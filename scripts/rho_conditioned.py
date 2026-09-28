"""What would the data say about beta and delta if risk aversion were known?

RESULTS.md 31, test 2. Every model so far estimates rho jointly and lands at
~4.5, far above the consumption-Euler consensus (~1) and Laibson et al.'s 1.94.
This fixes rho instead: a network takes rho as an INPUT and estimates only
(beta, delta), trained on the same simulations -- for a training draw the input
is its own true rho. On PSID each household is then scored at chosen rho values,
giving ``p(beta, delta | x, rho)``: the present bias and patience the data imply
*if* risk aversion were 1, 1.94, 2 -- or 4.5, our joint estimate, as a
reference.

Configuration otherwise matches the adopted model (RESULTS 29): anchor + level
inputs, log(1 - delta) target, wide embedder, comphs, 5 seeds. rho enters as
log(rho) with NO training-set standardisation (range -0.69 to 1.61), keeping
training-set constants to age and the level channel only.

Usage::

    uv run python scripts/rho_conditioned.py train --seed 0
    uv run python scripts/rho_conditioned.py psid
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from hh_npe.data.windows import build_windowed
from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import LOG1M_EPS, PHASE3, sample_sobol
from hh_npe.npe.train import load_posterior, save_posterior, train_npe
from hh_npe.simulator.laibson_calibration import EDUC_GROUPS
from scripts.compare_windows import (
    EMBEDDER,
    TRAINING,
    anchor_log,
    educ_by_draw,
    mean_income_channel,
    split_shards,
)

OUT = Path("outputs/rho_conditioned")
RHO_VALUES = {"rho = 1": 1.0, "rho = 1.94 (Laibson)": 1.9355, "rho = 2": 2.0,
              "rho = 4.5 (joint estimate)": 4.5}


@dataclass(frozen=True)
class Box2:
    """(beta, log(1 - delta)) box, duck-typed for train_npe / make_sbi_prior."""
    names: tuple = ("beta", "delta")

    @property
    def low(self):
        return np.array([PHASE3.beta_low, np.log(LOG1M_EPS)])

    @property
    def high(self):
        return np.array([PHASE3.beta_high, np.log(1.0 - PHASE3.delta_low)])

    @property
    def n_params(self):
        return 2


def inputs(x: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Anchor + level transform, then log(rho) as a constant channel."""
    xt, feats = mean_income_channel(x, FEATURES_TWOASSET_AGE)
    xt = anchor_log(xt, feats)
    r = torch.log(rho.float())[:, None, None].expand(-1, xt.shape[1], 1)
    return torch.cat([xt, r], dim=-1)


CHANNELS = tuple(FEATURES_TWOASSET_AGE) + ("log_mean_income", "log_rho")


def to_target(theta: np.ndarray) -> torch.Tensor:
    """(beta, delta, rho) -> (beta, log(1 - delta))."""
    t = np.column_stack([theta[:, 0],
                         np.log(np.clip(1.0 - theta[:, 1], LOG1M_EPS, None))])
    return torch.as_tensor(t, dtype=torch.float32)


def windows(train: bool):
    sh = sorted(Path("data/processed/phase4_educ_dataset_shards").glob("shard_*.npz"))
    tr, ho = split_shards(sh, 57344)
    theta_all = sample_sobol(65536, PHASE3, seed=0)
    th, x, pid = build_windowed(tr if train else ho, theta_all, k=1, n_waves=7,
                                seed=0 if train else 999, start_low=24,
                                start_high=45, wave_years=2, with_age=True)
    keep = educ_by_draw(sh)[pid.numpy()] == EDUC_GROUPS.index("comphs")
    return th[keep].numpy(), x[keep], pid[keep]


def cmd_train(seed: int, n_heldout: int, n_post: int, out_root: Path,
              limit: int | None = None) -> None:
    th, x, pid = windows(train=True)
    if limit:
        th, x, pid = th[:limit], x[:limit], pid[:limit]
    xin = inputs(x, torch.as_tensor(th[:, 2]))
    keep = torch.tensor([c in ("age", "log_mean_income") for c in CHANNELS])
    f_mean = torch.where(keep, xin.mean((0, 1)), torch.zeros(len(CHANNELS)))
    f_std = torch.where(keep, xin.std((0, 1)), torch.ones(len(CHANNELS)))
    emb = TrajectoryTransformer(
        n_features=len(CHANNELS), seq_len=7, feature_mean=f_mean, feature_std=f_std,
        per_sequence=False,
        **{**EMBEDDER, "d_model": 128, "n_layers": 3, "output_dim": 64})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    post, _d, _i = train_npe(to_target(th), xin, embedder=emb, box=Box2(),
                             device=dev, group_ids=pid,
                             **{**TRAINING, "batch_size": 1024, "learning_rate": 1e-3})
    out = out_root / f"s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_posterior(post, emb, Box2(), out / "posterior.pt")

    # Held-out recovery of beta and delta GIVEN the true rho, in delta space.
    th_h, x_h, _p = windows(train=False)
    th_h, x_h = th_h[:n_heldout], x_h[:n_heldout]
    xin_h = inputs(x_h, torch.as_tensor(th_h[:, 2])).to(dev)
    lo, hi = Box2().low, Box2().high
    means, cov = [], []
    with torch.no_grad():
        for i in range(len(xin_h)):
            s = post.posterior_estimator.sample((n_post,), condition=xin_h[i:i + 1])
            s = s.squeeze(1).cpu().numpy()
            s = s[((s >= lo) & (s <= hi)).all(1)]
            s[:, 1] = 1.0 - np.exp(s[:, 1])
            means.append(s.mean(0))
            q = np.percentile(s, [5, 95], axis=0)
            cov.append((q[0] <= th_h[i, :2]) & (th_h[i, :2] <= q[1]))
    m, c = np.array(means), np.array(cov)
    res = {n: {"corr": float(np.corrcoef(m[:, j], th_h[:, j])[0, 1]),
               "mae": float(np.abs(m[:, j] - th_h[:, j]).mean()),
               "coverage_90": float(c[:, j].mean())}
           for j, n in enumerate(("beta", "delta"))}
    (out / "heldout.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def cmd_psid(n_post: int, out_root: Path, fig: Path) -> None:
    from scripts.literature_ranges import META
    d = torch.load("data/processed/psid_x_educ_rental.pt", weights_only=False)
    x = d["x"][d["educ"] == EDUC_GROUPS.index("comphs")].float()
    posts = [load_posterior(p)["posterior"] for p in sorted(out_root.glob("s*/posterior.pt"))]
    dev = next(posts[0].posterior_estimator.parameters()).device
    lo, hi = Box2().low, Box2().high
    per = max(1, n_post * 2 // len(posts))
    summary, draws_for_fig = {}, {}
    torch.manual_seed(0)
    print(f"{len(posts)} members, {len(x)} comphs households\n")
    print(f"{'':28s}{'median beta':>12s}{'median delta':>13s}{'beta 90% w':>11s}"
          f"{'in-box':>8s}{'beta in lit.':>13s}{'delta in lit.':>14s}")
    for lab, rho in RHO_VALUES.items():
        xin = inputs(x, torch.full((len(x),), rho)).to(dev)
        means, widths, frac = [], [], []
        with torch.no_grad():
            for b0 in range(0, len(xin), 256):
                xb = xin[b0:b0 + 256]
                s = torch.cat([p.posterior_estimator.sample((per,), condition=xb)
                               for p in posts]).cpu().numpy()          # (S, B, 2)
                inb = ((s >= lo) & (s <= hi)).all(-1)
                for j in range(s.shape[1]):
                    k = s[inb[:, j], j]
                    frac.append(inb[:, j].mean())
                    if len(k) < 20:
                        means.append([np.nan, np.nan]); widths.append([np.nan, np.nan]); continue
                    k = k.copy(); k[:, 1] = 1.0 - np.exp(k[:, 1])
                    means.append(k.mean(0))
                    q = np.percentile(k, [5, 95], axis=0); widths.append(q[1] - q[0])
        m, w = np.array(means), np.array(widths); ok = np.isfinite(m[:, 0])
        inb_b = ((m[ok, 0] >= META[0][0]) & (m[ok, 0] <= META[1][0])).mean()
        inb_d = ((m[ok, 1] >= META[0][1]) & (m[ok, 1] <= META[1][1])).mean()
        print(f"{lab:28s}{np.median(m[ok,0]):12.3f}{np.median(m[ok,1]):13.4f}"
              f"{np.median(w[ok,0]):11.3f}{np.median(frac):8.3f}{inb_b:13.1%}{inb_d:14.1%}")
        summary[lab] = {"rho": rho, "median_beta": float(np.median(m[ok, 0])),
                        "median_delta": float(np.median(m[ok, 1])),
                        "median_in_box": float(np.median(frac)),
                        "beta_in_lit": float(inb_b), "delta_in_lit": float(inb_d)}
        np.savez(out_root / f"psid_{rho:.4f}.npz", mean=m, width=w, in_box=np.array(frac))
        draws_for_fig[f"{lab}  N={ok.sum()}"] = m[ok]
    (out_root / "psid_summary.json").write_text(json.dumps(summary, indent=2))

    from hh_npe.evaluation.plots import contour_corner

    @dataclass(frozen=True)
    class PlotBox:
        names: tuple = ("beta", "delta")
        low: tuple = (PHASE3.beta_low, PHASE3.delta_low)
        high: tuple = (PHASE3.beta_high, PHASE3.delta_high)

    contour_corner(
        draws_for_fig, PlotBox(),
        truth={"Laibson et al. MSM": np.array([0.5305, 0.9891])},
        bands={"meta-analytic range (lit.)": (META[0][:2], META[1][:2])},
        reflect_axes=("delta",), path=fig,
        title="PSID comphs: β and δ implied by each household's data if ρ were fixed\n"
              "per-household posterior means; adopted inputs, log(1-δ) target")
    print(f"\nwrote {fig}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--n_heldout", type=int, default=1024)
    t.add_argument("--n_post", type=int, default=600)
    t.add_argument("--limit", type=int, default=None, help="Training rows (smoke test).")
    t.add_argument("--out", type=Path, default=OUT)
    q = sub.add_parser("psid"); q.add_argument("--n_post", type=int, default=500)
    q.add_argument("--out", type=Path, default=OUT)
    q.add_argument("--fig", type=Path,
                   default=Path("figures/26_rho_conditioned_beta_delta.png"))
    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train(a.seed, a.n_heldout, a.n_post, a.out, a.limit)
    else:
        cmd_psid(a.n_post, a.out, a.fig)


if __name__ == "__main__":
    main()
