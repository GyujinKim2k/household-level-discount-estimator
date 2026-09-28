"""Pilot: is the illiquid return R_gamma identifiable from a household panel?

RESULTS.md 32. The first 4,096 draws of the option A run, with R_gamma drawn per
block on [1.025, 1.075] as a fourth parameter. Before spending the rest of the
~4.7-9.4 GPU-days, train a four-parameter model and ask one question: does
R_gamma recover on held-out simulated households, and what does estimating it
cost beta, delta and rho?

The concern is identification, not data: in the Euler equation patience and the
return enter roughly as delta * R_gamma, so the two can substitute. What should
separate them is the split between liquid and illiquid wealth, since only the
illiquid asset earns R_gamma.

theta is read from the shards (all four columns), never re-derived. The held-out
draws come from the training proposal, so recovery is scored against the
proposal posterior q itself, unweighted -- the importance correction to the
uniform prior matters for PSID inference, not for this identification check.

Configuration: the adopted one (RESULTS 29) -- anchor + level inputs, log(1 - delta)
target, wide embedder, static standardisation on age and log mean income only.

Usage::

    uv run python scripts/rgamma_pilot.py --seed 0
    uv run python scripts/rgamma_pilot.py --report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from hh_npe.data.windows import build_windowed
from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import PHASE3_RGAMMA, log1m_box, to_log1m, from_log1m
from hh_npe.npe.train import save_posterior, train_npe
from scripts.compare_windows import EMBEDDER, TRAINING, anchor_log, mean_income_channel

SHARDS = Path("data/processed/optionA_card_dataset_shards")
OUT = Path("outputs/rgamma_pilot")
NAMES = PHASE3_RGAMMA.names


def load(shards: Path):
    files = sorted(shards.glob("shard_*.npz"))
    if not files:
        raise SystemExit(f"no shards in {shards}")
    parts = {int(np.load(f)["lo"]): np.load(f)["theta"] for f in files}
    n = max(int(np.load(f)["hi"]) for f in files)
    theta_all = np.full((n, 4), np.nan)
    for lo, th in parts.items():
        theta_all[lo:lo + len(th)] = th
    if np.isnan(theta_all).any():
        raise SystemExit("shards do not cover a contiguous range of draws")
    # Hold out the last shard: whole draws, so no draw's households straddle.
    held_lo = max(parts)
    train_f = [f for f in files if int(np.load(f)["lo"]) < held_lo]
    held_f = [f for f in files if int(np.load(f)["lo"]) >= held_lo]
    return theta_all, train_f, held_f


def windows(files, theta_all, seed):
    th, x, pid = build_windowed(files, theta_all, k=1, n_waves=7, seed=seed,
                                start_low=24, start_high=45, wave_years=2,
                                with_age=True)
    x, feats = mean_income_channel(x, FEATURES_TWOASSET_AGE)
    return th, anchor_log(x, feats), pid, feats


def train(seed: int, shards: Path, n_heldout: int, n_post: int) -> None:
    theta_all, train_f, held_f = load(shards)
    th, x, pid, feats = windows(train_f, theta_all, seed=0)
    print(f"train: {len(th)} rows from {len(pid.unique())} draws; held-out "
          f"shards {len(held_f)}")
    keep = torch.tensor([f in ("age", "log_mean_income") for f in feats])
    f_mean = torch.where(keep, x.mean((0, 1)), torch.zeros(len(feats)))
    f_std = torch.where(keep, x.std((0, 1)), torch.ones(len(feats)))
    emb = TrajectoryTransformer(
        n_features=len(feats), seq_len=7, feature_mean=f_mean, feature_std=f_std,
        per_sequence=False, **{**EMBEDDER, "d_model": 128, "n_layers": 3,
                               "output_dim": 64})
    box = log1m_box(PHASE3_RGAMMA)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    post, _d, _i = train_npe(to_log1m(th, PHASE3_RGAMMA).float(), x, embedder=emb,
                             box=box, device=dev, group_ids=pid,
                             **{**TRAINING, "batch_size": 1024, "learning_rate": 1e-3})
    out = OUT / f"s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_posterior(post, emb, box, out / "posterior.pt")

    th_h, x_h, _p, _f = windows(held_f, theta_all, seed=999)
    th_h, x_h = th_h[:n_heldout].numpy(), x_h[:n_heldout].to(dev)
    lo, hi = box.low, box.high
    means, cov = [], []
    with torch.no_grad():
        for i in range(len(x_h)):
            s = post.posterior_estimator.sample((n_post,), condition=x_h[i:i + 1])
            s = s.squeeze(1).cpu().numpy()
            s = from_log1m(s[((s >= lo) & (s <= hi)).all(1)], PHASE3_RGAMMA)
            means.append(s.mean(0))
            q = np.percentile(s, [5, 95], axis=0)
            cov.append((q[0] <= th_h[i]) & (th_h[i] <= q[1]))
    m, c = np.array(means), np.array(cov)
    prior_sd = (PHASE3_RGAMMA.high - PHASE3_RGAMMA.low) / np.sqrt(12)
    res = {n: {"corr": float(np.corrcoef(m[:, j], th_h[:, j])[0, 1]),
               "mae": float(np.abs(m[:, j] - th_h[:, j]).mean()),
               "mae_over_prior_sd": float(np.abs(m[:, j] - th_h[:, j]).mean() / prior_sd[j]),
               "coverage_90": float(c[:, j].mean())}
           for j, n in enumerate(NAMES)}
    # The confounding the pilot exists to measure: are delta and R_gamma
    # estimation errors correlated?
    err = m - th_h
    res["err_corr_delta_rgamma"] = float(np.corrcoef(err[:, 1], err[:, 3])[0, 1])
    (out / "heldout.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def report() -> None:
    rows = [json.loads(f.read_text()) for f in sorted(OUT.glob("s*/heldout.json"))]
    if not rows:
        raise SystemExit("no pilot runs yet")
    print(f"{len(rows)} seeds")
    print(f"{'':10s}{'corr':>8s}{'mae':>10s}{'mae/prior sd':>14s}{'cov90':>8s}")
    for n in NAMES:
        v = np.array([[r[n][k] for k in ("corr", "mae", "mae_over_prior_sd",
                                         "coverage_90")] for r in rows]).mean(0)
        print(f"{n:10s}{v[0]:8.3f}{v[1]:10.4f}{v[2]:14.3f}{v[3]:8.3f}")
    print(f"error correlation delta vs R_gamma: "
          f"{np.mean([r['err_corr_delta_rgamma'] for r in rows]):+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", type=Path, default=SHARDS)
    ap.add_argument("--n_heldout", type=int, default=1024)
    ap.add_argument("--n_post", type=int, default=500)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--psid_check", action="store_true",
                    help="Where PSID households land with R_gamma free, against "
                         "the edge mixture's concentrated region (RESULTS 32.2).")
    a = ap.parse_args()
    if a.psid_check:
        psid_check()
    elif a.report:
        report()
    else:
        train(a.seed, a.shards, a.n_heldout, a.n_post)



def psid_check(n_draws: int = 400) -> None:
    """Where do PSID households land when R_gamma is free (RESULTS 32.2)?

    The edge mixture's concentrated half (beta 0.75-1, 1-delta in [1e-4, 0.05],
    rho 3.5-5) was aimed at where PSID posteriors sat with R_gamma fixed. This
    applies the pilot's four-parameter model to PSID and measures posterior
    mass inside that region. Draws are importance-weighted by
    EdgeMixture.log_weight, so the result reflects the uniform prior.
    """
    from hh_npe.npe.prior import EdgeMixture
    from hh_npe.npe.train import load_posterior
    from hh_npe.simulator.laibson_calibration import EDUC_GROUPS

    d = torch.load("data/processed/psid_x_educ_rental.pt", weights_only=False)
    x = d["x"][d["educ"] == EDUC_GROUPS.index("comphs")].float()
    x, feats = mean_income_channel(x, FEATURES_TWOASSET_AGE)
    x = anchor_log(x, feats)
    posts = [load_posterior(f)["posterior"] for f in sorted(OUT.glob("s*/posterior.pt"))]
    dev = next(posts[0].posterior_estimator.parameters()).device
    box, mix = log1m_box(PHASE3_RGAMMA), EdgeMixture()
    lo, hi = box.low, box.high
    means, in_region, rg_sd = [], [], []
    torch.manual_seed(0)
    with torch.no_grad():
        for b0 in range(0, len(x), 256):
            xb = x[b0:b0 + 256].to(dev)
            s = torch.cat([p.posterior_estimator.sample((n_draws,), condition=xb)
                           for p in posts]).cpu().numpy()          # (S, B, 4)
            for j in range(s.shape[1]):
                k = s[:, j][((s[:, j] >= lo) & (s[:, j] <= hi)).all(1)]
                k = from_log1m(k, PHASE3_RGAMMA)
                w = np.exp(mix.log_weight(k)); w /= w.sum()
                means.append((w[:, None] * k).sum(0))
                in_region.append(float((w * (mix._concentrated_density(k[:, :3]) > 0)).sum()))
                rg_sd.append(float(np.sqrt((w * (k[:, 3] - means[-1][3]) ** 2).sum())))
    m, r = np.array(means), np.array(in_region)
    print(f"{len(posts)} pilot members, {len(m)} comphs households (importance-weighted)")
    print(f"median posterior mean  beta {np.median(m[:,0]):.3f}  delta {np.median(m[:,1]):.4f}  "
          f"rho {np.median(m[:,2]):.3f}  R_gamma {np.median(m[:,3]):.4f}")
    print(f"R_gamma: posterior-mean spread across households {m[:,3].std():.4f}; "
          f"median posterior sd {np.median(rg_sd):.4f} (prior sd {0.05/np.sqrt(12):.4f})")
    print(f"posterior mass inside the concentrated region: median {np.median(r):.2f}, "
          f"share of households with >50% inside {np.mean(r > 0.5):.1%}")
    print("share of household posterior means inside each band of the region:")
    print(f"  beta >= 0.75: {np.mean(m[:,0] >= 0.75):.1%}   1-delta <= 0.05: "
          f"{np.mean(1 - m[:,1] <= 0.05):.1%}   rho >= 3.5: {np.mean(m[:,2] >= 3.5):.1%}")
    for q in (10, 25, 50, 75, 90):
        print(f"  rho p{q}: {np.percentile(m[:,2], q):.2f}", end="")
    print()
    np.savez(OUT / "psid_check.npz", mean=m, in_region=r)


if __name__ == "__main__":
    main()
