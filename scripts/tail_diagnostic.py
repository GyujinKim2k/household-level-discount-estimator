"""Where does beta's short upper tail come from (RESULTS.md 37.3)?

Under the uniform-prior SBC, where beta's posterior mean is in [0.8, 0.95) the
truth lies above the 95th percentile ~11% of the time instead of 5%, whatever
beta's flow target, and the misses are draws with beta near 1 (§37.1). What
the arms share upstream of the flow is the network ``q`` and the importance
weights. The held-out draws (the last two shards, all from the widened
EdgeMixture) separate them:

- **q**: the network's own posterior, reweighted only from its training
  mixture to the widened proposal the held-out draws came from (weights in
  ~[0.86, 1.15]). If ``q`` is right its ranks are calibrated on these draws, so
  a short upper tail here belongs to the network.
- **weighted**: the uniform-prior posterior the SBC checks, on the same draws,
  each truth weighted by ``p / p~`` -- the uniform-prior SBC by importance
  sampling over truths. It should reproduce the SBC's tail if the SBC
  simulations match the training generator.

Primary: one household per draw. Secondary: all 16 households per draw. The
intervals bootstrap over blocks of 16 consecutive draws, which share R_gamma
and card type; the SBC's own rate gets the same block bootstrap.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/tail_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.evaluation.weighted import sample_weighted
from hh_npe.npe.prior import EDGE_WIDENED, EdgeMixture, SwitchedProposal
from scripts.optionA import (
    CUTS,
    HELD_SEED,
    SHARDS,
    _weighting,
    flow_box,
    load_members,
    load_shards,
    one_per_draw,
    windows,
)

WIDENED = EdgeMixture(**EDGE_WIDENED)


def to_widened(n_train: int):
    """``log q2 - log q_mix``: from the training mixture to the widened proposal."""
    return lambda th: SwitchedProposal().log_weight(th, n_train) - WIDENED.log_weight(th)


BLOCK = 16    # theta_batch: draws per R_gamma / card-type block


def _boot_counts(blocks: np.ndarray, n_boot=4000, seed=0):
    """Block-bootstrap multiplicities: ``(n_boot, n_blocks)`` and each row's block."""
    ids, b = np.unique(blocks, return_inverse=True)
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
                     for _ in range(n_boot)]), b


def rate(hit, sel, w, counts, b) -> dict:
    """Weighted share of ``hit`` among ``sel``, with a block-bootstrap 90% CI."""
    num = np.bincount(b[sel], weights=(w * hit)[sel], minlength=counts.shape[1])
    den = np.bincount(b[sel], weights=w[sel], minlength=counts.shape[1])
    with np.errstate(invalid="ignore"):
        bs = (counts @ num) / (counts @ den)
    return {"rate": float(num.sum() / den.sum()),
            "ci90": [float(v) for v in np.nanquantile(bs, [0.05, 0.95])],
            "se": float(np.nanstd(bs))}


def split(mean, u, w, blocks) -> list[dict]:
    """beta's miss rates by posterior-mean region, truths weighted by ``w``."""
    counts, b = _boot_counts(blocks)
    c = CUTS["beta"]
    out = []
    for a, hi in zip(c[:-1], c[1:]):
        m = (mean >= a) & ((mean < hi) if hi < c[-1] else (mean <= hi))
        row = {"lo": a, "hi": hi, "n": int(m.sum())}
        if m.any():
            row["n_eff"] = float(w[m].sum() ** 2 / (w[m] ** 2).sum())
            for k, hit in (("above_95", u > 0.95), ("below_5", u < 0.05)):
                r = rate(hit, m, w, counts, b)
                row[k], row[k + "_ci90"], row[k + "_se"] = r["rate"], r["ci90"], r["se"]
        out.append(row)
    return out


def show(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"  {'beta mean region':18s}{'n':>7s}{'n_eff':>8s}"
          f"{'above 95th':>12s}{'90% CI':>16s}{'below 5th':>11s}{'90% CI':>16s}")
    for r in rows:
        if r["n"] < 20:
            continue
        ci = lambda k: f"{r[k][0]:.3f}-{r[k][1]:.3f}"
        print(f"  [{r['lo']:.2f}, {r['hi']:.2f}){'':6s}{r['n']:7d}{r['n_eff']:8.0f}"
              f"{r['above_95']:12.3f}{ci('above_95_ci90'):>16s}"
              f"{r['below_5']:11.3f}{ci('below_5_ci90'):>16s}")


def sbc_set(path: Path) -> dict:
    """The SBC's own tail rate with the same block bootstrap, and where its
    upper misses sit in (delta, rho)."""
    z = np.load(path)
    t, m, u = z["truth"], z["mean"][:, 0], z["u"][:, 0]
    counts, b = _boot_counts(np.arange(len(m)) // BLOCK)
    one = np.ones(len(m))
    reg = (m >= 0.8) & (m < 0.95)
    top = t[:, 0] >= 0.95
    miss = reg & (u > 0.95)
    in_conc = WIDENED._concentrated_density(t[miss, :3]) > 0
    return {"above_95": rate(u > 0.95, reg, one, counts, b),
            "below_5": rate(u < 0.05, reg, one, counts, b),
            "coverage_true_beta_ge_095": rate((u >= 0.05) & (u <= 0.95), top, one, counts, b),
            "misses": {"n": int(miss.sum()), "true_beta_median": float(np.median(t[miss, 0])),
                       "true_delta_median": float(np.median(t[miss, 1])),
                       "true_rho_median": float(np.median(t[miss, 2])),
                       "share_in_widened_concentrated_region": float(in_conc.mean())}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dirs", type=Path, nargs="+",
                    default=[Path(f"outputs/optionA/s{s}") for s in range(5)])
    ap.add_argument("--shards", type=Path, default=SHARDS)
    ap.add_argument("--n_draws", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else None)
    ap.add_argument("--out", type=Path, default=Path("outputs/optionA/tail_diagnostic.json"))
    args = ap.parse_args()

    members, proposal, cfg = load_members(args.run_dirs, args.device)
    bt = cfg["beta_transform"]
    inv, lw_uniform = _weighting(proposal, bt)
    fb = flow_box(bt)
    theta_all, _card, _tr, held_f, _p, _cfg = load_shards(args.shards)
    th, x, pid, _f = windows(held_f, theta_all, seed=HELD_SEED)
    th, pid = th.numpy().astype(float), pid.numpy()
    first = SwitchedProposal().n_first
    assert pid.min() >= proposal["n_train"] and pid.min() >= first, "held-out not widened-only"
    one = np.zeros(len(pid), bool)
    one[one_per_draw(torch.from_numpy(pid), seed=0)] = True
    print(f"{len(members)} members, beta target {bt}; held-out {len(th)} households from "
          f"{len(np.unique(pid))} draws (all from the widened proposal)", flush=True)

    torch.manual_seed(0)
    arms = {"q": to_widened(proposal["n_train"]), "weighted": lw_uniform}
    truth_w = {"q": np.ones(len(th)), "weighted": np.exp(WIDENED.log_weight(th))}
    blocks = pid // BLOCK
    res = {"run_dirs": list(map(str, args.run_dirs)), "beta_transform": bt,
           "sbc": sbc_set(args.run_dirs[0].parent / "evaluation" / "sbc_ensemble.npz")}
    rows = {"truth": th, "pid": pid, "one_per_draw": one}
    for arm, lw in arms.items():
        r = sample_weighted(members, x, args.n_draws, fb.low, fb.high, invert=inv,
                            log_weight=lw, truth=th)
        ok = np.isfinite(r["u"][:, 0])
        for scope, sel in (("one_per_draw", one & ok), ("all_households", ok)):
            res[f"{arm}/{scope}"] = split(r["mean"][sel, 0], r["u"][sel, 0],
                                          truth_w[arm][sel], blocks[sel])
        counts, b = _boot_counts(blocks[ok])
        u0 = r["u"][ok, 0]
        res[f"{arm}/coverage_true_beta_ge_095"] = rate(
            (u0 >= 0.05) & (u0 <= 0.95), th[ok, 0] >= 0.95, truth_w[arm][ok], counts, b)
        rows.update({f"{arm}_{k}": r[k] for k in ("mean", "hi", "u")})
        print(f"  {arm} done", flush=True)

    s = res["sbc"]
    f = lambda d: f"{d['rate']:.3f} ({d['ci90'][0]:.3f}-{d['ci90'][1]:.3f})"
    print(f"\n-- SBC set (uniform prior, 1 household per draw), block bootstrap\n"
          f"  [0.80, 0.95): above 95th {f(s['above_95'])}, below 5th {f(s['below_5'])}\n"
          f"  coverage, true beta >= 0.95: {f(s['coverage_true_beta_ge_095'])}\n"
          f"  its {s['misses']['n']} upper misses: true beta median "
          f"{s['misses']['true_beta_median']:.3f}, delta {s['misses']['true_delta_median']:.4f}, "
          f"rho {s['misses']['true_rho_median']:.2f}; "
          f"{s['misses']['share_in_widened_concentrated_region']:.0%} inside the widened "
          f"concentrated region")
    for arm, desc in (("q", "network q, reweighted to the held-out proposal only"),
                      ("weighted", "uniform-prior posterior, truths weighted by p/p~ "
                                   "(the SBC by importance sampling)")):
        for scope in ("one_per_draw", "all_households"):
            show(f"-- {desc}; {scope.replace('_', ' ')}", res[f"{arm}/{scope}"])
        print(f"  coverage, true beta >= 0.95: {f(res[f'{arm}/coverage_true_beta_ge_095'])}")
    args.out.write_text(json.dumps(res, indent=2))
    np.savez(args.out.with_suffix(".npz"), **rows)
    print(f"\nwrote {args.out} and its .npz")


if __name__ == "__main__":
    main()
