"""Posterior for one held-out SIMULATED household, where the true theta is known.

The PSID figures can only show where the posterior lands; they cannot show
whether it is right. A held-out simulated household can: its true (beta, delta,
rho) generated the data, so the figure shows the 68/95% region, the posterior
mean as the point estimate, and the truth, on one panel.

Uses the current headline model (wide embedder, log(1 - delta) target, 5-member
ensemble). Draws are rejected against the transformed box and then inverted to
delta, as in ``psid_posterior``.

**Which household.** Not hand-picked for looking good. Among the first
``--pool`` held-out comphs households, the one whose standardised posterior-mean
error is the *median* of the pool, restricted to truths away from the prior
edges so no panel is dominated by a boundary. The pool's error distribution is
printed, so the choice can be checked.

Usage::

    uv run python scripts/plot_simulated_household.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.windows import build_windowed
from hh_npe.evaluation.plots import contour_corner
from hh_npe.npe.prior import PHASE3, from_log1m, log1m_box, sample_sobol
from hh_npe.npe.train import load_posterior
from hh_npe.simulator.laibson_calibration import EDUC_GROUPS
from scripts.compare_windows import educ_by_draw, split_shards


def draws_for(posts, x1: torch.Tensor, n: int) -> np.ndarray:
    """Pooled ensemble draws for one household, rejected in log space."""
    box = log1m_box(PHASE3)
    lo, hi = np.asarray(box.low), np.asarray(box.high)
    out = []
    with torch.no_grad():
        for p in posts:
            s = p.posterior_estimator.sample((n,), condition=x1).squeeze(1)
            s = s.cpu().numpy()
            out.append(s[((s >= lo) & (s <= hi)).all(1)])
    return from_log1m(np.concatenate(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dirs", type=Path, nargs="+",
                    default=[Path(f"outputs/adopted/log1m_s{s}") for s in range(5)])
    ap.add_argument("--shards", type=Path,
                    default=Path("data/processed/phase4_educ_dataset_shards"))
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--n_draws", type=int, default=1000,
                    help="Draws per ensemble member.")
    ap.add_argument("--out", type=Path,
                    default=Path("figures/10_simulated_household_posterior.png"))
    args = ap.parse_args()

    shard_files = sorted(args.shards.glob("shard_*.npz"))
    _tr, held = split_shards(shard_files, 57344)
    theta_all = sample_sobol(65536, PHASE3, seed=0)
    th, x, pid = build_windowed(held, theta_all, k=1, n_waves=7, seed=999,
                                start_low=24, start_high=45, wave_years=2,
                                with_age=True)
    comphs = educ_by_draw(shard_files)[pid.numpy()] == EDUC_GROUPS.index("comphs")
    th, x = th[comphs][: args.pool].numpy(), x[comphs][: args.pool]

    posts = [load_posterior(r / "posterior_7w.pt")["posterior"] for r in args.run_dirs]
    dev = next(posts[0].posterior_estimator.parameters()).device
    torch.manual_seed(0)
    means, sds = [], []
    for i in range(len(x)):
        d = draws_for(posts, x[i:i + 1].to(dev), 200)
        means.append(d.mean(0)); sds.append(d.std(0))
    means, sds = np.array(means), np.array(sds)

    prior_sd = (PHASE3.high - PHASE3.low) / np.sqrt(12)
    err = np.sqrt((((means - th) / prior_sd) ** 2).mean(1))
    # Away from the prior edges: 10% of each range in from both ends.
    span = PHASE3.high - PHASE3.low
    interior = ((th > PHASE3.low + 0.1 * span) & (th < PHASE3.high - 0.1 * span)).all(1)
    cand = np.flatnonzero(interior)
    pick = cand[np.argsort(err[cand])[len(cand) // 2]]
    print(f"pool {len(x)}, interior {len(cand)}; standardised error "
          f"p25 {np.percentile(err, 25):.3f} median {np.median(err):.3f} "
          f"p75 {np.percentile(err, 75):.3f}; picked #{pick} at {err[pick]:.3f}")

    d = draws_for(posts, x[pick:pick + 1].to(dev), args.n_draws)
    truth, est = th[pick], d.mean(0)
    q = np.percentile(d, [5, 95], axis=0)
    print(f"{'':8s}{'true':>9s}{'estimate':>10s}{'90% interval':>22s}  covered")
    for j, n in enumerate(PHASE3.names):
        cov = q[0, j] <= truth[j] <= q[1, j]
        print(f"{n:8s}{truth[j]:9.4f}{est[j]:10.4f}   [{q[0, j]:.4f}, {q[1, j]:.4f}]"
              f"  {'yes' if cov else 'NO'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    contour_corner(
        {f"posterior (5-member ensemble, {len(d):,} draws)": d},
        PHASE3,
        truth={"true θ (generated the data)": truth,
               "posterior mean (point estimate)": est},
        truth_markers=("*", "X"),
        truth_colors=("#d62728", "#1f1f1f"),
        reflect_axes=("delta",),
        path=args.out,
        title="One held-out simulated comphs household: posterior vs. truth\n"
              "current model (wide embedder, log(1-δ) target); 7 waves",
    )
    np.savez(args.out.with_suffix(".npz"), draws=d, truth=truth, estimate=est,
             pool_error=err, pick=pick)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
