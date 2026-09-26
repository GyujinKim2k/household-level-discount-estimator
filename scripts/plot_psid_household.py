"""Posterior for one REAL PSID household, the counterpart of figure 10.

A real household has no known true theta, so the panel shows the 68/95%
credible regions, the posterior mean as the point estimate, Laibson et al.'s
population estimate as a reference, and the published ranges behind them.

Uses the current headline model (wide embedder, log(1 - delta) target, 5-member
ensemble); draws are rejected in log space and then inverted, as everywhere.

**Which household.** The *typical* one, by rule: the comphs household whose
posterior mean is closest, in prior-standardised distance, to the median
posterior mean across all 889. Reads the means written by ``psid_posterior``,
so the choice is reproducible and independent of this figure.

Usage::

    uv run python scripts/plot_psid_household.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hh_npe.evaluation.plots import contour_corner
from hh_npe.npe.prior import PHASE3
from hh_npe.npe.train import load_posterior
from hh_npe.simulator.laibson_calibration import EDUC_GROUPS
from scripts.literature_ranges import LAIBSON, META
from scripts.plot_simulated_household import draws_for


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dirs", type=Path, nargs="+",
                    default=[Path(f"outputs/adopted/log1m_s{s}") for s in range(5)])
    ap.add_argument("--psid_out", type=Path, default=Path("outputs/psid_adopted"),
                    help="psid_posterior output for the same model.")
    ap.add_argument("--x", type=Path,
                    default=Path("data/processed/psid_x_educ_rental.pt"))
    ap.add_argument("--n_draws", type=int, default=1000)
    ap.add_argument("--out", type=Path,
                    default=Path("figures/15_psid_household_posterior.png"))
    args = ap.parse_args()

    m = np.load(args.psid_out / "posterior_uncorrected.npz")["mean"]
    ok = np.isfinite(m[:, 0])
    med = np.median(m[ok], axis=0)
    prior_sd = (PHASE3.high - PHASE3.low) / np.sqrt(12)
    dist = np.sqrt((((m - med) / prior_sd) ** 2).sum(1))
    dist[~ok] = np.inf
    pick = int(np.argmin(dist))

    d = torch.load(args.x, weights_only=False)
    x = d["x"][d["educ"] == EDUC_GROUPS.index("comphs")].float()
    assert len(x) == len(m), "tensor and posterior file disagree on households"
    posts = [load_posterior(r / "posterior_7w.pt")["posterior"] for r in args.run_dirs]
    dev = next(posts[0].posterior_estimator.parameters()).device
    torch.manual_seed(0)
    s = draws_for(posts, x[pick:pick + 1].to(dev), args.n_draws)
    est = s.mean(0)
    q = np.percentile(s, [5, 95], axis=0)

    feats = list(d["features"])
    age = x[pick, :, feats.index("age")].int().tolist()
    print(f"household #{pick} of {len(x)} (typical: nearest the population median)")
    print(f"  ages {age[0]}-{age[-1]}")
    for f in ("income", "consumption", "liquid_assets", "illiquid_assets"):
        v = x[pick, :, feats.index(f)].numpy()
        print(f"  {f:16s} " + " ".join(f"{int(round(a)):>8,d}" for a in v))
    print(f"\n{'':8s}{'estimate':>10s}{'90% interval':>22s}{'population median':>20s}")
    for j, n in enumerate(PHASE3.names):
        print(f"{n:8s}{est[j]:10.4f}   [{q[0, j]:.4f}, {q[1, j]:.4f}]{med[j]:16.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    contour_corner(
        {f"posterior (5-member ensemble, {len(s):,} draws)": s},
        PHASE3,
        truth={"posterior mean (point estimate)": est,
               "Laibson et al. MSM (population)": LAIBSON},
        truth_markers=("X", "*"),
        truth_colors=("#1f1f1f", "#d62728"),
        bands={"meta-analytic range (lit.)": META},
        reflect_axes=("delta",),
        path=args.out,
        title=f"One real PSID comphs household (typical: nearest the median), "
              f"ages {age[0]}-{age[-1]}\ncurrent model (wide embedder, "
              "log(1-δ) target); no true θ exists for real data",
    )
    np.savez(args.out.with_suffix(".npz"), draws=s, estimate=est, pick=pick)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
