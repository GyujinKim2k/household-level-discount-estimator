"""Population posterior of the current headline model, against the literature.

The §19.6 / §20 configuration -- wider embedder, `log(1 - delta)` target,
5-member ensemble -- applied to the 889 comphs PSID households. Draws are pooled
across households (the mixture of per-household posteriors), rejected against
the TRANSFORMED box and only then inverted to delta, for the underflow reason
recorded in `psid_posterior.sample_all`.

Shaded behind: the sourced meta-analytic bands of `literature_ranges.py` and
Laibson et al.'s 95% CI.

Usage::

    uv run python scripts/plot_adopted_posterior.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hh_npe.evaluation.plots import contour_corner
from hh_npe.npe.prior import PHASE3, from_log1m, log1m_box
from hh_npe.npe.train import load_posterior
from hh_npe.simulator.laibson_calibration import EDUC_GROUPS
from scripts.literature_ranges import META, laibson_ci


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dirs", type=Path, nargs="+",
                    default=[Path(f"outputs/adopted/log1m_s{s}") for s in range(5)])
    ap.add_argument("--x", type=Path,
                    default=Path("data/processed/psid_x_educ_rental.pt"))
    ap.add_argument("--per_household", type=int, default=40,
                    help="In-box draws kept per household per member.")
    ap.add_argument("--out", type=Path,
                    default=Path("figures/08_adopted_literature_comparison.png"))
    args = ap.parse_args()

    d = torch.load(args.x, weights_only=False)
    x = d["x"][d["educ"] == EDUC_GROUPS.index("comphs")].float()
    posts = [load_posterior(r / "posterior_7w.pt")["posterior"] for r in args.run_dirs]
    dev = next(posts[0].posterior_estimator.parameters()).device
    x = x.to(dev)

    box = log1m_box(PHASE3)
    lo = torch.as_tensor(box.low, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(box.high, dtype=torch.float32, device=dev)
    torch.manual_seed(0)
    pooled = []
    with torch.no_grad():
        for p in posts:
            s = p.posterior_estimator.sample((args.per_household * 2,), condition=x)
            s = s.transpose(0, 1)                         # (hh, S, 3)
            for row in s:
                keep = row[((row >= lo) & (row <= hi)).all(-1)][: args.per_household]
                pooled.append(keep.cpu().numpy())
    draws = from_log1m(np.concatenate(pooled))           # reject first, then invert
    print(f"{len(x)} households, {len(draws):,} pooled draws")
    for j, n in enumerate(PHASE3.names):
        print(f"  {n:6s} median {np.median(draws[:, j]):.4f}  "
              f"90% [{np.percentile(draws[:, j], 5):.4f}, "
              f"{np.percentile(draws[:, j], 95):.4f}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    contour_corner(
        {"PSID comphs, pooled posterior (N=889)": draws},
        PHASE3,
        truth={"Laibson et al. MSM": np.array([0.5305, 0.9891, 1.9355])},
        bands={"meta-analytic range": META, "Laibson et al. 95% CI": laibson_ci()},
        path=args.out,
        title="Current model (wide embedder, log(1-δ) target, 5-member ensemble)\n"
              "vs. the published literature",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
