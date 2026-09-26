"""The current headline model's per-household posteriors, against the literature.

Plots the **posterior mean of each household** -- one point per household, 889
in all -- as 68/95% contours, the same object as ``figures/07``. That shows how
households differ from each other.

It deliberately does NOT pool every posterior draw across households. A pooled
cloud adds each household's own estimation uncertainty on top of the
between-household spread, and for beta that uncertainty (median posterior sd
~0.15) exceeds the spread itself (~0.10), so the pooled contours smear across
most of the prior box. An earlier version of figure 08 did exactly that and read
as a much worse model when the model was unchanged in that respect.

The Phase 4 baseline is overlaid for comparison. Shaded behind: the sourced
meta-analytic bands of ``literature_ranges.py`` and Laibson et al.'s 95% CI.

Reads the ``posterior_uncorrected.npz`` files written by ``psid_posterior.py``,
whose means are already in delta space (rejected in log space, then inverted).

Usage::

    uv run python scripts/plot_adopted_posterior.py
    uv run python scripts/plot_adopted_posterior.py \
        --current outputs/psid_adopted_arch --label "wide embedder only" \
        --out figures/09_arch_only_literature_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hh_npe.evaluation.plots import contour_corner
from hh_npe.npe.prior import PHASE3
from scripts.literature_ranges import LAIBSON, META, laibson_ci


def means(run: Path) -> np.ndarray:
    m = np.load(run / "posterior_uncorrected.npz")["mean"]
    return m[np.isfinite(m[:, 0])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", type=Path, default=Path("outputs/psid_adopted"))
    ap.add_argument("--baseline", type=Path,
                    default=Path("outputs/psid_phase4_comphs"))
    ap.add_argument("--delta_top", type=float, default=1.04,
                    help="Upper end of the delta AXIS. The prior bound at 1.00 "
                         "is drawn as a dotted line; the density is not "
                         "extended past it (see contour_corner's axis_limits).")
    ap.add_argument("--baseline_label", default="Phase 4 baseline")
    ap.add_argument("--title_group", default="comphs")
    ap.add_argument("--label", default="current model: wide embedder, log(1-δ)",
                    help="Legend name for --current.")
    ap.add_argument("--out", type=Path,
                    default=Path("figures/08_adopted_literature_comparison.png"))
    args = ap.parse_args()

    cur, base = means(args.current), means(args.baseline)
    series = {
        f"{args.label}  N={len(cur)}": cur,
        f"{args.baseline_label}  N={len(base)}": base,
    }
    for k, v in series.items():
        print(f"{k}\n  median {np.median(v, 0).round(4)}  sd {v.std(0).round(4)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    contour_corner(
        series, PHASE3,
        truth={"Laibson et al. MSM": LAIBSON},
        bands={"meta-analytic range (lit.)": META,
               "Laibson et al. 95% CI": laibson_ci()},
        path=args.out,
        reflect_axes=("delta",),
        axis_limits=(PHASE3.low,
                     np.where(np.array(PHASE3.names) == "delta",
                              args.delta_top, PHASE3.high)),
        title="Per-household posterior means against published ranges -- PSID "
              f"{args.title_group}, 7 waves\nbands: CTB present-bias meta-analyses (β), "
              "Carroll et al. heterogeneous δ, Elminejad et al. ρ",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
