"""Contour figure for the option A2 models' PSID posteriors (RESULTS 43.4, 43.5).

Per-household posterior means on the 4-parameter box, one series per
``optionA.py psid`` output, with Laibson et al.'s point and the meta-analytic
range. The default is figure 38: the 5-member A2 ensemble's couples and all
889, against the 4,096-draw pilot's couples. The box is the widest R_gamma
range among the series (summary.json records it from RESULTS 43.5 on).

Usage::

    PYTHONPATH=. .venv/bin/python scripts/plot_optionA2_psid.py
    PYTHONPATH=. .venv/bin/python scripts/plot_optionA2_psid.py \\
        --series "A2w, couples=outputs/psid_optionA2w_couples" ... --out figures/41_...png
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from hh_npe.evaluation.plots import contour_corner
from hh_npe.npe.prior import PHASE3_RGAMMA as BOX
from scripts.literature_ranges import LAIBSON, META

SERIES = ["A2, couples (5 members)=outputs/psid_optionA2_couples",
          "A2, all 889 (5 members)=outputs/psid_optionA2_all",
          "A2 pilot, couples (3 members, 4,096 draws)=outputs/psid_optionA2_pilot_couples"]
OUT = Path("figures/38_optionA2_psid.png")
TITLE = ("Per-household posterior means, PSID comphs: option A2 (32,768 draws, "
         "5 members) against its pilot")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", nargs="+", default=SERIES, help="LABEL=DIR, in plot order.")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--title", default=TITLE)
    args = ap.parse_args()
    series, lo = {}, BOX.rgamma_low
    for e in args.series:
        k, d = e.split("=", 1)
        m = np.load(Path(d) / "posterior_uncorrected.npz")["mean"]
        m = m[np.isfinite(m[:, 0])]
        series[f"{k}  N={len(m)}"] = m
        rr = json.loads((Path(d) / "summary.json").read_text()).get("rgamma_range")
        lo = min(lo, rr[0]) if rr else lo
    box = dataclasses.replace(BOX, rgamma_low=lo)
    names = np.array(box.names)
    contour_corner(
        series, box,
        truth={"Laibson et al. MSM (R_gamma: calibrated 1.05)": np.append(LAIBSON, 1.05)},
        truth_markers=("*",), truth_colors=("#d62728",),
        bands={"meta-analytic range (lit.; R_gamma none)":
               (np.append(META[0], box.rgamma_low), np.append(META[1], box.rgamma_high))},
        reflect_axes=("delta",),
        axis_limits=(box.low, np.where(names == "delta", 1.04, box.high)),
        path=args.out,
        title=f"{args.title}\nimportance-weighted to the uniform prior; card type "
              "marginalised; rho and R_gamma axes start at the prior's lower edges")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
