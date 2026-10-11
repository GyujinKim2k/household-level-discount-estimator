"""Contour figure for the option A2 model on the full run (RESULTS 43.4).

Per-household posterior means on the 4-parameter box: PSID couples and all 889
on the 5-member ensemble, against the 4,096-draw pilot's couples (figure 35),
with Laibson et al.'s point and the meta-analytic range.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/plot_optionA2_psid.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hh_npe.evaluation.plots import contour_corner
from hh_npe.npe.prior import PHASE3_RGAMMA as BOX
from scripts.literature_ranges import LAIBSON, META

SERIES = {"A2, couples (5 members)": Path("outputs/psid_optionA2_couples"),
          "A2, all 889 (5 members)": Path("outputs/psid_optionA2_all"),
          "A2 pilot, couples (3 members, 4,096 draws)": Path("outputs/psid_optionA2_pilot_couples")}
OUT = Path("figures/38_optionA2_psid.png")


def main() -> None:
    series = {}
    for k, d in SERIES.items():
        m = np.load(d / "posterior_uncorrected.npz")["mean"]
        m = m[np.isfinite(m[:, 0])]
        series[f"{k}  N={len(m)}"] = m
    names = np.array(BOX.names)
    contour_corner(
        series, BOX,
        truth={"Laibson et al. MSM (R_gamma: calibrated 1.05)": np.append(LAIBSON, 1.05)},
        truth_markers=("*",), truth_colors=("#d62728",),
        bands={"meta-analytic range (lit.; R_gamma none)":
               (np.append(META[0], BOX.rgamma_low), np.append(META[1], BOX.rgamma_high))},
        reflect_axes=("delta",),
        axis_limits=(BOX.low, np.where(names == "delta", 1.04, BOX.high)),
        path=OUT,
        title="Per-household posterior means, PSID comphs: option A2 (32,768 draws, "
              "5 members) against its pilot\nimportance-weighted to the uniform prior; "
              "card type marginalised; rho and R_gamma axes start at the prior's lower edges")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
