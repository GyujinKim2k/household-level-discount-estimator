"""Contour figure for the option A2 pilot (RESULTS 43.2).

Per-household posterior means on the 4-parameter box: PSID couples and all 889
on the pilot model (fixed solver, PSID seed pool), against option A's logit
headline (RESULTS 39: old solver, SCF seed), with the meta-analytic range and
the candidate concentrated region the report proposes.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/plot_optionA2_pilot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hh_npe.evaluation.plots import contour_corner
from hh_npe.npe.prior import PHASE3_RGAMMA as BOX
from scripts.literature_ranges import LAIBSON, META

SERIES = {"A2 pilot, couples": Path("outputs/psid_optionA2_pilot_couples"),
          "A2 pilot, all 889": Path("outputs/psid_optionA2_pilot_all"),
          "option A (old solver, SCF seed), all 889": Path("outputs/psid_optionA_beta_logit")}
REPORT = Path("outputs/optionA2_pilot/report.json")
OUT = Path("figures/35_optionA2_pilot_psid.png")


def main() -> None:
    series = {}
    for k, d in SERIES.items():
        m = np.load(d / "posterior_uncorrected.npz")["mean"]
        m = m[np.isfinite(m[:, 0])]
        series[f"{k}  N={len(m)}"] = m
    reg = json.loads(REPORT.read_text())["regions"]["couples p10/p90"]
    names = np.array(BOX.names)
    contour_corner(
        series, BOX,
        truth={"Laibson et al. MSM (R_gamma: calibrated 1.05)": np.append(LAIBSON, 1.05)},
        truth_markers=("*",), truth_colors=("#d62728",),
        bands={"meta-analytic range (lit.; R_gamma none)":
               (np.append(META[0], BOX.rgamma_low), np.append(META[1], BOX.rgamma_high)),
               "candidate concentrated region":
               (np.array([reg["beta_lo"], 1 - reg["one_minus_delta_hi"], reg["crra_lo"],
                          BOX.rgamma_low]), BOX.high)},
        reflect_axes=("delta",),
        axis_limits=(BOX.low, np.where(names == "delta", 1.04, BOX.high)),
        path=OUT,
        title="Per-household posterior means, PSID comphs: option A2 pilot (4,096 uniform "
              "draws, 3 members)\nagainst option A's logit headline; card type marginalised")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
