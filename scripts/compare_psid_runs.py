"""Per-household shifts between two ``optionA.py psid`` runs (RESULTS 43.11).

Pairs the households of two runs on the same PSID tensor and splits them by
whether they report card debt (negative liquid wealth in any wave). For a
change to the input that only touches debt, such as ``--debt_scale``, the
households without debt are a check: their windows are the same, so their
shift must be zero. The shift is run B's posterior mean minus run A's, also
in prior sds of each parameter.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/compare_psid_runs.py \\
        --a outputs/psid_optionA2n_couples --b outputs/psid_optionA2n_couples_alpha202 \\
        --x data/processed/psid_x_comphs_couples.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.npe.prior import PHASE3_RGAMMA

PARAMS = ["beta", "delta", "crra", "R_gamma"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="Reference run.")
    ap.add_argument("--b", type=Path, required=True, help="Changed run.")
    ap.add_argument("--x", type=Path, required=True, help="The PSID tensor both runs used.")
    ap.add_argument("--arm", default="uncorrected", choices=["uncorrected", "corrected"])
    args = ap.parse_args()

    d = torch.load(args.x, weights_only=False)
    x = d["x"].numpy()
    liquid = list(d["features"]).index("liquid_assets")
    runs = []
    for r in (args.a, args.b):
        z = np.load(r / f"posterior_{args.arm}.npz")
        idx = np.load(r / "household_index.npy")
        runs.append((idx, z["mean"], z["lo"], z["hi"]))
    if not np.array_equal(runs[0][0], runs[1][0]):
        raise SystemExit("the two runs cover different households")
    idx, ma, la, ha = runs[0]
    _, mb, lb, hb = runs[1]
    ok = np.isfinite(ma[:, 0]) & np.isfinite(mb[:, 0])
    debt = (x[idx, :, liquid] < 0).any(1)
    rr = json.loads((args.a / "summary.json").read_text()).get("rgamma_range")
    box = dataclasses.replace(PHASE3_RGAMMA, rgamma_low=rr[0], rgamma_high=rr[1]) if rr \
        else PHASE3_RGAMMA
    box_sd = (np.asarray(box.high) - np.asarray(box.low)) / np.sqrt(12)   # uniform prior
    width_a, width_b = ha - la, hb - lb

    print(f"{args.a} -> {args.b}, {args.arm} arm; {ok.sum()} households")
    for name, g in (("with card debt", debt & ok), ("without card debt", ~debt & ok)):
        s = mb[g] - ma[g]
        print(f"\n{name}: {g.sum()} ({g.sum() / ok.sum():.1%})")
        print(f"{'':10s}{'median A':>10s}{'median B':>10s}{'shift med':>11s}{'p10':>9s}{'p90':>9s}"
              f"{'in prior sd: med':>18s}{'p10':>8s}{'p90':>8s}{'90% width A -> B (median)':>28s}")
        for j, p in enumerate(PARAMS):
            q = np.percentile(s[:, j], [10, 50, 90])
            print(f"{p:10s}{np.median(ma[g, j]):10.4f}{np.median(mb[g, j]):10.4f}"
                  f"{q[1]:+11.4f}{q[0]:+9.4f}{q[2]:+9.4f}"
                  f"{q[1] / box_sd[j]:+18.3f}{q[0] / box_sd[j]:+8.3f}{q[2] / box_sd[j]:+8.3f}"
                  f"{np.median(width_a[g, j]):17.4f} -> {np.median(width_b[g, j]):.4f}")
        if name.startswith("without"):
            print(f"  largest |shift| among them: {np.abs(s).max():.3g}")


if __name__ == "__main__":
    main()
