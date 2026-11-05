"""RESULTS 55: PSID posteriors from the v2 data against A2n's, in RESULTS 1's
headline template, household by household.

v2 is A2n's model trained on the regenerated data (``run_regen_v2.sh``): the
same draws, card types, shocks and starting states, the audit's fixes, and
flows a year before stocks. Both posteriors are for the same PSID households,
so besides each population summary the paired per-household shift is
reported, in units of A2n's own posterior sd.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/compare_regen_v2.py \\
        --pair couples outputs/psid_optionA2n_couples outputs/psid_optionA2v2_couples \\
        --pair all outputs/psid_optionA2n_all outputs/psid_optionA2v2_all \\
        --out outputs/optionA2v2/comparison.json

RESULTS 1 since 2026-11-05 (v2 on the box with the beta strip, RESULTS 63)::

    PYTHONPATH=. .venv/bin/python scripts/compare_regen_v2.py --labels v2 v2_b010 \\
        --pair couples outputs/psid_optionA2v2_couples outputs/psid_optionA2v2_b010_couples \\
        --pair all outputs/psid_optionA2v2_all outputs/psid_optionA2v2_b010_all \\
        --out outputs/optionA2v2_b010/comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.heterogeneity_test import decompose
from scripts.literature_ranges import LAIBSON, META, laibson_ci

NAMES = ("beta", "delta", "crra", "R_gamma")


def load(d: Path, arm: str) -> dict:
    z = np.load(d / f"posterior_{arm}.npz")
    return {k: z[k] for k in ("mean", "sd", "lo", "hi")} | {
        "index": np.load(d / "household_index.npy")}


def summary(r: dict) -> dict:
    """RESULTS 1's headline rows for one set of per-household posteriors."""
    m, sd, lo, hi = r["mean"], r["sd"], r["lo"], r["hi"]
    ref = np.append(LAIBSON, 1.05)
    llo, lhi = laibson_ci()
    inside = (m[:, :3] >= META[0]) & (m[:, :3] <= META[1])
    het = decompose(m, sd, 10_000, 0, list(NAMES))
    return {"n": int(len(m)),
            "median of means": np.median(m, 0).tolist(),
            "mean of means": m.mean(0).tolist(),
            "sd across households": m.std(0).tolist(),
            "median posterior sd": np.median(sd, 0).tolist(),
            "covers Laibson (R_gamma 1.05)": ((lo <= ref) & (ref <= hi)).mean(0).tolist(),
            "inside meta-analysis range": inside.mean(0).tolist(),
            "inside meta, all three": float(inside.all(1).mean()),
            "inside Laibson CI": ((m[:, :3] >= llo) & (m[:, :3] <= lhi)).mean(0).tolist(),
            "heterogeneity": {n: {q: het[n][q] for q in ("var_true", "var_true_ci", "p_le_zero")}
                              for n in NAMES}}


def paired(a: dict, b: dict) -> dict:
    """b minus a, household by household, raw and over a's posterior sd (the
    keys say A2n, the default first run)."""
    if not np.array_equal(a["index"], b["index"]):
        raise SystemExit("the two runs are not over the same PSID households")
    d = b["mean"] - a["mean"]
    z = d / a["sd"]
    return {"median shift": np.median(d, 0).tolist(),
            "median |shift| / A2n sd": np.median(np.abs(z), 0).tolist(),
            "p90 |shift| / A2n sd": np.quantile(np.abs(z), 0.9, 0).tolist(),
            "corr of means": [float(np.corrcoef(a["mean"][:, j], b["mean"][:, j])[0, 1])
                              for j in range(len(NAMES))],
            "posterior sd ratio, median": np.median(b["sd"] / a["sd"], 0).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=3, action="append", required=True,
                    metavar=("NAME", "A2N_DIR", "V2_DIR"))
    ap.add_argument("--arm", default="uncorrected")
    ap.add_argument("--labels", nargs=2, default=("A2n", "v2"), metavar=("A", "B"),
                    help="the two runs' names in the output")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    la, lb = args.labels
    res = {"arm": args.arm}
    for name, a_dir, b_dir in args.pair:
        a, b = load(Path(a_dir), args.arm), load(Path(b_dir), args.arm)
        res[name] = {la: summary(a), lb: summary(b), f"{lb} - {la}": paired(a, b)}
        print(f"\n=== {name} (n = {len(a['mean'])}) ===   " + "".join(f"{n:>12s}" for n in NAMES))
        for run in (la, lb):
            for row in ("median of means", "median posterior sd"):
                print(f"{run:8s} {row:24s}" + "".join(f"{v:12.4f}" for v in res[name][run][row]))
        for row in ("median shift", "median |shift| / A2n sd", "posterior sd ratio, median"):
            print(f"{lb} - {la} {row:20s}"
                  + "".join(f"{v:12.4f}" for v in res[name][f"{lb} - {la}"][row]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
