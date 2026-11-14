"""RESULTS 70: the numbers RESULTS 1 quotes for its reference network that no
other script reports.

The rest of RESULTS 1 comes from committed scripts already: the PSID tables
from ``compare_regen_v2.py``, heterogeneity and resolution from
``household_resolution.py``, calibration from the arm's held-out check and
SBC, the representative theta from ``laibson_comparison.py``. This adds, per
arm:

* PSID's low-beta shares, couples and all 889 (RESULTS 63.5's list): posterior
  means below 0.3, 5th percentiles below 0.3 and below 0.11, 95th percentiles
  below 0.3;
* the representative theta against Laibson et al.'s (RESULTS 63.10's three
  numbers): the difference over the combined standard error, with their
  reported se and with their sampling-only se, and their sampling-only and
  reported se over ours;
* the last simulated point's largest misfit, in PSID standard errors.

v2_b010, RESULTS 1's reference until 2026-11-14, is scored beside the new
one, so the numbers RESULTS 1 quoted for it come back (a check). CPU only;
reads saved outputs.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/reference_summary.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NAMES = ("beta", "delta", "crra", "R_gamma")
LAIBSON_JSON = Path("outputs/lr_plateau/laibson_comparison.json")
ARMS = {   # arm: (PSID couples, PSID all 889)
    "v2_b010": ("outputs/psid_optionA2v2_b010_couples", "outputs/psid_optionA2v2_b010_all"),
    "v2_fresh_rop": ("outputs/lr_plateau/v2_fresh_rop/psid_w07_couples",
                     "outputs/lr_plateau/v2_fresh_rop/psid_w07_all"),
}
OUT = Path("outputs/reference/reference_summary.json")


def low_beta(mean: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    """Shares of households by where beta's posterior sits (lo, hi: the 90%
    interval's ends)."""
    b, lo, hi = mean[:, 0], lo[:, 0], hi[:, 0]
    return {"n": int(len(b)),
            "mean_below_0.3": float((b < 0.3).mean()),
            "p5_below_0.3": float((lo < 0.3).mean()),
            "p5_below_0.11": float((lo < 0.11).mean()),
            "p95_below_0.3": float((hi < 0.3).mean())}


def against_laibson(theta, se, theirs, their_se: dict) -> dict:
    """Ours minus theirs over the combined se, for beta, delta and rho."""
    d = np.asarray(theta[:3]) - np.asarray(theirs[:3])
    se = np.asarray(se[:3])
    rep, samp = (np.asarray(their_se[k][:3]) for k in ("reported", "sampling only"))
    return {"difference": d.tolist(),
            "z_reported": (d / np.hypot(rep, se)).tolist(),
            "z_sampling_only": (d / np.hypot(samp, se)).tolist(),
            "their_sampling_only_over_ours": (samp / se).tolist(),
            "their_reported_over_ours": (rep / se).tolist()}


def arm_summary(name: str, couples: Path, all889: Path, lb: dict) -> dict:
    out = {}
    for label, d in (("couples", couples), ("all 889", all889)):
        z = np.load(Path(d) / "posterior_uncorrected.npz")
        out[label] = low_beta(z["mean"], z["lo"], z["hi"])
    a = lb["arms"][name]
    if not a["converged"]:
        raise SystemExit(f"{name}: the representative theta did not converge")
    out["representative"] = {
        "theta": a["representative_theta"], "se": a["representative_se"],
        **against_laibson(a["representative_theta"], a["representative_se"],
                          lb["laibson"]["theta"], lb["laibson_se"]),
        "max_abs_misfit_over_psid_se": float(np.abs(a["last_simulated_misfit_over_psid_se"]).max()),
        "at_laibson_theta": a["at_laibson"]["mean_of_means"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--laibson", type=Path, default=LAIBSON_JSON,
                    help="laibson_comparison.py's output holding every arm")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    lb = json.loads(args.laibson.read_text())
    res = {a: arm_summary(a, *map(Path, ARMS[a]), lb) for a in args.arms}
    for a, r in res.items():
        print(f"\n=== {a} ===")
        for label in ("couples", "all 889"):
            s = r[label]
            print(f"  {label:8s} n={s['n']}: means < 0.3 {s['mean_below_0.3']:.3f}; 5th pct < 0.3 "
                  f"{s['p5_below_0.3']:.3f}, < 0.11 {s['p5_below_0.11']:.3f}; 95th pct < 0.3 "
                  f"{s['p95_below_0.3']:.3f}")
        p = r["representative"]
        print("  representative θ  " + "  ".join(
            f"{t:.4f} ({s:.4f})" for t, s in zip(p["theta"], p["se"], strict=True)))
        for k in ("difference", "z_reported", "z_sampling_only", "their_sampling_only_over_ours",
                  "their_reported_over_ours"):
            print(f"  {k:30s}" + "".join(f"{v:10.3f}" for v in p[k]))
        print(f"  largest misfit / PSID se       {p['max_abs_misfit_over_psid_se']:.3f}")
        print("  simulated at their θ           "
              + "".join(f"{v:10.4f}" for v in p["at_laibson_theta"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
