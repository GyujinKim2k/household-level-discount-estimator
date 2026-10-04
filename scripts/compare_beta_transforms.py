"""beta-transform screen (RESULTS.md 37): option A with beta's flow target linear,
``log(1 - beta)`` or logit, five seeds each, judged on criteria fixed before
training.

The defect being targeted (§36.1): where beta's posterior mean is in
[0.8, 0.95), the truth lies above the 95th percentile 11.4% of the time and
below the 5th 2.7% -- an upper tail too short near the bound at 1.

Criteria, in order:

1. **The tail.** In that region, both miss rates at most 0.075 (5% is
   calibrated; one binomial standard error is ~0.015 at n ~ 220), on at least
   100 draws.
2. **beta overall.** SBC coverage in [0.88, 0.92] and rank KS p >= 0.05.
3. **No collateral damage.** delta, rho and R_gamma coverage each within 0.02
   of the linear arm's.
4. **Recovery.** beta's error over the prior sd no worse than linear's by more
   than 0.01. Held-out ``log q`` (beta on its own scale) breaks ties.

An arm passing 1-3 replaces linear; if none does, beta stays linear (§34).

Usage::

    PYTHONPATH=. .venv/bin/python scripts/compare_beta_transforms.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hh_npe.npe.prior import PHASE3_RGAMMA as BOX

ARMS = {"linear": ("outputs/optionA", "outputs/psid_optionA"),
        "log1m": ("outputs/optionA_beta_log1m", "outputs/psid_optionA_beta_log1m"),
        "logit": ("outputs/optionA_beta_logit", "outputs/psid_optionA_beta_logit")}
REGION = (0.8, 0.95)


def score(run: Path, psid: Path) -> dict:
    ev = json.loads((run / "evaluation" / "evaluation.json").read_text())
    s, h = ev["ensemble"]["sbc"], ev["ensemble"]["heldout_q"]
    reg = next(b for b in s["by_posterior_mean"]["beta"] if (b["lo"], b["hi"]) == REGION)
    top = next(b for b in s["by_true_theta"]["beta"] if b["lo"] == 0.95)
    out = {
        "beta_transform": ev["config"].get("beta_transform", "linear"),
        "tail_region_n": reg["n"], "tail_above_95": reg["above_95"],
        "tail_below_5": reg["below_5"], "tail_coverage": reg["coverage_90"],
        "n_post_mean_ge_095": next(b for b in s["by_posterior_mean"]["beta"]
                                   if b["lo"] == 0.95)["n"],
        "coverage_true_ge_095": top["coverage_90"],
        "coverage": s["coverage_90"], "ks_p": s["ks_p"],
        "recovery": {n: {k: s["recovery"][n][k] for k in ("corr", "mae_over_prior_sd",
                                                            "contraction")}
                     for n in BOX.names},
        "heldout_log_q": h["log_q"],
    }
    sm = json.loads((psid / "summary.json").read_text())["uncorrected"]
    z = np.load(psid / "posterior_uncorrected.npz")
    ok = np.isfinite(z["mean"][:, 0])
    out["psid"] = {"median_of_means": sm["median_of_means"],
                   "beta_median_90_width": sm["median_90_width"]["beta"],
                   "beta_median_upper_limit": float(np.median(z["hi"][ok, 0])),
                   "beta_median_lower_limit": float(np.median(z["lo"][ok, 0])),
                   "beta_ci_covers_laibson": sm["share_ci_covers_reference"]["beta"],
                   "share_in_meta": sm["share_mean_in_meta_range"],
                   "share_all_three_in_meta": sm["share_all_three_in_meta_range"],
                   "ess_p10": sm["ess_p10"], "in_box_median": sm["in_box_median"]}
    return out


def verdict(r: dict, lin: dict) -> dict:
    c = {
        "1_tail": (r["tail_region_n"] >= 100 and r["tail_above_95"] <= 0.075
                   and r["tail_below_5"] <= 0.075),
        "2_beta_overall": 0.88 <= r["coverage"]["beta"] <= 0.92 and r["ks_p"]["beta"] >= 0.05,
        "3_no_collateral": all(abs(r["coverage"][n] - lin["coverage"][n]) <= 0.02
                               for n in ("delta", "crra", "R_gamma")),
        "4_recovery": (r["recovery"]["beta"]["mae_over_prior_sd"]
                       <= lin["recovery"]["beta"]["mae_over_prior_sd"] + 0.01),
    }
    c["passes_1_to_3"] = c["1_tail"] and c["2_beta_overall"] and c["3_no_collateral"]
    return c


def show(res: dict) -> None:
    arms = list(res)
    row = lambda label, f: print(f"{label:38s}" + "".join(f"{f(res[a]):>12s}" for a in arms))
    print(f"{'':38s}" + "".join(f"{a:>12s}" for a in arms))
    print(f"-- beta tail, posterior mean in [{REGION[0]}, {REGION[1]}) (5% / 5% if calibrated)")
    row("  n", lambda r: f"{r['tail_region_n']}")
    row("  truth above 95th pct", lambda r: f"{r['tail_above_95']:.3f}")
    row("  truth below 5th pct", lambda r: f"{r['tail_below_5']:.3f}")
    row("  coverage", lambda r: f"{r['tail_coverage']:.3f}")
    row("  n with posterior mean >= 0.95", lambda r: f"{r['n_post_mean_ge_095']}")
    row("  coverage, true beta >= 0.95", lambda r: f"{r['coverage_true_ge_095']:.3f}")
    print("-- SBC, ensemble")
    for n in BOX.names:
        row(f"  {n} coverage / KS p",
            lambda r, n=n: f"{r['coverage'][n]:.3f}/{r['ks_p'][n]:.3f}")
    print("-- recovery on SBC draws: corr / mae over prior sd")
    for n in BOX.names:
        row(f"  {n}", lambda r, n=n: f"{r['recovery'][n]['corr']:.3f}/"
                                     f"{r['recovery'][n]['mae_over_prior_sd']:.3f}")
    row("held-out log q (beta own scale)", lambda r: f"{r['heldout_log_q']:.3f}")
    print("-- PSID comphs (uncorrected)")
    for n in BOX.names:
        row(f"  median {n}", lambda r, n=n: f"{r['psid']['median_of_means'][n]:.4f}")
    row("  beta median 90% limits", lambda r: f"{r['psid']['beta_median_lower_limit']:.3f}-"
                                              f"{r['psid']['beta_median_upper_limit']:.3f}")
    row("  beta CI covers 0.53", lambda r: f"{r['psid']['beta_ci_covers_laibson']:.1%}")
    row("  all three in meta range", lambda r: f"{r['psid']['share_all_three_in_meta']:.1%}")
    row("  ESS p10", lambda r: f"{r['psid']['ess_p10']:.0f}")
    print("-- criteria (fixed before training)")
    for k in ("1_tail", "2_beta_overall", "3_no_collateral", "4_recovery", "passes_1_to_3"):
        row(f"  {k}", lambda r, k=k: "yes" if r["verdict"][k] else "no")


def figure(res: dict, psids: dict, path: Path) -> None:
    """Household posterior means per arm, on the 4-parameter box."""
    from hh_npe.evaluation.plots import contour_corner
    from scripts.literature_ranges import LAIBSON, META

    series = {}
    for a, p in psids.items():
        m = np.load(Path(p) / "posterior_uncorrected.npz")["mean"]
        m = m[np.isfinite(m[:, 0])]
        series[f"beta target {a}  N={len(m)}"] = m
    names = np.array(BOX.names)
    contour_corner(
        series, BOX,
        truth={"Laibson et al. MSM (R_gamma: calibrated 1.05)": np.append(LAIBSON, 1.05)},
        truth_markers=("*",), truth_colors=("#d62728",),
        bands={"meta-analytic range (lit.; R_gamma none)":
               (np.append(META[0], BOX.rgamma_low), np.append(META[1], BOX.rgamma_high))},
        reflect_axes=("delta",),
        axis_limits=(BOX.low, np.where(names == "delta", 1.04, BOX.high)),
        path=path,
        title="Per-household posterior means, PSID comphs, option A by beta target\n"
              "importance-weighted to the uniform prior; card type marginalised")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("outputs/beta_transform_screen.json"))
    ap.add_argument("--figure", type=Path,
                    default=Path("figures/33_beta_transform_psid.png"))
    args = ap.parse_args()
    res = {a: score(Path(r), Path(p)) for a, (r, p) in ARMS.items()}
    for a in res:
        res[a]["verdict"] = verdict(res[a], res["linear"])
    show(res)
    args.out.write_text(json.dumps(res, indent=2))
    figure(res, {a: p for a, (_r, p) in ARMS.items()}, args.figure)
    print(f"\nwrote {args.out} and {args.figure}")


if __name__ == "__main__":
    main()
