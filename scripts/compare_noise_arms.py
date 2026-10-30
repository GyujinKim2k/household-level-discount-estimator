"""RESULTS 58: v2 against its PSID-only noise arms (RESULTS 57.4), side by side.

Each ``--arm NAME RUN_DIR PSID_COUPLES PSID_ALL``:
- ``RUN_DIR`` holds the members (``s*/``), ``heldout_check.json`` and
  ``sbc/sbc16.json`` (``run_v2_noise_arms.sh`` / ``run_regen_v2.sh``);
- the PSID directories are ``optionA.py psid`` outputs.

The first arm is the reference for the paired per-household shifts.

Each optional ``--seed_pair NAME PSID_A PSID_B`` gives the seed baseline for
those shifts: PSID couples' posteriors of two disjoint member pairs of one arm
(``optionA.py psid --run_dirs s0 s1`` and ``s2 s3``), compared the same way.

Reported per arm:
- the noise each was trained with;
- the held-out check and SBC, each on windows carrying the arm's own noise;
- the liquidation-penalty shifts;
- PSID couples and all 889 in RESULTS 1's template (``compare_regen_v2.summary``);
- the floors (``check_optionA2_heldout.floors``);
- household by household against the reference: the shift in units of the
  reference's posterior sd, and the ratio of posterior sds.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/compare_noise_arms.py \\
        --arm v2 outputs/optionA2v2 outputs/psid_optionA2v2_couples outputs/psid_optionA2v2_all \\
        --arm typical ... --arm gross ... --out outputs/v2_noise_arms/comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.check_optionA2_heldout import floors
from scripts.compare_regen_v2 import NAMES, load, summary

HELDOUT = ("cov90", "corr", "mae_over_prior_sd", "contraction")
FLOORS = ("rho_lo_lt_0.55", "R_lo_lt_floor+0.0025", "either")


def noise_of(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "s0" / "results.json").read_text())["_config"]
    return cfg.get("obs_noise", {})


def heldout(run_dir: Path) -> dict:
    e = json.loads((run_dir / "heldout_check.json").read_text())["ensemble"]
    return {"log_q_weighted": e["log_q_weighted"],
            **{n: {k: e[n][k] for k in HELDOUT} for n in NAMES}}


def sbc(run_dir: Path) -> dict:
    s = json.loads((run_dir / "sbc" / "sbc16.json").read_text())
    out = {"coverage_90": {n: s["sbc"]["coverage_90"][n]["rate"] for n in NAMES},
           "ks_p": {n: s["sbc"]["ks_p_one_per_draw"][n] for n in NAMES}, "penalty": {}}
    for scale, r in s["penalty"]["by_scale"].items():
        out["penalty"][scale] = {n: {"shift_median": r[n]["shift_over_prior_sd_median"],
                                     "shift_p10": r[n]["shift_over_prior_sd_p10"],
                                     "shift_p90": r[n]["shift_over_prior_sd_p90"],
                                     "cov90": r[n]["cov90"]} for n in NAMES}
    return out


def paired(a: dict, b: dict) -> dict:
    """b minus the reference a, household by household."""
    if not np.array_equal(a["index"], b["index"]):
        raise SystemExit("the two runs are not over the same PSID households")
    d = b["mean"] - a["mean"]
    z = d / a["sd"]
    return {"median shift": np.median(d, 0).tolist(),
            "median |shift| / ref sd": np.median(np.abs(z), 0).tolist(),
            "corr of means": [float(np.corrcoef(a["mean"][:, j], b["mean"][:, j])[0, 1])
                              for j in range(len(NAMES))],
            "posterior sd ratio, median": np.median(b["sd"] / a["sd"], 0).tolist()}


def report(res: dict) -> None:
    arms = list(res["arms"])
    row = lambda label, vals, fmt="{:9.3f}": print(  # noqa: E731
        f"  {label:34s}" + "".join(fmt.format(v) for v in vals))
    print("noise (sigma range per feature)")
    for a in arms:
        print(f"  {a:8s} " + ", ".join(f"{f} {lo}-{hi}" for f, (lo, hi)
                                     in res["arms"][a]["noise"].items()))
    print("\nheld-out (own noise), per parameter beta / delta / rho / R_gamma")
    for k in HELDOUT:
        for a in arms:
            row(f"{k}, {a}", [res["arms"][a]["heldout"][n][k] for n in NAMES])
    for a in arms:
        row(f"log q weighted, {a}", [res["arms"][a]["heldout"]["log_q_weighted"]])
    print("\nSBC (own noise)")
    for k in ("coverage_90", "ks_p"):
        for a in arms:
            row(f"{k}, {a}", [res["arms"][a]["sbc"][k][n] for n in NAMES])
    print("\npenalty x0.5 / x1.5: median shift / prior sd")
    for scale in ("0.5", "1.5"):
        for a in arms:
            row(f"x{scale}, {a}",
                [res["arms"][a]["sbc"]["penalty"][scale][n]["shift_median"] for n in NAMES])
    for grp in ("couples", "all"):
        print(f"\nPSID {grp}")
        for k in ("median of means", "median posterior sd", "covers Laibson (R_gamma 1.05)"):
            for a in arms:
                row(f"{k[:22]}, {a}", res["arms"][a][grp][k], "{:9.4f}")
        for k, v in res["paired"].items():
            for q in ("median shift", "median |shift| / ref sd", "corr of means",
                      "posterior sd ratio, median"):
                row(f"{q[:24]}, {k}", v[grp][q], "{:9.4f}")
    if res["seed_pairs"]:
        print("\nseed baseline, PSID couples: members s0+s1 against s2+s3 of one arm")
        for k, v in res["seed_pairs"].items():
            for q in ("median |shift| / ref sd", "corr of means"):
                row(f"{q[:24]}, {k}", v[q], "{:9.4f}")
    print("\nfloors (couples / all 889): rho 5% < 0.55, R_gamma 5% < floor + 0.0025, either")
    for a in arms:
        f = res["arms"][a]["floors"]
        print(f"  {a:8s} " + "   ".join(
            f"{g}: " + " ".join(f"{f[g][k]:.3f}" for k in FLOORS) for g in f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs=4, action="append", required=True,
                    metavar=("NAME", "RUN_DIR", "PSID_COUPLES", "PSID_ALL"))
    ap.add_argument("--seed_pair", nargs=3, action="append", default=[],
                    metavar=("NAME", "PSID_A", "PSID_B"))
    ap.add_argument("--psid_arm", default="uncorrected")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = {"psid_arm": args.psid_arm, "reference": args.arm[0][0], "arms": {}, "paired": {},
           "seed_pairs": {name: paired(load(Path(a), args.psid_arm), load(Path(b), args.psid_arm))
                          for name, a, b in args.seed_pair}}
    post = {}
    for name, run_dir, pc, pa in args.arm:
        run_dir = Path(run_dir)
        post[name] = {"couples": load(Path(pc), args.psid_arm),
                      "all": load(Path(pa), args.psid_arm)}
        fl = floors([f"couples={pc}", f"all 889={pa}"])
        res["arms"][name] = {"noise": noise_of(run_dir), "heldout": heldout(run_dir),
                             "sbc": sbc(run_dir),
                             "floors": {g: {k: fl[g][k] for k in FLOORS}
                                        for g in ("couples", "all 889")},
                             **{g: summary(p) for g, p in post[name].items()}}
    ref = args.arm[0][0]
    for name in list(post)[1:]:
        res["paired"][f"{name} - {ref}"] = {g: paired(post[ref][g], post[name][g])
                                            for g in ("couples", "all")}
    report(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
