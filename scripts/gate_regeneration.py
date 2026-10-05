"""Gates for two changes proposed before regenerating option A (RESULTS.md 41).

1. **The liquidation penalty as a fifth estimated parameter**: a scale ``s`` on
   ``grids.liquidation_penalty`` (``ModelSpec.liqpen_scale``), prior
   ``S_RANGE``.
2. **Initial wealth drawn from PSID's range**: each household's age-20 seed is
   a PSID household's wave-1 (liquid, illiquid) ratio to mean income, drawn
   jointly (``simulate(initial_wealth=...)``), instead of the SCF median for
   everybody.

Both run on the fixed solver (RESULTS 40, ``ModelSpec.crra_centred``). Gate 2
(RESULTS 9.1) rejected initial wealth as "forgotten by age 35", but at
rho = 4.5, where the old solver drove every household to maximum debt and no
illiquid wealth, so that verdict is re-checked here.

**Method: local Fisher information**, at a few theta points. Per household,
summary statistics of a 7-wave window (per wave: log consumption, asinh liquid
and asinh illiquid wealth, each over the household's mean income -- the anchor
the network sees). Their mean's Jacobian in each parameter comes from central
differences on the same households and shocks (common random numbers). Their
covariance comes from the base theta. ``I = J' Sigma^-1 J`` is the information
one household carries about theta through these statistics, and
``sqrt(diag(I^-1))`` the Cramer-Rao sd. In prior-sd units below 1 means
informative. The statistics are a coarse summary of what the network uses, so
absolute values are pessimistic; the comparisons are the point:

* does ``s`` carry information, and what does adding it cost beta, delta, rho
  and R_gamma (their CR sd with ``s`` estimated over ``s`` known)?
* does random initial wealth persist into the observation window, and what
  does it cost in information about theta?

Usage::

    PYTHONPATH=. .venv/bin/python scripts/gate_regeneration.py --out outputs/gate_regeneration.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.waves import FEATURES_TWOASSET, aggregate_waves
from hh_npe.simulator import laibson_calibration as cal
from hh_npe.simulator.dispatch import AGE_START_SIM
from hh_npe.simulator.twoasset import GRIDS, simulate
from hh_npe.simulator.twoasset_gpu import solve_batch
from scripts.gate_within_theta import psid_initial_ratios

NAMES = ("beta", "delta", "rho", "R_gamma", "s")
#: Prior sd of each parameter: the option A box, and s uniform on S_RANGE.
S_RANGE = (0.25, 1.5)
PRIOR_SD = np.array([0.7, 0.15, 4.5, 0.05, S_RANGE[1] - S_RANGE[0]]) / np.sqrt(12)
STEP = np.array([0.03, 0.002, 0.2, 0.004, 0.15])
POINTS = {
    "mid rho": (0.77, 0.988, 2.0, 1.05, 1.0),
    "high rho": (0.77, 0.988, 4.0, 1.05, 1.0),
    "Gate 2 theta": (0.8465, 0.9898, 4.5, 1.05, 1.0),
    "present-biased": (0.55, 0.975, 3.0, 1.05, 1.0),
}
START_AGES = (28, 40)
PSID_X = "data/processed/psid_x_comphs_optionA.pt"


def spec_for(r_gamma: float, s: float, card: int = 1):
    calib = cal.COMPHS
    if not card:
        calib = dataclasses.replace(calib, c0_credit=0.0, c1_credit=0.0, c2_credit=0.0)
    return dataclasses.replace(GRIDS["full"], calib=calib, R_gamma=float(r_gamma),
                               liqpen_scale=float(s))


def perturbed(theta: np.ndarray) -> list[np.ndarray]:
    """Base, then -/+ STEP in each parameter, in NAMES order."""
    out = [theta.copy()]
    for k in range(len(NAMES)):
        for sign in (-1, 1):
            t = theta.copy()
            t[k] += sign * STEP[k]
            out.append(t)
    return out


def solve_all(thetas: list[np.ndarray], card: int):
    """Solve each theta, batching those that share (R_gamma, s)."""
    groups: dict = {}
    for i, t in enumerate(thetas):
        groups.setdefault((t[3], t[4]), []).append(i)
    sols = [None] * len(thetas)
    for (r, s), idx in groups.items():
        out = solve_batch(np.array([thetas[i][:3] for i in idx]), spec_for(r, s, card),
                          theta_batch=16, chunk=16)
        for i, sol in zip(idx, out):
            sols[i] = sol
        torch.cuda.empty_cache()
    return sols


def stats(panel: dict, start_age: int) -> np.ndarray:
    x, _ = aggregate_waves(panel, age_start_sim=AGE_START_SIM, start_age=start_age,
                           n_waves=7, wave_years=2, features=FEATURES_TWOASSET)
    x = x.astype(float)
    ybar = np.maximum(x[:, :, 0].mean(1, keepdims=True), 1.0)
    return np.concatenate([np.log(np.maximum(x[:, :, 1], 1.0) / ybar),
                           np.arcsinh(x[:, :, 2] / ybar), np.arcsinh(x[:, :, 3] / ybar)], 1)


def fisher(base: np.ndarray, plus: list, minus: list, idx) -> np.ndarray:
    """Per-household information about the parameters ``idx`` (prior-sd units)."""
    J = np.stack([(plus[k].mean(0) - minus[k].mean(0)) / (2 * STEP[k] / PRIOR_SD[k])
                  for k in idx], 1)
    S = np.cov(base, rowvar=False)
    S += 1e-6 * np.trace(S) / len(S) * np.eye(len(S))
    return J.T @ np.linalg.solve(S, J)


def cr_sd(I: np.ndarray) -> np.ndarray:
    return np.sqrt(np.diag(np.linalg.inv(I)))


def bands(fixed: dict, rand: dict) -> dict:
    """Gate 2's persistence table: sd ratio random / fixed seed, by age band."""
    out = {}
    for lo, hi in ((25, 30), (31, 34), (35, 44), (45, 55)):
        t = slice(lo - AGE_START_SIM, hi - AGE_START_SIM + 1)
        out[f"{lo}-{hi}"] = {k: float(rand[k][:, t].std() / max(fixed[k][:, t].std(), 1.0))
                             for k in ("liquid_assets", "illiquid_assets")}
    return out


def followup(args) -> None:
    """Two checks the local gate cannot make.

    * ``s`` far from 1: its prior reaches 0.25 and 1.5, where behaviour may move
      even though it is flat around 1.
    * young-age dispersion against PSID: a PSID seed can only help if PSID's
      young households are *more* dispersed than the model's (RESULTS 9.6 found
      the opposite, on the broken solver and without pensions).
    """
    d = torch.load(PSID_X, weights_only=False)
    x = d["x"].numpy()
    age = x[:, :, 4].ravel()
    rng = np.random.default_rng(0)
    ratios = psid_initial_ratios(PSID_X)
    seed_rows = ratios[rng.integers(0, len(ratios), args.n)]
    res = {"s_global": {}, "dispersion": {}}

    print("== s across its prior: cardholders, ages 40-44 ==")
    for name in ("high rho", "present-biased"):
        th = np.array(POINTS[name])
        for sv in (0.25, 0.5, 1.0, 1.5):
            sol = solve_batch(th[None, :3], spec_for(th[3], sv), theta_batch=16, chunk=16)[0]
            p = simulate(sol, n_households=args.n, seed=1)
            z = p["illiquid_assets"][:, 18:27]
            dz = z[:, 2:] - z[:, :-2]
            r = {"debt": float((p["liquid_assets"][:, 20:25] < 0).mean()),
                 "illiquid_median": float(np.median(p["illiquid_assets"][:, 20:25])),
                 "illiquid_zero": float((p["illiquid_assets"][:, 20:25] < 1).mean()),
                 "big_drop": float((dz < -np.maximum(0.5 * z[:, :-2], 20_000)).mean()),
                 "c_over_y": float(np.median(p["consumption"][:, 20:25] / p["income"][:, 20:25]))}
            res["s_global"][f"{name} s={sv}"] = r
            print(f"  {name:15s} s {sv:4.2f}: debt {r['debt']:.0%}  illiquid median "
                  f"{r['illiquid_median']:>9,.0f}  zero {r['illiquid_zero']:.0%}  big 2-yr drops "
                  f"{r['big_drop']:.1%}  c/y {r['c_over_y']:.2f}", flush=True)
            del sol
            torch.cuda.empty_cache()

    print("\n== dispersion by age, PSID (gross liquid, illiquid with pensions) against the "
          "model, cardholder / no-card 50/50 ==")
    def q(v):
        return np.percentile(v, [10, 25, 50, 75, 90])
    bands_ = ((25, 30), (31, 34), (35, 44))
    for lo, hi in bands_:
        sel = (age >= lo) & (age <= hi)
        res["dispersion"][f"PSID {lo}-{hi}"] = {
            k: q(x[:, :, j].ravel()[sel]).tolist() for k, j in (("liquid", 2), ("illiquid", 3))}
    for name in ("mid rho", "high rho", "present-biased"):
        th = np.array(POINTS[name])
        panels = {}
        for card in (1, 0):
            sol = solve_batch(th[None, :3], spec_for(th[3], th[4], card), theta_batch=16, chunk=16)[0]
            panels[card] = {"fixed": simulate(sol, n_households=args.n, seed=1),
                            "PSID seed": simulate(sol, n_households=args.n, seed=1,
                                                  initial_wealth=seed_rows)}
            del sol
            torch.cuda.empty_cache()
        for init in ("fixed", "PSID seed"):
            for lo, hi in bands_:
                t = slice(lo - AGE_START_SIM, hi - AGE_START_SIM + 1)
                res["dispersion"][f"{name} {init} {lo}-{hi}"] = {
                    k: q(np.concatenate([panels[c][init][f][:, t].ravel() for c in (1, 0)])).tolist()
                    for k, f in (("liquid", "liquid_assets"), ("illiquid", "illiquid_assets"))}
    print(f"{'':34s}{'illiquid p10 p25 p50 p75 p90':>44s}   liquid p10/p50/p90")
    for k, v in res["dispersion"].items():
        print(f"  {k:32s}" + "".join(f"{a:>9,.0f}" for a in v["illiquid"])
              + "   " + "/".join(f"{a:,.0f}" for a in np.array(v["liquid"])[[0, 2, 4]]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_name(args.out.stem + "_followup.json").write_text(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--card", type=int, default=1)
    ap.add_argument("--points", nargs="+", default=list(POINTS))
    ap.add_argument("--out", type=Path, default=Path("outputs/gate_regeneration.json"))
    ap.add_argument("--followup", action="store_true",
                    help="s across its prior, and young-age dispersion against PSID.")
    args = ap.parse_args()
    if args.followup:
        return followup(args)

    rng = np.random.default_rng(0)
    ratios = psid_initial_ratios(PSID_X)
    seed_rows = ratios[rng.integers(0, len(ratios), args.n)]
    print(f"PSID wave-1 seeds: liquid ratio p10/p50/p90 "
          f"{np.percentile(ratios[:, 0], [10, 50, 90]).round(2)}, illiquid "
          f"{np.percentile(ratios[:, 1], [10, 50, 90]).round(2)}", flush=True)

    res = {"step": STEP.tolist(), "prior_sd": PRIOR_SD.tolist(), "s_range": S_RANGE,
           "n": args.n, "card": args.card, "points": {}}
    for name in args.points:
        theta = np.array(POINTS[name], dtype=float)
        thetas = perturbed(theta)
        sols = solve_all(thetas, args.card)
        sims = {"fixed": [simulate(s, n_households=args.n, seed=1) for s in sols],
                "random": [simulate(s, n_households=args.n, seed=1, initial_wealth=seed_rows)
                           for s in sols]}
        del sols
        torch.cuda.empty_cache()
        r = res["points"][name] = {"theta": theta.tolist(),
                                   "persistence": bands(sims["fixed"][0], sims["random"][0])}
        print(f"\n=== {name}: theta {theta.tolist()} ===", flush=True)
        for a in START_AGES:
            st = {k: [stats(p, a) for p in v] for k, v in sims.items()}
            ra = r[f"start {a}"] = {}
            for init, v in st.items():
                base, minus, plus = v[0], v[1::2], v[2::2]
                I5 = fisher(base, plus, minus, range(5))
                I4 = fisher(base, plus, minus, range(4))
                ra[init] = {"cr_sd_4": cr_sd(I4).tolist(), "cr_sd_5": cr_sd(I5).tolist(),
                            "s_known_info": float(I5[4, 4])}
            f, rd = ra["fixed"], ra["random"]
            print(f"  window from {a}: CR sd per household / prior sd")
            print(f"    {'':28s}" + "".join(f"{n:>9s}" for n in NAMES))
            print(f"    {'fixed seed, 4 params':28s}" + "".join(f"{v:9.2f}" for v in f["cr_sd_4"]))
            print(f"    {'fixed seed, + s estimated':28s}" + "".join(f"{v:9.2f}" for v in f["cr_sd_5"]))
            print(f"    {'PSID seed, 4 params':28s}" + "".join(f"{v:9.2f}" for v in rd["cr_sd_4"]))
            print(f"    {'PSID seed, + s estimated':28s}" + "".join(f"{v:9.2f}" for v in rd["cr_sd_5"]))
        print("  persistence, sd ratio PSID seed / fixed seed: " + "  ".join(
            f"{b}: liq {v['liquid_assets']:.2f} illiq {v['illiquid_assets']:.2f}"
            for b, v in r["persistence"].items()), flush=True)
        # What s does to what PSID shows: big illiquid drops and debt, ages 40-44.
        desc = {}
        for lab, i in (("s -", 9), ("base", 0), ("s +", 10)):
            p = sims["fixed"][i]
            z = p["illiquid_assets"][:, 18:27]
            dz = z[:, 2:] - z[:, :-2]
            desc[lab] = {"debt_40_44": float((p["liquid_assets"][:, 20:25] < 0).mean()),
                         "illiquid_median_40_44": float(np.median(p["illiquid_assets"][:, 20:25])),
                         "big_two_year_drop": float((dz < -np.maximum(0.5 * z[:, :-2], 20_000)).mean())}
        r["s_effects"] = desc
        print("  s = 0.85 / 1.0 / 1.15: " + "  ".join(
            f"{k}: debt {v['debt_40_44']:.0%}, illiquid med {v['illiquid_median_40_44']:,.0f}, "
            f"big 2-yr drops {v['big_two_year_drop']:.1%}" for k, v in desc.items()), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
