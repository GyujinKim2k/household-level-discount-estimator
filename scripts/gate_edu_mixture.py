"""Gate 2b: does *persistent* heterogeneity widen p(x|theta) where the initial
condition could not?

Gate 2 (``gate_within_theta.py``) found that randomising age-20 wealth is
forgotten by the ages we observe: the widening ratio is 1.38 at ages 25-30,
1.05 at 31-34 and 1.00 from 35 on. A buffer-stock model converges to its ergodic
distribution and loses its seed. That verdict applies **only** to the initial
condition, and says nothing about heterogeneity that acts at every age.

Education group is the opposite kind of source. It moves four calibration blocks
coherently and permanently -- the income profile (``agecoeff`` 0.079 / 0.135 /
0.247 across somehs / comphs / compco), the AR(1), the credit limit and initial
wealth -- so it cannot be forgotten. This solves the same theta under each
group's own calibration, pools the three, and asks whether the mixture is wider
than comphs alone at the ages the PSID sample actually covers.

The groups are drawn uniformly, matching the plan: a population-representative
result is recovered afterwards by reweighting.

Usage::

    uv run python scripts/gate_edu_mixture.py --n 3000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io
import torch

from hh_npe.simulator import laibson_calibration as cal
from hh_npe.simulator.twoasset import GRIDS, simulate
from hh_npe.simulator.twoasset_gpu import solve_batch

FEATURES = ("income", "consumption", "liquid_assets", "illiquid_assets")
AGE_START_SIM = 20
GROUPS = ("comphs", "somehs", "compco")
MAT = Path("replication-package-LLMRT/LifecycleSimulation/input")

#: cal globals set from each .mat, in the order the arrays store them.
INCOME = ("YWORK_KIDSCOEFF", "YWORK_SPOUSECOEFF", "YWORK_DEPADULCOEFF",
          "YWORK_AGECOEFF", "YWORK_AGE2COEFF", "YWORK_AGE3COEFF", "YWORK_CONS")
AR1 = ("YWORK_AUTO", "YWORK_VAREPS", "YWORK_VARNU")
CREDIT = ("C0_CREDIT", "C1_CREDIT", "C2_CREDIT")
INITW = ("MED_TOTAL_WEALTH", "MED_LIQ_WEALTH")


def load_group(g: str) -> dict:
    out = {}
    # est_firststage_income.mat holds TWO arrays; name the key rather than
    # taking the first, which silently picked est_ar1 for the profile.
    for f, key, names in (("income", "est_income", INCOME),
                          ("income", "est_ar1", AR1),
                          ("creditlim", "est_creditlim", CREDIT),
                          ("initwealth", "est_initwealth", INITW)):
        m = scipy.io.loadmat(MAT / g / f"est_firststage_{f}.mat", squeeze_me=True)
        out.update(dict(zip(names, np.asarray(m[key]).ravel())))
    # sigmas are derived, not stored: keep them consistent with the variances.
    out["YWORK_SIGMAEPS"] = float(np.sqrt(out["YWORK_VAREPS"]))
    out["YWORK_SIGMANU"] = float(np.sqrt(out["YWORK_VARNU"]))
    return out


def apply_group(vals: dict):
    """Set the module globals grids.py reads at call time; return the old ones.

    Monkeypatching rather than threading through ModelSpec: this is a gate, and
    the plumbing (plan section 6) is only worth building if the gate passes.
    """
    old = {k: getattr(cal, k) for k in vals}
    for k, v in vals.items():
        setattr(cal, k, v)
    return old


def sim_group(g: str, theta, n: int, grid: str, seed: int) -> dict:
    vals = load_group(g)
    old = apply_group(vals)
    try:
        sol = solve_batch(np.array([theta]), spec=GRIDS[grid],
                          theta_batch=1, chunk=16)[0]
        return simulate(sol, n_households=n, seed=seed)
    finally:
        apply_group(old)


def stats(panels, lo, hi, key):
    t0, t1 = lo - AGE_START_SIM, hi - AGE_START_SIM + 1
    v = np.concatenate([p[key][:, t0:t1].ravel() for p in panels])
    return (np.percentile(v, 75) - np.percentile(v, 25), v.std(), np.median(v))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theta", type=float, nargs=3,
                    default=[0.8465, 0.9898, 4.5002])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--grid", default="full")
    ap.add_argument("--x", default="data/processed/psid_x_rental.pt")
    args = ap.parse_args()

    print("per-group calibration (the four blocks education moves at once)")
    print(f"{'group':9s}{'auto':>8s}{'vareps':>9s}{'varnu':>8s}"
          f"{'agecoef':>9s}{'cons':>8s}{'C0_cred':>9s}{'med_liq':>9s}")
    g_vals = {}
    for g in GROUPS:
        v = g_vals[g] = load_group(g)
        print(f"{g:9s}{v['YWORK_AUTO']:8.4f}{v['YWORK_VAREPS']:9.4f}"
              f"{v['YWORK_VARNU']:8.4f}{v['YWORK_AGECOEFF']:9.4f}"
              f"{v['YWORK_CONS']:8.3f}{v['C0_CREDIT']:9.5f}"
              f"{v['MED_LIQ_WEALTH']:+9.4f}")

    panels = {}
    for i, g in enumerate(GROUPS):
        print(f"\nsolving {g} at theta = {tuple(args.theta)} ...", flush=True)
        panels[g] = sim_group(g, args.theta, args.n // 3, args.grid, seed=1 + i)

    e = torch.load(args.x, weights_only=False)["x"].numpy()
    a = e[:, :, 4].ravel()
    mix = list(panels.values())
    base = [panels["comphs"]]

    print(f"\nwithin-theta spread: comphs alone vs the 3-group mixture "
          f"(uniform), N={args.n}")
    print(f"{'band':9s}{'feature':17s}{'comphs IQR':>12s}{'mix IQR':>10s}"
          f"{'ratio':>7s}{'PSID IQR':>10s}{'mix/PSID':>10s}")
    ratios = []
    for lo, hi in ((25, 30), (31, 34), (35, 44), (45, 55)):
        sel = (a >= lo) & (a <= hi)
        for j, k in enumerate(FEATURES):
            b_i = stats(base, lo, hi, k)[0]
            m_i = stats(mix, lo, hi, k)[0]
            pv = e[:, :, j].ravel()[sel]
            pi = np.percentile(pv, 75) - np.percentile(pv, 25)
            ratios.append(m_i / max(b_i, 1))
            print(f"{f'{lo}-{hi}':9s}{k:17s}{b_i:12.0f}{m_i:10.0f}"
                  f"{m_i / max(b_i, 1):7.2f}{pi:10.0f}{m_i / max(pi, 1):10.2f}")

    print(f"\nmean widening from the education mixture: {np.mean(ratios):.2f}x")
    print("Compare against 1.00x for randomised initial wealth at ages 35+.")


if __name__ == "__main__":
    main()
