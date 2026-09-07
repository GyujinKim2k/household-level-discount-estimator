"""Gate 2c: sizing the two remaining heterogeneity sources against education.

Gate 2 killed randomised initial wealth (forgotten by age 35, 1.00x). Gate 2b
showed education group works (1.79x, growing with age). This sizes the other two
the plan proposes, so the regeneration carries only sources that earn their cost:

* **Income process.** Drawn from ``VCV_firststage_income.mat`` (500 bootstrap
  reps). As the plan already flags, this is *sampling uncertainty in a
  group-level estimate*, not household heterogeneity -- the right magnitude
  check is whether it is even comparable to the between-group spread.
* **Returns.** ``R_gamma`` is a ``ModelSpec`` field, so no monkeypatching. There
  is no calibrated dispersion for it at all; +-2pp around their 1.05 benchmark
  is a chosen sensitivity in the range Fagereng et al. (2020) report for
  persistent household return heterogeneity, and must be reported as chosen.

Usage::

    uv run python scripts/gate_nuisance_mixture.py --n 3000
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import scipy.io
import torch

from hh_npe.simulator import laibson_calibration as cal
from hh_npe.simulator.twoasset import GRIDS, simulate
from hh_npe.simulator.twoasset_gpu import solve_batch
from scripts.gate_edu_mixture import (AR1, FEATURES, INCOME, MAT, apply_group,
                                      load_group, stats)


#: The single 10x10 `VCV_income` covers the whole first stage: the 7 profile
#: coefficients followed by the 3 AR(1) parameters, in the order the two
#: estimate arrays store them. Drawing all 10 jointly is both more correct and
#: strictly more generous than drawing the AR(1) alone.
NAMES10 = INCOME + AR1


def income_draws(g: str, n: int, seed: int) -> list[dict]:
    """Joint bootstrap draws of the group's whole first-stage income estimate."""
    base = load_group(g)
    V = np.asarray(scipy.io.loadmat(MAT / g / "VCV_firststage_income.mat",
                                    squeeze_me=True)["VCV_income"])
    mu = np.array([base[k] for k in NAMES10])
    sd = np.sqrt(np.diag(V))
    print(f"  {g} bootstrap sd, as a fraction of |estimate|:")
    for nm, m_, s_ in zip(NAMES10, mu, sd):
        print(f"    {nm:20s}{m_:+10.4f} +- {s_:.4f}  ({s_ / abs(m_):.1%})")
    rng = np.random.default_rng(seed)
    out = []
    for d in rng.multivariate_normal(mu, V, size=n):
        o = dict(zip(NAMES10, map(float, d)))
        # Variances must stay positive and |psi| < 1, or tauchen's
        # sqrt(vareps/(1-psi^2)) is undefined. Clip rather than reject: a
        # rejected draw would silently reweight the mixture.
        o["YWORK_AUTO"] = float(np.clip(o["YWORK_AUTO"], 0.0, 0.99))
        o["YWORK_VAREPS"] = max(o["YWORK_VAREPS"], 1e-6)
        o["YWORK_VARNU"] = max(o["YWORK_VARNU"], 1e-6)
        o["YWORK_SIGMAEPS"] = float(np.sqrt(o["YWORK_VAREPS"]))
        o["YWORK_SIGMANU"] = float(np.sqrt(o["YWORK_VARNU"]))
        out.append(o)
    return out


def run(theta, n, grid, arms, spec_of=None):
    """arms: list of cal-override dicts. One solve each, n//len(arms) households."""
    panels = []
    for i, ov in enumerate(arms):
        old = apply_group(ov) if ov else {}
        try:
            sp = GRIDS[grid] if spec_of is None else spec_of(i)
            sol = solve_batch(np.array([theta]), spec=sp, theta_batch=1,
                              chunk=16)[0]
            panels.append(simulate(sol, n_households=max(n // len(arms), 1),
                                   seed=100 + i))
        finally:
            if ov:
                apply_group(old)
    return panels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theta", type=float, nargs=3,
                    default=[0.8465, 0.9898, 4.5002])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--grid", default="full")
    ap.add_argument("--x", default="data/processed/psid_x_rental.pt")
    ap.add_argument("--n_arms", type=int, default=6)
    args = ap.parse_args()
    th, n, grid = args.theta, args.n, args.grid

    print("income-process bootstrap dispersion (comphs):")
    inc = income_draws("comphs", args.n_arms, seed=0)

    base = run(th, n, grid, [None])
    print("\nsolving income-process mixture ...", flush=True)
    inc_p = run(th, n, grid, inc)
    print("solving returns mixture (R_gamma 1.03/1.05/1.07) ...", flush=True)
    rgs = [1.03, 1.05, 1.07]
    ret_p = run(th, n, grid, [None] * 3,
                spec_of=lambda i: replace(GRIDS[grid], R_gamma=rgs[i]))

    e = torch.load(args.x, weights_only=False)["x"].numpy()
    a = e[:, :, 4].ravel()
    print(f"\nwithin-theta IQR vs the comphs baseline, N={n}")
    print(f"{'band':9s}{'feature':17s}{'base':>9s}{'income':>9s}{'ret':>9s}"
          f"{'inc/base':>10s}{'ret/base':>10s}{'PSID':>9s}")
    ri, rr = [], []
    for lo, hi in ((25, 30), (35, 44), (45, 55)):
        sel = (a >= lo) & (a <= hi)
        for j, k in enumerate(FEATURES):
            b = stats(base, lo, hi, k)[0]
            i_ = stats(inc_p, lo, hi, k)[0]
            r_ = stats(ret_p, lo, hi, k)[0]
            pv = e[:, :, j].ravel()[sel]
            pi = np.percentile(pv, 75) - np.percentile(pv, 25)
            ri.append(i_ / max(b, 1)); rr.append(r_ / max(b, 1))
            print(f"{f'{lo}-{hi}':9s}{k:17s}{b:9.0f}{i_:9.0f}{r_:9.0f}"
                  f"{i_ / max(b, 1):10.2f}{r_ / max(b, 1):10.2f}{pi:9.0f}")

    print(f"\nmean widening -- income process: {np.mean(ri):.2f}x   "
          f"returns: {np.mean(rr):.2f}x")
    print("Against education 1.79x and randomised initial wealth 1.00x.")


if __name__ == "__main__":
    main()
