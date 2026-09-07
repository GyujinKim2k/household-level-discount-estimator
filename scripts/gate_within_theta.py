"""Gate 2: does randomising initial wealth widen p(x|theta) enough to matter?

The go/no-go for a ~9.5-day regeneration. Every simulated household currently
starts at the same SCF median wealth (twoasset.py:415-419), so all within-theta
dispersion comes from income shocks. If that is already as wide as PSID's, the
regeneration cannot help and must not run.

An earlier comparison of POOLED simulated dispersion against PSID is not the
right quantity -- it mixes in the whole uniform prior over theta. This holds
theta fixed at the posterior median and varies only the initial condition.

Usage::

    uv run python scripts/gate_within_theta.py --n 4000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from hh_npe.simulator import grids
from hh_npe.simulator.twoasset import GRIDS, simulate
from hh_npe.simulator.twoasset_gpu import solve_batch

FEATURES = ("income", "consumption", "liquid_assets", "illiquid_assets")
AGE_START_SIM = 20


def psid_initial_ratios(path: str) -> np.ndarray:
    """(liquid, illiquid) as ratios to mean income at the age observed.

    Taken jointly per household -- they are strongly dependent, and independent
    draws would manufacture households that do not exist.
    """
    d = torch.load(path, weights_only=False)
    x = d["x"].numpy()
    age = x[:, 0, 4]
    ymean = grids.mean_income(np.arange(AGE_START_SIM, AGE_START_SIM + 71))
    y_at_age = ymean[np.clip(age.astype(int) - AGE_START_SIM, 0, 70)]
    return np.column_stack([x[:, 0, 2] / y_at_age, x[:, 0, 3] / y_at_age])


def spread(panel: dict, lo: int, hi: int) -> dict:
    """IQR, p90-p10 and sd. IQR alone is quantised to the wealth grid spacing,
    so two visibly different distributions can share it to the digit."""
    t0, t1 = lo - AGE_START_SIM, hi - AGE_START_SIM + 1
    out = {}
    for k in FEATURES:
        v = panel[k][:, t0:t1].ravel()
        out[k] = (np.percentile(v, 75) - np.percentile(v, 25),
                  np.percentile(v, 90) - np.percentile(v, 10),
                  v.std(), np.median(v))
    return out


def by_age_band(fixed, rand, e, bands):
    """Does the initial condition survive to the ages we actually observe?

    A buffer-stock model forgets its age-20 seed as it converges to the ergodic
    distribution. If the memory is gone by 35, randomising it cannot widen
    p(x|theta) over the observation window and the regeneration is pointless --
    which is a different verdict from "randomisation does nothing at all".
    """
    a = e[:, :, 4].ravel()
    print(f"\n{'band':9s}{'feature':13s}{'fix sd':>10s}{'rnd sd':>10s}"
          f"{'ratio':>7s}{'fix med':>10s}{'rnd med':>10s}{'PSID med':>10s}"
          f"{'PSID sd':>10s}")
    for lo, hi in bands:
        sf, sr = spread(fixed, lo, hi), spread(rand, lo, hi)
        sel = (a >= lo) & (a <= hi)
        for j, k in enumerate(FEATURES):
            if k == "income":
                continue
            pv = e[:, :, j].ravel()[sel]
            print(f"{f'{lo}-{hi}':9s}{k:13s}{sf[k][2]:10.0f}{sr[k][2]:10.0f}"
                  f"{sr[k][2] / max(sf[k][2], 1):7.2f}{sf[k][3]:10.0f}"
                  f"{sr[k][3]:10.0f}{np.median(pv):10.0f}{pv.std():10.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theta", type=float, nargs=3,
                    default=[0.8465, 0.9898, 4.5002],
                    help="Posterior median from outputs/psid_rental.")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--grid", default="full")
    ap.add_argument("--x", default="data/processed/psid_x_rental.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ratios = psid_initial_ratios(args.x)
    print(f"PSID initial (liquid, illiquid) ratios from {len(ratios)} households")
    for j, nm in enumerate(("liquid", "illiquid")):
        q = np.percentile(ratios[:, j], [10, 25, 50, 75, 90])
        print(f"  {nm:9s} " + "  ".join(f"p{p}={v:+.3f}"
                                        for p, v in zip((10, 25, 50, 75, 90), q)))
    print(f"  model's single seed: liquid +0.0549, illiquid "
          f"{1.4696 - 0.05486:+.3f}  (SCF comphs medians)")

    b, d, r = args.theta
    print(f"\nsolving theta = ({b}, {d}, {r}) on the {args.grid} grid ...",
          flush=True)
    # GPU solve: ~5 s against ~28 min on one CPU core. Not bit-reproducible
    # across (device, theta_batch, chunk) -- irrelevant here, since the gate
    # measures dispersion at one theta rather than comparing solutions.
    sol = solve_batch(np.array([[b, d, r]]), spec=GRIDS[args.grid],
                      theta_batch=1, chunk=16)[0]

    rng = np.random.default_rng(args.seed)
    draw = ratios[rng.integers(0, len(ratios), args.n)]
    fixed = simulate(sol, n_households=args.n, seed=1)
    rand = simulate(sol, n_households=args.n, seed=1, initial_wealth=draw)

    sf, sr = spread(fixed, 35, 44), spread(rand, 35, 44)

    e = torch.load(args.x, weights_only=False)["x"].numpy()
    a = e[:, :, 4].ravel()
    sel = (a >= 35) & (a <= 44)
    print(f"\nwithin-theta spread at ages 35-44, N={args.n}")
    print(f"{'feature':18s}{'fixed IQR':>12s}{'random IQR':>12s}{'ratio':>8s}"
          f"{'PSID IQR':>11s}{'rand/PSID':>11s}")
    for j, k in enumerate(FEATURES):
        pv = e[:, :, j].ravel()[sel]
        pi = np.percentile(pv, 75) - np.percentile(pv, 25)
        f_i, r_i = sf[k][0], sr[k][0]
        print(f"{k:18s}{f_i:12.0f}{r_i:12.0f}{r_i / max(f_i, 1):8.2f}"
              f"{pi:11.0f}{r_i / max(pi, 1):11.2f}")

    by_age_band(fixed, rand, e, [(25, 30), (31, 34), (35, 44), (45, 55)])

    widen = np.mean([sr[k][0] / max(sf[k][0], 1) for k in FEATURES])
    print(f"\nmean widening from randomisation: {widen:.2f}x")
    print("GO if randomisation materially widens p(x|theta) AND the fixed")
    print("spread is visibly narrower than PSID's. If ratio ~ 1, the")
    print("regeneration cannot deliver what it is for.")


if __name__ == "__main__":
    main()
