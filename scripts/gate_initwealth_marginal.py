"""Is within-group initial-wealth dispersion worth carrying, given education?

Gate 2 (``gate_within_theta.py``) killed initial wealth as a *justification* for
regenerating: randomising the age-20 seed is forgotten by age 35, and 79% of our
PSID household-waves are at 35 or older. But it costs no extra solve -- initial
wealth is a forward-pass argument, not a solver one -- so the live question is
whether to carry it along, not whether to regenerate for it.

That question is **not** the one Gate 2 answered. Education group already moves
initial wealth *between* groups as one of its four blocks, and by a lot::

    MED_LIQ_WEALTH     comphs +0.0549   somehs -0.0371   compco +0.1923
    MED_TOTAL_WEALTH   comphs  1.4696   somehs  1.0999   compco  3.5894

so the marginal question is whether adding *within*-group spread on top of that
between-group shift buys anything further. This measures exactly that, holding
theta and the three education solves fixed and varying only the forward pass.

Each group's draws are the PSID empirical ``(liquid, illiquid)`` pairs recentred
on that group's own SCF median, which is the composition the plan proposed:
education supplies the level, the PSID draw supplies the spread. Pairs are drawn
jointly -- they are strongly dependent, and independent draws manufacture
households that do not exist.

Usage::

    uv run python scripts/gate_initwealth_marginal.py --n 3000
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from hh_npe.simulator import grids
from hh_npe.simulator.twoasset import GRIDS, simulate
from hh_npe.simulator.twoasset_gpu import solve_batch
from scripts.gate_edu_mixture import (FEATURES, GROUPS, apply_group,
                                      load_group, stats)

AGE_START_SIM = 20
BANDS = ((25, 30), (31, 34), (35, 44), (45, 55))


def psid_ratios(path: str) -> np.ndarray:
    """(liquid, illiquid) as ratios to mean income at the age observed."""
    x = torch.load(path, weights_only=False)["x"].numpy()
    age = x[:, 0, 4]
    ymean = grids.mean_income(np.arange(AGE_START_SIM, AGE_START_SIM + 71))
    y = ymean[np.clip(age.astype(int) - AGE_START_SIM, 0, 70)]
    return np.column_stack([x[:, 0, 2] / y, x[:, 0, 3] / y])


def recentred(ratios: np.ndarray, vals: dict, rng, n: int) -> np.ndarray:
    """PSID spread, shifted so its median is the group's own SCF median.

    Additive rather than multiplicative: the liquid ratio changes sign across
    groups (somehs is -0.0371), so a multiplicative rescale is undefined there.
    """
    d = ratios[rng.integers(0, len(ratios), n)]
    med = np.median(ratios, axis=0)
    target = np.array([vals["MED_LIQ_WEALTH"],
                       vals["MED_TOTAL_WEALTH"] - vals["MED_LIQ_WEALTH"]])
    out = d - med + target
    out[:, 1] = np.clip(out[:, 1], 0.0, None)   # illiquid cannot go negative
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theta", type=float, nargs=3,
                    default=[0.8465, 0.9898, 4.5002])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--grid", default="full")
    ap.add_argument("--x", default="data/processed/psid_x_rental.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ratios = psid_ratios(args.x)
    rng = np.random.default_rng(args.seed)
    per = max(args.n // len(GROUPS), 1)

    edu_only, edu_plus = [], []
    for i, g in enumerate(GROUPS):
        vals = load_group(g)
        old = apply_group(vals)
        try:
            print(f"solving {g} ...", flush=True)
            sol = solve_batch(np.array([args.theta]), spec=GRIDS[args.grid],
                              theta_batch=1, chunk=16)[0]
            # Same solve, same seed, same households: only the age-20 seeding
            # differs, so any gap between the two is the initial condition.
            edu_only.append(simulate(sol, n_households=per, seed=100 + i))
            edu_plus.append(simulate(sol, n_households=per, seed=100 + i,
                                     initial_wealth=recentred(ratios, vals,
                                                              rng, per)))
        finally:
            apply_group(old)

    e = torch.load(args.x, weights_only=False)["x"].numpy()
    a = e[:, :, 4].ravel()
    print(f"\nIQR: education mixture alone vs + within-group initial wealth")
    print(f"{'band':9s}{'feature':17s}{'edu only':>10s}{'edu+init':>10s}"
          f"{'ratio':>8s}{'PSID':>10s}")
    gains = {}
    for lo, hi in BANDS:
        sel = (a >= lo) & (a <= hi)
        r = []
        for j, k in enumerate(FEATURES):
            b = stats(edu_only, lo, hi, k)[0]
            p = stats(edu_plus, lo, hi, k)[0]
            pv = e[:, :, j].ravel()[sel]
            r.append(p / max(b, 1))
            print(f"{f'{lo}-{hi}':9s}{k:17s}{b:10.0f}{p:10.0f}"
                  f"{p / max(b, 1):8.2f}"
                  f"{np.percentile(pv, 75) - np.percentile(pv, 25):10.0f}")
        gains[(lo, hi)] = float(np.mean(r))

    print("\nmean marginal widening from within-group initial wealth, by band:")
    for (lo, hi), v in gains.items():
        share = np.mean((a >= lo) & (a <= hi))
        print(f"  {lo}-{hi}: {v:.2f}x   ({share:.1%} of PSID household-waves)")
    w = np.average(list(gains.values()),
                   weights=[np.mean((a >= lo) & (a <= hi)) for lo, hi in BANDS])
    print(f"\nweighted by where our data actually is: {w:.2f}x")
    print("Education already shifts initial wealth BETWEEN groups. This is the")
    print("marginal value of within-group spread on top of that, which is the")
    print("only part still undecided -- it costs no extra solve either way.")


if __name__ == "__main__":
    main()
