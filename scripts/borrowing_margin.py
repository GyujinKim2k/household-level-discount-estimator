"""The credit-card margin: the model over-generates borrowers at every theta.

The two-asset model exists to explain the *credit-card debt puzzle* -- households
holding illiquid wealth while revolving expensive card debt. That margin is also
where beta is identified: present bias is what makes a household borrow at
10.59% while holding an asset returning 5%.

So the share of households in net card debt is the single moment this model
should get right, and it does not. Simulated households borrow 2.5-3.5x more
often than PSID's, **at Laibson et al.'s own MSM estimate as well as ours**,
which means it is not an artifact of the parameters we recovered and no
(beta, delta, rho) can repair it.

Construction is comparable on both sides, which is why the comparison is worth
making at all: PSID ``liquid`` is checking/saving + CD/bonds - credit-card debt,
a single net position, and the model's ``X`` is likewise a single net position
that is negative exactly when the household is borrowing on the card.

Usage::

    uv run python scripts/borrowing_margin.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hh_npe.simulator.twoasset import GRIDS, simulate
from hh_npe.simulator.twoasset_gpu import solve_batch

AGE_START_SIM = 20
BANDS = ((25, 30), (31, 34), (35, 44), (45, 55))
THETAS = {
    "ours (posterior median)": [0.8465, 0.9898, 4.5002],
    "Laibson et al. MSM": [0.5305, 0.9891, 1.9355],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x", type=Path,
                    default=Path("data/processed/psid_x_rental.pt"))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--grid", default="full")
    args = ap.parse_args()

    e = torch.load(args.x, weights_only=False)["x"].numpy()
    age = e[:, :, 4].ravel()
    liq = e[:, :, 2].ravel()

    sols = solve_batch(np.array(list(THETAS.values())), spec=GRIDS[args.grid],
                       theta_batch=len(THETAS), chunk=16)
    panels = [simulate(s, n_households=args.n, seed=1) for s in sols]

    print("share of households in NET credit-card debt (liquid < 0)\n")
    print(f"{'ages':10s}{'PSID':>9s}" + "".join(f"{k:>26s}" for k in THETAS))
    for lo, hi in BANDS:
        sel = (age >= lo) & (age <= hi)
        t0, t1 = lo - AGE_START_SIM, hi - AGE_START_SIM + 1
        row = f"{f'{lo}-{hi}':10s}{np.mean(liq[sel] < 0):9.1%}"
        for p in panels:
            v = p["liquid_assets"][:, t0:t1].ravel()
            row += f"{np.mean(v < 0):26.1%}"
        print(row)

    print("\nPSID's borrowing share is flat in age; the model's is steeply")
    print("age-varying, and in opposite directions at the two thetas. The")
    print("over-generation holds at Laibson et al.'s own estimate, so it is a")
    print("property of the model rather than of the parameters we recovered.")
    print("\nCaveat: they scale SCF card debt by alpha = 2.02 for under-reporting")
    print("(RESULTS.md section 6, deviation 2). That scales the *amount* owed,")
    print("not the *incidence* of owing, so it moves this comparison far less")
    print("than it moves the level of debt -- but households who deny card debt")
    print("outright would still be missed on the PSID side.")


if __name__ == "__main__":
    main()
