"""Where does the model's wealth-dynamics misfit come from?

RESULTS.md §22: started from a real household's wave-5 state, the model makes
large, lumpy wealth moves within two years that real households do not, and
"nothing changes" out-forecasts it. This measures the moves themselves, model
against PSID, on the same households, so candidate fixes can be tested in
simulation before anything is regenerated.

For each comphs household: its observed wave-5 state, then the two-year change
to wave 6 -- in PSID (what it did) and in the model at a given theta and spec
(what the model says it would do, median over simulated shock paths).

Model mechanics this is probing (SIMULATOR_SPEC §1): deposits into the illiquid
asset are free and it pays R_gamma = 1.05 against R = 1.02, while withdrawals pay
a liquidation penalty of ~31% at age 45. That predicts one-way, lumpy moves of
liquid cash into illiquid wealth.

Candidate fixes are passed as ModelSpec overrides (``--r_gamma`` etc.), so each
is one solve at fixed theta.

Usage::

    uv run python scripts/wealth_dynamics.py
    uv run python scripts/wealth_dynamics.py --r_gamma 1.03
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch

from hh_npe.simulator import laibson_calibration as cal
from hh_npe.simulator.dispatch import AGE_START_SIM
from hh_npe.simulator.twoasset import GRIDS, simulate_from
from hh_npe.simulator.twoasset_gpu import solve_batch

FEATS = ("income", "consumption", "liquid_assets", "illiquid_assets", "age")


def describe(tag: str, dx: np.ndarray, dz: np.ndarray, z0: np.ndarray) -> dict:
    """Summary of two-year changes in liquid (dx) and illiquid (dz) wealth."""
    big_up = (dz > np.maximum(0.5 * z0, 20_000))
    big_dn = (dz < -np.maximum(0.5 * z0, 20_000))
    r = {
        "dz_median": np.median(dz), "dz_p10": np.percentile(dz, 10),
        "dz_p90": np.percentile(dz, 90),
        "share_z_big_up": big_up.mean(), "share_z_big_down": big_dn.mean(),
        "share_z_down": (dz < -1000).mean(),
        "dx_median": np.median(dx), "dx_p10": np.percentile(dx, 10),
        "dx_p90": np.percentile(dx, 90),
        "corr_dx_dz": np.corrcoef(dx, dz)[0, 1],
    }
    print(f"{tag:26s}"
          f"{r['dz_median']:>10,.0f}{r['dz_p10']:>10,.0f}{r['dz_p90']:>10,.0f}"
          f"{r['share_z_big_up']:>8.1%}{r['share_z_big_down']:>8.1%}{r['share_z_down']:>8.1%}"
          f"{r['dx_median']:>10,.0f}{r['dx_p10']:>10,.0f}{r['dx_p90']:>10,.0f}"
          f"{r['corr_dx_dz']:>7.2f}")
    return r


def header() -> None:
    print(f"{'':26s}{'ΔZ med':>10s}{'ΔZ p10':>10s}{'ΔZ p90':>10s}"
          f"{'Z big+':>8s}{'Z big-':>8s}{'Z down':>8s}"
          f"{'ΔX med':>10s}{'ΔX p10':>10s}{'ΔX p90':>10s}{'corr':>7s}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", type=Path,
                    default=Path("data/processed/psid_x_educ_rental.pt"))
    ap.add_argument("--theta", type=float, nargs=3, default=None,
                    help="(beta, delta, rho); default the OOS population theta.")
    ap.add_argument("--r_gamma", type=float, default=None)
    ap.add_argument("--n_sims", type=int, default=100)
    ap.add_argument("--wave", type=int, default=4,
                    help="0-based starting wave; the change is to wave+1.")
    args = ap.parse_args()

    d = torch.load(args.x, weights_only=False)
    x = d["x"][d["educ"] == cal.EDUC_GROUPS.index("comphs")].numpy()
    w = args.wave
    fi = {f: FEATS.index(f) for f in FEATS}
    z0, x0 = x[:, w, fi["illiquid_assets"]], x[:, w, fi["liquid_assets"]]

    if args.theta is None:
        import json
        pop = json.loads(Path("outputs/oos/oos_results.json").read_text())
        theta = np.array(pop["population_theta"])
    else:
        theta = np.array(args.theta)
    spec = dataclasses.replace(GRIDS["full"], calib=cal.COMPHS)
    if args.r_gamma is not None:
        spec = dataclasses.replace(spec, R_gamma=args.r_gamma)
    print(f"theta {theta.round(4)}  R_gamma {spec.R_gamma}  R {spec.R}  "
          f"N={len(x)} comphs, wave {w + 1} -> {w + 2}\n")

    sol = solve_batch(theta[None], spec, theta_batch=16, chunk=16)[0]
    sim = simulate_from(sol, x[:, w, fi["age"]].astype(int) - AGE_START_SIM,
                        x0, z0, x[:, w, fi["income"]], horizon=2,
                        n_sims=args.n_sims, seed=1)
    header()
    describe("PSID (what they did)", x[:, w + 1, fi["liquid_assets"]] - x0,
             x[:, w + 1, fi["illiquid_assets"]] - z0, z0)
    describe("model (median path)",
             np.median(sim["liquid_assets"][:, :, 2], 1) - x0,
             np.median(sim["illiquid_assets"][:, :, 2], 1) - z0, z0)
    # Pooled over simulated paths too, since a median path hides lumpiness.
    zs = sim["illiquid_assets"][:, :, 2] - z0[:, None]
    xs = sim["liquid_assets"][:, :, 2] - x0[:, None]
    describe("model (all paths pooled)", xs.ravel(), zs.ravel(),
             np.repeat(z0, args.n_sims))

    # Where the model's big illiquid deposits come from.
    up = np.median(zs, 1) > np.maximum(0.5 * z0, 20_000)
    print(f"\nhouseholds the model moves into a big illiquid deposit: {up.mean():.1%}")
    if up.any():
        print(f"  their median wave-{w + 1} liquid wealth ${np.median(x0[up]):,.0f}, "
              f"illiquid ${np.median(z0[up]):,.0f}, income "
              f"${np.median(x[up, w, fi['income']]):,.0f}")
        print(f"  others'                 liquid ${np.median(x0[~up]):,.0f}, "
              f"illiquid ${np.median(z0[~up]):,.0f}, income "
              f"${np.median(x[~up, w, fi['income']]):,.0f}")


if __name__ == "__main__":
    main()
