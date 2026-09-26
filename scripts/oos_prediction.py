"""Out-of-sample test: do household-specific parameters predict a household's future?

The per-household posteriors are the project's contribution, and nothing so far
has tested whether they carry information about the *household* rather than
the population. RESULTS.md §20.2 found no detectable β heterogeneity and ρ
heterogeneity that flips sign across specifications -- which predicts that a
household's own parameters should forecast its behaviour no better than one
parameter set for everyone. This checks that prediction directly.

Design
------
1. A 5-wave model (``run_oos_train.sh``) infers each comphs household's
   parameters from **waves 1-5 only**. Waves 6-7 are never seen.
2. Each household is dropped into the solved model at its **own observed
   wave-5 state** -- liquid and illiquid wealth and income -- and run forward
   two waves (four years) by :func:`twoasset.simulate_from`, ``n_sims`` times.
3. Four prediction rules, compared on the same household:

   ``household``    that household's posterior mean -- the thing under test
   ``population``   the median posterior mean across all households: the
                    "no heterogeneity" alternative
   ``laibson``      Laibson et al.'s MSM estimate, for reference
   ``persistence``  wave 5 carried forward; a no-model floor any model should beat

4. Scored on consumption, liquid and illiquid wealth at waves 6 and 7. Income is
   excluded: it is exogenous and identical across rules by construction.

**Paired, common random numbers.** Every rule simulates a given household with
the same seed, and ``simulate_from`` draws income shocks independently of
preferences, so all rules face identical income paths. The household-vs-
population difference therefore carries no shock noise, only the effect of the
parameters.

**Metrics** on ``asinh(v / 1000)`` -- log-like for large values, linear through
zero, so negative liquid wealth (card debt) is handled. ``AE`` is the absolute
error of the predictive median; ``CRPS`` scores the whole predictive
distribution, rewarding calibrated spread as well as location. Lower is better
for both.

Usage::

    uv run python scripts/oos_prediction.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.npe.prior import PHASE3
from hh_npe.npe.train import load_posterior
from hh_npe.simulator import laibson_calibration as cal
from hh_npe.simulator.dispatch import AGE_START_SIM
from hh_npe.simulator.twoasset import GRIDS, simulate_from
from hh_npe.simulator.twoasset_gpu import solve_batch
from scripts.psid_posterior import sample_all

FEATS = ("income", "consumption", "liquid_assets", "illiquid_assets", "age")
TARGETS = ("consumption", "liquid_assets", "illiquid_assets")
LAIBSON = np.array([0.5305, 0.9891, 1.9355])
N_IN, HORIZON_WAVES, WAVE_YEARS = 5, 2, 2


def g(v):
    """Scale for scoring: log-like above ~$1k, linear through zero."""
    return np.arcsinh(np.asarray(v, dtype=float) / 1000.0)


def crps(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ensemble CRPS per row: E|X - y| - 0.5 E|X - X'|, samples along axis 1."""
    a = np.abs(samples - y[:, None]).mean(1)
    s = np.sort(samples, axis=1)
    m = s.shape[1]
    # E|X - X'| via the sorted-sample identity, O(m log m) instead of O(m^2).
    w = 2 * np.arange(1, m + 1) - m - 1
    b = (s * w).sum(1) * 2 / (m * m)
    return a - 0.5 * b


def forecast(thetas: np.ndarray, owner: np.ndarray, state: dict, n_sims: int,
             spec, theta_batch: int, chunk: int) -> dict:
    """Simulate each household under the theta assigned to it.

    ``thetas[k]`` is solved once and used for households ``owner == k``.
    Returns ``{feature: (H, n_sims, horizon_years + 1)}``.
    """
    H = len(state["t0"])
    horizon = HORIZON_WAVES * WAVE_YEARS
    out = {f: np.empty((H, n_sims, horizon + 1)) for f in
           ("income",) + TARGETS}
    for b0 in range(0, len(thetas), theta_batch):
        sols = solve_batch(thetas[b0:b0 + theta_batch], spec,
                           theta_batch=theta_batch, chunk=chunk)
        for k, sol in enumerate(sols, start=b0):
            for h in np.flatnonzero(owner == k):
                # One call per household, seeded by the household: the same
                # income path whichever theta is being tested (CRN).
                sim = simulate_from(sol, state["t0"][h:h + 1],
                                    state["liquid"][h:h + 1],
                                    state["illiquid"][h:h + 1],
                                    state["income"][h:h + 1],
                                    horizon=horizon, n_sims=n_sims,
                                    seed=10_000 + int(h))
                for f in out:
                    out[f][h] = sim[f][0]
        print(f"    solved {min(b0 + theta_batch, len(thetas))}/{len(thetas)}",
              flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dirs", type=Path, nargs="+",
                    default=[Path(f"outputs/oos/w5_s{s}") for s in range(3)])
    ap.add_argument("--x", type=Path,
                    default=Path("data/processed/psid_x_educ_rental.pt"))
    ap.add_argument("--n_sims", type=int, default=200)
    ap.add_argument("--n_post", type=int, default=500)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=None,
                    help="Use only the first N households (smoke test).")
    ap.add_argument("--out", type=Path, default=Path("outputs/oos"))
    args = ap.parse_args()

    d = torch.load(args.x, weights_only=False)
    assert tuple(d["features"]) == FEATS, d["features"]
    x = d["x"][d["educ"] == cal.EDUC_GROUPS.index("comphs")].numpy()
    if args.limit:
        x = x[: args.limit]
    H = len(x)
    print(f"{H} comphs households; inputs waves 1-{N_IN}, "
          f"targets waves {N_IN + 1}-{N_IN + HORIZON_WAVES}")

    # --- 1. parameters from waves 1-5 only --------------------------------
    from scripts.ensemble_eval import _ensemble
    posts = [load_posterior(r / f"posterior_{N_IN}w.pt")["posterior"]
             for r in args.run_dirs]
    post = _ensemble(posts) if len(posts) > 1 else posts[0]
    torch.manual_seed(0)
    mean, *_ = sample_all(post, torch.from_numpy(x[:, :N_IN]).float(), args.n_post)
    bad = ~np.isfinite(mean[:, 0])
    pop = np.median(mean[~bad], axis=0)
    mean[bad] = pop     # unrepresentable households fall back to the population
    print(f"  household thetas: median {pop.round(4)}, "
          f"{bad.sum()} fell back to population")

    # --- 2. starting state at wave 5 ---------------------------------------
    last = N_IN - 1
    age = x[:, last, FEATS.index("age")].astype(int)
    state = {"t0": age - AGE_START_SIM,
             "income": x[:, last, FEATS.index("income")],
             "liquid": x[:, last, FEATS.index("liquid_assets")],
             "illiquid": x[:, last, FEATS.index("illiquid_assets")]}

    spec = dataclasses.replace(GRIDS["full"], calib=cal.COMPHS)
    kw = dict(n_sims=args.n_sims, spec=spec, theta_batch=16, chunk=16)
    rules = {}
    print("  forecasting: household")
    rules["household"] = forecast(mean, np.arange(H), state, **kw)
    print("  forecasting: population")
    rules["population"] = forecast(pop[None], np.zeros(H, int), state, **kw)
    print("  forecasting: laibson")
    rules["laibson"] = forecast(LAIBSON[None], np.zeros(H, int), state, **kw)

    # --- 3. score ------------------------------------------------------------
    res = {}
    rng = np.random.default_rng(0)
    boot = rng.integers(0, H, size=(args.n_boot, H))
    print(f"\n{'':22s}" + "".join(f"{r:>13s}" for r in
                                 ("household", "population", "laibson",
                                  "persistence")))
    per_hh = {}
    for f in TARGETS:
        fi = FEATS.index(f)
        for w in range(HORIZON_WAVES):
            wave = N_IN + w                 # 0-based index into PSID waves
            yrs = (w + 1) * WAVE_YEARS      # years after wave 5
            y = g(x[:, wave, fi])
            row = {}
            for r, sim in rules.items():
                pred = g(sim[f][:, :, yrs])
                row[r] = {"ae": np.abs(np.median(pred, 1) - y),
                          "crps": crps(pred, y)}
            persist = np.abs(g(x[:, last, fi]) - y)
            row["persistence"] = {"ae": persist, "crps": persist}
            key = f"{f} wave {wave + 1}"
            per_hh[key] = row
            for m in ("ae", "crps"):
                print(f"{key:22s}" if m == "ae" else f"{'':22s}", end="")
                print("".join(f"{row[r][m].mean():13.4f}" for r in row)
                      + f"   {m.upper()}")
            dlt = row["household"]["crps"] - row["population"]["crps"]
            bd = dlt[boot].mean(1)
            res[key] = {r: {m: float(row[r][m].mean()) for m in ("ae", "crps")}
                        for r in row}
            res[key]["household_minus_population_crps"] = {
                "mean": float(dlt.mean()),
                "ci95": [float(np.percentile(bd, 2.5)),
                         float(np.percentile(bd, 97.5))],
                "share_households_better": float((dlt < 0).mean())}

    # --- 4. the headline paired comparison -----------------------------------
    print("\nhousehold minus population, CRPS (negative = own parameters help)")
    all_d = np.mean([per_hh[k]["household"]["crps"] - per_hh[k]["population"]["crps"]
                     for k in per_hh], axis=0)
    for k in per_hh:
        r = res[k]["household_minus_population_crps"]
        print(f"  {k:22s} {r['mean']:+.4f}  95% CI [{r['ci95'][0]:+.4f}, "
              f"{r['ci95'][1]:+.4f}]  households better {r['share_households_better']:.1%}")
    bd = all_d[boot].mean(1)
    overall = {"mean": float(all_d.mean()),
               "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
               "share_households_better": float((all_d < 0).mean())}
    res["overall_household_minus_population_crps"] = overall
    print(f"  {'ALL targets, averaged':22s} {overall['mean']:+.4f}  95% CI "
          f"[{overall['ci95'][0]:+.4f}, {overall['ci95'][1]:+.4f}]  "
          f"households better {overall['share_households_better']:.1%}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "oos_results.json").write_text(json.dumps(
        {"n_households": H, "population_theta": pop.tolist(),
         "n_fallback": int(bad.sum()), "results": res}, indent=2))
    np.savez(args.out / "oos_household_thetas.npz", theta=mean, fallback=bad)
    print(f"\nwrote {args.out}/oos_results.json")


if __name__ == "__main__":
    main()
