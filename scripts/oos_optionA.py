"""Out-of-sample forecasts and wealth dynamics on the adopted option A model.

RESULTS.md §22 and §24, rerun on option A (logit beta, RESULTS 38): do a
household's own parameters forecast its next two waves better than one
population parameter set, and how far are the model's two-year wealth moves
from PSID's now that card access is a household type?

Design, as §22 (``oos_prediction.py``) except where option A forces a change:

1. A 5-wave option A ensemble (``optionA.py train --n_waves 5``) infers each
   comphs household's (beta, delta, rho, R_gamma) from **waves 1-5 only**,
   importance-weighted to the uniform prior. Input is option A's PSID tensor
   (gross liquid wealth, illiquid including DC pensions; RESULTS 33), so the
   wave-5 starting state and the wave 6-7 targets are on the same definitions.
2. Each household is started at its own wave-5 state and run forward two waves
   by :func:`twoasset.simulate_from`, ``n_sims`` times, seeded by the household
   (common random numbers across rules).
3. Rules: ``household`` (its posterior mean), ``population`` (the median
   posterior mean), ``laibson`` (their MSM estimate, R_gamma at their 1.05) and
   ``persistence`` (wave 5 carried forward).

**R_gamma per household.** The solver shares one R_gamma across a GPU batch,
so household thetas are solved in groups with R_gamma rounded to the nearest
``RG_STEP`` (0.0025, against a posterior sd of ~0.013). Population and Laibson
are solved at their exact values.

**Card access is not observed.** A household with negative liquid wealth in
any of waves 1-5 holds a card: a no-card household cannot borrow. For the rest
both types are simulated and the forecast is their mixture, the cardholder
weight being ``a / (1 + a)`` -- the posterior under the 50/50 generation share,
where ``a`` is the chance that a cardholder at the population theta, observed
over the same five ages, never shows negative liquid wealth (no-card households
never do). The same weight serves every rule, so rules still differ only in
their parameters. Scores are also reported at weight 0 (the §24.1 proxy: no
card unless seen borrowing) and 0.5.

Also reported, at the population theta:

* the §24 wealth-dynamics table: two-year changes from wave 5 to 6, model
  (weighted median path) against what the household did;
* the §24 level table: the model's own households (from age 20) by card type,
  median liquid wealth and net-debt share by age, against PSID.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/oos_optionA.py \\
        --run_dirs outputs/oos_optionA/w5/s0 ... --out outputs/oos_optionA
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.waves import FEATURES_TWOASSET, aggregate_waves
from hh_npe.evaluation.weighted import sample_weighted
from hh_npe.simulator import laibson_calibration as cal
from hh_npe.simulator.dispatch import AGE_START_SIM
from hh_npe.simulator.twoasset import GRIDS, simulate, simulate_from
from hh_npe.simulator.twoasset_gpu import solve_batch
from scripts.oos_prediction import FEATS, HORIZON_WAVES, N_IN, TARGETS, WAVE_YEARS, g
from scripts.oos_prediction import LAIBSON as LAIBSON3
from scripts.optionA import BOX, PSID_X, _weighting, flow_box, load_members, transform
from scripts.wealth_dynamics import describe, header

LAIBSON = np.append(LAIBSON3, 1.05)
RG_STEP = 0.0025
HORIZON = HORIZON_WAVES * WAVE_YEARS
#: §22's household-minus-population CRPS (5-wave linear-delta model, no card
#: types, net liquid wealth), for comparison.
S22 = {"consumption wave 6": -0.038, "consumption wave 7": -0.024,
       "liquid_assets wave 6": -0.096, "liquid_assets wave 7": -0.004,
       "illiquid_assets wave 6": +0.064, "illiquid_assets wave 7": +0.124}


def spec_for(card: int, r_gamma: float, grid: str = "full"):
    """comphs, credit line zeroed for a no-card household, as generation did."""
    calib = cal.COMPHS
    if not card:
        calib = dataclasses.replace(calib, c0_credit=0.0, c1_credit=0.0, c2_credit=0.0)
    return dataclasses.replace(GRIDS[grid], calib=calib, R_gamma=float(r_gamma))


def forecast(thetas, rg, owner, need, state, n_sims, grid, sols=None) -> dict:
    """Household ``h`` under ``thetas[owner[h]]`` (R_gamma ``rg[owner[h]]``),
    for each card type ``c`` where ``need[c][h]``.

    ``sols`` optionally supplies solved ``{(card, k): Solution}``. Returns
    ``{card: {feature: (H, n_sims, HORIZON + 1)}}``, NaN where not needed.
    """
    H = len(owner)
    out = {c: {f: np.full((H, n_sims, HORIZON + 1), np.nan) for f in ("income",) + TARGETS}
           for c in (0, 1)}
    groups: dict = {}
    for c in (0, 1):
        for k in np.unique(owner[need[c]]):
            groups.setdefault((c, float(rg[k])), []).append(int(k))
    done = 0
    for (c, r), ks in sorted(groups.items()):
        spec = spec_for(c, r, grid)
        for b0 in range(0, len(ks), 16):
            kb = ks[b0:b0 + 16]
            if sols is not None and all((c, k) in sols for k in kb):
                batch = [sols[(c, k)] for k in kb]
            else:
                batch = solve_batch(thetas[kb, :3], spec, theta_batch=16, chunk=16)
            for k, sol in zip(kb, batch):
                for h in np.flatnonzero((owner == k) & need[c]):
                    sim = simulate_from(sol, state["t0"][h:h + 1], state["liquid"][h:h + 1],
                                        state["illiquid"][h:h + 1], state["income"][h:h + 1],
                                        horizon=HORIZON, n_sims=n_sims, seed=10_000 + int(h))
                    for f in out[c]:
                        out[c][f][h] = sim[f][0]
            del batch
            torch.cuda.empty_cache()
            done += len(kb)
            print(f"    solved {done} (card {c}, R_gamma {r:.4f})", flush=True)
    return out


def _mix(a, b, w):
    """Pooled samples and weights of the mixture ``w * a + (1 - w) * b``,
    sorted per row. ``b`` may be NaN where ``w == 1``."""
    n = a.shape[1]
    s = np.concatenate([a, np.where(np.isnan(b), a, b)], 1)
    wt = np.concatenate([np.repeat(w[:, None], n, 1), np.repeat(1 - w[:, None], n, 1)], 1) / n
    o = np.argsort(s, 1)
    return np.take_along_axis(s, o, 1), np.take_along_axis(wt, o, 1)


def crps_mix(a, b, w, y):
    """CRPS of the weighted sample mixture, per row:
    ``sum_i w_i |s_i - y| - 0.5 sum_ij w_i w_j |s_i - s_j|``, the second term by
    the sorted-sample identity with cumulative weights."""
    s, wt = _mix(a, b, w)
    c = np.cumsum(wt, 1)
    t1 = (wt * np.abs(s - y[:, None])).sum(1)
    t2 = 2 * (wt * s * ((c - wt) - (1 - c))).sum(1)
    return t1 - 0.5 * t2


def median_mix(a, b, w):
    s, wt = _mix(a, b, w)
    i = (np.cumsum(wt, 1) < 0.5).sum(1)
    return s[np.arange(len(s)), np.minimum(i, s.shape[1] - 1)]


def never_borrow_share(sol, ages: np.ndarray, n: int, seed: int) -> dict:
    """For each wave-1 age, the share of the model's own households (``sol``,
    from age 20) with no negative liquid wealth over five waves from it."""
    panel = simulate(sol, n_households=n, seed=seed)
    li = FEATURES_TWOASSET.index("liquid_assets")
    out = {}
    for a in np.unique(ages):
        x, _ = aggregate_waves(panel, age_start_sim=AGE_START_SIM, start_age=int(a),
                               n_waves=N_IN, wave_years=WAVE_YEARS, features=FEATURES_TWOASSET)
        out[int(a)] = float((x[:, :, li] >= 0).all(1).mean())
    return out, panel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dirs", type=Path, nargs="+",
                    default=[Path(f"outputs/oos_optionA/w5/s{s}") for s in range(5)])
    ap.add_argument("--x", type=Path, default=PSID_X)
    ap.add_argument("--n_sims", type=int, default=200)
    ap.add_argument("--n_draws", type=int, default=40000)
    ap.add_argument("--n_own", type=int, default=4000,
                    help="Model households per card type for the level table.")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=None,
                    help="Use only the first N households (smoke test).")
    ap.add_argument("--grid", default="full", help="'coarse' for smoke tests only.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else None)
    ap.add_argument("--out", type=Path, default=Path("outputs/oos_optionA"))
    args = ap.parse_args()

    d = torch.load(args.x, weights_only=False)
    assert tuple(d["features"]) == FEATS, d["features"]
    assert d.get("liquid_def") == "gross" and d.get("with_pension"), "not option A's input"
    x = d["x"].numpy()
    x = x[d["educ"].numpy() == cal.EDUC_GROUPS.index("comphs")]
    if args.limit:
        x = x[: args.limit]
    H = len(x)
    fi = {f: FEATS.index(f) for f in FEATS}
    print(f"{H} comphs households; inputs waves 1-{N_IN}, targets waves "
          f"{N_IN + 1}-{N_IN + HORIZON_WAVES}", flush=True)

    # --- 1. parameters from waves 1-5 only --------------------------------
    members, proposal, cfg = load_members(args.run_dirs, args.device, n_waves=N_IN)
    bt = cfg["beta_transform"]
    inv, lw = _weighting(proposal, bt)
    fb = flow_box(bt)
    torch.manual_seed(0)
    xt, _ = transform(torch.from_numpy(x[:, :N_IN]).float())
    r = sample_weighted(members, xt, args.n_draws, fb.low, fb.high, invert=inv, log_weight=lw)
    mean = r["mean"].copy()
    del members, r
    torch.cuda.empty_cache()
    bad = ~np.isfinite(mean[:, 0])
    pop = np.median(mean[~bad], axis=0)
    mean[bad] = pop
    print(f"  {len(args.run_dirs)} members, beta target {bt}; household thetas: median "
          f"{pop.round(4)}, {bad.sum()} fell back to the population", flush=True)

    # --- 2. starting state and card access --------------------------------
    last = N_IN - 1
    age1 = x[:, 0, fi["age"]].astype(int)
    state = {"t0": x[:, last, fi["age"]].astype(int) - AGE_START_SIM,
             "income": x[:, last, fi["income"]],
             "liquid": x[:, last, fi["liquid_assets"]],
             "illiquid": x[:, last, fi["illiquid_assets"]]}
    borrower = (x[:, :N_IN, fi["liquid_assets"]] < 0).any(1)
    need = {1: np.ones(H, bool), 0: ~borrower}

    pop_sols = {(c, 0): solve_batch(pop[None, :3], spec_for(c, pop[3], args.grid),
                                    theta_batch=16, chunk=16)[0] for c in (0, 1)}
    a_by_age, own_card = never_borrow_share(pop_sols[(1, 0)], age1, args.n_own, seed=1)
    a = np.array([a_by_age[int(v)] for v in age1])
    w_head = np.where(borrower, 1.0, a / (1 + a))
    weights = {"headline": w_head, "proxy (no card unless seen borrowing)":
               np.where(borrower, 1.0, 0.0), "prior 0.5": np.where(borrower, 1.0, 0.5)}
    print(f"  borrowers in waves 1-{N_IN}: {borrower.mean():.1%}; cardholders at the "
          f"population theta who never borrow over five waves: median {np.median(a):.3f}; "
          f"non-borrowers' cardholder weight median {np.median(w_head[~borrower]):.3f}",
          flush=True)

    # --- 3. forecasts --------------------------------------------------------
    rg_hh = np.clip(np.round(mean[:, 3] / RG_STEP) * RG_STEP, BOX.rgamma_low, BOX.rgamma_high)
    kw = dict(state=state, n_sims=args.n_sims, grid=args.grid)
    rules = {}
    print("  forecasting: household", flush=True)
    rules["household"] = forecast(mean, rg_hh, np.arange(H), need, **kw)
    print("  forecasting: population", flush=True)
    rules["population"] = forecast(pop[None], pop[None, 3], np.zeros(H, int), need,
                                   sols=pop_sols, **kw)
    print("  forecasting: laibson", flush=True)
    rules["laibson"] = forecast(LAIBSON[None], LAIBSON[None, 3], np.zeros(H, int), need, **kw)

    # --- 4. score ------------------------------------------------------------
    rng = np.random.default_rng(0)
    boot = rng.integers(0, H, size=(args.n_boot, H))
    res = {}
    for wname, w in weights.items():
        per_hh, res[wname] = {}, {}
        for f in TARGETS:
            for k in range(HORIZON_WAVES):
                wave, yrs = N_IN + k, (k + 1) * WAVE_YEARS
                y = g(x[:, wave, fi[f]])
                row = {}
                for rn, sim in rules.items():
                    pc, pn = g(sim[1][f][:, :, yrs]), g(sim[0][f][:, :, yrs])
                    row[rn] = {"ae": np.abs(median_mix(pc, pn, w) - y),
                               "crps": crps_mix(pc, pn, w, y)}
                persist = np.abs(g(x[:, last, fi[f]]) - y)
                row["persistence"] = {"ae": persist, "crps": persist}
                key = f"{f} wave {wave + 1}"
                per_hh[key] = row
                dl = row["household"]["crps"] - row["population"]["crps"]
                bd = dl[boot].mean(1)
                res[wname][key] = {rn: {m: float(v[m].mean()) for m in ("ae", "crps")}
                                   for rn, v in row.items()}
                res[wname][key]["household_minus_population_crps"] = {
                    "mean": float(dl.mean()), "ci95": np.percentile(bd, [2.5, 97.5]).tolist(),
                    "share_households_better": float((dl < 0).mean())}
        all_d = np.mean([per_hh[k]["household"]["crps"] - per_hh[k]["population"]["crps"]
                         for k in per_hh], axis=0)
        bd = all_d[boot].mean(1)
        res[wname]["overall_household_minus_population_crps"] = {
            "mean": float(all_d.mean()), "ci95": np.percentile(bd, [2.5, 97.5]).tolist(),
            "share_households_better": float((all_d < 0).mean())}

    for wname in weights:
        rw = res[wname]
        print(f"\n=== card weight: {wname} ===\nCRPS (lower is better)")
        print(f"{'':24s}" + "".join(f"{n:>13s}" for n in
                                    ("household", "population", "laibson", "persistence")))
        for key in S22:
            print(f"{key:24s}" + "".join(f"{rw[key][n]['crps']:13.3f}" for n in
                                         ("household", "population", "laibson", "persistence")))
        print("household minus population (negative = own parameters help)   §22")
        for key in S22:
            h = rw[key]["household_minus_population_crps"]
            print(f"  {key:22s} {h['mean']:+.3f}  [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]"
                  f"  better {h['share_households_better']:.1%}   {S22[key]:+.3f}")
        o = rw["overall_household_minus_population_crps"]
        print(f"  {'all, averaged':22s} {o['mean']:+.3f}  [{o['ci95'][0]:+.3f}, "
              f"{o['ci95'][1]:+.3f}]  better {o['share_households_better']:.1%}   +0.004")

    # --- 5. wealth dynamics at the population theta (§24) --------------------
    sim = rules["population"]
    x0, z0 = state["liquid"], state["illiquid"]
    print(f"\n=== wealth dynamics, wave {N_IN} -> {N_IN + 1}, population theta "
          f"{pop.round(4)}, headline card weights ===")
    header()
    dyn = {"psid": describe("PSID (what they did)", x[:, N_IN, fi["liquid_assets"]] - x0,
                            x[:, N_IN, fi["illiquid_assets"]] - z0, z0)}
    med = {f: median_mix(sim[1][f][:, :, WAVE_YEARS], sim[0][f][:, :, WAVE_YEARS], w_head)
           for f in ("liquid_assets", "illiquid_assets")}
    dyn["model"] = describe("model (median path)", med["liquid_assets"] - x0,
                            med["illiquid_assets"] - z0, z0)
    for c, tag in ((1, "model, all cardholders"), (0, "model, non-borrowers no card")):
        m = need[c]
        mx = np.median(sim[c]["liquid_assets"][m][:, :, WAVE_YEARS], 1)
        mz = np.median(sim[c]["illiquid_assets"][m][:, :, WAVE_YEARS], 1)
        dyn[tag] = describe(tag, mx - x0[m], mz - z0[m], z0[m])

    # --- 6. levels: the model's own households against PSID (§24) -----------
    own = {1: own_card, 0: simulate(pop_sols[(0, 0)], n_households=args.n_own, seed=1)}
    psid_age = x[:, :, fi["age"]].ravel()
    psid_liq = x[:, :, fi["liquid_assets"]].ravel()
    levels = {}
    print("\n=== liquid wealth by age: PSID (gross) against the model's own households ===")
    print(f"{'ages':8s}{'PSID median':>13s}{'PSID debt':>11s}"
          f"{'card med':>11s}{'card debt':>11s}{'no-card med':>13s}{'50/50 debt':>12s}")
    for lo in (35, 40, 45):
        sel = (psid_age >= lo) & (psid_age < lo + 5)
        t = slice(lo - AGE_START_SIM, lo + 5 - AGE_START_SIM)
        cl, nl = own[1]["liquid_assets"][:, t].ravel(), own[0]["liquid_assets"][:, t].ravel()
        levels[f"{lo}-{lo + 4}"] = row = {
            "psid_median": float(np.median(psid_liq[sel])),
            "psid_debt": float((psid_liq[sel] < 0).mean()),
            "card_median": float(np.median(cl)), "card_debt": float((cl < 0).mean()),
            "nocard_median": float(np.median(nl)),
            "mix_debt": float(0.5 * (cl < 0).mean() + 0.5 * (nl < 0).mean())}
        print(f"{lo}-{lo + 4:<5d}{row['psid_median']:>13,.0f}{row['psid_debt']:>11.0%}"
              f"{row['card_median']:>11,.0f}{row['card_debt']:>11.0%}"
              f"{row['nocard_median']:>13,.0f}{row['mix_debt']:>12.0%}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "oos_results.json").write_text(json.dumps(
        {"n_households": H, "run_dirs": list(map(str, args.run_dirs)), "beta_transform": bt,
         "population_theta": pop.tolist(), "n_fallback": int(bad.sum()),
         "borrower_share": float(borrower.mean()),
         "never_borrow_share_by_age": a_by_age, "results": res,
         "wealth_dynamics": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in dyn.items()},
         "levels": levels}, indent=2))
    np.savez(args.out / "oos_household_thetas.npz", theta=mean, fallback=bad,
             borrower=borrower, card_weight=w_head)
    print(f"\nwrote {args.out}/oos_results.json")


if __name__ == "__main__":
    main()
