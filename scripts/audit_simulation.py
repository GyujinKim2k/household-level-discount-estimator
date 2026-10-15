"""Simulation audit (RESULTS 54): the solver and the generated data, checked
three ways -- Laibson et al.'s MATLAB run under Octave (L), the independent
reference model (R, ``hh_npe.simulator.reference``) and our port (P).

Subcommands
-----------
``three_way``
    One theta on one grid through L (an Octave ``.mat`` from
    ``scripts/laibson_octave/run_lifecycle.m``), R and P (``twoasset.solve``,
    the CPU reference of the production GPU solver). Policies are compared as
    dollar values on each age's own grid; then every solver's policies are run
    through R's forward pass on L's own uniforms, and R's forward pass on L's
    policies must reproduce L's panels exactly.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/audit_simulation.py three_way \\
        --octave outputs/audit/octave/test_coarse.mat
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from hh_npe.simulator import reference as ref
from hh_npe.simulator import twoasset

OUT = Path("outputs/audit")


# ------------------------------------------------------------------ loading

def load_octave(path: Path) -> dict:
    """L's arrays, 0-based, policies as indices into each age's own grid."""
    d = loadmat(path)
    T = int(d["X"].shape[1])
    nX = d["nX"].ravel().astype(int)
    L = {"T": T, "nX": nX, "prefs": d["prefs"].ravel(), "returns": d["returns"].ravel(),
         "X": [d["X"][: nX[t], t] for t in range(T)], "Z": d["Z"].ravel(),
         "xmin": d["xmin"].ravel(), "elapsed": float(d["elapsed"].ravel()[0])}
    for k in ("Ix", "Iz"):
        L[k.lower()] = [d[k][: nX[t], :, :, t].astype(np.int64) - 1 for t in range(T)]
    for k in ("C", "V", "EV", "W"):
        L[k] = [d[k][: nX[t], :, :, t] for t in range(T)]
    for k in ("simY", "simX", "simZ", "simC", "simXind", "simZind", "simUind",
              "randU", "randY", "simMoments", "Ymean", "ymean", "effhhN", "zliqpen",
              "xiS", "xiP", "death"):
        L[k] = d[k]
    return L


def port_solution(model: ref.Model, prefs, centred: bool) -> ref.Solution:
    """P's CPU solve on the reference model's grid, re-expressed on each
    age's own grid (P keeps one common grid and a feasibility mask)."""
    c = model.cfg
    spec = twoasset.ModelSpec(xjump=c.xjump, xmax=c.xmax, x_cells_per_step=c.x_cells,
                              zjump=c.zjump, zmax=c.zmax, z_cells_per_step=c.z_cells,
                              R_gamma=c.R_gamma, crra_centred=centred)
    if not c.card:
        spec = dataclasses.replace(spec, calib=dataclasses.replace(
            spec.calib, c0_credit=0.0, c1_credit=0.0, c2_credit=0.0))
    return reexpress(twoasset.solve(*prefs, spec), model, tuple(prefs))


def reexpress(sol, model: ref.Model, prefs) -> ref.Solution:
    """A port Solution (common grid, CPU or GPU) on each age's own grid."""
    Xc = sol.X
    off = [int(np.flatnonzero(Xc == model.X[t][0])[0]) for t in range(model.T)]
    for t in range(model.T):
        assert np.array_equal(Xc[off[t]: off[t] + len(model.X[t])], model.X[t])
    ix, iz, cc = [], [], []
    for t in range(model.T):
        rows = slice(off[t], off[t] + len(model.X[t]))
        nxt = off[t + 1] if t < model.T - 1 else off[t]
        ix.append(sol.next_x[t][rows].astype(np.int64) - nxt)
        iz.append(sol.next_z[t][rows].astype(np.int64))
        cc.append(sol.cons[t][rows].astype(np.float64))
    return ref.Solution(ix, iz, cc, [None] * model.T, [None] * model.T, [None] * model.T,
                        model, tuple(prefs))


def as_solution(L: dict, model: ref.Model) -> ref.Solution:
    return ref.Solution(L["ix"], L["iz"], L["C"], L["V"], L["EV"], L["W"], model,
                        tuple(L["prefs"]))


# ------------------------------------------------------------------ comparing

def agree(a: ref.Solution, b: ref.Solution, model: ref.Model) -> dict:
    """Share of states whose choice (as dollars) and consumption agree."""
    same_x = same_z = n = 0
    dc = 0.0
    for t in range(model.T):
        Xn = model.X[t + 1] if t < model.T - 1 else model.X[t]
        ok = a.c[t] >= 0          # unsolvable states choose arbitrarily in every code
        sx = Xn[a.ix[t]] == Xn[b.ix[t]]
        sz = a.iz[t] == b.iz[t]
        same_x += (sx & ok).sum()
        same_z += (sz & ok).sum()
        n += ok.sum()
        dc = max(dc, float(np.abs(a.c[t] - b.c[t])[ok & sx & sz].max(initial=0)))
    return {"states": int(n), "same_x": same_x / n, "same_z": same_z / n,
            "max_c_diff_where_same": dc}


def value_gap(a: ref.Solution, b: ref.Solution) -> float:
    """Largest relative gap in V over finite entries."""
    g = 0.0
    for va, vb in zip(a.V, b.V, strict=True):
        f = np.isfinite(va) & np.isfinite(vb) & (np.abs(va) < 1e90)
        if f.any():
            g = max(g, float((np.abs(va - vb)[f] / np.maximum(np.abs(va[f]), 1e-12)).max()))
    return g


def panel_stats(p: dict, model: ref.Model) -> dict:
    """What the network reads, ages 35-44 and over 7 biennial waves from 30."""
    a = slice(15, 25)
    my = model.Ymean[None, a]
    c, y = p["consumption"], p["income"]
    w = np.arange(11, 25, 2)                       # ages 31..43, biennial
    dlc = np.diff(np.log(c[:, w]), axis=1)
    return {
        "illiquid/mean income": float(np.median(p["illiquid"][:, a] / my)),
        "liquid/mean income": float(np.median(p["liquid"][:, a] / my)),
        "share in debt": float((p["liquid"][:, a] < 0).mean()),
        "consumption/income": float(np.median(c[:, a] / y[:, a])),
        "sd dlog c (2y)": float(np.std(dlc)),
    }


def panels_equal(p: dict, L: dict) -> dict:
    return {"income": bool(np.array_equal(p["income"], L["simY"])),
            "cash": bool(np.array_equal(p["cash"], L["simX"])),
            "illiquid": bool(np.array_equal(p["illiquid"], L["simZ"])),
            "consumption_maxdiff": float(np.abs(p["consumption"] - L["simC"]).max())}


def cmd_three_way(args) -> None:
    L = load_octave(args.octave)
    beta, delta, rho = (float(v) for v in L["prefs"])
    R_gamma = float(L["returns"][1])
    card = bool(L["xmin"].max() > 0)
    X0 = L["X"][0]
    pos = X0[X0 >= 0]
    xjump = float(pos[1] - pos[0])
    zjump = float(L["Z"][1] - L["Z"][0])
    base = dict(xjump=xjump, zjump=zjump, R_gamma=R_gamma, card=card)
    fs = ref.first_stage()
    res = {"octave": str(args.octave), "prefs": [beta, delta, rho], "R_gamma": R_gamma,
           "card": card, "xjump": xjump, "zjump": zjump, "octave_seconds": L["elapsed"]}

    m_l = ref.Model(ref.Config(**base), fs)            # MATLAB's choices: z-first, uncentred
    assert all(np.array_equal(m_l.X[t], L["X"][t]) for t in range(m_l.T)), "X grids differ"
    assert np.array_equal(m_l.Z, L["Z"]), "Z grid differs"
    res["inputs_maxdiff"] = {
        "Ymean": float(np.abs(m_l.Ymean / L["Ymean"].ravel() - 1).max()),
        "hhs": float(np.abs(m_l.hhs - L["effhhN"].ravel()).max()),
        "P": float(np.abs(m_l.P - L["xiP"]).max()),
        "death": float(np.abs(m_l.death - L["death"].ravel()).max()),
        "pen": float(np.abs(m_l.pen - L["zliqpen"].ravel()).max())}
    sols = {"L": as_solution(L, m_l)}
    variants = {"R": {}, "R x_first": {"tie": "x_first"},
                "R round_income": {"round_income": True}, "R centred": {"centred": True},
                "R centred x_first": {"centred": True, "tie": "x_first"}}
    timing = {}
    for name, kw in variants.items():
        t0 = time.time()
        sols[name] = ref.solve(beta, delta, rho, ref.Model(ref.Config(**base, **kw), fs))
        timing[name] = time.time() - t0
    for name, centred in (("P", False), ("P centred", True)):
        t0 = time.time()
        sols[name] = port_solution(m_l, (beta, delta, rho), centred)
        timing[name] = time.time() - t0
    res["seconds"] = timing

    pairs = [("L", "R"), ("L", "R x_first"), ("L", "R round_income"), ("L", "P"),
             ("R x_first", "P"), ("R centred x_first", "P centred"), ("R", "R centred"),
             ("P", "P centred")]
    res["policy_agreement"] = {f"{a} vs {b}": agree(sols[a], sols[b], m_l) for a, b in pairs}
    res["value_gap"] = {f"{a} vs {b}": value_gap(sols[a], sols[b])
                        for a, b in (("L", "R"), ("L", "R x_first"), ("R", "R centred"))}

    u = {"u_state": L["randU"], "u_income": L["randY"]}
    i0 = (L["simXind"][:, 0].astype(np.int64) - 1, L["simZind"][:, 0].astype(np.int64) - 1)
    # simXind(:, 1) is overwritten with the post-income index; the pre-income
    # seed is their single SCF point, which initial_indices recomputes.
    seed = ref.initial_indices(m_l, L["randU"].shape[0])
    del i0
    panels = {name: ref.simulate(s, u, init=seed) for name, s in sols.items()}
    res["forward_R_on_L_policies_reproduces_L"] = panels_equal(panels["L"], L)
    res["panel_stats"] = {name: panel_stats(p, m_l) for name, p in panels.items()}
    res["moments_L"] = L["simMoments"].ravel().tolist()
    for k, v in res["policy_agreement"].items():
        print(f"{k:28s} same x' {v['same_x']:.5f}  same z' {v['same_z']:.5f}  "
              f"max |dc| where same {v['max_c_diff_where_same']:.3g}")
    print("value gaps:", {k: f"{v:.3g}" for k, v in res["value_gap"].items()})
    print("forward pass on L's policies reproduces L:", res["forward_R_on_L_policies_reproduces_L"])
    keys = list(next(iter(res["panel_stats"].values())))
    print(f"{'':20s}" + "".join(f"{k[:16]:>18s}" for k in keys))
    for name, st in res["panel_stats"].items():
        print(f"{name:20s}" + "".join(f"{st[k]:18.4f}" for k in keys))
    out = args.out or OUT / f"three_way_{Path(args.octave).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


# ------------------------------------------------------------- stored data

#: The A2 shard directories every current model is trained or scored on, and
#: the draw count each generation run was launched with (run_optionA2_full.sh,
#: run_optionA2_strip.sh, run_optionA2n_sbc.sh).
DATASETS = {
    "main": (Path("data/processed/couples_dataset_shards"), 32768),
    "strip": (Path("data/processed/couples_strip_dataset_shards"), 3072),
    "sbc16": (Path("data/processed/optionA2_sbc16_shards"), 1024),
    "sbc16_liqpen050": (Path("data/processed/optionA2_sbc16_liqpen050_shards"), 1024),
    "sbc16_liqpen150": (Path("data/processed/optionA2_sbc16_liqpen150_shards"), 1024),
}
POOL = Path("data/processed/seed_pool_couples_income.npz")
WINDOW = dict(start_age=30, n_waves=7, wave_years=2)


def regenerate_draws(cfg: dict, n: int):
    """theta (with R_gamma) and card exactly as generate_dataset.py drew them
    for this run's recorded configuration."""
    from hh_npe.npe.prior import PROPOSAL_SAMPLERS
    from scripts.generate_dataset import _per_block_rgamma

    seed = int(cfg["seed"])
    theta = PROPOSAL_SAMPLERS[cfg["proposal"]](n, seed)
    card = (np.random.default_rng(seed + 1).integers(0, 2, size=n)
            if cfg["card_types"] else None)
    ns = argparse.Namespace(theta_batch=int(cfg["theta_batch"]), block=512, n_samples=n,
                            rgamma_range=cfg["rgamma_range"], seed=seed)
    return _per_block_rgamma(theta, card, ns)


def shards(d: Path) -> list[Path]:
    return sorted(d.glob("shard_*.npz"))


def cmd_integrity(args) -> None:
    from hh_npe.data.waves import FEATURES_TWOASSET, aggregate_waves
    from hh_npe.simulator import grids
    from hh_npe.simulator import laibson_calibration as cal
    from hh_npe.simulator.dispatch import draw_initial_conditions
    from scripts.generate_dataset import _load_pool

    pool, digest = _load_pool(POOL)
    states, P = grids.tauchen(c=cal.COMPHS)
    res = {}
    for name, (d, n) in DATASETS.items():
        cfg = json.loads((d / "solver_config.json").read_text())
        bad: list[str] = []
        if cfg.get("init_pool_sha256") != digest:
            bad.append("seed pool hash differs from the one generation recorded")
        if not cfg.get("crra_centred"):
            bad.append("not generated with the centred (fixed) solver")
        theta, card = regenerate_draws(cfg, n)
        expected_lo, n_rows = 0, 0
        for sf in shards(d):
            z = np.load(sf)
            lo, hi, M = int(z["lo"]), int(z["hi"]), int(z["n_households"])
            tag = sf.name
            if lo != expected_lo:
                bad.append(f"{tag}: lo {lo}, expected {expected_lo}")
            expected_lo = hi
            if M != 16:
                bad.append(f"{tag}: {M} households per draw")
            if not np.array_equal(z["theta"], theta[lo:hi]):
                bad.append(f"{tag}: stored theta differs from the proposal's")
            if card is not None and not np.array_equal(z["card"], card[lo:hi]):
                bad.append(f"{tag}: stored card differs")
            got = [draw_initial_conditions(pool, int(cfg["seed"]) + lo + 1, j, M, states, P,
                                           cal.COMPHS.ywork_sigmanu) for j in range(hi - lo)]
            if not np.array_equal(z["init_wealth"], np.concatenate([g[0] for g in got])):
                bad.append(f"{tag}: init_wealth differs")
            if not np.array_equal(z["init_state"], np.concatenate([g[1] for g in got])):
                bad.append(f"{tag}: init_state differs")
            panel = {k[6:]: z[k] for k in z.files if k.startswith("panel_")}
            for k, v in list(panel.items()) + [("x", z["x"])]:
                if not np.isfinite(v).all():
                    bad.append(f"{tag}: {k} not finite")
            if not (panel["t_age"] == np.arange(71)).all():
                bad.append(f"{tag}: t_age is not 0..70")
            x, alive = aggregate_waves(panel, age_start_sim=20, features=FEATURES_TWOASSET,
                                       **WINDOW)
            if not np.array_equal(x, z["x"]) or not np.array_equal(alive, z["alive"]):
                bad.append(f"{tag}: x is not aggregate_waves(panel)")
            if not z["alive"].all():
                bad.append(f"{tag}: alive is not all True")
            n_rows += len(z["x"])
        if expected_lo != n and name in ("main", "strip", "sbc16"):
            bad.append(f"draws end at {expected_lo}, the run had {n}")
        res[name] = {"shards": len(shards(d)), "draws": expected_lo, "rows": n_rows,
                     "problems": bad}
        print(f"{name:16s} {len(shards(d)):3d} shards, {expected_lo:6d} draws, {n_rows:7d} rows: "
              + ("OK" if not bad else f"{len(bad)} problems: " + "; ".join(bad[:5])))
    res["proposal_weights"] = proposal_check()
    out = OUT / "integrity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


def proposal_check(n_bins: int = 8) -> dict:
    """Do the importance weights turn the stored training draws back into the
    uniform prior? Weighted bin masses against the uniform prior's, 1-D per
    parameter and 2-D per pair, for main + strip as the A2n checkpoints
    weight them (prior._parts_log_weight)."""
    from hh_npe.npe.prior import _parts_log_weight

    th, parts = [], []
    for name in ("main", "strip"):
        d, n = DATASETS[name]
        cfg = json.loads((d / "solver_config.json").read_text())
        th.append(np.concatenate([np.load(f)["theta"] for f in shards(d)]))
        parts.append({"name": cfg["proposal"], "n_train": n, "rgamma_range": cfg["rgamma_range"]})
    th = np.concatenate(th)
    w = np.exp(_parts_log_weight(parts, th))
    w = w / w.sum()
    lo = np.array([0.30, 0.85, 0.5, 1.021])
    hi = np.array([1.00, 1.00, 5.0, 1.075])
    u = (th - lo) / (hi - lo)
    out = {"n": int(len(th)), "ess": float(1 / (w ** 2).sum())}
    names = ("beta", "delta", "rho", "R_gamma")
    worst = 0.0
    for j in range(4):
        b = np.minimum((u[:, j] * n_bins).astype(int), n_bins - 1)
        mass = np.bincount(b, weights=w, minlength=n_bins)
        # Standard error of a weighted bin mass under the weights' own spread.
        se = np.sqrt(np.bincount(b, weights=w ** 2, minlength=n_bins))
        z = (mass - 1 / n_bins) / se
        out[names[j]] = {"mass": mass.round(4).tolist(), "max_abs_z": float(np.abs(z).max())}
        worst = max(worst, float(np.abs(z).max()))
    for j in range(4):
        for k in range(j + 1, 4):
            bj = np.minimum((u[:, j] * 4).astype(int), 3)
            bk = np.minimum((u[:, k] * 4).astype(int), 3)
            cell = bj * 4 + bk
            mass = np.bincount(cell, weights=w, minlength=16)
            se = np.sqrt(np.bincount(cell, weights=w ** 2, minlength=16))
            z = (mass - 1 / 16) / se
            out[f"{names[j]}x{names[k]}"] = float(np.abs(z).max())
            worst = max(worst, float(np.abs(z).max()))
    out["max_abs_z"] = worst
    print(f"proposal weights: ESS {out['ess']:.0f} of {out['n']}; largest |z| of a weighted "
          f"bin mass against the uniform prior {worst:.2f} "
          f"(1-D {n_bins} bins x 4, 2-D 4x4 x 6 pairs)")
    return out


def budget_check(cash, cons, ill, inc, rg, Xg, pen, R, R_CC):
    """The budget identity between consecutive years of a stored panel.

    This year's consumption (less the dividend), the illiquid move and its
    penalty imply what next year's liquid position cost today; that cost
    times R (R_CC if borrowing) must be a point of the X grid that, with next
    year's income, snaps to next year's cash. Returns (ok, |implied x' - the
    nearest grid point|), both (n, T - 1)."""
    c_ex = cons - (rg - 1.0) * ill
    dz = ill[:, 1:] - ill[:, :-1]
    cost = cash[:, :-1] - c_ex[:, :-1] - dz + pen[None, :-1] * np.minimum(dz, 0)
    xp = np.where(cost >= 0, cost * R, cost * R_CC)
    j = np.clip(np.searchsorted(Xg, xp), 1, len(Xg) - 1)
    near = np.where(np.abs(Xg[j - 1] - xp) < np.abs(Xg[j] - xp), Xg[j - 1], Xg[j])
    gap = np.abs(near - xp)
    snapped = twoasset._nearest_index(Xg, near + inc[:, 1:])
    return (gap < 1e-3) & (Xg[snapped] == cash[:, 1:]), gap


def cmd_invariants(args) -> None:
    """Every stored household-year against the model's own rules, and the
    budget identity re-derived from consecutive years."""
    from hh_npe.simulator import grids
    from hh_npe.simulator import laibson_calibration as cal

    age = grids.ages()
    X_card, _ = grids.liquid_grid(age, 1000.0, 4e5, grids.X_CELLS_PER_STEP, c=cal.COMPHS)
    nocard = dataclasses.replace(cal.COMPHS, c0_credit=0.0, c1_credit=0.0, c2_credit=0.0)
    X_none, _ = grids.liquid_grid(age, 1000.0, 4e5, grids.X_CELLS_PER_STEP, c=nocard)
    Z = grids.illiquid_grid()
    xmin = grids.credit_limit(age, 1000.0, cal.COMPHS)
    R, R_CC = cal.R_FREE, cal.R_CC
    tot: dict = {}

    def add(k, v):
        tot[k] = tot.get(k, 0) + v

    worst_budget = 0.0
    for d, _n in DATASETS.values():
        # The penalty sets of RESULTS 43.11 were generated at 0.5x and 1.5x it.
        cfg = json.loads((d / "solver_config.json").read_text())
        pen = grids.liquidation_penalty(age) * float(cfg.get("liqpen_scale", 1.0))
        for sf in shards(d):
            z = np.load(sf)
            M = int(z["n_households"])
            card = np.repeat(z["card"], M).astype(bool)
            rg = np.repeat(z["theta"][:, 3], M)[:, None]
            y, c = z["panel_income"], z["panel_consumption"]
            liq, ill = z["panel_liquid_assets"], z["panel_illiquid_assets"]
            cash = z["panel_cash_on_hand"]
            hh_years = y.size
            add("household-years", hh_years)
            add("income <= 0", int((y <= 0).sum()))
            add("income off the $1,000 lattice", int((y % 1000 != 0).sum()))
            add("cash != liquid + income", int((cash != liq + y).sum()))
            on_x = np.where(card[:, None], np.isin(cash, X_card), np.isin(cash, X_none))
            add("cash off its X grid", int((~on_x).sum()))
            add("illiquid off the Z grid", int((~np.isin(ill, Z)).sum()))
            add("consumption <= 0", int((c <= 0).sum()))
            add("cardholder liquid below -credit limit",
                int(((liq < -xmin[None, :]) & card[:, None]).sum()))
            add("no-card liquid < 0 (phantom debt)", int(((liq < 0) & ~card[:, None]).sum()))
            add("no-card household-years", int((~card).sum()) * y.shape[1])
            add("cash at the X top ($400k)", int((cash >= X_card[-1]).sum()))
            add("illiquid at the Z top", int((ill >= Z[-1]).sum()))
            add("cardholder at the credit floor",
                int(((cash <= -xmin[None, :]) & card[:, None]).sum()))
            ok = np.zeros((len(y), y.shape[1] - 1), dtype=bool)
            gap = np.full(ok.shape, np.inf)
            for rows, Xg in ((card, X_card), (~card, X_none)):
                if rows.any():
                    ok[rows], gap[rows] = budget_check(
                        cash[rows], c[rows], ill[rows], y[rows], rg[rows], Xg, pen, R, R_CC)
            add("budget: implied x' off the grid or not snapping to next cash", int((~ok).sum()))
            worst_budget = max(worst_budget, float(gap[np.isfinite(gap)].max()))
    tot["largest |implied x' - grid point| ($)"] = worst_budget
    for k, v in tot.items():
        share = "" if k.startswith(("household", "largest", "no-card household")) else \
            f"   ({v / tot['household-years']:.3%})"
        print(f"{k:62s} {v:>14,}{share}" if isinstance(v, int) else f"{k:62s} {v:14.6g}")
    out = OUT / "invariants.json"
    out.write_text(json.dumps(tot, indent=1))
    print(f"wrote {out}")


STAT_NAMES = ("illiquid / mean income", "liquid / mean income", "share of years in debt",
              "consumption / income", "sd dlog c (2y)", "sd dlog y (2y)",
              "share at a grid cap")


def household_stats(z, Ymean: np.ndarray) -> np.ndarray:
    """Per household: ages 35-44 medians and shares, and the 2-year log
    changes over the 7-wave window from 30 the shards were aggregated at."""
    a = slice(15, 25)
    ill, liq = z["panel_illiquid_assets"], z["panel_liquid_assets"]
    c, y, cash = z["panel_consumption"], z["panel_income"], z["panel_cash_on_hand"]
    w = np.arange(11, 25, 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlc = np.diff(np.log(c[:, w]), axis=1)
        dly = np.diff(np.log(y[:, w]), axis=1)
    cap = (cash[:, a] >= 4e5) | (ill[:, a] >= 3.4e6)
    return np.column_stack([
        np.median(ill[:, a] / Ymean[None, a], axis=1),
        np.median(liq[:, a] / Ymean[None, a], axis=1),
        (liq[:, a] < 0).mean(axis=1),
        np.median(c[:, a] / y[:, a], axis=1),
        dlc.std(axis=1), dly.std(axis=1), cap.mean(axis=1)])


def cmd_statics(args) -> None:
    """Behaviour across theta, the detector for RESULTS 40's kind of artefact:
    medians of each household statistic in bins of each parameter, by card
    type. Uniform draws only (the main run's 4,096-draw pilot and the SBC set)
    for the 1-D marginals; every draw for the 2-D cells."""
    from hh_npe.simulator import grids

    Ymean = grids.mean_income(grids.ages())
    rows = []
    for name in ("main", "strip", "sbc16"):
        d, _ = DATASETS[name]
        for sf in shards(d):
            z = np.load(sf)
            M = int(z["n_households"])
            lo = int(z["lo"])
            uniform = (name == "sbc16") or (name == "main" and lo < 4096)
            st = household_stats(z, Ymean)
            th = np.repeat(z["theta"], M, axis=0)
            card = np.repeat(z["card"], M)
            rows.append(np.column_stack([th, card, np.full(len(st), uniform), st]))
    A = np.concatenate(rows)
    np.save(OUT / "statics_households.npy", A.astype(np.float32))
    th, card, uni, S = A[:, :4], A[:, 4].astype(bool), A[:, 5].astype(bool), A[:, 6:]
    names = ("beta", "delta", "rho", "R_gamma")
    lo = np.array([0.30, 0.85, 0.5, 1.021])
    hi = np.array([1.00, 1.00, 5.0, 1.075])
    nb = 10
    res = {"n_households": int(len(A)), "n_uniform": int(uni.sum()), "marginal": {},
           "flags": []}
    for j, p in enumerate(names):
        b = np.minimum(((th[:, j] - lo[j]) / (hi[j] - lo[j]) * nb).astype(int), nb - 1)
        res["marginal"][p] = {}
        for cname, cm in (("card", card), ("no card", ~card)):
            sel = uni & cm
            med = np.array([[np.nanmedian(S[sel & (b == k), s]) if (sel & (b == k)).sum() > 50
                             else np.nan for k in range(nb)] for s in range(S.shape[1])])
            res["marginal"][p][cname] = med.round(5).tolist()
            for s, sname in enumerate(STAT_NAMES):
                v = med[s]
                steps = np.abs(np.diff(v))
                typical = np.nanmedian(steps)
                for k in np.flatnonzero(steps > 3 * typical + 1e-9):
                    if steps[k] > 0.05 * (np.nanmax(np.abs(v)) + 1e-12):
                        res["flags"].append(f"{p} bins {k}->{k + 1} ({cname}): {sname} "
                                            f"jumps {v[k]:.4g} -> {v[k + 1]:.4g}")
    # 2-D cells, every draw: the cell step against its neighbours.
    for j, k in ((2, 1), (0, 1), (2, 3), (2, 0)):
        bj = np.minimum(((th[:, j] - lo[j]) / (hi[j] - lo[j]) * 8).astype(int), 7)
        bk = np.minimum(((th[:, k] - lo[k]) / (hi[k] - lo[k]) * 8).astype(int), 7)
        for cname, cm in (("card", card), ("no card", ~card)):
            for s, sname in enumerate(STAT_NAMES[:5]):
                grid = np.full((8, 8), np.nan)
                for a in range(8):
                    for c in range(8):
                        m = cm & (bj == a) & (bk == c)
                        if m.sum() > 100:
                            grid[a, c] = np.median(S[m, s])
                res.setdefault("cells", {})[f"{names[j]}x{names[k]} {cname} {sname}"] = \
                    grid.round(5).tolist()
                along = np.abs(np.diff(grid, axis=0))
                typical = np.nanmedian(along)
                for a, c in zip(*np.nonzero(along > 4 * typical + 1e-9), strict=True):
                    if along[a, c] > 0.1 * (np.nanmax(np.abs(grid)) + 1e-12):
                        res["flags"].append(
                            f"{names[j]} bins {a}->{a + 1} at {names[k]} bin {c} ({cname}): "
                            f"{sname} {grid[a, c]:.4g} -> {grid[a + 1, c]:.4g}")
    # Directions economics fixes, on the uniform draws (Spearman over bins).
    from scipy.stats import spearmanr
    res["directions"] = {}
    for p, s, sign in (("delta", 0, +1), ("beta", 0, +1), ("R_gamma", 0, +1),
                       ("beta", 2, -1), ("delta", 2, -1)):
        for cname in ("card", "no card"):
            v = np.array(res["marginal"][p][cname][s])
            ok = np.isfinite(v)
            r = spearmanr(np.arange(nb)[ok], v[ok]).statistic if ok.sum() > 3 else np.nan
            res["directions"][f"{STAT_NAMES[s]} vs {p} ({cname})"] = {
                "spearman": float(r), "expected_sign": sign,
                "agrees": bool(np.sign(r) == sign) if np.isfinite(r) else None}
    # Degenerate draws: all 16 households with identical illiquid paths.
    out = OUT / "statics.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"{len(A):,} households ({uni.sum():,} from uniform draws)")
    for k, v in res["directions"].items():
        print(f"  {k:52s} spearman {v['spearman']:+.2f}  expected {v['expected_sign']:+d}  "
              f"{'OK' if v['agrees'] else ('n/a' if v['agrees'] is None else 'NO')}")
    print(f"{len(res['flags'])} flags")
    for f in res["flags"][:60]:
        print("  " + f)
    print(f"wrote {out}")


def window_summary(inc, cons, liq, ill) -> dict:
    """The relations between flows and stocks a 7-wave window carries."""
    with np.errstate(divide="ignore", invalid="ignore"):
        dlc = np.diff(np.log(cons), axis=1)
        dly = np.diff(np.log(inc), axis=1)
        return {
            "median liquid / income": float(np.median(liq / inc)),
            "median illiquid / income": float(np.median(ill / inc)),
            "median consumption / income": float(np.median(cons / inc)),
            "share of waves in debt": float((liq < 0).mean()),
            "sd dlog c": float(dlc.std()),
            "sd dlog y": float(dly.std()),
            "corr(dlog c, dlog y)": float(np.corrcoef(dlc.ravel(), dly.ravel())[0, 1]),
            "corr(liquid/income, dlog y next)": float(np.corrcoef(
                (liq[:, :-1] / inc[:, :-1]).ravel(), dly.ravel())[0, 1]),
        }


def cmd_timing(args) -> None:
    """H1: a PSID wave reports wealth and age at the interview but income (and
    most consumption) for the calendar year before. The simulator's wave reads
    all of them in one simulated year, where its stocks are the position at
    the START of that year -- the interview's -- so its flows are a year later
    than PSID's. This re-reads the stored panels with flows from the year
    before and reports how far the window statistics move, for households in
    PSID couples' A2n region (RESULTS 43.8)."""
    rows_now, rows_psid = [], []
    region = dict(beta=(0.50, 0.70), delta=(0.965, 0.995), R_gamma=(1.043, 1.062))
    for name in ("main", "strip"):
        d, _ = DATASETS[name]
        for sf in shards(d):
            z = np.load(sf)
            M = int(z["n_households"])
            th = np.repeat(z["theta"], M, axis=0)
            m = ((th[:, 0] >= region["beta"][0]) & (th[:, 0] <= region["beta"][1])
                 & (th[:, 1] >= region["delta"][0]) & (th[:, 1] <= region["delta"][1])
                 & (th[:, 3] >= region["R_gamma"][0]) & (th[:, 3] <= region["R_gamma"][1]))
            if not m.any():
                continue
            y, c = z["panel_income"][m], z["panel_consumption"][m]
            liq, ill = z["panel_liquid_assets"][m], z["panel_illiquid_assets"][m]
            # Window from 30, as stored: wave w at index 11 + 2w (ages 31..43).
            t = 11 + 2 * np.arange(7)
            rows_now.append((y[:, t], c[:, t], liq[:, t], ill[:, t]))
            rows_psid.append((y[:, t - 1], c[:, t - 1], liq[:, t], ill[:, t]))
    cat = lambda rs: [np.concatenate([r[k] for r in rs]) for k in range(4)]  # noqa: E731
    now, psid = window_summary(*cat(rows_now)), window_summary(*cat(rows_psid))
    res = {"region": region, "households": int(sum(len(r[0]) for r in rows_now)),
           "as_generated": now, "flows_a_year_earlier": psid}
    print(f"{res['households']:,} households in PSID couples' A2n region")
    print(f"{'':36s}{'as generated':>14s}{'flows 1y earlier':>18s}")
    for k in now:
        print(f"{k:36s}{now[k]:14.4f}{psid[k]:18.4f}")
    out = OUT / "timing.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


# ------------------------------------------------------------------ GPU

#: (dataset, shard index, block of 16 within the shard) re-solved by
#: ``reproduce``: the uniform pilot, the switched main run, the strip, the
#: SBC set.
REPRO_BLOCKS = (("main", 0, 0), ("main", 40, 5), ("strip", 2, 3), ("sbc16", 1, 7))


def solve_block(name: str, shard: int, block: int, margins: dict | None = None):
    """One stored block re-solved exactly as generation solved it: the same
    draws, seeds, card type, R_gamma and seed-pool starts, one GPU batch of 16.
    With ``margins``, through ``solve_batch(margins=...)`` and the forward
    pass directly instead of ``simulate_batch_twoasset_gpu``."""
    from hh_npe.simulator.dispatch import draw_initial_conditions, simulate_batch_twoasset_gpu
    from scripts.generate_dataset import _load_pool

    d, _ = DATASETS[name]
    cfg = json.loads((d / "solver_config.json").read_text())
    z = np.load(shards(d)[shard])
    lo, M, tb = int(z["lo"]), int(z["n_households"]), int(cfg["theta_batch"])
    k0 = block * tb
    sel = slice(k0, k0 + tb)
    theta, card = z["theta"][sel], z["card"][sel]
    pool = _load_pool(POOL)[0]
    seed_base = int(cfg["seed"]) + lo + 1 + k0
    stored = {k[6:]: z[k][k0 * M:(k0 + tb) * M] for k in z.files if k.startswith("panel_")}
    if margins is None:
        _x, _a, panels = simulate_batch_twoasset_gpu(
            theta, seed_base, 30, 7, 2, cfg["grid"], tb, int(cfg["chunk"]), True, M,
            None, card, theta[:, 3], pool, float(cfg.get("liqpen_scale", 1.0)))
        return panels, stored, None
    from hh_npe.simulator import laibson_calibration as cal
    from hh_npe.simulator.twoasset import GRIDS
    from hh_npe.simulator.twoasset_gpu import solve_batch

    assert len(set(card)) == 1 and len(set(theta[:, 3])) == 1
    calib = cal.COMPHS if card[0] else dataclasses.replace(
        cal.COMPHS, c0_credit=0.0, c1_credit=0.0, c2_credit=0.0)
    spec = dataclasses.replace(GRIDS[cfg["grid"]], calib=calib, R_gamma=float(theta[0, 3]),
                               liqpen_scale=float(cfg.get("liqpen_scale", 1.0)))
    sols = solve_batch(theta[:, :3], spec, theta_batch=tb, chunk=int(cfg["chunk"]),
                       margins=margins)
    panels = []
    for j, sol in enumerate(sols):
        init, s0 = draw_initial_conditions(pool, seed_base, j, M, sol.states, sol.P,
                                           spec.calib.ywork_sigmanu)
        panels.append(twoasset.simulate(sol, n_households=M, seed=seed_base + j,
                                        initial_wealth=init, initial_state=s0))
    merged = {k: np.concatenate([p[k] for p in panels]) for k in panels[0]}
    return merged, stored, sols


def cmd_reproduce(args) -> None:
    import torch

    res = {"gpu": torch.cuda.get_device_name(0), "blocks": []}
    for name, shard, block in REPRO_BLOCKS:
        for path in ("generation", "margins on"):
            t0 = time.time()
            panels, stored, _ = solve_block(name, shard, block,
                                            None if path == "generation" else {})
            same = {k: bool(np.array_equal(panels[k], stored[k])) for k in stored}
            res["blocks"].append({"dataset": name, "shard": shard, "block": block,
                                  "path": path, "identical": same,
                                  "seconds": round(time.time() - t0, 1)})
            print(f"{name:6s} shard {shard:2d} block {block}  {path:10s}  "
                  f"{'bit-identical' if all(same.values()) else 'DIFFERS: ' + str(same)}"
                  f"  ({time.time() - t0:.0f} s)", flush=True)
    out = OUT / "reproduce.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


RHOS = (0.5, 1.0, 2.0, 3.5, 4.5, 5.0)
#: (card, R_gamma) -> two (beta, delta) corners; with RHOS, 60 theta in all.
CORNERS = {(1, 1.021): ((0.3, 0.85), (1.0, 0.95)), (1, 1.075): ((0.3, 0.99), (0.6, 0.9995)),
           (0, 1.021): ((0.3, 0.99), (0.6, 0.9995)), (0, 1.075): ((0.3, 0.85), (1.0, 0.95)),
           # The top of the delta box, where alpha / (1 - delta) reaches 5e4;
           # 1.25% of the main run's draws have delta > 0.999.
           (1, 1.05): ((0.6, 0.99999), (0.9, 0.99999))}


def spec_for(card: int, R_gamma: float, **grid):
    from hh_npe.simulator import laibson_calibration as cal
    from hh_npe.simulator.twoasset import GRIDS

    calib = cal.COMPHS if card else dataclasses.replace(
        cal.COMPHS, c0_credit=0.0, c1_credit=0.0, c2_credit=0.0)
    return dataclasses.replace(GRIDS["full"], calib=calib, R_gamma=float(R_gamma), **grid)


def households(sol, n: int, seed: int) -> dict:
    """``n`` households from the A2 seed pool, as generation starts them."""
    from hh_npe.simulator.dispatch import draw_initial_conditions
    from scripts.generate_dataset import _load_pool

    pool = _load_pool(POOL)[0]
    init, s0 = draw_initial_conditions(pool, seed, 0, n, sol.states, sol.P,
                                       sol.spec.calib.ywork_sigmanu)
    return twoasset.simulate(sol, n_households=n, seed=seed, initial_wealth=init,
                             initial_state=s0)


def visited(sol, panel) -> tuple[np.ndarray, ...]:
    """(t, x index, z index, s) of every household-year."""
    n, T = panel["cash_on_hand"].shape
    ix = np.searchsorted(sol.X, panel["cash_on_hand"])
    iz = np.searchsorted(sol.Z, panel["illiquid_assets"])
    assert np.array_equal(sol.X[ix], panel["cash_on_hand"])
    t = np.broadcast_to(np.arange(T), (n, T))
    return t, ix, iz, panel["income_state"]


def cmd_margins(args) -> None:
    """RESULTS 40's failure mode, looked for at every corner: how close the
    winning choice is to the runner-up, relative to the payoff's scale, at
    the states households actually visit; and whether solving a draw alone
    rather than in a batch moves any visited choice."""
    from hh_npe.simulator.twoasset_gpu import solve_batch

    res = {"theta": []}
    for (card, rg), pairs in CORNERS.items():
        thetas = np.array([(b, d, r) for b, d in pairs for r in RHOS])
        spec = spec_for(card, rg)
        mg: dict = {}
        t0 = time.time()
        sols = solve_batch(thetas, spec, theta_batch=len(thetas), chunk=16, margins=mg)
        alone = [solve_batch(th[None], spec, theta_batch=1, chunk=16)[0] for th in thetas]
        for j, (sol, sol1) in enumerate(zip(sols, alone, strict=True)):
            p = households(sol, args.n, seed=7_000 + j)
            p1 = households(sol1, args.n, seed=7_000 + j)
            t, ix, iz, s = visited(sol, p)
            young = t <= 40                                  # ages 20-60
            row = {"card": card, "R_gamma": rg, "theta": thetas[j].tolist()}
            for k in ("beh", "val"):
                m = mg[k][t, j, ix, iz, s]
                row[k] = {"ages 20-60 share < 1e-12": float((m[young] < 1e-12).mean()),
                          "ages 20-60 share tied (0)": float((m[young] == 0).mean()),
                          "all ages share < 1e-12": float((m < 1e-12).mean()),
                          "ages 20-60 min": float(m[young].min()),
                          "ages 20-60 p1": float(np.percentile(m[young], 1))}
            same = (sol.next_x[t, ix, iz, s] == sol1.next_x[t, ix, iz, s]) & \
                   (sol.next_z[t, ix, iz, s] == sol1.next_z[t, ix, iz, s])
            row["batch vs alone: visited choices equal"] = float(same.mean())
            row["batch vs alone: panels identical"] = bool(all(
                np.array_equal(p[k], p1[k]) for k in p))
            res["theta"].append(row)
            b, v = row["beh"], row["val"]
            same_p = row["batch vs alone: panels identical"]
            print(f"card {card} Rg {rg} theta {np.round(thetas[j], 4)}: beh <1e-12 "
                  f"{b['ages 20-60 share < 1e-12']:.4f} tied {b['ages 20-60 share tied (0)']:.4f}"
                  f" p1 {b['ages 20-60 p1']:.2e} | val <1e-12 "
                  f"{v['ages 20-60 share < 1e-12']:.4f} | batch=alone "
                  f"{row['batch vs alone: visited choices equal']:.4f} "
                  f"{'panels same' if same_p else 'PANELS DIFFER'}", flush=True)
        print(f"  group done in {time.time() - t0:.0f} s", flush=True)
        del mg, sols, alone
    out = OUT / "margins.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


#: theta for the CPU parity and grid refinement runs: PSID couples' A2n region
#: (RESULTS 43.8) and the box's edges.
PROBES = ((1, 1.050, (0.60, 0.98, 2.7)), (0, 1.050, (0.60, 0.98, 2.7)),
          (1, 1.060, (0.50, 0.99, 1.5)), (1, 1.030, (0.70, 0.97, 4.0)),
          (1, 1.075, (0.35, 0.9995, 4.8)), (0, 1.021, (0.95, 0.87, 0.6)))


def _cpu_solve(job):
    card, rg, th = job
    return twoasset.solve(*th, spec_for(card, rg))


def cmd_parity(args) -> None:
    """The production GPU solver against its CPU reference, full grid."""
    from concurrent.futures import ProcessPoolExecutor

    from hh_npe.simulator.twoasset_gpu import solve_batch

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(PROBES)) as ex:
        cpu = list(ex.map(_cpu_solve, PROBES))
    print(f"CPU solves done in {time.time() - t0:.0f} s", flush=True)
    res = []
    for (card, rg, th), sc in zip(PROBES, cpu, strict=True):
        sg = solve_batch(np.array([th]), spec_for(card, rg), theta_batch=1, chunk=16)[0]
        ok = sc.solvable & sg.solvable
        same = (sc.next_x == sg.next_x) & (sc.next_z == sg.next_z)
        pc, pg = households(sc, args.n, 11), households(sg, args.n, 11)
        t, ix, iz, s = visited(sg, pg)
        row = {"card": card, "R_gamma": rg, "theta": list(th),
               "solvable states equal": float(same[ok].mean()),
               "visited choices equal": float(same[t, ix, iz, s].mean()),
               "panels identical": bool(all(np.array_equal(pc[k], pg[k]) for k in pc)),
               "max |c| diff where same": float(np.abs(sc.cons - sg.cons)[ok & same].max())}
        res.append(row)
        print(row, flush=True)
    out = OUT / "parity.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


def window_stats(p: dict) -> dict:
    """The statistics of RESULTS 43.6's table, on 7-wave windows from 30."""
    from hh_npe.simulator import grids

    Ym = grids.mean_income(grids.ages())
    a = slice(15, 25)
    w = np.arange(11, 25, 2)
    c, y = p["consumption"], p["income"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dlc = np.diff(np.log(c[:, w]), axis=1)
    return {"illiquid / mean income": float(np.median(p["illiquid_assets"][:, a] / Ym[a])),
            "liquid / mean income": float(np.median(p["liquid_assets"][:, a] / Ym[a])),
            "share in debt": float((p["liquid_assets"][:, a] < 0).mean()),
            # Debt the grid can resolve either way: more than 2% of mean income.
            "share in debt > 2% of mean income": float(
                (p["liquid_assets"][:, a] < -0.02 * Ym[a]).mean()),
            "mean debt / mean income, debtors": float(
                (-p["liquid_assets"][:, a] / Ym[a])[p["liquid_assets"][:, a] < 0].mean()),
            "sd dlog c (2y)": float(dlc.std()),
            "sd dlog c / sd dlog y": float(dlc.std() / np.diff(np.log(y[:, w]), axis=1).std()),
            "share at a cap": float(((p["cash_on_hand"][:, a] >= p["_xtop"])
                                     | (p["illiquid_assets"][:, a] >= p["_ztop"])).mean())}


#: Grid variants for ``refine``: Laibson et al.'s full grid, each axis at about
#: twice its resolution, and both caps doubled.
REFINE = {"full": {}, "Z x2": dict(zjump=1000.0, z_cells_per_step=20),
          "X x2": dict(xjump=500.0, x_cells_per_step=100),
          "caps x2": dict(xmax=8e5, zmax=7e6)}


def cmd_refine(args) -> None:
    """Grid adequacy for what the network reads (H4, H6)."""
    from hh_npe.simulator.twoasset_gpu import solve_batch

    res = []
    for card, rg, th in PROBES:
        row = {"card": card, "R_gamma": rg, "theta": list(th)}
        for name, kw in REFINE.items():
            t0 = time.time()
            sol = solve_batch(np.array([th]), spec_for(card, rg, **kw), theta_batch=1,
                              chunk=8)[0]
            p = households(sol, args.n, 13)
            p["_xtop"], p["_ztop"] = sol.X[-1], sol.Z[-1]
            row[name] = window_stats(p)
            row[name]["seconds"] = round(time.time() - t0, 1)
            print(card, rg, th, name, {k: round(v, 4) for k, v in row[name].items()}, flush=True)
        res.append(row)
    out = OUT / "refine.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


def cmd_plot_statics(args) -> None:
    """Figure 65: each household statistic against each parameter, on the
    uniform draws (marginals), by card type: median with the 10-90% band
    (shares: mean). The detector for RESULTS 40's kind of cliff."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A = np.load(OUT / "statics_households.npy").astype(np.float64)
    th, card, uni, S = A[:, :4], A[:, 4].astype(bool), A[:, 5].astype(bool), A[:, 6:]
    names = (r"$\beta$", r"$\delta$", r"$\rho$", r"$R_\gamma$")
    lo = np.array([0.30, 0.85, 0.5, 1.021])
    hi = np.array([1.00, 1.00, 5.0, 1.075])
    rows = ((0, "illiquid / mean income, 35-44", "median"),
            (1, "liquid / mean income, 35-44", "median"),
            (2, "share of years in debt, 35-44", "mean"),
            (4, "sd of 2-year dlog c, 30-43", "median"))
    nb = 20
    fig, ax = plt.subplots(len(rows), 4, figsize=(15, 12), sharey="row")
    for r, (s, label, how) in enumerate(rows):
        for j in range(4):
            edges = np.linspace(lo[j], hi[j], nb + 1)
            mid = (edges[1:] + edges[:-1]) / 2
            b = np.clip(np.digitize(th[:, j], edges) - 1, 0, nb - 1)
            for cm, colour, lab in ((card, "C0", "card"), (~card, "C3", "no card")):
                if s == 2 and lab == "no card":
                    continue
                sel = uni & cm
                v = [S[sel & (b == k), s] for k in range(nb)]
                c = np.array([np.mean(x) if how == "mean" else np.median(x) for x in v])
                ax[r, j].plot(mid, c, color=colour, lw=1.6, label=lab)
                if how == "median":
                    q = np.array([np.percentile(x, [10, 90]) for x in v])
                    ax[r, j].fill_between(mid, q[:, 0], q[:, 1], color=colour, alpha=0.15)
            if r == len(rows) - 1:
                ax[r, j].set_xlabel(names[j], fontsize=13)
        ax[r, 0].set_ylabel(label, fontsize=9)
    ax[0, 0].legend(fontsize=9)
    fig.suptitle("Simulation audit: household statistics against each parameter, uniform draws "
                 "(81,920 households;\nmedian and 10-90% band, debt share as a mean). "
                 "A solver artefact like RESULTS 40's would show as a cliff.", fontsize=11)
    fig.tight_layout()
    out = Path("figures/65_simulation_audit_statics.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


#: Full-grid three-way points: Laibson et al.'s theta (where L also runs at
#: the full grid) and one in PSID couples' A2n region.
FULLGRID = (("laibson", 1, 1.05, (0.5305, 0.9891, 1.9355)),
            ("a2n_region", 1, 1.05, (0.60, 0.98, 2.7)))


def _ref_full(job):
    import pickle

    name, card, rg, th, kw = job
    m = ref.Model(ref.Config(R_gamma=rg, card=bool(card), **kw))
    sol = ref.solve(*th, m)
    tag = "_".join(f"{k}={'octave' if k == 'income_levels' else kw[k]}"
                   for k in sorted(kw)) or "base"
    with open(OUT / f"fullgrid_R_{name}_{tag}.pkl", "wb") as f:
        pickle.dump({"ix": sol.ix, "iz": sol.iz, "c": sol.c}, f)
    return name, tag


def cmd_fullgrid_ref(args) -> None:
    """R on the full production grid (CPU, one process per solve): centred
    with the port's tie order, as production, and the same with income
    rounded in the expectation step (H7); with ``--matlab``, at Laibson et
    al.'s theta, MATLAB's own arithmetic (uncentred) under each tie order and
    centred with MATLAB's, to attribute L's differences."""
    from concurrent.futures import ProcessPoolExecutor

    if args.octave_levels:
        n, c, rg, th = FULLGRID[0]
        lv = tuple(np.loadtxt(OUT / "octave/explog_levels.txt"))
        jobs = [(n, c, rg, th, {"centred": False, "tie": "z_first", "income_levels": lv})]
    elif args.matlab:
        n, c, rg, th = FULLGRID[0]
        jobs = [(n, c, rg, th, kw) for kw in ({"centred": False, "tie": "z_first"},
                                               {"centred": False, "tie": "x_first"},
                                               {"centred": True, "tie": "z_first"})]
    else:
        jobs = [(n, c, rg, th, kw) for n, c, rg, th in FULLGRID
                for kw in ({"centred": True, "tie": "x_first"},
                           {"centred": True, "tie": "x_first", "round_income": True})]
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        for done in ex.map(_ref_full, jobs):
            print("solved", done, flush=True)


def cmd_fullgrid_compare(args) -> None:
    """Production (GPU, centred) against R on the full grid, on R's forward
    pass with identical shocks."""
    import pickle

    from hh_npe.simulator.twoasset_gpu import solve_batch

    res = {}
    u = ref.uniforms(args.n, 71, seed=5)
    for name, card, rg, th in FULLGRID:
        m = ref.Model(ref.Config(R_gamma=rg, card=bool(card), centred=True, tie="x_first"))
        sols = {}
        for tag, label in (("centred=True_tie=x_first", "R"),
                           ("centred=True_round_income=True_tie=x_first", "R round_income")):
            with open(OUT / f"fullgrid_R_{name}_{tag}.pkl", "rb") as f:
                d = pickle.load(f)
            sols[label] = ref.Solution(d["ix"], d["iz"], d["c"], [None] * m.T, [None] * m.T,
                                       [None] * m.T, m, th)
        g = solve_batch(np.array([th]), spec_for(card, rg), theta_batch=1, chunk=16)[0]
        sols["P (production)"] = reexpress(g, m, th)
        init = ref.initial_indices(m, args.n)
        pan = {k: ref.simulate(s, u, init=init) for k, s in sols.items()}
        row = {"theta": list(th), "R_gamma": rg, "card": card,
               "agreement": {f"{a} vs {b}": agree(sols[a], sols[b], m)
                             for a, b in (("R", "P (production)"), ("R", "R round_income"))},
               "panel_stats": {k: panel_stats(p, m) for k, p in pan.items()}}
        for a, b in (("R", "P (production)"), ("R", "R round_income")):
            dpath = (pan[a]["cash"] != pan[b]["cash"]) | (pan[a]["illiquid"] != pan[b]["illiquid"])
            row[f"paths differ, ages 25-59: {a} vs {b}"] = float(dpath[:, 5:40].any(1).mean())
            row[f"paths ever differ: {a} vs {b}"] = float(dpath.any(1).mean())
        res[name] = row
        print(name, json.dumps({k: v for k, v in row.items() if k != "panel_stats"}, indent=1))
        for k, st in row["panel_stats"].items():
            print(f"   {k:16s}", {a: round(v, 4) for a, v in st.items()})
    (OUT / "fullgrid.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT / 'fullgrid.json'}")


def cmd_pool_scale(args) -> None:
    """H2: the seed pool stores wealth over model mean income at the person's
    own age (20-24); the forward pass multiplies by age-20 mean income. The
    same solves, households started both ways with the same shocks."""
    from hh_npe.simulator import grids
    from hh_npe.simulator.dispatch import draw_initial_conditions
    from hh_npe.simulator.twoasset_gpu import solve_batch
    from scripts.generate_dataset import _load_pool

    pool = _load_pool(POOL)[0]
    ages = np.load(POOL)["age"].astype(float)
    f = grids.mean_income(ages) / grids.mean_income(np.array([20.0]))[0]
    scaled = pool.copy()
    scaled[:, :2] *= f[:, None]
    res = {"pool_age_factor": {"min": float(f.min()), "median": float(np.median(f)),
                               "max": float(f.max())}, "theta": []}
    for card, rg, th in PROBES[:4]:
        sol = solve_batch(np.array([th]), spec_for(card, rg), theta_batch=1, chunk=16)[0]
        out = {}
        for name, pl in (("as generated", pool), ("own-age scale", scaled)):
            init, s0 = draw_initial_conditions(pl, 99, 0, args.n, sol.states, sol.P,
                                               sol.spec.calib.ywork_sigmanu)
            out[name] = twoasset.simulate(sol, n_households=args.n, seed=99,
                                          initial_wealth=init, initial_state=s0)
        a, b = out["as generated"], out["own-age scale"]
        d = ((a["cash_on_hand"] != b["cash_on_hand"])
             | (a["illiquid_assets"] != b["illiquid_assets"]))
        row = {"card": card, "R_gamma": rg, "theta": list(th),
               "paths differ at age": {str(age): float(d[:, age - 20].mean())
                                       for age in (20, 25, 30, 35, 40, 50, 59)}}
        for name, pnl in out.items():
            pnl["_xtop"], pnl["_ztop"] = sol.X[-1], sol.Z[-1]
            row[name] = window_stats(pnl)
        res["theta"].append(row)
        print(json.dumps(row), flush=True)
    (OUT / "pool_scale.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT / 'pool_scale.json'}")


def h8_ratio_map(nb: int = 8):
    """The training set's weighted (beta, R_gamma) density over the one the
    importance weights assume, on an nb x nb grid of the A2 box (H8)."""
    from hh_npe.npe.prior import _parts_log_weight

    th, parts = [], []
    for name in ("main", "strip"):
        d, n = DATASETS[name]
        cfg = json.loads((d / "solver_config.json").read_text())
        th.append(np.concatenate([np.load(f)["theta"] for f in shards(d)]))
        parts.append({"name": cfg["proposal"], "n_train": n, "rgamma_range": cfg["rgamma_range"]})
    th = np.concatenate(th)
    w = np.exp(_parts_log_weight(parts, th))
    lo, hi = np.array([0.30, 1.021]), np.array([1.00, 1.075])
    u = (th[:, [0, 3]] - lo) / (hi - lo)
    b = np.minimum((u * nb).astype(int), nb - 1)
    m = np.bincount(b[:, 0] * nb + b[:, 1], weights=w / w.sum(), minlength=nb * nb)
    return m.reshape(nb, nb) * nb * nb, lo, hi


def cmd_h8_impact(args) -> None:
    """How far H8 could move PSID couples' A2n posteriors: their draws
    reweighted by the inverse of the (beta, R_gamma) ratio map. An upper
    bound -- the network learns a smoothed version of the pattern."""
    import torch

    from hh_npe.evaluation.weighted import sample_draws, summarize
    from hh_npe.simulator.laibson_calibration import EDUC_GROUPS
    from scripts.optionA import _weighting, load_members, spec_of, transform

    ratio, lo, hi = h8_ratio_map()
    members, proposal, cfg = load_members([f"outputs/optionA2n/s{s}" for s in range(5)])
    spec = spec_of(cfg)
    inv, lw = _weighting(proposal, spec)
    fb = spec.flow_box()
    d = torch.load("data/processed/psid_x_comphs_couples.pt", weights_only=False)
    keep = np.flatnonzero(d["educ"].numpy() == EDUC_GROUPS.index("comphs"))
    torch.manual_seed(0)
    xt, _ = transform(torch.from_numpy(d["x"].numpy()[keep]).float(), masked=spec.masked)
    r = sample_draws(members, xt, args.draws, fb.low, fb.high, invert=inv, log_weight=lw)
    nb = ratio.shape[0]
    base, corr = [], []
    for i in range(len(keep)):
        n = r["n_kept"][i]
        t, lwi = r["theta"][i, :n].astype(np.float64), r["log_w"][i, :n]
        uu = (t[:, [0, 3]] - lo) / (hi - lo)
        b = np.clip((uu * nb).astype(int), 0, nb - 1)
        base.append(summarize(t, lwi))
        corr.append(summarize(t, lwi - np.log(ratio[b[:, 0], b[:, 1]])))
    names = ("beta", "delta", "rho", "R_gamma")
    res = {"ratio_map": ratio.round(3).tolist(), "n_couples": len(keep)}
    for j, nm in enumerate(names):
        m0 = np.array([s["mean"][j] for s in base])
        m1 = np.array([s["mean"][j] for s in corr])
        sd = np.array([s["sd"][j] for s in base])
        res[nm] = {"median_of_means": [float(np.median(m0)), float(np.median(m1))],
                   "median |shift| / posterior sd": float(np.median(np.abs(m1 - m0) / sd)),
                   "p90 |shift| / posterior sd": float(np.percentile(np.abs(m1 - m0) / sd, 90))}
        print(nm, res[nm], flush=True)
    (OUT / "h8_impact.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT / 'h8_impact.json'}")


def _l_model(L: dict, **kw) -> ref.Model:
    """The reference model on the grid an Octave run used."""
    X0 = L["X"][0]
    pos = X0[X0 >= 0]
    return ref.Model(ref.Config(xjump=float(pos[1] - pos[0]), zjump=float(L["Z"][1] - L["Z"][0]),
                                R_gamma=float(L["returns"][1]), card=bool(L["xmin"].max() > 0),
                                **kw), ref.first_stage())


def cmd_attribute(args) -> None:
    """Where L's and P's households part, and why, on L's own shocks: the tie
    order alone, the port's finite treatment of u(0) and impossible choices
    alone, and the centring; how large the differences are in what the
    network reads; and how often MATLAB's own best payoff is -inf."""
    L = load_octave(args.octave)
    th = tuple(float(v) for v in L["prefs"])
    m = _l_model(L)
    fs = m.fs
    cfg = m.cfg
    base = dict(xjump=cfg.xjump, zjump=cfg.zjump, R_gamma=cfg.R_gamma, card=cfg.card)
    sols = {"L": as_solution(L, m),
            "R x_first": ref.solve(*th, ref.Model(ref.Config(**base, tie="x_first"), fs)),
            "R centred x_first": ref.solve(*th, ref.Model(
                ref.Config(**base, tie="x_first", centred=True), fs)),
            "P": port_solution(m, th, False), "P centred": port_solution(m, th, True)}
    u = {"u_state": L["randU"], "u_income": L["randY"]}
    init = ref.initial_indices(m, L["randU"].shape[0])
    pan = {k: ref.simulate(s, u, init=init) for k, s in sols.items()}
    t = np.arange(5, 40)                                   # ages 25-59
    ym = m.Ymean[t]

    def feat(p):
        return np.stack([np.log(p["consumption"][:, t] / ym), np.arcsinh(p["liquid"][:, t] / ym),
                         np.arcsinh(p["illiquid"][:, t] / ym)], -1)

    def part(a, b):
        d = (pan[a]["cash"] != pan[b]["cash"]) | (pan[a]["illiquid"] != pan[b]["illiquid"])
        first = np.where(d.any(1), d.argmax(1) + 20, -1)
        fa, fb = feat(pan[a]), feat(pan[b])
        sd = fa.reshape(-1, 3).std(0)
        hit = first > 0
        return {"ever": float(hit.mean()), "ages 25-59": float(d[:, 5:40].any(1).mean()),
                "first age median": float(np.median(first[hit])) if hit.any() else None,
                "mean |feature diff| / sd (c, liquid, illiquid), ages 25-59":
                    (np.abs(fa - fb).reshape(-1, 3).mean(0) / sd).round(5).tolist()}

    res = {"theta": th, "R_gamma": cfg.R_gamma, "card": cfg.card,
           "tie order (R x_first vs L)": part("R x_first", "L"),
           "u(0) and impossible as finite (P vs R x_first)": part("P", "R x_first"),
           "centring, reference": part("R centred x_first", "R x_first"),
           "production vs MATLAB (P centred vs L)": part("P centred", "L"),
           "production vs centred reference": part("P centred", "R centred x_first")}
    W = L["W"]
    vis = [W[k][pan["L"]["ix"][:, k], pan["L"]["iz"][:, k], pan["L"]["state"][:, k]]
           for k in range(m.T)]
    res["MATLAB best payoff -inf: share of states"] = float(
        sum(np.isneginf(w).sum() for w in W) / sum(w.size for w in W))
    res["MATLAB best payoff -inf: share of visited household-years"] = float(
        np.mean(np.isneginf(np.stack(vis, 1))))
    out = OUT / f"attribute_{Path(args.octave).stem}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def cmd_h8_design(args) -> None:
    """H8: are beta and R_gamma independent in the generated design, as the
    importance weights assume? A permutation test (R_gamma shuffled across
    blocks within each run) and the 8 x 8 ratio map by part of the run."""
    from hh_npe.npe.prior import SWITCHED_A2, _parts_log_weight

    th, parts, blk = [], [], []
    for i, name in enumerate(("main", "strip")):
        d, n = DATASETS[name]
        cfg = json.loads((d / "solver_config.json").read_text())
        t = np.concatenate([np.load(f)["theta"] for f in shards(d)])
        th.append(t)
        blk.append(i * 100_000 + np.arange(len(t)) // 16)
        parts.append({"name": cfg["proposal"], "n_train": n, "rgamma_range": cfg["rgamma_range"]})
    th, blk = np.concatenate(th), np.concatenate(blk)
    lo, hi = np.array([0.30, 0.85, 0.5, 1.021]), np.array([1.00, 1.00, 5.0, 1.075])

    def masses(tt):
        w = np.exp(_parts_log_weight(parts, tt))
        w /= w.sum()
        uu = (tt - lo) / (hi - lo)
        bj = np.minimum((uu[:, 0] * 4).astype(int), 3)
        bk = np.minimum((uu[:, 3] * 4).astype(int), 3)
        return np.bincount(bj * 4 + bk, weights=w, minlength=16)

    obs = np.abs(masses(th) - 1 / 16).max()
    ub, inv = np.unique(blk, return_inverse=True)
    rg_block = np.zeros(len(ub))
    rg_block[inv] = th[:, 3]
    run = ub // 100_000
    rng = np.random.default_rng(0)
    sims = []
    for _ in range(args.perms):
        perm = rg_block.copy()
        for r in (0, 1):
            idx = np.flatnonzero(run == r)
            perm[idx] = rg_block[rng.permutation(idx)]
        t2 = th.copy()
        t2[:, 3] = perm[inv]
        sims.append(np.abs(masses(t2) - 1 / 16).max())
    sims = np.array(sims)
    res = {"observed max |cell mass - 1/16| (4x4 beta x R_gamma)": float(obs),
           "permuted: p50, p99, max": np.percentile(sims, [50, 99, 100]).tolist(),
           "p_value": float((sims >= obs).mean()), "perms": args.perms}
    main_th = th[: 32768]
    for name, sl, w in (("pilot", slice(0, 4096), None),
                        ("switched", slice(4096, 32768), "second")):
        tt = main_th[sl]
        ww = np.ones(len(tt)) if w is None else 1.0 / SWITCHED_A2.second.density(tt)
        uu = (tt[:, [0, 3]] - [0.30, 1.025]) / [0.70, 0.05]
        b = np.minimum((uu * 8).astype(int), 7)
        m = np.bincount(b[:, 0] * 8 + b[:, 1], weights=ww / ww.sum(), minlength=64) * 64
        res[f"8x8 ratio map, {name}"] = m.reshape(8, 8).round(3).tolist()
        res[f"8x8 ratio range, {name}"] = [float(m.min()), float(m.max())]
    (OUT / "h8_design.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if "map" not in k}, indent=1))


def cmd_fullgrid_l(args) -> None:
    """Port fidelity at Laibson et al.'s theta on their full grid: their code
    under Octave (L), R (centred, the port's tie order) and production (P,
    GPU) -- policies, panels on L's own shocks, and their 16 moments, with
    their shipped MATLAB optimum (MSMout.optMoments) as a fourth reference."""
    import pickle

    from hh_npe.simulator.twoasset_gpu import solve_batch
    from scripts.validate_twoasset import msm_objective, simulated_moments, their_simulated_moments

    L = load_octave(args.octave)
    th = tuple(float(v) for v in L["prefs"])
    m = ref.Model(ref.Config(centred=True, tie="x_first"))
    assert all(np.array_equal(m.X[t], L["X"][t]) for t in range(m.T))
    with open(OUT / "fullgrid_R_laibson_centred=True_tie=x_first.pkl", "rb") as f:
        d = pickle.load(f)
    sols = {"L": as_solution(L, m),
            "R": ref.Solution(d["ix"], d["iz"], d["c"], [None] * m.T, [None] * m.T,
                              [None] * m.T, m, th),
            "P": reexpress(solve_batch(np.array([th]), spec_for(1, 1.05), theta_batch=1,
                                       chunk=16)[0], m, th)}
    # MATLAB's own arithmetic in R (fullgrid_ref --matlab), to attribute L's
    # differences to the tie order and the centring.
    variants = {"R uncentred z_first, Octave income levels":
                "centred=False_income_levels=octave_tie=z_first",
                "R uncentred z_first": "centred=False_tie=z_first",
                "R uncentred x_first": "centred=False_tie=x_first",
                "R centred z_first": "centred=True_tie=z_first"}
    for label, tag in variants.items():
        f = OUT / f"fullgrid_R_laibson_{tag}.pkl"
        if f.exists():
            with open(f, "rb") as fh:
                d = pickle.load(fh)
            sols[label] = ref.Solution(d["ix"], d["iz"], d["c"], [None] * m.T, [None] * m.T,
                                       [None] * m.T, m, th)
    u = {"u_state": L["randU"], "u_income": L["randY"]}
    init = ref.initial_indices(m, L["randU"].shape[0])
    pan = {k: ref.simulate(sv, u, init=init) for k, sv in sols.items()}
    pairs = [("L", "R"), ("L", "P"), ("R", "P")] + [
        (a, b) for a, b in (("L", "R uncentred z_first, Octave income levels"),
                            ("L", "R uncentred z_first"), ("L", "R uncentred x_first"),
                            ("L", "R centred z_first"),
                            ("R uncentred z_first", "R uncentred x_first"),
                            ("R uncentred x_first", "R"), ("R centred z_first", "R"))
        if a in sols and b in sols]
    res = {"theta": th, "octave_seconds": L["elapsed"],
           "agreement": {f"{a} vs {b}": agree(sols[a], sols[b], m) for a, b in pairs},
           "forward_on_L_policies_reproduces_L": panels_equal(pan["L"], L)}
    for a, b in pairs:
        dd = (pan[a]["cash"] != pan[b]["cash"]) | (pan[a]["illiquid"] != pan[b]["illiquid"])
        res[f"paths differ {a} vs {b}: ever, ages 25-59"] = [
            float(dd.any(1).mean()), float(dd[:, 5:40].any(1).mean())]
    mom = {"MATLAB shipped (their RNG)": their_simulated_moments(),
           "L, Octave own": L["simMoments"].ravel()}
    for k, pn in pan.items():
        mom[k] = simulated_moments({"liquid_assets": pn["liquid"],
                                    "illiquid_assets": pn["illiquid"], "income": pn["income"]})
    res["moments"] = {k: np.asarray(v).round(4).tolist() for k, v in mom.items()}
    res["msm_q"] = {k: msm_objective(np.asarray(v)) for k, v in mom.items()}
    (OUT / "fullgrid_l.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "moments"}, indent=1))
    for k, v in res["moments"].items():
        print(f"{k:28s}", " ".join(f"{x:7.3f}" for x in v))


def cmd_h7_levels(args) -> None:
    """H7: income levels exp(log(k * $1,000)) off the lattice, how often the
    expectation step's cash snap differs from the realised (rounded) one, and
    on which levels numpy and Octave land on different sides of the lattice."""
    from hh_npe.simulator import grids

    lat = np.arange(1, 441) * 1000.0
    y = np.exp(np.log(lat))
    side = np.sign(y - lat)
    X, _ = grids.liquid_grid(grids.ages())
    Xp = X[X >= 0]
    diff = tot = 0
    for yl, yf in zip(lat[:150], y[:150], strict=True):        # incomes up to $150k
        snap = twoasset._nearest_index
        diff += int((snap(X, Xp + yl) != snap(X, Xp + yf)).sum())
        tot += len(Xp)
    res = {"levels": len(lat),
           "numpy below / exact / above": [int((side == v).sum()) for v in (-1, 0, 1)],
           "cash queries (X >= 0, income <= $150k) snapping differently": diff / tot}
    f = OUT / "octave/explog_levels.txt"
    if f.exists():
        oct_side = np.sign(np.loadtxt(f)[:440] - lat)
        res["levels where numpy and Octave differ in side"] = int((oct_side != side).sum())
    (OUT / "h7_levels.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("three_way")
    t.add_argument("--octave", type=Path, required=True)
    t.add_argument("--out", type=Path, default=None)
    sub.add_parser("integrity")
    sub.add_parser("invariants")
    sub.add_parser("statics")
    sub.add_parser("timing")
    sub.add_parser("plot_statics")
    sub.add_parser("h7_levels")
    fr = sub.add_parser("fullgrid_ref")
    fr.add_argument("--matlab", action="store_true")
    fr.add_argument("--octave_levels", action="store_true",
                    help="MATLAB arithmetic with Octave's own exp(log(.)) income levels")
    sub.add_parser("pool_scale").add_argument("--n", type=int, default=4000)
    sub.add_parser("h8_impact").add_argument("--draws", type=int, default=8000)
    sub.add_parser("h8_design").add_argument("--perms", type=int, default=400)
    sub.add_parser("attribute").add_argument("--octave", type=Path, required=True)
    sub.add_parser("fullgrid_l").add_argument("--octave", type=Path,
                                              default=OUT / "octave/laibson_full.mat")
    sub.add_parser("fullgrid_compare").add_argument("--n", type=int, default=4000)
    sub.add_parser("reproduce")
    for name in ("margins", "parity", "refine"):
        sub.add_parser(name).add_argument("--n", type=int, default=2000,
                                          help="households simulated per theta")
    args = ap.parse_args()
    {"three_way": cmd_three_way, "integrity": cmd_integrity, "invariants": cmd_invariants,
     "statics": cmd_statics, "timing": cmd_timing, "reproduce": cmd_reproduce,
     "margins": cmd_margins, "parity": cmd_parity, "refine": cmd_refine,
     "plot_statics": cmd_plot_statics, "fullgrid_ref": cmd_fullgrid_ref,
     "fullgrid_compare": cmd_fullgrid_compare, "pool_scale": cmd_pool_scale,
     "h8_impact": cmd_h8_impact, "h8_design": cmd_h8_design,
     "attribute": cmd_attribute, "fullgrid_l": cmd_fullgrid_l,
     "h7_levels": cmd_h7_levels}[args.cmd](args)


if __name__ == "__main__":
    main()
