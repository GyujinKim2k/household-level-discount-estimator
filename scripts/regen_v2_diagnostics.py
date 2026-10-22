"""RESULTS 56: what changed between the v1 data (A2n) and the v2 data, and why
v2's held-out log q is lower.

Three subcommands, each writing one key of ``--out``:

- ``data``: the stored panels, v1 against v2, draw by draw. The two runs share
  every draw's (beta, delta, rho), card, shocks and starting state, so the
  differences are the solver's fixes and the R_gamma permutation. The
  statistics are ``audit_simulation.panel_stats``': ages 35-44, normalised by
  the run's mean income at each age (identical in v1 and v2), and sd of the
  biennial change in log consumption over ages 31-43.
- ``design``: how well a held-out draw's R_gamma is predicted by its
  (beta, delta, rho) from the training draws alone, before any simulation.
  R_gamma is drawn once per block of 16 draws; v1 drew it from the block index,
  v2 permutes it across blocks (RESULTS 54.6, H8). Reported as held-out R^2 of
  the mean R_gamma of the k nearest training draws, and of beta-quantile-bin
  means at several bin counts.
- ``heldout``: each ensemble's per-row held-out log q on its own held-out set
  (``check_optionA2_heldout``'s rows, unweighted), split into a sum of
  one-parameter Gaussian marginals of the network's own q in flow space and
  the joint remainder; the Gaussian (linear) part of the remainder per pair.

Usage::

    PYTHONPATH=.:src .venv/bin/python scripts/regen_v2_diagnostics.py data
    PYTHONPATH=.:src .venv/bin/python scripts/regen_v2_diagnostics.py design
    PYTHONPATH=.:src .venv/bin/python scripts/regen_v2_diagnostics.py heldout
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

D = Path("data/processed")
PAIRS = {"main": ("couples_dataset_shards", "couples_v2_dataset_shards"),
         "strip": ("couples_strip_dataset_shards", "couples_v2_strip_dataset_shards"),
         "sbc": ("optionA2_sbc16_shards", "optionA2v2_sbc16_shards")}
RUNS = {"A2n": ("outputs/optionA2n", [D / PAIRS["main"][0], D / PAIRS["strip"][0]]),
        "v2": ("outputs/optionA2v2", [D / PAIRS["main"][1], D / PAIRS["strip"][1]])}
AGES = slice(15, 25)                        # 35-44; t = 0 is age 20
WAVES = np.arange(11, 25, 2)                # 31, 33, ..., 43
NAMES = ("beta", "delta", "crra", "R_gamma")


def shards(d: Path) -> list[Path]:
    return sorted(d.glob("shard_*.npz"))


def theta(d: Path) -> np.ndarray:
    return np.concatenate([np.load(f)["theta"] for f in shards(d)])


# --------------------------------------------------------------------- data
def panel_sums(files: list[Path], my: np.ndarray | None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Household-year sums for the statistics, and per-draw debt shares.
    ``my``: mean income by age (computed from the panels when None)."""
    if my is None:
        tot, n = 0.0, 0
        for f in files:
            z = np.load(f)
            tot, n = tot + z["panel_income"].sum(0), n + len(z["panel_income"])
        my = tot / n
    s = {"illiquid": [], "liquid": [], "c/y": [], "dlc": []}
    per_draw = []
    for f in files:
        z = np.load(f)
        liq = z["panel_liquid_assets"][:, AGES] / my[AGES]
        s["liquid"].append(liq.ravel())
        s["illiquid"].append((z["panel_illiquid_assets"][:, AGES] / my[AGES]).ravel())
        s["c/y"].append((z["panel_consumption"][:, AGES] / z["panel_income"][:, AGES]).ravel())
        s["dlc"].append(np.diff(np.log(z["panel_consumption"][:, WAVES]), axis=1).ravel())
        m = int(z["n_households"])
        per_draw.append((liq < 0).reshape(-1, m * liq.shape[1]).mean(1))
    return {k: np.concatenate(v) for k, v in s.items()}, np.concatenate(per_draw), my


def stats(s: dict) -> dict:
    liq = s["liquid"]
    return {"illiquid/mean income, median": float(np.median(s["illiquid"])),
            "illiquid/mean income, mean": float(s["illiquid"].mean()),
            "share in debt": float((liq < 0).mean()),
            "share in debt > 2% of mean income": float((liq < -0.02).mean()),
            "mean debt / mean income, all": float(np.where(liq < 0, -liq, 0).mean()),
            "mean debt / mean income, debtors": float(-liq[liq < 0].mean()),
            "consumption/income, median": float(np.median(s["c/y"])),
            "sd dlog c (2y)": float(s["dlc"].std())}


def cmd_data(args) -> dict:
    out = {}
    for name, (a, b) in PAIRS.items():
        fa, fb = shards(D / a), shards(D / b)
        ta, tb = theta(D / a), theta(D / b)
        if not np.array_equal(ta[:, :3], tb[:, :3]):
            raise SystemExit(f"{name}: v1 and v2 draws differ in (beta, delta, rho)")
        sa, da, my = panel_sums(fa, None)
        sb, db, _ = panel_sums(fb, my)
        d = db - da
        rho = ta[:, 2]
        out[name] = {"draws": len(ta), "v1": stats(sa), "v2": stats(sb),
                     "per-draw debt share shift": {
                         "mean": float(d.mean()), "median": float(np.median(d)),
                         "p10": float(np.percentile(d, 10)), "p90": float(np.percentile(d, 90)),
                         "by rho": {f"[{lo}, {hi})": float(d[(rho >= lo) & (rho < hi)].mean())
                                    for lo, hi in ((0.5, 1), (1, 2), (2, 3), (3, 4), (4, 5.01))}}}
        print(f"\n=== {name}: {len(ta)} draws, ages 35-44 ===")
        for k in out[name]["v1"]:
            print(f"  {k:36s} v1 {out[name]['v1'][k]:8.4f}   v2 {out[name]['v2'][k]:8.4f}")
        print("  per-draw debt share, v2 - v1: " + json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else {a: round(b, 4) for a, b in v.items()})
             for k, v in out[name]["per-draw debt share shift"].items()}))
    return out


# ------------------------------------------------------------------- design
def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - ((y - pred) ** 2).mean() / y.var())


def cmd_design(args) -> dict:
    from scipy.spatial import cKDTree
    out = {}
    for name, dirs in PAIRS.items():
        for tag, d in zip(("v1", "v2"), dirs):
            t = theta(D / d)
            # Held out as in training: the main run's last 1,024 draws, else the last quarter.
            n_ho = 1024 if len(t) >= 32768 else len(t) // 4
            tr, ho = slice(0, len(t) - n_ho), slice(len(t) - n_ho, len(t))
            f = np.column_stack([t[:, 0], np.log(1 - t[:, 1]), t[:, 2]])
            f = (f - f.mean(0)) / f.std(0)
            y = t[:, 3]
            res = {"held-out draws": n_ho, "knn": {}, "beta bins": {}}
            for cols in ((0,), (1,), (2,), (0, 1), (0, 1, 2)):
                key = "".join(("beta", "delta", "rho")[c][0] for c in cols)
                for k in (10, 30, 100):
                    _, ix = cKDTree(f[tr][:, cols]).query(f[ho][:, cols], k=k)
                    res["knn"][f"{key} k{k}"] = r2(y[ho], y[tr][ix].mean(1))
            for nb in (5, 20, 100, 500):
                if nb > (len(t) - n_ho) // 8:
                    continue
                e = np.quantile(t[tr, 0], np.linspace(0, 1, nb + 1))
                e[0], e[-1] = -np.inf, np.inf
                bt, bh = np.digitize(t[tr, 0], e) - 1, np.digitize(t[ho, 0], e) - 1
                m = np.array([y[tr][bt == i].mean() for i in range(nb)])
                res["beta bins"][str(nb)] = r2(y[ho], m[bh])
            out[f"{name} {tag}"] = res
            print(f"{name:5s} {tag}: held-out {n_ho:5d}  " + "  ".join(
                f"{k} {v:+.3f}" for k, v in res["knn"].items() if k.endswith("k10"))
                  + "  | beta bins " + "  ".join(f"{k} {v:+.3f}" for k, v in res["beta bins"].items()))
    return out


# ------------------------------------------------------------------ heldout
def gauss(t, m, S) -> float:
    L = np.linalg.cholesky(S)
    y = np.linalg.solve(L, t - m)
    return float(-0.5 * y @ y - np.log(np.diag(L)).sum() - 0.5 * len(t) * np.log(2 * np.pi))


def cmd_heldout(args) -> dict:
    import torch

    from hh_npe.evaluation.weighted import sample_draws
    from scripts import optionA as oa

    out = {}
    for name, (d, dirs) in RUNS.items():
        members, _prop, cfg = oa.load_members([Path(f"{d}/s{i}") for i in range(5)], args.device)
        spec = oa.spec_of(cfg)
        fb = spec.flow_box()
        theta_all, _card, _tr, held_f, _p, _sh = oa.load_shards(dirs)
        oa.check_flow_lag(cfg, held_f)
        th, x, pid, _f = oa.windows(held_f, theta_all, seed=oa.HELD_SEED, noise=spec.obs_noise,
                                    mask=spec.mask_patterns())
        pick = oa.one_per_draw(pid, seed=0)
        th, x, pid = th[pick], x[pick], pid[pick].numpy()
        main = pid < 32768                           # the strip's draws follow the main run's
        tf = spec.to_flow(th).numpy().astype(float)
        joint = oa._log_q(members, th, x, spec, per_row=True)
        jac = spec.log_jacobian(th.numpy())
        torch.manual_seed(0)
        dr = sample_draws(members, x, args.n_draws, fb.low, fb.high)
        marg = np.full((len(tf), 4), np.nan)
        pair = {p: np.full(len(tf), np.nan) for p in combinations(range(4), 2)}
        for i in range(len(tf)):
            s = dr["theta"][i, :dr["n_kept"][i]].astype(float)
            m, S = s.mean(0), np.cov(s.T)
            marg[i] = [gauss(tf[i, [j]], m[[j]], S[[j]][:, [j]]) for j in range(4)]
            for a, b in pair:
                ix = [a, b]
                pair[a, b][i] = gauss(tf[i, ix], m[ix], S[np.ix_(ix, ix)]) - marg[i, a] - marg[i, b]
        res = {}
        for part, k in (("main", main), ("strip", ~main)):
            flow = joint[k] - jac[k]
            res[part] = {"n": int(k.sum()), "log q (own scales)": float(joint[k].mean()),
                         "log q (flow space)": float(flow.mean()),
                         "Gaussian marginals": {n: float(marg[k, j].mean())
                                                for j, n in enumerate(NAMES)},
                         "sum of marginals": float(marg[k].sum(1).mean()),
                         "joint remainder": float(flow.mean() - marg[k].sum(1).mean()),
                         "Gaussian pair terms": {f"{NAMES[a]}-{NAMES[b]}": float(v[k].mean())
                                                 for (a, b), v in pair.items()}}
            r = res[part]
            print(f"{name:4s} {part:5s} n={r['n']:4d}  log q {r['log q (own scales)']:.3f}  "
                  f"flow space {r['log q (flow space)']:.3f} = marginals {r['sum of marginals']:.3f}"
                  f" + joint {r['joint remainder']:.3f}  (Gaussian pairs "
                  f"{sum(r['Gaussian pair terms'].values()):.3f})")
            print("      marginals " + "  ".join(f"{n} {v:.3f}" for n, v in r["Gaussian marginals"].items()))
        res["per row"] = {"pid": pid.tolist(), "log q": joint.tolist()}
        out[name] = res
    a, b = out["A2n"]["per row"], out["v2"]["per row"]
    if a["pid"] != b["pid"]:
        raise SystemExit("the two held-out sets are not the same draws")
    dq = np.array(b["log q"]) - np.array(a["log q"])
    out["paired v2 - A2n"] = {"mean": float(dq.mean()), "median": float(np.median(dq)),
                              "main": float(dq[np.array(a["pid"]) < 32768].mean()),
                              "strip": float(dq[np.array(a["pid"]) >= 32768].mean())}
    print("paired v2 - A2n per row: " + json.dumps({k: round(v, 3) for k, v in out["paired v2 - A2n"].items()}))
    for n in RUNS:
        del out[n]["per row"]
    return out


def cmd_ripple(args) -> dict:
    """Mean held-out log q at the truth minus the mean of log q at the truth
    moved by +-eps in one parameter, the others fixed. A smooth posterior of
    sd s loses about eps^2 / (2 s^2); a network that learned the design's
    fine structure loses more at small eps."""
    from scripts import optionA as oa

    steps = {"beta": (0.005, 0.02, 0.05), "R_gamma": (0.0005, 0.002, 0.005)}
    out = {}
    for name, (d, dirs) in RUNS.items():
        members, _prop, cfg = oa.load_members([Path(f"{d}/s{i}") for i in range(5)], args.device)
        spec = oa.spec_of(cfg)
        theta_all, _card, _tr, held_f, _p, _sh = oa.load_shards(dirs)
        th, x, pid, _f = oa.windows(held_f, theta_all, seed=oa.HELD_SEED, noise=spec.obs_noise,
                                    mask=spec.mask_patterns())
        pick = oa.one_per_draw(pid, seed=0)
        th, x, pid = th[pick], x[pick], pid[pick].numpy()
        base = oa._log_q(members, th, x, spec, per_row=True)
        lo, hi = np.array(spec.box.low), np.array(spec.box.high)
        res = {}
        for p, eps_all in steps.items():
            j = NAMES.index(p)
            for eps in eps_all:
                ok = (th[:, j].numpy() - eps > lo[j]) & (th[:, j].numpy() + eps < hi[j])
                drops = []
                for sgn in (1, -1):
                    t2 = th.clone()
                    t2[:, j] += sgn * eps
                    drops.append(base - oa._log_q(members, t2, x, spec, per_row=True))
                dr = 0.5 * (drops[0] + drops[1])
                for part, k in (("main", pid < 32768), ("strip", pid >= 32768)):
                    k = k & ok
                    res[f"{p} {eps} {part}"] = {"n": int(k.sum()), "drop": float(dr[k].mean()),
                                                "se": float(dr[k].std() / np.sqrt(k.sum()))}
        out[name] = res
    for k in out["A2n"]:
        a, b = out["A2n"][k], out["v2"][k]
        print(f"{k:22s} n={a['n']:4d}  A2n {a['drop']:+.4f} ({a['se']:.4f})   "
              f"v2 {b['drop']:+.4f} ({b['se']:.4f})")
    return out


def cmd_floor(args) -> dict:
    """``check_optionA2_heldout``'s coverage by posterior-mean R_gamma, its two
    lowest bins split into the main run's draws and the strip's."""
    import torch

    from hh_npe.evaluation.weighted import sample_weighted
    from scripts import optionA as oa
    from scripts.check_optionA2_heldout import heldout_log_weight

    out = {}
    for name, (d, dirs) in RUNS.items():
        members, proposal, cfg = oa.load_members([Path(f"{d}/s{i}") for i in range(5)], args.device)
        spec = oa.spec_of(cfg)
        inv, lw = oa._weighting(proposal, spec)
        fb = spec.flow_box()
        theta_all, _card, _tr, held_f, _p, sh_cfg = oa.load_shards(dirs)
        th, x, pid, _f = oa.windows(held_f, theta_all, seed=oa.HELD_SEED, noise=spec.obs_noise,
                                    mask=spec.mask_patterns())
        pick = oa.one_per_draw(pid, seed=0)
        th, x, pid = th[pick], x[pick], pid[pick].numpy()
        t = th.numpy().astype(float)
        w = np.exp(heldout_log_weight(sh_cfg, pid, t))
        torch.manual_seed(0)                         # as check_optionA2_heldout's ensemble
        r = sample_weighted(members, x, args.n_draws, fb.low, fb.high, invert=inv, log_weight=lw,
                            truth=t)
        m, u = r["mean"][:, 3], r["u"][:, 3]
        res = {}
        for lo, hi in ((1.021, 1.025), (1.025, 1.03)):
            for part, k in (("all", np.ones(len(t), bool)), ("main", pid < 32768),
                            ("strip", pid >= 32768)):
                k = k & (m >= lo) & (m < hi) & np.isfinite(u)
                if not k.any():
                    continue
                ww = w[k] / w[k].sum()
                res[f"[{lo}, {hi}) {part}"] = {
                    "n": int(k.sum()), "cov90": float((ww * ((u[k] >= .05) & (u[k] <= .95))).sum()),
                    "above95": float((ww * (u[k] > .95)).sum()),
                    "below5": float((ww * (u[k] < .05)).sum()),
                    "true R_gamma median": float(np.median(t[k, 3])),
                    "share of weight": float(w[k].sum() / w[(m >= lo) & (m < hi)].sum())}
        out[name] = res
        for k, v in res.items():
            print(f"{name:4s} {k:24s} n={v['n']:4d}  weight {v['share of weight']:.2f}  cov90 "
                  f"{v['cov90']:.3f}  >95% {v['above95']:.3f}  <5% {v['below5']:.3f}  "
                  f"true R median {v['true R_gamma median']:.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("data", "design", "heldout", "floor", "ripple"))
    ap.add_argument("--n_draws", type=int, default=4000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("outputs/optionA2v2/diagnostics.json"))
    args = ap.parse_args()
    res = json.loads(args.out.read_text()) if args.out.exists() else {}
    res[args.command] = {"data": cmd_data, "design": cmd_design, "heldout": cmd_heldout,
                         "floor": cmd_floor, "ripple": cmd_ripple}[args.command](args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
