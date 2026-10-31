"""Are PSID's reversals in income and consumption reporting error or real?
The budget decides (RESULTS 59).

RESULTS 57.2 measured the iid ("reversing") part of each feature, but on one
series alone a reporting error and a real one-wave movement look the same.
They differ in what else moves. A real spending spike is paid for: wealth at
the next interview is lower, and stays lower. A real windfall is saved or
spent. A misreported amount moves nothing else.

**Model.** In units of the household's mean income, each wave's flow (income
y, consumption c: the calendar year before the interview) and the stock W at
the interview are

    x_t = persistent_t + T_x,t + e_x,t,

T real and iid across waves, e reporting error, iid and independent across
variables and of everything real. Real transitory saving T_S = T_y - T_c
passes into W one-for-one and stays there:

- V_k(x) = Var(Delta_k x) has intercept 2 (Var T_x + Var e_x);
- C_k(c, y) = Cov(Delta_k c, Delta_k y) has intercept 2 Cov(T_c, T_y), since
  errors are independent;
- C_k(x, W) has intercept Cov(T_x, T_S): only the change ending at the
  transitory wave moves W, the one starting there does not.

Persistent components add terms in k and k^2 (wealth integrates a
persistent saving change), never a constant. So the intercepts identify:

    Var T_c = a(c, y) / 2 - a(c, W),   Var T_y = a(y, W) + a(c, y) / 2,

and the error variances as the rest of a(x, x) / 2. The **real share** of
a feature's reversing variance is Var T_x / (a(x, x) / 2): 0 if every
reversal is error, 1 if every reversal is real. For saving itself,
2 a(S, W) / a(S, S).

**Where the assumptions fail, and which way:**
- **Wealth that takes in less than the saving:** a durable bought for cash is
  consumption but stays an asset, and the narrower stores miss flows into the
  others. This understates the real share.
- **Saving that reverses before the next interview but one** (a windfall
  saved, then spent in the unmeasured year) moves W at both ends. This
  overstates the real share. v2's own households, where every movement is
  real, measure the net effect: their real share should read 1.
- **Errors shared across questions** (one respondent misreporting everything
  in a wave) are not separated.

**Stores of saving, PSID** (2010 dollars at the interview; cash and the card
balance from the family file, the rest from
``psid_illiquid_decomposition.components``):
- ``net worth``: checking and savings, minus the card balance, plus every
  illiquid component. The tensor's own liquid column (gross definition) drops
  cash when there is card debt, so it is not used;
- ``financial net worth``: the same without home equity, vehicles and other
  real estate, whose revaluations dominate net worth's changes;
- ``net cash``: checking and savings minus the card balance.

The narrower the store, the less revaluation noise, and the more saving it
misses (a lower bound on the real share).

**Estimation**, as ``psid_iid_error``: k-wave changes for k = 1..6, each
calendar pair of waves centred on its median, pooled; covariances winsorised
at 1/99 per variable (the budget algebra needs covariances that add up);
intercepts by least squares on 1, k, k^2, weighted by the number of changes.
200 bootstrap resamples of households. ``unimputed`` drops consumption changes
touching an imputed wave (``psid_accuracy_codes``).

**v2's households**, noise-free, couples' region: W is the model's net worth,
liquid plus illiquid. Then with the training noise's form added to income and
consumption, to check that the added variance is read as error and not as
real.

    PYTHONPATH=.:src .venv/bin/python scripts/psid_budget_reversals.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE as FEATS

INC, CON = FEATS.index("income"), FEATS.index("consumption")
LIQ, ILL = FEATS.index("liquid_assets"), FEATS.index("illiquid_assets")
MAX_LAG = 6
#: Calendar pairs with fewer changes, and gaps with fewer, are left out.
MIN_GROUP, MIN_LAG = 20, 100
WINSOR = (1.0, 99.0)
FINANCIAL = ("DC pensions", "IRAs", "stocks", "other assets", "unsecured debt")
STORES = ("net worth", "financial net worth", "net cash")
PSID_SETS = {"couples": Path("data/processed/psid_x_comphs_couples.pt"),
             "all 889": Path("data/processed/psid_x_comphs_optionA.pt")}
V2 = (Path("data/processed/couples_v2_dataset_shards"),
      Path("data/processed/couples_v2_strip_dataset_shards"))
#: Noise added to v2's income and consumption for the check (the training's form).
CHECK_SIGMA = 0.15
#: The cross intercepts, which independent reporting errors leave alone.
CROSS = ("cy", "yW", "cW", "SW")
OUT = Path("outputs/psid_budget_reversals/psid_budget_reversals.json")
FIG = Path("figures/79_psid_budget_reversals.png")


def changes(v: np.ndarray, k: int, clean: np.ndarray | None = None) -> np.ndarray:
    """k-wave changes, each calendar pair centred on its median, pooled; NaN
    where an endpoint is not ``clean`` or the pair has too few changes."""
    d = v[:, k:] - v[:, :-k]
    ok = (np.ones_like(d, bool) if clean is None
          else clean[:, k:] & clean[:, :-k])
    out = np.full_like(d, np.nan)
    for s in range(d.shape[1]):
        if ok[:, s].sum() >= MIN_GROUP:
            out[ok[:, s], s] = d[ok[:, s], s] - np.median(d[ok[:, s], s])
    return out.ravel()


def _winsor(d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(d, WINSOR)
    return np.clip(d, lo, hi)


def cov(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_LAG:
        return float("nan"), int(ok.sum())
    x, y = _winsor(a[ok]), _winsor(b[ok])
    return float(np.mean((x - x.mean()) * (y - y.mean()))), int(ok.sum())


def intercept(C: np.ndarray, n: np.ndarray, quad: bool = True) -> float:
    """The constant of C_k = a + b k (+ d k^2), least squares weighted by n."""
    k = np.arange(1, len(C) + 1, dtype=float)
    use = np.isfinite(C) & (n >= MIN_LAG)
    if use.sum() < (4 if quad else 3):
        return float("nan")
    A = np.column_stack([np.ones(use.sum()), k[use]] + ([k[use] ** 2] if quad else []))
    w = np.sqrt(n[use].astype(float))
    return float(np.linalg.lstsq(A * w[:, None], C[use] * w, rcond=None)[0][0])


def lag_covs(series: dict, pairs, clean: dict | None = None) -> dict:
    """C_k, k = 1..MAX_LAG, and the number of changes, for each pair."""
    clean = clean or {}
    d = {name: [changes(v, k, clean.get(name)) for k in range(1, MAX_LAG + 1)]
         for name, v in series.items()}
    out = {}
    for a, b in pairs:
        r = [cov(d[a][k], d[b][k]) for k in range(MAX_LAG)]
        out[(a, b)] = (np.array([c for c, _ in r]), np.array([m for _, m in r]))
    return out


def decompose(y: np.ndarray, c: np.ndarray, w: np.ndarray, ybar: np.ndarray,
              clean_c: np.ndarray | None = None, quad: bool = True,
              with_curves: bool = False) -> dict:
    """The budget decomposition (module docstring) of one store ``w``; every
    series over the household's mean income ``ybar``."""
    s = {"y": y / ybar, "c": c / ybar, "W": w / ybar, "S": (y - c) / ybar}
    clean = {"c": clean_c, "S": clean_c} if clean_c is not None else None
    pairs = [("y", "y"), ("c", "c"), ("S", "S"), ("c", "y"), ("y", "W"), ("c", "W"),
             ("S", "W")]
    C = lag_covs(s, pairs, clean)
    a = {f"{p}{q}": intercept(*C[(p, q)], quad=quad) for p, q in pairs}
    var_tc = a["cy"] / 2 - a["cW"]
    var_ty = a["yW"] + a["cy"] / 2
    out = {"a": a,
           "consumption": {"reversing_var": a["cc"] / 2, "real_var": var_tc,
                           "error_var": a["cc"] / 2 - var_tc,
                           "real_share": var_tc / (a["cc"] / 2)},
           "income": {"reversing_var": a["yy"] / 2, "real_var": var_ty,
                      "error_var": a["yy"] / 2 - var_ty,
                      "real_share": var_ty / (a["yy"] / 2)},
           "saving": {"reversing_var": a["SS"] / 2, "real_var": a["SW"],
                      "real_share": 2 * a["SW"] / a["SS"]},
           "corr_cy": a["cy"] / np.sqrt(a["cc"] * a["yy"])}
    if with_curves:
        out["C_k"] = {f"{p},{q}": {"C": C[(p, q)][0].tolist(), "n": C[(p, q)][1].tolist()}
                      for p, q in pairs}
    return out


def bootstrap(y, c, w, ybar, clean_c, n_boot: int, seed: int = 0) -> dict:
    """90% intervals of the shares, the cross intercepts and corr_cy; the
    draws of (a_cy, a_SW) for the figure."""
    if n_boot == 0:
        return {}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        b = rng.integers(0, len(y), len(y))
        r = decompose(y[b], c[b], w[b], ybar[b], None if clean_c is None else clean_c[b])
        draws.append([r["income"]["real_share"], r["consumption"]["real_share"],
                      r["saving"]["real_share"], *(r["a"][k] for k in CROSS), r["corr_cy"]])
    draws = np.array(draws)
    q = lambda j: np.nanpercentile(draws[:, j], [5, 95]).tolist()  # noqa: E731
    return {"income_real_share_90": q(0), "consumption_real_share_90": q(1),
            "saving_real_share_90": q(2),
            "a_90": {k: q(3 + i) for i, k in enumerate(CROSS)}, "corr_cy_90": q(7),
            "cross_draws": draws[:, [3, 6]].tolist()}


def psid_stores(x_path: Path, psid_dir: Path) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    """x, the three stores (module docstring), the tensor's rows, and
    consumption without owners' rental value."""
    from scripts.build_psid_tensor import REF_YEAR, WAVES, col, deflator
    from scripts.impute_rental_value import (
        ANCHOR_WAVES,
        impute,
        owner_matrix,
        real_rental,
    )
    from scripts.psid_illiquid_decomposition import components

    d = torch.load(x_path, weights_only=False)
    if d.get("liquid_def") != "gross" or not d.get("with_pension", False):
        raise SystemExit(f"{x_path}: expected the gross liquid definition, with pensions")
    x = d["x"].numpy().astype(np.float64)
    rows = d["psid_row"].numpy()
    fam = pd.read_pickle(psid_dir / "tax.pkl")
    comp = components(fam, pd.read_pickle(psid_dir / "pension.pkl"), rows)
    total = sum(comp.values())
    if not np.allclose(np.maximum(total, 0.0), x[..., ILL], rtol=1e-5, atol=1.0):
        raise SystemExit("components do not reproduce the tensor's illiquid wealth")

    def amounts(key):
        return np.column_stack([col(fam, key, y).loc[rows].fillna(0).to_numpy() * deflator(y)
                                for y in WAVES])
    cash, card = amounts("chk") + amounts("cd"), amounts("cc")
    if not np.allclose(np.where(card > 0, -card, cash), x[..., LIQ], rtol=1e-5, atol=1.0):
        raise SystemExit("cash and card do not reproduce the tensor's liquid wealth")
    net_cash = cash - card
    stores = {"net worth": net_cash + total,
              "financial net worth": net_cash + sum(comp[k] for k in FINANCIAL),
              "net cash": net_cash}
    # Owners' rental value (observed from 2019, carried back before): the
    # tensor's consumption is the family file's expenditure plus it.
    # The check is exact except for owners with no anchor, whose fallback is a
    # median over the sample the rent was built on (not this one).
    rv = real_rental(pd.read_pickle(psid_dir / "house.pkl"), rows)
    own = owner_matrix(fam, rows)
    rent = impute(rv, own, d["rent_growth_per_year"])   # as built
    anchored = np.logical_or.reduce([(rv[:, WAVES.index(a)] > 0) & own[:, WAVES.index(a)]
                                     for a in ANCHOR_WAVES])[:, None]
    spent = np.column_stack([
        (sum((1 if k == "totexp" else -1) * col(fam, k, y).loc[rows].fillna(0)
             for k in ("totexp", "mort", "ptax", "hins", "vdown", "vloan"))).to_numpy()
        * deflator(REF_YEAR[y]) for y in WAVES])
    exact = np.isclose(spent + rent, x[..., CON], rtol=1e-5, atol=1.0)
    fallback = own & ~anchored & (x[..., CON] - spent > 0)
    if not (exact | fallback).all():
        raise SystemExit("expenditure plus rental value does not reproduce the tensor's "
                         "consumption")
    return x, stores, rows, spent


def psid(x_path: Path, psid_dir: Path, n_boot: int) -> dict:
    from scripts import psid_accuracy_codes as pac

    x, stores, rows, spent = psid_stores(x_path, psid_dir)
    imputed = pac.masks(pac.collect(psid_dir, rows), 0.0)["consumption"]
    y, c = x[..., INC], x[..., CON]
    ybar = y.mean(1, keepdims=True)
    res = {"n": len(x), "share_consumption_waves_imputed": float(imputed.mean()),
           "rent_share_of_consumption": float((c - spent).sum() / c.sum()), "stores": {}}
    for name, w in stores.items():
        r = decompose(y, c, w, ybar, with_curves=True) | {
            "boot": bootstrap(y, c, w, ybar, None, n_boot),
            "linear": decompose(y, c, w, ybar, quad=False)}
        r["unimputed"] = decompose(y, c, w, ybar, clean_c=~imputed) | {
            "boot": bootstrap(y, c, w, ybar, ~imputed, n_boot)}
        r["without rental value"] = decompose(y, spent, w, ybar) | {
            "boot": bootstrap(y, spent, w, ybar, None, n_boot)}
        res["stores"][name] = r
    return res


def model(shards, every: int, n_boot: int) -> dict:
    from scripts import optionA as oa
    from scripts.noise_calibration import cells, region

    theta, card, train, held, _p, cfg = oa.load_shards(list(shards))
    box = oa.box_of(cfg)
    th, x, pid = oa.windows_raw((train + held)[::every], theta, seed=oa.HELD_SEED)
    cd = np.asarray(card)[pid.numpy()]
    keep = cells(th.numpy(), cd, box, region("couples", box)) >= 0
    x = x[torch.from_numpy(keep)]
    noisy = oa.add_obs_noise(x, {"income": CHECK_SIGMA, "consumption": CHECK_SIGMA},
                             seed=oa.NOISE_SEED)
    out = {"n_windows": int(len(x)), "check_sigma": CHECK_SIGMA}
    for name, v in (("noise-free", x), (f"noise {CHECK_SIGMA}", noisy)):
        v = v.numpy().astype(np.float64)
        y, c = v[..., INC], v[..., CON]
        ybar = y.mean(1, keepdims=True)
        for store, w in (("net worth", v[..., LIQ] + v[..., ILL]), ("liquid", v[..., LIQ])):
            key = name if store == "net worth" else f"{name}, liquid"
            out[key] = decompose(y, c, w, ybar, with_curves=True) | {
                "boot": bootstrap(y, c, w, ybar, None, n_boot),
                "linear": decompose(y, c, w, ybar, quad=False)}
    return out


def report(res: dict) -> None:
    def line(label, r, b=None):
        b = b or {}
        iv = lambda k: (f" ({b[k][0]:+.2f} to {b[k][1]:+.2f})" if k in b else "")  # noqa: E731
        print(f"  {label:34s} income {r['income']['real_share']:+.2f}"
              f"{iv('income_real_share_90')}   consumption {r['consumption']['real_share']:+.2f}"
              f"{iv('consumption_real_share_90')}   saving {r['saving']['real_share']:+.2f}"
              f"{iv('saving_real_share_90')}")
    print("real share of each feature's reversing variance (0: all error, 1: all real), 90%")
    for grp, p in res["psid"].items():
        print(f"\nPSID {grp} ({p['n']}; {p['share_consumption_waves_imputed']:.2f} of "
              "consumption waves imputed)")
        for store, r in p["stores"].items():
            line(store, r, r["boot"])
            line("  linear in k", r["linear"])
            line("  consumption unimputed", r["unimputed"], r["unimputed"]["boot"])
            line("  without rental value", r["without rental value"],
                 r["without rental value"]["boot"])
            print("    reversing var: income {:.4f}  consumption {:.4f}  (units of mean income^2)"
                  .format(r["income"]["reversing_var"], r["consumption"]["reversing_var"]))
    print("\nthe cross intercepts (units of mean income^2; 90%), which independent "
          "reporting errors leave alone; W a cash store")
    rows = [(f"PSID {g}, net cash", p["stores"]["net cash"]) for g, p in res["psid"].items()]
    if "model_v2" in res:
        m = res["model_v2"]
        rows += [(f"v2, {n}, liquid", m[f"{n}, liquid"])
                 for n in ("noise-free", f"noise {m['check_sigma']}")]
    for label, r in rows:
        b = r["boot"]["a_90"]
        print(f"  {label:28s} " + "  ".join(
            f"{k} {r['a'][k]:+.4f} ({b[k][0]:+.4f} to {b[k][1]:+.4f})" for k in CROSS)
            + f"  corr_cy {r['corr_cy']:+.2f} ({r['boot']['corr_cy_90'][0]:+.2f} to "
              f"{r['boot']['corr_cy_90'][1]:+.2f})")
    if "model_v2" in res:
        m = res["model_v2"]
        print(f"\nv2's households, couples' region ({m['n_windows']} windows): W = net worth")
        names = ("noise-free", f"noise {m['check_sigma']}")
        for name in names + tuple(f"{n}, liquid" for n in names):
            line(name, m[name], m[name]["boot"])
            line("  linear in k", m[name]["linear"])
            print("    error var: income {:.4f}  consumption {:.4f};  real var: income {:.4f}  "
                  "consumption {:.4f}".format(
                      m[name]["income"]["error_var"], m[name]["consumption"]["error_var"],
                      m[name]["income"]["real_var"], m[name]["consumption"]["real_var"]))


def plot(res: dict, path: Path) -> None:
    """C_k of consumption with income and of saving with cash, PSID against
    v2's households; and the 68/95% bootstrap contours of the two intercepts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    m = res.get("model_v2", {})
    series = [(f"PSID {g}, net cash", p["stores"]["net cash"], c)
              for (g, p), c in zip(res["psid"].items(), ("C0", "C2"), strict=True)]
    if m:
        series += [("v2, noise-free, liquid", m["noise-free, liquid"], "C3"),
                   (f"v2, noise {m['check_sigma']}, liquid",
                    m[f"noise {m['check_sigma']}, liquid"], "C1")]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    k = np.arange(1, MAX_LAG + 1)
    for j, (pair, title) in enumerate((
            ("c,y", "Cov(Δc, Δy): consumption with income"),
            ("S,W", "Cov(ΔS, Δcash): saving left in cash"))):
        for label, r, color in series:
            ax[j].plot(k, r["C_k"][pair]["C"], marker="o", color=color, label=label)
        ax[j].axhline(0, color="k", lw=0.6)
        ax[j].set(xlabel="gap k (waves)", ylabel="covariance, units of mean income²",
                  title=title + "\n(intercept: the shared reversing part)")
        ax[j].legend(fontsize=7)
    a = ax[2]
    for label, r, color in series:
        d = np.array(r["boot"].get("cross_draws", []))
        if not len(d):
            continue
        d = d[np.isfinite(d).all(1)]
        kde = gaussian_kde(d.T)
        lo, hi = d.min(0) - 3 * d.std(0), d.max(0) + 3 * d.std(0)
        gx, gy = np.mgrid[lo[0]:hi[0]:200j, lo[1]:hi[1]:200j]
        z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        zs = np.sort(z.ravel())[::-1]
        cum = np.cumsum(zs) / zs.sum()
        levels = sorted(zs[np.searchsorted(cum, q)] for q in (0.95, 0.68))
        a.contour(gx, gy, z, levels=levels, colors=color, linestyles=["--", "-"])
        a.plot(r["a"]["cy"], r["a"]["SW"], "x", color=color)
        a.plot([], [], color=color, label=label)
    a.axhline(0, color="k", lw=0.5)
    a.axvline(0, color="k", lw=0.5)
    a.set(xlabel="a(c, y): consumption and income reversing together",
          ylabel="a(S, cash): saving reversals left in cash",
          title="intercepts, 68% (solid) / 95% (dashed), bootstrap")
    a.legend(fontsize=7, loc="lower right")
    fig.suptitle("A real reversal moves the budget's other entries; a reporting error does not "
                 "(RESULTS 59). Every reversal in v2 is real.")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psid_dir", type=Path, default=Path("PSID-data"))
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--model_every", type=int, default=4)
    ap.add_argument("--model_boot", type=int, default=20)
    ap.add_argument("--no_model", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fig", type=Path, default=FIG)
    args = ap.parse_args()
    res = {"psid": {g: psid(p, args.psid_dir, args.n_boot) for g, p in PSID_SETS.items()}}
    if not args.no_model:
        res["model_v2"] = model(V2, args.model_every, args.model_boot)
    report(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(res), indent=1))
    plot(res, args.fig)
    print(f"\nwrote {args.out} and {args.fig}")


if __name__ == "__main__":
    main()
