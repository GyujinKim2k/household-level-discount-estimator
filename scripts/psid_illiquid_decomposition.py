"""What PSID couples' illiquid wealth is made of, and what drives its build-up
(RESULTS 44.6).

RESULTS 44.5 found that a third of PSID couples go from below half a year's
income in illiquid wealth to above it within their 12 years ("up"), which the
model produces for 16-19% of couples whatever their estimates. Its untested
reading: in PSID illiquid wealth accrues near-automatically, through mortgage
principal and employer pension contributions, while every deposit in the model
is a choice. This splits the estimation input into its parts and asks which
of them carries the build-up.

**Components** (2010 dollars, exactly as ``build_psid_tensor.py`` sums them
for the couples tensor, ``--with_pension --illiquid_debts all``):

- ``home equity``: PSID's wealth with home equity minus without it;
- ``DC pensions``: Section P balances, current and previous employers, head
  and spouse;
- ``IRAs``, ``stocks`` (outside IRAs), ``vehicles``;
- ``other real estate`` and ``other assets`` (other assets plus farm and
  business, net of their debts);
- ``unsecured debt``: student, medical, legal, family and other loans, as a
  negative.

Their sum, floored at zero, must reproduce the tensor's illiquid column; the
script refuses otherwise.

**Statistics.**

- By age: each component's mean over the mean of window mean income (so the
  components add up to the mean ratio, before the floor), and the share of
  household-waves holding it.
- By kind of couple (RESULTS 44.5's holders, never, up, down, other): the mean
  change from the first wave to the last of each component over the window's
  mean income. These add up to the change in the unfloored ratio.
- Up-movers by housing tenure in the first and last wave (renter to owner is
  a purchase).
- The kinds of window, and the share below 0.1 x income by age, with home
  equity left out, with home equity only, and with vehicles left out.
  Laibson et al.'s illiquid wealth is total wealth less liquid wealth, so it
  includes vehicles net of their loans (``2_buildmoments.do``: ``nfin_nw``),
  as PSID's does here.
- **Price against paydown.** PSID has no house value or mortgage balance in
  this extract, only home equity and the annual mortgage payment. Two bounds
  instead, over each two-year gap of couples who own in both waves:
  - owners with no mortgage payment in either wave: their equity is the house
    value, so its real growth is a house-price index for these couples;
  - owners with a mortgage: the equity gain against twice the annual payment,
    which bounds the principal repaid over two years from above (payments
    include interest, taxes are separate).

Usage::

    PYTHONPATH=. .venv/bin/python scripts/psid_illiquid_decomposition.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from scripts.build_psid_tensor import (
    ILLIQUID_DEBTS,
    PENSION,
    PENSION_DK,
    REF_YEAR,
    WAVES,
    col,
    deflator,
)
from scripts.illiquid_by_age import AGE_EDGES, KINDS, kinds
from scripts.illiquid_shock_calibration import HOLD, INC, ratio
from scripts.impute_rental_value import owner_matrix
from scripts.noise_calibration import PSID_X

OUT = Path("outputs/optionA3/psid_illiquid_decomposition.json")
FIG = Path("figures/51_psid_illiquid_decomposition.png")
ILL = FEATURES_TWOASSET_AGE.index("illiquid_assets")
AGE = FEATURES_TWOASSET_AGE.index("age")
COMPONENTS = ("home equity", "DC pensions", "IRAs", "stocks", "vehicles",
              "other real estate", "other assets", "unsecured debt")
UNSECURED = tuple(k for k in ILLIQUID_DEBTS["all"] if k not in ILLIQUID_DEBTS["secured"])


def components(fam: pd.DataFrame, pen: pd.DataFrame, rows: np.ndarray) -> dict:
    """Each component, (N, waves), 2010 dollars at the interview (module
    docstring)."""
    out = {k: np.zeros((len(rows), len(WAVES))) for k in COMPONENTS}
    for w, y in enumerate(WAVES):
        def c(key, y=y):
            return col(fam, key, y).loc[rows].fillna(0).to_numpy()
        d = deflator(y)
        out["home equity"][:, w] = c("w2") - c("w1")
        out["IRAs"][:, w] = c("ira")
        out["stocks"][:, w] = c("stocks")
        out["vehicles"][:, w] = c("veh")
        out["other real estate"][:, w] = c("re_a") - c("re_b")
        out["other assets"][:, w] = c("oth") + c("fb_a") - c("fb_b")
        out["unsecured debt"][:, w] = -sum(c(k) for k in UNSECURED)
        p = [pen[k].to_numpy(dtype=float)[rows] for k in PENSION[y]]
        out["DC pensions"][:, w] = sum(np.where(np.isfinite(v) & (v < PENSION_DK), v, 0.0)
                                       for v in p)
        for k in COMPONENTS:
            out[k][:, w] *= d
    return out


def mortgage_payment(fam: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """Annual mortgage payment, (N, waves), 2010 dollars of the reference year
    (as ``build_psid_tensor`` deflates it out of consumption)."""
    return np.column_stack([col(fam, "mort", y).loc[rows].fillna(0).to_numpy()
                            * deflator(REF_YEAR[y]) for y in WAVES])


def by_age(x: np.ndarray, comp: dict, own: np.ndarray) -> list[dict]:
    age = x[..., AGE]
    ybar = np.broadcast_to(x[..., INC].mean(1, keepdims=True), age.shape)
    total = sum(comp.values())
    out = []
    for lo, hi in zip(AGE_EDGES[:-1], AGE_EDGES[1:], strict=True):
        m = (age >= lo) & (age < hi)
        if not m.any():
            continue
        den = ybar[m].mean()
        out.append({"ages": [lo, hi - 1], "n": int(m.sum()),
                    "mean_ratio_unfloored": float(total[m].mean() / den),
                    "median_ratio": float(np.median(ratio(x)[m])),
                    "owner": float(own[m].mean()),
                    "mean_over_income": {k: float(v[m].mean() / den) for k, v in comp.items()},
                    "share_holding": {k: float((v[m] < 0).mean() if k == "unsecured debt"
                                               else (v[m] > 0).mean())
                                      for k, v in comp.items()}})
    return out


def changes(x: np.ndarray, comp: dict, own: np.ndarray) -> dict:
    """Mean first-to-last-wave change of each component over the window's
    mean income, by kind of couple; and up-movers by tenure."""
    ybar = x[..., INC].mean(1)
    kd = kinds(x)
    d = {k: (v[:, -1] - v[:, 0]) / ybar for k, v in comp.items()}
    total = sum(d.values())
    out = {}
    for kind, m in kd.items():
        out[kind] = {"n": int(m.sum()), "mean_change_ratio": float(total[m].mean()),
                     "by_component": {k: float(v[m].mean()) for k, v in d.items()}}
    up = kd["up"]
    lead = np.argmax(np.column_stack([d[k] for k in COMPONENTS]), axis=1)
    out["up"]["largest_rise"] = {k: float((lead[up] == j).mean())
                                 for j, k in enumerate(COMPONENTS)}
    tenure = {"renter to owner": ~own[:, 0] & own[:, -1],
              "owner to owner": own[:, 0] & own[:, -1],
              "owner to renter": own[:, 0] & ~own[:, -1],
              "renter to renter": ~own[:, 0] & ~own[:, -1]}
    out["up_by_tenure"] = {
        t: {"share_of_up": float(m[up].mean()),
            "share_up_in_group": float(up[m].mean()) if m.any() else None,
            "mean_change_home_equity": float(d["home equity"][up & m].mean())
            if (up & m).any() else None,
            "mean_change_rest": float((total - d["home equity"])[up & m].mean())
            if (up & m).any() else None}
        for t, m in tenure.items()}
    out["all_by_tenure_share"] = {t: float(m.mean()) for t, m in tenure.items()}
    return out


def variant(ill: np.ndarray, x: np.ndarray) -> dict:
    """Kinds of window, and the share below 0.1 x income by age, with illiquid
    wealth replaced by ``ill`` (floored)."""
    xx = x.copy()
    xx[..., ILL] = np.maximum(ill, 0.0)
    r, age = ratio(xx), xx[..., AGE]
    low = {f"{lo}-{hi - 1}": float((r[(age >= lo) & (age < hi)] < 0.1).mean())
           for lo, hi in zip(AGE_EDGES[:-1], AGE_EDGES[1:], strict=True)
           if ((age >= lo) & (age < hi)).any()}
    return {"kinds": {k: float(v.mean()) for k, v in kinds(xx).items()},
            "share_below_0.1_by_age": low}


def price_vs_paydown(home: np.ndarray, own: np.ndarray, pay: np.ndarray) -> dict:
    """The two bounds of the module docstring, per two-year gap and pooled."""
    out = {"gaps": []}
    pooled = {"free": [], "ratio": [], "gain": [], "bound": []}
    for w in range(len(WAVES) - 1):
        both = own[:, w] & own[:, w + 1]
        free = (both & (pay[:, w] == 0) & (pay[:, w + 1] == 0)
                & (home[:, w] > 0) & (home[:, w + 1] > 0))
        g = home[free, w + 1] / home[free, w]
        mort = both & (pay[:, w] > 0) & (pay[:, w + 1] > 0)
        gain = home[mort, w + 1] - home[mort, w]
        bound = 2 * pay[mort, w + 1]
        out["gaps"].append({
            "waves": [WAVES[w], WAVES[w + 1]],
            "free_and_clear_n": int(free.sum()),
            "free_and_clear_real_growth_p50": float(np.median(g) - 1) if free.any() else None,
            "mortgaged_n": int(mort.sum()),
            "mortgaged_gain_p50": float(np.median(gain)) if mort.any() else None,
            "mortgaged_2x_payment_p50": float(np.median(bound)) if mort.any() else None,
            "mortgaged_share_gain_above_2x_payment": float((gain > bound).mean())
            if mort.any() else None})
        pooled["free"].append(np.log(g))
        pooled["gain"].append(gain)
        pooled["bound"].append(bound)
    lg = np.concatenate(pooled["free"])
    gain, bound = np.concatenate(pooled["gain"]), np.concatenate(pooled["bound"])
    out["pooled"] = {"free_and_clear_real_growth_per_year_p50":
                     float(np.exp(np.median(lg) / 2) - 1),
                     "mortgaged_mean_gain": float(gain.mean()),
                     "mortgaged_mean_2x_payment": float(bound.mean()),
                     "mortgaged_gain_p50": float(np.median(gain)),
                     "mortgaged_2x_payment_p50": float(np.median(bound)),
                     "mortgaged_share_gain_above_2x_payment": float((gain > bound).mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", type=Path, default=PSID_X)
    ap.add_argument("--psid_dir", type=Path, default=Path("PSID-data"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fig", type=Path, default=FIG)
    args = ap.parse_args()

    d = torch.load(args.x, weights_only=False)
    if d.get("illiquid_debts", "all") != "all" or not d.get("with_pension", True):
        raise SystemExit(f"{args.x}: expected illiquid net of all debt, with pensions")
    x = d["x"].numpy().astype(np.float64)
    rows = d["psid_row"].numpy()
    fam = pd.read_pickle(args.psid_dir / "tax.pkl")
    pen = pd.read_pickle(args.psid_dir / "pension.pkl")
    comp = components(fam, pen, rows)
    total = sum(comp.values())
    rebuilt = np.maximum(total, 0.0).astype(np.float32)
    if not np.allclose(rebuilt, x[..., ILL].astype(np.float32), rtol=1e-6, atol=0.5):
        bad = ~np.isclose(rebuilt, x[..., ILL].astype(np.float32), rtol=1e-6, atol=0.5)
        raise SystemExit(f"components do not reproduce the tensor's illiquid wealth "
                         f"({int(bad.sum())} household-waves differ)")
    own = owner_matrix(fam, rows)
    pay = mortgage_payment(fam, rows)
    print(f"{len(x)} couples: components reproduce the tensor's illiquid wealth "
          f"({(total < 0).mean():.1%} of household-waves below zero before the floor)")

    res = {"x": str(args.x), "hold": HOLD, "components": list(COMPONENTS),
           "by_age": by_age(x, comp, own), "changes": changes(x, comp, own),
           "variants": {"as estimated": variant(total, x),
                        "without home equity": variant(total - comp["home equity"], x),
                        "home equity only": variant(comp["home equity"], x),
                        "without vehicles": variant(total - comp["vehicles"], x)},
           "price_vs_paydown": price_vs_paydown(comp["home equity"], own, pay)}
    report(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    plot(res, args.fig)
    print(f"\nwrote {args.out} and {args.fig}")


def report(res: dict) -> None:
    short = {"home equity": "home", "DC pensions": "DC", "IRAs": "IRA", "stocks": "stock",
             "vehicles": "veh", "other real estate": "o.RE", "other assets": "other",
             "unsecured debt": "debt"}
    print("\nmean component / mean window income, by age (adds up to the unfloored mean ratio)")
    print("  ages   owner " + "".join(f"{short[k]:>7s}" for k in COMPONENTS) + "   total  median r")
    for b in res["by_age"]:
        print(f"  {b['ages'][0]}-{b['ages'][1]}  {b['owner']:.2f} " + "".join(
            f"{b['mean_over_income'][k]:7.2f}" for k in COMPONENTS)
            + f"  {b['mean_ratio_unfloored']:6.2f}  {b['median_ratio']:6.2f}")
    print("\nshare of household-waves holding each (debt: owing)")
    for b in res["by_age"]:
        print(f"  {b['ages'][0]}-{b['ages'][1]}        " + "".join(
            f"{b['share_holding'][k]:7.2f}" for k in COMPONENTS))
    print("\nmean change, first wave to last, over window mean income, by kind of couple")
    print("  kind       n " + "".join(f"{short[k]:>7s}" for k in COMPONENTS) + "   total")
    for kind in KINDS:
        c = res["changes"][kind]
        print(f"  {kind:7s} {c['n']:4d} " + "".join(f"{c['by_component'][k]:7.2f}"
                                                  for k in COMPONENTS)
              + f"  {c['mean_change_ratio']:6.2f}")
    print("  up-movers' largest rise: " + "  ".join(
        f"{short[k]} {v:.2f}" for k, v in res["changes"]["up"]["largest_rise"].items()))
    print("\nup-movers by tenure (first wave -> last):")
    for t, v in res["changes"]["up_by_tenure"].items():
        print(f"  {t:17s} share of up {v['share_of_up']:.2f}; of all couples "
              f"{res['changes']['all_by_tenure_share'][t]:.2f}, of whom up "
              f"{v['share_up_in_group'] or 0:.2f}; mean change home "
              f"{v['mean_change_home_equity'] or 0:.2f}, rest {v['mean_change_rest'] or 0:.2f}")
    print("\nwith parts left out: kinds of window; share below 0.1 x income by age")
    for k, v in res["variants"].items():
        print(f"  {k:20s} " + "  ".join(f"{a} {b:.2f}" for a, b in v["kinds"].items())
              + "  |  " + " ".join(f"{b:.2f}" for b in v["share_below_0.1_by_age"].values()))
    pv = res["price_vs_paydown"]
    print("\ncontinuing owners, each two-year gap:")
    for g in pv["gaps"]:
        print(f"  {g['waves'][0]}-{g['waves'][1]}: free-and-clear n {g['free_and_clear_n']:3d} "
              f"real equity growth p50 {g['free_and_clear_real_growth_p50']:+.3f}; mortgaged n "
              f"{g['mortgaged_n']:3d} equity gain p50 {g['mortgaged_gain_p50']:9,.0f} against "
              f"2x payment p50 {g['mortgaged_2x_payment_p50']:9,.0f}; gain above it "
              f"{g['mortgaged_share_gain_above_2x_payment']:.2f}")
    print("  pooled: " + "; ".join(f"{k} {v:,.3f}" for k, v in pv["pooled"].items()))


def plot(res: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = dict(zip(COMPONENTS, ("tab:brown", "tab:blue", "tab:cyan", "tab:green",
                                   "tab:gray", "tab:olive", "tab:pink", "tab:red"), strict=True))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    mid = np.array([np.mean(b["ages"]) for b in res["by_age"]])
    pos, neg = np.zeros(len(mid)), np.zeros(len(mid))
    for k in COMPONENTS:
        v = np.array([b["mean_over_income"][k] for b in res["by_age"]])
        base = np.where(v >= 0, pos, neg)
        ax[0].bar(mid, v, 3.6, bottom=base, color=colors[k], label=k)
        pos, neg = pos + np.maximum(v, 0), neg + np.minimum(v, 0)
    ax[0].plot(mid, [b["median_ratio"] for b in res["by_age"]], "k.-", label="median ratio")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set(title="PSID couples: mean component / mean income, by age", xlabel="age")
    ax[0].legend(fontsize=7, ncol=2)

    kinds_ = ("holders", "up", "other", "down")
    w = 0.8 / len(COMPONENTS)
    for j, k in enumerate(COMPONENTS):
        v = [res["changes"][kd]["by_component"][k] for kd in kinds_]
        ax[1].bar(np.arange(len(kinds_)) + (j - (len(COMPONENTS) - 1) / 2) * w, v, w,
                  color=colors[k])
    ax[1].set_xticks(range(len(kinds_)),
                     [f"{kd} ({res['changes'][kd]['n']})" for kd in kinds_])
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(title="mean change over 12 years / mean income, by kind of couple")

    g = res["price_vs_paydown"]["gaps"]
    lab = [f"{a['waves'][0] % 100:02d}-{a['waves'][1] % 100:02d}" for a in g]
    ax[2].plot(lab, [a["mortgaged_gain_p50"] for a in g], "o-", color="tab:brown",
               label="mortgaged owners: equity gain, p50")
    ax[2].plot(lab, [a["mortgaged_2x_payment_p50"] for a in g], "s--", color="0.4",
               label="2 x annual payment, p50 (bounds principal)")
    ax[2].set(title="continuing owners, each two-year gap (2010 $)")
    t = ax[2].twinx()
    t.plot(lab, [a["free_and_clear_real_growth_p50"] for a in g], "^:", color="tab:green",
           label="free-and-clear: real growth, p50 (right)")
    t.axhline(0, color="tab:green", lw=0.5)
    h1, l1 = ax[2].get_legend_handles_labels()
    h2, l2 = t.get_legend_handles_labels()
    ax[2].legend(h1 + h2, l1 + l2, fontsize=7)
    fig.suptitle("What PSID couples' illiquid wealth is made of (RESULTS 44.6)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
