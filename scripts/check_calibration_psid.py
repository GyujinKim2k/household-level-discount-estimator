"""Laibson et al.'s fixed calibration against PSID: the PSID side (RESULTS 42).

The simulator's fixed inputs come from four sources: IPUMS (household
composition), PSID 1982-91 (income process), SCF 1995-2013 (age-20 seed, credit
limit, alpha) and SSA (mortality). This puts each one we can check next to what
our PSID data say, for our 889 comphs households and for the couples subset
the headline now uses (legally married, A3 = 1, in at least 4 of 7 waves):

1. **Seed.** Wealth over model mean income for comphs heads aged 20-24 (all
   2011-2023 waves, not only our cohort), as SCF's `4_initialwealth.do` builds
   it: median of the ratio at ages <= 24.
2. **Credit limit.** PSID records balances, not limits. Debt among borrowers
   against the model's limit, and the share of borrowing waves beyond it --
   which the model cannot produce at all. Raw, and times alpha = 2.02.
3. **Composition.** Children and adults per family by age, against the IPUMS
   profile and the model's two adults.
4. **Income.** Median after-tax income by age against the model's exact median
   (the stationary Tauchen mixture times the discretised transitory shock --
   income does not depend on theta, so no solve is needed).
5. **Card.** Who is a known cardholder: any reported card debt in 7 waves.

SCF numbers, where shown, come from ``scripts/scf_first_stage.py``'s output.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/check_calibration_psid.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hh_npe.simulator import grids
from hh_npe.simulator import laibson_calibration as cal
from scripts import build_psid_tensor as b

TENSOR = Path("data/processed/psid_x_comphs_optionA.pt")
SCF_OUT = Path("outputs/scf_first_stage.json")
OUT = Path("outputs/calibration_check.json")
ALPHA = 2.02
COUPLE_MIN_WAVES = 4

MARITAL = b.MARITAL
N_FU = dict(zip(b.WAVES, ["ER47316", "ER53016", "ER60016", "ER66016",
                          "ER72016", "ER78016", "ER82017"]))
N_CHILD = dict(zip(b.WAVES, ["ER47320", "ER53020", "ER60021", "ER66021",
                             "ER72021", "ER78021", "ER82022"]))
BANDS = [(25, 34), (35, 44), (45, 54), (55, 70)]


def _q(v, w=None, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    return [float(x) for x in np.quantile(v, qs)]


def young_heads(fam, ind, edu, pen, lo=20, hi=24) -> pd.DataFrame:
    """Head-waves aged lo..hi in any wave, under Laibson et al.'s filter."""
    key = ["ER30001", "ER30002"]
    mm = ind[key].join(edu.drop_duplicates(subset=key).set_index(key), on=key)
    rows = []
    for y in b.WAVES:
        age = ind[b.AGE[y]]
        head = (ind[b.IV[y]] > 0) & (ind[b.REL[y]] == b.HEAD_CODE)
        ed = mm[b.ED_HD[y]].where((mm[b.ED_HD[y]] > 0) & (mm[b.ED_HD[y]] <= 17))
        selfemp = fam[b.SELFEMP_HD[y]].isin(b.SELFEMP_CODES)
        farm = fam[b.FARM_INC[y]].where(fam[b.FARM_INC[y]] < b.MISSING_FROM, 0)
        bus = fam[b.BUS_ASSET[y]].where(fam[b.BUS_ASSET[y]] < b.MISSING_FROM, 0)
        sel = (head & age.between(lo, hi) & ed.between(b.COMPHS_LO, b.COMPHS_HI - 1)
               & ~selfemp & (farm == 0) & (bus == 0))
        idx = np.flatnonzero(sel.to_numpy())
        col = lambda k: b.col(fam, k, y).iloc[idx].fillna(0).to_numpy()  # noqa: E731
        cash = col("chk") + col("cd")
        card = col("cc")
        home = (b.col(fam, "w2", y) - b.col(fam, "w1", y)).iloc[idx].fillna(0).to_numpy()
        assets = sum(col(k) for k in ("stocks", "ira", "veh", "oth", "re_a", "fb_a"))
        debts = sum(col(k) for k in ("re_b", "fb_b", "stud", "med", "legal",
                                     "famloan", "othdebt"))
        pens = sum(np.where(np.isfinite(v := pen[c].to_numpy(float)[idx]) & (v < b.PENSION_DK),
                            v, 0.0) for c in b.PENSION[y])
        d = b.deflator(y)
        a = age.to_numpy()[idx].astype(float)
        mi = grids.mean_income(a)
        ill_net = (assets + home - debts + pens) * d
        rows.append(pd.DataFrame(dict(
            wave=y, age=a, person=idx,
            married=fam[MARITAL[y]].to_numpy()[idx] == 1,
            liq_net=(cash - card) * d / mi,
            liq_gross=np.where(card > 0, -card, cash) * d / mi,
            illiquid=np.maximum(ill_net, 0) / mi,
            total=((cash - card) * d + ill_net) / mi)))
    return pd.concat(rows, ignore_index=True)


def model_income_quantile(age: float, q: float = 0.5) -> float:
    """Exact quantile of model income at ``age``: stationary Tauchen mixture of
    the discretised transitory shock (what `simulate` draws from)."""
    c = cal.COMPHS
    states, P = grids.tauchen(c=c)
    pi = grids.stationary(P)
    a = np.array([float(age)])
    ymean = grids.mean_log_income(a, c)[0]
    xmin = grids.credit_limit(a, 1000.0, c)[0]
    lv, pr = [], []
    for s, ps in zip(states, pi):
        probs, levels = grids.discretize_transitory(ymean + s, xjump=1000.0, xmax=4e5,
                                                    xmin=xmin, c=c)
        lv.append(levels)
        pr.append(probs * ps)
    lv, pr = np.concatenate(lv), np.concatenate(pr)
    o = np.argsort(lv)
    return float(lv[o][np.searchsorted(np.cumsum(pr[o]), q)])


def main() -> None:
    ind = pd.read_pickle("PSID-data/extract.pkl")
    fam = pd.read_pickle("PSID-data/tax.pkl")
    edu = pd.read_pickle("PSID-data/edu.pkl")
    pen = pd.read_pickle("PSID-data/pension.pkl")
    scf = json.loads(SCF_OUT.read_text()) if SCF_OUT.exists() else None
    out: dict = {}

    tens = torch.load(TENSOR, weights_only=False)
    x, row = tens["x"].numpy(), tens["psid_row"].numpy()
    married = fam.loc[row, [MARITAL[y] for y in b.WAVES]].to_numpy() == 1
    couple = married.sum(1) >= COUPLE_MIN_WAVES
    age, inc, liq = x[:, :, 4], x[:, :, 0], x[:, :, 2]
    groups = {"all 889": np.ones(len(x), bool), "couples": couple, "not couples": ~couple}
    print(f"households: {len(x)}; couples (married in >= {COUPLE_MIN_WAVES} of 7 "
          f"waves) {couple.sum()}; married all 7 {(married.sum(1) == 7).sum()}")
    out["n"] = {"all": int(len(x)), "couples": int(couple.sum()),
                "married_all7": int((married.sum(1) == 7).sum())}

    # --- 1. seed ---------------------------------------------------------------
    print("\n1. AGE-20 SEED: ratio to model mean income, comphs heads aged 20-24, 2011-2023")
    yh = young_heads(fam, ind, edu, pen)
    seed = {}
    for lab, m in (("all", np.ones(len(yh), bool)), ("married", yh.married.to_numpy())):
        s = yh[m]
        seed[lab] = {k: _q(s[k]) for k in ("liq_net", "liq_gross", "illiquid", "total")}
        seed[lab]["n_head_waves"], seed[lab]["n_persons"] = len(s), int(s.person.nunique())
        print(f"   {lab:8s} {len(s):5d} head-waves, {s.person.nunique():5d} persons;  "
              f"p10/p25/p50/p75/p90")
        for k in ("liq_net", "illiquid", "total"):
            print(f"      {k:9s} " + " ".join(f"{v:7.3f}" for v in seed[lab][k]))
        print(f"      in debt {np.mean(s.liq_gross < 0):.2f}, zero illiquid "
              f"{np.mean(s.illiquid == 0):.2f}")
    print(f"   Laibson (SCF, adjusted): total {cal.COMPHS.med_total_wealth:.3f}, liquid "
          f"{cal.COMPHS.med_liq_wealth:.3f}")
    if scf:
        dec = scf["seed_decomposition"]
        print(f"   SCF raw medians: all comphs {dec['all comphs, raw'][0]:.3f}, cardholders "
              f"{dec['cardholders, raw'][0]:.3f}, married {dec['married all comphs, raw'][0]:.3f}")
    out["seed"] = seed

    # --- 2. credit limit ---------------------------------------------------------
    print("\n2. CREDIT LIMIT vs PSID card debt (2010 $; borrowers = gross liquid < 0)")
    lim = grids.credit_limit(age.ravel(), 1000.0).reshape(age.shape)
    cred = {}
    for lab, g in groups.items():
        for lo, hi in BANDS:
            m = g[:, None] & (age >= lo) & (age <= hi)
            debt = m & (liq < 0)
            d = -liq[debt]
            r = dict(share_debt=float(debt.sum() / m.sum()), p50=float(np.median(d)),
                     p90=float(np.quantile(d, 0.9)), limit=float(np.median(lim[debt])),
                     beyond_raw=float(np.mean(d > lim[debt])),
                     beyond_alpha=float(np.mean(ALPHA * d > lim[debt])))
            cred[f"{lab} {lo}-{hi}"] = r
            print(f"   {lab:12s} {lo}-{hi}: in debt {r['share_debt']:.2f}  debt p50 "
                  f"{r['p50']:7,.0f} p90 {r['p90']:7,.0f}  limit {r['limit']:7,.0f}  "
                  f"beyond limit {r['beyond_raw']:.1%} (x{ALPHA}: {r['beyond_alpha']:.1%})")
    if scf:
        print("   SCF 2010-2013 comphs, raw CCBAL among debtors:")
        for k, v in scf["card_balances"].items():
            if k.startswith("2010-2013") and not k.endswith("20-24"):
                print(f"      {k[10:]}: in debt {v['share_debt']:.2f}  p50 {v['p50']:7,.0f} "
                      f"p90 {v['p90']:7,.0f}")
    out["credit"] = cred

    # --- 3. composition ----------------------------------------------------------
    print("\n3. COMPOSITION: per family, PSID (all waves of the 889) vs model (IPUMS)")
    nfu = fam.loc[row, [N_FU[y] for y in b.WAVES]].to_numpy(float)
    nch = fam.loc[row, [N_CHILD[y] for y in b.WAVES]].to_numpy(float)
    comp = {}
    for lo, hi in BANDS:
        m = (age >= lo) & (age <= hi)
        a = np.arange(lo, hi + 1, dtype=float)
        sp, kids, dep = grids.household_composition(a)
        r = dict(psid_kids=float(nch[m].mean()), psid_adults=float((nfu - nch)[m].mean()),
                 psid_married=float(married[m].mean()),
                 psid_kids_couples=float(nch[m & couple[:, None]].mean()),
                 model_kids=float(kids.mean()), model_adults=float((sp + dep).mean()))
        comp[f"{lo}-{hi}"] = r
        print(f"   {lo}-{hi}: kids PSID {r['psid_kids']:.2f} (couples "
              f"{r['psid_kids_couples']:.2f}) model {r['model_kids']:.2f}   adults PSID "
              f"{r['psid_adults']:.2f} model {r['model_adults']:.2f}   married "
              f"{r['psid_married']:.2f}")
    out["composition"] = comp

    # --- 4. income -------------------------------------------------------------
    print("\n4. INCOME: median after-tax non-asset income (2010 $)")
    incd = {}
    for lo, hi in BANDS:
        m = (age >= lo) & (age <= hi)
        mid = (lo + min(hi, 60)) / 2
        r = {lab: float(np.median(inc[m & g[:, None]])) for lab, g in groups.items()}
        r["model"] = model_income_quantile(mid)
        incd[f"{lo}-{hi}"] = r
        print(f"   {lo}-{hi}: all {r['all 889']:8,.0f}  couples {r['couples']:8,.0f}  "
              f"not couples {r['not couples']:8,.0f}  model (age {mid:.0f}) {r['model']:8,.0f}")
    out["income"] = incd

    # --- 5. card ---------------------------------------------------------------
    print("\n5. CARD ACCESS: known cardholders = any card debt in 7 waves")
    ever = (liq < 0).any(1)
    card = {lab: float(ever[g].mean()) for lab, g in groups.items()}
    for lab, v in card.items():
        print(f"   {lab:12s} ever in debt {v:.3f}  -> cardholder share must be >= this")
    if scf:
        pi = scf["card_holding"]
        for grp in ("all", "couples"):
            vals = [pi.get(f"hasBank {grp} {lo}-{hi} {y}") for lo, hi in
                    [(25, 34), (35, 44), (45, 54)] for y in (2016, 2019, 2022)]
            vals = [v for v in vals if v is not None]
            print(f"   SCF bank-card holding, comphs {grp}, 25-54, 2016-2022: "
                  f"{min(vals):.2f}-{max(vals):.2f}")
    out["card"] = card

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
