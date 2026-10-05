"""Laibson et al.'s SCF first stage, ported to Python and taken apart (RESULTS 42).

Two of the simulator's fixed inputs come from the SCF, not from PSID:

* the **age-20 seed** -- median total and liquid wealth over mean income, ages
  20-24 (`SCF/code/4_initialwealth.do`), 1.4696 and 0.0549 for comphs;
* the **credit limit** -- a quadratic in age, as a multiple of mean income
  (`SCF/code/5_examinecredlimits.do`), (0.1672, -0.00187, 0.000136).

The replication package ships only 10-row stubs of the SCF, so neither could be
checked against anything. This script downloads nothing itself: it reads the
public SCF files that ``data/external/scf/download.sh`` fetched (full public
data, summary extract, NBER TAXSIM household files), rebuilds the variables as
`2_buildmoments.do` does, and:

1. **Gate.** Reproduces their two outputs, per implicate. Nothing below is
   reported unless the port lands within ~2%.
2. **Seed decomposition.** From the raw median for all comphs households,
   through the cardholder filter, the two-head / kids / unemployment
   adjustment, to the cohort adjustment -- which step makes the seed 30 times
   PSID's?
3. **Denominator.** SCF mean after-tax income by age, against the model's
   (PSID-estimated) mean income that the ratios are multiplied by.
4. **Card balances.** SCF ``CCBAL`` among debtors, raw (no alpha), against
   PSID's W39A from ``check_calibration_psid.py``: does the alpha = 2.02
   under-reporting correction apply equally to PSID?
5. **Card-holding rate** for comphs by age: X410 (their ``hasVisa``) and X7973
   (bank-type card, the only possession question from 2016 on). The prior for
   the card-access type.

Ported literally where it matters, including two quirks: Stata's weighted
percentile rule, and the CPI chain (summary extract from 2022 dollars back to
the survey year, TAXSIM from the income year, then everything to 2010 dollars
at the survey year's September CPI).

Usage::

    PYTHONPATH=. .venv/bin/python scripts/scf_first_stage.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hh_npe.simulator import grids
from hh_npe.simulator import laibson_calibration as cal

SCF = Path("data/external/scf")
REPL = Path("replication-package-LLMRT/ParameterAndMoments/SCF")
OUT = Path("outputs/scf_first_stage.json")

YEARS = (1995, 1998, 2001, 2004, 2007, 2010, 2013)
#: For the card-holding rate only (no income needed): closer to PSID's years.
PI_YEARS = (2010, 2013, 2016, 2019, 2022)
#: comphs `relevantUnemployment` in 4_ and 5_.
UNEMP_REF = 7.074417114257813
COH_8084 = 17

#: Their published comphs outputs, the gate.
THEIR_SEED = (1.46957951, 0.05486016)
THEIR_CREDIT = (-1.86936547e-03, 1.35663441e-04, 1.67212279e-01)   # AGE, AGESQ, cons

ROSTER_AGE = "X110 X116 X122 X128 X134 X204 X210 X216 X222 X228".split()
ROSTER_DEP = "X113 X119 X125 X131 X137 X207 X213 X219 X225 X231".split()
FULL_VARS = (["Y1", "YY1", "X42001", "X7001", "X7020", "X102", "X107", "X4106",
              "X4706", "X7132", "X410", "X7973", "X414"] + ROSTER_AGE + ROSTER_DEP)
SUMMARY_WEALTH = ("liq bond stocks nmmf cashli retqliq cds savbnd othfin othma "
                  "houses oresre nnresre mrthel resdbt vehic veh_inst othnfin bus "
                  "install odebt othloc ccbal").split()
SUMMARY_INCOME = "income wageinc ssretinc bussefarminc transfothinc".split()
SUMMARY_OTHER = "age edcl married".split()
TAX_VARS = ("fedtaxliab", "sttaxliab", "fica")


# --------------------------------------------------------------------- inputs

def _cpi():
    m = pd.read_csv(SCF / "CPIAUCSL.csv", header=None, names=["date", "cpi"],
                    parse_dates=["date"])
    annual = m.groupby(m.date.dt.year).cpi.mean()
    sep = m[m.date.dt.month == 9]
    sept = pd.Series(sep.cpi.to_numpy(), index=sep.date.dt.year.to_numpy())
    return annual, sept


def _unemployment():
    u = pd.read_csv(SCF / "national_unemployment.csv", header=None,
                    names=["date", "u"], parse_dates=["date"])
    return u.set_index(u.date.dt.year).u


def _alpha() -> float:
    return float(pd.read_csv(REPL / "data/alpha_bootstrap.csv", header=None)[1].mean())


def _read_full(year: int) -> pd.DataFrame:
    yy = f"{year % 100:02d}"
    path = SCF / f"p{yy}i6.dta"
    with pd.io.stata.StataReader(path) as r:
        names = list(r.variable_labels())
    up = {n.upper(): n for n in names}
    cols = [up[v] for v in FULL_VARS if v in up]
    d = pd.read_stata(path, columns=cols, convert_categoricals=False)
    d.columns = [c.upper() for c in d.columns]
    for v in FULL_VARS:           # absent in some years: Stata's append gives missing
        if v not in d:
            d[v] = np.nan
    return d


def load_year(year: int, with_tax: bool = True) -> pd.DataFrame:
    """One survey year merged as in `1_scf_download.do`, all dollars converted to
    the survey year's own dollars (before `adjustCPI`)."""
    annual, _ = _cpi()
    d = _read_full(year)
    d["year"] = year
    # Stored as int16 in some years: widen first, or 10 * YY1 wraps around.
    d["Y1"] = d.Y1.astype(np.int64)
    d["YY1"] = d.YY1.astype(np.int64)
    d["rep"] = d.Y1 - 10 * d.YY1
    assert d.rep.between(1, 5).all(), f"{year}: implicate number out of range"
    d["wgt"] = d.X42001 / 5
    s = pd.read_stata(SCF / f"rscfp{year}.dta",
                      columns=["y1"] + SUMMARY_WEALTH + SUMMARY_INCOME + SUMMARY_OTHER,
                      convert_categoricals=False)
    s["y1"] = s.y1.astype(np.int64)
    # The summary extract is in 2022 dollars; back to the survey year.
    for v in SUMMARY_WEALTH + SUMMARY_INCOME:
        s[v] = s[v] * annual[year] / annual[2022]
    s.columns = ["Y1"] + [c.upper() for c in s.columns[1:]]
    d = d.merge(s, on="Y1", how="left")
    if with_tax:
        t = pd.read_stata(SCF / f"ftax{year % 100:02d}ph.dta",
                          columns=["y1", *TAX_VARS]).rename(columns={"y1": "Y1"})
        t["Y1"] = t.Y1.astype(np.int64)
        for v in TAX_VARS:        # income year -> survey year dollars
            t[v] = t[v] * annual[year] / annual[year - 1]
        d = d.merge(t, on="Y1", how="inner")
    return d


# ---------------------------------------------------------------- build vars

def build(d: pd.DataFrame, alpha: float, unemp_1995_98: str) -> pd.DataFrame:
    """`2_buildmoments.do`: demographics, income, cards, wealth, then 2010 $."""
    annual, sept = _cpi()
    year = int(d.year.iloc[0])
    d = d.copy()
    # demographics
    second = (d.X7020 == 2) if year >= 2001 else ((d.X102 != 0) & (d.X107 == 1))
    d["nhead"] = np.where(second, 2, 1)
    ages = d[ROSTER_AGE].to_numpy()
    deps = d[ROSTER_DEP].to_numpy() == 1
    with np.errstate(invalid="ignore"):
        d["und18"] = ((ages < 18) & (ages != 0) & deps).sum(axis=1)
        d["ndepad"] = ((ages >= 18) & (ages != 0) & deps).sum(axis=1)
    d["head_selfempl"] = d.X4106.between(2, 4)
    d["spouse_selfempl"] = d.X4706.between(2, 4)
    d["comphs"] = d.EDCL.isin([2, 3])
    u = _unemployment()
    if 1992 <= year <= 1998 and unemp_1995_98 == "drop":
        # Region (X30074) is not in the public files, so their census-division
        # rate is missing and `filterData` drops these years.
        d["unemprate"] = np.nan
    else:
        d["unemprate"] = u[year]
    # income
    btinc = (d.WAGEINC + d.SSRETINC + np.maximum(0, d.BUSSEFARMINC)
             + np.maximum(0, d.TRANSFOTHINC))
    d["btincome"] = btinc
    if "fedtaxliab" in d:
        taxes = d.fedtaxliab + d.fica / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            mult = btinc / d.INCOME
        mult = np.where(np.isnan(mult) | (mult > 1), 1.0, np.where(mult < 0, 0.0, mult))
        taxes = taxes * mult + 0.05 * btinc
        d["atincome"] = np.maximum(btinc - taxes, 0)
    else:
        d["atincome"] = np.nan
    # cards
    d["ccdebt"] = alpha * d.CCBAL
    i_cc = d.X7132 / 100
    i_cc = i_cc.where(i_cc != 0).clip(lower=0)
    d["ccbal_highint"] = np.where((i_cc > 5) & i_cc.notna(), d.CCBAL, 0.0)
    d["ccdebt_highint"] = alpha * d.ccbal_highint
    d["hasVisa"] = d.X410.where(d.X410 != 5, 0)
    d["hasBank"] = d.X7973.where(d.X7973 != 5, 0)
    d["credlimit"] = d.X414.where(d.X414 > 0)
    # wealth
    fin_a = (d.LIQ + d.BOND + d.STOCKS + d.NMMF + d.CASHLI + d.RETQLIQ + d.CDS
             + d.SAVBND + d.OTHFIN + d.OTHMA)
    d["fin_a"] = fin_a
    d["fin_nw"] = fin_a - d.ccdebt
    nfin_a = d.HOUSES + d.ORESRE + d.NNRESRE + d.VEHIC + d.OTHNFIN + d.BUS
    nfin_l = (d.MRTHEL + d.RESDBT + d.VEH_INST + (d.INSTALL - d.VEH_INST)
              + d.ODEBT + d.OTHLOC)
    d["nfin_nw"] = nfin_a - nfin_l
    d["wealth"] = d.fin_nw + d.nfin_nw
    f = annual[2010] / sept[year]
    for v in ("LIQ", "fin_a", "fin_nw", "nfin_nw", "wealth", "CCBAL", "ccdebt",
              "ccbal_highint", "ccdebt_highint", "credlimit", "btincome", "atincome"):
        d[f"{v}10"] = d[v] * f
    return d


def build_all(years, alpha, unemp_1995_98="national", with_tax=True) -> pd.DataFrame:
    return pd.concat([build(load_year(y, with_tax), alpha, unemp_1995_98)
                      for y in years], ignore_index=True)


# ------------------------------------------------------------- stata helpers

def wquantile(x, w, p):
    """Stata's weighted percentile (`summarize, detail` with aweights)."""
    o = np.argsort(x, kind="mergesort")
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    cw = np.cumsum(w)
    P = p * cw[-1]
    j = min(int(np.searchsorted(cw, P, side="left")), len(x) - 1)   # first cw >= P
    if np.isclose(cw[j], P, rtol=1e-12, atol=0):
        return 0.5 * (x[j] + x[min(j + 1, len(x) - 1)])
    return x[j]


def wls(X, y, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def winsorize(d: pd.DataFrame, cols_strict: tuple, col_ge: str = "atincome10"):
    """`filterData`'s p99 caps by 5-year age group (wealth-type: `>`, income: `>=`)."""
    d = d.copy()
    grp = np.minimum((d.AGE - 20) // 5, 12)
    for g in np.unique(grp):
        m = (grp == g).to_numpy()
        p = wquantile(d.loc[m, col_ge], d.loc[m, "wgt"], 0.99)
        d.loc[m & (d[col_ge] >= p).to_numpy(), col_ge] = p
        for c in cols_strict:
            p = wquantile(d.loc[m, c], d.loc[m, "wgt"], 0.99)
            d.loc[m & (d[c] > p).to_numpy(), c] = p
    return d


def filter_sample(d, first_age, cardholders=True, liq=True, tax=True):
    m = d.comphs & d.AGE.between(first_age, 90)
    if cardholders:
        m &= d.hasVisa == 1
    m &= ~(d.head_selfempl | d.spouse_selfempl) & (d.BUSSEFARMINC == 0)
    if tax:
        m &= d.atincome10 >= 1000
    need = ["CCBAL", "wealth10", "AGE", "nhead", "ndepad", "und18", "unemprate"]
    if tax:
        need.append("atincome10")
    m &= d[need].notna().all(axis=1)
    s = d[m].copy()
    s["liq_nw10"] = s.LIQ10 - s.ccdebt_highint10
    return winsorize(s, ("wealth10", "liq_nw10") if liq else ("wealth10",))


def agebin(age, first_age):
    a = np.asarray(age, float)
    return np.where((a >= first_age) & (a <= 70), a - 20,
                    np.where(a <= 75, 51, np.where(a <= 80, 52, 53)))


def meaninc(s: pd.DataFrame, first_age: int, wt: str) -> pd.Series:
    """Mean after-tax income for the age bin of AGE + 1 (SCF income is t-1)."""
    bins = agebin(s.AGE, first_age)
    by_bin = {b: np.average(s.atincome10[bins == b], weights=s[wt][bins == b])
              for b in np.unique(bins)}
    present = set(s.AGE.astype(int))
    out = {}
    for a in present:
        nxt = a + 1
        out[a] = by_bin.get(float(agebin([nxt], first_age)[0]), np.nan) if nxt in present else np.nan
    return s.AGE.astype(int).map(out)


def cohort(byear):
    c = np.zeros(len(byear), int)
    c[byear <= 1904] = 1
    for yr in range(2, 21):
        c[(byear >= 1900 + 5 * (yr - 1)) & (byear < 1900 + 5 * yr)] = yr
    return c


TERMS = ("nhead", "kids", "depad", "unemp")


def adjust(s: pd.DataFrame, y: np.ndarray, wt: str, demog=True, coh=True,
           terms: tuple = TERMS):
    """The typical-household adjustment of 4_ and 5_. ``terms`` picks which of
    the demographic shifts to apply, for the decomposition."""
    ok = np.isfinite(y)
    s, y = s[ok], y[ok]
    coh_idx = cohort((s.year - s.AGE).to_numpy())
    levels = np.unique(coh_idx)
    ages = np.unique(s.AGE.astype(int))
    X = np.column_stack(
        [s.nhead, s.ndepad, s.und18, s.unemprate]
        + [(coh_idx == c).astype(float) for c in levels]
        + [(s.AGE.astype(int) == a).astype(float) for a in ages])
    b = wls(X, y, s[wt].to_numpy())
    bn, bd, bk, bu = b[:4]
    bc = dict(zip(levels, b[4:4 + len(levels)]))
    age = s.AGE.to_numpy(float)
    c = cal.COMPHS
    kids = c.a0_kids * np.exp(c.a1_kids * age - c.a2_kids * age ** 2)
    depad = c.a0_depadul * np.exp(c.a1_depadul * age - c.a2_depadul * age ** 2)
    out = y.copy()
    if demog:
        shift = {"nhead": bn * (2 - s.nhead), "kids": bk * (kids - s.und18),
                 "depad": bd * (depad - s.ndepad),
                 "unemp": bu * (UNEMP_REF - s.unemprate)}
        out = out + sum(np.asarray(shift[t], float) for t in terms)
    if coh:
        out = out - np.array([bc[k] for k in coh_idx]) + bc.get(COH_8084, 0.0)
    return s, np.asarray(out, float)


# ------------------------------------------------------------------ outputs

def seed_one(s, wt, demog=True, coh=True, denom="scf", terms=TERMS):
    s = s.copy()
    s["meaninc"] = (meaninc(s, 20, wt) if denom == "scf"
                    else grids.mean_income(s.AGE.to_numpy(float)))
    res = []
    for num in ("wealth10", "liq_nw10"):
        r = (s[num] / s.meaninc).to_numpy()
        if demog or coh:
            ss, rhat = adjust(s, r, wt, demog, coh, terms)
        else:
            ss, rhat = s[np.isfinite(r)], r[np.isfinite(r)]
        m = (ss.AGE <= 24).to_numpy()
        res.append(wquantile(rhat[m], ss[wt].to_numpy()[m], 0.5))
    return res


def by_implicate(s, fn):
    vals = []
    for rep in range(1, 6):
        si = s[s.rep == rep].copy()
        si["wt5"] = si.wgt * 5
        vals.append(fn(si))
    return np.array(vals)


def credit_one(s, wt):
    s = s.copy()
    s["meaninc"] = meaninc(s, 21, wt)
    s = s[s.credlimit10 > 0]
    lam = (s.credlimit10 / s.meaninc).to_numpy()
    ss, lh = adjust(s, lam, wt)
    X = np.column_stack([ss.AGE, ss.AGE ** 2, np.ones(len(ss))])
    return wls(X, lh, ss[wt].to_numpy())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unemp_1995_98", choices=["national", "drop"], default="national")
    args = p.parse_args()
    alpha = _alpha()
    print(f"alpha (mean of their bootstrap) = {alpha:.4f}")
    out: dict = {"alpha": alpha, "unemp_1995_98": args.unemp_1995_98}

    d = build_all(YEARS, alpha, args.unemp_1995_98)
    # --- 1. gate --------------------------------------------------------------
    s4 = filter_sample(d, 20)
    seed = by_implicate(s4, lambda si: seed_one(si, "wt5"))
    s5 = filter_sample(d, 21, liq=False)
    cred = by_implicate(s5, lambda si: credit_one(si, "wt5"))
    their_raw = np.loadtxt(REPL / "output/comphs/credlimit_implicate5results_comphs.raw")
    print("\n1. GATE (port vs their output)")
    print(f"   seed  total {seed[:, 0].mean():.4f} vs {THEIR_SEED[0]:.4f}   "
          f"liquid {seed[:, 1].mean():.4f} vs {THEIR_SEED[1]:.4f}")
    print(f"   per implicate total {np.round(seed[:, 0], 4)}")
    print(f"   credit (AGE, AGESQ, cons) {np.round(cred.mean(0), 6)} vs {THEIR_CREDIT}")
    print(f"   per implicate cons ours {np.round(cred[:, 2], 4)}  theirs {np.round(their_raw[:, 2], 4)}")
    out["gate"] = {"seed": seed.mean(0).tolist(), "seed_theirs": THEIR_SEED,
                   "seed_per_implicate": seed.tolist(),
                   "credit": cred.mean(0).tolist(), "credit_theirs": THEIR_CREDIT,
                   "credit_per_implicate": cred.tolist(),
                   "credit_theirs_per_implicate": their_raw.tolist(),
                   "n_seed_sample_age_le_24": int((s4.AGE <= 24).sum()),
                   "n_credit_sample": int(len(s5))}

    # The credit-limit port does not reproduce theirs (RESULTS 42). Show where
    # their curve sits among what the public data give, in lambda units.
    ages = (25, 35, 45, 55, 65, 75)
    ev = lambda b, a: b[0] * a + b[1] * a * a + b[2]  # noqa: E731
    fits = {"raw": [], "demog": [], "full": []}
    med = []
    for rep in range(1, 6):
        si = s5[s5.rep == rep].copy()
        si["wt5"] = si.wgt * 5
        si["meaninc"] = meaninc(si, 21, "wt5")
        si = si[si.credlimit10 > 0]
        lam = (si.credlimit10 / si.meaninc).to_numpy()
        ok = np.isfinite(lam)
        X = np.column_stack([si.AGE, si.AGE ** 2, np.ones(len(si))])
        fits["raw"].append(wls(X[ok], lam[ok], si.wt5.to_numpy()[ok]))
        for k, kw in (("demog", dict(coh=False)), ("full", {})):
            ss, lh = adjust(si, lam, "wt5", **kw)
            fits[k].append(wls(np.column_stack([ss.AGE, ss.AGE ** 2, np.ones(len(ss))]),
                               lh, ss.wt5.to_numpy()))
        if rep == 1:
            for a in ages:
                m = ok & si.AGE.between(a - 5, a + 4).to_numpy()
                med.append(wquantile(lam[m], si.wt5.to_numpy()[m], 0.5))
    curves = {"theirs": [ev(THEIR_CREDIT, a) for a in ages], "raw median (band)": med}
    curves |= {f"{k} fit": [ev(np.mean(v, 0), a) for a in ages] for k, v in fits.items()}
    print(f"   credit limit / mean income at ages {ages}:")
    for k, v in curves.items():
        print(f"      {k:18s} " + " ".join(f"{x:6.3f}" for x in v))
    out["gate"]["credit_curves"] = {"ages": ages, **curves}

    # --- 2. seed decomposition -----------------------------------------------
    print("\n2. SEED DECOMPOSITION: median (total, liquid) / mean income, ages 20-24")
    sall = filter_sample(d, 20, cardholders=False)
    steps = {
        "all comphs, raw": (sall, dict(demog=False, coh=False)),
        "cardholders, raw": (s4, dict(demog=False, coh=False)),
        "cardholders, + 2 heads/kids/unemp": (s4, dict(demog=True, coh=False)),
        "cardholders, + cohort 1980-84 (theirs)": (s4, dict(demog=True, coh=True)),
        "all comphs, full adjustment": (sall, dict(demog=True, coh=True)),
        "cardholders, cohort only": (s4, dict(demog=False, coh=True)),
        "cardholders, 2 heads only": (s4, dict(demog=True, coh=False, terms=("nhead",))),
        "cardholders, kids only": (s4, dict(demog=True, coh=False, terms=("kids",))),
        "cardholders, dependent adults only": (s4, dict(demog=True, coh=False, terms=("depad",))),
        "cardholders, unemployment only": (s4, dict(demog=True, coh=False, terms=("unemp",))),
        "cardholders raw, model mean income": (s4, dict(demog=False, coh=False, denom="model")),
        "all comphs raw, model mean income": (sall, dict(demog=False, coh=False, denom="model")),
    }
    dec = {}
    for k, (s, kw) in steps.items():
        v = by_implicate(s, lambda si: seed_one(si, "wt5", **kw)).mean(0)
        dec[k] = v.tolist()
        print(f"   {k:42s} total {v[0]:7.3f}  liquid {v[1]:7.3f}")
    young = sall[sall.AGE <= 24]
    for lab, m in (("married (nhead 2)", young.nhead == 2), ("cardholder", young.hasVisa == 1)):
        sh = np.average(m, weights=young.wgt)
        print(f"   share of comphs 20-24 households: {lab} {sh:.3f}")
        dec[f"share_{lab}"] = float(sh)
    sm = sall[(sall.nhead == 2)]
    v = by_implicate(sm, lambda si: seed_one(si, "wt5", demog=False, coh=False)).mean(0)
    print(f"   {'married (2 heads), all comphs, raw':42s} total {v[0]:7.3f}  liquid {v[1]:7.3f}")
    dec["married all comphs, raw"] = v.tolist()
    out["seed_decomposition"] = dec

    # --- 3. denominator ------------------------------------------------------
    print("\n3. MEAN INCOME (2010 $): SCF comphs cardholders, after tax, vs model")
    s1 = s4[s4.rep == 1].copy()
    s1["mi"] = meaninc(s1, 20, "wgt")
    den = {}
    for a in (20, 22, 24, 30, 40, 50, 60):
        scf = float(s1.loc[s1.AGE == a, "mi"].iloc[0]) if (s1.AGE == a).any() else np.nan
        mod = float(grids.mean_income(np.array([float(a)]))[0])
        den[a] = (scf, mod)
        print(f"   age {a}: SCF {scf:9,.0f}   model {mod:9,.0f}   ratio {scf / mod:5.2f}")
    out["mean_income"] = den

    # --- 4. card balances ----------------------------------------------------
    print("\n4. CARD BALANCES, raw CCBAL (no alpha), 2010 $, comphs, all households")
    bands = [(20, 24), (25, 34), (35, 44), (45, 54), (55, 64)]
    cb = {}
    for yrs, lab in ((YEARS, "1995-2013"), ((2010, 2013), "2010-2013")):
        sb = sall[sall.year.isin(yrs)]
        for lo, hi in bands:
            b = sb[sb.AGE.between(lo, hi)]
            deb = b[b.CCBAL10 > 0]
            row = dict(share_debt=float(np.average(b.CCBAL10 > 0, weights=b.wgt)),
                       share_debt_cardholders=float(np.average(
                           b.CCBAL10[b.hasVisa == 1] > 0, weights=b.wgt[b.hasVisa == 1])),
                       p50=wquantile(deb.CCBAL10, deb.wgt, 0.5),
                       p90=wquantile(deb.CCBAL10, deb.wgt, 0.9))
            cb[f"{lab} {lo}-{hi}"] = row
            print(f"   {lab} {lo}-{hi}: in debt {row['share_debt']:.2f} (cardholders "
                  f"{row['share_debt_cardholders']:.2f})  debtors p50 {row['p50']:8,.0f} "
                  f"p90 {row['p90']:8,.0f}")
    out["card_balances"] = cb

    # --- 5. card-holding rate -----------------------------------------------
    print("\n5. CARD-HOLDING RATE, comphs (not self-employed, no business income)")
    pi = {}
    dp = pd.concat([build(load_year(y, with_tax=(y <= 2013)), alpha, "national")
                    for y in PI_YEARS], ignore_index=True)
    dp = dp[dp.comphs & ~(dp.head_selfempl | dp.spouse_selfempl) & (dp.BUSSEFARMINC == 0)]
    for var in ("hasVisa", "hasBank"):
        for married in (False, True):
            for lo, hi in [(20, 24), (25, 34), (35, 44), (45, 54), (55, 64)]:
                for y in PI_YEARS:
                    b = dp[(dp.year == y) & dp.AGE.between(lo, hi) & dp[var].notna()]
                    if married:
                        b = b[b.nhead == 2]
                    if len(b) == 0:
                        continue
                    pi[f"{var} {'couples' if married else 'all'} {lo}-{hi} {y}"] = float(
                        np.average(b[var] == 1, weights=b.wgt))
    for var in ("hasVisa", "hasBank"):
        for grp in ("all", "couples"):
            line = []
            for lo, hi in [(20, 24), (25, 34), (35, 44), (45, 54), (55, 64)]:
                vals = [pi.get(f"{var} {grp} {lo}-{hi} {y}") for y in PI_YEARS]
                line.append(" ".join("  -  " if v is None else f"{v:.2f}" for v in vals))
            print(f"   {var:8s} {grp:8s} " + " | ".join(line))
    print(f"   (columns per band: {PI_YEARS}; bands 20-24 | 25-34 | 35-44 | 45-54 | 55-64)")
    out["card_holding"] = pi

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
