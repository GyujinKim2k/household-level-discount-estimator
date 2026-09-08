"""Laibson et al.'s 'typical-household adjustment', applied to the PSID tensor.

They do not feed raw household data to the model. Every moment is first
regressed on household composition and local labour-market conditions, then
each household is re-centred onto the model's OWN demographic assumptions
(`3_scfanalysis_withCondMoments_bs.do:52-56`)::

    regress <moment> nhead ndepad und18 unemprate cohd* aged* [aweight=wgt], noc r
    gen hat = moment + _b[nhead]*(2 - nhead)
                     + _b[und18]*(b0child*exp(b1child*AGE - b2child*AGE^2) - und18)
                     + _b[ndepad]*(b0depad*exp(b1depad*AGE - b2depad*AGE^2) - ndepad)
                     + _b[unemprate]*(relevantUnemployment - unemprate)
                     - sum_i _b[cohd_i]*cohd_i

The reason is structural, not cosmetic. The model has household composition as a
*deterministic function of age* (SIMULATOR_SPEC 2.2): it cannot represent a
45-year-old with four children separately from one with none, and assigns both
the age-45 average. Rather than change the model, they change the data --
removing the composition variation the model cannot see and substituting the
profile it assumes.

We fed raw PSID households straight in, with real composition varying. This
closes that gap. The demographic targets are the model's own frozen constants
(`laibson_calibration.A0_KIDS` etc.), so the adjustment aims at exactly what the
simulator generates.

Applied to the dollar features only. `age` is the conditioning variable and is
left alone.

Usage::

    uv run python scripts/typical_household.py \
        --x data/processed/psid_x_matched.pt \
        --out data/processed/psid_x_typical.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hh_npe.simulator import laibson_calibration as cal

WAVES = [2011, 2013, 2015, 2017, 2019, 2021, 2023]
MISSING_FROM = 9_999_998

NFU = dict(zip(WAVES, ["ER47316", "ER53016", "ER60016", "ER66016",
                       "ER72016", "ER78016", "ER82017"]))
NKIDS = dict(zip(WAVES, ["ER47320", "ER53020", "ER60021", "ER66021",
                         "ER72021", "ER78021", "ER82022"]))
MSTAT = dict(zip(WAVES, ["ER47323", "ER53023", "ER60024", "ER66024",
                         "ER72024", "ER78025", "ER82026"]))
STATE = dict(zip(WAVES, ["ER47303", "ER53003", "ER60003", "ER66003",
                         "ER72003", "ER78003", "ER82003"]))
MARRIED = 1
#: The reference unemployment rate the moments are re-centred on. Their
#: ${relevantUnemployment} is the sample-average rate; we use the same
#: definition rather than a literal, since our years differ from theirs.
DOLLAR_FEATURES = (0, 1, 2, 3)     # income, consumption, liquid, illiquid


# The moments are re-centred on the MODEL's own demographic profile, so these
# must use the same education bundle the simulated data was generated under.
# Demographics are group-varying (`a1_kids` runs 0.262 / 0.358 / 0.576 across
# somehs / comphs / compco), so re-centring a somehs sample on the comphs
# profile would compare each side to a different household.


def model_kids(age: np.ndarray, c: cal.Calibration = cal.COMPHS) -> np.ndarray:
    return c.a0_kids * np.exp(c.a1_kids * age - c.a2_kids * age ** 2)


def model_depadul(age: np.ndarray,
                  c: cal.Calibration = cal.COMPHS) -> np.ndarray:
    return c.a0_depadul * np.exp(c.a1_depadul * age - c.a2_depadul * age ** 2)


#: National unemployment, FRED UNRATE annual means. Frozen as literals: the
#: shipped state file (`stateuerates_all.csv`) stops in 2015 because it was
#: built for their 1995-2013 SCF sample, and BLS returns 403 to scripted
#: downloads, so later waves have to be imputed from the national series.
NATIONAL_UNEMP = {
    2010: 9.62, 2012: 8.07, 2014: 6.17, 2016: 4.87,
    2018: 3.90, 2020: 8.09, 2022: 3.65, 2015: 5.29,
}


def state_unemployment(psid_dir: Path, repl: Path) -> dict:
    """Annual state unemployment rate, keyed (psid_state_code, year).

    State rates exist through 2015 only. Later reference years are imputed as
    the state's 2015 position scaled to that year's national level:

        rate(state, y) = rate(state, 2015) * national(y) / national(2015)

    That keeps both dimensions the regression needs -- cross-state variation,
    which is what identifies the coefficient in a cross-section, and the time
    variation, which is large here (9.6% in 2010 against 3.7% in 2022). Using
    the national rate alone would make the term collinear with the age and
    cohort dummies and identify nothing.
    """
    from scripts.taxsim_run import PSID_TO_SOI

    df = pd.read_csv(repl / "stateuerates_all.csv", encoding="latin-1")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    # Measure code is the last two digits: 03 = unemployment RATE. Codes
    # 04/05/06 are unemployment, employment and labour-force LEVELS, in
    # millions -- averaging those in gives a "rate" of 2.8 million percent.
    sid = df["Series Id"].astype(str)
    df = df[sid.str.startswith(("LASST", "LAUST")) & sid.str.endswith("03")].copy()
    df["fips"] = df["Series Id"].str.slice(5, 7)
    df["val"] = pd.to_numeric(
        df["Value"].astype(str).str.replace(r"\(.\)", "", regex=True),
        errors="coerce")
    ann = df.groupby(["fips", "Year"])["val"].mean()
    # BLS uses FIPS; PSID uses its own alphabetical code. Both order DC among
    # the states, so map PSID -> SOI -> FIPS by position.
    soi_to_fips = {i: f"{f:02d}" for i, f in enumerate(
        [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22,
         23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
         40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56], start=1)}
    out = {}
    for psid_code, soi in PSID_TO_SOI.items():
        fips = soi_to_fips.get(soi)
        base = ann.get((fips, 2015), np.nan)
        for y in WAVES:
            ref = y - 1                          # income reference year
            v = ann.get((fips, ref), np.nan)
            if not np.isfinite(v) and np.isfinite(base):
                v = base * NATIONAL_UNEMP[ref] / NATIONAL_UNEMP[2015]
            if np.isfinite(v):
                out[(psid_code, y)] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x", type=Path,
                   default=Path("data/processed/psid_x_matched.pt"))
    ap.add_argument("--psid_dir", type=Path, default=Path("PSID-data"))
    ap.add_argument("--repl", type=Path,
                   default=Path("replication-package-LLMRT/ParameterAndMoments/"
                                "PSID/data"))
    ap.add_argument("--out", type=Path,
                   default=Path("data/processed/psid_x_typical.pt"))
    args = ap.parse_args()

    d = torch.load(args.x, weights_only=False)
    x = d["x"].numpy().astype(np.float64)
    rows = d["psid_row"].numpy()
    fam = pd.read_pickle(args.psid_dir / "tax.pkl")
    N, W, F = x.shape
    print(f"x {x.shape}")

    def col(c):
        s = fam[c]
        return s.where(s < MISSING_FROM, np.nan).to_numpy()[rows]

    nkids = np.column_stack([col(NKIDS[y]) for y in WAVES])
    nfu = np.column_stack([col(NFU[y]) for y in WAVES])
    married = np.column_stack([col(MSTAT[y]) for y in WAVES]) == MARRIED
    nhead = np.where(married, 2.0, 1.0)
    # Dependent adults: everyone who is neither a head nor a child.
    ndepad = np.clip(nfu - nkids - nhead, 0, None)
    age = x[:, :, 4]

    ue = state_unemployment(args.psid_dir, args.repl)
    st = np.column_stack([col(STATE[y]) for y in WAVES])
    unemp = np.array([[ue.get((int(st[i, w]) if np.isfinite(st[i, w]) else -1,
                               WAVES[w]), np.nan)
                       for w in range(W)] for i in range(N)])
    ref_unemp = np.nanmean(unemp)
    print(f"state unemployment matched for "
          f"{np.isfinite(unemp).mean():.1%} of household-waves; "
          f"reference rate {ref_unemp:.2f}%")
    unemp = np.where(np.isfinite(unemp), unemp, ref_unemp)

    # Cohort: birth year, bucketed in 5s, as dummies. Age dummies too, so the
    # adjustment never absorbs the age profile the model is meant to explain.
    birth = np.array([[WAVES[w] - age[i, w] for w in range(W)] for i in range(N)])
    coh = np.floor(birth / 5).astype(int)
    agb = np.floor(age / 5).astype(int)

    flat = {k: v.ravel() for k, v in
            dict(nhead=nhead, ndepad=ndepad, und18=nkids, unemp=unemp,
                 age=age.ravel().reshape(N, W)).items()}
    # Stata's `noc` drops the constant and carries full dummy sets. Here a
    # constant is kept and one level of each dummy set is omitted, which is
    # equivalent in fit and numerically better conditioned -- dropping a level
    # WITHOUT a constant leaves the omitted category unrepresented and makes
    # every coefficient meaningless.
    du = pd.get_dummies(pd.Series(coh.ravel()), prefix="coh", drop_first=True)
    da = pd.get_dummies(pd.Series(agb.ravel()), prefix="age", drop_first=True)
    A = np.column_stack([
        np.ones(N * W),
        flat["nhead"], flat["ndepad"], flat["und18"], flat["unemp"],
        du.to_numpy(float), da.to_numpy(float),
    ])
    B0 = 1        # coefficient offset: index 0 is the constant
    ok = np.isfinite(A).all(axis=1)

    xa = x.copy()
    print(f"\n{'feature':16s}{'b_nhead':>11s}{'b_ndepad':>11s}"
          f"{'b_und18':>11s}{'b_unemp':>11s}{'median shift':>14s}")
    names = d["features"]
    tgt_k, tgt_d = model_kids(age).ravel(), model_depadul(age).ravel()
    for j in DOLLAR_FEATURES:
        y = x[:, :, j].ravel()
        m = ok & np.isfinite(y)
        b, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        adj = (b[B0 + 0] * (2.0 - flat["nhead"])
               + b[B0 + 1] * (tgt_d - flat["ndepad"])
               + b[B0 + 2] * (tgt_k - flat["und18"])
               + b[B0 + 3] * (ref_unemp - flat["unemp"]))
        # Cohort dummies are zeroed out (adjust to the reference cohort), age
        # dummies are NOT -- age is the conditioning variable the model uses.
        c0 = B0 + 4
        adj = adj - A[:, c0:c0 + du.shape[1]] @ b[c0:c0 + du.shape[1]]
        newy = y + adj
        xa[:, :, j] = newy.reshape(N, W)
        print(f"{names[j]:16s}{b[B0]:11.0f}{b[B0+1]:11.0f}{b[B0+2]:11.0f}"
              f"{b[B0+3]:11.0f}"
              f"{np.nanmedian(newy - y):14.0f}")

    torch.save({**d, "x": torch.from_numpy(xa).float(),
                "typical_household_adjusted": True,
                "reference_unemployment": float(ref_unemp)}, args.out)
    print(f"\nwrote {args.out}")
    print(f"{'':16s}{'before':>12s}{'after':>12s}")
    for j in DOLLAR_FEATURES:
        print(f"{names[j]:16s}{np.median(x[:, :, j]):12.0f}"
              f"{np.median(xa[:, :, j]):12.0f}")


if __name__ == "__main__":
    main()
