"""After-tax non-asset income for the PSID panel, via NBER TAXSIM35.

Laibson et al.'s income process is estimated on **after-tax non-asset income**
(`PSID/code/2_builddata.do:108`)::

    nasry    = rinc + rlumps - rassetinc
    atincome = nasry - fedtax - oasdi_withheld - hi_withheld - sttax

They took federal tax from PSID's own `hwtax`/`ottax` variables, which PSID
stopped publishing after 1991 -- exactly the window they used and exactly not
ours. So federal and state liabilities have to be computed, and TAXSIM is the
standard way to do it.

Two places this improves on their construction rather than merely reproducing
it: they assumed a flat 5% state tax "as in SCF", where TAXSIM computes the
actual state liability; and they scaled federal tax by a non-asset share, where
TAXSIM is given the asset and non-asset components separately.

TAXSIM is a network service -- upload a CSV, download the liabilities. The ssh
route needs a key we do not have; multipart HTTP works and is verified against
the documented sanity case (mstat=2, year=1970, ltcg=100000 -> fiitax
16700.04).

Usage::

    uv run python scripts/taxsim_run.py --out data/processed/psid_atincome.csv
"""

from __future__ import annotations

import argparse
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

WAVES = [2011, 2013, 2015, 2017, 2019, 2021, 2023]
#: PSID wave Y reports income for calendar year Y-1 (the 2011 wave carries
#: TOTAL FAMILY INCOME-2010). The tax year is therefore the reference year,
#: not the interview year -- getting this wrong applies the wrong year's
#: brackets, and the 2017 TCJA sits inside our window.
TAX_YEAR = {y: y - 1 for y in WAVES}

ENDPOINT = "http://taxsim.nber.org/uptest/webfile.cgi"

# --- PSID variable maps, by wave -------------------------------------------
IV = dict(zip(WAVES, ["ER34101", "ER34201", "ER34301", "ER34501",
                      "ER34701", "ER34901", "ER35101"]))
REL = dict(zip(WAVES, ["ER34103", "ER34203", "ER34303", "ER34503",
                       "ER34703", "ER34903", "ER35103"]))
AGE = dict(zip(WAVES, ["ER47317", "ER53017", "ER60017", "ER66017",
                       "ER72017", "ER78017", "ER82018"]))
STATE = dict(zip(WAVES, ["ER47303", "ER53003", "ER60003", "ER66003",
                         "ER72003", "ER78003", "ER82003"]))
MSTAT = dict(zip(WAVES, ["ER47323", "ER53023", "ER60024", "ER66024",
                         "ER72024", "ER78025", "ER82026"]))
NKIDS = dict(zip(WAVES, ["ER47320", "ER53020", "ER60021", "ER66021",
                         "ER72021", "ER78021", "ER82022"]))
PWAGE = dict(zip(WAVES, ["ER52219", "ER58020", "ER65200", "ER71277",
                         "ER77299", "ER81626", "ER85480"]))
# Spouse wages carry three different label conventions across the panel
# ("WAGES/SALARY OF WIFE", "WAGES AND SALARIES OF SPOUSE", "G13 WAGES/SALARY
# OF SPOUSE"); all seven are annual, verified by magnitude against RP wages.
SWAGE = dict(zip(WAVES, ["ER48615", "ER54309", "ER65228", "ER71305",
                         "ER73424", "ER81654", "ER85508"]))
TOTINC = dict(zip(WAVES, ["ER52343", "ER58152", "ER65349", "ER71426",
                          "ER77448", "ER81775", "ER85629"]))
# Asset-income components, head/RP and spouse, per 2_builddata.do:92.
ASSET = {
    "div_rp":   ["ER52240", "ER58041", "ER65219", "ER71296", "ER77318", "ER81645", "ER85499"],
    "div_sp":   ["ER52253", "ER58054", "ER65247", "ER71324", "ER77346", "ER81673", "ER85527"],
    "int_rp":   ["ER52242", "ER58043", "ER65221", "ER71298", "ER77320", "ER81647", "ER85501"],
    "rent_rp":  ["ER52238", "ER58039", "ER65217", "ER71294", "ER77316", "ER81643", "ER85497"],
    "trust_rp": ["ER52244", "ER58045", "ER65223", "ER71300", "ER77322", "ER81649", "ER85503"],
    "trust_sp": ["ER52257", "ER58058", "ER65251", "ER71328", "ER77350", "ER81677", "ER85531"],
    "bus_rp":   ["ER52217", "ER58018", "ER65198", "ER71275", "ER77297", "ER81624", "ER85478"],
    "bus_sp":   ["ER52247", "ER58048", "ER65226", "ER71303", "ER77325", "ER81652", "ER85506"],
    "ast_otr":  ["ER52313", "ER58122", "ER65319", "ER71396", "ER77418", "ER81745", "ER85599"],
}
ASSET = {k: dict(zip(WAVES, v)) for k, v in ASSET.items()}

MISSING_FROM = 9_999_998
HEAD_CODE = 10
MARRIED = 1          # PSID marital status: 1 = Married

# PSID codes the 48 contiguous states plus DC alphabetically as 1-49, then
# appends Alaska (50) and Hawaii (51). SOI/TAXSIM codes all 51 alphabetically.
# The mapping below is built from those two orderings and then *verified*
# against TAXSIM itself: submitting wage-only income in every state returns
# siitax == 0 for exactly {AK, FL, NV, NH, SD, TN, TX, WA, WY}, which is the
# correct 2020 set. An off-by-one here would silently mis-tax every household.
_CONTIGUOUS_PLUS_DC = [
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "ID", "IL",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
    "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR",
    "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
    "WY",
]
_SOI = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY",
]
_SOI_OF = {a: i + 1 for i, a in enumerate(_SOI)}
PSID_TO_SOI = {i + 1: _SOI_OF[a] for i, a in enumerate(_CONTIGUOUS_PLUS_DC)}
PSID_TO_SOI[50] = _SOI_OF["AK"]
PSID_TO_SOI[51] = _SOI_OF["HI"]
# 0 = foreign/territory, 99 = DK/NA -> state 0, meaning "no state tax computed".
# Federal liability is unaffected, so these households keep a usable atincome.


def clean(s: pd.Series) -> pd.Series:
    """PSID tops each field with DK/NA sentinels; treat them as zero income."""
    return s.where(s < MISSING_FROM, 0.0).fillna(0.0)


def analysis_sample(ind: pd.DataFrame, age_low: int, age_high: int) -> pd.Series:
    m = np.logical_and.reduce([(ind[IV[y]] > 0) & (ind[REL[y]] == HEAD_CODE)
                               for y in WAVES])
    m &= ind[AGE[WAVES[0]]].between(age_low, age_high)
    m &= (ind[AGE[WAVES[-1]]] - ind[AGE[WAVES[0]]]).between(11, 13)
    return m


def build_input(ind: pd.DataFrame, fam: pd.DataFrame,
                keep: pd.Series) -> pd.DataFrame:
    """One TAXSIM record per household-wave."""
    rows = []
    idx = np.flatnonzero(keep.to_numpy())
    for w, year in enumerate(WAVES):
        d = fam.loc[idx]
        i = ind.loc[idx]
        assets = sum(clean(d[ASSET[k][year]]) for k in ASSET)
        state = i[STATE[year]].map(PSID_TO_SOI).fillna(0).astype(int) \
            if STATE[year] in i.columns else d[STATE[year]].map(PSID_TO_SOI).fillna(0).astype(int)
        rows.append(pd.DataFrame({
            # taxsimid encodes household row and wave so the merge back is exact.
            "taxsimid": idx * 10 + w,
            "year": TAX_YEAR[year],
            "state": state.to_numpy(),
            "mstat": np.where(d[MSTAT[year]].to_numpy() == MARRIED, 2, 1),
            "page": i[AGE[year]].clip(18, 99).fillna(40).astype(int).to_numpy(),
            "sage": 0,
            "depx": d[NKIDS[year]].clip(0, 18).fillna(0).astype(int).to_numpy(),
            "pwages": clean(d[PWAGE[year]]).to_numpy(),
            "swages": clean(d[SWAGE[year]]).to_numpy(),
            "dividends": (clean(d[ASSET["div_rp"][year]])
                          + clean(d[ASSET["div_sp"][year]])).to_numpy(),
            "intrec": clean(d[ASSET["int_rp"][year]]).to_numpy(),
            "otherprop": (clean(d[ASSET["rent_rp"][year]])
                          + clean(d[ASSET["trust_rp"][year]])
                          + clean(d[ASSET["trust_sp"][year]])
                          + clean(d[ASSET["bus_rp"][year]])
                          + clean(d[ASSET["bus_sp"][year]])
                          + clean(d[ASSET["ast_otr"][year]])).to_numpy(),
            # Everything in total family income that is not asset income and not
            # wages: transfers, self-employment, other FU members' labour. Fed
            # to TAXSIM as non-property income so it is taxed but not subject to
            # NIIT. Floored at zero -- PSID income can be negative from business
            # losses, which TAXSIM would reject on a non-joint return.
            "nonprop": np.maximum(
                clean(d[TOTINC[year]]) - assets
                - clean(d[PWAGE[year]]) - clean(d[SWAGE[year]]), 0.0).to_numpy(),
            "_nasry": (clean(d[TOTINC[year]]) - assets).to_numpy(),
            "_totinc": clean(d[TOTINC[year]]).to_numpy(),
            "_wave": year,
        }))
    out = pd.concat(rows, ignore_index=True)
    # Non-joint returns must not carry spouse income; TAXSIM aborts on the
    # first offending row and the error makes the whole response unparseable.
    single = out["mstat"] == 1
    moved = int((single & (out["swages"] > 0)).sum())
    if moved:
        out.loc[single, "nonprop"] += out.loc[single, "swages"]
        out.loc[single, "swages"] = 0.0
    return out, moved


def submit(df: pd.DataFrame, chunk: int = 2000) -> pd.DataFrame:
    """POST the CSV to TAXSIM and parse the returned liabilities.

    Chunked at 2000 because the HTTP route truncates near ~2320 records
    regardless of content -- it reports "I/O conversion error" and names a
    record that computes fine when submitted alone, so the message points at
    the data when the cause is size. The ssh route has no such limit but needs
    a key. Do not raise this without re-testing; a silent truncation here would
    drop households from the panel rather than fail.
    """
    cols = [c for c in df.columns if not c.startswith("_")]
    parts = []
    for lo in range(0, len(df), chunk):
        sub = df.iloc[lo:lo + chunk]
        buf = Path(f"/tmp/taxsim_in_{lo}.csv")
        sub[cols].to_csv(buf, index=False, float_format="%.2f")
        res = subprocess.run(
            ["curl", "-s", "--max-time", "300", "-F", f"datafile=@{buf}", ENDPOINT],
            capture_output=True, text=True, timeout=360,
        )
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
        if not lines:
            raise SystemExit(f"empty TAXSIM response for rows {lo}..{lo+len(sub)}")
        hdr = [h.strip() for h in lines[0].split(",")]
        good = []
        for ln in lines[1:]:
            p = ln.split(",")
            if len(p) != len(hdr):
                continue
            try:
                good.append([float(x) for x in p])
            except ValueError:
                continue          # trailer lines, e.g. " TAXSIM:"
        got = pd.DataFrame(good, columns=hdr)
        if len(got) != len(sub):
            raise SystemExit(
                f"TAXSIM returned {len(got)} rows for {len(sub)} sent "
                f"(rows {lo}..). It stops at the first bad record; check the "
                f"tail of {buf} for the offending line."
            )
        parts.append(got)
        print(f"  rows {lo:6d}-{lo+len(sub):6d}: ok")
    return pd.concat(parts, ignore_index=True)


def after_tax(tin: pd.DataFrame, tax: pd.DataFrame) -> pd.DataFrame:
    """Merge TAXSIM's liabilities onto its input and compute ``atincome``.

    Laibson et al. subtract federal, state and the *employee* share of FICA.
    TAXSIM's `fica` is employee + employer, so it is halved.
    """
    m = tin.merge(tax, on="taxsimid", suffixes=("", "_t"))
    m["tfica"] = m["fica"] / 2.0

    # TAXSIM computes liability on TOTAL income, including the dividends,
    # interest and rent that `nasry` excludes. Subtracting the whole liability
    # from non-asset income alone charges asset-income tax to labour income --
    # which for a few households drives after-tax income below zero, and for
    # every household with assets biases it down.
    #
    # Laibson et al. handle it by pro-rating (2_builddata.do:96): assume tax is
    # paid on all income, then scale by the non-asset share. Same rule here,
    # applied to state tax as well since TAXSIM computes that on total income
    # too. FICA is levied on wages only and is not scaled.
    share = (m["_nasry"] / m["_totinc"].replace(0, np.nan)).clip(0, 1).fillna(1.0)
    m["tax_share"] = share
    m["fiitax_adj"] = m["fiitax"] * share
    m["siitax_adj"] = m["siitax"] * share
    m["atincome"] = m["_nasry"] - m["fiitax_adj"] - m["siitax_adj"] - m["tfica"]
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--psid_dir", type=Path, default=Path("PSID-data"))
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/psid_atincome.csv"))
    p.add_argument("--age_low", type=int, default=25)
    p.add_argument("--age_high", type=int, default=46)
    p.add_argument("--dry_run", action="store_true",
                   help="Build and describe the input without submitting.")
    args = p.parse_args()

    ind = pd.read_pickle(args.psid_dir / "extract.pkl")
    fam = pd.read_pickle(args.psid_dir / "tax.pkl")
    keep = analysis_sample(ind, args.age_low, args.age_high)
    print(f"analysis sample: {int(keep.sum())} households x {len(WAVES)} waves")

    tin, moved = build_input(ind, fam, keep)
    print(f"TAXSIM input: {len(tin)} records, "
          f"years {sorted(tin['year'].unique())}")
    if moved:
        print(f"  {moved} single-filer records had spouse wages; folded into "
              f"nonprop (TAXSIM rejects swages on non-joint returns)")
    print(f"  married (mstat=2): {float((tin['mstat'] == 2).mean()):.3f}")
    print(f"  state==0 (foreign/DK, federal only): "
          f"{int((tin['state'] == 0).sum())}")
    if args.dry_run:
        print(tin.describe().T[["mean", "50%", "max"]].round(0).to_string())
        return

    print("submitting to TAXSIM...")
    m = after_tax(tin, submit(tin))
    m["hh"] = m["taxsimid"] // 10
    m["wave"] = m["_wave"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ["hh", "wave", "year", "state", "mstat", "_totinc", "_nasry",
                 "tax_share", "fiitax", "siitax", "fiitax_adj", "siitax_adj",
                 "tfica", "atincome"]
    m[keep_cols].rename(columns={"_nasry": "nasry",
                                 "_totinc": "totinc"}).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(m)} rows)")

    print(f"\n{'wave':6s}{'nasry':>10s}{'fiitax':>10s}{'siitax':>9s}"
          f"{'tfica':>9s}{'atincome':>11s}{'avg rate':>10s}")
    for y in WAVES:
        s = m[m["_wave"] == y]
        rate = 1 - s["atincome"].sum() / s["_nasry"].sum()
        print(f"{y:<6d}{s['_nasry'].median():10.0f}{s['fiitax'].median():10.0f}"
              f"{s['siitax'].median():9.0f}{s['tfica'].median():9.0f}"
              f"{s['atincome'].median():11.0f}{rate:10.3f}")
    neg = int((m["atincome"] <= 0).sum())
    if neg:
        print(f"\n{neg} records with atincome <= 0 "
              f"({neg / len(m):.3%}) -- inspect before use")


if __name__ == "__main__":
    main()
