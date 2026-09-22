"""Assemble the empirical ``x`` tensor for the PSID panel.

Produces ``(N, 7, 5)`` in :data:`FEATURES_TWOASSET_AGE` order -- income,
consumption, liquid_assets, illiquid_assets, age -- matching what the posterior
was trained on, so ``posterior.sample(x=x[i])`` is directly applicable.

Units are the load-bearing detail. Laibson et al. deflate to **2010 dollars**
(`2_builddata.do:77`, `cpibaseyear 2010`), so the simulator emits 2010 dollars:
simulated median income is 40,000 at age 30 and 53,000 at age 40. Our TAXSIM
after-tax non-asset income for tax year 2010 has a median of 40,593, which is
an independent check that the two are on the same scale before anything is
deflated.

Timing within a wave. PSID dates flows and stocks a year apart: the interview
in year Y reports wealth as of the interview but income for calendar Y-1. Flows
are therefore deflated at the reference year and stocks at the interview year.
The simulator has no interview date and treats a wave as one point in time
(SIMULATOR_SPEC 6.2), so this is the closest honest mapping rather than an
exact one.

Usage::

    uv run python scripts/build_psid_tensor.py --out data/processed/psid_x.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE

WAVES = [2011, 2013, 2015, 2017, 2019, 2021, 2023]
REF_YEAR = {y: y - 1 for y in WAVES}

#: CPI-U (CPIAUCSL) annual averages. Frozen as literals for the same reason
#: laibson_calibration.py freezes its 126 values: the 653 MB replication
#: package must not be a runtime dependency. Re-derive with
#: `pandas.read_excel(".../PSID/data/CPIAUCSL.xls", skiprows=10)`, group by
#: year, take the mean.
CPI = {
    2010: 218.076, 2011: 224.923, 2012: 229.586, 2013: 232.952,
    2014: 236.715, 2015: 237.002, 2016: 240.005, 2017: 245.121,
    2018: 251.100, 2019: 255.652, 2020: 258.851, 2021: 270.971,
    2022: 292.612, 2023: 302.628,
}
CPI_BASE = CPI[2010]


def deflator(year: int) -> float:
    """Multiplier taking nominal `year` dollars to 2010 dollars."""
    return CPI_BASE / CPI[year]


# --- PSID variable maps (see PSID_DATA.md) ---------------------------------
IV = dict(zip(WAVES, ["ER34101", "ER34201", "ER34301", "ER34501",
                      "ER34701", "ER34901", "ER35101"]))
REL = dict(zip(WAVES, ["ER34103", "ER34203", "ER34303", "ER34503",
                       "ER34703", "ER34903", "ER35103"]))
AGE = dict(zip(WAVES, ["ER47317", "ER53017", "ER60017", "ER66017",
                       "ER72017", "ER78017", "ER82018"]))
FAM = {
    "chk":    ["ER52350", "ER58161", "ER65358", "ER71435", "ER77457", "ER81784", "ER85638"],
    # CD/bonds/T-bills were inside W28 before 2019 and split out as W28B after;
    # adding them back for 2019+ keeps one definition across the panel.
    "cd":     [None, None, None, None, "ER77461", "ER81788", "ER85642"],
    "cc":     ["ER52372", "ER58185", "ER65382", "ER71459", "ER77485", "ER81812", "ER85666"],
    "stocks": ["ER52358", "ER58171", "ER65368", "ER71445", "ER77471", "ER81798", "ER85652"],
    "ira":    ["ER52368", "ER58181", "ER65378", "ER71455", "ER77481", "ER81808", "ER85662"],
    "veh":    ["ER52360", "ER58173", "ER65370", "ER71447", "ER77473", "ER81800", "ER85654"],
    "oth":    ["ER52364", "ER58177", "ER65374", "ER71451", "ER77477", "ER81804", "ER85658"],
    # 2011 reports other-real-estate and farm/business as COMBINED NET values
    # (W2, W11); from 2013 they split into asset (W2A/W11A) and debt
    # (W2B/W11B). Using the 2011 codes as if they were gross assets -- as an
    # earlier version did -- silently mixes net and gross across waves.
    "re_a":   ["ER52354", "ER58165", "ER65362", "ER71439", "ER77465", "ER81792", "ER85646"],
    "re_b":   [None,      "ER58167", "ER65364", "ER71441", "ER77467", "ER81794", "ER85648"],
    "fb_a":   ["ER52346", "ER58155", "ER65352", "ER71429", "ER77451", "ER81778", "ER85632"],
    "fb_b":   [None,      "ER58157", "ER65354", "ER71431", "ER77453", "ER81780", "ER85634"],
    # Unsecured non-revolving debts. Laibson et al.'s SCF wealth is net of the
    # equivalents (`other_l = (INSTALL-VEH_INST) + ODEBT + OTHLOC`), so a
    # comparable measure must net them too. Student loans dominate: 29.5%
    # incidence, 15,000 median among holders. They are NOT part of `X`, which
    # is specifically the revolving credit-card margin at R_CC.
    "stud":   ["ER52376", "ER58189", "ER65386", "ER71463", "ER77489", "ER81816", "ER85670"],
    "med":    ["ER52380", "ER58193", "ER65390", "ER71467", "ER77493", "ER81820", "ER85674"],
    "legal":  ["ER52384", "ER58197", "ER65394", "ER71471", "ER77497", "ER81824", "ER85678"],
    "famloan":["ER52388", "ER58201", "ER65398", "ER71475", "ER77501", "ER81828", "ER85682"],
    "othdebt":[None,      "ER58205", "ER65402", "ER71479", "ER77505", "ER81832", "ER85686"],
    "w1":     ["ER52392", "ER58209", "ER65406", "ER71483", "ER77509", "ER81836", "ER85690"],
    "w2":     ["ER52394", "ER58211", "ER65408", "ER71485", "ER77511", "ER81838", "ER85692"],
    "totexp": ["ER52395E4", "ER58212E4", "ER65448B", "ER71527B", "ER77587", "ER81914", "ER85768"],
    "mort":   ["ER52395A6", "ER58212A6", "ER65415", "ER71492", "ER77521", "ER81848", "ER85702"],
    "ptax":   ["ER52395A8", "ER58212A8", "ER65417", "ER71495", "ER77527", "ER81854", "ER85708"],
    "hins":   ["ER52395A9", "ER58212A9", "ER65418", "ER71496", "ER77529", "ER81856", "ER85710"],
    "vdown":  ["ER52395B9", "ER58212B9", "ER65427", "ER71505", "ER77542", "ER81869", "ER85723"],
    "vloan":  ["ER52395B8", "ER58212B8", "ER65426", "ER71504", "ER77540", "ER81867", "ER85721"],
}
FAM = {k: dict(zip(WAVES, v)) for k, v in FAM.items()}

MISSING_FROM = 9_999_998
HEAD_CODE = 10

# --- Laibson et al.'s own sample filter (3_scfanalysis_withCondMoments_bs.do) -
# Their entire first-stage calibration -- income profile, credit limits, initial
# wealth -- is estimated on `comphs` households only. Applying a
# high-school-completer income process to a sample containing college graduates
# guarantees the wrong lifetime income path, and therefore the wrong
# consumption and wealth. These are not nuisance controls.
#: `COMPLETED ED-HD` through 2015, relabelled `COMPLETED ED-RP` from 2017.
ED_HD = dict(zip(WAVES, ["ER52405", "ER58223", "ER65459", "ER71538",
                         "ER77599", "ER81926", "ER85780"]))
#: `BC23 CORP/UNCORP BUS--JOB 1`: 2 = works for self, 3 = both.
SELFEMP_HD = dict(zip(WAVES, ["ER47483", "ER53183", "ER60198", "ER66199",
                              "ER72199", "ER78204", "ER82187"]))
FARM_INC = dict(zip(WAVES, ["ER52214", "ER58015", "ER65195", "ER71272",
                            "ER77294", "ER81621", "ER85475"]))
BUS_ASSET = dict(zip(WAVES, ["ER52217", "ER58018", "ER65198", "ER71275",
                             "ER77297", "ER81624", "ER85478"]))
#: `W38A WTR HAVE CREDIT/STORE CARD DEBT`. NOT card possession -- see
#: --require_card, which is off by default for that reason.
HAS_CCDEBT = dict(zip(WAVES, ["ER48936", "ER54686", "ER61797", "ER67851",
                              "ER73879", "ER80001", "ER83971"]))
#: Section P pension variables: the *stock* in defined-contribution accounts,
#: which the Section W wealth module does not carry. P20 is the current job's
#: balance, P49 the balance still held at up to two previous employers, each
#: asked of head and spouse. PSID's standard wealth summary excludes all of
#: this, which is the documented reason its wealth sits below the SCF's
#: (RESULTS.md 17.5); the model's illiquid state carries a liquidation penalty
#: precisely to represent such accounts, so omitting them is a mis-measurement
#: of exactly the quantity being estimated on.
#:
#: DK/refused sentinels. The field width is not constant across waves: 2011-13
#: use the 9-digit 999999998/9, while 2015 onward also carry the 8-digit
#: 99999998/9. A 9-digit-only threshold lets the 8-digit ones through, where
#: they survive as ~1e8 and, once summed with a real balance, as values like
#: 100,011,998 that no longer look like sentinels at all. The p99 of genuine
#: balances is 843,200, so 99,999,997 is far above any real account.
PENSION_DK = 99_999_997
PENSION = {
    2011: ['ER49080', 'ER49299', 'ER49122', 'ER49202', 'ER49341', 'ER49421'],
    2013: ['ER54836', 'ER55052', 'ER54876', 'ER54956', 'ER55092', 'ER55172'],
    2015: ['ER61956', 'ER62173', 'ER61997', 'ER62077', 'ER62214', 'ER62294'],
    2017: ['ER68010', 'ER68227', 'ER68051', 'ER68131', 'ER68268', 'ER68348'],
    2019: ['ER74036', 'ER74243', 'ER74073', 'ER74151', 'ER74280', 'ER74358'],
    2021: ['ER80159', 'ER80365', 'ER80195', 'ER80273', 'ER80401', 'ER80479'],
    2023: ['ER84129', 'ER84335', 'ER84165', 'ER84243', 'ER84371', 'ER84449'],
}

SELFEMP_CODES = (2, 3)
COMPHS_LO, COMPHS_HI = 12, 16     # SCF EDCL 2-3: HS degree or some college

#: Their three education groups, from `2_buildmoments.do:69-74`:
#:   somehs  EDCL 1    no high school degree nor equivalent
#:   comphs  EDCL 2-3  high school degree, or some college
#:   compco  EDCL 4    college degree
#: PSID reports years completed rather than a category, so the cut points are
#: the year equivalents of those categories. Order matches
#: ``laibson_calibration.EDUC_GROUPS`` -- the index is what gets stored, so the
#: two orderings must not drift apart.
EDUC_GROUPS = ("comphs", "somehs", "compco")


def educ_index(years: np.ndarray) -> np.ndarray:
    """Group index per household; -1 where education is unusable."""
    out = np.full(len(years), -1, dtype=np.int64)
    ok = np.isfinite(years)
    out[ok & (years < COMPHS_LO)] = EDUC_GROUPS.index("somehs")
    out[ok & (years >= COMPHS_LO) & (years < COMPHS_HI)] = EDUC_GROUPS.index("comphs")
    out[ok & (years >= COMPHS_HI)] = EDUC_GROUPS.index("compco")
    return out


def col(fam: pd.DataFrame, key: str, wave: int) -> pd.Series:
    v = FAM[key][wave]
    if v is None:
        return pd.Series(0.0, index=fam.index)
    return fam[v].where(fam[v] < MISSING_FROM, np.nan)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--psid_dir", type=Path, default=Path("PSID-data"))
    p.add_argument("--atincome", type=Path,
                   default=Path("data/processed/psid_atincome.csv"))
    p.add_argument("--out", type=Path, default=Path("data/processed/psid_x.pt"))
    p.add_argument("--age_low", type=int, default=25)
    p.add_argument("--age_high", type=int, default=46)
    p.add_argument("--match_laibson", action="store_true",
                   help="Apply their sample filter: comphs education, not "
                        "self-employed, no business or farm income. Cuts 2119 "
                        "to ~889. Required for the calibration to be the right "
                        "one for the sample.")
    p.add_argument("--with_pension", action="store_true",
                   help="Add defined-contribution pension balances (Section P) "
                        "to illiquid wealth. Needs PSID-data/pension.pkl from "
                        "the J365419 extract. Off by default so existing "
                        "tensors stay reproducible.")
    p.add_argument("--educ_groups", action="store_true",
                   help="Keep all three education groups and record each "
                        "household's group index, instead of restricting to "
                        "comphs. Applies the rest of their sample filter (not "
                        "self-employed, no business or farm income), which is "
                        "group-independent. Needed for the per-group and "
                        "conditioned posteriors; the comphs subset of this is "
                        "exactly what --match_laibson produces.")
    p.add_argument("--require_card", action="store_true",
                   help="Additionally require card debt in at least one wave, "
                        "as a proxy for their hasVisa. PSID never asks about "
                        "card POSSESSION, so this is a proxy and a biased one: "
                        "it drops convenience users, who are systematically the "
                        "patient households, and so selects on the very outcome "
                        "that identifies beta. 'Ever in 7 waves' (66.6%) is far "
                        "closer to the ~70%% US holding rate than any single "
                        "wave (36%%), but report it as a bound, not a fix.")
    p.add_argument("--compare", type=Path,
                   default=Path("data/processed/phase3_dataset_shards"),
                   help="Shards to compare the empirical distribution against.")
    args = p.parse_args()

    ind = pd.read_pickle(args.psid_dir / "extract.pkl")
    fam = pd.read_pickle(args.psid_dir / "tax.pkl")
    pen = (pd.read_pickle(args.psid_dir / "pension.pkl")
           if args.with_pension else None)
    if args.with_pension:
        missing = [c for y in WAVES for c in PENSION[y] if c not in pen.columns]
        if missing:
            raise SystemExit(f"pension.pkl lacks {len(missing)} codes, e.g. "
                             f"{missing[:3]}; rebuild it from J365419")
    at = pd.read_csv(args.atincome)

    keep = np.logical_and.reduce([(ind[IV[y]] > 0) & (ind[REL[y]] == HEAD_CODE)
                                  for y in WAVES])
    keep &= ind[AGE[WAVES[0]]].between(args.age_low, args.age_high)
    keep &= (ind[AGE[WAVES[-1]]] - ind[AGE[WAVES[0]]]).between(11, 13)
    keep = keep.to_numpy()
    print(f"head-in-all-7 cohort: {int(keep.sum())}")

    if args.match_laibson or args.require_card or args.educ_groups:
        edu = pd.read_pickle(args.psid_dir / "edu.pkl")
        key = ["ER30001", "ER30002"]
        mm = (ind.loc[keep, key].reset_index(drop=True)
              .join(edu.drop_duplicates(subset=key).set_index(key), on=key))
        sub = np.ones(len(mm), bool)

        def _col(df, c):
            return df[c].where(df[c] < MISSING_FROM, np.nan).to_numpy()

        if args.match_laibson or args.educ_groups:
            v = mm[[ED_HD[y] for y in WAVES]].to_numpy()
            v = np.where((v > 0) & (v <= 17), v, np.nan)
            with np.errstate(invalid="ignore"):
                ed = np.nanmedian(v, axis=1)     # education is time-invariant
            if args.educ_groups:
                # Keep every group, but still require usable education: a
                # household we cannot classify cannot be conditioned on.
                sub &= np.isfinite(ed)
                print(f"  + usable education: {int(sub.sum())}")
            else:
                sub &= (ed >= COMPHS_LO) & (ed < COMPHS_HI)
                print(f"  + comphs education ({COMPHS_LO}-{COMPHS_HI - 1} yrs): "
                      f"{int(sub.sum())}")

            se = np.column_stack([_col(fam.loc[keep], SELFEMP_HD[y])
                                  for y in WAVES])
            sub &= ~np.isin(se, SELFEMP_CODES).any(axis=1)
            print(f"  + not self-employed: {int(sub.sum())}")

            biz = ((np.column_stack([_col(fam.loc[keep], FARM_INC[y])
                                     for y in WAVES]) != 0)
                   | (np.column_stack([_col(fam.loc[keep], BUS_ASSET[y])
                                       for y in WAVES]) != 0)).any(axis=1)
            sub &= ~biz
            print(f"  + no business/farm income: {int(sub.sum())}")

        if args.require_card:
            card = mm[[HAS_CCDEBT[y] for y in WAVES]].to_numpy() == 1
            sub &= card.any(axis=1)
            print(f"  + ever had card debt (biased hasVisa proxy): "
                  f"{int(sub.sum())}")

        full = np.zeros(len(ind), bool)
        full[np.flatnonzero(keep)[sub]] = True
        educ_all = np.full(len(ind), -1, dtype=np.int64)
        if args.match_laibson or args.educ_groups:
            educ_all[np.flatnonzero(keep)] = educ_index(ed)
        keep = full

    idx = np.flatnonzero(keep)

    at_w = {y: at[at.wave == y].set_index("hh")["atincome"] for y in WAVES}

    n_feat = len(FEATURES_TWOASSET_AGE)
    x = np.full((len(idx), len(WAVES), n_feat), np.nan, dtype=np.float64)
    for w, y in enumerate(WAVES):
        d_flow = deflator(REF_YEAR[y])   # income, consumption: calendar Y-1
        d_stock = deflator(y)            # wealth: measured at the interview
        f = fam.loc[idx]

        inc = at_w[y].reindex(idx).to_numpy() * d_flow

        cons = (col(fam, "totexp", y).loc[idx].fillna(0)
                - col(fam, "mort", y).loc[idx].fillna(0)
                - col(fam, "ptax", y).loc[idx].fillna(0)
                - col(fam, "hins", y).loc[idx].fillna(0)
                - col(fam, "vdown", y).loc[idx].fillna(0)
                - col(fam, "vloan", y).loc[idx].fillna(0)).to_numpy() * d_flow

        liq = (col(fam, "chk", y).loc[idx].fillna(0)
               + col(fam, "cd", y).loc[idx].fillna(0)
               - col(fam, "cc", y).loc[idx].fillna(0)).to_numpy() * d_stock

        home = (col(fam, "w2", y).loc[idx] - col(fam, "w1", y).loc[idx]).fillna(0)
        assets = sum(col(fam, k, y).loc[idx].fillna(0)
                     for k in ("stocks", "ira", "veh", "oth", "re_a", "fb_a"))
        # Net of every debt PSID reports that is not the revolving card
        # balance. Home equity is already net (w2 - w1). Real-estate and
        # farm/business debts are only separate from 2013; in 2011 the asset
        # legs are already net, so subtracting nothing there is correct.
        debts = sum(col(fam, k, y).loc[idx].fillna(0)
                    for k in ("re_b", "fb_b", "stud", "med", "legal",
                              "famloan", "othdebt"))
        ill = (assets + home - debts).to_numpy() * d_stock
        # The model requires Z >= 0. Netting pushes ~29% of household-waves to
        # the floor, which is a real feature of the data -- those households
        # have more non-housing debt than illiquid assets -- not a clipping
        # artifact to hide.
        if args.with_pension:
            # Added BEFORE the floor: DC balances are strictly non-negative, so
            # they legitimately offset other debt rather than being clipped
            # away. DK/refused is treated as zero, which understates slightly --
            # 469 of 37,338 cells on the matched sample.
            pens = sum(
                np.where(
                    np.isfinite(pen[c].to_numpy(dtype=float)[idx])
                    & (pen[c].to_numpy(dtype=float)[idx] < PENSION_DK),
                    pen[c].to_numpy(dtype=float)[idx], 0.0)
                for c in PENSION[y])
            ill = ill + pens * d_stock
        n_floor = int((ill < 0).sum())
        ill = np.maximum(ill, 0.0)
        if w == 0:
            print(f"  net illiquid floored at 0 for {n_floor} of {len(ill)} "
                  f"households in {y} ({n_floor / len(ill):.1%})")

        x[:, w, 0] = inc
        x[:, w, 1] = cons
        x[:, w, 2] = liq
        x[:, w, 3] = ill
        x[:, w, 4] = ind[AGE[y]].to_numpy()[idx]

    ok = np.isfinite(x).all(axis=(1, 2))
    print(f"complete on all waves and features: {int(ok.sum())}")
    x, idx = x[ok], idx[ok]

    payload = {"x": torch.from_numpy(x).float(),
               "psid_row": torch.from_numpy(idx).long(),
               "waves": WAVES,
               "features": list(FEATURES_TWOASSET_AGE),
               "units": "2010 USD (CPI-U, base 2010; simulator's units)"}
    if args.match_laibson or args.educ_groups:
        e = educ_all[idx]
        payload |= {"educ": torch.from_numpy(e).long(),
                    "educ_groups": list(EDUC_GROUPS)}
    torch.save(payload, args.out)
    print(f"wrote {args.out}: x {tuple(x.shape)}")
    if "educ" in payload:
        for i, g in enumerate(EDUC_GROUPS):
            print(f"  {g:8s} {int((e == i).sum()):5d}")
        if (e < 0).any():
            print(f"  unclassified {int((e < 0).sum())}")

    # --- the check that decides whether any of this is usable --------------
    # An amortized posterior is only valid on x inside its training support.
    # If the empirical features sit outside the simulated ones, the posterior
    # is extrapolating and its output is not trustworthy however well it
    # calibrated on simulated data.
    import glob
    shards = sorted(glob.glob(str(args.compare / "shard_*.npz")))[:8]
    sim = {k: [] for k in ("income", "consumption", "liquid_assets",
                           "illiquid_assets")}
    for s in shards:
        z = np.load(s)
        for k in sim:
            # ages 25-59, the observation envelope
            sim[k].append(z["panel_" + k][:, 5:40].ravel())
    sim = {k: np.concatenate(v) for k, v in sim.items()}

    print(f"\n{'feature':17s}{'source':10s}" +
          "".join(f"{q:>10s}" for q in ("p10", "p25", "p50", "p75", "p90")))
    qs = [10, 25, 50, 75, 90]
    for j, (name, simkey) in enumerate(zip(
            FEATURES_TWOASSET_AGE[:4],
            ("income", "consumption", "liquid_assets", "illiquid_assets"))):
        e = np.percentile(x[:, :, j].ravel(), qs)
        s = np.percentile(sim[simkey], qs)
        print(f"{name:17s}{'PSID':10s}" + "".join(f"{v:10.0f}" for v in e))
        print(f"{'':17s}{'simulated':10s}" + "".join(f"{v:10.0f}" for v in s))
    print("\nage      PSID "
          f"{np.percentile(x[:, :, 4].ravel(), qs).round(0)}")


if __name__ == "__main__":
    main()
