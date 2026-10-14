"""PSID's own accuracy codes for the couples' inputs (RESULTS 49).

Next to most wealth and expenditure amounts, PSID publishes an accuracy code:
1 if PSID imputed the amount (the respondent did not give one, or gave only a
range), 0 if not. They say which values are partly synthetic, not how large a
reported value's error is. No code in our extracts covers income, home value
or mortgage separately, or the Section P pension balances.

**Which code belongs to which amount.**
- **Wealth items:** by label, within a wave. Each amount is "IMP <item>" and
  its code is "ACC <item>". PSID's own pointers ("Accuracy of ER...") carry
  typos in some waves: 2011 checking points at "ER523504", and 2015's WEALTH2
  code points at WEALTH1. So labels are used.
- **Expenditure categories:** the code is the amount's name plus "A" through
  2017 (from 2019 the next number), and its pointer names the amount; either
  is used if the amount's label is the category's. One pointer has a typo
  (2011 bus/train points at "ER52365C7"). Consumption is total expenditure
  less mortgage, property tax, home insurance and vehicle loan and down
  payments (``build_psid_tensor``), so those five categories' codes are left
  out.
- **Home equity** (wealth with equity less wealth without) has no code of its
  own. PSID's codes on the two wealth totals bound it: equity was imputed at
  least where the with-equity total was imputed and the without-equity total
  was not.

**Does imputation explain PSID's noise?** RESULTS 43.9 set the noise from
PSID couples' signature a = 2 V1 - V2 (IQR-based, on the network's input
scale). Here a and V1 are recomputed on the couples' own tensor using only
changes between two waves in which nothing in that feature was imputed. For
illiquid wealth, "nothing imputed" also requires PSID's with-equity wealth
total to be unimputed, since home equity has no code. If imputed values were
the noise, a would fall toward the model's no-noise level.

**Sample.** The 409 couples of ``data/processed/psid_x_comphs_couples.pt`` (its
``psid_row``), 7 waves 2011-2023. Only shares are printed, never a household's
values.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/psid_accuracy_codes.py
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import build_psid_tensor as b

LIQUID = ("chk", "cd", "cc")
ILLIQUID = ("stocks", "ira", "veh", "oth", "re_a", "fb_a", "re_b", "fb_b", "stud", "med",
            "legal", "famloan", "othdebt")
NOT_CONSUMPTION = ("MORTGAGE", "PROPERTY TAX", "HOME INSURANCE", "VEHICLE DOWN PAYMENT",
                   "VEHICLE LOAN PAYMENT")
TOTALS = {"w1": "wealth without equity", "w2": "wealth with equity"}


def codebook(psid_dir: Path) -> dict[str, dict]:
    """Every variable's year, label and question text, over all extracts."""
    out = {}
    for f in sorted(glob.glob(str(psid_dir / "*_codebook.xml"))):
        for v in ET.parse(f).getroot().iter("VARIABLE"):
            out.setdefault(v.findtext("NAME"), {
                "year": v.findtext("YEAR"), "label": (v.findtext("LABEL") or "").strip(),
                "q": (v.findtext("QTEXT") or "").strip()})
    return out


def wealth_codes(cb: dict) -> dict[str, str]:
    """Amount variable -> its accuracy code, by label within a year."""
    acc = {(m["year"], m["label"][4:]): n for n, m in cb.items() if m["label"].startswith("ACC ")}
    return {n: acc[(m["year"], m["label"][4:])] for n, m in cb.items()
            if m["label"].startswith("IMP ") and (m["year"], m["label"][4:]) in acc}


def expenditure_codes(cb: dict, year: int) -> list[tuple[str, str, str]]:
    """(category, amount variable, code) for one wave's consumption categories."""
    out = []
    for n, m in cb.items():
        if m["year"] != str(year) or not re.fullmatch(r"ACC .+ EXPENDITURE", m["label"]):
            continue
        cat = m["label"][4:-len(" EXPENDITURE")]
        if cat in NOT_CONSUMPTION:
            continue
        ptr = re.match(r"Accuracy of (ER\w+)", m["q"])
        cands = ([n[:-1]] if n.endswith("A") else []) + ([ptr.group(1)] if ptr else [])
        val = next((c for c in cands if cb.get(c, {}).get("label", "").startswith(cat)), None)
        if val is None:
            raise SystemExit(f"{n}: no amount variable for {cat} among {cands}")
        out.append((cat, val, n))
    return out


def iqr_var(v: np.ndarray) -> float:
    q75, q25 = np.percentile(v, [75, 25])
    return float(((q75 - q25) / 1.349) ** 2)


def signature(z: np.ndarray, clean: np.ndarray | None) -> dict:
    """a = 2 V1 - V2 over couples x waves ``z``, using only changes whose
    waves are all ``clean`` (None: every change)."""
    out = {}
    for k in (1, 2):
        d = z[:, k:] - z[:, :-k]
        ok = np.ones_like(d, bool) if clean is None else np.logical_and.reduce(
            [clean[:, j:j + d.shape[1]] for j in range(k + 1)])
        out[f"V{k}"] = iqr_var(d[ok])
        out[f"n{k}"] = int(ok.sum())
    out["a"] = 2 * out["V1"] - out["V2"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psid_dir", type=Path, default=Path("PSID-data"))
    ap.add_argument("--x", type=Path, default=Path("data/processed/psid_x_comphs_couples.pt"))
    ap.add_argument("--out", type=Path, default=Path("outputs/psid_accuracy/couples.json"))
    args = ap.parse_args()

    cb = codebook(args.psid_dir)
    wc = wealth_codes(cb)
    fam = pd.read_pickle(args.psid_dir / "tax.pkl")
    ind = pd.read_pickle(args.psid_dir / "extract.pkl")
    if len(fam) != len(ind):
        raise SystemExit("tax.pkl and extract.pkl rows differ")
    rows = torch.load(args.x, weights_only=False)["psid_row"].numpy()
    print(f"{len(rows)} couples, waves {b.WAVES}; {len(wc)} wealth codes in the codebooks")

    def column(var: str) -> np.ndarray | None:
        for d in (fam, ind):
            if var in d.columns:
                return d[var].to_numpy(dtype=float)[rows]
        return None

    res = {"n_couples": int(len(rows)), "components": {}, "features": {}, "totals": {},
           "code_values": {}}
    imputed = {f: np.zeros((len(rows), len(b.WAVES)), bool)
               for f in ("liquid", "illiquid", "consumption")}
    amount = {f: np.zeros((len(rows), len(b.WAVES))) for f in imputed}
    amount_imp = {f: np.zeros((len(rows), len(b.WAVES))) for f in imputed}
    codes_seen: dict[str, int] = {}

    def add(feature: str, key: str, w: int, val: np.ndarray, acc: np.ndarray | None) -> None:
        val = np.where(np.isfinite(val) & (np.abs(val) < b.MISSING_FROM), val, 0.0)
        amount[feature][:, w] += np.abs(val)
        ent = res["components"].setdefault(f"{feature}: {key}", {})
        if acc is None:
            ent[str(b.WAVES[w])] = None
            return
        for c, k in zip(*np.unique(acc[np.isfinite(acc)], return_counts=True), strict=True):
            codes_seen[str(int(c))] = codes_seen.get(str(int(c)), 0) + int(k)
        imp = acc == 1
        imputed[feature][:, w] |= imp
        amount_imp[feature][:, w] += np.where(imp, np.abs(val), 0.0)
        held = val != 0
        ent[str(b.WAVES[w])] = {"imputed": float(imp.mean()),
                                "imputed_among_nonzero": float(imp[held].mean()) if held.any()
                                else None, "nonzero": float(held.mean())}

    for w, y in enumerate(b.WAVES):
        for feature, keys in (("liquid", LIQUID), ("illiquid", ILLIQUID)):
            for k in keys:
                var = b.FAM[k][y]
                if var is None:
                    continue
                code = wc.get(var)
                add(feature, k, w, column(var), None if code is None else column(code))
        for cat, val, code in expenditure_codes(cb, y):
            add("consumption", cat.lower(), w, column(val), column(code))
        for k, name in TOTALS.items():
            code = wc.get(b.FAM[k][y])
            acc = column(code) if code else None
            res["totals"].setdefault(name, {})[str(y)] = (
                None if acc is None else float((acc == 1).mean()))
    res["code_values"] = codes_seen

    w1 = np.array([[np.nan] * len(rows)] * len(b.WAVES)).T
    w2 = w1.copy()
    for w, y in enumerate(b.WAVES):
        for k, arr in (("w1", w1), ("w2", w2)):
            code = wc.get(b.FAM[k][y])
            if code:
                arr[:, w] = column(code)
    equity_lb = (w2 == 1) & (w1 != 1)
    res["totals"]["home equity imputed, lower bound"] = float(equity_lb.mean())

    print(f"\naccuracy code values seen: {codes_seen}")
    print("\nshare of couple-waves with any imputed component, and imputed share of the "
          "feature's absolute amount:")
    print(f"  {'feature':12s}" + "".join(f"{y:>8d}" for y in b.WAVES)
          + f"{'all':>8s}{'amount':>9s}{'couples ever':>14s}")
    for f in imputed:
        by_wave = imputed[f].mean(0)
        share_amt = amount_imp[f].sum() / amount[f].sum()
        ever = imputed[f].any(1).mean()
        res["features"][f] = {"by_wave": by_wave.tolist(), "all": float(imputed[f].mean()),
                              "amount_share": float(share_amt), "couples_ever": float(ever)}
        print(f"  {f:12s}" + "".join(f"{v:8.3f}" for v in by_wave)
              + f"{imputed[f].mean():8.3f}{share_amt:9.3f}{ever:14.3f}")
    print("  income: no accuracy codes in the extracts; pensions (Section P): none")

    print("\nPSID's wealth totals, share imputed by wave:")
    for name, d in res["totals"].items():
        if isinstance(d, dict):
            print(f"  {name:22s}" + "".join("    -   " if v is None else f"{v:8.3f}"
                                           for v in d.values()))
    print(f"  home equity imputed (lower bound, all waves): {equity_lb.mean():.3f}")

    print("\ncomponents, share imputed pooled over waves with a code (among non-zero):")
    for key, d in res["components"].items():
        v = [e for e in d.values() if e]
        if not v:
            print(f"  {key:40s} no code")
            continue
        imp = np.mean([e["imputed"] for e in v])
        nz = [e["imputed_among_nonzero"] for e in v if e["imputed_among_nonzero"] is not None]
        print(f"  {key:40s} {imp:.3f}" + (f"  ({np.mean(nz):.3f})" if nz else "")
              + f"  waves with a code {len(v)} of {len(d)}")
    from scripts import optionA as oa
    xt, feats = oa.transform(torch.load(args.x, weights_only=False)["x"])
    xt = xt.numpy()
    print("\nnoise signature on the network's scale: every change / only changes between "
          "unimputed waves")
    print(f"  {'feature':12s}{'a':>9s}{'V1':>9s}{'n1':>7s}   |{'a':>9s}{'V1':>9s}{'n1':>7s}")
    res["signature"] = {}
    for f, name in (("consumption", "consumption"), ("liquid", "liquid_assets"),
                    ("illiquid", "illiquid_assets")):
        z = xt[:, :, feats.index(name)]
        clean = ~imputed[f] & ((w2 != 1) if f == "illiquid" else True)
        s_all, s_clean = signature(z, None), signature(z, clean)
        res["signature"][f] = {"all": s_all, "unimputed": s_clean}
        print(f"  {f:12s}{s_all['a']:9.4f}{s_all['V1']:9.4f}{s_all['n1']:7d}   |"
              f"{s_clean['a']:9.4f}{s_clean['V1']:9.4f}{s_clean['n1']:7d}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
