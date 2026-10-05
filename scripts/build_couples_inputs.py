"""The two PSID-derived inputs for the couples regeneration (RESULTS 42-43).

1. **Couples tensor.** The model is a 2-adult household for life, so the
   primary PSID sample is couples: legally married (A3 = 1) in at least
   ``--min_married`` of the 7 waves. Built as a row subset of the option A
   tensor, so ``x`` is byte-identical to it for every kept household -- same
   gross liquid wealth, DC pensions and imputed rent. ``married_waves`` is
   stored so the stricter all-7 set (310 households) is a filter at inference.

2. **Seed pool.** Replaces Laibson et al.'s single age-20 seed (1.47 x mean
   income, an artefact of their two-head adjustment: RESULTS 42.2) with the
   joint (liquid, illiquid) ratios to model mean income of PSID married comphs
   heads aged 20-24, under their sample filter. One row per person, at the
   youngest age observed, so nobody counts twice. Liquid is the gross
   definition the model's X means, **floored at 0**: no household starts life
   in card debt. Illiquid includes DC pensions, net of non-card debt, floored
   at 0 as in the PSID input. Nothing is estimated: generation draws
   households from this pool with replacement.

Both files are PSID-derived and live in ``data/processed`` (never committed).

Usage::

    PYTHONPATH=. .venv/bin/python scripts/build_couples_inputs.py
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import build_psid_tensor as b
from scripts.check_calibration_psid import young_heads

SRC = Path("data/processed/psid_x_comphs_optionA.pt")
TENSOR_OUT = Path("data/processed/psid_x_comphs_couples.pt")
POOL_OUT = Path("data/processed/seed_pool_couples.npz")
MIN_PERSONS = 200


def pool_hash(pool: np.ndarray) -> str:
    """What generation records, so a resume with a different pool is refused."""
    return hashlib.sha256(np.ascontiguousarray(pool, dtype=np.float64).tobytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min_married", type=int, default=4)
    p.add_argument("--age_low", type=int, default=20)
    p.add_argument("--age_high", type=int, default=24)
    args = p.parse_args()

    ind = pd.read_pickle("PSID-data/extract.pkl")
    fam = pd.read_pickle("PSID-data/tax.pkl")

    # --- couples tensor ---------------------------------------------------------
    src = torch.load(SRC, weights_only=False)
    row = src["psid_row"].numpy()
    married = fam.loc[row, [b.MARITAL[y] for y in b.WAVES]].to_numpy() == 1
    mw = married.sum(axis=1)
    keep = mw >= args.min_married
    out = {k: (v[keep] if torch.is_tensor(v) and len(v) == len(row) else v)
           for k, v in src.items()}
    out["married_waves"] = torch.from_numpy(mw[keep]).long()
    out["couple_rule"] = f"A3 married in >= {args.min_married} of 7 waves"
    out["note"] = (src.get("note", "") + " Couples subset (RESULTS 42): rows of "
                   f"{SRC.name} with {out['couple_rule']}; x unchanged.")
    torch.save(out, TENSOR_OUT)
    print(f"couples tensor: {int(keep.sum())} of {len(row)} households "
          f"({int((mw == 7).sum())} married in all 7) -> {TENSOR_OUT}")

    # --- seed pool --------------------------------------------------------------
    edu = pd.read_pickle("PSID-data/edu.pkl")
    pen = pd.read_pickle("PSID-data/pension.pkl")
    yh = young_heads(fam, ind, edu, pen, args.age_low, args.age_high)
    yh = yh[yh.married].sort_values(["person", "age", "wave"])
    first = yh.groupby("person", sort=False).head(1)
    if len(first) < MIN_PERSONS:
        raise SystemExit(f"only {len(first)} married heads aged {args.age_low}-"
                         f"{args.age_high}; widen the age range (plan: 20-26)")
    pool = np.column_stack([np.maximum(first.liq_gross.to_numpy(), 0.0),
                            np.maximum(first.illiquid.to_numpy(), 0.0)])
    assert (pool >= 0).all() and np.isfinite(pool).all()
    np.savez(POOL_OUT, pool=pool, age=first.age.to_numpy(), wave=first.wave.to_numpy(),
             source=(f"PSID 2011-2023 married (A3) comphs heads aged {args.age_low}-"
                     f"{args.age_high}, not self-employed, no business/farm income; "
                     "one row per person (youngest wave); (gross liquid floored at 0, "
                     "illiquid incl. DC pensions net of non-card debt floored at 0) "
                     "over model mean income at own age"),
             sha256=pool_hash(pool))
    q = np.quantile(pool, [0.1, 0.25, 0.5, 0.75, 0.9], axis=0)
    print(f"seed pool: {len(pool)} persons -> {POOL_OUT}  sha256 {pool_hash(pool)[:12]}")
    for j, name in enumerate(("liquid", "illiquid")):
        print(f"   {name:8s} p10/25/50/75/90 " + " ".join(f"{v:6.3f}" for v in q[:, j])
              + f"   zero {np.mean(pool[:, j] == 0):.2f}  max {pool[:, j].max():.2f}")


if __name__ == "__main__":
    main()
