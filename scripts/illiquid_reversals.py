"""Why PSID couples' illiquid signature is positive and the model's negative
(RESULTS 57.1).

RESULTS 43.9's noise signature a = 2 V_1 - V_2 = -2 cov(dz_t, dz_{t-1}) is
positive when a change tends to be followed by its reverse. For illiquid
wealth PSID couples show +0.063 and no cell of the model reaches it without
noise. This compares the two series' changes directly:

- **Both sides:** a and V_1 on the network's input scale (asinh of illiquid
  wealth over the household's mean income, IQR-based as in RESULTS 43.9);
  the lag-1 correlation of one-wave changes, among couples holding illiquid
  wealth in all three waves the two changes span (changes winsorised at
  their 1st and 99th percentiles); and, among changes between two held
  waves, the shares falling or rising by more than 0.1 x mean income.
- **PSID, by component** (``psid_illiquid_decomposition.components``): each
  component's own lag-1 correlation, held in all three waves; and a *freeze*:
  the component replaced by its household mean over the window (its level
  kept, its changes removed), the sum re-floored at zero, and a and V_1
  recomputed. 200 bootstrap resamples of couples give 90% intervals for the
  share of a and of V_1 that each freeze removes.
- **PSID, moves:** changes spanning a change of housing tenure (a purchase or
  a sale), their share of the squared changes, and V_1 without them.
- **The model:** v2's noise-free windows in couples' region (RESULTS 43.9's
  region), and the share of its held changes that are exactly zero.

Aggregates only; no household's values are printed.

Usage::

    PYTHONPATH=.:src .venv/bin/python scripts/illiquid_reversals.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from scripts import optionA as oa
from scripts.impute_rental_value import owner_matrix
from scripts.noise_calibration import PSID_X, cells, region, robust_var, signature
from scripts.psid_illiquid_decomposition import COMPONENTS, components

INC = FEATURES_TWOASSET_AGE.index("income")
ILL = FEATURES_TWOASSET_AGE.index("illiquid_assets")
V2 = (Path("data/processed/couples_v2_dataset_shards"),
      Path("data/processed/couples_v2_strip_dataset_shards"))
OUT = Path("outputs/illiquid_reversals/illiquid_reversals.json")
#: Freezes reported besides one component at a time.
PAIRS = {"home equity + vehicles": ("home equity", "vehicles")}


def input_scale(ill: np.ndarray, ybar: np.ndarray) -> np.ndarray:
    return np.arcsinh(np.maximum(ill, 0.0) / ybar)


def lag1_corr(v: np.ndarray, ybar: np.ndarray, scale: str = "linear") -> tuple[float, int]:
    """Correlation of consecutive one-wave changes of ``v`` (N, waves) over
    mean income, where ``v`` is non-zero in all three waves they span;
    changes winsorised at their 1st and 99th percentiles."""
    r = v / ybar
    r = np.arcsinh(r) if scale == "asinh" else r
    d = np.diff(r, axis=1)
    held = (v[:, 2:] != 0) & (v[:, 1:-1] != 0) & (v[:, :-2] != 0)
    a, b = d[:, 1:][held], d[:, :-1][held]
    lo, hi = np.percentile(np.concatenate([a, b]), [1, 99])
    a, b = np.clip(a, lo, hi), np.clip(b, lo, hi)
    return float(np.corrcoef(a, b)[0, 1]), int(held.sum())


def moves(ill: np.ndarray, ybar: np.ndarray) -> dict:
    """Changes between two waves with positive illiquid wealth: shares that
    fall or rise by more than 0.1 x mean income, or are exactly zero."""
    d = np.diff(ill, axis=1) / ybar
    held = (ill[:, 1:] > 0) & (ill[:, :-1] > 0)
    d = d[held]
    return {"n": int(held.sum()), "falling_gt_0.1": float((d < -0.1).mean()),
            "rising_gt_0.1": float((d > 0.1).mean()),
            "within_0.1": float((np.abs(d) <= 0.1).mean()),
            "exactly_zero": float((d == 0).mean())}


def series_stats(ill: np.ndarray, ybar: np.ndarray) -> dict:
    a, v1, v2 = signature(input_scale(ill, ybar))
    r, n = lag1_corr(ill, ybar)
    ra, _ = lag1_corr(ill, ybar, "asinh")
    return {"a": a, "V1": v1, "V2": v2, "lag1_corr": r, "lag1_corr_asinh": ra,
            "lag1_pairs": n, "changes": moves(ill, ybar)}


def freezes(comp: dict, ybar: np.ndarray, n_boot: int, seed: int = 0) -> dict:
    """The share of a and of V_1 each freeze removes, with 90% bootstrap
    intervals (the same resamples for every freeze)."""
    total = sum(comp.values())
    mean = {k: np.broadcast_to(v.mean(1, keepdims=True), v.shape) for k, v in comp.items()}
    sets = {k: (k,) for k in COMPONENTS} | PAIRS
    z = {"none": input_scale(total, ybar)}
    for name, ks in sets.items():
        z[name] = input_scale(total - sum(comp[k] - mean[k] for k in ks), ybar)
    boot = np.random.default_rng(seed).integers(0, len(total), (n_boot, len(total)))
    sig = {name: np.array([signature(v[b])[:2] for b in boot]) for name, v in z.items()}
    a0, v0 = signature(z["none"])[:2]
    out = {}
    for name in sets:
        a, v1 = signature(z[name])[:2]
        out[name] = {"a": a, "V1": v1, "a_removed": 1 - a / a0, "V1_removed": 1 - v1 / v0,
                     "a_removed_90": np.percentile(1 - sig[name][:, 0] / sig["none"][:, 0],
                                                   [5, 95]).tolist(),
                     "V1_removed_90": np.percentile(1 - sig[name][:, 1] / sig["none"][:, 1],
                                                    [5, 95]).tolist()}
    return out


def psid(psid_dir: Path, n_boot: int) -> dict:
    d = torch.load(PSID_X, weights_only=False)
    x = d["x"].numpy().astype(np.float64)
    rows = d["psid_row"].numpy()
    fam = pd.read_pickle(psid_dir / "tax.pkl")
    comp = components(fam, pd.read_pickle(psid_dir / "pension.pkl"), rows)
    total = sum(comp.values())
    if not np.allclose(np.maximum(total, 0.0), x[..., ILL], rtol=1e-5, atol=1.0):
        raise SystemExit("components do not reproduce the tensor's illiquid wealth")
    ybar = x[..., INC].mean(1, keepdims=True)
    ill = np.maximum(total, 0.0)
    res = {"n": len(x), "illiquid": series_stats(ill, ybar), "components": {}}
    for k, v in comp.items():
        r, n = lag1_corr(v, ybar)
        res["components"][k] = {"lag1_corr": r, "lag1_pairs": n,
                                "share_household_waves_nonzero": float((v != 0).mean())}
    res["freeze"] = freezes(comp, ybar, n_boot)
    own = owner_matrix(fam, rows)
    trans = own[:, 1:] != own[:, :-1]
    dz = np.diff(input_scale(ill, ybar), axis=1)
    res["tenure"] = {"share_of_changes": float(trans.mean()),
                     "share_of_squared_changes": float((dz[trans] ** 2).sum() / (dz ** 2).sum()),
                     "V1_without": robust_var(dz[~trans])}
    return res


def model(shards, every: int) -> dict:
    theta, card, train, held, _p, cfg = oa.load_shards(list(shards))
    box = oa.box_of(cfg)
    th, x, pid = oa.windows_raw((train + held)[::every], theta, seed=oa.HELD_SEED)
    th, x = th.numpy(), x.numpy().astype(np.float64)
    cd = np.asarray(card)[pid.numpy()]
    x = x[cells(th, cd, box, region("couples", box)) >= 0]
    return {"n_windows": int(len(x)),
            "illiquid": series_stats(x[..., ILL], x[..., INC].mean(1, keepdims=True))}


def report(res: dict) -> None:
    p, m = res["psid"]["illiquid"], res["model_v2"]["illiquid"]
    print(f"\nilliquid, PSID couples ({res['psid']['n']}) against v2's noise-free windows in "
          f"couples' region ({res['model_v2']['n_windows']})")
    rows = [("a", "a"), ("V1", "V1"), ("lag-1 corr of changes", "lag1_corr"),
            ("  on the asinh scale", "lag1_corr_asinh")]
    for name, k in rows:
        print(f"  {name:34s} {p[k]:+8.4f}  {m[k]:+8.4f}")
    for k in ("falling_gt_0.1", "rising_gt_0.1", "within_0.1", "exactly_zero"):
        print(f"  held changes, {k:20s} {p['changes'][k]:8.3f}  {m['changes'][k]:8.3f}")
    print("\nPSID, each component: lag-1 corr of its changes (held in three waves)")
    for k, v in res["psid"]["components"].items():
        print(f"  {k:18s} {v['lag1_corr']:+.2f}  pairs {v['lag1_pairs']:5d}  "
              f"non-zero in {v['share_household_waves_nonzero']:.2f} of household-waves")
    print("\nPSID, frozen at the household mean: share of a and of V1 removed (90%)")
    for k, v in res["psid"]["freeze"].items():
        print(f"  {k:24s} a {v['a_removed']:+.2f} ({v['a_removed_90'][0]:+.2f} to "
              f"{v['a_removed_90'][1]:+.2f})   V1 {v['V1_removed']:+.2f} "
              f"({v['V1_removed_90'][0]:+.2f} to {v['V1_removed_90'][1]:+.2f})")
    t = res["psid"]["tenure"]
    print(f"\nPSID, changes spanning a tenure change: {t['share_of_changes']:.3f} of changes, "
          f"{t['share_of_squared_changes']:.3f} of squared changes; V1 without them "
          f"{t['V1_without']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psid_dir", type=Path, default=Path("PSID-data"))
    ap.add_argument("--shards", type=Path, nargs="+", default=list(V2))
    ap.add_argument("--every", type=int, default=2, help="Use every n-th shard.")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    res = {"psid": psid(args.psid_dir, args.n_boot), "model_v2": model(args.shards, args.every)}
    report(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
