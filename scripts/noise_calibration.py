"""Per-feature observation noise: the minimum PSID requires (RESULTS 43.7).

Classical measurement error -- iid across waves, multiplicative on the dollar
value, ``v * exp(sigma * z)`` -- adds the same amount to the variance of a
change over any number of waves, while genuine dynamics add more the longer
the gap. Its signature is therefore the first-order autocovariance of changes:
with ``V_k`` the variance of k-wave changes,

    a = 2 V_1 - V_2 = -2 cov(dz_t, dz_{t-1}),

which noise raises by 2 sigma^2 on a log scale and true persistent shocks
leave alone. Everything here is on the network's own input scale (log of
income and consumption over the household's mean income, asinh of liquid and
illiquid wealth over it), and ``V_k`` is IQR-based, ((p75 - p25) / 1.349)^2,
so a few extreme changes (PSID's income changes have kurtosis 36) do not set
the noise for every household.

The model has transitory variation of its own, more at some theta than at
others. The **minimum required noise** for a feature is the smallest sigma at
which some cell (rho bins x halves of beta, delta, R_gamma x card) of a region reaches PSID
couples' signature, found by adding the noise to the cell's simulated windows.
Below it, no theta in the region can produce what PSID shows; a feature whose
signature the region already reaches gets zero. The same rule for every
feature; age is exact and never perturbed.

The **typical** requirement is the median of the cells' own requirements
over the region under the prior. Training draws each household's sigma
uniformly on [minimum, typical] per feature, so the network marginalises over
how noisy a household's data are (RESULTS 43.7: the minimum alone cannot tell
noise from low rho, since rho < 1 reproduces PSID's consumption signature by
itself).

Two regions, both reported:

- ``box``: the whole prior box. Its corners of very impatient households
  without a card (delta < 0.90, beta < 0.53) track transitory income so
  closely that they reach PSID's consumption signature with no noise, which
  says nothing about couples, who are nowhere near them.
- ``couples`` (applied): the training proposal's concentrated region,
  beta >= 0.40 and delta >= 0.95 (``EDGE_A2``, set from the pilot in RESULTS
  43.2, where 87-88% of couples' posterior means lie), with rho, R_gamma and
  card type free, so the rule does not presuppose rho.

**Masked PSID** (``--mask_tau``, RESULTS 50). PSID's signature uses only
changes between waves that are not masked: a feature-wave is masked where PSID
imputed some part of it and the imputed share of its amount is at least tau
(``psid_accuracy_codes.masks``). The model side is unchanged.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/noise_calibration.py
    PYTHONPATH=. .venv/bin/python scripts/noise_calibration.py --mask_tau 0.25 \
        --out outputs/optionA2m/noise_calibration_tau0.25.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from hh_npe.npe.prior import EDGE_A2
from scripts import optionA as oa

MAIN = Path("data/processed/couples_dataset_shards")
STRIP = Path("data/processed/couples_strip_dataset_shards")
PSID_X = Path("data/processed/psid_x_comphs_couples.pt")
OUT = Path("outputs/optionA2n/noise_calibration.json")
INC = FEATURES_TWOASSET_AGE.index("income")
RHO_EDGES = (0.5, 0.75, 1.0, 1.5, 2.5, 3.5, 5.0)
SIGMA_GRID = np.round(np.arange(0.0, 2.0001, 0.005), 3)
MIN_CELL = 400
#: Splits of beta, delta and R_gamma within a region. Halves, not thirds: with
#: thirds, couples' region leaves too few windows in the rho < 1 cells.
SPLIT = 2


def input_scale(x: np.ndarray, f: str) -> np.ndarray:
    """One dollar feature of raw (N, waves, 5) windows on the network's scale
    (``compare_windows.anchor_log``)."""
    ybar = x[..., INC].mean(1, keepdims=True)
    v = x[..., FEATURES_TWOASSET_AGE.index(f)] / ybar
    if f in ("income", "consumption"):
        return np.log(np.maximum(v, 0.01))
    return np.arcsinh(v)


def robust_var(v: np.ndarray) -> float:
    q25, q75 = np.percentile(v, [25, 75])
    return ((q75 - q25) / 1.349) ** 2


def signature(z: np.ndarray, clean: np.ndarray | None = None) -> tuple[float, float, float]:
    """(a, V_1, V_2) of an (N, waves) series on the input scale, using only
    changes whose waves are all ``clean`` (None: every change)."""
    v = []
    for k in (1, 2):
        d = z[:, k:] - z[:, :-k]
        ok = (np.ones_like(d, bool) if clean is None else
              np.logical_and.reduce([clean[:, j:j + d.shape[1]] for j in range(k + 1)]))
        v.append(robust_var(d[ok]))
    return 2 * v[0] - v[1], v[0], v[1]


def region(name: str, box) -> dict:
    """(low, high) of beta, delta and R_gamma for a region (module docstring)."""
    names = list(box.names)
    r = {k: (box.low[names.index(k)], box.high[names.index(k)])
         for k in ("beta", "delta", "R_gamma")}
    if name == "couples":
        r["beta"] = (EDGE_A2["beta_lo"], r["beta"][1])
        r["delta"] = (1 - EDGE_A2["one_minus_delta_hi"], r["delta"][1])
    return r


def cells(th: np.ndarray, card: np.ndarray, box, reg: dict) -> np.ndarray:
    """Cell index per window, -1 outside the region: rho bins x ``SPLIT``
    parts of beta, delta, R_gamma within the region x card."""
    names = list(box.names)
    idx = np.digitize(th[:, names.index("crra")], RHO_EDGES[1:-1])
    inside = np.ones(len(th), bool)
    for k in ("beta", "delta", "R_gamma"):
        j = names.index(k)
        lo, hi = reg[k]
        inside &= (th[:, j] >= lo) & (th[:, j] <= hi)
        idx = idx * SPLIT + np.digitize(th[:, j], np.linspace(lo, hi, SPLIT + 1)[1:-1])
    return np.where(inside, idx * 2 + card.astype(int), -1)


def curve(x: np.ndarray, f: str, rng: np.random.Generator, stop: float) -> np.ndarray:
    """a(sigma) on ``SIGMA_GRID`` for windows ``x`` with noise on feature ``f``
    only (common random numbers across sigma). Past the first sigma whose a
    reaches ``stop`` the curve is not computed and reads +inf."""
    j = FEATURES_TWOASSET_AGE.index(f)
    z = rng.standard_normal(x.shape[:2])
    v = x[..., j]
    out = np.full(len(SIGMA_GRID), np.inf)
    for i, s in enumerate(SIGMA_GRID):
        if f == "income":
            xn = x.copy()
            xn[..., j] = v * np.exp(s * z)
            zs = input_scale(xn, f)
        else:
            ybar = x[..., INC].mean(1, keepdims=True)
            r = v * np.exp(s * z) / ybar
            zs = np.log(np.maximum(r, 0.01)) if f == "consumption" else np.arcsinh(r)
        out[i] = signature(zs)[0]
        if out[i] >= stop:
            break
    return out


def sigma_min(curves: dict, target: float) -> tuple[float, int]:
    """Smallest sigma at which some cell's a(sigma) reaches ``target``."""
    best, arg = np.inf, -1
    for c, a in curves.items():
        hit = np.flatnonzero(a >= target)
        if len(hit) and SIGMA_GRID[hit[0]] < best:
            best, arg = float(SIGMA_GRID[hit[0]]), c
    return best, arg


def describe_cell(c: int, reg: dict) -> str:
    card, c = c % 2, c // 2
    r, c = c % SPLIT, c // SPLIT
    d, c = c % SPLIT, c // SPLIT
    b, rho = c % SPLIT, c // SPLIT
    def part(k, i):
        e = np.linspace(*reg[k], SPLIT + 1)
        return f"{k} {e[i]:.4g}-{e[i + 1]:.4g}"
    return (f"rho {RHO_EDGES[rho]}-{RHO_EDGES[rho + 1]}, {part('beta', b)}, "
            f"{part('delta', d)}, {part('R_gamma', r)}, card {card}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=int, default=2, help="Use every n-th shard.")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--mask_tau", type=float, default=None,
                    help="Mask PSID feature-waves whose imputed share is >= this "
                         "(psid_accuracy_codes.masks).")
    ap.add_argument("--profiles", action="store_true",
                    help="Only the lag profiles (RESULTS 43.7's table), into the output's "
                         "'profiles' key.")
    args = ap.parse_args()

    pd_ = torch.load(PSID_X, weights_only=False)
    px = pd_["x"].numpy().astype(np.float64)
    clean = {}
    if args.mask_tau is not None:
        from scripts import psid_accuracy_codes as pac
        m = pac.masks(pac.collect(Path("PSID-data"), pd_["psid_row"].numpy()), args.mask_tau)
        clean = {"consumption": ~m["consumption"], "liquid_assets": ~m["liquid"],
                 "illiquid_assets": ~m["illiquid"]}
        print(f"PSID masked at tau {args.mask_tau}: " + ", ".join(
            f"{f} {1 - c.mean():.3f}" for f, c in clean.items()))
    theta, card, train, held, _p, cfg = oa.load_shards([MAIN, STRIP])
    box = oa.box_of(cfg)
    files = (train + held)[::args.every]
    th, x, pid = oa.windows_raw(files, theta, seed=oa.HELD_SEED)
    th, x = th.numpy(), x.numpy().astype(np.float64)
    cd = np.asarray(card)[pid.numpy()]
    print(f"PSID couples {len(px)}; {len(th)} simulated windows from {len(files)} shards")
    if args.profiles:
        res = json.loads(args.out.read_text()) if args.out.exists() else {}
        res["profiles"] = profiles(px, th, x, cd, list(box.names))
        args.out.write_text(json.dumps(res, indent=2))
        print(f"wrote {args.out}")
        return
    boot = np.random.default_rng(0).integers(0, len(px), (args.n_boot, len(px)))
    res = {"mask_tau": args.mask_tau,
           "rule": "per feature, the range [minimum, typical] of the multiplicative "
                   "log-normal sigma (iid per household-wave) at which a cell of the region "
                   "reaches PSID couples' a = 2 V1 - V2 (IQR-based, network input scale): "
                   "minimum over the region's cells, and their prior-weighted median",
           "applied_region": "couples", "regions": {}}
    for name in ("couples", "box"):
        reg = region(name, box)
        cell = cells(th, cd, box, reg)
        ids, counts = np.unique(cell[cell >= 0], return_counts=True)
        use = ids[counts >= MIN_CELL]
        print(f"\n===== region {name}: beta {reg['beta']}, delta {reg['delta']}, R_gamma "
              f"{reg['R_gamma']}, rho and card free; {len(use)} of {len(ids)} cells with "
              f">= {MIN_CELL} windows")
        out = res["regions"][name] = {"bounds": reg, "features": {}}
        for f in ("income", "consumption", "liquid_assets", "illiquid_assets"):
            out["features"][f] = calibrate(f, px, boot, x, cell, use, reg,
                                            clean.get(f))
        out["sigma_range"] = {f: [v["sigma_min"], v["sigma_typical"]]
                              for f, v in out["features"].items()}
        print(f"\nregion {name}: sigma range (minimum, typical) {out['sigma_range']}")
    res["sigma_range"] = res["regions"]["couples"]["sigma_range"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\napplied (couples' region): {res['sigma_range']}\nwrote {args.out}")


def profiles(px, th, x, cd, names) -> dict:
    """IQR-based variance of k-wave changes, k = 1..6, for PSID couples and
    for simulated cardholders by rho (all beta, delta, R_gamma)."""
    rho = th[:, names.index("crra")]
    out = {}
    print("\nIQR-based variance of k-wave changes, k = 1..6 (model: cardholders)")
    for f in ("income", "consumption", "liquid_assets", "illiquid_assets"):
        rows = {"PSID couples": input_scale(px, f)}
        for lo, hi in ((0.5, 0.75), (1.5, 2.5), (3.5, 5.0)):
            rows[f"rho {lo}-{hi}"] = input_scale(x[(rho >= lo) & (rho < hi) & (cd == 1)], f)
        out[f] = {}
        for k, z in rows.items():
            v = [robust_var((z[:, j:] - z[:, :-j]).ravel()) for j in range(1, z.shape[1])]
            out[f][k] = v
            print(f"{f:16s}{k:14s}" + " ".join(f"{e:6.3f}" for e in v))
    # Why IQR-based: PSID's one-wave changes are heavy-tailed.
    print("\none-wave changes: pooled variance, IQR-based variance, kurtosis, share of "
          "variance in the largest 5%")
    tails = {}
    for f in ("income", "consumption", "liquid_assets", "illiquid_assets"):
        for k, z in (("PSID couples", input_scale(px, f)),
                     ("rho 1.5-2.5", input_scale(x[(rho >= 1.5) & (rho < 2.5) & (cd == 1)], f))):
            d = np.diff(z, axis=1).ravel()
            d = d - d.mean()
            sq = np.sort(d ** 2)[::-1]
            t = {"var": float(d.var()), "iqr_var": robust_var(d),
                 "kurtosis": float(np.mean(d ** 4) / np.mean(d ** 2) ** 2),
                 "top5_share": float(sq[:len(sq) // 20].sum() / sq.sum())}
            tails[f"{f}, {k}"] = t
            print(f"{f:16s}{k:14s}{t['var']:7.3f}{t['iqr_var']:7.3f}{t['kurtosis']:7.1f}"
                  f"{t['top5_share']:7.2f}")
    out["tails"] = tails
    return out


def calibrate(f, px, boot, x, cell, use, reg, clean=None) -> dict:
    a_p, v1, v2 = signature(input_scale(px, f), clean)
    a_b = np.array([signature(input_scale(px[b], f), None if clean is None else clean[b])[0]
                    for b in boot])
    rng = np.random.default_rng(1)
    stop = max(a_p, np.percentile(a_b, 95))
    curves = {int(c): curve(x[cell == c], f, rng, stop) for c in use}
    s_min, c_min = sigma_min(curves, a_p)
    lo, hi = (sigma_min(curves, t)[0] for t in np.percentile(a_b, [5, 95]))
    a0 = {c: a[0] for c, a in curves.items()}
    cmax = max(a0, key=a0.get)
    # Each cell's own requirement, and its median over the region under the
    # prior: cells are equal parts of beta, delta, R_gamma and card, so a
    # cell's prior volume is its rho bin's width.
    need = {c: sigma_min({c: a}, a_p)[0] for c, a in curves.items()}
    width = np.diff(RHO_EDGES)
    order = sorted(need, key=need.get)
    cw = np.cumsum([width[(c // 2) // SPLIT ** 3] for c in order])
    s_typ = need[order[int(np.searchsorted(cw, cw[-1] / 2))]]
    print(f"\n{f}: PSID a {a_p:.4f} (90% {np.percentile(a_b, 5):.4f}-"
          f"{np.percentile(a_b, 95):.4f}; V1 {v1:.4f}, V2 {v2:.4f}); model a without "
          f"noise: median cell {np.median(list(a0.values())):.4f}, max {a0[cmax]:.4f} "
          f"({describe_cell(cmax, reg)})")
    print(f"  minimum required sigma {s_min:.3f} (90% {lo:.3f}-{hi:.3f}), reached at "
          f"{describe_cell(c_min, reg) if c_min >= 0 else '-'}; typical (prior-weighted "
          f"median over the region's cells) {s_typ:.3f}; cells that cannot reach it with "
          f"sigma <= {SIGMA_GRID[-1]}: {sum(np.isinf(v) for v in need.values())}")
    by_rho = {}
    for r in range(len(RHO_EDGES) - 1):
        cs = [c for c in curves if (c // 2) // SPLIT ** 3 == r]
        if cs:
            s = [sigma_min({c: curves[c]}, a_p)[0] for c in cs]
            by_rho[f"{RHO_EDGES[r]}-{RHO_EDGES[r + 1]}"] = {
                "min": float(np.min(s)), "median": float(np.median(s))}
    print("  sigma needed by rho bin (min / median over its cells): " + "  ".join(
        f"{k}: {v['min']:.3f}/{v['median']:.3f}" for k, v in by_rho.items()))
    return {"psid_a": a_p, "psid_a_90": np.percentile(a_b, [5, 95]).tolist(),
            "psid_V1": v1, "psid_V2": v2,
            "model_a_no_noise_median_cell": float(np.median(list(a0.values()))),
            "model_a_no_noise_max_cell": float(a0[cmax]), "max_cell": describe_cell(cmax, reg),
            "sigma_min": s_min, "sigma_min_90": [lo, hi], "sigma_typical": s_typ,
            "sigma_by_cell": {describe_cell(c, reg): v for c, v in need.items()},
            "sigma_min_cell": describe_cell(c_min, reg) if c_min >= 0 else None,
            "sigma_by_rho_bin": by_rho}


if __name__ == "__main__":
    main()
