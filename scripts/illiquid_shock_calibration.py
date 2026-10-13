"""The illiquid valuation shock's range, calibrated from PSID couples (RESULTS 44).

PSID couples' illiquid wealth keeps drifting apart far faster than the
model's: the variance of k-wave changes grows with k much more steeply
(RESULTS 43.7's lag profiles). Observation noise cannot produce that; it adds
the same 2 sigma^2 to every V_k. A valuation shock on the balance does, since
it compounds. The statistic is therefore

    d = V_6 - V_1,

the growth of the variance of illiquid changes from one wave apart to six
(IQR-based, on the network's asinh input scale), in which iid noise cancels to
first order. The model's windows carry A2n's illiquid observation noise
anyway, drawn as in training, because IQR-based variances are not exactly
additive.

**First pass, no behaviour.** The shock enters the solve (households hold less
of a riskier asset), which this script cannot rerun. It compounds the shock
onto A2's stored windows at the wave level, keeping each window's own deposits:

    Z~_{w+1} = max(Z~_w * G_w + (Z_{w+1} - Z_w), 0),

with G_w the product of two annual shocks, log G ~ N(-sigma^2, 2 sigma^2), so
E[G] = 1. The pilot's gate G1 checks the resulting range on the real solve.

**Holders.** A multiplicative shock cannot move a zero balance, and the model
has many more couples with almost no illiquid wealth than PSID does. Pooled
over everyone, d therefore mixes the shock with who holds illiquid wealth at
all, and the rule would push sigma up on the holders to make up for the
non-holders. The applied statistic is d among **holders**: windows whose
illiquid wealth is at least ``HOLD`` (0.5) times the window's mean income in
every wave, on the observed (shocked, noisy) values, as for PSID. d over all
windows and the composition (holders, never-holders, the rest) are reported
alongside.

**The rule is RESULTS 43.9's.** Cells are rho bins x halves of beta, delta and
R_gamma x card type, in couples' region (``EDGE_A2``). A cell's requirement is
the smallest annual sigma at which its d reaches PSID couples' d (common random
numbers across sigma; infinity if no sigma <= 0.6 does). The range is
[minimum over cells, prior-weighted median of the cells' requirements]. For
holders a cell needs at least ``MIN_HOLDERS`` holder windows without the shock.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/illiquid_shock_calibration.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from scripts import optionA as oa
from scripts.noise_calibration import (
    MAIN,
    MIN_CELL,
    PSID_X,
    RHO_EDGES,
    SPLIT,
    STRIP,
    cells,
    describe_cell,
    input_scale,
    region,
    robust_var,
    signature,
)

OUT = Path("outputs/optionA3/zshock_calibration.json")
NOISE = Path("outputs/optionA2n/noise_calibration.json")
SD_GRID = np.round(np.arange(0.0, 0.6001, 0.01), 3)
ILL = FEATURES_TWOASSET_AGE.index("illiquid_assets")
INC = FEATURES_TWOASSET_AGE.index("income")
F = "illiquid_assets"
#: Holders: illiquid wealth >= HOLD x the window's mean income in every wave.
HOLD = 0.5
MIN_HOLDERS = 100
POPULATIONS = ("holders", "all")


def ratio(x: np.ndarray) -> np.ndarray:
    return x[..., ILL] / x[..., INC].mean(1, keepdims=True)


def groups(x: np.ndarray) -> dict:
    """Holders, never-holders (below 0.1 x mean income in every wave) and the
    rest, on (N, waves, 5) windows."""
    r = ratio(x)
    hold, never = (r >= HOLD).all(1), (r < 0.1).all(1)
    return {"holders": hold, "never": never, "rest": ~hold & ~never}


def statistic(x: np.ndarray, pop: str) -> float:
    """d of the windows' illiquid wealth over ``pop`` (``holders`` or ``all``)."""
    z = input_scale(x, F)
    if pop == "holders":
        z = z[groups(x)["holders"]]
    return persistent(z)


def profile(z: np.ndarray) -> list[float]:
    """IQR-based variance of k-wave changes, k = 1 .. waves - 1."""
    return [robust_var((z[:, k:] - z[:, :-k]).ravel()) for k in range(1, z.shape[1])]


def persistent(z: np.ndarray) -> float:
    """d = V_last - V_1 of an (N, waves) series on the input scale."""
    v = profile(z)
    return v[-1] - v[0]


class Draws:
    """Common random numbers for one set of windows: the shock's standard
    normals per wave gap, and the observation noise (each window's sigma
    uniform on A2n's illiquid range, a normal per wave)."""

    def __init__(self, n: int, waves: int, rng: np.random.Generator, noise: tuple):
        self.eps = rng.standard_normal((n, waves - 1))
        lo, hi = noise
        self.sig_n = (lo + (hi - lo) * rng.random(n))[:, None]
        self.zn = rng.standard_normal((n, waves))


def shocked(x: np.ndarray, sd: float, dr: Draws) -> np.ndarray:
    """Windows with the valuation shock compounded onto illiquid wealth and
    A2n's observation noise on top (module docstring)."""
    z = x[..., ILL]
    out = x.copy()
    zt = z[:, 0].copy()
    path = [zt]
    g = np.exp(sd * np.sqrt(2.0) * dr.eps - sd ** 2)
    for w in range(1, z.shape[1]):
        zt = np.maximum(zt * g[:, w - 1] + (z[:, w] - z[:, w - 1]), 0.0)
        path.append(zt)
    out[..., ILL] = np.stack(path, 1) * np.exp(dr.sig_n * dr.zn)
    return out


def curve(x: np.ndarray, rng: np.random.Generator, stop: float, noise: tuple,
          pop: str) -> np.ndarray:
    """d(sigma) over ``pop`` on ``SD_GRID``; +inf past the first sigma that
    reaches ``stop``."""
    dr = Draws(len(x), x.shape[1], rng, noise)
    out = np.full(len(SD_GRID), np.inf)
    for i, s in enumerate(SD_GRID):
        out[i] = statistic(shocked(x, s, dr), pop)
        if out[i] >= stop:
            break
    return out


def need(curves: dict, target: float) -> dict:
    """Each cell's smallest sigma reaching ``target`` (inf if none)."""
    out = {}
    for c, d in curves.items():
        hit = np.flatnonzero(d >= target)
        out[c] = float(SD_GRID[hit[0]]) if len(hit) else np.inf
    return out


def typical(req: dict) -> float:
    """Prior-weighted median of the cells' requirements (a cell's prior
    volume is its rho bin's width; RESULTS 43.9)."""
    width = np.diff(RHO_EDGES)
    order = sorted(req, key=req.get)
    cw = np.cumsum([width[(c // 2) // SPLIT ** 3] for c in order])
    return req[order[int(np.searchsorted(cw, cw[-1] / 2))]]


def falls(x: np.ndarray) -> dict:
    """Dollar one-wave illiquid changes: share falling by more than $1,000,
    and the 10th percentile."""
    dz = np.diff(x[..., ILL], axis=1).ravel()
    return {"share_down": float((dz < -1000).mean()), "p10": float(np.percentile(dz, 10))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=int, default=2, help="Use every n-th shard.")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--psid_x", type=Path, default=PSID_X,
                    help="PSID couples' tensor (RESULTS 44.4: illiquid wealth on "
                         "another definition).")
    args = ap.parse_args()

    noise = tuple(json.loads(NOISE.read_text())["sigma_range"][F])
    px = torch.load(args.psid_x, weights_only=False)["x"].numpy().astype(np.float64)
    theta, card, train, held, _p, cfg = oa.load_shards([MAIN, STRIP])
    box = oa.box_of(cfg)
    files = (train + held)[::args.every]
    th, x, pid = oa.windows_raw(files, theta, seed=oa.HELD_SEED)
    th, x = th.numpy(), x.numpy().astype(np.float64)
    cd = np.asarray(card)[pid.numpy()]
    print(f"PSID couples {len(px)}; {len(th)} simulated windows from {len(files)} shards; "
          f"A2n illiquid noise {noise}")

    reg = region("couples", box)
    cell = cells(th, cd, box, reg)
    ids, counts = np.unique(cell[cell >= 0], return_counts=True)
    use = ids[counts >= MIN_CELL]
    print(f"couples' region: beta {reg['beta']}, delta {reg['delta']}; {len(use)} of "
          f"{len(ids)} cells with >= {MIN_CELL} windows")

    # Who holds illiquid wealth: PSID against the model (A2n noise, no shock).
    dr0 = Draws(len(x), x.shape[1], np.random.default_rng(3), noise)
    x0 = shocked(x, 0.0, dr0)
    comp = {"PSID couples": {k: float(m.mean()) for k, m in groups(px).items()}}
    rho = th[:, list(box.names).index("crra")]
    for lo, hi in zip(RHO_EDGES[:-1], RHO_EDGES[1:], strict=True):
        m = (cell >= 0) & (rho >= lo) & (rho < hi)
        comp[f"model rho {lo}-{hi}"] = {k: float(v.mean()) for k, v in groups(x0[m]).items()}
    print("\nshare of windows: holders (>= 0.5 x income every wave) / never (< 0.1) / rest")
    for k, v in comp.items():
        print(f"  {k:22s} {v['holders']:.2f} / {v['never']:.2f} / {v['rest']:.2f}")

    res = {"rule": "annual sd of a mean-one log-normal valuation shock on illiquid wealth: "
                   "[minimum, typical] over couples' cells of the smallest sigma at which "
                   "the cell's d = V6 - V1 among holders (IQR-based, asinh input scale, A2n "
                   "noise on) reaches PSID couples'; first pass compounding at the wave "
                   "level with deposits kept, no behavioural response",
           "psid_x": str(args.psid_x),
           "psid_noise_signature_a": signature(input_scale(px, F))[0],
           "applied_population": "holders", "hold": HOLD, "region": reg,
           "obs_noise_illiquid": list(noise), "composition": comp, "populations": {}}
    boot = np.random.default_rng(0).integers(0, len(px), (args.n_boot, len(px)))
    for pop in POPULATIONS:
        res["populations"][pop] = calibrate(pop, px, boot, x, cell, use, reg, noise)
    res["zshock_range"] = res["populations"]["holders"]["zshock_range"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\napplied (holders): {res['zshock_range']}; wrote {args.out}")


def calibrate(pop, px, boot, x, cell, use, reg, noise) -> dict:
    """The range for one population (module docstring)."""
    d_p = statistic(px, pop)
    zp = input_scale(px, F)
    if pop == "holders":
        zp = zp[groups(px)["holders"]]
    d_b = np.array([statistic(px[b], pop) for b in boot])
    d_lo, d_hi = np.percentile(d_b, [5, 95])
    print(f"\n===== {pop}\nPSID couples: d = V6 - V1 = {d_p:.4f} (90% {d_lo:.4f}-{d_hi:.4f}); "
          f"V_k {np.round(profile(zp), 3).tolist()}")
    if pop == "holders":
        x0 = shocked(x, 0.0, Draws(len(x), x.shape[1], np.random.default_rng(3), noise))
        nh = {int(c): int(groups(x0[cell == c])["holders"].sum()) for c in use}
        use = np.array([c for c in use if nh[int(c)] >= MIN_HOLDERS])
        print(f"{len(use)} cells with >= {MIN_HOLDERS} holder windows")
    rng = np.random.default_rng(1)
    stop = max(d_p, d_hi)
    curves = {int(c): curve(x[cell == c], rng, stop, noise, pop) for c in use}
    req = need(curves, d_p)
    s_min = min(req.values())
    c_min = min(req, key=req.get)
    s_min_90 = [min(need(curves, t).values()) for t in (d_lo, d_hi)]
    s_typ = typical(req)
    d0 = {c: d[0] for c, d in curves.items()}
    print(f"model d with no shock (A2n noise on): median cell {np.median(list(d0.values())):.4f}, "
          f"max {max(d0.values()):.4f}")
    print(f"minimum sigma {s_min:.3f} (90% {s_min_90[0]:.3f}-{s_min_90[1]:.3f}) at "
          f"{describe_cell(c_min, reg)}; typical {s_typ:.3f}; cells that cannot reach "
          f"with sigma <= {SD_GRID[-1]}: {sum(np.isinf(v) for v in req.values())}")
    by_rho = {}
    for r in range(len(RHO_EDGES) - 1):
        s = [v for c, v in req.items() if (c // 2) // SPLIT ** 3 == r]
        if s:
            by_rho[f"{RHO_EDGES[r]}-{RHO_EDGES[r + 1]}"] = {
                "min": float(np.min(s)), "median": float(np.median(s))}
    print("sigma needed by rho bin (min / median): " + "  ".join(
        f"{k}: {v['min']:.3f}/{v['median']:.3f}" for k, v in by_rho.items()))

    # The region pooled, at no shock and at the range's two ends.
    inside = np.isin(cell, use)
    dr = Draws(int(inside.sum()), x.shape[1], np.random.default_rng(2), noise)
    pooled = {}
    for label, s in (("no shock", 0.0), ("minimum", s_min), ("typical", s_typ)):
        if not np.isfinite(s):
            continue
        xs = shocked(x[inside], s, dr)
        zs = input_scale(xs, F)
        if pop == "holders":
            zs = zs[groups(xs)["holders"]]
        pooled[label] = {"sigma": s, "V_k": profile(zs), **falls(xs)}
        print(f"region pooled, {label} (sigma {s:.3f}): V_k "
              f"{np.round(pooled[label]['V_k'], 3).tolist()}; all windows' falls {falls(xs)}")
    return {"psid": {"d": d_p, "d_90": [d_lo, d_hi], "V_k": profile(zp), **falls(px)},
            "n_cells": int(len(use)), "zshock_min": s_min, "zshock_min_90": s_min_90,
            "zshock_min_cell": describe_cell(c_min, reg), "zshock_typical": s_typ,
            "zshock_range": [s_min, s_typ],
            "by_cell": {describe_cell(c, reg): v for c, v in req.items()},
            "by_rho_bin": by_rho, "pooled": pooled}

if __name__ == "__main__":
    main()
