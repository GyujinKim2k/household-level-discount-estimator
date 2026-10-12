"""Why PSID couples' ρ sits at the 0.5 floor under option A2w (RESULTS 43.6).

Two checks, both on existing data; nothing is simulated.

``stats``: simple per-household statistics of the 7-wave window (wealth over
mean income, debt, consumption growth and volatility, income volatility) for
PSID couples, against simulated households in couples' region of (β, δ,
R_gamma), by ρ. Shows which statistics move with ρ in the model and where
PSID sits on them.

``ablate``: the trained ensemble's posterior for PSID couples after changing
one input channel (smooth it to a per-household trend, rescale it, ...). The
same change is applied to held-out simulated households, whose ρ is known, as
the control: a change that shifts ρ on PSID far more than on simulated
households is a channel through which PSID's data, not the change itself,
pull ρ down.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/rho_floor_diagnostic.py stats
    PYTHONPATH=. .venv/bin/python scripts/rho_floor_diagnostic.py ablate --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from hh_npe.evaluation.weighted import sample_weighted
from scripts import optionA as oa

MAIN = Path("data/processed/couples_dataset_shards")
STRIP = Path("data/processed/couples_strip_dataset_shards")
RUN_DIRS = [Path(f"outputs/optionA2w/s{s}") for s in range(5)]
PSID_X = Path("data/processed/psid_x_comphs_couples.pt")
OUT = Path("outputs/optionA2w/rho_floor_diagnostic.json")
INC, CONS, LIQ, ILLIQ, AGE = (FEATURES_TWOASSET_AGE.index(k) for k in
                              ("income", "consumption", "liquid_assets",
                               "illiquid_assets", "age"))
#: Couples' region of the other three parameters: roughly their p25-p75 of
#: posterior means under A2w (beta 0.75, delta 0.983, R_gamma 1.037 medians).
REGION = {"beta": (0.60, 0.90), "delta": (0.97, 0.995), "R_gamma": (1.025, 1.05)}
RHO_BINS = (0.5, 0.75, 1.0, 1.5, 2.5, 3.5, 5.0)


# ----------------------------------------------------------------- statistics

def household_stats(x: np.ndarray) -> dict[str, np.ndarray]:
    """Per-household statistics of a raw (N, waves, 5) dollar window."""
    y, c, liq, ill = x[..., INC], x[..., CONS], x[..., LIQ], x[..., ILLIQ]
    ybar = y.mean(1)
    years = x[:, -1, AGE] - x[:, 0, AGE]
    lc, ly = np.log(np.maximum(c, 1.0)), np.log(np.maximum(y, 1.0))
    dlc, dly = np.diff(lc, axis=1), np.diff(ly, axis=1)
    return {
        "illiq / ybar (median wave)": np.median(ill, 1) / ybar,
        "illiq = 0 (share of waves)": (ill <= 0).mean(1),
        "liquid / ybar (median wave)": np.median(liq, 1) / ybar,
        "in debt (share of waves)": (liq < 0).mean(1),
        "c / y (mean)": (c / np.maximum(y, 1.0)).mean(1),
        "net worth change / ybar / year": ((liq + ill)[:, -1] - (liq + ill)[:, 0]) / ybar / years,
        "log c growth / year": (lc[:, -1] - lc[:, 0]) / years,
        "sd dlog c (2y)": dlc.std(1),
        "sd dlog y (2y)": dly.std(1),
        "sd dlog c / sd dlog y": dlc.std(1) / np.maximum(dly.std(1), 1e-3),
    }


def stats(args) -> dict:
    theta, card, train, held, _, cfg = oa.load_shards([MAIN, STRIP])
    files = (train + held)[::args.every]
    th, x, pid = oa.windows_raw(files, theta, seed=oa.HELD_SEED)
    th, x, holder = th.numpy(), x.numpy(), np.asarray(card)[pid.numpy()] == 1
    names = list(oa.BOX.names)
    inr = np.ones(len(th), bool)
    for k, (lo, hi) in REGION.items():
        inr &= (th[:, names.index(k)] >= lo) & (th[:, names.index(k)] < hi)
    rho = th[:, names.index("crra")]
    sim = household_stats(x)
    d = torch.load(PSID_X, weights_only=False)
    ps = household_stats(d["x"].numpy())
    bins = list(zip(RHO_BINS[:-1], RHO_BINS[1:]))
    out = {"n_sim_in_region": int(inr.sum()), "region": REGION, "by_rho": {}}
    print(f"{len(files)} shards; {len(th)} simulated windows, {inr.sum()} in couples' "
          f"region {REGION}; PSID couples {len(d['x'])}")
    # Non-cardholders cannot borrow, so debt is compared among cardholders; PSID
    # cardholding is unobserved, so its row is all couples (a lower bound).
    sim["in debt, cardholders (share of waves)"] = np.where(holder, sim["in debt (share of waves)"], np.nan)
    ps["in debt, cardholders (share of waves)"] = ps["in debt (share of waves)"]
    print("medians, or means for shares (PSID: p25-p75 in brackets)")
    head = f"{'':40s}" + "".join(f"  rho {a:.2g}-{b:.2g}" for a, b in bins) + "   PSID couples"
    print(head)
    for k in sim:
        share = "share" in k
        stat = np.nanmean if share else np.nanmedian
        row, vals = f"{k:40s}", []
        for a, b in bins:
            v = sim[k][inr & (rho >= a) & (rho < b)]
            vals.append(float(stat(v)))
            row += f"{vals[-1]:13.3f}"
        p = np.percentile(ps[k], [25, 50, 75])
        row += (f"   {ps[k].mean():.3f} (mean)" if share
                else f"   {p[1]:.3f} ({p[0]:.3f}-{p[2]:.3f})")
        print(row)
        out["by_rho"][k] = {"statistic": "mean" if share else "median",
                            "sim_by_rho_bin": vals, "psid_p25_p50_p75": p.tolist(),
                            "psid_mean": float(ps[k].mean())}
    out["rho_bins"] = bins
    return out


# ------------------------------------------------------------------- ablation

def _trend(v: np.ndarray, log: bool) -> np.ndarray:
    """Per-household least-squares line through the waves (in logs if ``log``)."""
    t = np.arange(v.shape[1], dtype=float)
    t = t - t.mean()
    z = np.log(np.maximum(v, 1.0)) if log else v
    slope = (z * t).sum(1, keepdims=True) / (t ** 2).sum()
    fit = z.mean(1, keepdims=True) + slope * t
    return np.exp(fit) if log else fit


def _set(x, col, v):
    x = x.copy()
    x[..., col] = v
    return x


#: Each change maps a raw (N, waves, 5) dollar window to another.
ABLATIONS = {
    "consumption smoothed": lambda x: _set(x, CONS, _trend(x[..., CONS], True)),
    "income smoothed": lambda x: _set(x, INC, _trend(x[..., INC], True)),
    "illiquid smoothed": lambda x: _set(x, ILLIQ, _trend(x[..., ILLIQ], False)),
    "liquid smoothed": lambda x: _set(x, LIQ, _trend(x[..., LIQ], False)),
    "all four smoothed": lambda x: ABLATIONS["consumption smoothed"](
        ABLATIONS["income smoothed"](ABLATIONS["illiquid smoothed"](
            ABLATIONS["liquid smoothed"](x)))),
    "illiquid x2": lambda x: _set(x, ILLIQ, 2 * x[..., ILLIQ]),
    "illiquid x0.5": lambda x: _set(x, ILLIQ, 0.5 * x[..., ILLIQ]),
    "no card debt (liquid >= 0)": lambda x: _set(x, LIQ, np.maximum(x[..., LIQ], 0)),
    "consumption x0.85": lambda x: _set(x, CONS, 0.85 * x[..., CONS]),
    "consumption x1.15": lambda x: _set(x, CONS, 1.15 * x[..., CONS]),
    # Every dollar value scaled: only the log-mean-income channel changes.
    "all dollars x0.75 (income level)": lambda x: np.concatenate(
        [0.75 * x[..., :AGE], x[..., AGE:]], -1),
}


def _posterior(members, xr, spec, inv, lw, n_draws, chunk=64):
    """``sample_weighted`` by chunks. nflows' spline inverse can assert on an
    input far from the training data; such a row is returned as NaN (counted
    in ``failed``) rather than ending the run."""
    torch.manual_seed(0)
    xt, _ = oa.transform(torch.from_numpy(xr).float())
    fb = spec.flow_box()
    parts, failed = [], 0
    for i in range(0, len(xt), chunk):
        xb = xt[i:i + chunk]
        try:
            parts.append(sample_weighted(members, xb, n_draws, fb.low, fb.high,
                                         invert=inv, log_weight=lw))
            continue
        except AssertionError:
            pass
        for r in range(len(xb)):
            try:
                parts.append(sample_weighted(members, xb[r:r + 1], n_draws, fb.low,
                                             fb.high, invert=inv, log_weight=lw))
            except AssertionError:
                failed += 1
                d = len(fb.low)
                parts.append({k: np.full((1, d), np.nan) for k in ("mean", "sd", "lo", "hi")})
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("mean", "sd", "lo", "hi")}
    out["failed"] = failed
    return out


def ablate(args) -> dict:
    members, proposal, cfg = oa.load_members(RUN_DIRS, args.device)
    spec = oa.spec_of(cfg)
    inv, lw = oa._weighting(proposal, spec)
    j = list(spec.box.names).index("crra")
    psid_x = torch.load(PSID_X, weights_only=False)["x"].numpy().astype(np.float64)
    theta, _, _, held, _, _ = oa.load_shards([MAIN, STRIP])
    # Every household of every held-out draw: one per draw leaves too few in
    # couples' region.
    th, xs, _ = oa.windows_raw(held, theta, seed=oa.HELD_SEED)
    th, xs = th.numpy(), xs.numpy().astype(np.float64)
    names = list(spec.box.names)
    inr = np.ones(len(th), bool)
    for k, (lo, hi) in REGION.items():
        inr &= (th[:, names.index(k)] >= lo) & (th[:, names.index(k)] < hi)
    # Control: held-out households in couples' region whose true rho is
    # Laibson-like, so a shift there is the change's own effect on the network.
    sel = np.flatnonzero(inr & (th[:, j] >= 1.5) & (th[:, j] < 2.5))
    th, xs = th[sel], xs[sel]
    print(f"PSID couples {len(psid_x)}; control: {len(sel)} held-out households in "
          f"couples' region with true rho in [1.5, 2.5), true median "
          f"{np.median(th[:, j]):.3f}")

    def summ(r):
        m, lo = r["mean"][:, j], r["lo"][:, j]
        ok = np.isfinite(m)
        return {"rho_median_of_means": float(np.median(m[ok])),
                "rho_mean_lt_1": float((m[ok] < 1).mean()),
                "rho_lo_lt_0.55": float((lo[ok] < 0.55).mean()),
                "medians_all": dict(zip(names, np.median(r["mean"][ok], 0).tolist())),
                "failed": int(r["failed"])}

    out = {}
    base_p = _posterior(members, psid_x, spec, inv, lw, args.n_draws)
    base_s = _posterior(members, xs, spec, inv, lw, args.n_draws)
    out["base"] = {"psid": summ(base_p), "control": summ(base_s)}
    print(f"{'':34s}{'PSID rho med':>13s}{'<1':>7s}{'lo<.55':>8s}{'d med':>8s}"
          f"{'ctrl rho med':>14s}{'d med':>8s}")

    def line(k, p, s):
        print(f"{k:34s}{p['rho_median_of_means']:13.3f}{p['rho_mean_lt_1']:7.2f}"
              f"{p['rho_lo_lt_0.55']:8.2f}"
              f"{p['rho_median_of_means'] - out['base']['psid']['rho_median_of_means']:+8.3f}"
              f"{s['rho_median_of_means']:14.3f}"
              f"{s['rho_median_of_means'] - out['base']['control']['rho_median_of_means']:+8.3f}"
              + (f"   failed {p['failed']}/{s['failed']}" if p["failed"] + s["failed"] else ""))

    line("base", out["base"]["psid"], out["base"]["control"])
    for k, f in ABLATIONS.items():
        p = summ(_posterior(members, f(psid_x), spec, inv, lw, args.n_draws))
        s = summ(_posterior(members, f(xs), spec, inv, lw, args.n_draws))
        out[k] = {"psid": p, "control": s}
        line(k, p, s)
    out["control_true_rho_median"] = float(np.median(th[:, j]))
    out["control_n"] = int(len(sel))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--every", type=int, default=3, help="Use every n-th shard.")
    a = sub.add_parser("ablate")
    a.add_argument("--device", default="cuda")
    a.add_argument("--n_draws", type=int, default=10_000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    res = stats(args) if args.cmd == "stats" else ablate(args)
    prev = json.loads(args.out.read_text()) if args.out.exists() else {}
    prev[args.cmd] = res
    args.out.write_text(json.dumps(prev, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
