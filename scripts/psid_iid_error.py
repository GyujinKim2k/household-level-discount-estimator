"""PSID's iid error per feature, measured on PSID alone (RESULTS 57.2).

RESULTS 43.9 set the training noise relative to the model: the smallest sigma
at which some cell of the model's own windows reaches PSID's signature. That
ties the noise to the model's own dynamics, and for illiquid wealth those
dynamics have the opposite sign to PSID's (RESULTS 57.1). This measures the
iid component of PSID couples' own records instead, with no reference to the
model.

**Noise model.** The training noise (``optionA.add_obs_noise``) is
observed = true x exp(sigma z), with z iid N(0, 1) across waves. On
log|value| it is additive, sigma z, and it never moves a zero or flips a
sign. So:

- the estimation is on log|value|;
- a k-wave change y_t - y_{t-k} is used only inside a *run*, where the value is
  non-zero and of one sign in every wave from t-k to t. Selecting on zeros
  and signs selects on the true values only, which the noise cannot change,
  so every selected change carries the noise's full 2 sigma^2;
- V_k, the variance of k-wave changes, is Var(true k-wave change) + 2 sigma^2.

**Separating the noise from true dynamics: how V_k grows with k.** Noise
adds the same 2 sigma^2 at every gap, the intercept; true persistent shocks
add more the longer the gap. Three specifications of the true process:

- ``signature``: a random walk, lags 1-2 only: sigma^2 = (2 V_1 - V_2) / 2.
  This is RESULTS 43.9's statistic, on this scale.
- ``rw``: a random walk, lags 1-6: V_k = 2 sigma^2 + k s_eta^2. This is the
  regression of the variance of k-period differences on k (Carroll and
  Samwick 1997).
- ``rw+drift``: adds a household-specific drift: V_k = 2 sigma^2 + k s_eta^2 +
  k^2 s_g^2.
- ``rw+drift+ar``: adds a mean-reverting component, AR(1) per wave:
  V_k = 2 sigma^2 + k s_eta^2 + k^2 s_g^2 + 2 s^2 (1 - phi^k), phi on a grid
  0.05-0.95. A short-lived true component (small phi) is nearly iid itself,
  so this spec gives true dynamics the most room and reads sigma lowest.

The fitted specs use non-negative least squares on V_k relative to V_k
(for ``rw+drift+ar``, at each phi, keeping the best).

**What sigma holds.** Anything iid in levels: reporting error, imputation,
and true one-wave movements, such as a car bought and later sold. sigma is
therefore an upper bound on reporting error. For income and consumption it
also includes transitory movements the model already simulates.

**Details:**
- **Variance.** V_k is IQR-based, ((p75 - p25) / 1.349)^2, as in RESULTS
  43.9 (``iqr``), or winsorised at the 1st and 99th percentiles
  (``winsorised``).
- **Calendar effects.** Each calendar pair of waves is first centred on its
  median, so a common year effect is not read as dispersion (house prices
  after 2019, for example).
- **Uncertainty.** 200 bootstrap resamples of couples.
- **Imputation.** ``unimputed`` repeats everything with PSID's imputed
  feature-waves treated as missing (``psid_accuracy_codes.masks`` at tau 0:
  any part imputed). Income has no accuracy codes.
- **By size.** The same, ``rw+drift``, within groups of couples by their
  median |value| over mean income across the window (groups of at least 50
  couples). A single multiplicative sigma assumes the error scales with the
  value; this shows whether it does.
- **Model reference** (context only; it sets nothing): the same estimator on
  v2's noise-free windows in couples' region (RESULTS 43.9's region).

**Range** (``sigma_range``, in ``noise_calibration.py``'s format), per
feature, over every value:
- **low:** ``rw+drift+ar`` on the IQR-based variance. True dynamics get the
  most room, and only a typical record's error is counted.
- **high:** ``rw+drift`` on the winsorised variance. Every reversal is
  counted as error, and so is the occasional gross error.

The tests check that the two bracket the true sigma whether the true process
is a random walk with drift or mean-reverting. Training draws each
household's sigma uniformly between them, as for RESULTS 43.9's range.

Usage::

    PYTHONPATH=.:src .venv/bin/python scripts/psid_iid_error.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import nnls

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from scripts.noise_calibration import PSID_X

FEATURES = ("income", "consumption", "liquid_assets", "illiquid_assets")
#: psid_accuracy_codes' name for each feature with accuracy codes.
MASK_KEY = {"consumption": "consumption", "liquid_assets": "liquid",
            "illiquid_assets": "illiquid"}
SPECS = ("signature", "rw", "rw+drift", "rw+drift+ar")
#: phi for ``rw+drift+ar``.
PHI_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
VARIANCES = ("iqr", "winsorised")
MAX_LAG = 6
#: Calendar pairs of waves with fewer changes than this are left out.
MIN_GROUP = 20
#: Gaps with fewer changes than this are left out of the fits.
MIN_LAG = 100
#: Groups by median |value| over mean income (``by_size``).
SIZE_EDGES = (0.0, 0.1, 0.5, 1.0, 2.0, np.inf)
MIN_COUPLES = 50
V2 = (Path("data/processed/couples_v2_dataset_shards"),
      Path("data/processed/couples_v2_strip_dataset_shards"))
OUT = Path("outputs/psid_iid_error/psid_iid_error.json")
FIG = Path("figures/70_psid_iid_error.png")


def in_run(v: np.ndarray, k: int, clean: np.ndarray | None = None) -> np.ndarray:
    """(N, waves - k): the k-wave change ending at wave s + k lies inside a run,
    where ``v`` is non-zero, of one sign and ``clean`` in every wave s..s+k."""
    good = v != 0 if clean is None else (v != 0) & clean
    sign = np.sign(v)
    n = v.shape[1] - k
    ok = np.ones((len(v), n), bool)
    for j in range(k + 1):
        ok &= good[:, j:j + n] & (sign[:, j:j + n] == sign[:, :n])
    return ok


def variance(d: np.ndarray, kind: str) -> float:
    if kind == "iqr":
        q25, q75 = np.percentile(d, [25, 75])
        return float(((q75 - q25) / 1.349) ** 2)
    lo, hi = np.percentile(d, [1, 99])
    return float(np.clip(d, lo, hi).var())


def lag_variances(v: np.ndarray, kind: str, clean: np.ndarray | None = None,
                  max_lag: int = MAX_LAG) -> tuple[np.ndarray, np.ndarray]:
    """V_k for k = 1..max_lag on log|v|, within runs, each calendar pair of
    waves centred on its median; and the number of changes behind each."""
    y = np.log(np.abs(np.where(v != 0, v, 1.0)))
    V, n = [], []
    for k in range(1, max_lag + 1):
        d, ok = y[:, k:] - y[:, :-k], in_run(v, k, clean)
        parts = [d[ok[:, s], s] - np.median(d[ok[:, s], s])
                 for s in range(d.shape[1]) if ok[:, s].sum() >= MIN_GROUP]
        c = np.concatenate(parts) if parts else np.array([])
        V.append(variance(c, kind) if len(c) else np.nan)
        n.append(len(c))
    return np.array(V), np.array(n)


def fit(V: np.ndarray, spec: str, n: np.ndarray | None = None) -> dict:
    """sigma (the iid component's sd) and the true process's terms, by spec,
    on the gaps with at least ``MIN_LAG`` changes (``n``)."""
    use = np.isfinite(V) & (np.ones(len(V), bool) if n is None else np.asarray(n) >= MIN_LAG)
    if spec == "signature":
        if not use[:2].all():
            return {"sigma": float("nan"), "sigma2": float("nan"), "lags": []}
        s2 = max((2 * V[0] - V[1]) / 2, 0.0)
        return {"sigma": float(np.sqrt(s2)), "sigma2": float(s2), "lags": [1, 2]}
    k = np.arange(1, len(V) + 1, dtype=float)[use]
    need = {"rw": 3, "rw+drift": 4, "rw+drift+ar": 5}[spec]
    if len(k) < need:
        return {"sigma": float("nan"), "sigma2": float("nan"), "s_eta2": float("nan"),
                "s_g2": float("nan"), "lags": k.astype(int).tolist(),
                "max_rel_misfit": float("nan")}
    cols = [np.full_like(k, 2.0), k] + ([k ** 2] if spec != "rw" else [])
    best = None
    for phi in (PHI_GRID if spec == "rw+drift+ar" else [None]):
        A = np.column_stack(cols + ([2 * (1 - phi ** k)] if phi is not None else []))
        coef, res = nnls(A / V[use][:, None], np.ones_like(k))
        if best is None or res < best[0] - 1e-12:
            best = (res, phi, A, coef)
    _res, phi, A, coef = best
    out = {"sigma": float(np.sqrt(coef[0])), "sigma2": float(coef[0]),
           "s_eta2": float(coef[1]), "s_g2": float(coef[2]) if spec != "rw" else 0.0,
           "lags": k.astype(int).tolist(),
           "max_rel_misfit": float(np.max(np.abs(A @ coef / V[use] - 1)))}
    if phi is not None:
        out |= {"phi": float(phi), "s2": float(coef[3])}
    return out


def estimate(v: np.ndarray, clean: np.ndarray | None = None) -> dict:
    out = {}
    for kind in VARIANCES:
        V, n = lag_variances(v, kind, clean)
        out[kind] = {"V": V.tolist(), "n": n.tolist(), **{s: fit(V, s, n) for s in SPECS}}
    return out


def bootstrap(v: np.ndarray, clean: np.ndarray | None, n_boot: int, seed: int = 0) -> dict:
    """90% intervals for sigma, per variance and spec, over resamples of couples."""
    rng = np.random.default_rng(seed)
    draws = {(k, s): [] for k in VARIANCES for s in SPECS}
    for _ in range(n_boot):
        i = rng.integers(0, len(v), len(v))
        e = estimate(v[i], None if clean is None else clean[i])
        for k in VARIANCES:
            for s in SPECS:
                draws[k, s].append(e[k][s]["sigma"])
    return {k: {s: np.nanpercentile(draws[k, s], [5, 95]).tolist() for s in SPECS}
            for k in VARIANCES}


def by_size(v: np.ndarray, ybar: np.ndarray, n_boot: int) -> list[dict]:
    """``rw+drift`` sigma within groups of couples by median |v| / mean income."""
    med = np.median(np.abs(v) / ybar, axis=1)
    out = []
    for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:], strict=True):
        m = (med >= lo) & (med < hi)
        row = {"median_over_income": [lo, None if np.isinf(hi) else hi], "couples": int(m.sum())}
        if m.sum() >= MIN_COUPLES:
            e, b = estimate(v[m]), bootstrap(v[m], None, n_boot)
            row |= {k: {"sigma": e[k]["rw+drift"]["sigma"], "lags": e[k]["rw+drift"]["lags"],
                        "boot90": b[k]["rw+drift"]} for k in VARIANCES}
        out.append(row)
    return out


def psid(n_boot: int, psid_dir: Path) -> dict:
    from scripts import psid_accuracy_codes as pac

    d = torch.load(PSID_X, weights_only=False)
    x = d["x"].numpy().astype(np.float64)
    m = pac.masks(pac.collect(psid_dir, d["psid_row"].numpy()), 0.0)
    ybar = x[..., FEATURES_TWOASSET_AGE.index("income")].mean(1, keepdims=True)
    res = {"n": len(x)}
    for f in FEATURES:
        v = x[..., FEATURES_TWOASSET_AGE.index(f)]
        res[f] = {"all": estimate(v) | {"boot90": bootstrap(v, None, n_boot)},
                  "share_waves_nonzero": float(in_run(v, 0).mean()),
                  "by_size": by_size(v, ybar, n_boot)}
        if f in MASK_KEY:
            clean = ~m[MASK_KEY[f]]
            res[f]["unimputed"] = (estimate(v, clean)
                                   | {"boot90": bootstrap(v, clean, n_boot)})
            res[f]["share_waves_imputed"] = float(m[MASK_KEY[f]].mean())
    return res


def model(shards, every: int) -> dict:
    from scripts import optionA as oa
    from scripts.noise_calibration import cells, region

    theta, card, train, held, _p, cfg = oa.load_shards(list(shards))
    box = oa.box_of(cfg)
    th, x, pid = oa.windows_raw((train + held)[::every], theta, seed=oa.HELD_SEED)
    th, x = th.numpy(), x.numpy().astype(np.float64)
    cd = np.asarray(card)[pid.numpy()]
    x = x[cells(th, cd, box, region("couples", box)) >= 0]
    return {"n_windows": int(len(x))} | {
        f: estimate(x[..., FEATURES_TWOASSET_AGE.index(f)]) for f in FEATURES}


def report(res: dict, current: dict | None) -> None:
    p = res["psid"]
    print(f"\nPSID couples ({p['n']}): sigma of the iid component of log|value| "
          "(90% bootstrap interval)")
    for f in FEATURES:
        cur = f"   training now {current[f][0]}-{current[f][1]}" if current else ""
        print(f"\n{f}: {p[f]['share_waves_nonzero']:.3f} of household-waves non-zero{cur}")
        for var in [v for v in ("all", "unimputed") if v in p[f]]:
            for kind in VARIANCES:
                e, b = p[f][var][kind], p[f][var]["boot90"][kind]
                print(f"  {var:9s} {kind:10s} V_k " + " ".join(f"{u:.4f}" for u in e["V"])
                      + "   n " + " ".join(str(u) for u in e["n"]))
                print("      " + "   ".join(
                    f"{s} {e[s]['sigma']:.3f} ({b[s][0]:.3f}-{b[s][1]:.3f})" for s in SPECS)
                      + f"   misfit rw {e['rw']['max_rel_misfit']:.2f}, "
                        f"rw+drift {e['rw+drift']['max_rel_misfit']:.2f}, lags "
                        f"{e['rw+drift']['lags'][0]}-{e['rw+drift']['lags'][-1]}")
    print("\nby size (rw+drift; groups by median |value| / mean income): sigma iqr / winsorised")
    for f in FEATURES:
        rows = [r for r in p[f]["by_size"] if "iqr" in r]
        print(f"  {f:16s} " + "   ".join(
            f"{r['median_over_income'][0]}-{r['median_over_income'][1] or 'inf'} "
            f"({r['couples']}): {r['iqr']['sigma']:.3f} / {r['winsorised']['sigma']:.3f}"
            for r in rows))
    print("\nrange, PSID only [rw+drift+ar iqr, rw+drift winsorised]: " + ", ".join(
        f"{f} {lo:.3f}-{hi:.3f}" for f, (lo, hi) in res["sigma_range"].items()))
    if "model_v2" in res:
        mo = res["model_v2"]
        print(f"\nmodel v2, noise-free, couples' region ({mo['n_windows']} windows), iqr: "
              "sigma by spec (context only)")
        for f in FEATURES:
            e = mo[f]["iqr"]
            print(f"  {f:16s} V_k " + " ".join(f"{u:.4f}" for u in e["V"]) + "   "
                  + "  ".join(f"{s} {e[s]['sigma']:.3f}" for s in SPECS))


def plot(res: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    k = np.arange(1, MAX_LAG + 1)
    for ax, f in zip(axes, FEATURES, strict=True):
        p = res["psid"][f]
        for var, mk in (("all", "o"), ("unimputed", "s")):
            if var not in p:
                continue
            e = p[var]["iqr"]
            V = np.where(np.array(e["n"]) >= MIN_LAG, np.array(e["V"]), np.nan)
            ax.plot(k, V, mk, mfc="none" if var == "unimputed" else None, color="k",
                    label=f"PSID couples, {'every value' if var == 'all' else 'unimputed'}")
            for spec, ls in (("rw+drift", "-"), ("rw+drift+ar", "--")):
                c = e[spec]
                if not np.isfinite(c["sigma"]):
                    continue
                fitted = 2 * c["sigma2"] + k * c["s_eta2"] + k ** 2 * c["s_g2"]
                if "phi" in c:
                    fitted = fitted + 2 * c["s2"] * (1 - c["phi"] ** k)
                ax.plot(np.r_[0, k], np.r_[2 * c["sigma2"], fitted], ls,
                        color="tab:blue" if var == "all" else "tab:orange", lw=1,
                        label=f"{spec} fit ({var}): sigma {c['sigma']:.3f}")
        if "model_v2" in res:
            ax.plot(k, res["model_v2"][f]["iqr"]["V"], ":", color="gray",
                    label="model v2, no noise (context)")
        ax.set(title=f.replace("_", " "), xlabel="gap k (waves of 2 years)",
               ylabel="V_k, IQR-based, log|value|", xlim=(0, MAX_LAG + 0.3))
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=6.5)
    fig.suptitle("Variance of k-wave changes on log|value|: the intercept at k = 0 is "
                 "2 sigma^2, the iid component")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psid_dir", type=Path, default=Path("PSID-data"))
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--shards", type=Path, nargs="+", default=list(V2),
                    help="Model reference runs; --every 0 skips the reference.")
    ap.add_argument("--every", type=int, default=4, help="Use every n-th shard.")
    ap.add_argument("--current", type=Path,
                    default=Path("outputs/optionA2v2/noise_calibration.json"),
                    help="The training's current ranges, printed alongside.")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fig", type=Path, default=FIG)
    args = ap.parse_args()
    res = {"specs": list(SPECS), "variances": list(VARIANCES), "max_lag": MAX_LAG,
           "psid": psid(args.n_boot, args.psid_dir)}
    if args.every:
        res["model_v2"] = model(args.shards, args.every)
    res["sigma_range"] = {f: [round(res["psid"][f]["all"]["iqr"]["rw+drift+ar"]["sigma"], 3),
                              round(res["psid"][f]["all"]["winsorised"]["rw+drift"]["sigma"], 3)]
                          for f in FEATURES}
    current = (json.loads(args.current.read_text())["sigma_range"]
               if args.current.exists() else None)
    res["training_now"] = current
    report(res, current)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    plot(res, args.fig)
    print(f"\nwrote {args.out} and {args.fig}")


if __name__ == "__main__":
    main()
