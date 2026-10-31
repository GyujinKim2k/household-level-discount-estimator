"""Can the household posteriors tell PSID households apart?

Two questions, kept apart:

1. **Do households' estimates differ by more than estimation noise?** The SBC
   set simulates 16 households per draw sharing one theta and one card type
   (`sbc_optionA2.py`), each with its own start age, seed-pool start, shocks
   and training noise. The spread of their posterior means is the spread
   estimation noise alone produces: the shared-theta null. It is compared with
   the spread of PSID households' posterior means. Draws are weighted toward
   PSID: each draw by the density of PSID's posterior means (a Gaussian KDE in
   standardised beta, log(1 - delta), rho, R_gamma) at the draw's average
   posterior mean. A card-type mix adds spread the null does not have; its
   size is the card shift of the draw-average posterior mean (a quadratic
   regression on the truth with a card term), and the null is reported with a
   50/50 mix, the largest it can add.

   That null shares all four parameters. A household's estimate of one
   parameter also moves with the other three, so a second, conservative null
   shares only that parameter and lets the rest vary as independent draws over
   the box do (`others_spread`). PSID's spread beyond the first says
   households differ; beyond the second, that they differ in that parameter.

   The variance identity Var(theta) = Var(means) + E[posterior var] that
   `heterogeneity_test.decompose` (RESULTS 1) rests on holds only when the
   population is the posterior's prior. It is checked here on the SBC set,
   as drawn and reweighted toward PSID, where the truth's variance is known.

2. **Can one household be placed against another?** One household's 90%
   interval against the 5-95% range of households' posterior means; the share
   of household pairs whose 90% intervals do not overlap, and whose order has
   posterior probability 0.9 or more (normal approximation). On simulated
   households the same pairs are scored against the truth.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/household_resolution.py
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = {   # arm: (members' output directory, PSID couples, PSID all 889)
    "v2": ("outputs/optionA2v2", "outputs/psid_optionA2v2_couples",
           "outputs/psid_optionA2v2_all"),
    "typical": ("outputs/optionA2v2_noise_typical",
                "outputs/psid_optionA2v2_noise_typical_couples",
                "outputs/psid_optionA2v2_noise_typical_all"),
    "gross": ("outputs/optionA2v2_noise_gross",
              "outputs/psid_optionA2v2_noise_gross_couples",
              "outputs/psid_optionA2v2_noise_gross_all"),
}
SBC_SHARDS = Path("data/processed/optionA2v2_sbc16_shards")
NAMES = ("beta", "delta", "crra", "R_gamma")
PER_DRAW = 16
Z90 = stats.norm.ppf(0.9)
N_BOOT = 1000
OUT = Path("outputs/household_resolution/household_resolution.json")
FIG = Path("figures/80_household_resolution.png")
#: the typical gross-arm couple's weighted posterior draws (figure 76)
COUPLE = Path("figures/76_psid_household_optionA2v2_noise_gross.npz")
ARM_COLOR = {"v2": "#3B75AF", "typical": "#8D69B8", "gross": "#C53A32"}
LIMITS = {"beta": (0.30, 1.00), "delta": (0.85, 1.00), "crra": (0.50, 5.00)}  # the box
VIEW = {**LIMITS, "delta": (0.88, 1.00)}
LABEL = {"beta": r"$\beta$", "delta": r"$\delta$", "crra": r"$\rho$", "R_gamma": r"$R_\gamma$"}


def scale(th: np.ndarray) -> np.ndarray:
    """Columns on which distances are taken: delta as log(1 - delta)."""
    th = np.array(th, dtype=float)
    th[:, 1] = np.log1p(-np.minimum(th[:, 1], 1 - 1e-9))
    return th


def sbc_draws(npz: Path) -> dict:
    """The SBC households grouped by draw, (draws, 16, 4), with each draw's
    card type read from the shards and checked against the truth."""
    z = np.load(npz)
    order = np.argsort(z["pid"], kind="stable")
    pid = z["pid"][order].reshape(-1, PER_DRAW)
    if not (pid == pid[:, :1]).all():
        raise SystemExit(f"{npz}: draws do not have {PER_DRAW} households each")
    pid = pid[:, 0]
    g = {k: z[k][order].reshape(len(pid), PER_DRAW, -1) for k in ("mean", "sd", "lo", "hi")}
    g["truth"] = z["truth"][order].reshape(len(pid), PER_DRAW, -1)[:, 0]
    theta, card = [], []
    for f in sorted(SBC_SHARDS.glob("shard_*.npz")):
        s = np.load(f)
        theta.append(s["theta"])
        card.append(s["card"])
    theta, card = np.concatenate(theta), np.concatenate(card)
    if not np.allclose(theta[pid], g["truth"]):
        raise SystemExit("SBC draws do not match the shards' theta")
    g["card"] = card[pid]
    return g


def psid(d: str) -> dict:
    z = np.load(Path(d) / "posterior_uncorrected.npz")
    if tuple(z["names"]) != NAMES:
        raise SystemExit(f"{d}: parameters {tuple(z['names'])}")
    ok = np.isfinite(z["mean"]).all(1)
    return {k: z[k][ok] for k in ("mean", "sd", "lo", "hi")}


def toward(sim: dict, means: np.ndarray) -> np.ndarray:
    """Draw weights: the density of PSID's posterior means at each draw's
    average posterior mean, standardised by PSID's spread."""
    p = scale(means)
    mu, sd = p.mean(0), p.std(0)
    kde = stats.gaussian_kde(((p - mu) / sd).T)
    w = kde((((scale(sim["mean"].mean(1)) - mu) / sd)).T)
    return w / w.sum()


def card_shift(sim: dict, w: np.ndarray) -> np.ndarray:
    """Card's shift of the draw-average posterior mean, per parameter,
    averaged over the weights: a quadratic in the scaled truth plus card and
    card x truth."""
    t = scale(sim["truth"])
    t = (t - t.mean(0)) / t.std(0)
    c = sim["card"][:, None].astype(float)
    quad = [t[:, i] * t[:, j] for i in range(4) for j in range(i, 4)]
    X = np.column_stack([np.ones(len(t)), t, *quad, c, c * t])
    X1 = X.copy()
    X1[:, -5:] = np.column_stack([np.ones(len(t)), t])
    X0 = X.copy()
    X0[:, -5:] = 0
    y = sim["mean"].mean(1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return w @ ((X1 - X0) @ beta)


def others_spread(sim: dict, w: np.ndarray | None = None, n_boot: int = 0,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per parameter k: the variance of the draw-average posterior mean of k
    that the other three parameters produce with k held fixed. Draws are
    sorted by k's truth and taken two by two: the two share k to within
    1/500 of its range and differ in the rest as independent uniform draws
    do, so half the squared difference of their averages, less the averages'
    own noise, is that variance. Pairs are disjoint, so they are independent
    and the bootstrap (``n_boot`` replicates, (n_boot, 4)) resamples them.
    With ``w`` each pair is weighted by its draws' mean weight. No model of
    the map from theta to the posterior mean is assumed: a cubic in the truth
    left 4-9 times the averages' noise variance unexplained."""
    t = scale(sim["truth"])
    mbar = sim["mean"].mean(1)
    noise = sim["mean"].var(1, ddof=1) / PER_DRAW
    rng = np.random.default_rng(seed)
    est, boot = np.zeros(4), np.zeros((n_boot, 4))
    for k in range(4):
        o = np.argsort(t[:, k])[: len(t) // 2 * 2]
        a, b = o[0::2], o[1::2]
        d2 = (mbar[a, k] - mbar[b, k]) ** 2 - noise[a, k] - noise[b, k]
        pw = np.ones(len(a)) if w is None else (w[a] + w[b]) / 2
        est[k] = max(pw @ d2 / pw.sum() / 2, 0.0)
        if n_boot:
            i = rng.integers(0, len(a), (n_boot, len(a)))
            boot[:, k] = np.maximum((pw[i] * d2[i]).sum(1) / pw[i].sum(1) / 2, 0.0)
    return est, boot


def wvar(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    m = w @ x
    return w @ (x - m) ** 2


def pairs(mean, sd, lo, hi, truth=None, max_pairs=200_000, seed=0):
    """Per parameter: the share of pairs whose 90% intervals do not overlap,
    the share ordered with posterior probability >= 0.9, and against a truth,
    the share of those ordered pairs whose order is right."""
    n = len(mean)
    if n * (n - 1) // 2 <= max_pairs:
        i, j = np.array(list(combinations(range(n), 2))).T
    else:
        rng = np.random.default_rng(seed)
        i, j = rng.integers(0, n, (2, max_pairs))
        keep = i != j
        i, j = i[keep], j[keep]
    out = {}
    for k, name in enumerate(NAMES):
        apart = (lo[i, k] > hi[j, k]) | (lo[j, k] > hi[i, k])
        z = (mean[i, k] - mean[j, k]) / np.hypot(sd[i, k], sd[j, k])
        ordered = np.abs(z) >= Z90
        r = {"intervals_apart": float(apart.mean()), "ordered_p90": float(ordered.mean())}
        if truth is not None:
            right = np.sign(z) == np.sign(truth[i, k] - truth[j, k])
            r["right_when_ordered"] = float(right[ordered].mean()) if ordered.any() else None
            r["right_when_apart"] = float(right[apart].mean()) if apart.any() else None
        out[name] = r
    return out


def extremes(mean: np.ndarray, sd: np.ndarray) -> dict:
    """P(order right) for the households at the 10th and 90th percentile of
    the posterior means, from their own posteriors (normal approximation)."""
    out = {}
    for k, name in enumerate(NAMES):
        o = np.argsort(mean[:, k])
        a, b = o[int(0.1 * (len(o) - 1))], o[int(0.9 * (len(o) - 1))]
        z = (mean[b, k] - mean[a, k]) / np.hypot(sd[a, k], sd[b, k])
        out[name] = {"p10": float(mean[a, k]), "p90": float(mean[b, k]),
                     "p_order": float(stats.norm.cdf(z))}
    return out


def null_vs_psid(sim: dict, p: dict, seed: int) -> dict:
    """The shared-theta null's spread of posterior means against PSID's, with
    a bootstrap over PSID households, SBC draws and neighbour pairs."""
    w = toward(sim, p["mean"])
    within = sim["mean"].var(1, ddof=1)               # (draws, 4)
    shift = card_shift(sim, w)
    null = w @ within
    null_mix = null + 0.25 * shift ** 2
    others, others_boot = others_spread(sim, w, N_BOOT, seed)
    null_own = null_mix + others
    pv = p["mean"].var(0, ddof=1)
    rng = np.random.default_rng(seed)
    boot = []
    for b in range(N_BOOT):
        ip = rng.integers(0, len(p["mean"]), len(p["mean"]))
        idr = rng.integers(0, len(w), len(w))
        wb = w[idr] / w[idr].sum()
        pb = p["mean"][ip].var(0, ddof=1)
        nb = wb @ within[idr] + 0.25 * shift ** 2
        boot.append(np.concatenate([1 - nb / pb, 1 - (nb + others_boot[b]) / pb]))
    boot = np.array(boot)
    # the variance identity on the reweighted SBC population, one household
    # per draw, where Var(truth) is known
    t, m, s = sim["truth"], sim["mean"][:, 0], sim["sd"][:, 0]
    var_true = wvar(scale(t), w)
    ess = 1 / (w ** 2).sum()
    out = {"ess_draws": float(ess)}
    for k, name in enumerate(NAMES):
        out[name] = {
            "psid_sd_of_means": float(np.sqrt(pv[k])),
            "null_sd_same_card": float(np.sqrt(null[k])),
            "null_sd_card_mix": float(np.sqrt(null_mix[k])),
            "card_shift": float(shift[k]),
            "spread_ratio": float(np.sqrt(pv[k] / null_mix[k])),
            "share_not_noise": float(1 - null_mix[k] / pv[k]),
            "share_not_noise_90": np.percentile(boot[:, k], [5, 95]).tolist(),
            "null_sd_own": float(np.sqrt(null_own[k])),
            "share_own": float(1 - null_own[k] / pv[k]),
            "share_own_90": np.percentile(boot[:, 4 + k], [5, 95]).tolist(),
            "reweighted_sim": {
                "sd_of_means": float(np.sqrt(wvar(m[:, k], w))),
                "sd_truth": float(np.sqrt(wvar(t[:, k], w))),
                "corr": float(_wcorr(m[:, k], t[:, k], w)),
                "identity_var_true": float(wvar(m[:, k], w) - w @ s[:, k] ** 2),
                "var_true": float(wvar(t[:, k], w)),
            },
        }
        if k == 1:
            out[name]["reweighted_sim"]["var_true_log1m"] = float(var_true[1])
    return out


def _resample(points: np.ndarray, w: np.ndarray, n: int, seed: int) -> np.ndarray:
    idx = np.random.default_rng(seed).choice(len(points), n, p=w / w.sum())
    return points[idx]


def _contour(ax, pts, i, j, color, label):
    """68% (solid) and 95% (dashed) HPD contours of ``pts`` in the (i, j)
    plane, reflected at delta = 1."""
    from hh_npe.evaluation.plots import _hpd_levels, _kde_grid

    a, b = NAMES[i], NAMES[j]
    lo, hi = (LIMITS[a][0], LIMITS[b][0]), (LIMITS[a][1], LIMITS[b][1])
    X, Y, Z = _kde_grid(pts[:, i], pts[:, j], lo, hi, reflect=(a == "delta", b == "delta"))
    ax.contour(X, Y, Z, levels=_hpd_levels(Z), colors=color, linestyles=["--", "-"])
    ax.plot([], [], color=color, label=label)


def figure(res: dict, arm: str, sim: dict, p: dict, couple: Path, path: Path) -> None:
    """(a, b) One couple's posterior against the spread of couples' posterior
    means and of the posterior means of households sharing one theta;
    (c) the bounds on the share of couples' spread that is real, per arm;
    (d) the variance identity on the SBC set, per arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = np.load(couple)
    cw = np.exp(c["log_w"] - c["log_w"].max())
    post = _resample(c["draws"], cw, 6000, seed=0)
    w = toward(sim, p["mean"])
    dev = sim["mean"] - sim["mean"].mean(1, keepdims=True)
    centre = np.median(p["mean"], 0)
    null = _resample((centre + dev).reshape(-1, 4), np.repeat(w, PER_DRAW), 6000, seed=1)
    null[:, 1] = np.minimum(null[:, 1], 1 - 1e-6)

    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    for a, (i, j) in zip(ax[0], ((0, 2), (0, 1)), strict=True):
        a.scatter(p["mean"][:, i], p["mean"][:, j], s=5, color="0.6", zorder=0,
                  label=f"PSID couples' posterior means ({len(p['mean'])})")
        _contour(a, post, i, j, "#3B75AF", f"one couple's posterior (figure 76, #{int(c['pick'])})")
        _contour(a, p["mean"], i, j, "#EF8636", "couples' posterior means")
        _contour(a, null, i, j, "#519E3E",
                 "posterior means of households sharing one θ, placed at couples' median")
        a.set(xlabel=LABEL[NAMES[i]], ylabel=LABEL[NAMES[j]],
              xlim=VIEW[NAMES[i]], ylim=VIEW[NAMES[j]])
        a.grid(ls=":", lw=0.5)
    ax[0, 1].legend(*ax[0, 0].get_legend_handles_labels(), fontsize=7, loc="lower left")
    ax[0, 0].set_title(f"{arm} arm: 68% (solid) / 95% (dashed)", fontsize=10)

    a = ax[1, 0]
    arms = list(res)
    xs = np.arange(len(NAMES))
    for n_arm, name in enumerate(arms):
        color = ARM_COLOR[name]
        nl = res[name]["couples"]["null"]
        x = xs + (n_arm - 1) * 0.22
        lo = [nl[n]["share_own"] for n in NAMES]
        hi = [nl[n]["share_not_noise"] for n in NAMES]
        lo90 = [nl[n]["share_own_90"][0] for n in NAMES]
        hi90 = [nl[n]["share_not_noise_90"][1] for n in NAMES]
        a.vlines(x, lo90, hi90, color=color, lw=1)
        a.vlines(x, lo, hi, color=color, lw=7, alpha=0.7, label=name)
    a.axhline(0, color="k", lw=0.6)
    a.set_xticks(xs, [LABEL[n] for n in NAMES])
    a.set(ylabel="share of PSID couples' spread of posterior means that is real",
          ylim=(-0.7, 1.0))
    a.set_title("bar: from 'the other three vary over the whole box' (low) to 'households "
                "share the\nother three' (high); line: 90% bootstrap", fontsize=9)
    a.legend(fontsize=8, loc="lower left")
    a.grid(ls=":", lw=0.5, axis="y")

    a = ax[1, 1]
    for n_arm, name in enumerate(arms):
        color = ARM_COLOR[name]
        sm = res[name]["simulated"]
        plus = [sm[n]["var_means_plus_post"] / sm[n]["var_truth"] for n in NAMES]
        minus = [sm[n]["var_means_minus_post"] / sm[n]["var_truth"] for n in NAMES]
        x = xs + (n_arm - 1) * 0.13
        a.bar(x - 0.2, plus, 0.12, color=color, label=f"{name}: Var(means) + E[post var]")
        a.bar(x + 0.2, minus, 0.12, color=color, hatch="//", fill=False, edgecolor=color,
              label=f"{name}: Var(means) − E[post var] (RESULTS 1)")
    a.axhline(1, color="k", lw=0.6, ls="--")
    a.axhline(0, color="k", lw=0.6)
    a.set_xticks(xs, [LABEL[n] for n in NAMES])
    a.set(ylabel="÷ the truth's variance", ylim=(-0.5, 1.6))
    a.set_title("simulated households over the prior box (SBC, truth known): the identity "
                "adds;\nRESULTS 1's test subtracts, and reads below zero for β in every arm",
                fontsize=9)
    a.legend(fontsize=6.5, ncol=2, loc="upper center")
    a.grid(ls=":", lw=0.5, axis="y")
    fig.suptitle("Household resolution (RESULTS 60): one couple's posterior is wider than the "
                 "spread of all couples' estimates in β and ρ;\ncouples still differ by more "
                 "than households sharing one θ", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _wcorr(a, b, w):
    am, bm = w @ a, w @ b
    return (w @ ((a - am) * (b - bm))) / np.sqrt(w @ (a - am) ** 2 * (w @ (b - bm) ** 2))


def simulated(sim: dict) -> dict:
    """Truth known, over the uniform prior box: correlation of the posterior
    mean with the truth, R^2, and the noise and signal parts of the spread of
    posterior means."""
    t = np.repeat(sim["truth"][:, None], PER_DRAW, 1)
    est, tru = scale(sim["mean"].reshape(-1, 4)), scale(t.reshape(-1, 4))
    cross = np.corrcoef(np.column_stack([est, tru]).T)[:4, 4:]
    out = {"corr_estimate_row_truth_column": cross.tolist(),
           "others_sd_box": np.sqrt(others_spread(sim)[0]).tolist()}
    for k, name in enumerate(NAMES):
        m, tt = sim["mean"][:, :, k], t[:, :, k]
        noise = m.var(1, ddof=1).mean()
        out[name] = {
            "corr": float(np.corrcoef(m.ravel(), tt.ravel())[0, 1]),
            "r2": float(1 - ((tt - m) ** 2).mean() / tt.var()),
            "sd_of_means": float(m.std()),
            "within_theta_sd": float(np.sqrt(noise)),
            "between_theta_sd": float(np.sqrt(max(m.mean(1).var() - noise / PER_DRAW, 0))),
            "rms_posterior_sd": float(np.sqrt((sim["sd"][:, :, k] ** 2).mean())),
            "truth_sd": float(tt.std()),
            # the population here is the prior, so Var(truth) = Var(means)
            # + E[posterior var]; RESULTS 1's test subtracts instead
            "var_truth": float(tt.var()),
            "var_means_plus_post": float(m.var() + (sim["sd"][:, :, k] ** 2).mean()),
            "var_means_minus_post": float(m.var() - (sim["sd"][:, :, k] ** 2).mean()),
        }
    return out


def interval_vs_range(p: dict) -> dict:
    lo5, hi95 = np.percentile(p["mean"], [5, 95], axis=0)
    width = np.median(p["hi"] - p["lo"], axis=0)
    return {n: {"median_90_width": float(width[k]), "range_5_95_of_means": float(hi95[k] - lo5[k]),
                "ratio": float(width[k] / (hi95[k] - lo5[k])),
                "median_posterior_sd": float(np.median(p["sd"][:, k]))}
            for k, n in enumerate(NAMES)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--figure", type=Path, default=FIG)
    ap.add_argument("--figure_arm", default="gross", choices=list(ARMS))
    ap.add_argument("--couple", type=Path, default=COUPLE,
                    help="Weighted posterior draws of one couple of --figure_arm "
                         "(plot_optionA2_households.py's .npz).")
    args = ap.parse_args()

    res, keep = {}, {}
    for arm in args.arms:
        run, pc, pa = ARMS[arm]
        sim = sbc_draws(Path(run) / "sbc" / "sbc16.npz")
        keep[arm] = (sim, psid(pc))
        flat = {k: sim[k].reshape(-1, 4) for k in ("mean", "sd", "lo", "hi")}
        r = {"simulated": simulated(sim),
             "sim_pairs_box": pairs(**flat, truth=np.repeat(sim["truth"], PER_DRAW, 0)),
             "sim_pairs_same_theta": _same_theta_pairs(sim)}
        for label, d in (("couples", pc), ("all 889", pa)):
            p = psid(d)
            r[label] = {"interval_vs_range": interval_vs_range(p),
                        "pairs": pairs(p["mean"], p["sd"], p["lo"], p["hi"]),
                        "extremes": extremes(p["mean"], p["sd"]),
                        "null": null_vs_psid(sim, p, seed=0)}
        res[arm] = r
    report(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")
    if args.figure_arm in keep and args.couple.exists():
        figure(res, args.figure_arm, *keep[args.figure_arm], args.couple, args.figure)
        print(f"wrote {args.figure}")


def _same_theta_pairs(sim: dict) -> dict:
    """Pairs of households from the same draw: any order claimed is wrong."""
    n = len(sim["truth"])
    i, j = np.array(list(combinations(range(PER_DRAW), 2))).T
    out = {}
    for k, name in enumerate(NAMES):
        m, s = sim["mean"][:, :, k], sim["sd"][:, :, k]
        lo, hi = sim["lo"][:, :, k], sim["hi"][:, :, k]
        apart = (lo[:, i] > hi[:, j]) | (lo[:, j] > hi[:, i])
        z = (m[:, i] - m[:, j]) / np.hypot(s[:, i], s[:, j])
        out[name] = {"intervals_apart": float(apart.mean()),
                     "ordered_p90": float((np.abs(z) >= Z90).mean()), "n_draws": n}
    return out


def report(res: dict) -> None:
    def row(label, vals, fmt="{:9.3f}"):
        print(f"  {label:44s}" + "".join(fmt.format(v) for v in vals))

    def col(d, *keys):
        """``d[name][keys...]`` for each parameter."""
        out = []
        for n in NAMES:
            v = d[n]
            for k in keys:
                v = v[k]
            out.append(v)
        return out

    sd4, var5 = "{:9.4f}", "{:+9.5f}"
    print(f"{'':46s}" + "".join(f"{n:>9s}" for n in NAMES))
    for arm, r in res.items():
        print(f"\n=== {arm} ===")
        s = r["simulated"]
        print(" simulated households, truth known (uniform prior box, SBC;"
              " delta as log(1 - delta) in corr)")
        row("corr(posterior mean, truth)", col(s, "corr"))
        row("R^2", col(s, "r2"))
        row("sd of posterior means", col(s, "sd_of_means"), sd4)
        row("  of which within one theta (noise)", col(s, "within_theta_sd"), sd4)
        row("rms posterior sd", col(s, "rms_posterior_sd"), sd4)
        row("Var(truth)", col(s, "var_truth"), var5)
        row("  Var(means) + E[posterior var]", col(s, "var_means_plus_post"), var5)
        row("  Var(means) - E[posterior var] (RESULTS 1)", col(s, "var_means_minus_post"), var5)
        for k, n in enumerate(NAMES):
            row(f"corr({n} estimate, each truth)", s["corr_estimate_row_truth_column"][k],
                "{:+9.3f}")
        row("sd of means from the other three (box)", s["others_sd_box"], sd4)
        row("pairs ordered at p>=0.9, box", col(r["sim_pairs_box"], "ordered_p90"))
        row("  of which right", col(r["sim_pairs_box"], "right_when_ordered"))
        row("pairs ordered at p>=0.9, same theta", col(r["sim_pairs_same_theta"], "ordered_p90"))
        for label in ("couples", "all 889"):
            p = r[label]
            nl = p["null"]
            print(f" PSID {label}  (null: {nl['ess_draws']:.0f} effective SBC draws)")
            row("one household's 90% interval / 5-95% range", col(p["interval_vs_range"], "ratio"))
            row("pairs with 90% intervals apart", col(p["pairs"], "intervals_apart"))
            row("pairs ordered at p>=0.9", col(p["pairs"], "ordered_p90"))
            row("10th vs 90th pct household: P(order)", col(p["extremes"], "p_order"))
            row("sd of posterior means, PSID", col(nl, "psid_sd_of_means"), sd4)
            row("  shared theta, same card", col(nl, "null_sd_same_card"), sd4)
            row("  shared theta, 50/50 card mix", col(nl, "null_sd_card_mix"), sd4)
            row("share of PSID's spread not noise", col(nl, "share_not_noise"))
            row("  90% bootstrap, low", col(nl, "share_not_noise_90", 0))
            row("  90% bootstrap, high", col(nl, "share_not_noise_90", 1))
            row("  k shared, the rest free (neighbour pairs)", col(nl, "null_sd_own"), sd4)
            row("share of PSID's spread beyond that", col(nl, "share_own"))
            row("  90% bootstrap, low", col(nl, "share_own_90", 0))
            row("  90% bootstrap, high", col(nl, "share_own_90", 1))
            row("sim reweighted to PSID: sd of means", col(nl, "reweighted_sim", "sd_of_means"),
                sd4)
            row("  sd of truth", col(nl, "reweighted_sim", "sd_truth"), sd4)
            row("  corr(posterior mean, truth)", col(nl, "reweighted_sim", "corr"))
            row("  Var(true), identity", col(nl, "reweighted_sim", "identity_var_true"), var5)
            row("  Var(true), actual", col(nl, "reweighted_sim", "var_true"), var5)


if __name__ == "__main__":
    main()
