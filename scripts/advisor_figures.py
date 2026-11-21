"""Four clean figures for the advisor, from outputs already on disk (no
training, no new sampling). β, δ and ρ only: R_γ is a nuisance parameter
(the user, 2026-11-14).

* 179: the population, PSID couples' representative θ under §1's reference
  (v2_fresh_rop) against Laibson et al.'s estimate, 68/95% (RESULTS 69.8,
  70.5; ``outputs/lr_plateau/laibson_comparison.json``).
* 180: one PSID couple's posterior under v2_fresh_rop, the couple nearest the
  couples' median (figure 148's couple and draws).
* 181: one held-out simulated household's posterior against its true θ under
  v2_fresh_rop (figure 147's household and draws, picked by rule).
* 183: figure 181's household under 20 annual waves (a20, RESULTS 71): its own
  panel cut as 20 annual waves from the same start age, with its own draw of
  v2's measurement error; the only figure here that samples (once, on the GPU
  if there is one; cached in ``--annual_cache``).
* 184: figures 181 and 183 overlaid.
* 182: 7 biennial waves (w07, v2_fresh_rop) against 20 annual waves (a20,
  RESULTS 71): every simulated household's posterior sd, the share of prior
  variance removed, β's posterior sd against the true β, and pairs of
  households ordered (RESULTS 71.6; ``outputs/waves_default``).

Usage::

    PYTHONPATH=.:src .venv/bin/python scripts/advisor_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hh_npe.evaluation.plots import _hpd_levels, _kde_grid
from hh_npe.evaluation.weighted import resample
from scripts.literature_ranges import LAIBSON

NAMES = ("beta", "delta", "crra")          # R_gamma left out
SYM = {"beta": "β", "delta": "δ", "crra": "ρ"}
#: the prior box (RESULTS 63: β from 0.10)
BOX = {"beta": (0.1, 1.0), "delta": (0.85, 1.0), "crra": (0.5, 5.0)}
OURS, LAI, TRUTH = "#1f5fa8", "0.15", "#c0392b"
BIENNIAL, ANNUAL = "#1f5fa8", "#e07b00"
REFERENCE = "v2_fresh_rop"
#: the annual window's measurement-error draw (figure 183)
ANNUAL_SEED = 183_000
#: figure 147's pool of held-out households (plot_optionA2_households --pool)
POOL = 300


def style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "xtick.direction": "out", "ytick.direction": "out",
                         "legend.frameon": False, "savefig.bbox": "tight"})


def half_up(v: float, nd: int) -> str:
    """``v`` to ``nd`` decimals, halves rounded up (0.5305 -> 0.531, as
    published; the float 0.5305 would otherwise print as 0.530)."""
    return f"{np.floor(v * 10 ** nd + 0.5 + 1e-9) / 10 ** nd:.{nd}f}"


# ----------------------------------------------------------------- the corner

def corner_axes(fig, rect=(0.08, 0.115, 0.60, 0.735)):
    """A 3x3 lower-triangle corner: ``ax[r][c]`` for r >= c (x: parameter c,
    y: parameter r; the diagonal holds parameter r's marginal)."""
    x0, y0, w, h = rect
    gap = 0.012
    s = (w - 2 * gap) / 3, (h - 2 * gap) / 3
    ax = [[None] * 3 for _ in range(3)]
    for r in range(3):
        for c in range(r + 1):
            ax[r][c] = fig.add_axes((x0 + c * (s[0] + gap), y0 + (2 - r) * (s[1] + gap),
                                     s[0], s[1]))
    return ax


def kde_1d(x: np.ndarray, lo: float, hi: float, n: int = 300, reflect=True):
    from scipy.stats import gaussian_kde
    g = np.linspace(lo, hi, n)
    k = gaussian_kde(x)
    y = k(g)
    if reflect:                              # boundary correction at the box's edges
        y = y + k(2 * lo - g) + k(2 * hi - g)
    return g, y / np.trapezoid(y, g)


def draw_samples(ax, d: np.ndarray, color: str, lim: dict, label=None, fill=True,
                 reflect=("delta",), lw=1.6):
    """68% (filled, solid) and 95% (dashed) regions of draws ``d`` (columns in
    NAMES order), and each marginal on the diagonal."""
    for r in range(3):
        for c in range(r):
            nx, ny = NAMES[c], NAMES[r]
            X, Y, Z = _kde_grid(d[:, c], d[:, r], (lim[nx][0], lim[ny][0]),
                                (lim[nx][1], lim[ny][1]), n=160,
                                reflect=(nx in reflect and lim[nx][1] >= BOX[nx][1],
                                         ny in reflect and lim[ny][1] >= BOX[ny][1]))
            if Z is None:
                continue
            l95, l68 = _hpd_levels(Z)
            if fill:
                ax[r][c].contourf(X, Y, Z, levels=[l68, Z.max()], colors=[color], alpha=0.28)
            ax[r][c].contour(X, Y, Z, levels=[l68], colors=[color], linewidths=lw)
            ax[r][c].contour(X, Y, Z, levels=[l95], colors=[color], linewidths=lw * 0.8,
                             linestyles="--")
    for r, p in enumerate(NAMES):
        g, y = kde_1d(d[:, r], *lim[p], reflect=lim[p] == BOX[p])
        ax[r][r].plot(g, y, color=color, lw=lw, label=label)
        if fill:
            ax[r][r].fill_between(g, y, color=color, alpha=0.18, lw=0)


def ellipse_xy(mean, cov, i, j, q):
    t = np.linspace(0, 2 * np.pi, 300)
    w, v = np.linalg.eigh(np.asarray(cov)[np.ix_([i, j], [i, j])])
    e = np.asarray(mean)[[i, j]][:, None] + v @ (np.sqrt(np.maximum(w, 0) * q)[:, None]
                                                 * np.vstack([np.cos(t), np.sin(t)]))
    return e


def draw_gaussian(ax, mean, cov, color, lim, ls68="-", ls95="--", lw=1.4, label=None,
                  fill=None):
    """68/95% ellipses of a normal (mean, cov over NAMES) and its marginals."""
    from scipy.stats import norm
    for r in range(3):
        for c in range(r):
            for q, ls, f in ((2.30, ls68, True), (5.99, ls95, False)):
                e = ellipse_xy(mean, cov, c, r, q)
                ax[r][c].plot(e[0], e[1], color=color, ls=ls, lw=lw)
                if fill and f:
                    ax[r][c].fill(e[0], e[1], color=fill, alpha=0.12, lw=0)
    for r, p in enumerate(NAMES):
        g = np.linspace(*lim[p], 400)
        ax[r][r].plot(g, norm.pdf(g, mean[r], np.sqrt(cov[r][r])), color=color, ls=ls68, lw=lw,
                      label=label)


def mark(ax, theta, **kw):
    for r in range(3):
        for c in range(r):
            ax[r][c].plot(theta[c], theta[r], ls="none", **kw)
        ax[r][r].axvline(theta[r], color=kw.get("color", "k"), lw=1, ls=":")


def finish(ax, lim, ticks=None):
    for r in range(3):
        for c in range(r + 1):
            a = ax[r][c]
            a.set_xlim(*lim[NAMES[c]])
            if r != c:
                a.set_ylim(*lim[NAMES[r]])
            else:
                a.set_ylim(bottom=0)
                a.set_yticks([])
                a.spines["left"].set_visible(False)
            if ticks and NAMES[c] in ticks:
                a.set_xticks(ticks[NAMES[c]])
            if ticks and r != c and NAMES[r] in ticks:
                a.set_yticks(ticks[NAMES[r]])
            if r < 2:
                a.set_xticklabels([])
            else:
                a.set_xlabel(SYM[NAMES[c]])
            if c > 0 or r == 0:
                if r != c:
                    a.set_yticklabels([])
            else:
                a.set_ylabel(SYM[NAMES[r]])
            a.tick_params(labelsize=9)


def diag_titles(ax, d: np.ndarray, fmt=("{:.2f}", "{:.3f}", "{:.2f}")):
    """Posterior mean and 90% interval above each marginal."""
    for r, p in enumerate(NAMES):
        lo, hi = np.percentile(d[:, r], [5, 95])
        f = fmt[r]
        ax[r][r].set_title(f"{SYM[p]} = {f.format(d[:, r].mean())}  "
                           f"[{f.format(lo)}, {f.format(hi)}]", fontsize=10)


# ----------------------------------------------------------------- 179

def laibson(args) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    d = json.loads(args.laibson.read_text())
    arm = d["arms"][REFERENCE]
    th_l = LAIBSON                            # their table 3 (0.5305, 0.9891, 1.9355)
    cov_rep, cov_smp = (np.array(d["laibson_cov"][k]) for k in ("reported", "sampling only"))
    se_rep, se_smp = (np.array(d["laibson_se"][k]) for k in ("reported", "sampling only"))
    boot = np.array(arm["boot"])[:, :3]
    th_o, se_o = np.array(arm["representative_theta"][:3]), np.array(arm["representative_se"][:3])
    lim = {"beta": (0.2, 0.86), "delta": (0.975, 1.0), "crra": (0.8, 3.15)}

    fig = plt.figure(figsize=(10.5, 9.2))
    ax = corner_axes(fig)
    draw_gaussian(ax, th_l, cov_rep, LAI, lim, lw=1.3, label="reported", fill="0.5")
    draw_gaussian(ax, th_l, cov_smp, LAI, lim, ls68=":", ls95=":", lw=1.3,
                  label="sampling only")
    draw_samples(ax, boot, OURS, lim, reflect=())
    mark(ax, th_l, marker="*", ms=13, color=LAI, mec="white", mew=0.6)
    mark(ax, th_o, marker="o", ms=6, color=OURS, mec="white", mew=0.6)
    finish(ax, lim, ticks={"beta": [0.3, 0.5, 0.7], "delta": [0.98, 0.99, 1.0],
                           "crra": [1, 2, 3]})
    for r in range(3):
        ax[r][r].set_ylim(0, None)

    handles = [Patch(fc=OURS, alpha=0.4, ec=OURS, label="Ours: PSID couples' representative θ"),
               Line2D([], [], color=LAI, lw=1.3, label="Laibson et al.: reported covariance"),
               Line2D([], [], color=LAI, lw=1.3, ls=":",
                      label="Laibson et al.: sampling-only covariance"),
               Line2D([], [], color="k", lw=1.3, label="68% region"),
               Line2D([], [], color="k", lw=1.0, ls="--", label="95% region")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.36, 0.86), fontsize=10)
    nd = (3, 4, 2)                            # decimals of β, δ, ρ

    def vals(v, se=False):                    # standard errors: one more decimal for ρ
        return [f"({half_up(x, n + (j == 2))})" if se else half_up(x, n)
                for j, (x, n) in enumerate(zip(v, nd, strict=True))]

    rows = [("Ours: 409 PSID couples", vals(th_o), OURS),
            ("   standard error", vals(se_o, True), OURS),
            ("Laibson et al.: 5,710 SCF hh.", vals(th_l), LAI),
            ("   reported se", vals(se_rep, True), LAI),
            ("   sampling-only se", vals(se_smp, True), LAI)]
    cols, y0, dy = (0.795, 0.875, 0.955), 0.585, 0.034
    for x, p in zip(cols, NAMES, strict=True):
        fig.text(x, y0, SYM[p], ha="center", fontsize=11, weight="bold")
    for k, (label, vals, color) in enumerate(rows, start=1):
        fig.text(0.51, y0 - k * dy, label, fontsize=10, color=color)
        for x, v in zip(cols, vals, strict=True):
            fig.text(x, y0 - k * dy, v, ha="center", fontsize=10, color=color)
    ratio = se_smp / se_o
    fig.text(0.07, 0.955, "The population estimate: PSID couples against Laibson et al.",
             fontsize=15, weight="bold")
    fig.text(0.07, 0.925, f"Our standard errors are {ratio.min():.1f}–{ratio.max():.1f} times "
             "smaller than their sampling-only ones, from 409 couples against 5,710 SCF "
             "households", fontsize=11)
    fig.text(0.07, 0.015,
             "Ours: the single θ whose simulated households the network reads as it reads PSID "
             "couples (7 biennial waves, 2011–2023); 68/95% regions from 1,000 bootstrap\n"
             "resamples of couples. Laibson et al.: method of simulated moments; normal regions "
             "from their reported covariance (solid) and its sampling-only part (dotted), the\n"
             "like-for-like comparison, since our errors hold the first stage fixed. Network: "
             f"{REFERENCE}, under the reference noise model (RESULTS 69.8, 70.5).",
             fontsize=8.5, color="0.3", va="bottom")
    out = args.fig_dir / "179_advisor_laibson_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------------------------------- 180, 181

def household(args, which: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    z = annual_draws(args) if which == "sim20" else np.load(
        args.sim_draws if which == "sim" else args.psid_draws)
    color = ANNUAL if which == "sim20" else OURS
    d = resample(z["draws"], z["log_w"], args.n_plot, seed=0)[:, :3]
    lim = dict(BOX)
    fig = plt.figure(figsize=(10.5, 9.2))
    ax = corner_axes(fig)
    draw_samples(ax, d, color, lim)
    mean = d.mean(0)
    handles = [Patch(fc=color, alpha=0.4, ec=color,
                     label="posterior: 68% (filled), 95% (dashed)"),
               Line2D([], [], color="k", marker="X", ls="none", ms=8, label="posterior mean")]
    if which in ("sim", "sim20"):
        truth = np.asarray(z["truth"])[:3]
        mark(ax, truth, marker="*", ms=16, color=TRUTH, mec="white", mew=0.7, zorder=5)
        handles.insert(1, Line2D([], [], color=TRUTH, marker="*", ls="none", ms=14,
                                 label="true θ (generated the data)"))
        lo, hi = np.percentile(d, [5, 95], axis=0)
        covered = ((truth >= lo) & (truth <= hi)).all()
        inside = ("all three parameters." if covered else "some parameters only.")
        if which == "sim":
            title = "One simulated household: posterior against the truth"
            sub = ("A household held out from training, 7 biennial waves (ages 34–46). The true "
                   "θ lies inside the 90% interval\nfor " + inside)
            note = ("Picked by rule, not by eye: among 300 held-out households, the one with the "
                    "median error of the posterior mean (true θ at least 10% inside the prior "
                    "box).\nAxes span the prior box. Above each marginal: posterior mean and 90% "
                    f"interval. Network: {REFERENCE} (5-member ensemble; card type marginalised).")
            name = "181_advisor_simulated_household.png"
        else:
            ages = np.asarray(z["ages"]).astype(int)
            title = "The same household, 20 annual waves: posterior against the truth"
            sub = (f"Figure 181's household observed every year from age {ages[0]} to "
                   f"{ages[-1]} (20 annual waves) instead of every other year\nfrom 34 to 46. "
                   "The true θ lies inside the 90% interval for " + inside)
            note = ("The same simulated household and true θ as figure 181: its panel cut as 20 "
                    "annual waves from the same start age, with its own draw of the measurement "
                    "error.\nIt was picked under 7 biennial waves (figure 181), not re-picked "
                    "here. Axes span the prior box. Above each marginal: posterior mean and 90% "
                    "interval.\nNetwork: RESULTS 71's a20 (5-member ensemble, trained like "
                    f"{REFERENCE}; card type marginalised).")
            name = "183_advisor_simulated_household_annual20.png"
    else:
        lai = LAIBSON
        mark(ax, lai, marker="*", ms=15, color="0.55", mec="white", mew=0.7, zorder=4)
        handles.append(Line2D([], [], color="0.55", marker="*", ls="none", ms=13,
                              label="Laibson et al.'s population estimate (reference)"))
        title = "One PSID couple: what seven waves tell"
        sub = ("A real couple observed in 7 biennial waves, 2011–2023 (ages 34–46): the couple "
               "whose posterior mean is\nnearest the median of all 409 couples'. No true θ "
               "exists for real data.")
        note = ("Axes span the prior box. Above each marginal: posterior mean and 90% interval. "
                f"Network: {REFERENCE} (5-member ensemble; card type marginalised),\nunder the "
                "reference noise model. Laibson et al.'s point is one θ for the whole population, "
                "shown for scale only.")
        name = "180_advisor_psid_couple.png"
    mark(ax, mean, marker="X", ms=9, color="k", mec="white", mew=0.6, zorder=6)
    finish(ax, lim, ticks={"beta": [0.2, 0.4, 0.6, 0.8, 1.0], "delta": [0.88, 0.92, 0.96, 1.0],
                           "crra": [1, 2, 3, 4, 5]})
    diag_titles(ax, d)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.52, 0.82), fontsize=10.5)
    fig.text(0.07, 0.955, title, fontsize=15, weight="bold")
    fig.text(0.07, 0.905, sub, fontsize=11)
    fig.text(0.07, 0.015, note, fontsize=8.5, color="0.3", va="bottom")
    out = args.fig_dir / name
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def annual_draws(args) -> dict:
    """Figure 147's household under a20: figure 147's pick recovered as
    ``plot_optionA2_households.simulated`` made it, the household's panel
    found among its draw's households by its clean 7-wave window, cut as 20
    annual waves from the same start age, noised with v2's measurement error
    (``ANNUAL_SEED``) and sampled by a20's ensemble as ``optionA psid`` does.
    Cached in ``args.annual_cache`` (outputs/, not committed)."""
    if args.annual_cache.exists():
        return dict(np.load(args.annual_cache))
    import torch

    from hh_npe.data.waves import FEATURES_TWOASSET_AGE
    from hh_npe.data.windows import _panel_of, cut_windows
    from scripts import optionA as oa
    from scripts import waves_experiment as we
    from scripts.plot_optionA2_households import weighted_draws
    from scripts.sbc_optionA2 import Members

    z = np.load(args.sim_draws)
    cfg = json.loads((args.w07_run / "s0" / "results.json").read_text())["_config"]
    spec, win = oa.spec_of(cfg), oa.window_of(cfg)
    theta_all, _c, _t, held_f, _p, _s = oa.load_shards(list(args.shards))
    th, _x, pid = oa.windows_raw(held_f, theta_all, seed=oa.HELD_SEED, noise=spec.obs_noise,
                                 mask=spec.mask_patterns(), window=win)
    pick1 = oa.one_per_draw(pid, seed=0)
    pool = np.sort(np.random.default_rng(1).choice(pick1, min(POOL, len(pick1)), replace=False))
    row = int(pool[int(z["pick"])])
    draw = int(z["draw"])
    if int(pid[row]) != draw or not np.allclose(th[row].numpy(), z["truth"], atol=1e-6):
        raise SystemExit("figure 147's household was not recovered")
    clean = torch.cat([c[2] for c in oa.clean_windows(held_f, theta_all, oa.HELD_SEED,
                                                       window=win)])[row].numpy()
    age0 = int(round(clean[0, -1]))         # the age channel of the first wave
    f = next(s for s in held_f
             if int(np.load(s.path)["lo"]) <= draw - s.offset < int(np.load(s.path)["hi"]))
    d = np.load(f.path)
    lo, hi, local = int(d["lo"]), int(d["hi"]), draw - f.offset
    panel = _panel_of(d)
    m = len(next(iter(panel.values()))) // (hi - lo)
    lag = oa.flow_lag_of([f.path])

    def cut(r, n, wy, s):
        one = {k: v[[r]] for k, v in panel.items()}
        return cut_windows(one, theta_all[f.offset + local][None], np.array([draw]),
                           np.array([[s]]), n, wy, FEATURES_TWOASSET_AGE, "last", lag)[1]

    # the age channel is not the start argument (it depends on the spacing and
    # the flow lag): read the offset off a cut of the same spacing
    probe = next(x for r in (local - lo) * m + np.arange(m)
                 if len(x := cut(r, win["n_waves"], win["wave_years"], age0)))
    start = 2 * age0 - int(round(probe[0, 0, -1]))
    hits = [int(r) for r in (local - lo) * m + np.arange(m)
            if len(x := cut(r, win["n_waves"], win["wave_years"], start))
            and np.allclose(x[0], clean, equal_nan=True)]
    if len(hits) != 1:
        raise SystemExit(f"figure 147's window matches {len(hits)} of its draw's households")
    raw = cut(hits[0], 20, 1, start)
    if len(raw) != 1:
        raise SystemExit("the household does not live through 20 annual waves")
    members = Members(we.members_of(args.a20_run), args.device)
    s20 = oa.spec_of(members.cfg)
    if members.window["wave_years"] != 1 or members.window["n_waves"] != 20 or not (
            members.window["start_low"] <= start <= members.window["start_high"]):
        raise SystemExit(f"{args.a20_run}: not 20 annual waves from age {start}")
    x = oa.add_obs_noise(torch.from_numpy(raw).float(), s20.obs_noise, seed=ANNUAL_SEED)
    xt, _f = oa.transform(x, masked=s20.masked)
    inv, lw = oa._weighting(members.proposal, s20)
    torch.manual_seed(0)
    dr, lwt = weighted_draws(members.nets, xt, args.n_raw, s20.flow_box(), inv, lw)
    out = {"draws": dr, "log_w": lwt, "truth": np.asarray(z["truth"]), "draw": draw,
           "panel_row": hits[0], "start": start, "ages": raw[0, :, -1]}
    args.annual_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.annual_cache, **out)
    print(f"figure 147's household: draw {draw}, panel row {hits[0]}, start age {start}; "
          f"{len(dr):,} a20 draws kept; wrote {args.annual_cache}")
    return out


def compare_household(args) -> None:
    """184: figure 181's household under 7 biennial and 20 annual waves."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    z7, z20 = np.load(args.sim_draws), annual_draws(args)
    d7 = resample(z7["draws"], z7["log_w"], args.n_plot, seed=0)[:, :3]
    d20 = resample(z20["draws"], z20["log_w"], args.n_plot, seed=0)[:, :3]
    truth = np.asarray(z7["truth"])[:3]
    ages = np.asarray(z20["ages"]).astype(int)
    lim = dict(BOX)
    fig = plt.figure(figsize=(10.5, 9.2))
    ax = corner_axes(fig)
    draw_samples(ax, d7, BIENNIAL, lim)
    draw_samples(ax, d20, ANNUAL, lim)
    mark(ax, truth, marker="*", ms=16, color=TRUTH, mec="white", mew=0.7, zorder=5)
    finish(ax, lim, ticks={"beta": [0.2, 0.4, 0.6, 0.8, 1.0], "delta": [0.88, 0.92, 0.96, 1.0],
                           "crra": [1, 2, 3, 4, 5]})
    q7, q20 = (np.percentile(d, [5, 95], axis=0) for d in (d7, d20))
    fmt = ("{:.2f}", "{:.3f}", "{:.2f}")
    for r, p in enumerate(NAMES):
        f = fmt[r]
        ax[r][r].set_title(f"{SYM[p]}, 90% interval\n[{f.format(q7[0, r])}, "
                           f"{f.format(q7[1, r])}] → [{f.format(q20[0, r])}, "
                           f"{f.format(q20[1, r])}]", fontsize=9.5)
    shrink = 1 - (q20[1] - q20[0]) / (q7[1] - q7[0])
    fig.legend(handles=[Patch(fc=BIENNIAL, ec=BIENNIAL, alpha=0.45,
                              label="7 biennial waves, ages 34–46 (PSID's design)"),
                        Patch(fc=ANNUAL, ec=ANNUAL, alpha=0.45,
                              label=f"20 annual waves, ages {ages[0]}–{ages[-1]}"),
                        Line2D([], [], color=TRUTH, marker="*", ls="none", ms=14,
                               label="true θ (generated the data)"),
                        Line2D([], [], color="k", lw=1.6, label="68%"),
                        Line2D([], [], color="k", lw=1.2, ls="--", label="95%")],
               loc="upper left", bbox_to_anchor=(0.52, 0.82), fontsize=10.5)
    fig.text(0.07, 0.955, "One simulated household: 7 biennial against 20 annual waves",
             fontsize=15, weight="bold")
    fig.text(0.07, 0.905, "The same household and true θ under each design. Its 90% interval "
             "narrows by " + ", ".join(f"{100 * s:.0f}% ({SYM[p]})"
                                       for p, s in zip(NAMES, shrink, strict=True))
             + ";\nacross all simulated households the median posterior sd falls by about a "
             "third (figure 182).", fontsize=11)
    fig.text(0.07, 0.015,
             "Figure 181's household (picked by rule under 7 biennial waves) and figure 183's. "
             "The annual panel is the same household's, with its own draw of the measurement\n"
             "error; the annual gain is an upper bound, since that error is drawn afresh every "
             f"wave. Networks: {REFERENCE} (biennial) and RESULTS 71's a20 (annual), trained the "
             "same way.\nAbove each marginal: the 90% interval under each design. ρ's runs into "
             "the top of the prior box (5) under both, which is why it barely narrows here.",
             fontsize=8.5, color="0.3", va="bottom")
    out = args.fig_dir / "184_advisor_simulated_household_biennial_vs_annual.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------------------------------- 182

def information(args) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    res = json.loads((args.waves_root / "waves_default.json").read_text())["arms"]
    sbc = {a: np.load(args.waves_root / a / "sbc" / "sbc16.npz") for a in ("w07", "a20")}
    arms = (("w07", BIENNIAL, "7 biennial waves (14 years; PSID's design)"),
            ("a20", ANNUAL, "20 annual waves (21 years)"))
    fig = plt.figure(figsize=(14, 10.2))
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.98, top=0.86, bottom=0.12,
                          hspace=0.42, wspace=0.28)

    # A: every simulated household's posterior sd, per parameter
    for k, p in enumerate(NAMES):
        a = fig.add_subplot(gs[0, k])
        sd = {arm: sbc[arm]["sd"][:, k] for arm, _c, _l in arms}
        hi = np.percentile(sd["w07"], 99.5)
        bins = np.linspace(0, hi, 60)
        for arm, color, _l in arms:
            a.hist(sd[arm], bins=bins, density=True, color=color, alpha=0.35)
            a.hist(sd[arm], bins=bins, density=True, histtype="step", color=color, lw=1.6)
            a.axvline(np.median(sd[arm]), color=color, lw=1.5, ls="--")
        m0, m1 = np.median(sd["w07"]), np.median(sd["a20"])
        f = "{:.3f}" if p == "delta" else "{:.2f}"
        a.text(0.97, 0.95, f"median sd\n{f.format(m0)} → {f.format(m1)}\n(×{m1 / m0:.2f})",
               transform=a.transAxes, ha="right", va="top", fontsize=10.5)
        a.set(xlim=(0, hi), xlabel=f"posterior sd of {SYM[p]}", yticks=[])
        a.spines["left"].set_visible(False)
        if k == 0:
            a.set_title("A   How uncertain each household's estimate is (all 20,992)",
                        loc="left", fontsize=12, weight="bold", x=0.0)
    fig.legend(handles=[Patch(fc=c, ec=c, alpha=0.45, label=lab) for _a, c, lab in arms]
               + [Line2D([], [], color="k", lw=1.4, ls="--", label="median")],
               loc="upper center", bbox_to_anchor=(0.52, 0.915), ncol=3, fontsize=10.5)

    # B: share of prior variance removed, with the information ratio
    a = fig.add_subplot(gs[1, 0])
    x = np.arange(3)
    s = {arm: np.array(res[arm]["heldout"]["contraction"][:3]) for arm, _c, _l in arms}
    for k, (arm, color, _l) in enumerate(arms):
        a.bar(x + (k - 0.5) * 0.36, s[arm], 0.36, color=color, alpha=0.85)
        for xi, v in zip(x, s[arm], strict=True):
            a.text(xi + (k - 0.5) * 0.36, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    info = (s["a20"] / (1 - s["a20"])) / (s["w07"] / (1 - s["w07"]))
    for xi, r in zip(x, info, strict=True):
        a.text(xi, 0.90, f"×{r:.1f}", ha="center", fontsize=12, weight="bold", color="0.2")
    a.text(1, 0.985, "information, annual ÷ biennial", ha="center", fontsize=9, color="0.35")
    a.set(xticks=x, xticklabels=[SYM[p] for p in NAMES], ylim=(0, 1.05),
          ylabel="share of prior variance removed")
    a.set_title("B   How much the data narrow one household", loc="left", fontsize=12,
                weight="bold", x=-0.12)

    # C: β's posterior sd against the true β
    a = fig.add_subplot(gs[1, 1])
    edges = np.linspace(0.1, 1.0, 10)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for arm, color, _l in arms:
        tb, sd = sbc[arm]["truth"][:, 0], sbc[arm]["sd"][:, 0]
        q = np.array([np.percentile(sd[(tb >= lo) & (tb < hi)], [25, 50, 75])
                      for lo, hi in zip(edges[:-1], edges[1:], strict=True)])
        a.fill_between(mid, q[:, 0], q[:, 2], color=color, alpha=0.2, lw=0)
        a.plot(mid, q[:, 1], "o-", color=color, lw=1.8, ms=4)
    prior_sd = 0.9 / np.sqrt(12)
    a.axhline(prior_sd, color="0.5", ls=":", lw=1)
    a.text(0.12, prior_sd + 0.004, "prior sd", fontsize=9, color="0.4")
    a.set(xlabel="true β", ylabel="posterior sd of β (median, IQR)", xlim=(0.1, 1.0),
          ylim=(0, 0.3))
    a.set_title("C   β's uncertainty, across the box", loc="left", fontsize=12,
                weight="bold", x=-0.12)

    # D: pairs of households ordered
    a = fig.add_subplot(gs[1, 2])
    for k, (arm, color, _l) in enumerate(arms):
        v = np.array([res[arm]["sbc"]["pairs_box"][p]["ordered_p90"] for p in NAMES]) * 100
        a.bar(x + (k - 0.5) * 0.36, v, 0.36, color=color, alpha=0.85)
        for xi, vi in zip(x, v, strict=True):
            a.text(xi + (k - 0.5) * 0.36, vi + 1, f"{vi:.0f}%", ha="center", fontsize=9)
    a.set(xticks=x, xticklabels=[SYM[p] for p in NAMES], ylim=(0, 62),
          ylabel="pairs ordered with ≥ 90% probability (%)")
    a.set_title("D   Telling two households apart", loc="left", fontsize=12, weight="bold",
                x=-0.16)

    fig.text(0.06, 0.965, "What 20 annual waves tell, against PSID's 7 biennial waves",
             fontsize=16, weight="bold")
    fig.text(0.06, 0.937, "Simulated households under the reference model; each design has "
             "its own network, trained the same way and calibrated (90% coverage 0.89–0.91).",
             fontsize=11)
    pse = (np.array(res["a20"]["population"]["se"][:3])
           / np.array(res["w07"]["population"]["se"][:3]))
    fig.text(0.06, 0.015,
             "Information = s/(1−s), s the share of prior variance removed (held-out "
             "households). A, C and D: 20,992 simulated households (1,312 θ draws over the "
             "prior box, 16 each).\n"
             "The annual gain is an upper bound: the simulated measurement error is drawn afresh "
             "every wave. The annual window reads 21 years against 14. For a population of 409\n"
             "households sharing one θ, the standard error at 20 annual waves is "
             + ", ".join(f"{SYM[p]} {v:.2f}" for p, v in zip(NAMES, pse, strict=True))
             + f" times that at 7 biennial waves. Networks: {REFERENCE} (biennial) and RESULTS "
             "71's a20 (annual); RESULTS 71.6–71.8.", fontsize=8.5, color="0.3", va="bottom")
    out = args.fig_dir / "182_advisor_biennial7_vs_annual20.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laibson", type=Path,
                    default=Path("outputs/lr_plateau/laibson_comparison.json"))
    ap.add_argument("--sim_draws", type=Path,
                    default=Path("figures/147_simulated_household_lr_plateau_v2_fresh_rop.npz"))
    ap.add_argument("--psid_draws", type=Path,
                    default=Path("figures/148_psid_household_lr_plateau_v2_fresh_rop.npz"))
    ap.add_argument("--waves_root", type=Path, default=Path("outputs/waves_default"))
    ap.add_argument("--n_plot", type=int, default=20000,
                    help="equally weighted draws resampled, as for figures 147 and 148")
    ap.add_argument("--fig_dir", type=Path, default=Path("figures"))
    ap.add_argument("--w07_run", type=Path, default=Path("outputs/lr_plateau/v2_fresh_rop/w07"))
    ap.add_argument("--a20_run", type=Path, default=Path("outputs/waves_default/a20"))
    ap.add_argument("--shards", type=Path, nargs="+",
                    default=[Path("data/processed/couples_v2_dataset_shards"),
                             Path("data/processed/couples_v2_strip_dataset_shards"),
                             Path("data/processed/couples_v2_betastrip_dataset_shards")])
    ap.add_argument("--annual_cache", type=Path,
                    default=Path("outputs/advisor/annual20_household.npz"))
    ap.add_argument("--n_raw", type=int, default=40000,
                    help="raw draws for figure 183's household, as for figure 147")
    ap.add_argument("--device", default=None)
    figs = ("laibson", "psid", "sim", "information", "sim20", "compare")
    ap.add_argument("--only", nargs="+", choices=figs, default=figs)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    style()
    if "laibson" in args.only:
        laibson(args)
    if "psid" in args.only:
        household(args, "psid")
    if "sim" in args.only:
        household(args, "sim")
    if "information" in args.only:
        information(args)
    if "sim20" in args.only:
        household(args, "sim20")
    if "compare" in args.only:
        compare_household(args)


if __name__ == "__main__":
    main()
