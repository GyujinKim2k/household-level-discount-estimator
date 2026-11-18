"""RESULTS 72: a lower learning-rate floor, four models, annual against
biennial in the same years.

The user, 2026-11-18: "ok let's train with lower floor. give the models chance
to learn more. not all models. let's pick representative models and train
only 4 models. also make sure compare Annual and Biennial with same years not
same waves." RESULTS 71's floor of 1e-5 bound for the long and annual windows
(71.4). RESULTS 71's recipe is kept in everything but the floor, 1e-7 (a new
noise draw every epoch, the rate halved at 5 passes without a gain, a stop
after 20, a cap of 600), on RESULTS 1's model. The four arms are the two
same-years pairs, every year read inside ages 24-64:

* w07 (14 years read) against a14 (15);
* w10 (20 years read) against a20 (21).

Their 1e-5 counterparts are RESULTS 71's (``outputs/waves_default``; w07 is
v2_fresh_rop). Set before the run:

* calibration of beta, delta and rho, held-out and SBC 90% coverage in
  0.88-0.92 (R_gamma reported);
* the floor: held-out log q of 1e-7 minus 1e-5, paired on identical windows,
  per arm, a 95% interval over households resampled ("helps": above 0), the
  four-test Bonferroni interval and the member-level check beside it;
* annual against biennial in about the same years only (a14 / w07, a20 /
  w10), under both floors;
* training from the CSVs and sbi's record: what training below 1e-5 did.

Commands:

``measures``
    The tables (low_floor.json).

``figures``
    161 (precision against years read, both floors, and the same-years
    ratios), 162 (the floor's paired log q gain and information), 167 (one
    held-out household under the four arms, a row per same-years pair,
    68/95%), 168 (the population's 68/95% regions, both floors), 169
    (training curves, both floors).

Usage::

    PYTHONPATH=.:src .venv/bin/python scripts/low_floor.py measures
    PYTHONPATH=.:src .venv/bin/python scripts/low_floor.py figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts import annual_waves as aw
from scripts import fresh_noise as fn
from scripts import low_beta_check as lb
from scripts import lr_plateau as lp
from scripts import noise_copies as nc
from scripts import noise_free_waves as nf
from scripts import training_metrics as tm
from scripts import waves_default as wd
from scripts import waves_experiment as we

#: the two same-years pairs, in training order
ARMS = ("w07", "a14", "w10", "a20")
SAME_YEARS = (("a14", "w07"), ("a20", "w10"))
ROOT = Path("outputs/low_floor")
FLOOR, FLOOR5 = 1e-7, 1e-5
#: the two floors' names in the paired files and the bootstrap
LOW, FIVE = "low", "floor5"
LABEL = {LOW: "floor 1e-7", FIVE: "floor 1e-5 (RESULTS 71)"}
TESTS = len(ARMS)


def floor_rows(root: Path, label: str) -> tuple:
    """``(low, floor5, w)``: each floor's per-household log q on the same
    held-out windows of one arm (cross_noise_heldout.py --per_row). Both
    networks carry v2's noise, so the two noise columns must be the same
    windows."""
    d = dict(np.load(root / "floor" / f"cross_noise_{label}.npz"))
    for net in (LOW, FIVE):
        if not np.array_equal(d[f"{net}__{LOW}"], d[f"{net}__{FIVE}"]):
            raise SystemExit(f"{label}: the two noise columns are not the same windows")
    return d[f"{LOW}__{FIVE}"], d[f"{FIVE}__{FIVE}"], d["w"]


def below(mdir: Path, run: Path, rate: float = FLOOR5) -> dict:
    """Per member, what training at and below ``rate`` did, from the CSV's
    rates and sbi's record of the validation losses (the CSV rounds them,
    RESULTS 71.4): the floor, the lowest rate reached and whether it is the
    floor, passes trained below ``rate``, whether the best came below it, and
    the fall in the best validation loss after the rate first reached
    ``rate`` and after it first went below it."""
    out = {}
    for f in sorted(mdir.glob("s[0-9]*.csv")):
        m = tm.read(f)
        cfg = json.loads((run / f.stem / "results.json").read_text())["_config"]
        v = np.asarray(json.loads((run / f.stem / "training_summary.json").read_text())
                       ["validation_loss"])
        lr, passes = np.asarray(m["lr"]), np.asarray(m["passes"])
        floor = cfg["lr_plateau"]["min_lr"]
        under = lr < rate * (1 - 1e-6)

        def gain(mask, v=v):
            k = np.flatnonzero(mask)
            return None if not len(k) else float(v[:k[0]].min() - v.min())

        out[f.stem] = {"floor": float(floor), "lowest_rate": float(lr.min()),
                       "reached_floor": bool(np.isclose(lr.min(), floor, rtol=1e-5)),
                       "passes_below": float(under.sum() * passes[0]),   # passes per epoch
                       "best_below": bool(under[int(np.argmin(v))]),
                       "gain_from_rate": gain(lr <= rate * (1 + 1e-6)),
                       "gain_below_rate": gain(under)}
    return out


def runs(args) -> dict:
    """Each floor's run directory and CSV directory per arm."""
    out = {}
    for tag, root in ((LOW, args.root), (FIVE, args.floor5_root)):
        for a in ARMS:
            run = we.arm_dir(a, root)
            mdir = (args.w07_metrics if (tag == FIVE and a == "w07")
                    else args.metrics_dir / tm.run_name(run))
            out[tag, a] = (run, mdir)
    return out


def measures(args) -> None:
    """The arms' measures and comparisons (low_floor.json)."""
    from scripts import laibson_comparison as lc

    pops = {LOW: wd.populations(args.root), FIVE: wd.populations(args.floor5_root)}
    th0 = np.array(pops[LOW]["theta0"])
    if not np.allclose(pops[FIVE]["theta0"], th0):
        raise SystemExit("the two floors' populations are not at one theta0")
    card = np.load(args.sim_cache / "sim_centre.npz")["card"]
    res, means = {LOW: {}, FIVE: {}}, {}
    for (tag, a), (run, mdir) in runs(args).items():
        n, wy = we.parse_arm(a)
        cfg = json.loads((we.members_of(run)[0] / "results.json").read_text())["_config"]
        we.check_window(cfg["window"], n, wy)
        want = FLOOR if tag == LOW else FLOOR5
        if not np.isclose(cfg["lr_plateau"]["min_lr"], want) or not cfg["fresh_noise"]:
            raise SystemExit(f"{tag} {a}: not trained with fresh draws and floor {want:g}")
        mem = fn.member_log_q(run)
        r = {"n_waves": n, "wave_years": wy, "years_read": aw.years_read(a),
             **we.arm_measures(run), "population": pops[tag]["arms"][a],
             "low_beta": lb.bands(dict(np.load(run / "sbc" / "sbc16.npz"))),
             "training": nc.training(mdir),
             "training_extra": lp.training_extra(mdir, run, a, plateau=True),
             "below": below(mdir, run),
             "members": {"mean": float(mem.mean()), "sd": float(mem.std(ddof=1))}}
        r["calibrated"] = wd.calibrated(r)
        res[tag][a] = r
        means[f"{tag}:{a}"] = wd.scorings(run.parent, a)
    for k, m in means.items():   # the cached scorings give the stored se
        tag, a = k.split(":")
        se = we.population_se(m, card, th0, we.N_POP, lc.CARD_SHARE)["se"]
        if not np.allclose(se, pops[tag]["arms"][a]["se"]):
            raise SystemExit(f"{k}: the cached scorings do not give the stored se")
    bad = [f"{tag}/{a}/{s}" for tag in res for a in ARMS
           for s, v in res[tag][a]["training_extra"].items() if not v["schedule"]["ok"]]
    if bad:
        raise SystemExit(f"the logged learning rates do not follow the rule: {bad}")
    boot = nf.boot_se(means, card, th0, args.n_boot, args.seed)
    L, F = res[LOW], res[FIVE]
    floor = {}
    for a in ARMS:
        d, b, w = floor_rows(args.root, a)
        lq = nc.paired_log_q({LOW: d, FIVE: b}, w, LOW, FIVE, args.n_boot, args.seed)
        lq[f"ci_bonferroni{TESTS}"] = lp.bonferroni(d - b, w, TESTS, 10 * args.n_boot, args.seed)
        floor[a] = {"log_q": lq,
                    "members": fn.member_diff(fn.member_log_q(we.arm_dir(a, args.root)),
                                              fn.member_log_q(we.arm_dir(a, args.floor5_root))),
                    **wd.ratios(L[a], F[a], boot, f"{LOW}:{a}", f"{FIVE}:{a}")}
    res["floor"] = floor
    res["same_years"] = {f"{an}/{bi}": {tag: wd.ratios(res[tag][an], res[tag][bi], boot,
                                                        f"{tag}:{an}", f"{tag}:{bi}")
                                         for tag in (LOW, FIVE)}
                         for an, bi in SAME_YEARS}
    res["n_boot"], res["seed"] = args.n_boot, args.seed
    out = args.root / "low_floor.json"
    out.write_text(json.dumps(res, indent=1))
    report(res)
    print(f"\nwrote {out}")


def report(res: dict) -> None:
    """The RESULTS tables, as markdown."""
    cols = [(tag, a) for a in ARMS for tag in (LOW, FIVE)]
    hdr = ("| | " + " | ".join(f"{a} {'1e-7' if t == LOW else '1e-5'}" for t, a in cols)
           + " |\n|---|" + "---|" * len(cols))

    def row(label, f, fmt="{:.3f}"):
        vals = [f(res[t][a]) for t, a in cols]
        print(f"| {label} | " + " | ".join("–" if v is None else fmt.format(v) for v in vals)
              + " |")

    print("\ncalibration (gate: beta, delta, rho in 0.88-0.92; R_gamma reported)\n" + hdr)
    for j, p in enumerate(we.NAMES):
        print(f"| {p}: held-out / SBC 90% coverage | " + " | ".join(
            f"{res[t][a]['heldout']['cov90'][j]:.3f} / {res[t][a]['sbc_coverage_90'][j]:.3f}"
            for t, a in cols) + " |")
    print("| calibrated | " + " | ".join("yes" if res[t][a]["calibrated"] else "**no**"
                                        for t, a in cols) + " |")
    print("\none household and the population\n" + hdr)
    row("held-out log q", lambda r: r["heldout"]["log_q"])
    for j, p in enumerate(we.NAMES):
        row(f"{p}: share of prior variance removed", lambda r, j=j: r["heldout"]["contraction"][j])
    for p in we.NAMES:
        row(f"{p}: pairs ordered at p ≥ 0.9",
            lambda r, p=p: r["sbc"]["pairs_box"][p]["ordered_p90"])
    row("β 90% width, true β 0.1–0.3", lambda r: r["low_beta"]["bands"][0]["median_width_90"])
    row("β 90% width, true β 0.8–1.0", lambda r: r["low_beta"]["bands"][-1]["median_width_90"])
    for j, p in enumerate(we.NAMES):
        row(f"{p}: population se", lambda r, j=j: r["population"]["se"][j], "{:.5f}")
    print("\ntraining (CSVs; validation losses from sbi's record)\n" + hdr)

    def span(r, f, fmt="{:.0f}"):
        v = [x for x in (f(m) for m in r) if x is not None]   # None: never at the rate
        return f"{fmt.format(min(v))}–{fmt.format(max(v))}" if v else "–"

    for label, key, f, fmt in (
            ("epochs", "training", lambda m: m["epochs"], "{:.0f}"),
            ("passes at the best", "training", lambda m: m["passes_at_best"], "{:.0f}"),
            ("minutes per member", "training", lambda m: m["minutes"], "{:.0f}"),
            ("rate at the best", "training_extra", lambda m: m["lr_at_best"], "{:.2g}"),
            ("lowest rate reached", "below", lambda m: m["lowest_rate"], "{:.2g}"),
            ("passes below 1e-5", "below", lambda m: m["passes_below"], "{:.0f}"),
            ("longest run without a gain before the best", "training_extra",
             lambda m: m["patience_20"]["longest_no_gain_before_best"], "{:.0f}"),
            ("best-loss fall after reaching 1e-5", "below",
             lambda m: m["gain_from_rate"], "{:.4f}")):
        print(f"| {label} | " + " | ".join(span(list(res[t][a][key].values()), f, fmt)
                                           for t, a in cols) + " |")
    for label, key, f in (("best below 1e-5", "below", lambda m: m["best_below"]),
                          ("reached the floor", "below", lambda m: m["reached_floor"]),
                          ("at the cap", "training_extra", lambda m: m["reached_cap"]),
                          ("logged rate follows the rule", "training_extra",
                           lambda m: m["schedule"]["ok"])):
        print(f"| {label} | " + " | ".join(
            f"{sum(f(m) for m in res[t][a][key].values())}/{len(res[t][a][key])}"
            for t, a in cols) + " |")
    print(f"\nthe floor: held-out log q of 1e-7 minus 1e-5, paired on identical windows "
          f"(95% and {TESTS}-test intervals), members; information, pairs, population se ratios")
    for a, c in res["floor"].items():
        lq, m = c["log_q"], c["members"]
        lo, hi = lq[f"ci_bonferroni{TESTS}"]
        print(f"{a}: {lq['diff']:+.3f} [{lq['ci95'][0]:+.3f}, {lq['ci95'][1]:+.3f}] "
              f"({TESTS} tests [{lo:+.3f}, {hi:+.3f}]); members {m['diff']:+.3f} (Welch p "
              f"{m['welch_p']:.3f}); information {' '.join(f'{v:.2f}' for v in c['info'])}; "
              f"pairs {' '.join(f'{v:.2f}' for v in c['pairs'])}; population se "
              f"{' '.join(f'{v:.2f}' for v in c['population_se'])}")
    print("\nannual ÷ biennial in about the same years (information, pairs, population se "
          "with 90% interval)")
    for k, c in res["same_years"].items():
        for tag, r in c.items():
            ci = " ".join(f"[{lo:.2f}, {hi:.2f}]" for lo, hi in r["population_se_ci90"])
            print(f"{k} ({LABEL[tag]}): information "
                  f"{' '.join(f'{v:.2f}' for v in r['info'])}; pairs "
                  f"{' '.join(f'{v:.2f}' for v in r['pairs'])}; population se "
                  f"{' '.join(f'{v:.2f}' for v in r['population_se'])} {ci}")


# --------------------------------------------------------------------- figures

def figures(args) -> None:
    """Figures 161, 162, 167, 168 and 169."""
    import matplotlib
    matplotlib.use("Agg")

    res = json.loads((args.root / "low_floor.json").read_text())
    precision(res, args.fig_dir / "161_low_floor_same_years.png")
    floor_gain(res, args.fig_dir / "162_low_floor_gain.png")
    one_household(args, args.fig_dir / "167_low_floor_one_household.png")
    population(res, args, args.fig_dir / "168_low_floor_population.png")
    curves(args, args.fig_dir / "169_low_floor_training.png")


def precision(res: dict, path: Path) -> None:
    """161: share of prior variance removed, pairs ordered and the population
    se (÷ w07's at 1e-5) against years read, both floors (1e-7 solid, 1e-5
    dashed; biennial circles, annual squares); and annual ÷ biennial in about
    the same years, per parameter, under both floors."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    base = np.array(res[FIVE]["w07"]["population"]["se"])
    panels = (("share of prior variance removed (held-out)",
               lambda r, j: r["heldout"]["contraction"][j]),
              ("pairs ordered at p ≥ 0.9 (SBC)",
               lambda r, j: r["sbc"]["pairs_box"][we.NAMES[j]]["ordered_p90"]),
              ("population se ÷ w07's at floor 1e-5\n(409 households sharing θ₀)",
               lambda r, j: r["population"]["se"][j] / base[j]))
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (ylab, f) in zip(axs.flat[:3], panels, strict=True):
        for tag, style in ((LOW, dict(lw=1.6, alpha=1.0, ls="-")),
                           (FIVE, dict(lw=0.9, alpha=0.5, ls="--"))):
            for wy, mk in ((2, "o"), (1, "s")):
                xs = [a for a in ARMS if we.parse_arm(a)[1] == wy]
                for j in range(len(we.NAMES)):
                    ax.plot([aw.years_read(x) for x in xs], [f(res[tag][x], j) for x in xs],
                            marker=mk, mfc=aw.PARAM_COLOR[j] if wy == 2 else "white",
                            color=aw.PARAM_COLOR[j], **style)
        for an, bi in SAME_YEARS:   # the pairs compared
            ax.axvspan(aw.years_read(bi) - 0.4, aw.years_read(an) + 0.4, color="0.92", zorder=0)
        ax.set(xlabel="years read", ylabel=ylab)
        ax.grid(ls=":", lw=0.5)
    ax = axs[1, 1]
    keys = list(res["same_years"])
    for k_i, k in enumerate(keys):
        for t_i, tag in enumerate((LOW, FIVE)):
            for j in range(len(we.NAMES)):
                x = k_i * 1.2 + j * 0.24 + t_i * 0.09
                ax.plot(x, res["same_years"][k][tag]["info"][j], "o" if tag == LOW else "s",
                        color=aw.PARAM_COLOR[j], mfc=aw.PARAM_COLOR[j] if tag == LOW else "white")
    ax.axhline(1, color="k", lw=0.6)
    ax.set_xticks([k_i * 1.2 + 0.4 for k_i in range(len(keys))],
                  [f"{k.split('/')[0]} ÷ {k.split('/')[1]}\n({aw.years_read(k.split('/')[0])} "
                   f"against {aw.years_read(k.split('/')[1])} years)" for k in keys])
    ax.set(ylabel="information s/(1−s), annual ÷ biennial",
           title="annual against biennial in about the same years")
    ax.grid(ls=":", lw=0.5)
    handles = [Line2D([], [], color=c, lw=2, label=p)
               for c, p in zip(aw.PARAM_COLOR, we.NAMES, strict=True)]
    handles += [Line2D([], [], color="k", marker="o", label="biennial (w07, w10)"),
                Line2D([], [], color="k", marker="s", mfc="white", label="annual (a14, a20)"),
                Line2D([], [], color="k", ls="-", label="floor 1e-7 (panel 4: filled)"),
                Line2D([], [], color="k", ls="--", alpha=0.5,
                       label="floor 1e-5, RESULTS 71 (panel 4: open)")]
    fig.legend(handles=handles, fontsize=8, ncol=4, loc="lower center")
    fig.suptitle("RESULTS 72: precision against years read under the two floors; grey bands: "
                 "the same-years pairs", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def floor_gain(res: dict, path: Path) -> None:
    """162: the floor's paired held-out log q gain per arm (95% and four-test
    intervals) and the data's information s/(1 - s) as a ratio."""
    import matplotlib.pyplot as plt

    rec = res["floor"]
    x = np.arange(len(ARMS))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for k, (key, lw) in enumerate(((f"ci_bonferroni{TESTS}", 1), ("ci95", 3))):
        axs[0].vlines(x, [rec[a]["log_q"][key][0] for a in ARMS],
                      [rec[a]["log_q"][key][1] for a in ARMS], color="0.2" if k else "0.6",
                      lw=lw, label="95%" if k else f"{TESTS} tests (Bonferroni)")
    axs[0].plot(x, [rec[a]["log_q"]["diff"] for a in ARMS], "o", color="k")
    axs[0].axhline(0, color="k", lw=0.6)
    axs[0].set(xticks=x, xticklabels=ARMS, ylabel="held-out log q, floor 1e-7 − floor 1e-5",
               title="paired on identical windows")
    axs[0].legend(fontsize=8)
    for j, p in enumerate(we.NAMES):
        axs[1].plot(x + (j - 1.5) * 0.08, [rec[a]["info"][j] for a in ARMS], "o",
                    color=aw.PARAM_COLOR[j], label=p)
    axs[1].axhline(1, color="k", lw=0.6)
    axs[1].set(xticks=x, xticklabels=ARMS, ylabel="information s/(1−s), 1e-7 ÷ 1e-5",
               title="one household (held-out)")
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.grid(ls=":", lw=0.5)
    fig.suptitle("RESULTS 72: the learning-rate floor 1e-7 against RESULTS 71's 1e-5",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def one_household(args, path: Path) -> None:
    """167: figure 105's household (held-out, nearest theta0) under the four
    arms at floor 1e-7, 68/95%; a row per same-years pair."""
    import matplotlib.pyplot as plt

    from scripts import household_resolution as hr

    truth, draw, raw = aw.household(args.shards)
    drawn = {}
    for a in ARMS:
        drawn[a] = aw.household_draws(we.arm_dir(a, args.root), a, raw, args.device)
        aw.print_ranks(a, drawn[a], truth)
    label_of = {"beta": "β", "delta": "δ", "crra": "ρ", "R_gamma": "R_γ"}
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    for r_i, (an, bi) in enumerate(SAME_YEARS):
        for axi, (i, j) in zip(axs[r_i], ((0, 2), (0, 1), (1, 2)), strict=True):
            for x in (bi, an):
                wy = we.parse_arm(x)[1]
                hr._contour(axi, drawn[x], i, j, wd.COLOR[wy],
                            f"{x}: {we.parse_arm(x)[0]} {'annual' if wy == 1 else 'biennial'} "
                            f"waves, {aw.years_read(x)} years", we.LIMITS)
            axi.plot(*truth[[i, j]], "k*", ms=11, label="truth")
            axi.set(xlabel=label_of[we.NAMES[i]], ylabel=label_of[we.NAMES[j]],
                    xlim=we.LIMITS[we.NAMES[i]], ylim=we.LIMITS[we.NAMES[j]])
            axi.grid(ls=":", lw=0.5)
        axs[r_i, 0].legend(fontsize=8)
    fig.suptitle(f"RESULTS 72: one held-out simulated household (draw {draw}, nearest θ₀; "
                 "figure 105's) at floor 1e-7,\nannual against biennial in about the same "
                 "years: 68% (solid) / 95% (dashed)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    np.savez(args.root / "one_household.npz", truth=truth, draw=draw,
             **{f"draws_{x}": v for x, v in drawn.items()})
    print(f"wrote {path}")


def population(res: dict, args, path: Path) -> None:
    """168: 68/95% regions (delta method) of the representative theta for 409
    households sharing theta0, a row per floor, against Laibson et al.'s and
    PSID couples' at seven waves (at v2_fresh_rop's estimate)."""
    import matplotlib.pyplot as plt

    lai = json.loads(args.psid_reference.read_text())
    th_l = np.array(lai["theta_laibson"])
    th_p = np.array(lai["arms"][wd.PSID_ARM]["representative_theta"])
    th0 = np.array(json.loads((args.root / "population.json").read_text())["theta0"])
    rep = lai["arms"][wd.PSID_ARM]
    se_p = np.array(rep["representative_se"])
    cov_p = np.outer(se_p, se_p) * np.array(rep["representative_corr"])
    label = {"beta": r"$\beta$", "delta": r"$\delta$", "crra": r"$\rho$"}
    t = np.linspace(0, 2 * np.pi, 200)

    def ellipse(axi, c, cov, i, j, color, ls_pair, lab):
        w, v = np.linalg.eigh(np.asarray(cov)[np.ix_([i, j], [i, j])])
        for q, ls in zip((2.30, 5.99), ls_pair, strict=True):
            e = c[[i, j]][:, None] + v @ (np.sqrt(np.maximum(w, 0) * q)[:, None]
                                          * np.vstack([np.cos(t), np.sin(t)]))
            axi.plot(e[0], e[1], color=color, ls=ls, lw=1)
        axi.plot([], [], color=color, ls=ls_pair[0], label=lab)

    fig, axs = plt.subplots(2, 3, figsize=(16, 10.5))
    for r_i, tag in enumerate((LOW, FIVE)):
        for axi, (i, j) in zip(axs[r_i], ((0, 2), (0, 1), (1, 2)), strict=True):
            for key, ls in (("reported", ("-", "--")), ("sampling only", (":", ":"))):
                ellipse(axi, th_l, lai["laibson_cov"][key], i, j, "k", ls,
                        f"Laibson et al., {key}")
            axi.plot(*th_l[[i, j]], "k*", ms=10)
            for a in ARMS:
                ellipse(axi, th0, res[tag][a]["population"]["cov"], i, j,
                        we.COLOR[we.parse_arm(a)[0]], ("-", "--"),
                        f"{a} ({aw.years_read(a)} years), 409 households sharing θ₀")
            ellipse(axi, th_p, cov_p, i, j, "#7B3294", ("-", "--"),
                    "PSID couples, 7 biennial waves (v2_fresh_rop)")
            axi.plot(*th0[[i, j]], "o", color="0.2", ms=4)
            axi.plot(*th_p[[i, j]], "D", color="#7B3294", ms=4)
            axi.set(xlabel=label[we.NAMES[i]], ylabel=label[we.NAMES[j]], title=LABEL[tag])
            axi.grid(ls=":", lw=0.5)
        axs[r_i, 0].legend(fontsize=7)
    fig.suptitle("RESULTS 72: the population under the two floors, 68% / 95% regions of the "
                 "representative θ for 409 households sharing θ₀ (dot: v2_b010's estimate),\n"
                 "and PSID couples' at seven biennial waves, at v2_fresh_rop's estimate "
                 "(diamond)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def curves(args, path: Path) -> None:
    """169: member s0's validation loss (above the lower of the two floors'
    best) and learning rate against passes, both floors, per arm."""
    import matplotlib.pyplot as plt

    rs = runs(args)
    fig, axs = plt.subplots(2, len(ARMS), figsize=(16, 7), sharex="col")
    for c, a in enumerate(ARMS):
        ms = {tag: tm.read(rs[tag, a][1] / "s0.csv") for tag in (LOW, FIVE)}
        best = min(float(np.nanmin(m["val_loss"])) for m in ms.values())
        for tag, color in ((LOW, "#C53A32"), (FIVE, "#3B75AF")):
            m = ms[tag]
            axs[0, c].plot(m["passes"], np.asarray(m["val_loss"]) - best, color=color, lw=1,
                           label=LABEL[tag])
            axs[1, c].semilogy(m["passes"], m["lr"], color=color, lw=1)
        axs[0, c].set_yscale("symlog", linthresh=1e-4)   # the lower best sits at 0
        axs[0, c].set(title=f"{a}, member s0",
                      ylabel="validation loss − the lower best" if c == 0 else None)
        axs[1, c].axhline(FLOOR5, color="0.6", ls=":", lw=1)
        axs[1, c].set(xlabel="passes", ylabel="learning rate" if c == 0 else None)
        for ax in axs[:, c]:
            ax.grid(ls=":", lw=0.5)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("RESULTS 72: training under the two floors (validation loss above the lower "
                 "of the two bests, symmetric log scale, linear below 1e-4)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, f in (("measures", measures), ("figures", figures)):
        p = sub.add_parser(name, help=f.__doc__.splitlines()[0])
        p.add_argument("--root", type=Path, default=ROOT)
        p.add_argument("--floor5_root", type=Path, default=wd.ROOT,
                       help="RESULTS 71's arms, floor 1e-5")
        p.add_argument("--metrics_dir", type=Path, default=tm.METRICS_DIR)
        p.add_argument("--w07_metrics", type=Path, default=wd.W07_METRICS,
                       help="the CSVs of w07 at floor 1e-5 (v2_fresh_rop)")
        p.add_argument("--psid_reference", type=Path, default=wd.PSID_REFERENCE)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else None)
        p.set_defaults(fn=f)
    m = sub.choices["measures"]
    m.add_argument("--sim_cache", type=Path, default=we.ROOT / "population")
    m.add_argument("--n_boot", type=int, default=1000)
    m.add_argument("--seed", type=int, default=0)
    g = sub.choices["figures"]
    g.add_argument("--fig_dir", type=Path, default=we.FIG)
    g.add_argument("--shards", type=Path, nargs="+",
                   default=[Path("data/processed/couples_v2_dataset_shards"),
                            Path("data/processed/couples_v2_strip_dataset_shards"),
                            Path("data/processed/couples_v2_betastrip_dataset_shards")])
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
