"""RESULTS 73: a longer patience with the low floor, four models.

The user, 2026-11-21: "try longer patience with the low floor". RESULTS 72
lowered the learning-rate floor to 1e-7 and found no gain; a14 and a20 went up
to 17 passes without a gain before their best, against a stop after 20
(72.4). Here RESULTS 72's command is kept in everything but the stop: 60
passes without a gain instead of 20 (RESULTS 69's patience), with the floor at
1e-7, a new noise draw every epoch, the rate halved at 5 passes without a
gain and a cap of 600, on RESULTS 1's model. The arms are RESULTS 72's, the
two same-years pairs:

* w07 (14 years read) against a14 (15);
* w10 (20 years read) against a20 (21).

Counterparts: RESULTS 72's networks (``outputs/low_floor``, patience 20, the
same floor), and beside them RESULTS 71's default (``outputs/waves_default``,
floor 1e-5; w07 is v2_fresh_rop). Set before the run:

* calibration of beta, delta and rho, held-out and SBC 90% coverage in
  0.88-0.92 (R_gamma reported);
* within each run, what a stop after 20 (and 40) passes would have kept, from
  sbi's record: the plateau's cuts depend only on the validation losses
  before them, so up to that stop a patience-60 trajectory is the
  patience-20 run (up to the GPU's nondeterminism);
* the patience: held-out log q of patience 60 minus RESULTS 72's 20, paired
  on identical windows, per arm, a 95% interval over households resampled
  ("helps": above 0), the four-test Bonferroni interval and the member-level
  check beside it; patience 60 against RESULTS 71's default beside that;
* annual against biennial in about the same years only (a14 / w07, a20 /
  w10), under the three settings;
* training from the CSVs and sbi's record.

Commands:

``measures``
    The tables (low_floor_p60.json).

``figures``
    170 (the paired log q gain and information, against patience 20 and
    against the default), 171 (training curves with the points where stops
    after 20 and 40 passes would have fired), 172 (precision against years
    read under the three settings, and the same-years ratios), 177 (one
    held-out household under the four arms, a row per same-years pair,
    68/95%), 178 (the population's 68/95% regions, the three settings).

Usage::

    PYTHONPATH=.:src .venv/bin/python scripts/low_floor_p60.py measures
    PYTHONPATH=.:src .venv/bin/python scripts/low_floor_p60.py figures
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
from scripts import low_floor as lf
from scripts import lr_plateau as lp
from scripts import noise_copies as nc
from scripts import noise_free_waves as nf
from scripts import optionA as oa
from scripts import training_metrics as tm
from scripts import waves_default as wd
from scripts import waves_experiment as we

ARMS, SAME_YEARS = lf.ARMS, lf.SAME_YEARS
ROOT = Path("outputs/low_floor_p60")
PATIENCE = 60
#: the three settings' names in the paired files and the bootstrap
P60, P20, FIVE = "p60", "p20", "floor5"
TAGS = (P60, P20, FIVE)
#: each setting's floor and stop (passes); RESULTS 71's w07 is v2_fresh_rop,
#: trained with a stop after 60, so the default's stop is not checked
SETTING = {P60: (lf.FLOOR, PATIENCE), P20: (lf.FLOOR, 20), FIVE: (lf.FLOOR5, None)}
LABEL = {P60: "patience 60, floor 1e-7", P20: "patience 20, floor 1e-7 (RESULTS 72)",
         FIVE: "patience 20, floor 1e-5 (RESULTS 71)"}
SHORT = {P60: "p60", P20: "p20", FIVE: "default"}
TESTS = len(ARMS)
#: the shorter stops read off each patience-60 trajectory
STOPS = (20, 40)


def compare_rows(root: Path, label: str) -> tuple[dict, np.ndarray]:
    """``({setting: per-household log q}, w)`` on the same held-out windows
    of one arm (cross_noise_heldout.py --per_row with the three settings).
    All three carry v2's noise, so every noise column must be the same
    windows."""
    d = dict(np.load(root / "compare" / f"cross_noise_{label}.npz"))
    for net in TAGS:
        if not all(np.array_equal(d[f"{net}__{P60}"], d[f"{net}__{noise}"]) for noise in TAGS):
            raise SystemExit(f"{label}: the noise columns are not the same windows")
    return {net: d[f"{net}__{P60}"] for net in TAGS}, d["w"]


def stops(v, patience=STOPS) -> dict:
    """What a stop after each of ``patience`` epochs without a gain would
    have kept on the validation losses ``v`` (sbi's record): the epoch it
    fires at (None if never), the best by then, how much lower the final best
    is, and whether it is the same best (``lr_plateau.patience_kept``)."""
    v = np.asarray(v, dtype=float)
    out = {}
    for k in patience:
        kept = lp.patience_kept(v, k)
        out[str(k)] = {"stop_epoch": kept["stop_epoch"], "best_then": kept["best_then"],
                       "fall_after": float(kept["best_then"] - v.min()),
                       "same_best": bool(kept["best_then"] == v.min())}
    return out


def stop_measures(run: Path) -> dict:
    """:func:`stops` per member of ``run``, from sbi's record."""
    out = {}
    for m in we.members_of(run):
        v = json.loads((m / "training_summary.json").read_text())["validation_loss"]
        out[m.name] = stops(v)
    return out


def stop_of(cfg: dict) -> int:
    """The stop in epochs a member was trained with (optionA writes it only
    when it is not the default)."""
    return cfg.get("stop_after_epochs") or oa.TRAINING["stop_after_epochs"]


def roots(args) -> dict:
    return {P60: args.root, P20: args.p20_root, FIVE: args.floor5_root}


def runs(args) -> dict:
    """Each setting's run directory and CSV directory per arm."""
    out = {}
    for tag, root in roots(args).items():
        for a in ARMS:
            run = we.arm_dir(a, root)
            mdir = (args.w07_metrics if (tag == FIVE and a == "w07")
                    else args.metrics_dir / tm.run_name(run))
            out[tag, a] = (run, mdir)
    return out


def check_setting(tag: str, a: str, cfg: dict) -> None:
    floor, stop = SETTING[tag]
    if not np.isclose(cfg["lr_plateau"]["min_lr"], floor) or not cfg["fresh_noise"]:
        raise SystemExit(f"{tag} {a}: not trained with fresh draws and floor {floor:g}")
    if stop is not None and stop_of(cfg) != stop:
        raise SystemExit(f"{tag} {a}: trained with a stop after {stop_of(cfg)}, not {stop}")


def measures(args) -> None:
    """The arms' measures and comparisons (low_floor_p60.json)."""
    from scripts import laibson_comparison as lc

    pops = {tag: wd.populations(root) for tag, root in roots(args).items()}
    th0 = np.array(pops[P60]["theta0"])
    if not all(np.allclose(p["theta0"], th0) for p in pops.values()):
        raise SystemExit("the three settings' populations are not at one theta0")
    card = np.load(args.sim_cache / "sim_centre.npz")["card"]
    res, means = {tag: {} for tag in TAGS}, {}
    for (tag, a), (run, mdir) in runs(args).items():
        n, wy = we.parse_arm(a)
        members = we.members_of(run)
        for m in members:
            cfg = json.loads((m / "results.json").read_text())["_config"]
            we.check_window(cfg["window"], n, wy)
            check_setting(tag, a, cfg)
        mem = fn.member_log_q(run)
        r = {"n_waves": n, "wave_years": wy, "years_read": aw.years_read(a),
             **we.arm_measures(run), "population": pops[tag]["arms"][a],
             "low_beta": lb.bands(dict(np.load(run / "sbc" / "sbc16.npz"))),
             "training": nc.training(mdir),
             "training_extra": lp.training_extra(mdir, run, a, plateau=True),
             "below": lf.below(mdir, run),
             "stops": stop_measures(run),
             "members": {"mean": float(mem.mean()), "sd": float(mem.std(ddof=1))}}
        r["calibrated"] = wd.calibrated(r)
        res[tag][a] = r
        means[f"{tag}:{a}"] = wd.scorings(run.parent, a)
    for k, m in means.items():   # the cached scorings give the stored se
        tag, a = k.split(":")
        se = we.population_se(m, card, th0, we.N_POP, lc.CARD_SHARE)["se"]
        if not np.allclose(se, pops[tag]["arms"][a]["se"]):
            raise SystemExit(f"{k}: the cached scorings do not give the stored se")
    bad = [f"{tag}/{a}/{s}" for tag in TAGS for a in ARMS
           for s, v in res[tag][a]["training_extra"].items() if not v["schedule"]["ok"]]
    if bad:
        raise SystemExit(f"the logged learning rates do not follow the rule: {bad}")
    boot = nf.boot_se(means, card, th0, args.n_boot, args.seed)
    patience = {}
    for a in ARMS:
        q, w = compare_rows(args.root, a)
        c = {}
        for ref in (P20, FIVE):
            lq = nc.paired_log_q({P60: q[P60], ref: q[ref]}, w, P60, ref, args.n_boot, args.seed)
            if ref == P20:   # the test; the default is beside it
                lq[f"ci_bonferroni{TESTS}"] = lp.bonferroni(q[P60] - q[P20], w, TESTS,
                                                            10 * args.n_boot, args.seed)
            c[ref] = {"log_q": lq,
                      "members": fn.member_diff(fn.member_log_q(we.arm_dir(a, args.root)),
                                                fn.member_log_q(we.arm_dir(a, roots(args)[ref]))),
                      **wd.ratios(res[P60][a], res[ref][a], boot, f"{P60}:{a}", f"{ref}:{a}")}
        # RESULTS 72's own paired rows, where they exist: the same windows?
        old = args.p20_root / "floor" / f"cross_noise_{a}.npz"
        if old.exists():
            o = np.load(old)
            c["rows_against_72"] = {
                P20: float(np.max(np.abs(o[f"{lf.LOW}__{lf.FIVE}"] - q[P20]))),
                FIVE: float(np.max(np.abs(o[f"{lf.FIVE}__{lf.FIVE}"] - q[FIVE])))}
        patience[a] = c
    res["patience"] = patience
    res["same_years"] = {f"{an}/{bi}": {tag: wd.ratios(res[tag][an], res[tag][bi], boot,
                                                        f"{tag}:{an}", f"{tag}:{bi}")
                                         for tag in TAGS}
                         for an, bi in SAME_YEARS}
    res["n_boot"], res["seed"] = args.n_boot, args.seed
    out = args.root / "low_floor_p60.json"
    out.write_text(json.dumps(res, indent=1))
    report(res)
    print(f"\nwrote {out}")


def report(res: dict) -> None:
    """The RESULTS tables, as markdown."""
    cols = [(tag, a) for a in ARMS for tag in TAGS]
    hdr = ("| | " + " | ".join(f"{a} {SHORT[t]}" for t, a in cols)
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
        v = [x for x in (f(m) for m in r) if x is not None]
        return f"{fmt.format(min(v))}–{fmt.format(max(v))}" if v else "–"

    def mems(r, key):
        return list(r[key].values())

    spans = [("epochs", "training", lambda m: m["epochs"], "{:.0f}"),
             ("passes at the best", "training", lambda m: m["passes_at_best"], "{:.0f}"),
             ("minutes per member", "training", lambda m: m["minutes"], "{:.0f}"),
             ("rate at the best", "training_extra", lambda m: m["lr_at_best"], "{:.2g}"),
             ("lowest rate reached", "below", lambda m: m["lowest_rate"], "{:.2g}"),
             ("passes at the floor", "training_extra", lambda m: m["passes_trained_at_floor"],
              "{:.0f}"),
             ("longest run without a gain before the best", "training_extra",
              lambda m: m["patience_20"]["longest_no_gain_before_best"], "{:.0f}")]
    spans += [(f"best-loss fall after a stop at {k} would have fired", "stops",
               lambda m, k=k: m[str(k)]["fall_after"], "{:.4f}") for k in STOPS]
    for label, key, f, fmt in spans:
        print(f"| {label} | " + " | ".join(span(mems(res[t][a], key), f, fmt)
                                           for t, a in cols) + " |")
    counts = [(f"best after a run of ≥ {k} passes without a gain", "stops",
               lambda m, k=k: not m[str(k)]["same_best"]) for k in STOPS]
    counts += [("reached the floor", "below", lambda m: m["reached_floor"]),
               ("at the cap", "training_extra", lambda m: m["reached_cap"]),
               ("logged rate follows the rule", "training_extra", lambda m: m["schedule"]["ok"]),
               ("CSV is sbi's record", "training_extra", lambda m: m["csv_matches_summary"])]
    for label, key, f in counts:
        print(f"| {label} | " + " | ".join(
            f"{sum(bool(f(m)) for m in mems(res[t][a], key))}/{len(res[t][a][key])}"
            for t, a in cols) + " |")
    for ref in (P20, FIVE):
        print(f"\nthe patience: held-out log q of {LABEL[P60]} minus {LABEL[ref]}, paired on "
              "identical windows (95%" + (f" and {TESTS}-test intervals" if ref == P20 else "")
              + "), members; information, pairs, population se ratios")
        for a in ARMS:
            c = res["patience"][a][ref]
            lq, m = c["log_q"], c["members"]
            bon = (f" ({TESTS} tests [{lq[f'ci_bonferroni{TESTS}'][0]:+.3f}, "
                   f"{lq[f'ci_bonferroni{TESTS}'][1]:+.3f}])" if ref == P20 else "")
            print(f"{a}: {lq['diff']:+.3f} [{lq['ci95'][0]:+.3f}, {lq['ci95'][1]:+.3f}]{bon}; "
                  f"members {m['diff']:+.3f} (Welch p {m['welch_p']:.3f}); information "
                  f"{' '.join(f'{v:.3f}' for v in c['info'])}; pairs "
                  f"{' '.join(f'{v:.3f}' for v in c['pairs'])}; population se "
                  f"{' '.join(f'{v:.3f}' for v in c['population_se'])}")
    for a in ARMS:
        if "rows_against_72" in res["patience"][a]:
            r = res["patience"][a]["rows_against_72"]
            print(f"{a}: largest difference from RESULTS 72's paired rows: p20 {r[P20]:.2e}, "
                  f"default {r[FIVE]:.2e}")
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

#: line style and marker of each setting
STYLE = {P60: dict(lw=1.6, alpha=1.0, ls="-"), P20: dict(lw=1.0, alpha=0.6, ls="--"),
         FIVE: dict(lw=0.9, alpha=0.45, ls=":")}
MARK = {P60: "o", P20: "s", FIVE: "^"}


def figures(args) -> None:
    """Figures 170, 171, 172, 177 and 178."""
    import matplotlib
    matplotlib.use("Agg")

    res = json.loads((args.root / "low_floor_p60.json").read_text())
    gain(res, args.fig_dir / "170_low_floor_p60_gain.png")
    curves(args, args.fig_dir / "171_low_floor_p60_training.png")
    precision(res, args.fig_dir / "172_low_floor_p60_same_years.png")
    one_household(args, args.fig_dir / "177_low_floor_p60_one_household.png")
    population(res, args, args.fig_dir / "178_low_floor_p60_population.png")


def gain(res: dict, path: Path) -> None:
    """170: patience 60's paired held-out log q gain per arm, against
    patience 20 (95% and four-test intervals) and against the default (95%),
    and the data's information s/(1 - s) as a ratio to each."""
    import matplotlib.pyplot as plt

    rec = res["patience"]
    x = np.arange(len(ARMS))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for k, ref in enumerate((P20, FIVE)):
        xs = x + (k - 0.5) * 0.2
        if ref == P20:
            axs[0].vlines(xs, [rec[a][ref]["log_q"][f"ci_bonferroni{TESTS}"][0] for a in ARMS],
                          [rec[a][ref]["log_q"][f"ci_bonferroni{TESTS}"][1] for a in ARMS],
                          color="0.6", lw=1, label=f"{TESTS} tests (Bonferroni)")
        axs[0].vlines(xs, [rec[a][ref]["log_q"]["ci95"][0] for a in ARMS],
                      [rec[a][ref]["log_q"]["ci95"][1] for a in ARMS],
                      color="0.2" if ref == P20 else "#3B75AF", lw=3)
        axs[0].plot(xs, [rec[a][ref]["log_q"]["diff"] for a in ARMS], MARK[ref],
                    color="k" if ref == P20 else "#3B75AF", label=f"against {LABEL[ref]}, 95%")
        for j, p in enumerate(we.NAMES):
            axs[1].plot(x + (k - 0.5) * 0.4 + (j - 1.5) * 0.06,
                        [rec[a][ref]["info"][j] for a in ARMS], MARK[ref], color=aw.PARAM_COLOR[j],
                        mfc=aw.PARAM_COLOR[j] if ref == P20 else "white",
                        label=p if ref == P20 else None)
    axs[0].axhline(0, color="k", lw=0.6)
    axs[0].set(xticks=x, xticklabels=ARMS, ylabel="held-out log q, patience 60 − the other",
               title="paired on identical windows")
    axs[0].legend(fontsize=7)
    axs[1].axhline(1, color="k", lw=0.6)
    axs[1].set(xticks=x, xticklabels=ARMS,
               ylabel="information s/(1−s), patience 60 ÷ the other\n"
                      "(filled: ÷ patience 20, open: ÷ the default)",
               title="one household (held-out)")
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.grid(ls=":", lw=0.5)
    fig.suptitle("RESULTS 73: a stop after 60 passes at floor 1e-7, against RESULTS 72's 20 "
                 "and RESULTS 71's default", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def curves(args, path: Path) -> None:
    """171: member s0's validation loss (above the lower of the two bests)
    and learning rate against passes, patience 60 and RESULTS 72's 20, per
    arm; dotted lines where stops after 20 and 40 passes would have fired on
    the patience-60 trajectory."""
    import matplotlib.pyplot as plt

    rs = runs(args)
    fig, axs = plt.subplots(2, len(ARMS), figsize=(16, 7), sharex="col")
    for c, a in enumerate(ARMS):
        ms = {tag: tm.read(rs[tag, a][1] / "s0.csv") for tag in (P60, P20)}
        best = min(float(np.nanmin(m["val_loss"])) for m in ms.values())
        for tag, color in ((P60, "#C53A32"), (P20, "#3B75AF")):
            m = ms[tag]
            axs[0, c].plot(m["passes"], np.asarray(m["val_loss"]) - best, color=color, lw=1,
                           label=LABEL[tag])
            axs[1, c].semilogy(m["passes"], m["lr"], color=color, lw=1)
        v = json.loads((rs[P60, a][0] / "s0" / "training_summary.json").read_text())
        for k, st in stops(v["validation_loss"]).items():
            if st["stop_epoch"] is not None:
                for ax in axs[:, c]:
                    ax.axvline(st["stop_epoch"], color="0.3", ls=":", lw=1)
                axs[0, c].annotate(f"stop at {k}", (st["stop_epoch"], 1), fontsize=7,
                                   rotation=90, va="top", ha="right")
        axs[0, c].set_yscale("symlog", linthresh=1e-4)   # the lower best sits at 0
        axs[0, c].set(title=f"{a}, member s0",
                      ylabel="validation loss − the lower best" if c == 0 else None)
        axs[1, c].axhline(lf.FLOOR5, color="0.6", ls=":", lw=1)
        axs[1, c].set(xlabel="passes", ylabel="learning rate" if c == 0 else None)
        for ax in axs[:, c]:
            ax.grid(ls=":", lw=0.5)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("RESULTS 73: training with a stop after 60 passes and after 20 (floor 1e-7); "
                 "dotted: where stops after 20 and 40 would have fired on the first (symmetric "
                 "log scale, linear below 1e-4)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def precision(res: dict, path: Path) -> None:
    """172: share of prior variance removed, pairs ordered and the population
    se (÷ w07's under the default) against years read, the three settings
    (biennial circles, annual squares); and annual ÷ biennial in about the
    same years, per parameter, under the three settings."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    base = np.array(res[FIVE]["w07"]["population"]["se"])
    panels = (("share of prior variance removed (held-out)",
               lambda r, j: r["heldout"]["contraction"][j]),
              ("pairs ordered at p ≥ 0.9 (SBC)",
               lambda r, j: r["sbc"]["pairs_box"][we.NAMES[j]]["ordered_p90"]),
              ("population se ÷ w07's under the default\n(409 households sharing θ₀)",
               lambda r, j: r["population"]["se"][j] / base[j]))
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (ylab, f) in zip(axs.flat[:3], panels, strict=True):
        for tag in TAGS:
            for wy, mk in ((2, "o"), (1, "s")):
                xs = [a for a in ARMS if we.parse_arm(a)[1] == wy]
                for j in range(len(we.NAMES)):
                    ax.plot([aw.years_read(x) for x in xs], [f(res[tag][x], j) for x in xs],
                            marker=mk, mfc=aw.PARAM_COLOR[j] if wy == 2 else "white",
                            color=aw.PARAM_COLOR[j], **STYLE[tag])
        for an, bi in SAME_YEARS:
            ax.axvspan(aw.years_read(bi) - 0.4, aw.years_read(an) + 0.4, color="0.92", zorder=0)
        ax.set(xlabel="years read", ylabel=ylab)
        ax.grid(ls=":", lw=0.5)
    ax = axs[1, 1]
    keys = list(res["same_years"])
    for k_i, k in enumerate(keys):
        for t_i, tag in enumerate(TAGS):
            for j in range(len(we.NAMES)):
                x = k_i * 1.4 + j * 0.3 + t_i * 0.08
                ax.plot(x, res["same_years"][k][tag]["info"][j], MARK[tag],
                        color=aw.PARAM_COLOR[j], mfc=aw.PARAM_COLOR[j] if tag == P60 else "white")
    ax.axhline(1, color="k", lw=0.6)
    ax.set_xticks([k_i * 1.4 + 0.5 for k_i in range(len(keys))],
                  [f"{k.split('/')[0]} ÷ {k.split('/')[1]}\n({aw.years_read(k.split('/')[0])} "
                   f"against {aw.years_read(k.split('/')[1])} years)" for k in keys])
    ax.set(ylabel="information s/(1−s), annual ÷ biennial",
           title="annual against biennial in about the same years")
    ax.grid(ls=":", lw=0.5)
    handles = [Line2D([], [], color=c, lw=2, label=p)
               for c, p in zip(aw.PARAM_COLOR, we.NAMES, strict=True)]
    handles += [Line2D([], [], color="k", marker="o", label="biennial (w07, w10)"),
                Line2D([], [], color="k", marker="s", mfc="white", label="annual (a14, a20)")]
    handles += [Line2D([], [], color="k", marker=MARK[t], mfc="k" if t == P60 else "white",
                       label=f"{LABEL[t]} (panel 4: {MARK[t]})", **STYLE[t]) for t in TAGS]
    fig.legend(handles=handles, fontsize=8, ncol=3, loc="lower center")
    fig.suptitle("RESULTS 73: precision against years read under the three settings; grey bands: "
                 "the same-years pairs", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def one_household(args, path: Path) -> None:
    """177: figure 105's household (held-out, nearest theta0) under the four
    arms with a stop after 60, 68/95%; a row per same-years pair."""
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
    fig.suptitle(f"RESULTS 73: one held-out simulated household (draw {draw}, nearest θ₀; "
                 "figure 105's) with a stop after 60 passes (floor 1e-7),\nannual against "
                 "biennial in about the same years: 68% (solid) / 95% (dashed)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    np.savez(args.root / "one_household.npz", truth=truth, draw=draw,
             **{f"draws_{x}": v for x, v in drawn.items()})
    print(f"wrote {path}")


def population(res: dict, args, path: Path) -> None:
    """178: 68/95% regions (delta method) of the representative theta for 409
    households sharing theta0, a row per setting, against Laibson et al.'s and
    PSID couples' at seven waves (at v2_fresh_rop's estimate)."""
    import matplotlib.pyplot as plt

    lai = json.loads(args.psid_reference.read_text())
    th_l = np.array(lai["theta_laibson"])
    rep = lai["arms"][wd.PSID_ARM]
    th_p = np.array(rep["representative_theta"])
    th0 = np.array(json.loads((args.root / "population.json").read_text())["theta0"])
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

    fig, axs = plt.subplots(3, 3, figsize=(16, 15))
    for r_i, tag in enumerate(TAGS):
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
    fig.suptitle("RESULTS 73: the population under the three settings, 68% / 95% regions of "
                 "the representative θ for 409 households sharing θ₀ (dot: v2_b010's "
                 "estimate),\nand PSID couples' at seven biennial waves, at v2_fresh_rop's "
                 "estimate (diamond)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, f in (("measures", measures), ("figures", figures)):
        p = sub.add_parser(name, help=f.__doc__.splitlines()[0])
        p.add_argument("--root", type=Path, default=ROOT)
        p.add_argument("--p20_root", type=Path, default=lf.ROOT,
                       help="RESULTS 72's arms, a stop after 20, floor 1e-7")
        p.add_argument("--floor5_root", type=Path, default=wd.ROOT,
                       help="RESULTS 71's arms, the default")
        p.add_argument("--metrics_dir", type=Path, default=tm.METRICS_DIR)
        p.add_argument("--w07_metrics", type=Path, default=wd.W07_METRICS,
                       help="the CSVs of w07 under the default (v2_fresh_rop)")
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
