"""RESULTS 71: the waves experiment under the default recipe, biennial and
annual waves.

The user, 2026-11-14: "if we apply this to the real household, they usually
have the annual data. So, let's test the same model with annual & more wave
versions"; they chose the full annual ladder. RESULTS 64/65's experiment is
rerun with RESULTS 1's model as it now stands: v2's noise on the box with the
beta strip, trained with the default recipe for new runs (``optionA.py train
--recipe plateau_fresh``: a new noise draw every epoch, the learning rate
halved at 5 passes without a gain, floor 1e-5, a stop after 20, a cap of
600). Eight arms, every year a window reads inside ages 24-64
(``waves_experiment.start_low/start_high``):

* w07: 7 biennial waves, RESULTS 1's reference v2_fresh_rop as trained
  (patience 60; RESULTS 69.4: a stop after 20 passes keeps the same
  weights), linked in, not retrained;
* w10, w14, w20: 10, 14 and 20 biennial waves;
* a07, a10, a14, a20: 7, 10, 14 and 20 annual waves.

Set before the run:

* calibration of beta, delta and rho, held-out and SBC 90% coverage in
  0.88-0.92 (R_gamma reported, not gated: the user, 2026-11-14, it is a
  nuisance parameter);
* precision against waves under the default, biennial and annual;
* annual against biennial at the same number of waves (a07-w07, a10-w10,
  a14-w14, a20-w20) and about the same years (a14-w07, a20-w10);
* the recipe against RESULTS 64/65's at each length with a counterpart (w07,
  w10, w14, w20, a14, a20): held-out log q paired on identical windows, a
  95% interval over households resampled, the six-test Bonferroni interval
  and the member-level check beside it;
* training from the per-epoch CSVs, each member's logged rate against the
  rule rebuilt from its validation losses (``lr_plateau.schedule_check``).

Commands:

``measures``
    The tables (waves_default.json).

``figures``
    150 (precision against waves and against years read, both recipes), 151
    (the recipe's paired log q gain at each length, and its information),
    159 (one held-out household under the eight arms, 68/95%), 160 (the
    population's 68/95% regions under the default).

Usage::

    PYTHONPATH=.:src .venv/bin/python scripts/waves_default.py measures
    PYTHONPATH=.:src .venv/bin/python scripts/waves_default.py figures
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
from scripts import waves_experiment as we

#: under the default recipe; w07 is v2_fresh_rop, linked in
ARMS = ("w07", "a07", "w10", "a10", "w14", "a14", "w20", "a20")
TRAINED = ARMS[1:]
#: RESULTS 64/65's arms, the old recipe (constant rate, one noise draw)
OLD = ("w07", "w10", "w14", "w20", "a14", "a20")
SAME_COUNT = (("a07", "w07"), ("a10", "w10"), ("a14", "w14"), ("a20", "w20"))
SAME_YEARS = (("a14", "w07"), ("a20", "w10"))
ROOT = Path("outputs/waves_default")
#: where w07 was trained, and its CSVs and population (RESULTS 69)
W07_RUN = Path("outputs/lr_plateau/v2_fresh_rop/w07")
W07_METRICS = Path("training_metrics/lr_plateau/v2_fresh_rop/w07")
W07_POPULATION = Path("outputs/lr_plateau/v2_fresh_rop/population.json")
#: RESULTS 69's per-household log q: v2_fresh_rop and w07 on w07's windows
W07_ROWS = Path("outputs/lr_plateau/cross_noise_rows.npz")
W07_NAMES = ("v2_fresh_rop", "w07")
#: PSID couples' representative theta under the reference (RESULTS 69.8, 70.5)
PSID_REFERENCE = Path("outputs/lr_plateau/laibson_comparison.json")
PSID_ARM = "v2_fresh_rop"
#: the gate: these parameters' 90% coverage, held-out and SBC
GATED, BAND = ("beta", "delta", "crra"), (0.88, 0.92)
COLOR = {2: "#3B75AF", 1: "#C53A32"}
PARAM_COLOR = aw.PARAM_COLOR


def recipe_rows(root: Path, label: str) -> Path:
    """The per-household log q of the default and the old recipe on one
    length's held-out windows (cross_noise_heldout.py --per_row)."""
    return root / "recipe" / f"cross_noise_{label}.npz"


def pair_rows(root: Path, label: str, w07_rows: Path = W07_ROWS) -> tuple:
    """``(default, old, w)``: each recipe's per-household log q on the same
    held-out windows of one length. Both networks of a file carry v2's noise,
    so the two noise columns must be the same windows."""
    if label == "w07":
        d = dict(np.load(w07_rows))
        a, b = W07_NAMES
        return d[f"{a}__{b}"], d[f"{b}__{b}"], d["w"]
    d = dict(np.load(recipe_rows(root, label)))
    for net in ("default", "old"):
        if not np.array_equal(d[f"{net}__default"], d[f"{net}__old"]):
            raise SystemExit(f"{label}: the two noise columns are not the same windows")
    return d["default__old"], d["old__old"], d["w"]


def populations(root: Path) -> dict:
    """An experiment's population.json arms, with RESULTS 65's annual file
    beside it when there is one; checked at one theta0."""
    out, th0 = {}, None
    for name in ("population.json", aw.POPULATION):
        f = root / name
        if not f.exists():
            continue
        p = json.loads(f.read_text())
        if th0 is not None and not np.allclose(p["theta0"], th0):
            raise SystemExit(f"{f}: another theta0")
        th0 = p["theta0"]
        out |= p["arms"]
    return {"theta0": th0, "arms": out}


def scorings(root: Path, label: str) -> dict:
    return {t: np.load(root / "population" / f"post_{label}_{t}.npz")["mean"] for t in we.POINTS}


def calibrated(r: dict) -> bool:
    j = [we.NAMES.index(p) for p in GATED]
    vals = [r["heldout"]["cov90"][i] for i in j] + [r["sbc_coverage_90"][i] for i in j]
    return bool(all(BAND[0] <= v <= BAND[1] for v in vals))


def ratios(ra: dict, rb: dict, boot: dict, ka: str, kb: str) -> dict:
    """Arm a over arm b: the data's information s/(1 - s), pairs ordered at
    p >= 0.9, the population se (with its 90% paired interval), and the
    held-out log q difference (not paired when the windows differ)."""
    return {"info": nf.info_ratio(ra["heldout"]["contraction"], rb["heldout"]["contraction"]),
            "pairs": [ra["sbc"]["pairs_box"][p]["ordered_p90"]
                      / rb["sbc"]["pairs_box"][p]["ordered_p90"] for p in we.NAMES],
            "population_se": (np.array(ra["population"]["se"]) / rb["population"]["se"]).tolist(),
            "population_se_ci90": nf.ratio_ci(boot, ka, kb),
            "log_q_diff": ra["heldout"]["log_q"] - rb["heldout"]["log_q"]}


def measures(args) -> None:
    """The arms' measures and comparisons (waves_default.json)."""
    from scripts import laibson_comparison as lc

    pop_new, pop_old = populations(args.root), populations(args.old_root)
    th0 = np.array(pop_new["theta0"])
    if not np.allclose(pop_old["theta0"], th0):
        raise SystemExit("the two recipes' populations are not at one theta0")
    w07_pop = json.loads(args.w07_population.read_text())["arms"]["w07"]
    if not np.allclose(pop_new["arms"]["w07"]["se"], w07_pop["se"]):
        raise SystemExit("w07's linked scorings do not give v2_fresh_rop's population se")
    card = np.load(args.sim_cache / "sim_centre.npz")["card"]
    res = {"arms": {}, "old": {}}
    means = {}
    for a in ARMS:
        n, wy = we.parse_arm(a)
        run = we.arm_dir(a, args.root)
        we.check_window(json.loads((we.members_of(run)[0] / "results.json").read_text())
                        ["_config"]["window"], n, wy)
        mdir = args.w07_metrics if a == "w07" else args.metrics_dir / tm.run_name(run)
        mem = fn.member_log_q(run)
        res["arms"][a] = {"n_waves": n, "wave_years": wy, "years_read": aw.years_read(a),
                          "start_low": we.start_low(wy), "start_high": we.start_high(n, wy),
                          **we.arm_measures(run), "population": pop_new["arms"][a],
                          "low_beta": lb.bands(dict(np.load(run / "sbc" / "sbc16.npz"))),
                          "training": nc.training(mdir),
                          "training_extra": lp.training_extra(mdir, run, a, plateau=True),
                          "members": {"mean": float(mem.mean()), "sd": float(mem.std(ddof=1))}}
        res["arms"][a]["calibrated"] = calibrated(res["arms"][a])
        means[f"default:{a}"] = scorings(args.root, a)
    for a in OLD:
        run = we.arm_dir(a, args.old_root)
        res["old"][a] = {**we.arm_measures(run), "population": pop_old["arms"][a],
                         "low_beta": lb.bands(dict(np.load(run / "sbc" / "sbc16.npz")))}
        means[f"old:{a}"] = scorings(args.old_root, a)
    for k, m in means.items():   # the cached scorings give the stored se
        recipe, a = k.split(":")
        stored = (pop_new if recipe == "default" else pop_old)["arms"][a]["se"]
        se = we.population_se(m, card, th0, we.N_POP, lc.CARD_SHARE)["se"]
        if not np.allclose(se, stored):
            raise SystemExit(f"{k}: the cached scorings do not give the stored se")
    bad = [f"{a}/{s}" for a in ARMS for s, v in res["arms"][a]["training_extra"].items()
           if not v["schedule"]["ok"]]
    if bad:
        raise SystemExit(f"the logged learning rates do not follow the rule: {bad}")
    boot = nf.boot_se(means, card, th0, args.n_boot, args.seed)
    A, B = res["arms"], res["old"]
    recipe = {}
    for a in OLD:
        d, b, w = pair_rows(args.root, a, args.w07_rows)
        lq = nc.paired_log_q({"default": d, "old": b}, w, "default", "old", args.n_boot,
                             args.seed)
        lq[f"ci_bonferroni{len(OLD)}"] = lp.bonferroni(d - b, w, len(OLD), 10 * args.n_boot,
                                                      args.seed)
        recipe[a] = {"log_q": lq,
                     "members": fn.member_diff(fn.member_log_q(we.arm_dir(a, args.root)),
                                               fn.member_log_q(we.arm_dir(a, args.old_root))),
                     **ratios(A[a], B[a], boot, f"default:{a}", f"old:{a}")}
    res["recipe"] = recipe
    spacing = {}
    for title, pairs in (("same_count", SAME_COUNT), ("same_years", SAME_YEARS)):
        for an, bi in pairs:
            spacing[f"{title}/{an}/{bi}"] = {
                "default": ratios(A[an], A[bi], boot, f"default:{an}", f"default:{bi}"),
                **({"old": ratios(B[an], B[bi], boot, f"old:{an}", f"old:{bi}")}
                   if an in B and bi in B else {})}
    res["spacing"] = spacing
    growth = {}   # information at n waves over the shortest arm of the same spacing
    for wy, base in ((2, "w07"), (1, "a07")):
        for a in ARMS:
            if we.parse_arm(a)[1] == wy and a != base:
                growth[f"{a}/{base}"] = {
                    "default": nf.info_ratio(A[a]["heldout"]["contraction"],
                                             A[base]["heldout"]["contraction"]),
                    **({"old": nf.info_ratio(B[a]["heldout"]["contraction"],
                                             B[base]["heldout"]["contraction"])}
                       if a in B and base in B else {})}
    res["growth"] = growth
    psid = we.psid_cov(args.psid_reference, PSID_ARM)
    cases, res["psid_remainder"] = we.psid_cases(
        {a: A[a]["population"]["se"] for a in ARMS}, "w07",
        np.array(A["w07"]["population"]["cov"]), psid)
    for a in ARMS:
        A[a]["psid_scale"] = cases[a]
    lai = json.loads(args.psid_reference.read_text())
    res["laibson_se"] = lai["laibson_se"]
    res["psid_se_7_waves"] = lai["arms"][PSID_ARM]["representative_se"]
    res["n_boot"], res["seed"] = args.n_boot, args.seed
    out = args.root / "waves_default.json"
    out.write_text(json.dumps(res, indent=1))
    report(res)
    print(f"\nwrote {out}")


def report(res: dict) -> None:
    """The RESULTS tables, as markdown."""
    A, B = res["arms"], res["old"]
    ns = [a for a in ARMS if a in A]
    hdr = "| | " + " | ".join(ns) + " |\n|---|" + "---|" * len(ns)

    def row(label, f, fmt="{:.3f}", arms=ns):
        vals = [f(A[n]) for n in arms]
        print(f"| {label} | " + " | ".join("–" if v is None else fmt.format(v) for v in vals)
              + " |")

    print("\nthe arms (the default recipe)\n" + hdr)
    row("years read", lambda r: r["years_read"], "{}")
    print("| start ages | " + " | ".join(f"{A[n]['start_low']}–{A[n]['start_high']}"
                                        for n in ns) + " |")
    print("\ncalibration (gate: beta, delta, rho in 0.88-0.92; R_gamma reported)\n" + hdr)
    for j, p in enumerate(we.NAMES):
        print(f"| {p}: held-out / SBC 90% coverage | " + " | ".join(
            f"{A[n]['heldout']['cov90'][j]:.3f} / {A[n]['sbc_coverage_90'][j]:.3f}"
            for n in ns) + " |")
    print("| calibrated | " + " | ".join("yes" if A[n]["calibrated"] else "**no**" for n in ns)
          + " |")
    print("\none household\n" + hdr)
    row("held-out log q", lambda r: r["heldout"]["log_q"])
    for j, p in enumerate(we.NAMES):
        row(f"{p}: share of prior variance removed", lambda r, j=j: r["heldout"]["contraction"][j])
    for j, p in enumerate(we.NAMES):
        row(f"{p}: information ÷ w07's", lambda r, j=j: we._info(r["heldout"]["contraction"][j])
            / we._info(A["w07"]["heldout"]["contraction"][j]))
    for j, p in enumerate(we.NAMES):
        row(f"{p}: corr(mean, truth)", lambda r, j=j: r["heldout"]["corr"][j])
    for j, p in enumerate(we.NAMES):
        row(f"{p}: median posterior sd (SBC)", lambda r, j=j: r["sbc"]["median_posterior_sd"][j],
            "{:.4f}")
    print("\ntelling households apart (SBC households)\n" + hdr)
    for p in we.NAMES:
        row(f"{p}: one 90% interval ÷ 5–95% range of means",
            lambda r, p=p: r["sbc"]["interval_vs_range"][p]["ratio"])
    for p in we.NAMES:
        row(f"{p}: pairs ordered at p ≥ 0.9",
            lambda r, p=p: r["sbc"]["pairs_box"][p]["ordered_p90"])
    for p in we.NAMES:
        row(f"{p}: of which right", lambda r, p=p: r["sbc"]["pairs_box"][p]["right_when_ordered"])
    print("\nlow beta by band (RESULTS 64.12)\n" + hdr)
    row("β 90% width, 0.1–0.3", lambda r: r["low_beta"]["bands"][0]["median_width_90"])
    row("β 90% width, 0.8–1.0", lambda r: r["low_beta"]["bands"][-1]["median_width_90"])
    row("ranking, 0.1–0.2 vs 0.3–0.4", lambda r: r["low_beta"]["pairs"][0]["auc_beta_mean"])
    row("ranking, 0.7–0.8 vs 0.9–1.0", lambda r: r["low_beta"]["pairs"][-1]["auc_beta_mean"])
    print("\nthe population: 409 households sharing theta0\n" + hdr)
    for j, p in enumerate(we.NAMES):
        row(f"{p}: se", lambda r, j=j: r["population"]["se"][j], "{:.4f}")
    ss = np.array(res["laibson_se"]["sampling only"])
    print("\nPSID couples (409), from their 7-wave se under v2_fresh_rop: (a) the remainder "
          "shrinks like the noise, (b) it is held\n" + hdr)
    for j, p in enumerate(we.NAMES[:3]):
        print(f"| {p}: se, (a) / (b) | " + " | ".join(
            f"{A[n]['psid_scale']['se_a'][j]:.4f} / {A[n]['psid_scale']['se_b'][j]:.4f}"
            for n in ns) + " |")
    for j, p in enumerate(we.NAMES[:3]):
        print(f"| {p}: ÷ their sampling-only se, (a) / (b) | " + " | ".join(
            f"{A[n]['psid_scale']['se_a'][j] / ss[j]:.2f} / "
            f"{A[n]['psid_scale']['se_b'][j] / ss[j]:.2f}" for n in ns) + " |")
    print("\ntraining (per-epoch CSVs)\n" + hdr)
    tr = {n: list(A[n]["training"].values()) for n in ns}
    ex = {n: list(A[n]["training_extra"].values()) for n in ns}
    print("| epochs | " + " | ".join(
        f"{min(m['epochs'] for m in tr[n])}–{max(m['epochs'] for m in tr[n])}" for n in ns) + " |")
    print("| passes at the best | " + " | ".join(
        f"{min(m['passes_at_best'] for m in tr[n]):.0f}–"
        f"{max(m['passes_at_best'] for m in tr[n]):.0f}" for n in ns) + " |")
    print("| minutes per member (mean) | " + " | ".join(
        f"{np.mean([m['minutes'] for m in tr[n]]):.0f}" for n in ns) + " |")
    print("| members at the cap | " + " | ".join(
        f"{sum(v['reached_cap'] for v in ex[n])}" for n in ns) + " |")
    print("| rate at the best | " + " | ".join(
        f"{min(v['lr_at_best'] for v in ex[n]):.2g}–{max(v['lr_at_best'] for v in ex[n]):.2g}"
        for n in ns) + " |")
    print("| logged rate follows the rule | " + " | ".join(
        f"{sum(v['schedule']['ok'] for v in ex[n])}/{len(ex[n])}" for n in ns) + " |")
    print("\nthe recipe against RESULTS 64/65's: held-out log q paired on identical windows "
          "(95% and six-test intervals), members; information, pairs, population se ratios")
    for a, c in res["recipe"].items():
        lq, m = c["log_q"], c["members"]
        lo, hi = lq[f"ci_bonferroni{len(OLD)}"]
        print(f"{a}: {lq['diff']:+.3f} [{lq['ci95'][0]:+.3f}, {lq['ci95'][1]:+.3f}] "
              f"(six tests [{lo:+.3f}, {hi:+.3f}]); members {m['diff']:+.3f} (Welch p "
              f"{m['welch_p']:.3f}); information {' '.join(f'{v:.2f}' for v in c['info'])}; "
              f"pairs {' '.join(f'{v:.2f}' for v in c['pairs'])}; population se "
              f"{' '.join(f'{v:.2f}' for v in c['population_se'])}")
    print("\nannual ÷ biennial (information, pairs, population se with 90% interval; log q "
          "difference, windows differ)")
    for k, c in res["spacing"].items():
        for rec, r in c.items():
            ci = " ".join(f"[{lo:.2f}, {hi:.2f}]" for lo, hi in r["population_se_ci90"])
            print(f"{k} ({rec}): log q {r['log_q_diff']:+.3f}; information "
                  f"{' '.join(f'{v:.2f}' for v in r['info'])}; pairs "
                  f"{' '.join(f'{v:.2f}' for v in r['pairs'])}; population se "
                  f"{' '.join(f'{v:.2f}' for v in r['population_se'])} {ci}")
    print("\ngrowth: information over the shortest arm of the same spacing")
    for k, c in res["growth"].items():
        print(f"{k}: " + "; ".join(f"{rec} {' '.join(f'{v:.2f}' for v in r)}"
                                   for rec, r in c.items()))
    if B:
        print("\nRESULTS 64/65's arms (the old recipe), for the figures: held-out log q "
              + ", ".join(f"{a} {B[a]['heldout']['log_q']:.3f}" for a in B))


# --------------------------------------------------------------------- figures

def figures(args) -> None:
    """Figures 150, 151, 159 and 160."""
    import matplotlib
    matplotlib.use("Agg")

    res = json.loads((args.root / "waves_default.json").read_text())
    precision(res, args.fig_dir / "150_waves_default_precision.png")
    recipe_gain(res, args.fig_dir / "151_waves_default_recipe.png")
    one_household(args, args.fig_dir / "159_waves_default_one_household.png")
    population(res, args, args.fig_dir / "160_waves_default_population.png")


def precision(res: dict, path: Path) -> None:
    """150: share of prior variance removed, pairs ordered and the population
    se (÷ w07's) against waves and against years read; biennial circles,
    annual squares; the default solid, RESULTS 64/65's recipe dashed."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    A, B = res["arms"], res["old"]
    base = np.array(A["w07"]["population"]["se"])
    rows = (("share of prior variance removed (held-out)",
             lambda r, j: r["heldout"]["contraction"][j]),
            ("pairs ordered at p ≥ 0.9 (SBC)",
             lambda r, j: r["sbc"]["pairs_box"][we.NAMES[j]]["ordered_p90"]),
            ("population se ÷ w07's under the default\n(409 households sharing θ₀)",
             lambda r, j: r["population"]["se"][j] / base[j]))
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    for i, (ylab, f) in enumerate(rows):
        for k, (xlab, xf) in enumerate((("waves", lambda x: we.parse_arm(x)[0]),
                                        ("years read", aw.years_read))):
            ax = axs[i, k]
            for arms, style in ((A, dict(lw=1.6, alpha=1.0)), (B, dict(lw=0.9, alpha=0.45))):
                for wy, mk in ((2, "o"), (1, "s")):
                    xs = sorted((a for a in arms if we.parse_arm(a)[1] == wy),
                                key=lambda a: we.parse_arm(a)[0])
                    for j in range(len(we.NAMES)):
                        ax.plot([xf(x) for x in xs], [f(arms[x], j) for x in xs],
                                marker=mk, ls="-" if arms is A else "--",
                                mfc=PARAM_COLOR[j] if wy == 2 else "white",
                                color=PARAM_COLOR[j], **style)
            ax.set(xlabel=xlab, ylabel=ylab)
            ax.axvline(7 if k == 0 else aw.years_read("w07"), color="grey", ls=":", lw=1)
            ax.grid(ls=":", lw=0.5)
    handles = [Line2D([], [], color=c, lw=2, label=p)
               for c, p in zip(PARAM_COLOR, we.NAMES, strict=True)]
    handles += [Line2D([], [], color="k", marker="o", label="biennial (w07–w20)"),
                Line2D([], [], color="k", marker="s", mfc="white", label="annual (a07–a20)"),
                Line2D([], [], color="k", ls="-", label="the default recipe (RESULTS 71)"),
                Line2D([], [], color="k", ls="--", alpha=0.45,
                       label="RESULTS 64/65's recipe")]
    fig.legend(handles=handles, fontsize=8, ncol=4, loc="lower center")
    fig.suptitle("RESULTS 71: precision against waves (left) and against years read (right) "
                 "under the default recipe;\ndotted: PSID's seven biennial waves", fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def recipe_gain(res: dict, path: Path) -> None:
    """151: the default recipe against RESULTS 64/65's at each length: held-out
    log q paired on identical windows (95% and six-test intervals), and the
    data's information s/(1 - s) as a ratio."""
    import matplotlib.pyplot as plt

    rec = res["recipe"]
    arms = list(rec)
    x = np.arange(len(arms))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    d = [rec[a]["log_q"]["diff"] for a in arms]
    for k, (key, lw) in enumerate(((f"ci_bonferroni{len(OLD)}", 1), ("ci95", 3))):
        lo = [rec[a]["log_q"][key][0] for a in arms]
        hi = [rec[a]["log_q"][key][1] for a in arms]
        axs[0].vlines(x, lo, hi, color="0.2" if k else "0.6", lw=lw,
                      label="95%" if k else "six tests (Bonferroni)")
    axs[0].plot(x, d, "o", color="k")
    axs[0].axhline(0, color="k", lw=0.6)
    axs[0].set(xticks=x, xticklabels=arms, ylabel="held-out log q, default − old recipe",
               title="paired on identical windows")
    axs[0].legend(fontsize=8)
    for j, p in enumerate(we.NAMES):
        axs[1].plot(x + (j - 1.5) * 0.08, [rec[a]["info"][j] for a in arms], "o",
                    color=PARAM_COLOR[j], label=p)
    axs[1].axhline(1, color="k", lw=0.6)
    axs[1].set(xticks=x, xticklabels=arms, ylabel="information s/(1−s), default ÷ old",
               title="one household (held-out)")
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.grid(ls=":", lw=0.5)
    fig.suptitle("RESULTS 71: the default recipe (a new noise draw every epoch, the plateau "
                 "schedule) against RESULTS 64/65's at each length", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def one_household(args, path: Path) -> None:
    """159: figure 105's household (held-out, nearest theta0) under the eight
    arms, 68/95%; a row per number of waves, biennial against annual (each
    spacing's noise seeded as in figure 115)."""
    import matplotlib.pyplot as plt

    from scripts import household_resolution as hr

    truth, draw, raw = aw.household(args.shards)
    drawn = {}
    for label in ARMS:
        drawn[label] = aw.household_draws(we.arm_dir(label, args.root), label, raw, args.device)
        aw.print_ranks(label, drawn[label], truth)
    label_of = {"beta": "β", "delta": "δ", "crra": "ρ", "R_gamma": "R_γ"}
    fig, axs = plt.subplots(4, 3, figsize=(16, 19))
    for r_i, n in enumerate((7, 10, 14, 20)):
        for axi, (i, j) in zip(axs[r_i], ((0, 2), (0, 1), (1, 2)), strict=True):
            for wy in (2, 1):
                x = we.arm_label(n, wy)
                hr._contour(axi, drawn[x], i, j, COLOR[wy],
                            f"{x}: {n} {'annual' if wy == 1 else 'biennial'} waves, "
                            f"{aw.years_read(x)} years", we.LIMITS)
            axi.plot(*truth[[i, j]], "k*", ms=11, label="truth")
            axi.set(xlabel=label_of[we.NAMES[i]], ylabel=label_of[we.NAMES[j]],
                    xlim=we.LIMITS[we.NAMES[i]], ylim=we.LIMITS[we.NAMES[j]])
            axi.grid(ls=":", lw=0.5)
        axs[r_i, 0].legend(fontsize=8)
    fig.suptitle(f"RESULTS 71: one held-out simulated household (draw {draw}, nearest θ₀; "
                 "figure 105's) under the default recipe,\nbiennial against annual at 7, 10, "
                 "14 and 20 waves: 68% (solid) / 95% (dashed)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    np.savez(args.root / "one_household.npz", truth=truth, draw=draw,
             **{f"draws_{x}": v for x, v in drawn.items()})
    print(f"wrote {path}")


def population(res: dict, args, path: Path) -> None:
    """160: 68/95% regions (delta method) of the representative theta for 409
    households sharing theta0 under the default, a row per spacing, against
    Laibson et al.'s and PSID couples' at seven waves and projected to 20
    (case b)."""
    import matplotlib.pyplot as plt

    A = res["arms"]
    lai = json.loads(args.psid_reference.read_text())
    th_l = np.array(lai["theta_laibson"])
    th0 = np.array(json.loads((args.root / "population.json").read_text())["theta0"])
    th_p = np.array(lai["arms"][PSID_ARM]["representative_theta"])   # couples' estimate
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
    for r_i, wy in enumerate((2, 1)):
        arms = [a for a in ARMS if we.parse_arm(a)[1] == wy]
        for axi, (i, j) in zip(axs[r_i], ((0, 2), (0, 1), (1, 2)), strict=True):
            for key, ls in (("reported", ("-", "--")), ("sampling only", (":", ":"))):
                ellipse(axi, th_l, lai["laibson_cov"][key], i, j, "k", ls,
                        f"Laibson et al., {key}")
            axi.plot(*th_l[[i, j]], "k*", ms=10)
            for a in arms:
                ellipse(axi, th0, A[a]["population"]["cov"], i, j,
                        we.COLOR[we.parse_arm(a)[0]], ("-", "--"),
                        f"{a}, 409 households sharing θ₀")
            ellipse(axi, th_p, res["psid_remainder"]["cov_psid"], i, j, "#7B3294", ("-", "--"),
                    "PSID couples, 7 biennial waves (v2_fresh_rop)")
            ellipse(axi, th_p, A[arms[-1]]["psid_scale"]["cov_b"], i, j, "#8C510A",
                    ("-.", "-."), f"PSID couples projected to {arms[-1]}, remainder held (b)")
            axi.plot(*th0[[i, j]], "o", color="0.2", ms=4)
            axi.plot(*th_p[[i, j]], "D", color="#7B3294", ms=4)
            axi.set(xlabel=label[we.NAMES[i]], ylabel=label[we.NAMES[j]],
                    title="biennial waves" if wy == 2 else "annual waves")
            axi.grid(ls=":", lw=0.5)
        axs[r_i, 0].legend(fontsize=7)
    fig.suptitle("RESULTS 71: the population under the default recipe, 68% / 95% regions of "
                 "the representative θ for 409 households sharing θ₀ (dot: v2_b010's estimate),"
                 "\nand for PSID's 409 couples at 7 biennial waves and projected (case b), at "
                 "v2_fresh_rop's estimate (diamond)", fontsize=10)
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
        p.add_argument("--old_root", type=Path, default=we.ROOT,
                       help="RESULTS 64/65's arms, the old recipe")
        p.add_argument("--psid_reference", type=Path, default=PSID_REFERENCE)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else None)
        p.set_defaults(fn=f)
    m = sub.choices["measures"]
    m.add_argument("--metrics_dir", type=Path, default=tm.METRICS_DIR)
    m.add_argument("--w07_metrics", type=Path, default=W07_METRICS)
    m.add_argument("--w07_population", type=Path, default=W07_POPULATION)
    m.add_argument("--w07_rows", type=Path, default=W07_ROWS)
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
