"""RESULTS 66: the noise-free arms against the noisy ones.

Three arms of RESULTS 64-65 retrained with no observation noise (the user's
choice, 2026-11-07): w07 (7 biennial waves, start 24-51), a14 (14 annual,
25-51) and w14 (14 biennial, 24-37), five members each, everything else as
v2_b010. Each ensemble is graded with its own training noise, so the
noise-free arms are graded on noise-free windows: what the data would tell
under our model if they were measured exactly. Against v2's noise:

* the cost of noise, arm by arm (information, ordering, low beta, the
  population se at theta0 on the same simulated households and windows);
* the annual gain with and without noise: a14 against w07 (about the same
  years) and against w14 (the same number of waves). RESULTS 65's gain is an
  upper bound because every simulated wave draws fresh noise; without noise
  only the behaviour is left;
* cross-noise: each network on the other's held-out windows;
* PSID couples at seven waves: none_w07 beside w07, v2_b010, typical and
  gross (and 63.10's noise-unknown).

Commands:

``measures``
    The tables and comparisons (noise_free_waves.json), with the population
    se ratios' 90% intervals from households resampled within card type,
    paired across arms.

``figures``
    116 (precision with and without noise), 120 (figure 115's household
    under the six networks, 68/95%).

Usage::

    PYTHONPATH=. .venv/bin/python scripts/noise_free_waves.py measures
    PYTHONPATH=. .venv/bin/python scripts/noise_free_waves.py figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts import annual_waves as aw
from scripts import low_beta_check as lb
from scripts import waves_experiment as we

ARMS = ("w07", "a14", "w14")
NOISES = ("v2", "none")
ROOTS = {"v2": we.ROOT, "none": Path("outputs/waves_none")}
#: each noise's population at theta0 (RESULTS 65's file holds w07, a14, w14)
POPULATION = {"v2": "population_annual.json", "none": "population.json"}
#: (annual, biennial): about the same years, and the same number of waves
ANNUAL_PAIRS = (("a14", "w07"), ("a14", "w14"))
#: PSID couples' posteriors at seven waves, by arm (none_w07 and w07 are
#: under their roots: :func:`psid_dirs`)
PSID = {"v2_b010": Path("outputs/psid_optionA2v2_b010_couples"),
        "typical_b010": Path("outputs/psid_optionA2v2_b010_noise_typical_couples"),
        "gross_b010": Path("outputs/psid_optionA2v2_b010_noise_gross_couples")}
LAIBSON = ROOTS["none"] / "laibson_comparison_noise_free.json"
UNKNOWN = Path("outputs/noise_unknown_b010/population_refresh.json")
NOISE_COLOR = {"v2": "#3B75AF", "none": "#4A9A5B"}


def run_dir(noise: str, label: str, roots: dict = ROOTS) -> Path:
    """An arm's directory by its noise and label."""
    if noise not in roots:
        raise ValueError(f"unknown noise {noise!r}; known {sorted(roots)}")
    we.parse_arm(label)
    return we.arm_dir(label, roots[noise])


def psid_dirs(roots: dict = ROOTS) -> dict:
    return {"none_w07": roots["none"] / "psid_w07_couples",
            "w07": roots["v2"] / "psid_w07_couples", **PSID}


def info_ratio(contraction_a, contraction_b) -> list:
    """The data's information s/(1 - s) of ``a`` over ``b``, per parameter."""
    return [we._info(a) / we._info(b) for a, b in zip(contraction_a, contraction_b,
                                                       strict=True)]


def boot_se(means: dict, card: np.ndarray, th0: np.ndarray, n_boot: int, seed: int) -> dict:
    """Each arm's population se over ``n_boot`` resamplings of the simulated
    households within card type, the same resampling for every arm (the
    arms score the same households), so ratios of arms are paired.
    ``means`` maps arm -> {point: posterior means}."""
    from scripts import laibson_comparison as lc

    rng = np.random.default_rng(seed)
    i1, i0 = np.flatnonzero(card == 1), np.flatnonzero(card == 0)
    out = {a: [] for a in means}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        for a, m in means.items():
            mm = {t: v[idx] for t, v in m.items()}
            out[a].append(we.population_se(mm, card[idx], th0, we.N_POP, lc.CARD_SHARE)["se"])
    return {a: np.array(v) for a, v in out.items()}


def ratio_ci(boot: dict, a: str, b: str) -> list:
    """The 90% interval of se(a) / se(b) over paired resamplings."""
    return np.percentile(boot[a] / boot[b], [5, 95], axis=0).T.tolist()


def cross_noise(path: Path) -> dict:
    """A cross-noise run's 2 x 2 cells: coverage, log q and contraction for
    each network on each noise's windows."""
    d = json.loads(path.read_text())
    return {cell: {"log_q": v["log_q_weighted"],
                   "cov90": [v[p]["cov90"] for p in we.NAMES],
                   "contraction": [v[p]["contraction"] for p in we.NAMES]}
            for cell, v in d["cells"].items()}


def psid_summary(dirs: dict) -> dict:
    """PSID couples under each 7-wave arm: medians of the posterior means and
    sds, and none_w07's couple-by-couple correlation with w07 (aggregates
    only)."""
    from scripts import household_resolution as hr

    post = {a: hr.psid(str(d)) for a, d in dirs.items()
            if (d / "posterior_uncorrected.npz").exists()}
    out = {a: {"n": len(p["mean"]), "median_of_means": np.median(p["mean"], 0).tolist(),
               "median_posterior_sd": np.median(p["sd"], 0).tolist()}
           for a, p in post.items()}
    if "none_w07" in post and "w07" in post:
        a = np.load(dirs["none_w07"] / "posterior_uncorrected.npz")["mean"]
        b = np.load(dirs["w07"] / "posterior_uncorrected.npz")["mean"]
        ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
        out["corr_none_w07_w07"] = [float(np.corrcoef(a[ok, j], b[ok, j])[0, 1])
                                    for j in range(4)]
    return out


# -------------------------------------------------------------------- measures

def measures(args) -> None:
    """The noise-free and noisy arms' measures and comparisons."""
    from scripts import laibson_comparison as lc

    roots = {"v2": args.root_v2, "none": args.root_none}
    pops = {n: json.loads((roots[n] / POPULATION[n]).read_text()) for n in NOISES}
    th0 = np.array(pops["none"]["theta0"])
    if not np.allclose(th0, pops["v2"]["theta0"]):
        raise SystemExit("the two populations are not at one theta0")
    res = {"arms": {n: {} for n in NOISES}}
    means = {}
    for n in NOISES:
        for a in ARMS:
            run = run_dir(n, a, roots)
            res["arms"][n][a] = {**we.arm_measures(run), "years_read": aw.years_read(a),
                                 "population": pops[n]["arms"][a],
                                 "low_beta": lb.bands(dict(np.load(run / "sbc" / "sbc16.npz")))}
            means[f"{n}:{a}"] = {t: np.load(roots[n] / "population" / f"post_{a}_{t}.npz")["mean"]
                                 for t in we.POINTS}
    card = np.load(args.sim_cache / "sim_centre.npz")["card"]
    for n in NOISES:
        for a in ARMS:
            se = we.population_se(means[f"{n}:{a}"], card, th0, we.N_POP, lc.CARD_SHARE)["se"]
            if not np.allclose(se, pops[n]["arms"][a]["se"]):
                raise SystemExit(f"{n} {a}: the cached scorings do not give the stored se")
    boot = boot_se(means, card, th0, args.n_boot, args.seed)
    A = res["arms"]
    cmp_ = {"noise_cost": {}, "annual": {n: {} for n in NOISES}}
    for a in ARMS:                                      # none against v2, arm by arm
        r0, r1 = A["none"][a], A["v2"][a]
        cmp_["noise_cost"][a] = {
            "log_q_gain": r0["heldout"]["log_q"] - r1["heldout"]["log_q"],
            "info_none_over_v2": info_ratio(r0["heldout"]["contraction"],
                                            r1["heldout"]["contraction"]),
            "pairs_none_over_v2": [r0["sbc"]["pairs_box"][p]["ordered_p90"]
                                   / r1["sbc"]["pairs_box"][p]["ordered_p90"] for p in we.NAMES],
            "population_se_none_over_v2": (np.array(r0["population"]["se"])
                                           / r1["population"]["se"]).tolist(),
            "population_se_ci90": ratio_ci(boot, f"none:{a}", f"v2:{a}")}
    for n in NOISES:                                    # the annual gain under each noise
        for an, bi in ANNUAL_PAIRS:
            ra, rb = A[n][an], A[n][bi]
            cmp_["annual"][n][f"{an}/{bi}"] = {
                "log_q_gain": ra["heldout"]["log_q"] - rb["heldout"]["log_q"],
                "info": info_ratio(ra["heldout"]["contraction"], rb["heldout"]["contraction"]),
                "pairs": [ra["sbc"]["pairs_box"][p]["ordered_p90"]
                          / rb["sbc"]["pairs_box"][p]["ordered_p90"] for p in we.NAMES],
                "population_se": (np.array(ra["population"]["se"])
                                  / rb["population"]["se"]).tolist(),
                "population_se_ci90": ratio_ci(boot, f"{n}:{an}", f"{n}:{bi}")}
    res["comparisons"] = cmp_
    res["population_se_ci90"] = {k: np.percentile(v, [5, 95], axis=0).T.tolist()
                                 for k, v in boot.items()}
    res["cross_noise"] = {a: cross_noise(roots["none"] / f"cross_noise_{a}.json")
                          for a in ARMS if (roots["none"] / f"cross_noise_{a}.json").exists()}
    res["psid"] = psid_summary(psid_dirs(roots))
    if args.laibson.exists():
        lai = json.loads(args.laibson.read_text())
        res["representative"] = {
            name: {k: arm[k] for k in ("representative_theta", "representative_se",
                                       "representative_ci90", "converged", "at_box_edge",
                                       "last_simulated_misfit_over_psid_se")}
            for name, arm in lai["arms"].items()}
        res["laibson"] = {"theta": lai["laibson"]["theta"],
                          "se_sampling_only": lai["laibson_se"]["sampling only"]}
    if UNKNOWN.exists():
        u = json.loads(UNKNOWN.read_text())
        res["representative_unknown_63_10"] = {
            "last_simulated": u["last_simulated"][:4], "converged": u["converged"],
            "note": "RESULTS 63.10's noise-unknown arm: no fit, its last simulated point"}
    res["n_boot"], res["seed"] = args.n_boot, args.seed
    out = roots["none"] / "noise_free_waves.json"
    out.write_text(json.dumps(res, indent=1))
    report(res)
    print(f"\nwrote {out}")


def report(res: dict) -> None:
    """The RESULTS tables, as markdown."""
    A = res["arms"]
    cols = [(n, a) for a in ARMS for n in NOISES]
    hdr = ("| | " + " | ".join(f"{a} {'no noise' if n == 'none' else 'v2'}" for n, a in cols)
           + " |\n|---|" + "---|" * len(cols))

    def row(label, f, fmt="{:.3f}"):
        print(f"| {label} | " + " | ".join(fmt.format(f(A[n][a])) for n, a in cols) + " |")

    def row4(label, f, fmt="{:.3f}"):
        print(f"| {label} | " + " | ".join(" / ".join(fmt.format(v) for v in f(A[n][a]))
                                          for n, a in cols) + " |")

    print("\ncalibration and one household (β / δ / ρ / R_γ)\n" + hdr)
    row4("held-out 90% coverage", lambda r: r["heldout"]["cov90"])
    row4("SBC 90% coverage", lambda r: r["sbc_coverage_90"])
    row("held-out log q", lambda r: r["heldout"]["log_q"])
    row4("share of prior variance removed", lambda r: r["heldout"]["contraction"])
    row4("corr(posterior mean, truth)", lambda r: r["heldout"]["corr"])
    row4("median posterior sd (SBC)", lambda r: r["sbc"]["median_posterior_sd"], "{:.4f}")
    print("\ntelling households apart, low beta\n" + hdr)
    row4("pairs ordered at p ≥ 0.9", lambda r: [r["sbc"]["pairs_box"][p]["ordered_p90"]
                                                 for p in we.NAMES])
    row4("one 90% interval ÷ 5–95% range of means",
         lambda r: [r["sbc"]["interval_vs_range"][p]["ratio"] for p in we.NAMES])
    row("β 90% width, true β 0.1–0.3", lambda r: r["low_beta"]["bands"][0]["median_width_90"])
    row("β 90% width, true β 0.8–1.0", lambda r: r["low_beta"]["bands"][-1]["median_width_90"])
    row("ranking 0.1–0.2 vs 0.3–0.4", lambda r: r["low_beta"]["pairs"][0]["auc_beta_mean"])
    row("ranking 0.7–0.8 vs 0.9–1.0", lambda r: r["low_beta"]["pairs"][-1]["auc_beta_mean"])
    print("\nthe population at theta0 (409 households sharing it)\n" + hdr)
    row4("se", lambda r: r["population"]["se"], "{:.4f}")
    print("\ntraining\n" + hdr)
    print("| epochs | " + " | ".join(
        f"{min(m['epochs'] for m in A[n][a]['underfit'].values())}–"
        f"{max(m['epochs'] for m in A[n][a]['underfit'].values())}" for n, a in cols) + " |")
    for key, lab in (("train_minus_val_end", "training − validation loss at the end"),
                     ("minutes", "minutes per member")):
        row(f"{lab} (mean)", lambda r, key=key: np.mean([m[key] for m in r["underfit"].values()]))

    def show(v):
        return " ".join(f"{x:.2f}" for x in v)

    def show_ci(v, ci):
        return " ".join(f"{x:.2f} [{lo:.2f}, {hi:.2f}]" for x, (lo, hi) in zip(v, ci, strict=True))

    c = res["comparisons"]
    print("\nthe cost of noise: no noise ÷ v2 (β δ ρ R_γ)")
    for a, r in c["noise_cost"].items():
        print(f"{a}: log q {r['log_q_gain']:+.3f}; information {show(r['info_none_over_v2'])}; "
              f"pairs {show(r['pairs_none_over_v2'])}; population se "
              f"{show_ci(r['population_se_none_over_v2'], r['population_se_ci90'])}")
    print("\nthe annual gain, with v2's noise and with none (β δ ρ R_γ)")
    for n in NOISES:
        for k, r in c["annual"][n].items():
            print(f"{n:4s} {k}: log q {r['log_q_gain']:+.3f}; information {show(r['info'])}; "
                  f"pairs {show(r['pairs'])}; population se "
                  f"{show_ci(r['population_se'], r['population_se_ci90'])}")
    if res.get("cross_noise"):
        print("\ncross-noise held-out: 90% coverage β δ ρ R_γ, log q")
        for a, cells in res["cross_noise"].items():
            for cell, v in cells.items():
                print(f"{a}: {cell:22s} {show(v['cov90'])}  log q {v['log_q']:.3f}")
    p = res["psid"]
    if p:
        print("\nPSID couples at seven waves: median of means / median posterior sd")
        for a, v in p.items():
            if a.startswith("corr"):
                continue
            print(f"{a:13s} (n {v['n']}): " + "  ".join(
                f"{nm} {m:.4f} / {s:.4f}" for nm, m, s in zip(
                    we.NAMES, v["median_of_means"], v["median_posterior_sd"], strict=True)))
        if "corr_none_w07_w07" in p:
            print(f"couple by couple, none_w07 against w07: corr {show(p['corr_none_w07_w07'])}")
    if "representative" in res:
        print("\nPSID couples' representative theta (bootstrap se); converged; at the box edge")
        for a, v in res["representative"].items():
            print(f"{a:13s} " + "  ".join(f"{t:.4f} ({s:.4f})" for t, s in zip(
                v["representative_theta"], v["representative_se"], strict=True))
                  + f"  {v['converged']}  {v['at_box_edge']}")
        if "representative_unknown_63_10" in res:
            u = res["representative_unknown_63_10"]
            print(f"{'unknown_b010':13s} last simulated point "
                  + "  ".join(f"{t:.4f}" for t in u["last_simulated"]) + " (63.10, no fit)")


# --------------------------------------------------------------------- figures

def figures(args) -> None:
    """Figures 116 and 120."""
    res = json.loads((args.root_none / "noise_free_waves.json").read_text())
    precision(res, args.fig_dir / "116_noise_free_precision.png")
    one_household(args, args.fig_dir / "120_noise_free_one_household.png")


def precision(res: dict, path: Path) -> None:
    """116: share of prior variance removed, pairs ordered and the population
    se (÷ the noisy w07's) for w07, a14 and w14, with v2's noise (filled)
    and none (open)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    A = res["arms"]
    base = np.array(A["v2"]["w07"]["population"]["se"])
    rows = (("share of prior variance removed (held-out)",
             lambda r, j: r["heldout"]["contraction"][j]),
            ("pairs ordered at p ≥ 0.9 (SBC)",
             lambda r, j: r["sbc"]["pairs_box"][we.NAMES[j]]["ordered_p90"]),
            ("population se ÷ w07's with v2's noise",
             lambda r, j: r["population"]["se"][j] / base[j]))
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(ARMS))
    for ax, (ylab, f) in zip(axs, rows, strict=True):
        for j in range(len(we.NAMES)):
            for n, mk, ls in (("v2", "o", "-"), ("none", "s", "--")):
                ax.plot(x + 0.06 * (j - 1.5), [f(A[n][a], j) for a in ARMS], ls, marker=mk,
                        color=aw.PARAM_COLOR[j], mfc="white" if n == "none" else aw.PARAM_COLOR[j],
                        ms=7, lw=1.2)
        ax.set_xticks(x, [f"{a}\n({aw.years_read(a)} years)" for a in ARMS])
        ax.set(ylabel=ylab)
        ax.grid(ls=":", lw=0.5)
    handles = [Line2D([], [], color=c, lw=2, label=p)
               for c, p in zip(aw.PARAM_COLOR, we.NAMES, strict=True)]
    handles += [Line2D([], [], color="k", marker="o", label="v2's noise"),
                Line2D([], [], color="k", marker="s", ls="--", mfc="white", label="no noise")]
    axs[0].legend(handles=handles, fontsize=8)
    fig.suptitle("RESULTS 66: precision with v2's observation noise and with none, "
                 "7 biennial (w07), 14 annual (a14) and 14 biennial waves (w14)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def one_household(args, path: Path) -> None:
    """120: figure 115's household under w07, a14 and w14, each with v2's
    noise and with none, 68/95%. The noisy panels are figure 115's (the same
    noise seeds); the noise-free ones read the raw panel."""
    import matplotlib.pyplot as plt

    from scripts import household_resolution as hr

    roots = {"v2": args.root_v2, "none": args.root_none}
    truth, draw, raw = aw.household(args.shards)
    drawn = {}
    for a in ARMS:
        for n in NOISES:
            drawn[(n, a)] = aw.household_draws(run_dir(n, a, roots), a, raw, args.device)
            aw.print_ranks(f"{a} {n}", drawn[(n, a)], truth)
    label_of = {"beta": "β", "delta": "δ", "crra": "ρ", "R_gamma": "R_γ"}
    fig, axs = plt.subplots(len(ARMS), 3, figsize=(16, 15))
    for r_i, a in enumerate(ARMS):
        n_w, wy = we.parse_arm(a)
        for axi, (i, j) in zip(axs[r_i], ((0, 2), (0, 1), (1, 2)), strict=True):
            for n in NOISES:
                hr._contour(axi, drawn[(n, a)], i, j, NOISE_COLOR[n],
                            f"{a} ({n_w} {'annual' if wy == 1 else 'biennial'} waves), "
                            f"{'no noise' if n == 'none' else 'v2 noise'}", we.LIMITS)
            axi.plot(*truth[[i, j]], "k*", ms=11, label="truth")
            axi.set(xlabel=label_of[we.NAMES[i]], ylabel=label_of[we.NAMES[j]],
                    xlim=we.LIMITS[we.NAMES[i]], ylim=we.LIMITS[we.NAMES[j]])
            axi.grid(ls=":", lw=0.5)
        axs[r_i, 0].legend(fontsize=8)
    fig.suptitle(f"One held-out simulated household (draw {draw}, nearest θ₀; figures 105 and 115) "
                 "with v2's noise and with none (RESULTS 66): 68% (solid) / 95% (dashed)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    np.savez(args.root_none / "noise_free_one_household.npz", truth=truth, draw=draw,
             **{f"draws_{a}_{n}": v for (n, a), v in drawn.items()})
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("measures", measures), ("figures", figures)):
        p = sub.add_parser(name, help=fn.__doc__.splitlines()[0] if fn.__doc__ else None)
        p.add_argument("--root_v2", type=Path, default=ROOTS["v2"])
        p.add_argument("--root_none", type=Path, default=ROOTS["none"])
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else None)
        p.set_defaults(fn=fn)
    m = sub.choices["measures"]
    m.add_argument("--sim_cache", type=Path, default=we.ROOT / "population")
    m.add_argument("--laibson", type=Path, default=LAIBSON)
    m.add_argument("--n_boot", type=int, default=1000)
    m.add_argument("--seed", type=int, default=0)
    f = sub.choices["figures"]
    f.add_argument("--fig_dir", type=Path, default=we.FIG)
    f.add_argument("--shards", type=Path, nargs="+",
                   default=[Path("data/processed/couples_v2_dataset_shards"),
                            Path("data/processed/couples_v2_strip_dataset_shards"),
                            Path("data/processed/couples_v2_betastrip_dataset_shards")])
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
