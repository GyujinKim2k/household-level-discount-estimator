"""RESULTS 67: noisy copies of clean households, and the methodology file's noise.

Three 7-wave arms (w07's window: biennial, start 24-51; everything else as
RESULTS 64's w07), five members each, the user's choices of 2026-11-09:

* ``v2_k8``: v2's noise, K = 8 noisy copies of every clean training window;
* ``file_k1``: the methodology file's noise (``psid_noise.py``), one draw;
* ``file_k8``: the file's noise, K = 8 copies;

against ``w07`` (RESULTS 64: v2's noise, one draw). Each ensemble is graded
on windows carrying its own training noise; the held-out windows of a noise
are the same whatever the copies (copy 0 is the ``copies=1`` window), so
v2_k8 and w07 are scored on identical windows, as are file_k8 and file_k1.

Commands:

``measures``
    The tables (noise_copies.json): calibration, held-out log q paired on
    identical windows (95% intervals over households resampled with their
    importance weights), information, ordering, low beta, the population se
    at theta0 against w07's (paired resampling), training from the
    per-epoch CSVs, cross-noise, PSID couples, and the representative theta.

``figures``
    125 (learning curves from the CSVs) and 129 (paired log q and precision).

Usage::

    PYTHONPATH=.:src .venv/bin/python scripts/noise_copies.py measures
    PYTHONPATH=.:src .venv/bin/python scripts/noise_copies.py figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import low_beta_check as lb
from scripts import noise_free_waves as nf
from scripts import training_metrics as tm
from scripts import waves_experiment as we

ARMS = ("w07", "v2_k8", "file_k1", "file_k8")
NEW = ARMS[1:]
ROOT = Path("outputs/noise_copies")
#: (arm, against): the same noise, so the same held-out windows
PAIRS = (("v2_k8", "w07"), ("file_k8", "file_k1"))
#: each arm's training noise, as the cross-noise columns name them
NOISE_OF = {"w07": "w07", "v2_k8": "w07", "file_k1": "file_k1", "file_k8": "file_k1"}
LABEL = {"w07": "w07: v2 noise, one draw", "v2_k8": "v2_k8: v2 noise, 8 copies",
         "file_k1": "file_k1: file's noise, one draw", "file_k8": "file_k8: file's noise, 8 copies",
         "none_w07": "none_w07: no noise (RESULTS 66)"}
COLOR = {"w07": "#1F4E79", "v2_k8": "#6BAED6", "file_k1": "#E6862E", "file_k8": "#A63603",
         "none_w07": "#4A9A5B"}
#: under the root: the chain's cross-noise run, its per-household log q, the population fit
CROSS, PER_ROW, LAIBSON = "cross_noise.json", "cross_noise_rows.npz", "laibson_comparison.json"


def run_dir(arm: str, root: Path = ROOT, base: Path = we.ROOT) -> Path:
    """An arm's members' directory: w07 is RESULTS 64's."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known {ARMS}")
    return base / "w07" if arm == "w07" else root / arm / "w07"


def arm_root(arm: str, root: Path = ROOT, base: Path = we.ROOT) -> Path:
    """Where an arm's population scorings and PSID posteriors live."""
    return base if arm == "w07" else root / arm


def paired_log_q(rows: dict, w: np.ndarray, a: str, b: str, n_boot: int, seed: int) -> dict:
    """Weighted mean of log q(a) - log q(b) over the same households and
    windows, with a 95% interval over households resampled (weights carried
    along)."""
    d = rows[a] - rows[b]
    ok = np.isfinite(d) & np.isfinite(w)
    d, ww = d[ok], w[ok]
    est = float((ww * d).sum() / ww.sum())
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        boot.append((ww[i] * d[i]).sum() / ww[i].sum())
    return {"diff": est, "ci95": np.percentile(boot, [2.5, 97.5]).tolist(), "n": int(ok.sum())}


def training(arm_dir: Path) -> dict:
    """Per member, from its CSV: epochs, passes at the best epoch, the best
    validation loss, training minus validation loss at the best epoch, and
    minutes."""
    out = {}
    for f in sorted(arm_dir.glob("s[0-9]*.csv")):
        m = tm.read(f)
        b = int(np.nanargmin(m["val_loss"]))
        out[f.stem] = {"epochs": int(len(m["epoch"])), "passes_at_best": float(m["passes"][b]),
                       "best_val": float(m["val_loss"][b]),
                       "train_minus_val_at_best": float(m["train_loss"][b] - m["val_loss"][b]),
                       "minutes": float(np.nansum(m["epoch_sec"]) / 60)}
    return out


def psid_summary(dirs: dict) -> dict:
    """PSID couples under each arm: medians of the posterior means and sds and
    the share of couples whose rho mean is below 1 (aggregates only)."""
    from scripts import household_resolution as hr

    out = {}
    for a, d in dirs.items():
        if not (d / "posterior_uncorrected.npz").exists():
            continue
        p = hr.psid(str(d))
        out[a] = {"n": len(p["mean"]), "median_of_means": np.median(p["mean"], 0).tolist(),
                  "median_posterior_sd": np.median(p["sd"], 0).tolist(),
                  "share_rho_below_1": float((p["mean"][:, 2] < 1).mean())}
    return out


# -------------------------------------------------------------------- measures

def measures(args) -> None:
    """The arms' measures and comparisons (noise_copies.json)."""
    from scripts import laibson_comparison as lc

    res = {"arms": {}}
    means, pops = {}, {}
    for a in ARMS:
        r = arm_root(a, args.root, args.base)
        pops[a] = json.loads((r / "population.json").read_text())
        run = run_dir(a, args.root, args.base)
        res["arms"][a] = {**we.arm_measures(run), "population": pops[a]["arms"]["w07"],
                          "low_beta": lb.bands(dict(np.load(run / "sbc" / "sbc16.npz"))),
                          "training": training(args.metrics_dir / tm.run_name(run))}
        means[a] = {t: np.load(r / "population" / f"post_w07_{t}.npz")["mean"]
                    for t in we.POINTS}
    th0 = np.array(pops["w07"]["theta0"])
    if not all(np.allclose(p["theta0"], th0) for p in pops.values()):
        raise SystemExit("the arms' populations are not at one theta0")
    card = np.load(args.sim_cache / "sim_centre.npz")["card"]
    for a in ARMS:
        se = we.population_se(means[a], card, th0, we.N_POP, lc.CARD_SHARE)["se"]
        if not np.allclose(se, pops[a]["arms"]["w07"]["se"]):
            raise SystemExit(f"{a}: the cached scorings do not give the stored se")
    boot = nf.boot_se(means, card, th0, args.n_boot, args.seed)
    A = res["arms"]
    rows = dict(np.load(args.per_row))
    w = rows.pop("w")
    rows.pop("pid")
    # the duplicated noise columns must agree: v2_k8's held-out windows are w07's
    for net in ARMS:
        for dup, orig in (("v2_k8", "w07"), ("file_k8", "file_k1")):
            if not np.array_equal(rows[f"{net}__{dup}"], rows[f"{net}__{orig}"]):
                raise SystemExit(f"{net}: {dup}'s held-out windows differ from {orig}'s")
    cmp_ = {}
    for a in NEW:
        cmp_[f"{a}/w07"] = {
            "info": nf.info_ratio(A[a]["heldout"]["contraction"],
                                  A["w07"]["heldout"]["contraction"]),
            "pairs": [A[a]["sbc"]["pairs_box"][p]["ordered_p90"]
                      / A["w07"]["sbc"]["pairs_box"][p]["ordered_p90"] for p in we.NAMES],
            "population_se": (np.array(A[a]["population"]["se"])
                              / A["w07"]["population"]["se"]).tolist(),
            "population_se_ci90": nf.ratio_ci(boot, a, "w07")}
    for a, b in PAIRS:
        n = NOISE_OF[a]
        cmp_[f"{a}/{b}"] = {**cmp_.get(f"{a}/{b}", {}),
                           "log_q": paired_log_q({k: rows[f"{k}__{n}"] for k in (a, b)},
                                                 w, a, b, args.n_boot, args.seed),
                           "on_noise": n}
    cmp_["file_k8/file_k1"] |= {
        "info": nf.info_ratio(A["file_k8"]["heldout"]["contraction"],
                              A["file_k1"]["heldout"]["contraction"]),
        "population_se": (np.array(A["file_k8"]["population"]["se"])
                          / A["file_k1"]["population"]["se"]).tolist(),
        "population_se_ci90": nf.ratio_ci(boot, "file_k8", "file_k1")}
    res["comparisons"] = cmp_
    res["cross_noise"] = nf.cross_noise(args.cross)
    res["psid"] = psid_summary({a: arm_root(a, args.root, args.base) / "psid_w07_couples"
                                for a in ARMS})
    res["baseline_training"] = {"none_w07": training(args.metrics_dir / "waves_none" / "w07")}
    if args.laibson.exists():
        lai = json.loads(args.laibson.read_text())
        res["representative"] = {
            name: {k: arm[k] for k in ("representative_theta", "representative_se",
                                       "representative_ci90", "converged", "at_box_edge",
                                       "last_simulated_misfit_over_psid_se")}
            for name, arm in lai["arms"].items()}
        res["laibson"] = {"theta": lai["laibson"]["theta"],
                          "se_sampling_only": lai["laibson_se"]["sampling only"]}
    res["n_boot"], res["seed"] = args.n_boot, args.seed
    out = args.root / "noise_copies.json"
    out.write_text(json.dumps(res, indent=1))
    report(res)
    print(f"\nwrote {out}")


def report(res: dict, arms=ARMS) -> None:
    """The RESULTS tables, as markdown (``arms``: the columns, RESULTS 68
    passes its own)."""
    A = res["arms"]
    hdr = "| | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms)

    def row(label, f, fmt="{:.3f}"):
        print(f"| {label} | " + " | ".join(fmt.format(f(A[a])) for a in arms) + " |")

    def row4(label, f, fmt="{:.3f}"):
        print(f"| {label} | " + " | ".join(" / ".join(fmt.format(v) for v in f(A[a]))
                                          for a in arms) + " |")

    def show(v):
        return " ".join(f"{x:.2f}" for x in v)

    print("\ncalibration and one household (β / δ / ρ / R_γ), each on its own noise\n" + hdr)
    row4("held-out 90% coverage", lambda r: r["heldout"]["cov90"])
    row4("SBC 90% coverage", lambda r: r["sbc_coverage_90"])
    row("held-out log q", lambda r: r["heldout"]["log_q"])
    row4("share of prior variance removed", lambda r: r["heldout"]["contraction"])
    row4("corr(posterior mean, truth)", lambda r: r["heldout"]["corr"])
    row4("median posterior sd (SBC)", lambda r: r["sbc"]["median_posterior_sd"], "{:.4f}")
    row4("pairs ordered at p ≥ 0.9", lambda r: [r["sbc"]["pairs_box"][p]["ordered_p90"]
                                                 for p in we.NAMES])
    row("β 90% width, true β 0.1–0.3", lambda r: r["low_beta"]["bands"][0]["median_width_90"])
    row4("population se at θ₀", lambda r: r["population"]["se"], "{:.4f}")
    print("\ntraining, from the per-epoch CSVs (range over members)\n" + hdr)
    for key, lab, fmt in (("epochs", "epochs", "{:.0f}"),
                          ("passes_at_best", "passes over the clean data at the best epoch",
                           "{:.0f}"),
                          ("best_val", "best validation loss", "{:.3f}"),
                          ("train_minus_val_at_best", "training − validation loss at the best",
                           "{:.3f}"),
                          ("minutes", "minutes per member", "{:.0f}")):
        print(f"| {lab} | " + " | ".join(
            (fmt.format(min(m[key] for m in A[a]["training"].values())) + "–"
             + fmt.format(max(m[key] for m in A[a]["training"].values())))
            if A[a]["training"] else "–" for a in arms) + " |")
    print("\ncomparisons (β δ ρ R_γ)")
    for k, c in res["comparisons"].items():
        line = f"{k}:"
        if "log_q" in c:
            lq = c["log_q"]
            line += (f" paired log q {lq['diff']:+.3f} [{lq['ci95'][0]:+.3f}, "
                     f"{lq['ci95'][1]:+.3f}] on {c['on_noise']}'s windows (n {lq['n']});")
        if "info" in c:
            line += f" information {show(c['info'])};"
        if "pairs" in c:
            line += f" pairs {show(c['pairs'])};"
        if "population_se" in c:
            line += " population se " + " ".join(
                f"{x:.2f} [{lo:.2f}, {hi:.2f}]"
                for x, (lo, hi) in zip(c["population_se"], c["population_se_ci90"], strict=True))
        print(line)
    print("\ncross-noise held-out: 90% coverage β δ ρ R_γ, log q")
    for cell, v in res["cross_noise"].items():
        print(f"{cell:24s} {show(v['cov90'])}  log q {v['log_q']:.3f}")
    print("\nPSID couples: median of means / median posterior sd; share of ρ below 1")
    for a, v in res["psid"].items():
        print(f"{a:8s} (n {v['n']}): " + "  ".join(
            f"{nm} {m:.4f} / {s:.4f}" for nm, m, s in zip(
                we.NAMES, v["median_of_means"], v["median_posterior_sd"], strict=True))
              + f"  ρ < 1: {v['share_rho_below_1']:.2f}")
    if "representative" in res:
        print("\nPSID couples' representative theta (bootstrap se); converged; at the box edge")
        for a, v in res["representative"].items():
            print(f"{a:13s} " + "  ".join(f"{t:.4f} ({s:.4f})" for t, s in zip(
                v["representative_theta"], v["representative_se"], strict=True))
                  + f"  {v['converged']}  {v['at_box_edge']}")


# --------------------------------------------------------------------- figures

def figures(args) -> None:
    """Figures 125-129."""
    curves(args, args.fig_dir / "125_noise_copies_training.png")
    sbc_figures(args)
    res = json.loads((args.root / "noise_copies.json").read_text())
    precision(res, args.fig_dir / "129_noise_copies_precision.png")


def sbc_figures(args) -> None:
    """126-128: each new arm's SBC ranks, redrawn from its saved sbc16.npz
    with the arm's name in the title (the chain's first figures took the run
    directory's name, w07, for every arm)."""
    from scripts import optionA as oa
    from scripts import sbc_optionA2 as sbc

    for k, a in enumerate(NEW):
        d = run_dir(a, args.root, args.base)
        z = np.load(d / "sbc" / "sbc16.npz")
        dirs = sorted(d.glob("s[0-9]*"))
        spec = oa.spec_of(json.loads((dirs[0] / "results.json").read_text())["_config"])
        u = z["u"]
        path = args.fig_dir / f"{126 + k}_sbc_noise_copies_{a}.png"
        sbc.figure(u, z["one_per_draw"] & np.isfinite(u).all(1), path,
                   sbc.title(dirs, spec, len(np.unique(z["pid"])), a))
        print(f"wrote {path}")


def curves(args, path: Path) -> None:
    """125: training (dashed) and validation (solid) loss per member against
    passes over the clean windows, one panel per noise; the best epoch
    marked."""
    import matplotlib.pyplot as plt

    panels = ((("w07", "v2_k8"), "v2's noise"), (("file_k1", "file_k8"), "the file's noise"),
              (("none_w07",), "no noise (RESULTS 66, for reference)"))
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (arms, title) in zip(axs, panels, strict=True):
        for a in arms:
            d = (args.metrics_dir / "waves_none" / "w07" if a == "none_w07"
                 else args.metrics_dir / tm.run_name(run_dir(a, args.root, args.base)))
            for k, f in enumerate(sorted(d.glob("s[0-9]*.csv"))):
                m = tm.read(f)
                lab = LABEL[a] if k == 0 else None
                ax.plot(m["passes"], m["val_loss"], "-", color=COLOR[a], lw=1.2, label=lab)
                ax.plot(m["passes"], m["train_loss"], "--", color=COLOR[a], lw=0.8, alpha=0.7)
                b = int(np.nanargmin(m["val_loss"]))
                ax.plot(m["passes"][b], m["val_loss"][b], "o", color=COLOR[a], ms=5)
        ax.set(xlabel="passes over the clean training windows (epochs × copies)",
               ylabel="loss (−log q, flow scale)", title=title)
        ax.grid(ls=":", lw=0.5)
        ax.legend(fontsize=8, title="solid validation, dashed training; ● best", title_fontsize=8)
    fig.suptitle("RESULTS 67: training curves from the per-epoch CSVs (training_metrics/)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def precision(res: dict, path: Path) -> None:
    """129: the paired held-out log q differences with 95% intervals, and each
    arm's information and population se against w07's."""
    import matplotlib.pyplot as plt

    from scripts import annual_waves as aw

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    c = res["comparisons"]
    keys = [f"{a}/{b}" for a, b in PAIRS]
    for i, k in enumerate(keys):
        lq = c[k]["log_q"]
        axs[0].errorbar(i, lq["diff"], yerr=[[lq["diff"] - lq["ci95"][0]],
                                             [lq["ci95"][1] - lq["diff"]]],
                        fmt="o", color=COLOR[k.split("/")[0]], capsize=4)
    axs[0].axhline(0, color="k", lw=0.8)
    axs[0].set_xticks(range(len(keys)), [f"{k}\n(on {c[k]['on_noise']}'s windows)" for k in keys])
    axs[0].set(ylabel="held-out log q difference (95% interval)",
               title="paired on the same households and windows")
    for ax, key, ylab in ((axs[1], "info", "information ÷ w07's (held-out)"),
                          (axs[2], "population_se", "population se at θ₀ ÷ w07's")):
        for i, a in enumerate(NEW):
            v = c[f"{a}/w07"][key]
            for j in range(len(we.NAMES)):
                ax.plot(i + 0.08 * (j - 1.5), v[j], "o", color=aw.PARAM_COLOR[j],
                        label=we.NAMES[j] if i == 0 else None)
                if key == "population_se":
                    lo, hi = c[f"{a}/w07"]["population_se_ci90"][j]
                    ax.plot([i + 0.08 * (j - 1.5)] * 2, [lo, hi], "-", color=aw.PARAM_COLOR[j],
                            lw=1)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xticks(range(len(NEW)), NEW)
        ax.set(ylabel=ylab)
        ax.grid(ls=":", lw=0.5)
    axs[1].legend(fontsize=8)
    axs[0].grid(ls=":", lw=0.5)
    fig.suptitle("RESULTS 67: what 8 noisy copies and the methodology file's noise change",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("measures", measures), ("figures", figures)):
        p = sub.add_parser(name, help=fn.__doc__.splitlines()[0] if fn.__doc__ else None)
        p.add_argument("--root", type=Path, default=ROOT)
        p.add_argument("--base", type=Path, default=we.ROOT,
                       help="RESULTS 64's root, holding w07, its population and PSID posteriors.")
        p.add_argument("--metrics_dir", type=Path, default=tm.METRICS_DIR)
        p.set_defaults(fn=fn)
    m = sub.choices["measures"]
    m.add_argument("--sim_cache", type=Path, default=we.ROOT / "population")
    m.add_argument("--cross", type=Path, default=None, help=f"default <root>/{CROSS}")
    m.add_argument("--per_row", type=Path, default=None, help=f"default <root>/{PER_ROW}")
    m.add_argument("--laibson", type=Path, default=None, help=f"default <root>/{LAIBSON}")
    m.add_argument("--n_boot", type=int, default=1000)
    m.add_argument("--seed", type=int, default=0)
    f = sub.choices["figures"]
    f.add_argument("--fig_dir", type=Path, default=we.FIG)
    args = ap.parse_args()
    if args.cmd == "measures":
        for k, name in (("cross", CROSS), ("per_row", PER_ROW), ("laibson", LAIBSON)):
            if getattr(args, k) is None:
                setattr(args, k, args.root / name)
    args.fn(args)


if __name__ == "__main__":
    main()
