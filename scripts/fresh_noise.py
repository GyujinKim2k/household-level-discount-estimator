"""RESULTS 68: a new noise draw on the training windows every epoch.

Two 7-wave arms (w07's window: biennial, start 24-51; everything else as
RESULTS 64's w07), five members each, the user's choice of 2026-11-10 (both
noises):

* ``v2_fresh``: v2's noise, a new draw every epoch;
* ``file_fresh``: the methodology file's noise, a new draw every epoch;

against RESULTS 67's arms on the same noise: ``w07`` (one draw) and
``v2_k8`` (8 fixed copies); ``file_k1`` and ``file_k8``. Epoch e trains on
copy e - 1's draw and validation keeps draw 0, so held-out windows of a
noise are the same for every arm on it (copy 0), and the paired log q
compares networks on identical windows.

Commands:

``measures``
    The tables (fresh_noise.json): calibration, held-out log q paired on
    identical windows, information, ordering, low beta, the population se at
    theta0 (paired resampling), training from the per-epoch CSVs (with the
    rise after the best and whether the cap was reached), cross-noise, PSID
    couples, and the representative theta.

``figures``
    133 (training curves from the CSVs) and 136 (paired log q and precision).

Usage::

    PYTHONPATH=.:src .venv/bin/python scripts/fresh_noise.py measures
    PYTHONPATH=.:src .venv/bin/python scripts/fresh_noise.py figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import low_beta_check as lb
from scripts import noise_copies as nc
from scripts import noise_free_waves as nf
from scripts import training_metrics as tm
from scripts import waves_experiment as we

ARMS = ("w07", "v2_k8", "v2_fresh", "file_k1", "file_k8", "file_fresh")
NEW = ("v2_fresh", "file_fresh")
ROOT = Path("outputs/fresh_noise")
#: (arm, against): the same noise, so the same held-out windows; the first of
#: each noise is the pre-registered test
PAIRS = (("v2_fresh", "v2_k8"), ("v2_fresh", "w07"),
         ("file_fresh", "file_k1"), ("file_fresh", "file_k8"))
#: each arm's training noise, as the cross-noise columns name them
NOISE_OF = {"w07": "w07", "v2_k8": "w07", "v2_fresh": "w07",
            "file_k1": "file_k1", "file_k8": "file_k1", "file_fresh": "file_k1"}
LABEL = {**nc.LABEL, "v2_fresh": "v2_fresh: v2 noise, new draw every epoch",
         "file_fresh": "file_fresh: file's noise, new draw every epoch"}
#: hues of their own: shades of the comparison arms' blue and orange were too close to read
COLOR = {**nc.COLOR, "v2_fresh": "#C51B7D", "file_fresh": "#35978F"}
#: the cap on epochs (sbi trains up to one more)
CAP = 200
CROSS, PER_ROW, LAIBSON = "cross_noise.json", "cross_noise_rows.npz", "laibson_comparison.json"


def run_dir(arm: str, root: Path = ROOT, base: Path = we.ROOT,
            copies_root: Path = nc.ROOT) -> Path:
    """An arm's members' directory: w07 is RESULTS 64's, the copy and
    one-draw file arms RESULTS 67's."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known {ARMS}")
    return arm_root(arm, root, base, copies_root) / "w07"


def arm_root(arm: str, root: Path = ROOT, base: Path = we.ROOT,
             copies_root: Path = nc.ROOT) -> Path:
    """Where an arm's population scorings and PSID posteriors live."""
    if arm == "w07":
        return base
    return root / arm if arm in NEW else copies_root / arm


def training_extra(arm_dir: Path, run: Path) -> dict:
    """Per member: the rise in validation loss from the best to the stop,
    whether the cap was reached, and whether the best weights were restored
    there (the fresh arms' config)."""
    out = {}
    for f in sorted(arm_dir.glob("s[0-9]*.csv")):
        m = tm.read(f)
        cfg_f = run / f.stem / "results.json"
        cfg = json.loads(cfg_f.read_text())["_config"] if cfg_f.exists() else {}
        cap = cfg.get("max_epochs") or CAP
        out[f.stem] = {"rise_after_best": float(m["val_loss"][-1] - np.nanmin(m["val_loss"])),
                       "reached_cap": bool(len(m["epoch"]) > cap),
                       "restored_best_at_cap": bool(cfg.get("fresh_noise_restored_best_at_cap",
                                                            False))}
    return out


def member_log_q(run: Path) -> np.ndarray:
    """Each member's held-out log q, from the arm's held-out check."""
    h = json.loads((run / "heldout_check.json").read_text())
    return np.array([h[k]["log_q_weighted"] for k in sorted(h) if k.startswith("member_")])


def member_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """Difference of two arms' member means, with its se and Welch's p.
    Added after the run: the paired interval resamples households only, and
    this one carries the training's own randomness."""
    from scipy import stats

    return {"diff": float(a.mean() - b.mean()),
            "se": float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))),
            "welch_p": float(stats.ttest_ind(a, b, equal_var=False).pvalue),
            "range_a": [float(a.min()), float(a.max())],
            "range_b": [float(b.min()), float(b.max())]}


# -------------------------------------------------------------------- measures

def measures(args) -> None:
    """The arms' measures and comparisons (fresh_noise.json)."""
    from scripts import laibson_comparison as lc

    res = {"arms": {}}
    means, pops = {}, {}
    for a in ARMS:
        r = arm_root(a, args.root, args.base, args.copies_root)
        pops[a] = json.loads((r / "population.json").read_text())
        run = run_dir(a, args.root, args.base, args.copies_root)
        mdir = args.metrics_dir / tm.run_name(run)
        res["arms"][a] = {**we.arm_measures(run), "population": pops[a]["arms"]["w07"],
                          "low_beta": lb.bands(dict(np.load(run / "sbc" / "sbc16.npz"))),
                          "training": nc.training(mdir),
                          "training_extra": training_extra(mdir, run)}
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
    # every arm on a noise is scored on the same held-out windows (copy 0)
    for net in ARMS:
        for dup, orig in NOISE_OF.items():
            if dup != orig and not np.array_equal(rows[f"{net}__{dup}"], rows[f"{net}__{orig}"]):
                raise SystemExit(f"{net}: {dup}'s held-out windows differ from {orig}'s")
    cmp_ = {}
    for a, b in PAIRS:
        n = NOISE_OF[a]
        cmp_[f"{a}/{b}"] = {
            "log_q": nc.paired_log_q({k: rows[f"{k}__{n}"] for k in (a, b)}, w, a, b,
                                     args.n_boot, args.seed),
            "on_noise": n,
            "members": member_diff(*(member_log_q(run_dir(k, args.root, args.base,
                                                           args.copies_root)) for k in (a, b))),
            "info": nf.info_ratio(A[a]["heldout"]["contraction"], A[b]["heldout"]["contraction"]),
            "pairs": [A[a]["sbc"]["pairs_box"][p]["ordered_p90"]
                      / A[b]["sbc"]["pairs_box"][p]["ordered_p90"] for p in we.NAMES],
            "population_se": (np.array(A[a]["population"]["se"])
                              / A[b]["population"]["se"]).tolist(),
            "population_se_ci90": nf.ratio_ci(boot, a, b)}
    res["comparisons"] = cmp_
    res["cross_noise"] = nf.cross_noise(args.cross)
    res["psid"] = nc.psid_summary({a: arm_root(a, args.root, args.base, args.copies_root)
                                   / "psid_w07_couples" for a in ARMS})
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
    out = args.root / "fresh_noise.json"
    out.write_text(json.dumps(res, indent=1))
    nc.report(res, ARMS)
    print("\ntraining, beyond the table: rise after the best; members at the cap (restored)")
    for a in ARMS:
        ex = A[a]["training_extra"]
        if ex:
            rise = [v["rise_after_best"] for v in ex.values()]
            print(f"{a:11s} rise {min(rise):.3f}-{max(rise):.3f}; at the cap "
                  f"{sum(v['reached_cap'] for v in ex.values())} "
                  f"({sum(v['restored_best_at_cap'] for v in ex.values())} restored)")
    print("\nheld-out log q of the members (added after the run: training randomness included)")
    for k, c in cmp_.items():
        m = c["members"]
        print(f"{k:22s} member means {m['diff']:+.3f} (se {m['se']:.3f}, Welch p "
              f"{m['welch_p']:.3f}); members {m['range_a'][0]:.3f}-{m['range_a'][1]:.3f} "
              f"against {m['range_b'][0]:.3f}-{m['range_b'][1]:.3f}")
    print(f"\nwrote {out}")


# --------------------------------------------------------------------- figures

def figures(args) -> None:
    """Figures 133 and 136."""
    curves(args, args.fig_dir / "133_fresh_noise_training.png")
    res = json.loads((args.root / "fresh_noise.json").read_text())
    precision(res, args.fig_dir / "136_fresh_noise_precision.png")


def curves(args, path: Path) -> None:
    """133: training (dashed) and validation (solid) loss per member against
    passes over the clean windows, one panel per noise; the best epoch
    marked."""
    import matplotlib.pyplot as plt

    panels = ((("w07", "v2_k8", "v2_fresh"), "v2's noise"),
              (("file_k1", "file_k8", "file_fresh"), "the file's noise"))
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, (arms, title) in zip(axs, panels, strict=True):
        for a in arms:
            d = args.metrics_dir / tm.run_name(run_dir(a, args.root, args.base, args.copies_root))
            for k, f in enumerate(sorted(d.glob("s[0-9]*.csv"))):
                m = tm.read(f)
                ax.plot(m["passes"], m["val_loss"], "-", color=COLOR[a], lw=1.2,
                        label=LABEL[a] if k == 0 else None)
                ax.plot(m["passes"], m["train_loss"], "--", color=COLOR[a], lw=0.8, alpha=0.7)
                b = int(np.nanargmin(m["val_loss"]))
                ax.plot(m["passes"][b], m["val_loss"][b], "o", color=COLOR[a], ms=5)
        ax.set(xlabel="passes over the clean training windows (epochs × copies)",
               ylabel="loss (−log q, flow scale)", title=title)
        ax.grid(ls=":", lw=0.5)
        ax.legend(fontsize=8, title="solid validation, dashed training; ● best", title_fontsize=8)
    fig.suptitle("RESULTS 68: training curves from the per-epoch CSVs (training_metrics/); "
                 "a new noise draw every epoch against one draw and 8 fixed copies", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def precision(res: dict, path: Path) -> None:
    """136: the paired held-out log q differences with 95% intervals, and
    information and population se against the comparison arm."""
    import matplotlib.pyplot as plt

    from scripts import annual_waves as aw

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    c = res["comparisons"]
    keys = [f"{a}/{b}" for a, b in PAIRS]
    for i, k in enumerate(keys):
        lq = c[k]["log_q"]
        axs[0].errorbar(i, lq["diff"], yerr=[[lq["diff"] - lq["ci95"][0]],
                                             [lq["ci95"][1] - lq["diff"]]],
                        fmt="o", color=COLOR[k.split("/")[0]], capsize=4)
    axs[0].axhline(0, color="k", lw=0.8)
    axs[0].set_xticks(range(len(keys)), [k.replace("/", "\n− ") for k in keys], fontsize=8)
    axs[0].set(ylabel="held-out log q difference (95% interval)",
               title="paired on the same households and windows")
    for ax, key, ylab in ((axs[1], "info", "information ÷ the comparison arm's (held-out)"),
                          (axs[2], "population_se", "population se at θ₀ ÷ the comparison arm's")):
        for i, k in enumerate(keys):
            v = c[k][key]
            for j in range(len(we.NAMES)):
                ax.plot(i + 0.08 * (j - 1.5), v[j], "o", color=aw.PARAM_COLOR[j],
                        label=we.NAMES[j] if i == 0 else None)
                if key == "population_se":
                    lo, hi = c[k]["population_se_ci90"][j]
                    ax.plot([i + 0.08 * (j - 1.5)] * 2, [lo, hi], "-", color=aw.PARAM_COLOR[j],
                            lw=1)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xticks(range(len(keys)), [k.replace("/", "\n÷ ") for k in keys], fontsize=8)
        ax.set(ylabel=ylab)
        ax.grid(ls=":", lw=0.5)
    axs[1].legend(fontsize=8)
    axs[0].grid(ls=":", lw=0.5)
    fig.suptitle("RESULTS 68: what a new noise draw every epoch changes", fontsize=10)
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
        p.add_argument("--copies_root", type=Path, default=nc.ROOT,
                       help="RESULTS 67's root, holding v2_k8, file_k1 and file_k8.")
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
        args.cross = args.cross or args.root / CROSS
        args.per_row = args.per_row or args.root / PER_ROW
        args.laibson = args.laibson or args.root / LAIBSON
    args.fn(args)


if __name__ == "__main__":
    main()
