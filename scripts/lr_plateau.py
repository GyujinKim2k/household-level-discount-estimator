"""RESULTS 69: reduce-on-plateau and a harder early stop.

Four 7-wave arms (w07's window: biennial, start 24-51; everything else as
RESULTS 64's w07), five members each, the user's request of 2026-11-11
(fixed copies as RESULTS 67, all 8 in one epoch; patience 60 passes, cap 600):

* ``v2_k8_rop``, ``v2_fresh_rop``: v2's noise, 8 fixed copies (as v2_k8) and
  a new draw every epoch (as v2_fresh);
* ``file_k8_rop``, ``file_fresh_rop``: the same on the methodology file's
  noise (as file_k8 and file_fresh).

Each trains as its counterpart, except that the learning rate is halved
(floor 1e-5) at the 5th pass in a row without a gain in validation loss, and
training stops after 60 passes without one (cap 600; the best weights
restored there). In epochs of 8 passes that is a cut at the first epoch
without a gain, a stop after 8 and a cap of 75 (``optionA.schedule``).

Commands:

``measures``
    The tables (lr_plateau.json): calibration, held-out log q paired on
    identical windows (the schedule against each counterpart; fresh against
    fixed under it), the members' log q, information, ordering, low beta,
    the population se at theta0 (paired resampling), training from the
    per-epoch CSVs (cuts, the rate at the best, passes to the best and the
    stop, the cap), each member's learning rate rebuilt from its validation
    losses (:func:`expected_lr`) against the rate its CSV logged,
    cross-noise, PSID couples and the representative theta (RESULTS 68's
    arms checked unchanged).

``figures``
    140 (training curves and learning rates from the CSVs) and 145 (paired
    log q and precision).

Usage::

    PYTHONPATH=.:src .venv/bin/python scripts/lr_plateau.py measures
    PYTHONPATH=.:src .venv/bin/python scripts/lr_plateau.py figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import fresh_noise as fn
from scripts import low_beta_check as lb
from scripts import noise_copies as nc
from scripts import noise_free_waves as nf
from scripts import training_metrics as tm
from scripts import waves_experiment as we

NEW = ("v2_k8_rop", "v2_fresh_rop", "file_k8_rop", "file_fresh_rop")
ARMS = ("w07", "v2_k8", "v2_fresh", "v2_k8_rop", "v2_fresh_rop",
        "file_k1", "file_k8", "file_fresh", "file_k8_rop", "file_fresh_rop")
ROOT = Path("outputs/lr_plateau")
#: (arm, against), each on one noise; set before the run: the schedule against
#: each counterpart, then fresh against fixed copies under the schedule
PAIRS = (("v2_k8_rop", "v2_k8"), ("v2_fresh_rop", "v2_fresh"),
         ("file_k8_rop", "file_k8"), ("file_fresh_rop", "file_fresh"),
         ("v2_fresh_rop", "v2_k8_rop"), ("file_fresh_rop", "file_k8_rop"))
#: each arm's training noise, as the cross-noise columns name them
NOISE_OF = {a: "w07" if a == "w07" or a.startswith("v2") else "file_k1" for a in ARMS}
LABEL = {**fn.LABEL, "v2_k8_rop": "v2_k8_rop: v2 noise, 8 copies, plateau schedule",
         "v2_fresh_rop": "v2_fresh_rop: v2 noise, new draw, plateau schedule",
         "file_k8_rop": "file_k8_rop: file's noise, 8 copies, plateau schedule",
         "file_fresh_rop": "file_fresh_rop: file's noise, new draw, plateau schedule"}
#: hues apart from every arm they share a figure with
COLOR = {**fn.COLOR, "v2_k8_rop": "#222222", "v2_fresh_rop": "#C9A100",
         "file_k8_rop": "#7FBC41", "file_fresh_rop": "#F781BF"}
#: the starting rate (optionA train)
LR0 = 1e-3
CROSS, PER_ROW, LAIBSON = "cross_noise.json", "cross_noise_rows.npz", "laibson_comparison.json"


def arm_root(arm: str, root: Path = ROOT, base: Path = we.ROOT, copies_root: Path = nc.ROOT,
             fresh_root: Path = fn.ROOT) -> Path:
    """Where an arm's population scorings and PSID posteriors live: w07 is
    RESULTS 64's, the copy and one-draw arms RESULTS 67's, the fresh arms
    RESULTS 68's."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known {ARMS}")
    if arm == "w07":
        return base
    if arm in NEW:
        return root / arm
    return fresh_root / arm if arm in fn.NEW else copies_root / arm


def run_dir(arm: str, *roots) -> Path:
    """An arm's members' directory."""
    return arm_root(arm, *roots) / "w07"


def expected_lr(val_loss, lr0: float, factor: float, patience: int, floor: float,
                eps: float = 1e-8) -> tuple[np.ndarray, list[int]]:
    """The rate each epoch trained at, rebuilt from the validation losses
    alone: after epoch e (all but the last, after which nothing trains) a gain
    is a loss below the best so far; at the ``patience + 1``-th epoch in a row
    without one the rate becomes ``max(rate * factor, floor)`` (kept when the
    change is below ``eps``) and the count restarts. Written apart from
    torch's scheduler, to check what the runs did. Returns the rates of
    epochs 1..n and the epochs after which the rate fell."""
    lr, best, bad, cuts = lr0, np.inf, 0, []
    out = [lr]
    for e, v in enumerate(val_loss[:-1], start=1):
        if v < best:
            best, bad = v, 0
        else:
            bad += 1
        if bad > patience:
            new = max(lr * factor, floor)
            if lr - new > eps:
                lr = new
                cuts.append(e)
            bad = 0
        out.append(lr)
    return np.array(out), cuts


def schedule_check(csv: Path, run: Path) -> dict:
    """One member's logged rates against :func:`expected_lr` from its own
    validation losses, and the cuts against those it recorded."""
    m = tm.read(csv)
    cfg = json.loads((run / csv.stem / "results.json").read_text())["_config"]
    p = cfg["lr_plateau"]
    lr, cuts = expected_lr(m["val_loss"], LR0, p["factor"], p["patience"], p["min_lr"])
    summ = json.loads((run / csv.stem / "training_summary.json").read_text())
    err = float(np.max(np.abs(m["lr"] / lr - 1)))
    return {"max_rel_err": err, "cuts": cuts, "recorded_cuts": summ["lr_cut_after_epochs"],
            "ok": bool(err < 1e-5 and cuts == summ["lr_cut_after_epochs"])}


def training_extra(arm_dir: Path, run: Path, arm: str) -> dict:
    """Per member: the rise in validation loss from the best to the stop,
    the cap and whether the best weights were restored there, and for the
    plateau arms the cuts, the rate at the best, passes to the stop and to
    the floor, and :func:`schedule_check`."""
    out = {}
    for f in sorted(arm_dir.glob("s[0-9]*.csv")):
        m = tm.read(f)
        cfg = json.loads((run / f.stem / "results.json").read_text())["_config"]
        summ = json.loads((run / f.stem / "training_summary.json").read_text())
        b = int(np.nanargmin(m["val_loss"]))
        cap = cfg.get("max_epochs") or fn.CAP
        rec = {"rise_after_best": float(m["val_loss"][-1] - m["val_loss"][b]),
               "reached_cap": bool(len(m["epoch"]) > cap),
               "restored_best_at_cap": bool(summ.get(
                   "restored_best_at_cap", cfg.get("fresh_noise_restored_best_at_cap", False))),
               "passes_at_stop": float(m["passes"][-1])}
        if arm in NEW:
            floor = cfg["lr_plateau"]["min_lr"]
            at_floor = np.flatnonzero(np.isclose(m["lr"], floor, rtol=1e-5))
            rec |= {"lr_at_best": float(m["lr"][b]), "n_cuts": len(summ["lr_cut_after_epochs"]),
                    "passes_at_floor": float(m["passes"][at_floor[0]]) if len(at_floor) else None,
                    "schedule": schedule_check(f, run)}
        out[f.stem] = rec
    return out


# -------------------------------------------------------------------- measures

def measures(args) -> None:
    """The arms' measures and comparisons (lr_plateau.json)."""
    from scripts import laibson_comparison as lc

    roots = (args.root, args.base, args.copies_root, args.fresh_root)
    res = {"arms": {}}
    means, pops = {}, {}
    for a in ARMS:
        r = arm_root(a, *roots)
        pops[a] = json.loads((r / "population.json").read_text())
        run = run_dir(a, *roots)
        mdir = args.metrics_dir / tm.run_name(run)
        res["arms"][a] = {**we.arm_measures(run), "population": pops[a]["arms"]["w07"],
                          "low_beta": lb.bands(dict(np.load(run / "sbc" / "sbc16.npz"))),
                          "training": nc.training(mdir),
                          "training_extra": training_extra(mdir, run, a)}
        means[a] = {t: np.load(r / "population" / f"post_w07_{t}.npz")["mean"]
                    for t in we.POINTS}
    bad = [f"{a}/{s}" for a in NEW for s, v in res["arms"][a]["training_extra"].items()
           if not v["schedule"]["ok"]]
    if bad:
        raise SystemExit(f"the logged learning rates do not follow the rule: {bad}")
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
            "members": fn.member_diff(*(fn.member_log_q(run_dir(k, *roots)) for k in (a, b))),
            "info": nf.info_ratio(A[a]["heldout"]["contraction"], A[b]["heldout"]["contraction"]),
            "pairs": [A[a]["sbc"]["pairs_box"][p]["ordered_p90"]
                      / A[b]["sbc"]["pairs_box"][p]["ordered_p90"] for p in we.NAMES],
            "population_se": (np.array(A[a]["population"]["se"])
                              / A[b]["population"]["se"]).tolist(),
            "population_se_ci90": nf.ratio_ci(boot, a, b)}
    res["comparisons"] = cmp_
    res["cross_noise"] = nf.cross_noise(args.cross)
    res["psid"] = nc.psid_summary({a: arm_root(a, *roots) / "psid_w07_couples" for a in ARMS})
    if args.laibson.exists():
        lai = json.loads(args.laibson.read_text())
        keys = ("representative_theta", "representative_se", "representative_ci90",
                "converged", "at_box_edge", "last_simulated_misfit_over_psid_se")
        res["representative"] = {name: {k: arm[k] for k in keys}
                                 for name, arm in lai["arms"].items()}
        res["laibson"] = {"theta": lai["laibson"]["theta"],
                          "se_sampling_only": lai["laibson_se"]["sampling only"]}
        # RESULTS 68's arms come first in the list, so the shared bootstrap
        # stream (68.8) leaves them as they were
        if args.laibson_before.exists():
            old = json.loads(args.laibson_before.read_text())["arms"]
            res["representative_unchanged_from_68"] = {
                name: bool(all(np.array_equal(arm[k], lai["arms"][name][k])
                               for k in ("representative_theta", "representative_se")))
                for name, arm in old.items()}
    res["n_boot"], res["seed"] = args.n_boot, args.seed
    out = args.root / "lr_plateau.json"
    out.write_text(json.dumps(res, indent=1))
    nc.report(res, ARMS)
    print("\ntraining, beyond the table: rise after the best; passes at the stop; members at "
          "the cap (restored); cuts, the rate at the best and passes to the floor")
    for a in ARMS:
        ex = A[a]["training_extra"].values()
        if not ex:
            continue
        rise = [v["rise_after_best"] for v in ex]
        stop = [v["passes_at_stop"] for v in ex]
        line = (f"{a:15s} rise {min(rise):.3f}-{max(rise):.3f}; stop {min(stop):.0f}-"
                f"{max(stop):.0f} passes; at the cap {sum(v['reached_cap'] for v in ex)} "
                f"({sum(v['restored_best_at_cap'] for v in ex)} restored)")
        if a in NEW:
            fl = [v["passes_at_floor"] for v in ex]
            line += (f"; cuts {min(v['n_cuts'] for v in ex)}-{max(v['n_cuts'] for v in ex)}; "
                     f"rate at the best {min(v['lr_at_best'] for v in ex):.2g}-"
                     f"{max(v['lr_at_best'] for v in ex):.2g}; floor reached by "
                     f"{sum(f is not None for f in fl)}"
                     + (f" at {min(f for f in fl if f is not None):.0f}-"
                        f"{max(f for f in fl if f is not None):.0f} passes"
                        if any(f is not None for f in fl) else ""))
        print(line)
    print("\nheld-out log q of the members (training randomness included)")
    for k, c in cmp_.items():
        m = c["members"]
        print(f"{k:28s} member means {m['diff']:+.3f} (se {m['se']:.3f}, Welch p "
              f"{m['welch_p']:.3f}); members {m['range_a'][0]:.3f}-{m['range_a'][1]:.3f} "
              f"against {m['range_b'][0]:.3f}-{m['range_b'][1]:.3f}")
    if "representative_unchanged_from_68" in res:
        same = res["representative_unchanged_from_68"]
        changed = [k for k, v in same.items() if not v]
        print(f"\nRESULTS 68's arms' representative theta and se unchanged: "
              f"{len(same) - len(changed)} of {len(same)}"
              + (f" (changed: {changed})" if changed else ""))
    print(f"\nwrote {out}")


# --------------------------------------------------------------------- figures

def figures(args) -> None:
    """Figures 140 and 145."""
    curves(args, args.fig_dir / "140_lr_plateau_training.png")
    res = json.loads((args.root / "lr_plateau.json").read_text())
    precision(res, args.fig_dir / "145_lr_plateau_precision.png")


def curves(args, path: Path) -> None:
    """140: training (dashed) and validation (solid) loss per member against
    passes over the clean windows, one column per noise, the best epoch
    marked; below, the learning rate each epoch trained at."""
    import matplotlib.pyplot as plt

    roots = (args.root, args.base, args.copies_root, args.fresh_root)
    panels = ((("v2_k8", "v2_fresh", "v2_k8_rop", "v2_fresh_rop"), "v2's noise"),
              (("file_k8", "file_fresh", "file_k8_rop", "file_fresh_rop"), "the file's noise"))
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.5), sharex="col",
                            gridspec_kw={"height_ratios": [3, 1.2]})
    for j, (arms, title) in enumerate(panels):
        ax, lax = axs[0, j], axs[1, j]
        for a in arms:
            d = args.metrics_dir / tm.run_name(run_dir(a, *roots))
            for k, f in enumerate(sorted(d.glob("s[0-9]*.csv"))):
                m = tm.read(f)
                ax.plot(m["passes"], m["val_loss"], "-", color=COLOR[a], lw=1.1,
                        label=LABEL[a] if k == 0 else None)
                ax.plot(m["passes"], m["train_loss"], "--", color=COLOR[a], lw=0.7, alpha=0.6)
                b = int(np.nanargmin(m["val_loss"]))
                ax.plot(m["passes"][b], m["val_loss"][b], "o", color=COLOR[a], ms=5)
                lax.step(m["passes"], m["lr"], where="post", color=COLOR[a], lw=1,
                         alpha=0.8)
        ax.set(ylabel="loss (−log q, flow scale)", title=title)
        ax.grid(ls=":", lw=0.5)
        ax.legend(fontsize=8, title="solid validation, dashed training; ● best",
                  title_fontsize=8)
        lax.set(yscale="log", ylabel="learning rate",
                xlabel="passes over the clean training windows (epochs × copies)")
        lax.grid(ls=":", lw=0.5)
    fig.suptitle("RESULTS 69: training curves and learning rates from the per-epoch CSVs "
                 "(training_metrics/); the plateau schedule with patience 60 against RESULTS "
                 "67/68's constant rate with patience 20", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def precision(res: dict, path: Path) -> None:
    """145: the paired held-out log q differences with 95% intervals, and
    information and population se against the comparison arm."""
    import matplotlib.pyplot as plt

    from scripts import annual_waves as aw

    fig, axs = plt.subplots(1, 3, figsize=(20, 5.4))
    c = res["comparisons"]
    keys = [f"{a}/{b}" for a, b in PAIRS]
    for i, k in enumerate(keys):
        lq = c[k]["log_q"]
        axs[0].errorbar(i, lq["diff"], yerr=[[lq["diff"] - lq["ci95"][0]],
                                             [lq["ci95"][1] - lq["diff"]]],
                        fmt="o", color=COLOR[k.split("/")[0]], capsize=4)
    axs[0].axhline(0, color="k", lw=0.8)
    axs[0].axvline(3.5, color="0.6", lw=0.8, ls=":")
    axs[0].set_xticks(range(len(keys)), [k.replace("/", "\n− ") for k in keys], fontsize=7)
    axs[0].set(ylabel="held-out log q difference (95% interval)",
               title="paired on the same households and windows\n(left: the schedule against "
                     "each counterpart; right: fresh against fixed under it)")
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
        ax.axvline(3.5, color="0.6", lw=0.8, ls=":")
        ax.set_xticks(range(len(keys)), [k.replace("/", "\n÷ ") for k in keys], fontsize=7)
        ax.set(ylabel=ylab)
        ax.grid(ls=":", lw=0.5)
    axs[1].legend(fontsize=8)
    axs[0].grid(ls=":", lw=0.5)
    fig.suptitle("RESULTS 69: what reduce-on-plateau and a harder early stop change",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn_ in (("measures", measures), ("figures", figures)):
        p = sub.add_parser(name, help=fn_.__doc__.splitlines()[0] if fn_.__doc__ else None)
        p.add_argument("--root", type=Path, default=ROOT)
        p.add_argument("--base", type=Path, default=we.ROOT,
                       help="RESULTS 64's root, holding w07, its population and PSID posteriors.")
        p.add_argument("--copies_root", type=Path, default=nc.ROOT,
                       help="RESULTS 67's root, holding v2_k8, file_k1 and file_k8.")
        p.add_argument("--fresh_root", type=Path, default=fn.ROOT,
                       help="RESULTS 68's root, holding v2_fresh and file_fresh.")
        p.add_argument("--metrics_dir", type=Path, default=tm.METRICS_DIR)
        p.set_defaults(fn=fn_)
    m = sub.choices["measures"]
    m.add_argument("--sim_cache", type=Path, default=we.ROOT / "population")
    m.add_argument("--cross", type=Path, default=None, help=f"default <root>/{CROSS}")
    m.add_argument("--per_row", type=Path, default=None, help=f"default <root>/{PER_ROW}")
    m.add_argument("--laibson", type=Path, default=None, help=f"default <root>/{LAIBSON}")
    m.add_argument("--laibson_before", type=Path, default=fn.ROOT / LAIBSON,
                   help="RESULTS 68's fit, whose arms must come out unchanged.")
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
