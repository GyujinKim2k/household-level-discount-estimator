"""RESULTS 65: annual waves against RESULTS 64's biennial arms, on simulated
households.

Two arms, v2 on the box with the beta strip as every RESULTS 64 arm, trained
on 14 and 20 one-year waves (a14, a20; the user's choice, 2026-11-07). Every
year a window reads lies inside ages 24-64, as in RESULTS 64: an annual wave
at age s reads its stocks at the start of s and its flows of s - 1 (flow_lag
1, PSID's timing), so annual windows start at 25-51 (a14) and 25-45 (a20)
(``waves_experiment.start_low/start_high``). Against the biennial arms, on
the same held-out and SBC households:

* the same number of waves: a14 against w14, a20 against w20 (more years
  against more frequent observations);
* about the same years: a14 against w07, a20 against w10 (a14 reads w07's
  years and one more; what the years between biennial waves add).

Commands:

``measures``
    Per arm (w07, a14, w14, w10, a20, w20): calibration, one household,
    telling households apart, low beta by band (RESULTS 64.12), the
    population at 409 households sharing theta0 and PSID couples' two cases,
    training; the comparisons; the RESULTS tables.

``population_check``
    The population se's 90% interval (households resampled within card type,
    paired across arms, so the ratios have intervals too); the share of
    beta's variance that comes through each statistic in J^-1; cardholders'
    posterior spread at theta0 by window start age (RESULTS 65.6). CPU, from
    the cached scorings.

``figures``
    112 (precision against waves and against years read), 115 (one held-out
    household under the six arms, 68/95% contours).

Usage::

    PYTHONPATH=. .venv/bin/python scripts/annual_waves.py measures
    PYTHONPATH=. .venv/bin/python scripts/annual_waves.py population_check
    PYTHONPATH=. .venv/bin/python scripts/annual_waves.py figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts import low_beta_check as lb
from scripts import waves_experiment as we

ARMS = ("w07", "a14", "w14", "w10", "a20", "w20")
ANNUAL = ("a14", "a20")
#: (annual, biennial): the same number of waves, and about the same years
SAME_COUNT = (("a14", "w14"), ("a20", "w20"))
SAME_YEARS = (("a14", "w07"), ("a20", "w10"))
POPULATION = "population_annual.json"
COLOR = {"w07": "#3B75AF", "w10": "#3B75AF", "a14": "#C53A32", "a20": "#C53A32",
         "w14": "#519E3E", "w20": "#519E3E"}
PARAM_COLOR = ("#C53A32", "#3B75AF", "#519E3E", "#EF8636")


def years_read(label: str) -> int:
    """Calendar years a window reads: a biennial wave's two years hold its
    flows and stocks; an annual window also reads the flows of the year
    before its first wave."""
    n, wy = we.parse_arm(label)
    return wy * n + max(0, we.FLOW_LAG - (wy - 1))


# -------------------------------------------------------------------- measures

def measures(args) -> None:
    """The six arms' measures and comparisons (annual_waves.json)."""
    pop = json.loads((args.root / POPULATION).read_text())
    res = {"arms": {}}
    for a in args.arms:
        n, wy = we.parse_arm(a)
        run = we.arm_dir(a, args.root)
        res["arms"][a] = {"n_waves": n, "wave_years": wy, "years_read": years_read(a),
                          "start_low": we.start_low(wy), "start_high": we.start_high(n, wy),
                          **we.arm_measures(run), "population": pop["arms"][a],
                          "low_beta": lb.bands(dict(np.load(run / "sbc" / "sbc16.npz")))}
    if we.PSID_ARM in pop["arms"]:
        cases, res["psid_remainder"] = we.psid_cases(
            {a: res["arms"][a]["population"]["se"] for a in args.arms}, "w07",
            np.array(pop["arms"][we.PSID_ARM]["cov"]))
        for a in args.arms:
            res["arms"][a]["psid_scale"] = cases[a]
    res["laibson_se"] = json.loads(we.REFERENCE.read_text())["laibson_se"]
    out = args.root / "annual_waves.json"
    out.write_text(json.dumps(res, indent=1))
    report(res)
    print(f"\nwrote {out}")


def report(res: dict) -> None:
    """The RESULTS tables, as markdown."""
    a = res["arms"]
    ns = [x for x in ARMS if x in a]
    hdr = "| | " + " | ".join(ns) + " |\n|---|" + "---|" * len(ns)

    def row(label, f, fmt="{:.3f}"):
        vals = [f(a[n]) for n in ns]
        print(f"| {label} | " + " | ".join("–" if v is None else fmt.format(v) for v in vals)
              + " |")

    print("\nthe arms\n" + hdr)
    row("waves", lambda r: r["n_waves"], "{}")
    row("years per wave", lambda r: r["wave_years"], "{}")
    row("years read", lambda r: r["years_read"], "{}")
    print("| start ages | " + " | ".join(f"{a[n]['start_low']}–{a[n]['start_high']}"
                                        for n in ns) + " |")
    print("\ncalibration and one household\n" + hdr)
    for j, p in enumerate(we.NAMES):
        print(f"| {p}: held-out / SBC 90% coverage | " + " | ".join(
            f"{a[n]['heldout']['cov90'][j]:.3f} / {a[n]['sbc_coverage_90'][j]:.3f}"
            for n in ns) + " |")
    row("held-out log q", lambda r: r["heldout"]["log_q"])
    for j, p in enumerate(we.NAMES):
        row(f"{p}: share of prior variance removed", lambda r, j=j: r["heldout"]["contraction"][j])
    for j, p in enumerate(we.NAMES):
        row(f"{p}: the data's information ÷ w07's (s/(1-s))",
            lambda r, j=j: we._info(r["heldout"]["contraction"][j])
            / we._info(a["w07"]["heldout"]["contraction"][j]))
    for j, p in enumerate(we.NAMES):
        row(f"{p}: corr(posterior mean, truth), held-out", lambda r, j=j: r["heldout"]["corr"][j])
    for j, p in enumerate(we.NAMES):
        row(f"{p}: median posterior sd (SBC)",
            lambda r, j=j: r["sbc"]["median_posterior_sd"][j], "{:.4f}")
    print("\ntelling households apart (SBC households, truth known)\n" + hdr)
    for p in we.NAMES:
        row(f"{p}: one 90% interval ÷ 5–95% range of means",
            lambda r, p=p: r["sbc"]["interval_vs_range"][p]["ratio"])
    for p in we.NAMES:
        row(f"{p}: pairs ordered at p ≥ 0.9",
            lambda r, p=p: r["sbc"]["pairs_box"][p]["ordered_p90"])
    for p in we.NAMES:
        row(f"{p}: of which right",
            lambda r, p=p: r["sbc"]["pairs_box"][p]["right_when_ordered"])
    row("households with ≥ 2 of 4 parameters outside their central 95%",
        lambda r: r["sbc"]["outside_95"]["at_least_2"])
    print("\nlow beta by band (RESULTS 64.12): median 90% width, true beta 0.1-0.3 / 0.8-1.0;"
          " ranking 0.1-0.2 vs 0.3-0.4 / 0.7-0.8 vs 0.9-1.0\n" + hdr)
    row("β 90% width, 0.1–0.3", lambda r: r["low_beta"]["bands"][0]["median_width_90"])
    row("β 90% width, 0.8–1.0", lambda r: r["low_beta"]["bands"][-1]["median_width_90"])
    row("ranking, 0.1–0.2 vs 0.3–0.4", lambda r: r["low_beta"]["pairs"][0]["auc_beta_mean"])
    row("ranking, 0.7–0.8 vs 0.9–1.0", lambda r: r["low_beta"]["pairs"][-1]["auc_beta_mean"])
    print("\nthe population: 409 households at theta0, sharing it\n" + hdr)
    for j, p in enumerate(we.NAMES):
        row(f"{p}: se", lambda r, j=j: r["population"]["se"][j], "{:.4f}")
    base = np.array(a["w07"]["population"]["se"])
    for j, p in enumerate(we.NAMES):
        row(f"{p}: se ÷ w07's", lambda r, j=j: r["population"]["se"][j] / base[j])
    if "psid_remainder" in res:
        ss = np.array(res["laibson_se"]["sampling only"])
        print(f"\nPSID couples ({we.N_POP}), from their 7-wave se: (a) the remainder shrinks "
              "like the noise, (b) it is held\n" + hdr)
        for j, p in enumerate(we.NAMES[:3]):
            print(f"| {p}: se, (a) / (b) | " + " | ".join(
                f"{a[n]['psid_scale']['se_a'][j]:.4f} / {a[n]['psid_scale']['se_b'][j]:.4f}"
                for n in ns) + " |")
        for j, p in enumerate(we.NAMES[:3]):
            print(f"| {p}: ÷ their sampling-only se, (a) / (b) | " + " | ".join(
                f"{a[n]['psid_scale']['se_a'][j] / ss[j]:.2f} / "
                f"{a[n]['psid_scale']['se_b'][j] / ss[j]:.2f}" for n in ns) + " |")
    print("\nunderfitting: each arm's members (training_summary.json)\n" + hdr)
    uf = {n: list(a[n]["underfit"].values()) for n in ns}
    print("| epochs | " + " | ".join(
        f"{min(m['epochs'] for m in uf[n])}–{max(m['epochs'] for m in uf[n])}" for n in ns)
        + " |")
    for key, lab in (("train_minus_val_end", "training − validation loss at the end"),
                     ("best_val", "best validation loss"), ("minutes", "minutes per member")):
        print(f"| {lab} (mean) | " + " | ".join(
            f"{np.mean([m[key] for m in uf[n]]):.3f}" for n in ns) + " |")
    print("\ncomparisons: annual ÷ biennial (information s/(1-s), pairs ordered, population se); "
          "log q annual − biennial")
    for title, pairs in (("same number of waves", SAME_COUNT), ("about the same years",
                                                                 SAME_YEARS)):
        for an, bi in pairs:
            if an not in a or bi not in a:
                continue
            ra, rb = a[an], a[bi]
            info = [we._info(ra["heldout"]["contraction"][j])
                    / we._info(rb["heldout"]["contraction"][j]) for j in range(4)]
            pairs_ = [ra["sbc"]["pairs_box"][p]["ordered_p90"]
                      / rb["sbc"]["pairs_box"][p]["ordered_p90"] for p in we.NAMES]
            se = [ra["population"]["se"][j] / rb["population"]["se"][j] for j in range(4)]
            dq = ra["heldout"]["log_q"] - rb["heldout"]["log_q"]
            print(f"{title}, {an} vs {bi}: log q {dq:+.3f}"
                  f"; information {' '.join(f'{v:.2f}' for v in info)}"
                  f"; pairs {' '.join(f'{v:.2f}' for v in pairs_)}"
                  f"; population se {' '.join(f'{v:.2f}' for v in se)}")


# ------------------------------------------------------------ population check

START_GROUPS = ((24, 31), (32, 37), (38, 44), (45, 51))


def beta_through(J: np.ndarray, v_stat: np.ndarray) -> dict:
    """Beta's variance in the representative theta, V_u[0, 0] = r V r' with r
    J^-1's beta row: the share each statistic's own variance contributes,
    and the rest (their covariances)."""
    r = np.linalg.inv(J)[0]
    total = r @ v_stat @ r
    own = r ** 2 * np.diag(v_stat) / total
    return {"j_inv_beta_row": r.tolist(), "own_share": own.tolist(),
            "cross_share": float(1 - own.sum())}


def by_start(mean: np.ndarray, sd: np.ndarray, starts: np.ndarray, card: np.ndarray) -> list:
    """Cardholders at theta0's centre by window start age: the spread of one
    household's posterior mean across households and the median posterior
    sd, for beta and delta."""
    rows = []
    for lo, hi in START_GROUPS:
        s = (starts >= lo) & (starts <= hi) & (card == 1)
        if s.sum() < 30:
            continue
        rows.append({"start": [lo, hi], "n": int(s.sum()),
                     "mean_sd_across": mean[s, :2].std(0).tolist(),
                     "median_posterior_sd": np.median(sd[s, :2], 0).tolist()})
    return rows


def population_check(args) -> None:
    """The population se's precision (households resampled within card type,
    paired across arms), where beta's variance comes from, and one
    household's spread by window start (annual_population_check.json)."""
    from scripts import laibson_comparison as lc

    pop = json.loads((args.root / POPULATION).read_text())
    cache = args.root / "population"
    card = np.load(cache / "sim_centre.npz")["card"]
    th0 = np.array(pop["theta0"])
    means, sds = {}, {}
    for a in args.arms:
        post = {t: np.load(cache / f"post_{a}_{t}.npz") for t in we.POINTS}
        means[a] = {t: p["mean"] for t, p in post.items()}
        sds[a] = post["centre"]["sd"]
        se = we.population_se(means[a], card, th0, we.N_POP, lc.CARD_SHARE)["se"]
        if not np.allclose(se, pop["arms"][a]["se"]):
            raise SystemExit(f"{a}: the cached scorings do not give {POPULATION}'s se")
    rng = np.random.default_rng(args.seed)
    i1, i0 = np.flatnonzero(card == 1), np.flatnonzero(card == 0)
    boot = {a: [] for a in args.arms}
    for _ in range(args.n_boot):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        for a in args.arms:
            m = {t: v[idx] for t, v in means[a].items()}
            boot[a].append(we.population_se(m, card[idx], th0, we.N_POP, lc.CARD_SHARE)["se"])
    boot = {a: np.array(v) for a, v in boot.items()}
    res = {"n_boot": args.n_boot, "seed": args.seed, "arms": {}, "ratios": {}}
    for a in args.arms:
        st = {t: lc.stat(v, card, lc.CARD_SHARE) for t, v in means[a].items()}
        J = lc.jacobian(np.array([st[f"plus{k}"] for k in range(4)]),
                        np.array([st[f"minus{k}"] for k in range(4)]), lc.STEP)
        w = np.where(card == 1, lc.CARD_SHARE / (card == 1).sum(),
                     (1 - lc.CARD_SHARE) / (card == 0).sum())
        d = means[a]["centre"] - w @ means[a]["centre"]
        v_stat = (d * w[:, None]).T @ d / we.N_POP
        starts = we.pop_starts(pop["arms"][a]["window"], len(card))
        res["arms"][a] = {"se": pop["arms"][a]["se"],
                          "se_ci90": np.percentile(boot[a], [5, 95], axis=0).T.tolist(),
                          "beta_through": beta_through(J, v_stat),
                          "by_start": by_start(means[a]["centre"], sds[a], starts, card)}
    for an, bi in SAME_COUNT + SAME_YEARS:
        if an in boot and bi in boot:
            r = boot[an] / boot[bi]
            res["ratios"][f"{an}/{bi}"] = {
                "ratio": (np.array(pop["arms"][an]["se"]) / pop["arms"][bi]["se"]).tolist(),
                "ci90": np.percentile(r, [5, 95], axis=0).T.tolist()}
    out = args.root / "annual_population_check.json"
    out.write_text(json.dumps(res, indent=1))
    population_report(res)
    print(f"\nwrote {out}")


def population_report(res: dict) -> None:
    print(f"\npopulation se, with a 90% interval from {res['n_boot']} resamplings of households")
    for a, r in res["arms"].items():
        print(f"{a}: " + "  ".join(f"{p} {s:.4f} [{lo:.4f}, {hi:.4f}]" for p, s, (lo, hi)
                                   in zip(we.NAMES, r["se"], r["se_ci90"], strict=True)))
    print("\nse ratios, paired resamplings: ratio [90% interval]")
    for k, r in res["ratios"].items():
        print(f"{k}: " + "  ".join(f"{p} {v:.2f} [{lo:.2f}, {hi:.2f}]" for p, v, (lo, hi)
                                   in zip(we.NAMES, r["ratio"], r["ci90"], strict=True)))
    print("\nbeta's variance through J^-1: each statistic's own share (beta, delta, crra, "
          "R_gamma), and the covariances'")
    for a, r in res["arms"].items():
        b = r["beta_through"]
        print(f"{a}: J^-1 beta row {np.round(b['j_inv_beta_row'], 2).tolist()}; shares "
              f"{np.round(b['own_share'], 2).tolist()}, covariances {b['cross_share']:.2f}")
    print("\ncardholders at theta0 by window start: spread of posterior means across "
          "households / median posterior sd, beta and delta")
    for a, r in res["arms"].items():
        print(f"{a}: " + ";  ".join(
            f"{g['start'][0]}-{g['start'][1]} (n {g['n']}): β {g['mean_sd_across'][0]:.3f} / "
            f"{g['median_posterior_sd'][0]:.3f}, δ {g['mean_sd_across'][1]:.4f} / "
            f"{g['median_posterior_sd'][1]:.4f}" for g in r["by_start"]))


# --------------------------------------------------------------------- figures

def figures(args) -> None:
    """Figures 112 and 115."""
    res = json.loads((args.root / "annual_waves.json").read_text())
    precision(res, args.fig_dir / "112_annual_waves_precision.png")
    one_household(args, args.fig_dir / "115_annual_waves_one_household.png")


def precision(res: dict, path: Path) -> None:
    """112: share of prior variance removed, pairs ordered and the
    population se (÷ w07's) against waves and against years read; the
    biennial arms as a line, the annual ones as squares."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    a = res["arms"]
    bi = [x for x in ARMS if x in a and x not in ANNUAL]
    bi.sort(key=lambda x: we.parse_arm(x)[0])
    an = [x for x in ANNUAL if x in a]
    base = np.array(a["w07"]["population"]["se"])
    rows = (("share of prior variance removed (held-out)",
             lambda r, j: r["heldout"]["contraction"][j]),
            ("pairs ordered at p ≥ 0.9 (SBC)",
             lambda r, j: r["sbc"]["pairs_box"][we.NAMES[j]]["ordered_p90"]),
            ("population se ÷ w07's (409 households sharing θ₀)",
             lambda r, j: r["population"]["se"][j] / base[j]))
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    for i, (ylab, f) in enumerate(rows):
        for k, (xlab, xf) in enumerate((("waves", lambda x: we.parse_arm(x)[0]),
                                        ("years read", years_read))):
            ax = axs[i, k]
            for j in range(len(we.NAMES)):
                ax.plot([xf(x) for x in bi], [f(a[x], j) for x in bi], "o-",
                        color=PARAM_COLOR[j], lw=1.5)
                ax.plot([xf(x) for x in an], [f(a[x], j) for x in an], "s--",
                        color=PARAM_COLOR[j], lw=1, ms=8, mfc="white")
            ax.set(xlabel=xlab, ylabel=ylab)
            ax.axvline(7 if k == 0 else years_read("w07"), color="grey", ls=":", lw=1)
            ax.grid(ls=":", lw=0.5)
    handles = [Line2D([], [], color=c, lw=2, label=p) for c, p in zip(PARAM_COLOR, we.NAMES,
                                                                        strict=True)]
    handles += [Line2D([], [], color="k", marker="o", label="biennial (w07–w20)"),
                Line2D([], [], color="k", marker="s", ls="--", mfc="white",
                       label="annual (a14, a20)")]
    axs[0, 0].legend(handles=handles, fontsize=8)
    fig.suptitle("RESULTS 65: precision against waves (left) and against calendar years read "
                 "(right), biennial and annual waves; dotted: PSID's seven biennial waves",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


#: the household's noise by wave length: 2, figure 105's; 1, figure 115's own
HOUSEHOLD_SEEDS = {2: we.POP_SEED + 105, 1: we.POP_SEED + 115}


def household(shards) -> tuple[np.ndarray, int, dict]:
    """Figure 105's household (held-out, nearest theta0): its truth, draw id,
    and raw 20-wave windows from ``start_low(wy)`` for each wave length."""
    from hh_npe.data.windows import build_windowed
    from scripts import optionA as oa

    theta_all, _c, _tr, held_f, _p, _cfg = oa.load_shards(shards)
    th0 = we.theta0()
    first = held_f[0]
    d = np.load(first.path)
    lo, hi = int(d["lo"]), int(d["hi"])
    cand = theta_all[first.offset + lo: first.offset + hi]
    pick = int(np.argmin((((cand - th0) / (oa.BOX.high - oa.BOX.low)) ** 2).sum(1)))
    raw, truth = {}, None
    for wy in HOUSEHOLD_SEEDS:
        th, x, pid = build_windowed([first.path], theta_all[first.offset:], k=1, seed=0,
                                    fixed_start=we.start_low(wy), n_waves=20, wave_years=wy,
                                    start_low=we.start_low(wy), start_high=we.start_low(wy),
                                    flow_lag=oa.flow_lag_of([first.path]))
        row = int(np.flatnonzero(pid.numpy() == lo + pick)[0])
        t = th[row].numpy().astype(float)
        if truth is not None and not np.allclose(t, truth):
            raise SystemExit("the biennial and annual windows are not one household")
        truth = t
        raw[wy] = x[row:row + 1]
    return truth, lo + pick, raw


def household_draws(run: Path, label: str, raw: dict, device) -> np.ndarray:
    """One ensemble's resampled posterior draws for the household's first
    waves, with the ensemble's own training noise (seeded by wave length, so
    every arm of one noise sees the same noisy panel)."""
    from scripts import household_resolution as hr
    from scripts import laibson_comparison as lc
    from scripts import optionA as oa
    from scripts.plot_optionA2_households import weighted_draws
    from scripts.sbc_optionA2 import Members

    n, wy = we.parse_arm(label)
    members = Members(we.members_of(run), device)
    spec = oa.spec_of(members.cfg)
    x = oa.add_obs_noise(raw[wy], spec.obs_noise, seed=HOUSEHOLD_SEEDS[wy])
    xt, _f = oa.transform(x[:, :n], masked=spec.masked)
    inv, lw = oa._weighting(members.proposal, spec)
    torch.manual_seed(0)
    dr, lwt = weighted_draws(members.nets, xt, lc.N_DRAWS, spec.flow_box(), inv, lw)
    return hr._resample(dr, np.exp(lwt - lwt.max()), 6000, seed=0)


def print_ranks(name: str, draws: np.ndarray, truth: np.ndarray) -> None:
    print(f"{name}: the truth's rank in the posterior " + "  ".join(
        f"{p} {v:.3f}" for p, v in zip(we.NAMES, (draws < truth).mean(0), strict=True)),
        flush=True)


def one_household(args, path: Path) -> None:
    """115: figure 105's household (held-out, nearest theta0) under the six
    arms, 68/95%: its biennial panel from 24 with figure 105's noise, its
    annual panel from 25 with its own; shorter windows see the same first
    waves. Rows: 14 years (w07, a14) against 14 waves (w14); 20 years (w10,
    a20) against 20 waves (w20)."""
    import matplotlib.pyplot as plt

    from scripts import household_resolution as hr

    truth, draw, raw = household(args.shards)
    drawn = {}
    for label in args.arms:
        drawn[label] = household_draws(we.arm_dir(label, args.root), label, raw, args.device)
        print_ranks(label, drawn[label], truth)
    label_of = {"beta": "β", "delta": "δ", "crra": "ρ", "R_gamma": "R_γ"}
    groups = [[x for x in ("w07", "a14", "w14") if x in drawn],
              [x for x in ("w10", "a20", "w20") if x in drawn]]
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    for r_i, grp in enumerate(groups):
        for axi, (i, j) in zip(axs[r_i], ((0, 2), (0, 1), (1, 2)), strict=True):
            for x in grp:
                n, wy = we.parse_arm(x)
                ls_label = (f"{x}: {n} {'annual' if wy == 1 else 'biennial'} waves, "
                            f"{years_read(x)} years")
                hr._contour(axi, drawn[x], i, j, COLOR[x], ls_label, we.LIMITS)
            axi.plot(*truth[[i, j]], "k*", ms=11, label="truth")
            axi.set(xlabel=label_of[we.NAMES[i]], ylabel=label_of[we.NAMES[j]],
                    xlim=we.LIMITS[we.NAMES[i]], ylim=we.LIMITS[we.NAMES[j]])
            axi.grid(ls=":", lw=0.5)
        axs[r_i, 0].legend(fontsize=8)
    fig.suptitle(f"One held-out simulated household (draw {draw}, nearest θ₀; figure 105's) "
                 "under biennial and annual waves (RESULTS 65): 68% (solid) / 95% (dashed)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    np.savez(args.root / "annual_one_household.npz", truth=truth, draw=draw,
             **{f"draws_{x}": v for x, v in drawn.items()})
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("measures", measures), ("population_check", population_check),
                     ("figures", figures)):
        p = sub.add_parser(name, help=fn.__doc__.splitlines()[0] if fn.__doc__ else None)
        p.add_argument("--root", type=Path, default=we.ROOT)
        p.add_argument("--arms", nargs="+", default=list(ARMS))
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else None)
        p.set_defaults(fn=fn)
    sub.choices["population_check"].add_argument("--n_boot", type=int, default=1000)
    sub.choices["population_check"].add_argument("--seed", type=int, default=0)
    sub.choices["figures"].add_argument("--fig_dir", type=Path, default=we.FIG)
    sub.choices["figures"].add_argument(
        "--shards", type=Path, nargs="+",
        default=[Path("data/processed/couples_v2_dataset_shards"),
                 Path("data/processed/couples_v2_strip_dataset_shards"),
                 Path("data/processed/couples_v2_betastrip_dataset_shards")])
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
