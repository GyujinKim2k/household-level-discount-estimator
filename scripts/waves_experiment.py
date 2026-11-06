"""RESULTS 64: how precision grows with the number of waves, on simulated
households.

Four arms, each v2 on the box with the beta strip (RESULTS 63, §1 since
2026-11-05), trained on windows of 7, 10, 14 and 20 biennial waves. Every
window lies inside the same ages (the user's choice, 2026-11-05): it starts
at 24 at the earliest and its last wave starts by 63, so longer windows start
earlier (:func:`start_high`). Laibson et al.'s ``retireage = 64`` is used only
by their income-split robustness option, so the bound is about depth into
life (rising mortality in the backward induction, the falling income
profile), not a change of income regime.

Commands:

``start_high N``
    The latest start age for N waves (``run_waves.sh`` reads it from here).

``population``
    The population standard error at a fixed number of households (RESULTS
    61/63.10's statistic and Jacobian, on a simulated population): households
    at theta0 = v2_b010's representative theta, 85% cardholders, starts
    uniform over each arm's range. The Jacobian comes from central
    differences at theta0 with common random numbers: 9 simulations, each
    storing its annual panels, from which every arm cuts its own windows. The
    population shares one theta, so it understates PSID's standard error
    (couples differ, RESULTS 60); the ratio across waves is the result.

``measures``
    Per arm: calibration (held-out and SBC), one household's precision,
    telling households apart (SBC, truth known), low beta (RESULTS 63.4's
    table) and the population, with PSID couples' 7-wave standard error
    scaled to each length in two cases (:func:`psid_scale`); the underfitting
    check; the RESULTS tables.

``figures``
    104 (precision against waves), 105 (one simulated household under each
    arm, 68/95% contours), 106 (the population, 68/95% ellipses against
    Laibson et al.'s).

Usage::

    PYTHONPATH=. .venv/bin/python scripts/waves_experiment.py population
    PYTHONPATH=. .venv/bin/python scripts/waves_experiment.py measures
    PYTHONPATH=. .venv/bin/python scripts/waves_experiment.py figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("outputs/waves")
WAVES = (7, 10, 14, 20)
START_LOW = 24
#: the last wave starts by 63, so its two years are 63-64 (RESULTS 64.0)
LAST_WAVE_START = 63
WAVE_YEARS = 2
#: v2_b010's representative theta (RESULTS 63.10)
REFERENCE = Path("outputs/beta_strip/laibson_comparison_b010_refresh.json")
#: PSID couples' 7-wave populations: the arm and its representative theta's se
PSID_ARM = "v2_b010"
N_POP = 409                 # PSID couples
POP_SEED = 64_000_000       # window starts and noise of the simulated population
BETA_BINS = ((0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))
POINTS = ["centre"] + [f"{s}{k}" for k in range(4) for s in ("plus", "minus")]
NAMES = ("beta", "delta", "crra", "R_gamma")
LIMITS = {"beta": (0.10, 1.00), "delta": (0.85, 1.00), "crra": (0.50, 5.00)}
COLOR = {7: "#3B75AF", 10: "#519E3E", 14: "#EF8636", 20: "#C53A32"}
FIG = Path("figures")


def start_high(n_waves: int) -> int:
    """The latest start age whose last wave starts by ``LAST_WAVE_START``."""
    h = LAST_WAVE_START - WAVE_YEARS * (n_waves - 1)
    if h < START_LOW:
        raise ValueError(f"{n_waves} waves do not fit between {START_LOW} and "
                         f"{LAST_WAVE_START + WAVE_YEARS - 1}")
    return h


def arm_dir(n: int, root: Path = ROOT) -> Path:
    return root / f"w{n:02d}"


def members_of(run: Path) -> list[Path]:
    return sorted(Path(run).glob("s[0-9]*"))


def check_window(window: dict, n: int) -> None:
    """Refuse an arm whose members were not trained on RESULTS 64's window."""
    want = {"n_waves": n, "start_low": START_LOW, "start_high": start_high(n),
            "wave_years": WAVE_YEARS}
    got = {k: window.get(k) for k in want}
    if got != want:
        raise SystemExit(f"{n}-wave arm: members' window {got}, expected {want}")


# ------------------------------------------------------------------ population

def theta0() -> np.ndarray:
    return np.array(json.loads(REFERENCE.read_text())["arms"][PSID_ARM]["representative_theta"])


def points(th0: np.ndarray) -> dict:
    """theta0 and its central-difference stencil in (beta, log(1 - delta),
    rho, R_gamma), with laibson_comparison's steps."""
    from scripts import laibson_comparison as lc

    u0 = lc.to_u(th0)
    out = {"centre": np.asarray(th0, float)}
    for k in range(4):
        for sign, tag in ((1, "plus"), (-1, "minus")):
            out[f"{tag}{k}"] = lc.from_u(u0 + sign * lc.STEP[k] * np.eye(4)[k])
    return out


def sim_at(tag: str, theta: np.ndarray, cache: Path) -> dict:
    """``laibson_comparison.simulate`` with its panels, cached."""
    from scripts import laibson_comparison as lc

    f = cache / f"sim_{tag}.npz"
    if f.exists():
        d = dict(np.load(f))
        if not np.allclose(d["theta"], theta):
            raise SystemExit(f"{f} was simulated at {d['theta']}, not {theta}")
        return d
    d = lc.simulate(theta, return_panel=True)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(f, **d)
    print(f"simulated {tag} at {np.round(theta, 5).tolist()}: {len(d['ids'])} households",
          flush=True)
    return d


def cut(sim: dict, window: dict, flow_lag: int) -> np.ndarray:
    """Each simulated household's window: start uniform on the arm's range,
    the same starts at every point (common random numbers)."""
    from hh_npe.data.waves import FEATURES_TWOASSET_AGE
    from hh_npe.data.windows import cut_windows

    panel = {k.removeprefix("panel_"): v for k, v in sim.items() if k.startswith("panel_")}
    m = len(sim["ids"])
    starts = np.random.default_rng(POP_SEED + window["n_waves"]).integers(
        window["start_low"], window["start_high"] + 1, m)
    _t, x, ids = cut_windows(panel, np.zeros((m, 1)), np.arange(m), starts[:, None],
                             window["n_waves"], window["wave_years"], FEATURES_TWOASSET_AGE,
                             flow_lag=flow_lag)
    if len(ids) != m:
        raise SystemExit("a simulated household is missing from its window")
    return x[np.argsort(ids)].astype(np.float32)


def score(members, spec, x: np.ndarray) -> dict:
    """Posterior means and sds with the arm's training noise, seeded the same
    at every point (as ``laibson_comparison.Arm.score``)."""
    from scripts import laibson_comparison as lc
    from scripts import optionA as oa
    from scripts.sbc_optionA2 import posterior

    xt = oa.add_obs_noise(torch.from_numpy(x), spec.obs_noise, seed=POP_SEED + 77)
    xt, _f = oa.transform(xt, masked=spec.masked)
    r = posterior(members, xt, None, spec, lc.N_DRAWS)
    return {"mean": r["mean"], "sd": r["sd"]}


def population_se(means: dict, card: np.ndarray, th0: np.ndarray, n_pop: int = N_POP,
                  share: float = 0.85) -> dict:
    """The representative theta's covariance for ``n_pop`` households at
    theta0: J by central differences of the mean posterior mean (card types
    at ``share``), and the variance of one household's posterior mean in that
    card mix, divided by ``n_pop``; then V_u = J^-1 V J^-T, mapped to theta
    (delta = 1 - exp(u))."""
    from scripts import laibson_comparison as lc

    st = {t: lc.stat(m, card, share) for t, m in means.items()}
    J = lc.jacobian(np.array([st[f"plus{k}"] for k in range(4)]),
                    np.array([st[f"minus{k}"] for k in range(4)]), lc.STEP)
    m0 = means["centre"]
    w = np.where(card == 1, share / (card == 1).sum(), (1 - share) / (card == 0).sum())
    d = m0 - w @ m0
    v_stat = (d * w[:, None]).T @ d / n_pop
    j_inv = np.linalg.inv(J)
    v_u = j_inv @ v_stat @ j_inv.T
    D = np.diag([1.0, -(1.0 - th0[1]), 1.0, 1.0])
    v_th = D @ v_u @ D.T
    return {"jacobian": J.tolist(),
            "stat_sd_one_household": np.sqrt(np.diag(v_stat * n_pop)).tolist(),
            "cov": v_th.tolist(), "se": np.sqrt(np.diag(v_th)).tolist()}


def psid_scale(v_psid, v_hom, ratio) -> dict:
    """PSID couples' 7-wave covariance of the representative theta, scaled to
    n waves in two cases. ``v_hom`` is the same arm's covariance on households
    sharing theta0, ``ratio`` the homogeneous se at n waves over 7 waves'.

    The remainder v_psid - v_hom is what households sharing theta0 lack:
    mostly couples' differences (RESULTS 60), which, to first order, enter the
    representative theta as their own spread in theta, and more waves do not
    change that spread.
      (a) the remainder shrinks like the noise: D v_psid D (the launch plan's
          projection);
      (b) the remainder is held: (v_psid - v_hom) + D v_hom D;
    with D = diag(ratio). Both are v_psid at ratio 1."""
    v_psid, v_hom = np.asarray(v_psid, float), np.asarray(v_hom, float)
    D = np.diag(np.asarray(ratio, float))
    rem = v_psid - v_hom
    return {"remainder": rem, "a": D @ v_psid @ D, "b": rem + D @ v_hom @ D}


def population(args) -> None:
    from scripts import laibson_comparison as lc
    from scripts import optionA as oa
    from scripts.sbc_optionA2 import Members

    th0 = theta0()
    pts = points(th0)
    cache = args.root / "population"
    cache.mkdir(parents=True, exist_ok=True)
    sims = {t: sim_at(t, th, args.sim_cache) for t, th in pts.items()}
    card = sims["centre"]["card"]
    for s in sims.values():
        if not np.array_equal(s["card"], card):
            raise SystemExit("simulated households are not aligned across points")
    flow_lag = json.loads((lc.MAIN / "solver_config.json").read_text())["flow_lag"]
    runs = {f"w{n:02d}": arm_dir(n, args.root) for n in args.waves}
    if args.with_psid_arm:
        runs[PSID_ARM] = Path(lc.ARMS[PSID_ARM][0])
    res = {"theta0": th0.tolist(), "n_households": N_POP, "card_share": lc.CARD_SHARE,
           "n_simulated": int(len(card)), "arms": {}}
    for name, run in runs.items():
        members = Members(members_of(run), args.device)
        spec = oa.spec_of(members.cfg)
        win = members.window
        if name != PSID_ARM:
            check_window(win, int(name[1:]))
        means = {}
        for t in POINTS:
            f = cache / f"post_{name}_{t}.npz"
            if f.exists():
                means[t] = np.load(f)["mean"]
                continue
            r = score(members, spec, cut(sims[t], win, flow_lag))
            np.savez(f, **r)
            means[t] = r["mean"]
            print(f"scored {name} at {t}", flush=True)
        res["arms"][name] = {"window": win, **population_se(means, card, th0, N_POP,
                                                             lc.CARD_SHARE)}
        se = res["arms"][name]["se"]
        print(f"{name}: se " + "  ".join(f"{n} {v:.4f}" for n, v in zip(NAMES, se, strict=True)),
              flush=True)
    out = args.root / "population.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


# -------------------------------------------------------------------- measures

def low_beta(flat: dict, truth: np.ndarray) -> dict:
    """RESULTS 63.4's table: households by true beta."""
    m, sd, lo, hi = (flat[k][:, 0] for k in ("mean", "sd", "lo", "hi"))
    t = truth[:, 0]
    rows = []
    for a, b in BETA_BINS:
        s = (t >= a) & (t < b) if b < 1.0 else (t >= a) & (t <= b)
        rows.append({"lo": a, "hi": b, "n": int(s.sum()), "mean_of_means": float(m[s].mean()),
                     "median_posterior_sd": float(np.median(sd[s])),
                     "coverage_90": float(((lo[s] <= t[s]) & (t[s] <= hi[s])).mean()),
                     "p95_below_0.3": float((hi[s] < 0.3).mean())})
    below = t < 0.3
    return {"bins": rows, "corr_below_0.3": float(np.corrcoef(m[below], t[below])[0, 1]),
            "corr_all": float(np.corrcoef(m, t)[0, 1])}


def sbc_measures(run: Path) -> dict:
    from scripts import household_resolution as hr

    sim = hr.sbc_draws(run / "sbc" / "sbc16.npz", hr.SBC_SETS["v2_b010"])
    flat = {k: sim[k].reshape(-1, 4) for k in ("mean", "sd", "lo", "hi")}
    truth = np.repeat(sim["truth"], hr.PER_DRAW, 0)
    return {"n_households": int(len(truth)),
            "median_posterior_sd": np.median(flat["sd"], 0).tolist(),
            "coverage_90": ((flat["lo"] <= truth) & (truth <= flat["hi"])).mean(0).tolist(),
            "simulated": hr.simulated(sim),
            "interval_vs_range": hr.interval_vs_range(flat),
            "pairs_box": hr.pairs(**flat, truth=truth),
            "pairs_same_theta": hr._same_theta_pairs(sim),
            "low_beta": low_beta(flat, truth),
            "outside_95": outside_95(np.load(run / "sbc" / "sbc16.npz")["u"])}


def outside_95(u: np.ndarray) -> dict:
    """Shares of SBC households with at least one, and at least two, of the
    four parameters outside their central 95% interval (rank u below 0.025
    or above 0.975); 0.185 and 0.014 if calibrated and independent."""
    k = ((u < 0.025) | (u > 0.975)).sum(1)
    return {"at_least_1": float((k >= 1).mean()), "at_least_2": float((k >= 2).mean())}


def underfit(run: Path, tail: int = 5) -> dict:
    """Per member: the last epochs' slope of training and validation loss
    (``optionA.py train``'s training_summary.json). Training still falling
    while validation has stopped is the early-stopping point: the network
    fits its training households better than new ones, so the training data
    bind, not its capacity. A network short of capacity would end with both
    losses flat and no gap between them."""
    out = {}
    for d in members_of(run):
        f = d / "training_summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        tr, va = np.array(s["training_loss"]), np.array(s["validation_loss"])
        k = min(tail, len(tr) - 1)
        out[d.name] = {"epochs": int(len(tr)),
                       "train_slope_last": float((tr[-1] - tr[-1 - k]) / k) if k > 0 else None,
                       "val_slope_last": float((va[-1] - va[-1 - k]) / k) if k > 0 else None,
                       "best_val": float(va.min()) if len(va) else None,
                       "train_minus_val_end": float(tr[-1] - va[-1]) if len(va) else None,
                       "minutes": float(np.sum(s["epoch_durations_sec"]) / 60)}
    return out


def psid_check(root: Path) -> dict:
    """PSID couples and all 889 under the 7-wave arm on the matched age range
    (start 24-51) against v2_b010 (start 24-45): the same seven waves, so
    they should agree."""
    from scripts import laibson_comparison as lc

    out = {}
    for grp, ref in (("couples", lc.ARMS[PSID_ARM][1]), ("all", lc.ARMS[PSID_ARM][2])):
        new = root / f"psid_w07_{grp}"
        if not (new / "posterior_uncorrected.npz").exists():
            continue
        a = np.load(Path(ref) / "posterior_uncorrected.npz")
        b = np.load(new / "posterior_uncorrected.npz")
        ok = np.isfinite(a["mean"]).all(1) & np.isfinite(b["mean"]).all(1)
        ma, mb = a["mean"][ok], b["mean"][ok]
        out[grp] = {"n": int(ok.sum()), "median_v2_b010": np.median(ma, 0).tolist(),
                    "median_w07": np.median(mb, 0).tolist(),
                    "corr_of_means": [float(np.corrcoef(ma[:, j], mb[:, j])[0, 1])
                                      for j in range(4)],
                    "median_sd_v2_b010": np.median(a["sd"][ok], 0).tolist(),
                    "median_sd_w07": np.median(b["sd"][ok], 0).tolist()}
    return out


def measures(args) -> None:
    res = {"waves": list(args.waves), "arms": {}}
    pop = json.loads((args.root / "population.json").read_text())
    for n in args.waves:
        run = arm_dir(n, args.root)
        held = json.loads((run / "heldout_check.json").read_text())["ensemble"]
        sbc = json.loads((run / "sbc" / "sbc16.json").read_text())
        res["arms"][n] = {
            "start_high": start_high(n),
            "heldout": {"log_q": held["log_q_weighted"],
                        **{k: [held[p][k] for p in NAMES]
                           for k in ("cov90", "contraction", "corr", "mae_over_prior_sd")}},
            "sbc_coverage_90": [sbc["sbc"]["coverage_90"][p]["rate"] for p in NAMES],
            "sbc": sbc_measures(run),
            "population": pop["arms"][f"w{n:02d}"],
            "underfit": underfit(run)}
    if PSID_ARM in pop["arms"]:
        res["psid_arm_population"] = pop["arms"][PSID_ARM]
    res["psid_check"] = psid_check(args.root)
    lai = json.loads(REFERENCE.read_text())
    res["laibson_se"] = lai["laibson_se"]
    res["psid_se_7_waves"] = lai["arms"][PSID_ARM]["representative_se"]
    if "psid_arm_population" in res:
        ref = lai["arms"][PSID_ARM]
        se = np.array(ref["representative_se"])
        v_psid = np.outer(se, se) * np.array(ref["representative_corr"])
        v_hom = np.array(res["psid_arm_population"]["cov"])
        base = np.array(res["arms"][args.waves[0]]["population"]["se"])
        for n in args.waves:
            s = psid_scale(v_psid, v_hom, np.array(res["arms"][n]["population"]["se"]) / base)
            res["arms"][n]["psid_scale"] = {
                "cov_a": s["a"].tolist(), "cov_b": s["b"].tolist(),
                "se_a": np.sqrt(np.diag(s["a"])).tolist(),
                "se_b": np.sqrt(np.diag(s["b"])).tolist()}
        rem = s["remainder"]
        res["psid_remainder"] = {"cov_psid": v_psid.tolist(), "cov": rem.tolist(),
                                 "se": np.sqrt(np.diag(rem)).tolist(),
                                 "share_of_psid_variance": (np.diag(rem) / se**2).tolist(),
                                 "eigenvalues": np.linalg.eigvalsh(rem).tolist()}
    out = args.root / "waves_experiment.json"
    out.write_text(json.dumps(res, indent=1))
    report(res)
    print(f"\nwrote {out}")


def _info(share_removed: float) -> float:
    """Posterior precision from the data in units of the prior's, if both
    were Gaussian: 1 - posterior var / prior var = s gives s / (1 - s)."""
    return share_removed / (1 - share_removed)


def report(res: dict) -> None:
    """The RESULTS tables, as markdown."""
    a = res["arms"]
    ns = list(a)
    hdr = "| | " + " | ".join(f"{n} waves" for n in ns) + " |\n|---|" + "---|" * len(ns)

    def row(label, f, fmt="{:.3f}"):
        vals = [f(a[n]) for n in ns]
        print(f"| {label} | " + " | ".join("–" if v is None else fmt.format(v) for v in vals)
              + " |")

    print("\ncalibration and one household\n" + hdr)
    print("| start ages | " + " | ".join(f"{START_LOW}–{a[n]['start_high']}" for n in ns) + " |")
    row("held-out log q", lambda r: r["heldout"]["log_q"])
    for j, p in enumerate(NAMES):
        print(f"| {p}: held-out / SBC 90% coverage | " + " | ".join(
            f"{a[n]['heldout']['cov90'][j]:.3f} / {a[n]['sbc_coverage_90'][j]:.3f}"
            for n in ns) + " |")
    for j, p in enumerate(NAMES):
        row(f"{p}: share of prior variance removed", lambda r, j=j: r["heldout"]["contraction"][j])
    for j, p in enumerate(NAMES):
        row(f"{p}: the data's information ÷ 7 waves' (Gaussian reading, s/(1-s))",
            lambda r, j=j: _info(r["heldout"]["contraction"][j])
            / _info(a[ns[0]]["heldout"]["contraction"][j]))
    for j, p in enumerate(NAMES):
        row(f"{p}: corr(posterior mean, truth), held-out", lambda r, j=j: r["heldout"]["corr"][j])
    for j, p in enumerate(NAMES):
        row(f"{p}: median posterior sd (SBC)", lambda r, j=j: r["sbc"]["median_posterior_sd"][j],
            "{:.4f}")
    print("\ntelling households apart (SBC households, truth known)\n" + hdr)
    for p in NAMES:
        row(f"{p}: one 90% interval ÷ 5–95% range of means",
            lambda r, p=p: r["sbc"]["interval_vs_range"][p]["ratio"])
    for p in NAMES:
        row(f"{p}: pairs ordered at p ≥ 0.9",
            lambda r, p=p: r["sbc"]["pairs_box"][p]["ordered_p90"])
    for p in NAMES:
        row(f"{p}: of which right", lambda r, p=p: r["sbc"]["pairs_box"][p]["right_when_ordered"])
    for p in NAMES:
        row(f"{p}: sharing θ, falsely ordered",
            lambda r, p=p: r["sbc"]["pairs_same_theta"][p]["ordered_p90"])
    row("households with ≥ 1 of 4 parameters outside their central 95%",
        lambda r: r["sbc"]["outside_95"]["at_least_1"])
    row("households with ≥ 2 of 4 parameters outside their central 95%",
        lambda r: r["sbc"]["outside_95"]["at_least_2"])
    print("\nlow beta, by true beta (SBC)\n" + hdr)
    for b, (lo, hi) in enumerate(BETA_BINS):
        cells = []
        for n in ns:
            r = a[n]["sbc"]["low_beta"]["bins"][b]
            cells.append(f"{r['mean_of_means']:.3f} / {r['coverage_90']:.3f} / "
                         f"{r['p95_below_0.3']:.3f}")
        print(f"| β {lo}–{hi}: mean posterior mean / coverage / 95th pct below 0.3 | "
              + " | ".join(cells) + " |")
    row("corr(mean, truth) below β 0.3", lambda r: r["sbc"]["low_beta"]["corr_below_0.3"])
    print("\nthe population: 409 households at theta0, sharing it\n" + hdr)
    ss = np.array(res["laibson_se"]["sampling only"])
    for j, p in enumerate(NAMES):
        row(f"{p}: se", lambda r, j=j: r["population"]["se"][j], "{:.4f}")
    base = np.array(a[ns[0]]["population"]["se"])
    for j, p in enumerate(NAMES[:3]):
        row(f"{p}: se ÷ {ns[0]} waves'", lambda r, j=j: r["population"]["se"][j] / base[j])
    if "psid_remainder" in res:
        print(f"\nPSID couples ({N_POP}) at n waves, from their 7-wave se: (a) the remainder "
              "shrinks like the noise, (b) it is held\n" + hdr)

        def pair(label, f, fmt):
            print(f"| {label} | " + " | ".join(
                f"{fmt.format(f(a[n], 'a'))} / {fmt.format(f(a[n], 'b'))}" for n in ns) + " |")

        for j, p in enumerate(NAMES[:3]):
            pair(f"{p}: se, (a) / (b)", lambda r, c, j=j: r["psid_scale"][f"se_{c}"][j],
                 "{:.4f}")
        for j, p in enumerate(NAMES[:3]):
            pair(f"{p}: couples to match their sampling-only se, (a) / (b)",
                 lambda r, c, j=j: N_POP * (r["psid_scale"][f"se_{c}"][j] / ss[j]) ** 2, "{:.0f}")
        rem = res["psid_remainder"]
        print("\nthe remainder (PSID couples' 7-wave covariance less households sharing "
              f"theta0, {PSID_ARM}): se " + "  ".join(
                  f"{p} {v:.4f}" for p, v in zip(NAMES, rem["se"], strict=True))
              + "; share of PSID's variance " + "  ".join(
                  f"{p} {v:.2f}" for p, v in zip(NAMES, rem["share_of_psid_variance"],
                                                 strict=True))
              + f"; smallest eigenvalue {min(rem['eigenvalues']):.2e}")
    print("\nunderfitting: each arm's 5 members (training_summary.json)\n" + hdr)
    uf = {n: list(a[n]["underfit"].values()) for n in ns}
    print("| epochs | " + " | ".join(
        f"{min(m['epochs'] for m in uf[n])}–{max(m['epochs'] for m in uf[n])}" for n in ns)
        + " |")
    for key, lab in (("train_slope_last", "training loss, slope over the last 5 epochs"),
                     ("val_slope_last", "validation loss, slope over the last 5 epochs"),
                     ("train_minus_val_end", "training − validation loss at the end"),
                     ("best_val", "best validation loss"),
                     ("minutes", "minutes per member")):
        print(f"| {lab} (mean) | " + " | ".join(
            f"{np.mean([m[key] for m in uf[n]]):.3f}" for n in ns) + " |")
    if "psid_arm_population" in res:
        print(f"\n{PSID_ARM} (7 waves, PSID's start ages 24-45) on the same households: se "
              + "  ".join(f"{p} {v:.4f}" for p, v in
                          zip(NAMES, res["psid_arm_population"]["se"], strict=True)))
    for grp, c in res.get("psid_check", {}).items():
        print(f"\nPSID {grp} ({c['n']}): 7 waves, start 24-51 (w07) against 24-45 (v2_b010)")
        for k in ("median_v2_b010", "median_w07", "corr_of_means", "median_sd_v2_b010",
                  "median_sd_w07"):
            print(f"  {k:20s}" + "".join(f"{v:10.4f}" for v in c[k]))


# --------------------------------------------------------------------- figures

def figures(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads((args.root / "waves_experiment.json").read_text())
    ns = [int(n) for n in res["arms"]]
    a = {int(n): v for n, v in res["arms"].items()}
    label = {"beta": r"$\beta$", "delta": r"$\delta$", "crra": r"$\rho$", "R_gamma": r"$R_\gamma$"}

    # 104: precision against waves
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    panels = [("share of prior variance removed (held-out)",
               lambda n, j: a[n]["heldout"]["contraction"][j], False),
              ("median posterior sd ÷ 7 waves' (SBC)",
               lambda n, j: a[n]["sbc"]["median_posterior_sd"][j]
               / a[ns[0]]["sbc"]["median_posterior_sd"][j], False),
              ("household pairs ordered at p ≥ 0.9 (SBC)",
               lambda n, j: a[n]["sbc"]["pairs_box"][NAMES[j]]["ordered_p90"], False)]
    for axi, (_title, f, _log) in zip(ax.ravel()[:3], panels, strict=True):
        for j, p in enumerate(NAMES):
            axi.plot(ns, [f(n, j) for n in ns], "o-", label=label[p])
    # (d) the population: PSID couples' 7-wave se scaled in the two cases
    # (psid_scale), and households sharing theta0
    from matplotlib.lines import Line2D

    axi = ax[1, 1]
    ss = res["laibson_se"]["sampling only"]
    for j, _p in enumerate(NAMES[:3]):
        c = f"C{j}"
        axi.plot(ns, [a[n]["population"]["se"][j] / ss[j] for n in ns], "o:", color=c,
                 lw=1, ms=3)
        if "psid_scale" in a[ns[0]]:
            sa = [a[n]["psid_scale"]["se_a"][j] / ss[j] for n in ns]
            sb = [a[n]["psid_scale"]["se_b"][j] / ss[j] for n in ns]
            axi.fill_between(ns, sa, sb, color=c, alpha=0.15, lw=0)
            axi.plot(ns, sb, "o-", color=c)
            axi.plot(ns, sa, "o--", color=c, ms=4)
    axi.set_yscale("log")
    axi.axhline(1, color="k", lw=0.6)
    axi.legend(handles=[Line2D([], [], color=f"C{j}", label=label[p])
                        for j, p in enumerate(NAMES[:3])]
               + [Line2D([], [], color="0.3", ls="-", label="PSID couples, remainder held (b)"),
                  Line2D([], [], color="0.3", ls="--",
                         label="PSID couples, remainder shrinks (a)"),
                  Line2D([], [], color="0.3", ls=":", label="409 households sharing θ₀")],
               fontsize=7)
    panels.append(("population se ÷ Laibson et al.'s sampling-only se\n"
                   "(409 households, from PSID couples' 7-wave se)", None, True))
    for axi, (title, _f, _log) in zip(ax.ravel(), panels, strict=True):
        axi.axvline(7, color="0.4", ls=":", lw=1)
        axi.text(7.2, axi.get_ylim()[1], "PSID: 7 waves", va="top", fontsize=8, color="0.3")
        axi.set(xlabel="waves (biennial, inside ages 24–64)", title=title, xticks=ns)
        axi.grid(ls=":", lw=0.5)
        if axi is not ax[1, 1]:
            axi.legend(fontsize=8)
    fig.suptitle("The waves experiment (RESULTS 64): v2 on the box with the β strip, every "
                 "window inside ages 24–64", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "104_waves_precision.png", dpi=130)
    plt.close(fig)

    one_household(args, ns, label)

    # 106: the population, 68/95% ellipses (delta method) against theirs
    lai = json.loads(REFERENCE.read_text())
    th_l = np.array(lai["theta_laibson"])
    th0 = np.array(json.loads((args.root / "population.json").read_text())["theta0"])
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    t = np.linspace(0, 2 * np.pi, 200)

    def ellipse(axi, c, cov, i, j, color, ls_pair, lab):
        w, v = np.linalg.eigh(np.asarray(cov)[np.ix_([i, j], [i, j])])
        for q, ls in zip((2.30, 5.99), ls_pair, strict=True):
            e = c[[i, j]][:, None] + v @ (np.sqrt(np.maximum(w, 0) * q)[:, None]
                                          * np.vstack([np.cos(t), np.sin(t)]))
            axi.plot(e[0], e[1], color=color, ls=ls, lw=1)
        axi.plot([], [], color=color, ls=ls_pair[0], label=lab)

    for axi, (i, j) in zip(axs, ((0, 2), (0, 1), (1, 2)), strict=True):
        for key, ls in (("reported", ("-", "--")), ("sampling only", (":", ":"))):
            ellipse(axi, th_l, lai["laibson_cov"][key], i, j, "k", ls,
                    f"Laibson et al., {key}")
        axi.plot(*th_l[[i, j]], "k*", ms=10)
        for n in ns:
            ellipse(axi, th0, a[n]["population"]["cov"], i, j, COLOR[n], ("-", "--"),
                    f"{n} waves, 409 households sharing θ₀")
        if "psid_remainder" in res:
            ellipse(axi, th0, res["psid_remainder"]["cov_psid"], i, j, "#7B3294", ("-", "--"),
                    f"PSID couples, {ns[0]} waves (RESULTS 63.10)")
            ellipse(axi, th0, a[ns[-1]]["psid_scale"]["cov_b"], i, j, "#8C510A", ("-.", "-."),
                    f"PSID couples at {ns[-1]} waves, remainder held (b)")
        axi.plot(*th0[[i, j]], "o", color="0.2", ms=4)
        axi.set(xlabel=label[NAMES[i]], ylabel=label[NAMES[j]])
        axi.grid(ls=":", lw=0.5)
    axs[0].legend(fontsize=7)
    fig.suptitle("The population by waves (RESULTS 64): 68% / 95% regions of the "
                 "representative θ for 409 households sharing θ₀ = v2's PSID estimate, and "
                 "for PSID's 409 couples, who differ, at 7 waves and projected to "
                 f"{ns[-1]} (case b)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "106_waves_population.png", dpi=130)
    plt.close(fig)
    print(f"wrote figures 104-106 in {args.fig_dir}")


def one_household(args, ns, label) -> None:
    """105: one held-out simulated household, the same panel from age 24
    under every arm, 68/95% contours. The noise is drawn once on the
    20-wave window, so shorter windows see the same first waves."""
    import matplotlib.pyplot as plt

    from hh_npe.data.windows import build_windowed
    from scripts import household_resolution as hr
    from scripts import laibson_comparison as lc
    from scripts import optionA as oa
    from scripts.plot_optionA2_households import weighted_draws
    from scripts.sbc_optionA2 import Members

    theta_all, _c, _tr, held_f, _p, _cfg = oa.load_shards(args.shards)
    th0 = theta0()
    scale = (oa.BOX.high - oa.BOX.low)
    first = held_f[0]
    d = np.load(first.path)
    lo, hi = int(d["lo"]), int(d["hi"])
    cand = theta_all[first.offset + lo: first.offset + hi]
    pick = int(np.argmin((((cand - th0) / scale) ** 2).sum(1)))
    longest = max(ns)
    th, x, pid = build_windowed([first.path], theta_all[first.offset:], k=1, seed=0,
                                fixed_start=START_LOW, n_waves=longest, wave_years=WAVE_YEARS,
                                start_low=START_LOW, start_high=START_LOW,
                                flow_lag=oa.flow_lag_of([first.path]))
    row = int(np.flatnonzero(pid.numpy() == lo + pick)[0])
    truth = th[row].numpy().astype(float)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    drawn = {}
    for n in ns:
        members = Members(members_of(arm_dir(n, args.root)), args.device)
        spec = oa.spec_of(members.cfg)
        if n == ns[0]:
            raw = oa.add_obs_noise(x[row:row + 1], spec.obs_noise, seed=POP_SEED + 105)
        xt, _f = oa.transform(raw[:, :n], masked=spec.masked)
        inv, lw = oa._weighting(members.proposal, spec)
        torch.manual_seed(0)
        dr, lwt = weighted_draws(members.nets, xt, lc.N_DRAWS, spec.flow_box(), inv, lw)
        drawn[n] = hr._resample(dr, np.exp(lwt - lwt.max()), 6000, seed=0)
        print(f"{n} waves: the truth's rank in the posterior "
              + "  ".join(f"{p} {v:.3f}" for p, v in
                          zip(NAMES, (drawn[n] < truth).mean(0), strict=True)), flush=True)
    for axi, (i, j) in zip(axs, ((0, 2), (0, 1), (1, 2)), strict=True):
        for n in ns:
            hr._contour(axi, drawn[n], i, j, COLOR[n], f"{n} waves", LIMITS)
        axi.plot(*truth[[i, j]], "k*", ms=11, label="truth")
        axi.set(xlabel=label[NAMES[i]], ylabel=label[NAMES[j]],
                xlim=LIMITS[NAMES[i]], ylim=LIMITS[NAMES[j]])
        axi.grid(ls=":", lw=0.5)
    axs[0].legend(fontsize=8)
    fig.suptitle(f"One held-out simulated household (draw {lo + pick}, nearest θ₀), its panel "
                 f"from age {START_LOW}, under 7–{longest} waves (RESULTS 64): 68% (solid) / "
                 "95% (dashed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "105_waves_one_household.png", dpi=130)
    plt.close(fig)
    np.savez(args.root / "one_household.npz", truth=truth, draw=lo + pick,
             **{f"draws_{n}": drawn[n] for n in ns})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start_high", help="The latest start age for N waves.")
    s.add_argument("n_waves", type=int)
    for name, fn in (("population", population), ("measures", measures), ("figures", figures)):
        p = sub.add_parser(name, help=fn.__doc__.splitlines()[0] if fn.__doc__ else None)
        p.add_argument("--root", type=Path, default=ROOT)
        p.add_argument("--waves", type=int, nargs="+", default=list(WAVES))
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else None)
        p.set_defaults(fn=fn)
    sub.choices["population"].add_argument(
        "--sim_cache", type=Path, default=ROOT / "population",
        help="where the 9 simulations are cached (shared with the smoke test)")
    sub.choices["population"].add_argument(
        "--with_psid_arm", action="store_true",
        help=f"also score {PSID_ARM} (7 waves, PSID's start ages), to scale the projection")
    sub.choices["figures"].add_argument("--fig_dir", type=Path, default=FIG)
    sub.choices["figures"].add_argument(
        "--shards", type=Path, nargs="+",
        default=[Path("data/processed/couples_v2_dataset_shards"),
                 Path("data/processed/couples_v2_strip_dataset_shards"),
                 Path("data/processed/couples_v2_betastrip_dataset_shards")])
    args = ap.parse_args()
    if args.cmd == "start_high":
        print(start_high(args.n_waves))
        return
    args.fn(args)


if __name__ == "__main__":
    main()
