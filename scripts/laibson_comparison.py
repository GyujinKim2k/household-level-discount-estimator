"""Our household estimates against Laibson et al.'s MSM point (RESULTS 61).

1. **Their precision, from their own matrices.** Table 3 (comphs) saves the
   moment Jacobian, the first-stage Jacobian and covariances, and the moments'
   covariance and sample sizes. Their four standard-error variants (sampling
   only, + first stage, + simulation, all) are recomputed here and must match
   the reported ones; that also gives their full covariance, which their table
   does not print.

2. **Households simulated at their theta.** With v2's generation (solver,
   settings, seed pool, flow lag), 2,048 households per card type at
   (beta, delta, rho) = their estimate and R_gamma 1.05, one window each,
   starting where a PSID couple's first wave does. Each arm's ensemble scores
   them with its own training noise. What the network reports for households
   that truly have their theta is what PSID couples' estimates must be read
   against: posterior means are pulled toward the prior box's centre.

3. **A representative theta from PSID couples (indirect inference).** The
   statistic is the mean of the posterior means over households, card types
   mixed at ``CARD_SHARE``. Its Jacobian in (beta, log(1 - delta), rho,
   R_gamma) comes from central differences at their theta, with common random
   numbers: every point simulates the same households (seed pool draws,
   shocks, start ages, noise). Newton steps from their theta, each checked by
   a simulation at the new point (at most three), approach the theta whose
   simulated households the network estimates as it estimates PSID couples;
   the estimate is one further, unsimulated Newton step from the last
   simulated point, whose misfit is reported. That is a
   representative-agent estimate on PSID panels, the same kind of object as
   theirs. Its uncertainty is a bootstrap over PSID couples and over the
   simulated households (which also resamples the Jacobian).

Stages are cached under ``OUT``; rerunning skips finished simulations and
scorings.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/laibson_comparison.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts import household_resolution as hr

PKG = Path("replication-package-LLMRT/LifecycleSimulation")
MAIN = Path("data/processed/couples_v2_dataset_shards")
POOL = Path("data/processed/seed_pool_couples_income.npz")
PSID_X = Path("data/processed/psid_x_comphs_couples.pt")
OUT = Path("outputs/laibson_comparison")
FIG = Path("figures/81_laibson_comparison.png")
NAMES = hr.NAMES
PER_CARD = 2048
SEED = 61_000_000           # household streams: no training run, SBC set or earlier check
CARD_SHARE = 0.85           # PSID couples: 325 of 409 seen borrowing (RESULTS 45.4)
N_DRAWS = 4000
R_GAMMA = 1.05              # their calibration
#: central-difference half-steps in (beta, log(1 - delta), rho, R_gamma)
STEP = np.array([0.03, 0.15, 0.25, 0.005])
MAX_NEWTON = 3
N_BOOT = 1000
BOX_LO = np.array([0.301, 0.851, 0.51, 1.0215])
BOX_HI = np.array([0.999, 0.99999, 4.99, 1.0745])


def to_u(theta: np.ndarray) -> np.ndarray:
    u = np.array(theta, dtype=float)
    u[..., 1] = np.log1p(-u[..., 1])
    return u


def from_u(u: np.ndarray) -> np.ndarray:
    t = np.array(u, dtype=float)
    t[..., 1] = -np.expm1(t[..., 1])
    return t


# ------------------------------------------------------------------- theirs

def laibson() -> dict:
    """Their estimate, standard errors and covariances, recomputed from the
    replication package's table 3 run and checked against the reported ones
    (``MSMfunction.m``: A = (G'WG)^-1 G'W, W = diag(VCV)^-1)."""
    import scipy.io as sio
    from scipy.linalg import block_diag

    t3 = sio.loadmat(PKG / "output/simulations/table3/EDFbatch_table3.mat",
                     squeeze_me=True, struct_as_record=False)
    m, fs = t3["MSMout"], t3["fs_params"]
    V = sio.loadmat(PKG / "input/comphs/VCV_secondstage.mat", squeeze_me=True)["VCV_secondstage"]
    est = sio.loadmat(PKG / "input/comphs/est_secondstage.mat", squeeze_me=True)["est_secondstage"]
    G, Gc = np.asarray(m.G_theta), np.asarray(m.G_chi)
    W = np.linalg.inv(np.diag(np.diag(V)))
    A = np.linalg.inv(G.T @ W @ G) @ G.T @ W
    oc = block_diag(*(np.atleast_2d(getattr(fs, k))
                      for k in ("VCV_demographics", "VCV_income", "VCV_creditlim")))
    n = est[:, 2]
    sim = np.maximum.outer(n, n) / 10_000 * V          # setup.pop = 10,000
    first = Gc @ oc @ Gc.T
    cov = {"sampling only": A @ V @ A.T, "first stage": A @ (V + first) @ A.T,
           "simulation": A @ (V + sim) @ A.T, "reported": A @ (V + sim + first) @ A.T}
    rep = m.optprefs_stderr
    for name, field in (("sampling only", "none"), ("first stage", "fs"),
                        ("simulation", "sim"), ("reported", "full")):
        if not np.allclose(np.sqrt(np.diag(cov[name])), getattr(rep, field), rtol=1e-4):
            raise SystemExit(f"their {name} standard errors do not reproduce")
    return {"theta": np.asarray(m.optprefs, float), "cov": cov,
            "se": {k: np.sqrt(np.diag(c)) for k, c in cov.items()},
            "j_test": float(m.specTest), "j_df": int(len(n) - 3), "j_p": float(m.specPval),
            "n_obs": _households(n)}


def _households(n: np.ndarray) -> int:
    """Households behind the 16 moments: 4 moment types by 4 age groups, each
    type on the same households of an age group."""
    by_type = n.reshape(4, 4)
    if not (by_type == by_type[0]).all():
        raise SystemExit("their moments do not share sample sizes across types")
    return int(by_type[0].sum())


# -------------------------------------------------------------- simulation

def simulate(theta: np.ndarray) -> dict:
    """``PER_CARD`` households per card type at ``theta`` (beta, delta, rho,
    R_gamma), v2's generation; windows start where PSID couples' first waves
    do. The same seed at every theta: common random numbers."""
    from hh_npe.data.waves import FEATURES_TWOASSET_AGE
    from hh_npe.data.windows import add_age, cut_windows
    from hh_npe.simulator.dispatch import simulate_batch_twoasset_gpu
    from hh_npe.simulator.twoasset import GRIDS
    from scripts import optionA as oa
    from scripts.generate_dataset import _load_pool

    cfg = json.loads((MAIN / "solver_config.json").read_text())
    pool, digest = _load_pool(POOL)
    if digest != cfg["init_pool_sha256"]:
        raise SystemExit(f"{POOL}: not the seed pool v2 used")
    g = GRIDS[cfg["grid"]]
    for key in ("round_income", "matlab_infinities", "tie", "crra_centred"):
        if getattr(g, key) != cfg[key]:
            raise SystemExit(f"the solver's {key} is {getattr(g, key)!r}, v2 used {cfg[key]!r}")
    card = np.array([0, 1])
    _x, _a, panel = simulate_batch_twoasset_gpu(
        np.array([theta[:3], theta[:3]]), SEED, cfg["start_age"], cfg["n_waves"],
        cfg["wave_years"], cfg["grid"], cfg["theta_batch"], cfg["chunk"], return_panels=True,
        n_households=PER_CARD, card=card, r_gamma=np.array([theta[3], theta[3]]),
        init_pool=pool, flow_lag=cfg["flow_lag"])
    n = 2 * PER_CARD
    age = FEATURES_TWOASSET_AGE.index("age")
    ages = torch.load(PSID_X, weights_only=False)["x"].numpy()[:, 0, age].astype(int)
    # a window starting at s carries age s + 1 in its first wave
    starts = np.clip(np.random.default_rng(SEED).choice(ages, n) - 1,
                     oa.WINDOW["start_low"], oa.WINDOW["start_high"])
    _t, x, ids = cut_windows(add_age(panel), np.zeros((n, 1)), np.arange(n), starts[:, None],
                             oa.WINDOW["n_waves"], oa.WINDOW["wave_years"],
                             FEATURES_TWOASSET_AGE, flow_lag=cfg["flow_lag"])
    o = np.argsort(ids)
    return {"x": x[o].astype(np.float32), "ids": ids[o],
            "card": np.repeat(card, PER_CARD)[ids[o]], "theta": np.asarray(theta, float)}


def cached_sim(name: str, theta: np.ndarray) -> dict:
    f = OUT / f"sim_{name}.npz"
    if f.exists():
        d = dict(np.load(f))
        if not np.allclose(d["theta"], theta):
            raise SystemExit(f"{f} was simulated at {d['theta']}, not {theta}")
        return d
    d = simulate(theta)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(f, **d)
    print(f"simulated {name} at {np.round(theta, 5).tolist()}: {len(d['x'])} households",
          flush=True)
    return d


class Arm:
    """One arm's ensemble, scoring simulated windows with its own training
    noise (seeded the same for every point and arm)."""

    def __init__(self, name: str, device: str | None):
        from scripts import optionA as oa
        from scripts.sbc_optionA2 import Members

        self.name = name
        run = Path(hr.ARMS[name][0])
        self.members = Members(sorted(run.glob("s[0-9]*")), device)
        self.spec = oa.spec_of(self.members.cfg)
        lag = self.members.cfg.get("window", {}).get("flow_lag", 0)
        if lag != json.loads((MAIN / "solver_config.json").read_text())["flow_lag"]:
            raise SystemExit(f"{name}: members trained with flow_lag {lag}")

    def score(self, sim: dict) -> dict:
        from scripts import optionA as oa
        from scripts.sbc_optionA2 import posterior

        x = oa.add_obs_noise(torch.from_numpy(sim["x"]), self.spec.obs_noise, seed=SEED + 77)
        x, _f = oa.transform(x, masked=self.spec.masked)
        r = posterior(self.members, x, None, self.spec, N_DRAWS)
        return {"mean": r["mean"], "sd": r["sd"]}


def cached_post(arm: Arm, name: str, sim: dict) -> dict:
    f = OUT / f"post_{arm.name}_{name}.npz"
    if f.exists():
        return dict(np.load(f))
    r = arm.score(sim)
    np.savez(f, **r, ids=sim["ids"], card=sim["card"])
    print(f"scored {name} with {arm.name}", flush=True)
    return r | {"ids": sim["ids"], "card": sim["card"]}


def one_couple(sd: np.ndarray, se: dict) -> dict:
    """One couple's posterior sd (median over couples) against their standard
    errors, and how many couples carry as much information as their whole
    estimate: their precision over the median couple's data precision (its
    posterior precision less the uniform prior's), under their own
    assumption of one shared theta."""
    prior_sd = (BOX_HI[:3] - BOX_LO[:3]) / np.sqrt(12)
    med = np.median(sd[:, :3], 0)
    data = np.median(np.maximum(1 / sd[:, :3] ** 2 - 1 / prior_sd ** 2, 1e-12), 0)
    return {"median_posterior_sd": med.tolist(),
            **{f"ratio_to_{k.replace(' ', '_')}": (med / se[k]).tolist()
               for k in ("reported", "sampling only")},
            **{f"couples_equal_{k.replace(' ', '_')}": ((1 / se[k] ** 2) / data).tolist()
               for k in ("reported", "sampling only")}}


# ------------------------------------------------------- indirect inference

def stat(mean: np.ndarray, card: np.ndarray, share: float, idx=None) -> np.ndarray:
    """Mean posterior mean per parameter, card types mixed at ``share``
    cardholders; ``idx`` resamples households within each card type."""
    if idx is not None:
        mean, card = mean[idx], card[idx]
    return share * mean[card == 1].mean(0) + (1 - share) * mean[card == 0].mean(0)


def jacobian(plus: np.ndarray, minus: np.ndarray, step: np.ndarray) -> np.ndarray:
    """J[i, k] = d stat_i / d u_k from central differences; ``plus``/``minus``
    are (k, i)."""
    return ((plus - minus) / (2 * step[:, None])).T


def newton(u: np.ndarray, J: np.ndarray, target: np.ndarray, at_u: np.ndarray) -> np.ndarray:
    return u + np.linalg.solve(J, target - at_u)


def boot_theta(u_hat, posts_fd, post_hat, psid_means, share, step, rng, n_boot=N_BOOT):
    """Bootstrap of the representative theta, linearised at ``u_hat``:
    resample PSID households, and simulated households (the same indices at
    every point, so the common random numbers stay paired), recompute the
    Jacobian and the statistic, and take one Newton step."""
    card = post_hat["card"]
    by_card = [np.flatnonzero(card == c) for c in (0, 1)]
    out = np.empty((n_boot, 4))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(i, len(i)) for i in by_card])
        ip = rng.integers(0, len(psid_means), len(psid_means))
        plus = np.array([stat(p["mean"], card, share, idx) for p in posts_fd["plus"]])
        minus = np.array([stat(p["mean"], card, share, idx) for p in posts_fd["minus"]])
        J = jacobian(plus, minus, step)
        out[b] = newton(u_hat, J, psid_means[ip].mean(0), stat(post_hat["mean"], card, share, idx))
    return out


# ------------------------------------------------------------------ driver

def run(args) -> dict:
    lai = laibson()
    th_l = np.append(lai["theta"], R_GAMMA)
    u_l = to_u(th_l)
    sims = {"laibson": cached_sim("laibson", th_l)}
    for k in range(4):
        for sign, tag in ((1, "plus"), (-1, "minus")):
            sims[f"{tag}{k}"] = cached_sim(f"{tag}{k}", from_u(u_l + sign * STEP[k] * np.eye(4)[k]))
    v2_med = np.median(hr.psid(hr.ARMS["v2"][1])["mean"], 0)
    sims["v2_median"] = cached_sim("v2_median", v2_med)

    res = {"laibson": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in lai.items() if k not in ("cov", "se")},
           "laibson_se": {k: v.tolist() for k, v in lai["se"].items()},
           "laibson_cov": {k: v.tolist() for k, v in lai["cov"].items()},
           "theta_laibson": th_l.tolist(), "card_share": CARD_SHARE, "arms": {}}
    rng = np.random.default_rng(0)
    for name in args.arms:
        arm = Arm(name, args.device)
        posts = {k: cached_post(arm, k, s) for k, s in sims.items()}
        card = posts["laibson"]["card"]
        for p in posts.values():
            if not np.array_equal(p["card"], card):
                raise SystemExit("simulated households are not aligned across points")
        psid = hr.psid(hr.ARMS[name][1])
        pm = psid["mean"]
        target = pm.mean(0)
        se_target = pm.std(0, ddof=1) / np.sqrt(len(pm))
        fd = {"plus": [posts[f"plus{k}"] for k in range(4)],
              "minus": [posts[f"minus{k}"] for k in range(4)]}
        J = jacobian(np.array([stat(p["mean"], card, CARD_SHARE) for p in fd["plus"]]),
                     np.array([stat(p["mean"], card, CARD_SHARE) for p in fd["minus"]]), STEP)
        path = [{"theta": th_l.tolist(), "stat": stat(posts["laibson"]["mean"], card,
                                                      CARD_SHARE).tolist()}]
        u, at_u, post_hat = u_l, np.array(path[0]["stat"]), posts["laibson"]
        # log(1 - delta) reverses the order of delta's bounds
        lo_u = np.minimum(to_u(BOX_LO), to_u(BOX_HI))
        hi_u = np.maximum(to_u(BOX_LO), to_u(BOX_HI))
        for it in range(1, MAX_NEWTON + 1):
            if (np.abs(target - at_u) < 0.25 * se_target).all():
                break
            u = np.clip(newton(u, J, target, at_u), lo_u, hi_u)
            th = from_u(u)
            s = cached_sim(f"ii_{name}_{it}", th)
            post_hat = cached_post(arm, f"ii_{name}_{it}", s)
            at_u = stat(post_hat["mean"], card, CARD_SHARE)
            path.append({"theta": th.tolist(), "stat": at_u.tolist()})
        # the last simulated point's misfit, and one more Newton step from it
        # without a simulation: the bootstrap is centred there too
        resid_se = (target - at_u) / se_target
        u_raw = newton(u, J, target, at_u)
        u_hat = np.clip(u_raw, lo_u, hi_u)
        boot_raw = from_u(boot_theta(u, fd, post_hat, pm, CARD_SHARE, STEP, rng))
        # the networks never saw theta outside the box: past an edge the
        # linear step is extrapolation, so the estimate stops at the edge and
        # the bootstrap's share beyond it is reported
        beyond = ((boot_raw < BOX_LO) | (boot_raw > BOX_HI)).mean(0)
        boot = np.clip(boot_raw, BOX_LO, BOX_HI)
        # card share 0.5, linearised from the same point
        J5 = jacobian(np.array([stat(p["mean"], card, 0.5) for p in fd["plus"]]),
                      np.array([stat(p["mean"], card, 0.5) for p in fd["minus"]]), STEP)
        th5 = from_u(newton(u, J5, target, stat(post_hat["mean"], card, 0.5)))
        pl = posts["laibson"]["mean"]
        res["arms"][name] = {
            "psid_mean_of_means": target.tolist(), "psid_se": se_target.tolist(),
            "psid_median_of_means": np.median(pm, 0).tolist(),
            "psid_sd_of_means": pm.std(0, ddof=1).tolist(),
            "psid_median_posterior_sd": np.median(psid["sd"], 0).tolist(),
            "one_couple": one_couple(psid["sd"], lai["se"]),
            "at_laibson": {
                "mean_of_means": stat(pl, card, CARD_SHARE).tolist(),
                "mean_of_means_card_share_0.5": stat(pl, card, 0.5).tolist(),
                "by_card": {c: pl[card == i].mean(0).tolist() for i, c in ((0, "no card"),
                                                                           (1, "card"))},
                "median_of_means": np.median(pl, 0).tolist(),
                "sd_of_means": pl.std(0, ddof=1).tolist(),
                "median_posterior_sd": np.median(posts["laibson"]["sd"], 0).tolist(),
            },
            "at_v2_median": {"theta": v2_med.tolist(),
                             "mean_of_means": stat(posts["v2_median"]["mean"], card,
                                                   CARD_SHARE).tolist()},
            "jacobian": J.tolist(),
            "newton_path": path,
            "converged": bool((np.abs(resid_se) < 0.25).all()),
            "last_simulated_theta": from_u(u).tolist(),
            "last_simulated_misfit_over_psid_se": resid_se.tolist(),
            "representative_theta": from_u(u_hat).tolist(),
            "representative_unclipped": from_u(u_raw).tolist(),
            "at_box_edge": (u_raw != u_hat).tolist(),
            "boot_share_beyond_box": beyond.tolist(),
            "representative_se": boot.std(0, ddof=1).tolist(),
            "representative_ci90": np.percentile(boot, [5, 95], axis=0).T.tolist(),
            "representative_corr": np.corrcoef(boot.T).tolist(),
            "representative_card_share_0.5": th5.tolist(),
            "boot": boot.tolist(),
        }
        print(f"{name}: representative theta {np.round(from_u(u_hat), 4).tolist()} "
              f"se {np.round(boot.std(0, ddof=1), 4).tolist()}", flush=True)
    return res


def report(res: dict) -> None:
    def fmt(v, d=4):
        return " ".join(f"{x:9.{d}f}" for x in v)

    print(f"\n{'':44s}" + "".join(f"{n:>10s}" for n in NAMES))
    th = res["theta_laibson"]
    print(f"{'Laibson et al. theta (R_gamma calibrated)':44s} {fmt(th)}")
    for k, v in res["laibson_se"].items():
        print(f"{'  their se, ' + k:44s} {fmt(v)}")
    print(f"  their J test {res['laibson']['j_test']:.1f} on {res['laibson']['j_df']} df, "
          f"p = {res['laibson']['j_p']:.1e}; {res['laibson']['n_obs']} household observations")
    for name, a in res["arms"].items():
        print(f"\n=== {name} ===")
        rows = [("PSID couples: mean of posterior means", a["psid_mean_of_means"]),
                ("  its se", a["psid_se"]),
                ("  median of posterior means", a["psid_median_of_means"]),
                ("simulated at their theta: mean of means", a["at_laibson"]["mean_of_means"]),
                ("  card share 0.5", a["at_laibson"]["mean_of_means_card_share_0.5"]),
                ("  median of means", a["at_laibson"]["median_of_means"]),
                ("  sd of means (PSID's below)", a["at_laibson"]["sd_of_means"]),
                ("  PSID sd of means", a["psid_sd_of_means"]),
                ("simulated at v2's median estimate", a["at_v2_median"]["theta"]),
                ("  mean of means", a["at_v2_median"]["mean_of_means"]),
                ("last simulated Newton point", a["last_simulated_theta"]),
                ("  its misfit ÷ PSID's se", a["last_simulated_misfit_over_psid_se"]),
                ("representative theta (one more step)", a["representative_theta"]),
                ("  bootstrap se", a["representative_se"]),
                ("  90% low", [c[0] for c in a["representative_ci90"]]),
                ("  90% high", [c[1] for c in a["representative_ci90"]]),
                ("  unclipped (past a box edge: extrapolation)", a["representative_unclipped"]),
                ("  bootstrap share beyond the box", a["boot_share_beyond_box"]),
                ("  card share 0.5 (linearised)", a["representative_card_share_0.5"]),
                ("one couple: median posterior sd", a["one_couple"]["median_posterior_sd"]),
                ("  ÷ their reported se", a["one_couple"]["ratio_to_reported"]),
                ("  ÷ their sampling-only se", a["one_couple"]["ratio_to_sampling_only"]),
                ("  couples equal to their estimate, reported",
                 a["one_couple"]["couples_equal_reported"]),
                ("  same, sampling only", a["one_couple"]["couples_equal_sampling_only"])]
        for label, v in rows:
            print(f"{label:44s} {fmt(v, 2 if 'couples equal' in label or 'same,' in label else 4)}")
        print(f"  Newton steps {len(a['newton_path']) - 1}, converged {a['converged']}")


def plot(res: dict, path: Path) -> None:
    """(a-c) Their estimate with its 68/95% region (reported covariance),
    against each arm's representative theta (bootstrap, 68/95%); PSID
    couples' median estimates and what the network reports at their theta.
    (d) Standard errors against theirs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hh_npe.evaluation.plots import _hpd_levels, _kde_grid

    lim = {0: (0.25, 0.85), 1: (0.955, 1.0), 2: (0.6, 3.6)}
    th = np.array(res["theta_laibson"])
    cov = np.array(res["laibson_cov"]["reported"])
    cov_s = np.array(res["laibson_cov"]["sampling only"])
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    t = np.linspace(0, 2 * np.pi, 200)
    for a, (i, j) in zip(ax.ravel()[:3], ((0, 2), (0, 1), (1, 2)), strict=True):
        for c, ls, lab in ((cov, "-", "reported"), (cov_s, ":", "sampling only")):
            sub = c[np.ix_([i, j], [i, j])]
            w, v = np.linalg.eigh(sub)
            for q, ls2 in ((2.30, ls), (5.99, "--" if ls == "-" else ":")):
                e = th[[i, j]][:, None] + v @ (np.sqrt(w * q)[:, None] * np.vstack([np.cos(t),
                                                                                    np.sin(t)]))
                a.plot(e[0], e[1], color="k", ls=ls2, lw=1)
            a.plot([], [], color="k", ls=ls, label=f"Laibson et al., {lab} covariance")
        a.plot(*th[[i, j]], "k*", ms=12)
        for name, arm in res["arms"].items():
            color = hr.ARM_COLOR[name]
            b = np.array(arm["boot"])
            lo, hi = (lim[i][0], lim[j][0]), (lim[i][1], lim[j][1])
            X, Y, Z = _kde_grid(b[:, i], b[:, j], lo, hi)
            a.contour(X, Y, Z, levels=_hpd_levels(Z), colors=color, linestyles=["--", "-"])
            a.plot(*np.array(arm["representative_theta"])[[i, j]], "o", color=color)
            edge = ", β at the box floor" if arm["at_box_edge"][0] else ""
            a.plot([], [], color=color, label=f"{name}: representative θ (bootstrap){edge}")
            est = np.array(arm["at_laibson"]["mean_of_means"])
            a.annotate("", xy=est[[i, j]], xytext=th[[i, j]],
                       arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
            a.plot(*np.array(arm["psid_mean_of_means"])[[i, j]], "x", color=color, ms=8)
        a.set(xlabel=hr.LABEL[NAMES[i]], ylabel=hr.LABEL[NAMES[j]], xlim=lim[i], ylim=lim[j])
        a.grid(ls=":", lw=0.5)
    ax[0, 0].plot([], [], "kx", label="PSID couples' mean estimate (colour: arm)")
    ax[0, 0].plot([], [], color="0.3", marker=r"$\rightarrow$", ls="none", ms=12,
                  label="arrow: their θ → what the network reports for it")
    ax[0, 0].legend(fontsize=7, loc="upper right")
    ax[0, 0].set_title("68% (solid) / 95% (dashed)", fontsize=10)

    a = ax[1, 1]
    se = res["laibson_se"]
    groups = [("Laibson, reported", se["reported"], "0.3"),
              ("Laibson, sampling only", se["sampling only"], "0.65")]
    for name, arm in res["arms"].items():
        # a fit stopped at a box edge does not reach PSID's statistic, and its
        # bootstrap is clipped: no bars
        edge = any(arm["at_box_edge"])
        v = [np.nan] * 4 if edge else arm["representative_se"]
        groups.append((f"{name}: representative θ" + (" (stopped at a box edge: no bars)"
                                                      if edge else ""), v, hr.ARM_COLOR[name]))
    for name, arm in res["arms"].items():
        groups.append((f"{name}: one couple (median sd)", arm["psid_median_posterior_sd"],
                       hr.ARM_COLOR[name]))
    xs = np.arange(3)
    width = 0.8 / len(groups)
    for g, (lab, v, color) in enumerate(groups):
        ratio = np.array(v[:3]) / np.array(se["reported"])
        a.bar(xs - 0.4 + (g + 0.5) * width, ratio, width, color=color,
              hatch="//" if "one couple" in lab else None, alpha=0.9 if "one" not in lab else 0.5,
              label=lab)
    a.axhline(1, color="k", lw=0.6)
    a.set_yscale("log")
    a.set_xticks(xs, [hr.LABEL[n] for n in NAMES[:3]])
    a.set(ylabel="standard error ÷ Laibson et al.'s reported", ylim=(0.05, 40))
    a.legend(fontsize=6.5, ncol=2)
    a.grid(ls=":", lw=0.5, axis="y")
    fig.suptitle("Laibson et al.'s MSM point against PSID couples (RESULTS 61): a representative θ "
                 "matched on the network's estimates", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(hr.ARMS), choices=list(hr.ARMS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else None)
    ap.add_argument("--figure", type=Path, default=FIG)
    args = ap.parse_args()
    res = run(args)
    report(res)
    (OUT / "laibson_comparison.json").write_text(json.dumps(res, indent=2))
    plot(res, args.figure)
    print(f"\nwrote {OUT / 'laibson_comparison.json'} and {args.figure}")


if __name__ == "__main__":
    main()
