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
import textwrap
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
#: Newton steps and the bootstrap stop just inside the A box (RESULTS 61);
#: :func:`box_bounds` moves them to another arm's box.
BOX_LO = np.array([0.301, 0.851, 0.51, 1.0215])
BOX_HI = np.array([0.999, 0.99999, 4.99, 1.0745])
#: RESULTS 61's arms, v2 trained with no observation noise (RESULTS 62), and
#: RESULTS 63's arms on the box with the beta strip below 0.3
ARMS = {**hr.ARMS, "none": ("outputs/optionA2v2_noise_none",
                            "outputs/psid_optionA2v2_noise_none_couples",
                            "outputs/psid_optionA2v2_noise_none_all"),
        "v2_b010": ("outputs/optionA2v2_b010", "outputs/psid_optionA2v2_b010_couples",
                    "outputs/psid_optionA2v2_b010_all"),
        **{f"{a}_b010": (f"outputs/optionA2v2_b010_noise_{a}",
                         f"outputs/psid_optionA2v2_b010_noise_{a}_couples",
                         f"outputs/psid_optionA2v2_b010_noise_{a}_all")
           for a in ("typical", "gross")}}
ARM_COLOR = {**hr.ARM_COLOR, "none": "#4A9A5B", "unknown": "#D08A1E"}
ARM_COLOR |= {f"{a}_b010": c for a, c in list(ARM_COLOR.items())}


def box_bounds(box) -> tuple[np.ndarray, np.ndarray]:
    """``BOX_LO`` / ``BOX_HI`` for an arm's prior box: the same margins inside
    its beta and R_gamma ends (RESULTS 63's box reaches down to beta 0.1). On
    the A box they are exactly the constants, so cached simulations of
    RESULTS 61-62's clipped Newton points still match."""
    lo, hi = BOX_LO.copy(), BOX_HI.copy()
    lo[0], hi[0] = round(box.beta_low + 0.001, 6), round(box.beta_high - 0.001, 6)
    lo[3], hi[3] = round(box.rgamma_low + 0.0005, 6), round(box.rgamma_high - 0.0005, 6)
    return lo, hi


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
        run = Path(ARMS[name][0])
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

    def ranks(self, sim: dict) -> np.ndarray:
        """Each simulated household's weighted rank of the theta it was
        simulated at (the same windows and noise as :meth:`score`)."""
        from scripts import optionA as oa
        from scripts.sbc_optionA2 import posterior

        x = oa.add_obs_noise(torch.from_numpy(sim["x"]), self.spec.obs_noise, seed=SEED + 77)
        x, _f = oa.transform(x, masked=self.spec.masked)
        truth = np.tile(sim["theta"], (len(x), 1))
        return posterior(self.members, x, truth, self.spec, N_DRAWS)["u"]


def coverage_at(arm: Arm, name: str, sim: dict) -> dict:
    """How often households simulated at one theta have it inside their 90%
    interval, and on which side the misses fall (overall and by card type).
    Calibration is a prior average, so one theta need not give 0.90; a
    one-sided miss rate shows where the posterior leans there."""
    f = OUT / f"ranks_{arm.name}_{name}.npz"
    if f.exists():
        u = np.load(f)["u"]
    else:
        u = arm.ranks(sim)
        np.savez(f, u=u)
    ok = np.isfinite(u).all(1)
    inside = (u >= 0.05) & (u <= 0.95)
    card = sim["card"]
    return {"coverage_90": inside[ok].mean(0).tolist(),
            "truth_below_5th": (u[ok] < 0.05).mean(0).tolist(),
            "truth_above_95th": (u[ok] > 0.95).mean(0).tolist(),
            "coverage_90_by_card": {c: inside[ok & (card == i)].mean(0).tolist()
                                    for i, c in ((0, "no card"), (1, "card"))}}


def cached_post(arm: Arm, name: str, sim: dict) -> dict:
    f = OUT / f"post_{arm.name}_{name}.npz"
    if f.exists():
        return dict(np.load(f))
    r = arm.score(sim)
    np.savez(f, **r, ids=sim["ids"], card=sim["card"])
    print(f"scored {name} with {arm.name}", flush=True)
    return r | {"ids": sim["ids"], "card": sim["card"]}


def one_couple(sd: np.ndarray, se: dict, bounds=(BOX_LO, BOX_HI)) -> dict:
    """One couple's posterior sd (median over couples) against their standard
    errors, and how many couples carry as much information as their whole
    estimate: their precision over the median couple's data precision (its
    posterior precision less the uniform prior's), under their own
    assumption of one shared theta."""
    prior_sd = (bounds[1][:3] - bounds[0][:3]) / np.sqrt(12)
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


def bounded_step(v, J, resid, se, lo, hi) -> np.ndarray:
    """The Newton step solving ``J d = resid``, except that a component at a
    bound the step would push past is held there; the free components then
    solve the misfit's least squares in PSID standard errors (RESULTS 63.7:
    noise-unknown's income noise sits at its box's top)."""
    d = np.linalg.solve(J, resid)
    held = ((v <= lo + 1e-12) & (d < 0)) | ((v >= hi - 1e-12) & (d > 0))
    if held.any():
        w = 1 / np.asarray(se, float)
        d = np.zeros_like(d)
        d[~held] = np.linalg.lstsq(J[:, ~held] * w[:, None], resid * w, rcond=None)[0]
    return d


def refine(v, post, target, se, lo, hi, step, score_at, fd_at, stat_of, n_iter: int):
    """RESULTS 63.7: Newton from ``v`` with the Jacobian recomputed by central
    differences at every accepted point, rather than held at their theta.

    ``score_at(v, tag)`` simulates and scores a point; ``fd_at(v, r)`` returns
    the central-difference scorings around it ({"plus": [...], "minus": [...]},
    one per unknown); ``stat_of`` turns a scoring into the statistic. A step
    that does not lower the misfit's sum of squares (in PSID standard errors)
    is halved, twice at most; if none does, the refinement stops there. The
    point returned always carries its own Jacobian, for the last unsimulated
    step and the bootstrap. Returns (v, post, fd, J, path)."""
    def loss(p):
        return float(np.sum(((target - stat_of(p)) / se) ** 2))

    def jac(fd):
        return jacobian(np.array([stat_of(p) for p in fd["plus"]]),
                        np.array([stat_of(p) for p in fd["minus"]]), step)

    fd = fd_at(v, 0)
    J, cur, path = jac(fd), loss(post), []
    for r in range(1, n_iter + 1):
        resid = target - stat_of(post)
        if (np.abs(resid) < 0.25 * se).all():
            break
        d = bounded_step(v, J, resid, se, lo, hi)
        for h, tag in ((1.0, ""), (0.5, "h"), (0.25, "q")):
            v_new = np.clip(v + h * d, lo, hi)
            p_new = score_at(v_new, f"r{r}{tag}")
            new = loss(p_new)
            path.append({"v": v_new.tolist(), "stat": stat_of(p_new).tolist(),
                         "step_fraction": h, "accepted": new < cur})
            if new < cur:
                break
        if new >= cur:
            break
        v, post, cur = v_new, p_new, new
        fd = fd_at(v, r)
        J = jac(fd)
    return v, post, fd, J, path


def at_edge(v, lo, hi) -> np.ndarray:
    """Which components of an estimate sit on a box edge: clipped there, or
    held there by :func:`bounded_step` (comparing with the plain Newton step
    flags every component once one is held)."""
    return (v <= lo + 1e-12) | (v >= hi - 1e-12)


def stencil_centre(v, step, lo, hi) -> np.ndarray:
    """Where the central differences around ``v`` are taken: moved just enough
    that every point of the stencil stays inside the box."""
    return np.clip(v, lo + step, hi - step)


def boot_theta(u_hat, posts_fd, post_hat, psid_means, share, step, rng, n_boot=N_BOOT,
               step_fn=newton):
    """Bootstrap of the representative theta, linearised at ``u_hat``:
    resample PSID households, and simulated households (the same indices at
    every point, so the common random numbers stay paired), recompute the
    Jacobian and the statistic, and take one Newton step (``step_fn``)."""
    card = post_hat["card"]
    by_card = [np.flatnonzero(card == c) for c in (0, 1)]
    out = np.empty((n_boot, 4))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(i, len(i)) for i in by_card])
        ip = rng.integers(0, len(psid_means), len(psid_means))
        plus = np.array([stat(p["mean"], card, share, idx) for p in posts_fd["plus"]])
        minus = np.array([stat(p["mean"], card, share, idx) for p in posts_fd["minus"]])
        J = jacobian(plus, minus, step)
        out[b] = step_fn(u_hat, J, psid_means[ip].mean(0), stat(post_hat["mean"], card, share, idx))
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
    v2_med = np.median(hr.psid(ARMS["v2"][1])["mean"], 0)
    sims["v2_median"] = cached_sim("v2_median", v2_med)

    res = {"laibson": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in lai.items() if k not in ("cov", "se")},
           "laibson_se": {k: v.tolist() for k, v in lai["se"].items()},
           "laibson_cov": {k: v.tolist() for k, v in lai["cov"].items()},
           "theta_laibson": th_l.tolist(), "card_share": CARD_SHARE, "arms": {}}
    rng = np.random.default_rng(0)
    for name in args.arms:
        arm = Arm(name, args.device)
        b_lo, b_hi = box_bounds(arm.spec.box)
        posts = {k: cached_post(arm, k, s) for k, s in sims.items()}
        card = posts["laibson"]["card"]
        for p in posts.values():
            if not np.array_equal(p["card"], card):
                raise SystemExit("simulated households are not aligned across points")
        psid = hr.psid(ARMS[name][1])
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
        lo_u = np.minimum(to_u(b_lo), to_u(b_hi))
        hi_u = np.maximum(to_u(b_lo), to_u(b_hi))
        for it in range(1, MAX_NEWTON + 1):
            if (np.abs(target - at_u) < 0.25 * se_target).all():
                break
            u = np.clip(newton(u, J, target, at_u), lo_u, hi_u)
            th = from_u(u)
            s = cached_sim(f"ii_{name}_{it}", th)
            post_hat = cached_post(arm, f"ii_{name}_{it}", s)
            at_u = stat(post_hat["mean"], card, CARD_SHARE)
            path.append({"theta": th.tolist(), "stat": at_u.tolist()})
        refined, step_fn = None, newton
        if args.refresh:
            # RESULTS 63.7: far from their theta the fixed Jacobian fails
            def score_at(v, tag, name=name, arm=arm):
                return cached_post(arm, f"ii_{name}_{tag}",
                                   cached_sim(f"ii_{name}_{tag}", from_u(v)))

            def fd_at(v, r, lo_u=lo_u, hi_u=hi_u, score_at=score_at):
                c = stencil_centre(v, STEP, lo_u, hi_u)
                return {sign: [score_at(c + f * STEP[k] * np.eye(4)[k], f"j{r}_{sign}{k}")
                               for k in range(4)]
                        for sign, f in (("plus", 1), ("minus", -1))}

            u, post_hat, fd, J, rpath = refine(
                u, post_hat, target, se_target, lo_u, hi_u, STEP, score_at, fd_at,
                lambda p, card=card: stat(p["mean"], card, CARD_SHARE), args.refresh)
            at_u = stat(post_hat["mean"], card, CARD_SHARE)
            refined = {"n_max": args.refresh, "path": [
                {**q, "theta": from_u(np.array(q.pop("v"))).tolist()} for q in rpath],
                "jacobian": J.tolist(),
                "jacobian_centre": from_u(stencil_centre(u, STEP, lo_u, hi_u)).tolist()}

            def step_fn(v, J_, t, a, lo_u=lo_u, hi_u=hi_u, se=se_target):
                return v + bounded_step(v, J_, t - a, se, lo_u, hi_u)
        # the last simulated point's misfit, and one more Newton step from it
        # without a simulation: the bootstrap is centred there too
        resid_se = (target - at_u) / se_target
        u_raw = newton(u, J, target, at_u)
        u_hat = np.clip(step_fn(u, J, target, at_u), lo_u, hi_u)
        boot_raw = from_u(boot_theta(u, fd, post_hat, pm, CARD_SHARE, STEP, rng,
                                     step_fn=step_fn))
        # the networks never saw theta outside the box: past an edge the
        # linear step is extrapolation, so the estimate stops at the edge and
        # the bootstrap's share beyond it is reported
        beyond = ((boot_raw < b_lo) | (boot_raw > b_hi)).mean(0)
        boot = np.clip(boot_raw, b_lo, b_hi)
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
            "one_couple": one_couple(psid["sd"], lai["se"], (b_lo, b_hi)),
            "box_floor_beta": float(arm.spec.box.beta_low),
            "at_laibson": {
                "mean_of_means": stat(pl, card, CARD_SHARE).tolist(),
                "mean_of_means_card_share_0.5": stat(pl, card, 0.5).tolist(),
                "by_card": {c: pl[card == i].mean(0).tolist() for i, c in ((0, "no card"),
                                                                           (1, "card"))},
                "median_of_means": np.median(pl, 0).tolist(),
                "sd_of_means": pl.std(0, ddof=1).tolist(),
                "median_posterior_sd": np.median(posts["laibson"]["sd"], 0).tolist(),
                **coverage_at(arm, "laibson", sims["laibson"]),
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
            "at_box_edge": at_edge(u_hat, lo_u, hi_u).tolist(),
            "boot_share_beyond_box": beyond.tolist(),
            "representative_se": boot.std(0, ddof=1).tolist(),
            "representative_ci90": np.percentile(boot, [5, 95], axis=0).T.tolist(),
            "representative_corr": np.corrcoef(boot.T).tolist(),
            "representative_card_share_0.5": th5.tolist(),
            "boot": boot.tolist(),
            **({"refined": refined} if refined else {}),
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
                ("  90% intervals containing their theta", a["at_laibson"]["coverage_90"]),
                ("  their theta below the 5th percentile", a["at_laibson"]["truth_below_5th"]),
                ("  their theta above the 95th percentile", a["at_laibson"]["truth_above_95th"]),
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


#: a last simulated point further than this from PSID couples' statistic, in
#: PSID standard errors, is not a fit: no region and no bars
FIT_TOL = 3.0


def fit_note(arm: dict) -> str:
    """Why an arm's representative theta is not a fit with a region, or ''."""
    miss = float(np.max(np.abs(arm["last_simulated_misfit_over_psid_se"])))
    if miss > FIT_TOL:
        return f"no fit (misses PSID by {miss:.0f} se)"
    return "β at the box floor" if any(arm["at_box_edge"]) else ""


def plot(res: dict, path: Path, title: str | None = None) -> None:
    """(a-c) Their estimate with its 68/95% region (reported covariance),
    against each arm's representative theta (bootstrap, 68/95%); PSID
    couples' median estimates and what the network reports at their theta.
    An arm with no fit shows its last simulated point, hollow, and no region.
    (d) Standard errors against theirs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hh_npe.evaluation.plots import _hpd_levels, _kde_grid

    # RESULTS 63: room for fits below beta 0.3 when an arm's box reaches there
    floor = min([0.3] + [a.get("box_floor_beta", 0.3) for a in res["arms"].values()])
    lim = {0: (min(0.25, floor - 0.05), 0.85), 1: (0.955, 1.0), 2: (0.6, 3.6)}
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
            color = ARM_COLOR[name]
            b = np.array(arm["boot"])
            lo, hi = (lim[i][0], lim[j][0]), (lim[i][1], lim[j][1])
            note = fit_note(arm)
            if note.startswith("no fit"):
                a.plot(*np.array(arm["last_simulated_theta"])[[i, j]], "o", mfc="none",
                       color=color)
                a.plot([], [], "o", mfc="none", color=color,
                       label=f"{name}: last simulated point, {note}")
            else:
                if np.ptp(b[:, i]) > 0 and np.ptp(b[:, j]) > 0:   # clipped flat at an edge
                    X, Y, Z = _kde_grid(b[:, i], b[:, j], lo, hi)
                    a.contour(X, Y, Z, levels=_hpd_levels(Z), colors=color,
                              linestyles=["--", "-"])
                a.plot(*np.array(arm["representative_theta"])[[i, j]], "o", color=color)
                a.plot([], [], color=color, label=f"{name}: representative θ (bootstrap)"
                                                  + (f", {note}" if note else ""))
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
        # bootstrap is clipped; one that misses it by more than FIT_TOL is no
        # fit at all: no bars for either
        note = fit_note(arm)
        v = [np.nan] * 4 if note else arm["representative_se"]
        groups.append((f"{name}: representative θ" + (f" ({note}: no bars)" if note else ""),
                       v, ARM_COLOR[name]))
    for name, arm in res["arms"].items():
        groups.append((f"{name}: one couple (median sd)", arm["psid_median_posterior_sd"],
                       ARM_COLOR[name]))
    xs = np.arange(3)
    width = 0.8 / len(groups)
    # RESULTS 63.10: converged fits near delta 1 have se under 0.05 of theirs
    ratios = np.array([np.array(v[:3], float) for _, v, _ in groups]) / np.array(se["reported"])
    bottom = min(0.05, 0.5 * np.nanmin(ratios))
    for g, (lab, v, color) in enumerate(groups):
        ratio = np.array(v[:3]) / np.array(se["reported"])
        a.bar(xs - 0.4 + (g + 0.5) * width, ratio, width, color=color,
              hatch="//" if "one couple" in lab else None, alpha=0.9 if "one" not in lab else 0.5,
              label=lab)
    a.axhline(1, color="k", lw=0.6)
    a.set_yscale("log")
    a.set_xticks(xs, [hr.LABEL[n] for n in NAMES[:3]])
    a.set(ylabel="standard error ÷ Laibson et al.'s reported", ylim=(bottom, 400))
    a.legend(fontsize=6, ncol=2, loc="upper center")
    a.grid(ls=":", lw=0.5, axis="y")
    # a long title wraps rather than running off the figure's edges
    fig.suptitle(textwrap.fill(title or "Laibson et al.'s MSM point against PSID couples "
                               "(RESULTS 61): a representative θ matched on the network's "
                               "estimates", 130), fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*", default=list(hr.ARMS))
    ap.add_argument("--arm", nargs=3, action="append", default=[],
                    metavar=("NAME", "RUN", "PSID_COUPLES"),
                    help="an arm's ensemble and PSID couples' posteriors, replacing or "
                         "adding to ARMS, and run (the chain's smoke test)")
    ap.add_argument("--cache", type=Path, default=OUT,
                    help="where simulations and scorings are cached")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else None)
    ap.add_argument("--figure", type=Path, default=FIG)
    ap.add_argument("--out", type=Path, default=OUT / "laibson_comparison.json")
    ap.add_argument("--title", default=None, help="the figure's title")
    ap.add_argument("--refresh", type=int, default=0,
                    help="after the fixed-Jacobian steps, up to this many Newton steps with "
                         "the Jacobian recomputed at each point (RESULTS 63.7); 0: RESULTS 61's "
                         "method")
    args = ap.parse_args()
    for name, run_dir, couples in args.arm:
        ARMS[name] = (run_dir, couples, None)
        ARM_COLOR.setdefault(name, ARM_COLOR.get(name.removesuffix("_b010"), "0.4"))
        if name not in args.arms:
            args.arms.append(name)
    if bad := [a for a in args.arms if a not in ARMS]:
        ap.error(f"unknown arms {bad}; known {list(ARMS)}")
    OUT = args.cache
    res = run(args)
    report(res)
    args.out.write_text(json.dumps(res, indent=2))
    plot(res, args.figure, title=args.title)
    print(f"\nwrote {args.out} and {args.figure}")


if __name__ == "__main__":
    main()
