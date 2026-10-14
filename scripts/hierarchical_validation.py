"""Validation of the population layer on simulated couples (RESULTS 45).

The population fit (``hierarchical_population.py``) is read only through
these checks, each run through the same pipeline as PSID (draws, summary,
reference points, fit) on simulated households carrying A2n's observation
noise. With the summary likelihood, every simulated household here is one
its density model never saw: the SBC set's untrained half
(``hierarchical_summary.split``), or new simulations.

- **U, the uniform test.** The SBC set's untrained draws are uniform on the box,
  so their population is known. 5 populations of 512 (one household per draw,
  a different one each time) are fitted; the fitted spreads must match the
  realised ones.
- **V1, the null.** Every household shares one theta. ``null-sim`` simulates
  4,096 households per card type at each theta* (A2n's PSID couples' median
  posterior mean; Laibson et al.'s MSM point with R_gamma 1.05) with the
  training generation's solver, settings and seed pool, and cuts one window
  each, starting at an age drawn from PSID couples' first waves. ``validate``
  forms 10 populations of 409 per theta* and card mix (the generation's 50/50
  and 85% cardholders, since 325 of 409 PSID couples are seen borrowing). The
  fitted spreads are what the method returns when the truth is none: PSID's
  spread in a parameter is heterogeneity only above their 95th percentile.
- **V2, recovery at PSID's population.** ``v2-sim`` simulates 256 thetas
  from PSID's fitted g (R_gamma and card type per block of 16, as in
  generation), 16 households each, and forms 10 populations of 409. Fitted against
  realised spreads; gate G2 trusts a parameter's spread if its median
  relative bias is within 25% and the Hessian's 90% interval covers the
  realised value in at least 8 of 10.
- **V3, pooled households** (on V2): coverage of the true theta by the pooled
  90% intervals (gate G3: 0.85-0.95) against the unpooled; MSE of the pooled
  means relative to the unpooled; the variance ratio.
- **Diagnose** (RESULTS 46): why V2 fails, for a likelihood variant. Its
  training thetas in the corner where PSID's population sits (``CORNER``);
  on households it never saw (V2's, and the nulls' at one theta each), the
  PIT of ``s`` under ``p(s | true theta)`` in the corner and outside, and the
  predicted against the simulated mean of ``s``; and on V2 populations, the
  fit after ``rounds`` rounds and refitted on reference points drawn around
  the true population, with ``sum log m`` at the truth and at the fit on
  those points. A fit that stops short scores below the truth there; a
  likelihood that disfavours the truth scores above it. Also PSID's M1 fit
  on both arms, round by round, with each round's share of reference points
  clipped at the box-logit limit. ``--rounds`` other than the default
  writes ``diagnose_rounds<n>.json``.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/hierarchical_validation.py null-sim
    PYTHONPATH=. .venv/bin/python scripts/hierarchical_validation.py v2-sim
    PYTHONPATH=. .venv/bin/python scripts/hierarchical_validation.py validate
    PYTHONPATH=. .venv/bin/python scripts/hierarchical_validation.py report
    PYTHONPATH=. .venv/bin/python scripts/hierarchical_validation.py diagnose

``--variant targeted`` (RESULTS 46) reruns U, V1 and V2 (V2's same simulated
populations) with the summary likelihood retrained with the targeted couples,
under ``summary_targeted``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from hh_npe.data.windows import add_age, cut_windows
from hh_npe.evaluation import population as pop
from scripts import hierarchical_population as hp
from scripts import hierarchical_summary as hs
from scripts import optionA as oa
from scripts.noise_calibration import MAIN, PSID_X

OUT = Path("outputs/hier_validation")
FIG = Path("figures/53_population_validation.png")
POOL = Path("data/processed/seed_pool_couples_income.npz")
PSID_RUN = Path("outputs/psid_optionA2n_couples")
N_HH, N_REP, N_UNIFORM, PER_CARD = 409, 10, 5, 4096
V2_BLOCKS, BLOCK = 16, 16
CARD_MIX = (0.5, 0.85)
SEED = 45_000_000           # household streams: no training run, SBC set or earlier check
AGE = FEATURES_TWOASSET_AGE.index("age")
NAMES = ("beta", "delta", "crra", "R_gamma")
#: Where PSID's fitted population sits (40% of it, RESULTS 46): beta's logit
#: below -2.5 (beta < 0.355) and delta's above 3 (delta > 0.9929).
CORNER = (-2.5, 3.0)
N_DIAGNOSE = 3


def in_corner(u: np.ndarray) -> np.ndarray:
    return (u[..., 0] < CORNER[0]) & (u[..., 1] > CORNER[1])


def theta_stars() -> dict[str, np.ndarray]:
    """The nulls' common thetas: A2n's PSID couples' median posterior mean;
    Laibson et al.'s point; and, where the summary fit exists, the median of
    PSID's fitted population, which sits at beta's lower edge, where a small
    range of theta is a long one in u."""
    s = json.loads((PSID_RUN / "summary.json").read_text())["uncorrected"]["median_of_means"]
    out = {"psid_median": np.array([s[k] for k in NAMES]), "laibson": hp.LAIBSON}
    fit = hp.OUT / "summary" / "uncorrected" / "summary.json"
    if fit.exists():
        q = json.loads(fit.read_text())["families"]["M1"]["natural"]["q05_q25_q50_q75_q95"]
        out["psid_population"] = np.array([q[j][2] for j in range(len(NAMES))])
    return out


def simulate(theta: np.ndarray, card: np.ndarray, per: int, seed: int) -> np.ndarray:
    """``per`` households at each row of ``theta`` (beta, delta, rho, R_gamma)
    with card type ``card``, by the training generation's solver, settings and
    seed pool; one window each, starting where a PSID couple's first wave does,
    with A2n's observation noise. Rows of the result follow the draws, ``per``
    consecutive households each."""
    from hh_npe.simulator.dispatch import simulate_batch_twoasset_gpu
    from scripts.generate_dataset import _load_pool

    cfg = json.loads((MAIN / "solver_config.json").read_text())
    pool, digest = _load_pool(POOL)
    if digest != cfg["init_pool_sha256"]:
        raise SystemExit(f"{POOL}: not the seed pool the training run used")
    spec = oa.spec_of(json.loads((hp.RUNS[0] / "results.json").read_text())["_config"])
    _x, _a, panel = simulate_batch_twoasset_gpu(
        theta[:, :3], seed, cfg["start_age"], cfg["n_waves"], cfg["wave_years"], cfg["grid"],
        cfg["theta_batch"], cfg["chunk"], return_panels=True, n_households=per,
        card=np.asarray(card), r_gamma=theta[:, 3], init_pool=pool)
    n = len(theta) * per
    ages = torch.load(PSID_X, weights_only=False)["x"].numpy()[:, 0, AGE].astype(int)
    starts = np.clip(np.random.default_rng(seed).choice(ages, n) - 1,
                     oa.WINDOW["start_low"], oa.WINDOW["start_high"])
    _t, x, ids = cut_windows(add_age(panel), np.zeros((n, 1)), np.arange(n), starts[:, None],
                             oa.WINDOW["n_waves"], oa.WINDOW["wave_years"],
                             FEATURES_TWOASSET_AGE)
    if len(ids) != n:
        raise SystemExit(f"{n - len(ids)} windows not alive")
    x = x[np.argsort(ids)].astype(np.float32)
    return oa.add_obs_noise(torch.from_numpy(x), spec.obs_noise, seed=seed + 77).numpy()


def cmd_null_sim(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    for j, (name, th) in enumerate(theta_stars().items()):
        if (args.out / f"null_{name}.npz").exists():
            continue
        card = np.array([0, 1])
        x = simulate(np.array([th, th]), card, PER_CARD, SEED + 1000 * j)
        np.savez(args.out / f"null_{name}.npz", x=x, card=np.repeat(card, PER_CARD), theta=th)
        print(f"{name} {np.round(th, 4).tolist()}: {len(x)} households; wrote "
              f"{args.out / f'null_{name}.npz'}", flush=True)


def untouched(net: hp.Network) -> dict:
    """The SBC set's untrained half: never seen by the summary likelihood."""
    sim = hs.simulated(net)
    keep = ~hs.split(sim["draw"], sim["sbc"])
    return {"theta": sim["theta"][keep], "x": sim["x"][keep].numpy(), "draw": sim["draw"][keep]}


def psid_population(args):
    s = json.loads((hp.OUT / args.likelihood / "uncorrected" / "summary.json").read_text())
    p = {k: torch.tensor(v, dtype=torch.float64) for k, v in s["families"]["M1"]["params"].items()}
    return hp.FAMILIES["M1"], p, (np.array(s["box"][0]), np.array(s["box"][1]))


def cmd_v2_sim(args) -> None:
    """V2's households: ``V2_BLOCKS`` blocks of ``BLOCK`` thetas drawn from
    PSID's fitted g, 16 households each. As in generation, R_gamma and card
    type are per block (the solver shares them across a batch): the block
    takes its first draw's R_gamma, and is cardholder with probability 0.85.
    Resampling the SBC set's untrained half instead left an ESS of 36 draws,
    since PSID's g sits where a uniform set has few."""
    spec_g, p_g, (lo, hi) = psid_population(args)
    th = pop.from_box_logit(pop.sample_u(spec_g, p_g, V2_BLOCKS * BLOCK, seed=SEED + 21), lo, hi)
    th = th.reshape(V2_BLOCKS, BLOCK, -1)
    th[:, :, 3] = th[:, :1, 3]
    th = th.reshape(-1, 4)
    rng = np.random.default_rng(SEED + 22)
    card = np.repeat((rng.random(V2_BLOCKS) < 0.85).astype(int), BLOCK)
    x = simulate(th, card, 16, SEED + 2000)
    hh_theta, hh_draw = np.repeat(th, 16, axis=0), np.repeat(np.arange(len(th)), 16)
    reps = [rng.choice(len(x), N_HH, replace=False) for _ in range(N_REP)]
    idx = np.concatenate(reps)
    out = args.out / args.likelihood
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "v2_populations.npz", x=x[idx], theta=hh_theta[idx],
             replicate=np.repeat(np.arange(N_REP), N_HH), draw=hh_draw[idx],
             card=np.repeat(card, 16)[idx])
    print(f"V2: {len(th)} thetas from PSID's g x 16 households; {N_REP} populations of {N_HH}, "
          f"distinct thetas per population "
          f"{np.mean([len(np.unique(hh_draw[r])) for r in reps]):.0f}")


def realised(theta: np.ndarray, lo, hi) -> np.ndarray:
    """A population's own derived quantities (mean, log sd, corr of u)."""
    u = pop.box_logit(np.clip(theta, lo + 1e-9, hi - 1e-9), lo, hi)
    c = np.cov(u.T)
    sd = np.sqrt(np.diag(c))
    i, j = np.tril_indices(len(lo), -1)
    return np.concatenate([u.mean(0), np.log(sd), (c / np.outer(sd, sd))[i, j]])


def fit_one(net, mdn, x, seed, args, truth=None) -> dict:
    xt, _ = oa.transform(torch.from_numpy(np.asarray(x)).float())
    res = hp.fit_population(net, xt, args.n_draws, args.n_ref, args.rounds, seed,
                            families_=("M1",), mdn=mdn)
    spec, p = hp.FAMILIES["M1"], res["fits"]["M1"][0]
    out = {"derived": pop.derived(spec, p).numpy().tolist(),
           "natural_sd": hp.natural(spec, p, net.lo, net.hi, n=50_000)["sd"]}
    if truth is not None:
        out["realised"] = realised(truth, net.lo, net.hi).tolist()
        out["hessian_cov"] = pop.hessian_cov(spec, p, res["hd"]).tolist()
        r0, r1 = pop.pooled(res["hd"]), pop.pooled(res["hd"], spec, p)
        for tag, r in (("unpooled", r0), ("pooled", r1)):
            out[tag] = {"covered": ((r["lo"] <= truth) & (truth <= r["hi"])).mean(0).tolist(),
                        "mse": ((r["mean"] - truth) ** 2).mean(0).tolist(),
                        "mean_var": (r["sd"] ** 2).mean(0).tolist(),
                        "ess_min": float(r["ess"].min()),
                        "ess_median": float(np.median(r["ess"]))}
    return out


def cmd_validate(args) -> None:
    net = hp.Network([hp.RUNS[i] for i in args.members])
    mdn = (hp.load_mdn(args.mdn or hp.MDN[args.variant], args.n_draws)
           if args.likelihood == "summary" else None)
    out = args.out / _sub(args)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "validation.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["settings"] = {"likelihood": args.likelihood, "variant": args.variant,
                       "n_draws": args.n_draws,
                       "n_ref": args.n_ref, "rounds": args.rounds}

    def save():
        path.write_text(json.dumps(res, indent=2))

    if "uniform" in args.parts and "uniform" not in res:
        pool = untouched(net)
        draws = np.unique(pool["draw"])
        rng = np.random.default_rng(SEED + 9)
        res["uniform"] = []
        for r in range(N_UNIFORM):
            idx = np.array([rng.choice(np.flatnonzero(pool["draw"] == g)) for g in draws])
            print(f"uniform test, population {r} ({len(idx)} households)", flush=True)
            res["uniform"].append(fit_one(net, mdn, pool["x"][idx], SEED + 300 + r, args,
                                          truth=pool["theta"][idx]))
            save()
    if "null" in args.parts:
        res.setdefault("null", {})
        for name in theta_stars():
            d = np.load(args.out / f"null_{name}.npz")
            for mix in CARD_MIX:
                if f"{name}, card {mix}" in res["null"]:
                    continue
                rng = np.random.default_rng(SEED + int(100 * mix))
                cards = [rng.permutation(np.flatnonzero(d["card"] == c)) for c in (0, 1)]
                n1 = int(round(mix * N_HH))
                fits = []
                for r in range(N_REP):
                    idx = np.concatenate([cards[1][r * n1:(r + 1) * n1],
                                          cards[0][r * (N_HH - n1):(r + 1) * (N_HH - n1)]])
                    print(f"null {name}, cardholders {mix}, population {r}", flush=True)
                    fits.append(fit_one(net, mdn, d["x"][idx], SEED + r, args))
                res["null"][f"{name}, card {mix}"] = fits
                save()
    if "v2" in args.parts and "v2" not in res:
        v2 = np.load(args.out / args.likelihood / "v2_populations.npz")
        res["v2"] = []
        for r in range(N_REP):
            m = v2["replicate"] == r
            print(f"V2 population {r}", flush=True)
            res["v2"].append(fit_one(net, mdn, v2["x"][m], SEED + 500 + r, args,
                                     truth=v2["theta"][m]))
            save()
    res["summary"] = summarise(res, args)
    save()
    report(res)


def _cached_s(net, x, seed, path: Path) -> np.ndarray:
    """Households' summaries, cached: they depend on the network and draws,
    not on the likelihood variant."""
    if path.exists():
        return np.load(path)["s"]
    xt, _ = oa.transform(torch.from_numpy(np.asarray(x)).float())
    s = hs.summaries(net, xt, hp.N_DRAWS, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, s=s)
    return s


def _calibration(mdn, s, theta, lo, hi) -> dict:
    u = pop.box_logit(np.clip(theta, lo + 1e-9, hi - 1e-9), lo, hi)
    pit = hs.marginal_cdf(mdn, torch.from_numpy(s).double(), torch.from_numpy(u).double())
    out = {}
    for tag, m in (("corner", in_corner(u)), ("outside", ~in_corner(u))):
        if not m.any():
            continue
        with torch.no_grad():
            sim = mdn.sample(torch.from_numpy(u[m]).double(),
                             generator=torch.Generator().manual_seed(1)).numpy()
        pp = pit[m]
        out[tag] = {"n": int(m.sum()), "median_pit": np.median(pp, 0).tolist(),
                    "cover90": ((pp >= 0.05) & (pp <= 0.95)).mean(0).tolist(),
                    "s_mean": s[m].mean(0).tolist(), "s_mean_predicted": sim.mean(0).tolist(),
                    "s_sd": s[m].std(0).tolist(), "s_sd_predicted": sim.std(0).tolist()}
    return out


def cmd_diagnose(args) -> None:
    net = hp.Network()
    mdn = hp.load_mdn(args.mdn or hp.MDN[args.variant], args.n_draws)
    lo, hi = net.lo, net.hi
    out = args.out / _sub(args)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("diagnose.json" if args.rounds == hp.ROUNDS
                  else f"diagnose_rounds{args.rounds}.json")
    s_tr, th_tr, draw_tr, train, *_ = hs.training_pairs(hs.OUT, args.variant)
    u_tr = pop.box_logit(np.clip(th_tr[train], lo + 1e-9, hi - 1e-9), lo, hi)
    spec_g, p_g, _box = psid_population(argparse.Namespace(likelihood="summary",
                                                           variant="base"))
    res = {"corner": CORNER,
           "training": {"thetas": int(len(np.unique(draw_tr[train]))),
                        "thetas_in_corner": int(len(np.unique(draw_tr[train][in_corner(u_tr)]))),
                        "households_in_corner": int(in_corner(u_tr).sum())},
           "psid_g_share_in_corner": float(in_corner(pop.sample_u(spec_g, p_g, 200_000)).mean())}
    cache = args.out / "summaries"
    v2 = np.load(args.out / "summary" / "v2_populations.npz")
    _, first = np.unique(v2["x"].reshape(len(v2["x"]), -1), axis=0, return_index=True)
    first = np.sort(first)
    s2 = _cached_s(net, v2["x"][first], SEED + 600, cache / "v2.npz")
    res["v2_households"] = _calibration(mdn, s2, v2["theta"][first], lo, hi)
    res["null"] = {}
    for j, name in enumerate(theta_stars()):
        d = np.load(args.out / f"null_{name}.npz")
        s = _cached_s(net, d["x"], SEED + 610 + j, cache / f"null_{name}.npz")
        res["null"][name] = _calibration(mdn, s, np.repeat(d["theta"][None], len(s), 0), lo, hi)
    res["v2_fits"] = []
    for r in range(N_DIAGNOSE):
        m = v2["replicate"] == r
        print(f"diagnose: V2 population {r}", flush=True)
        xt, _ = oa.transform(torch.from_numpy(v2["x"][m]).float())
        fit = hp.fit_population(net, xt, args.n_draws, args.n_ref, args.rounds,
                                SEED + 500 + r, mdn=mdn)
        dr, spec = fit["draws"], hp.FAMILIES["M1"]
        log_z = np.log(dr["in_box_frac"])
        ref = pop.Reference(spec, p_g, pop.Reference.scales, 2.0)
        th_ref = pop.from_box_logit(ref.sample_u(len(xt) * args.n_ref, SEED + 700 + r), lo, hi
                                    ).reshape(len(xt), args.n_ref, len(lo))
        s = dr["summary"]
        hd = pop.HouseholdDraws.from_mis(
            dr["theta"], hp.summary_log_l(mdn, s, dr["theta"], lo, hi, net.device),
            dr["log_qf"] - log_z[:, None], ref.log_density_theta(dr["theta"], lo, hi),
            th_ref, hp.summary_log_l(mdn, s, th_ref, lo, hi, net.device),
            net.log_density(xt, th_ref) - log_z[:, None], ref.log_density_theta(th_ref, lo, hi),
            lo, hi).to(net.device)
        p_fit = fit["fits"]["M1"][0]
        p_o, _ll = pop.fit(hd, spec, init=p_g)
        res["v2_fits"].append({
            "realised": realised(v2["theta"][m], lo, hi).tolist(),
            "rounds": [h["derived"] for h in fit["history"]],
            "refit_from_truth": pop.derived(spec, p_o).numpy().tolist(),
            "sum_log_m_truth": float(pop.log_marginal(spec, p_g, hd).sum()),
            "sum_log_m_fit": float(pop.log_marginal(spec, p_fit, hd).sum())})
        path.write_text(json.dumps(res, indent=2))
    res["psid_rounds"] = {}
    for arm in ("uncorrected", "corrected"):
        print(f"diagnose: PSID, {arm}", flush=True)
        xt, _keep = hp.psid_x(PSID_X, arm)
        fit = hp.fit_population(net, xt, args.n_draws, args.n_ref, args.rounds, 0, mdn=mdn)
        res["psid_rounds"][arm] = fit["history"]
        path.write_text(json.dumps(res, indent=2))
    report_diagnose(res)


def report_diagnose(res: dict) -> None:
    t = res["training"]
    print(f"training: {t['thetas']} thetas, {t['thetas_in_corner']} in the corner "
          f"(u beta < {res['corner'][0]}, u delta > {res['corner'][1]}; "
          f"{t['households_in_corner']} households); PSID's g there: "
          f"{res['psid_g_share_in_corner']:.2f}")
    fmt = lambda v, f="+.2f": " ".join(f"{a:{f}}" for a in v)  # noqa: E731
    for tag, c in [("V2 households", res["v2_households"])] + [
            (f"null {k}", v) for k, v in res["null"].items()]:
        for where, r in c.items():
            print(f"  {tag:22s} {where:7s} n={r['n']:5d} median PIT {fmt(r['median_pit'])}  "
                  f"cover90 {fmt(r['cover90'], '.2f')}  s mean {fmt(r['s_mean'])} "
                  f"(predicted {fmt(r['s_mean_predicted'])})")
    for i, f in enumerate(res["v2_fits"]):
        print(f"  V2 population {i}: beta, delta mean_u / sd_u: realised "
              f"{f['realised'][0]:+.2f} {f['realised'][1]:+.2f} / "
              f"{np.exp(f['realised'][4]):.2f} {np.exp(f['realised'][5]):.2f}; fit "
              f"{f['rounds'][-1][0]:+.2f} {f['rounds'][-1][1]:+.2f} / "
              f"{np.exp(f['rounds'][-1][4]):.2f} {np.exp(f['rounds'][-1][5]):.2f}; "
              f"refit from the truth {f['refit_from_truth'][0]:+.2f} "
              f"{f['refit_from_truth'][1]:+.2f}; sum log m truth {f['sum_log_m_truth']:.1f}, "
              f"fit {f['sum_log_m_fit']:.1f}")
        for h in f["rounds"]:
            print(f"    round: beta mean_u {h[0]:+.2f} sd_u {np.exp(h[4]):.2f}")
    for arm, hist in res.get("psid_rounds", {}).items():
        print(f"  PSID {arm}, by round (beta mean_u, sd_u; sum log m; reference points "
              "clipped):")
        for h in hist:
            print(f"    round {h['round']}: {h['derived'][0]:+.2f} {np.exp(h['derived'][4]):.2f}; "
                  f"{h['ll']:.1f}; {h['ref_clipped']:.3f}")


def _recovery(fits: list[dict], d: int) -> dict:
    fit = np.array([f["derived"] for f in fits])
    real = np.array([f["realised"] for f in fits])
    se = np.array([np.sqrt(np.clip(np.diag(f["hessian_cov"]), 0, None)) for f in fits])
    cover = np.abs(fit - real) <= 1.645 * se
    rel = np.exp(fit[:, d:2 * d] - real[:, d:2 * d]) - 1
    out = {}
    for j, n in enumerate(NAMES):
        c1 = np.mean([f["pooled"]["covered"][j] for f in fits])
        c0 = np.mean([f["unpooled"]["covered"][j] for f in fits])
        out[n] = {"sd_u_fitted": float(np.median(np.exp(fit[:, d + j]))),
                  "sd_u_realised": float(np.median(np.exp(real[:, d + j]))),
                  "sd_rel_bias_median": float(np.median(rel[:, j])),
                  "sd_ci_cover": float(cover[:, d + j].mean()),
                  "mean_ci_cover": float(cover[:, j].mean()),
                  "G2": bool(abs(np.median(rel[:, j])) <= 0.25 and cover[:, d + j].mean() >= 0.8),
                  "pooled_cover90": float(c1), "unpooled_cover90": float(c0),
                  "G3": bool(0.85 <= c1 <= 0.95),
                  "mse_ratio": float(np.mean([f["pooled"]["mse"][j] / f["unpooled"]["mse"][j]
                                              for f in fits])),
                  "var_ratio": float(np.mean([f["pooled"]["mean_var"][j]
                                              / f["unpooled"]["mean_var"][j] for f in fits]))}
    return out


def summarise(res: dict, args) -> dict:
    d = len(NAMES)
    out = {}
    if res.get("uniform"):
        out["uniform"] = _recovery(res["uniform"], d)
    if res.get("v2"):
        out["v2"] = _recovery(res["v2"], d)
    if res.get("null"):
        pooled = np.concatenate([[np.exp(f["derived"][d:2 * d]) for f in fits]
                                 for fits in res["null"].values()])
        out["null"] = {k: {"sd_u_median": np.median([np.exp(f["derived"][d:2 * d])
                                                     for f in fits], 0).tolist(),
                           "sd_u_max": np.max([np.exp(f["derived"][d:2 * d]) for f in fits],
                                              0).tolist()}
                       for k, fits in res["null"].items()}
        out["null_sd_u_q95"] = np.percentile(pooled, 95, 0).tolist()
        for k, fits in res["null"].items():
            out["null"][k]["natural_sd_median"] = np.median([f["natural_sd"] for f in fits],
                                                            0).tolist()
            out["null"][k]["natural_sd_q95"] = np.percentile([f["natural_sd"] for f in fits],
                                                             95, 0).tolist()
        psid = hp.OUT / hp.run_name(args.likelihood, args.variant) / "uncorrected" / "summary.json"
        if psid.exists():
            m1 = json.loads(psid.read_text())["families"]["M1"]
            der = m1["derived"]
            out["psid_sd_u"] = [float(np.exp(der[f"log_sd_u {n}"])) for n in NAMES]
            out["psid_natural_sd"] = m1["natural"]["sd"]
            out["heterogeneity"] = {n: bool(out["psid_sd_u"][j] > out["null_sd_u_q95"][j])
                                    for j, n in enumerate(NAMES)}
    return out


def report(res: dict) -> None:
    s = res["summary"]
    for part in ("uniform", "v2"):
        if part in s:
            print(f"\n{part}: fitted / realised sd_u (median), relative bias, Hessian 90% "
                  "cover of sd; pooled / unpooled 90% coverage of theta; MSE ratio")
            for n, r in s[part].items():
                print(f"  {n:8s} {r['sd_u_fitted']:.3f} / {r['sd_u_realised']:.3f}  "
                      f"{r['sd_rel_bias_median']:+.2f}  {r['sd_ci_cover']:.2f}"
                      f"{' G2' if r['G2'] else ' --'}  |  {r['pooled_cover90']:.3f} / "
                      f"{r['unpooled_cover90']:.3f}{' G3' if r['G3'] else ' --'}  "
                      f"MSE x{r['mse_ratio']:.2f}")
    if "null" in s:
        print("\nV1, households sharing one theta: fitted sd_u (median / max over populations)")
        for k, v in s["null"].items():
            print(f"  {k:28s} " + "  ".join(f"{n} {a:.3f}/{b:.3f}" for n, a, b in
                                             zip(NAMES, v["sd_u_median"], v["sd_u_max"],
                                                 strict=True)))
        print("  null 95th percentile: " + "  ".join(
            f"{n} {v:.3f}" for n, v in zip(NAMES, s["null_sd_u_q95"], strict=True)))
        if "psid_sd_u" in s:
            print("  PSID M1 sd_u:         " + "  ".join(
                f"{n} {v:.3f}" for n, v in zip(NAMES, s["psid_sd_u"], strict=True)))
            print("  heterogeneity (u): " + str(s["heterogeneity"]))
            print("natural-unit population sd, null 95th percentile by theta*, against PSID:")
            for k, v in s["null"].items():
                print(f"  {k:28s} " + "  ".join(f"{n} {a:.4f}" for n, a in
                                                 zip(NAMES, v["natural_sd_q95"], strict=True)))
            print("  PSID M1:                     " + "  ".join(
                f"{n} {a:.4f}" for n, a in zip(NAMES, s["psid_natural_sd"], strict=True)))


def _sub(args) -> str:
    """The run's directory under ``OUT``: likelihood, variant, members."""
    return hp.run_name(args.likelihood, args.variant) + (
        "" if len(args.members) == len(hp.RUNS) else "_members" + "".join(map(str, args.members)))


def cmd_report(args) -> None:
    res = json.loads((args.out / _sub(args) / "validation.json").read_text())
    res["summary"] = summarise(res, args)
    report(res)
    plot(res, args.fig)


def plot(res: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    s, d = res["summary"], len(NAMES)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for part, mk in (("uniform", "o"), ("v2", "s")):
        if not res.get(part):
            continue
        fit = np.array([np.exp(f["derived"][d:2 * d]) for f in res[part]])
        real = np.array([np.exp(f["realised"][d:2 * d]) for f in res[part]])
        for j, n in enumerate(NAMES):
            ax[0].scatter(real[:, j], fit[:, j], s=16, marker=mk, color=f"C{j}",
                          label=f"{n}" if part == "uniform" else None)
    lim = ax[0].get_xlim()[1]
    ax[0].plot([0, lim], [0, lim], color="0.5", lw=0.8)
    ax[0].set(title="fitted against realised population sd (u)\n"
                    "circles: uniform test; squares: V2 at PSID's g",
              xlabel="realised", ylabel="fitted")
    ax[0].legend(fontsize=8)
    if res.get("null"):
        for j in range(d):
            null = np.concatenate([[np.exp(f["derived"][d + j]) for f in fits]
                                   for fits in res["null"].values()])
            jit = np.random.default_rng(j).uniform(-0.15, 0.15, len(null))
            ax[1].scatter(j + jit, null, s=8, color="0.5",
                          label="V1: one theta for all" if j == 0 else None)
            ax[1].plot([j - 0.3, j + 0.3], [s["null_sd_u_q95"][j]] * 2, color="k",
                       label="null 95th percentile" if j == 0 else None)
            if "psid_sd_u" in s:
                ax[1].scatter([j], [s["psid_sd_u"][j]], marker="*", s=160, color="#d62728",
                              label="PSID couples" if j == 0 else None, zorder=3)
        ax[1].set_xticks(range(d), NAMES)
        ax[1].set(title="population sd (u): null against PSID")
        ax[1].legend(fontsize=8)
    part = "v2" if "v2" in s else "uniform"
    if part in s:
        w = 0.38
        ax[2].bar(np.arange(d) - w / 2, [s[part][n]["unpooled_cover90"] for n in NAMES], w,
                  label="unpooled", color="0.6")
        ax[2].bar(np.arange(d) + w / 2, [s[part][n]["pooled_cover90"] for n in NAMES], w,
                  label="pooled", color="tab:blue")
        ax[2].axhspan(0.85, 0.95, color="tab:green", alpha=0.15, label="gate G3")
        ax[2].set_xticks(range(d), NAMES)
        ax[2].set(title=f"90% interval coverage of the true theta ({part})", ylim=(0.3, 1.0))
        ax[2].legend(fontsize=8)
    variant = res["settings"].get("variant", "base")
    fig.suptitle(f"Population layer on simulated couples ({res['settings']['likelihood']} "
                 "likelihood" + (", RESULTS 45)" if variant == "base"
                                 else f", {variant}, RESULTS 46)"))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("null-sim", "v2-sim", "validate", "report", "diagnose"))
    ap.add_argument("--likelihood", choices=("summary", "npe"), default="summary")
    ap.add_argument("--variant", choices=tuple(hp.MDN), default="base",
                    help="The summary likelihood: RESULTS 45's (base) or retrained with the "
                         "targeted couples (RESULTS 46).")
    ap.add_argument("--mdn", type=Path, default=None, help="Default: the variant's.")
    ap.add_argument("--parts", nargs="+", default=["uniform", "null", "v2"])
    ap.add_argument("--members", type=int, nargs="+", default=list(range(5)),
                    help="Ensemble members to use (RESULTS 45.2: one member's density "
                         "against the five-member average).")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fig", type=Path, default=FIG)
    ap.add_argument("--n_draws", type=int, default=hp.N_DRAWS)
    ap.add_argument("--n_ref", type=int, default=hp.N_REF)
    ap.add_argument("--rounds", type=int, default=hp.ROUNDS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {"null-sim": cmd_null_sim, "v2-sim": cmd_v2_sim, "validate": cmd_validate,
     "report": cmd_report, "diagnose": cmd_diagnose}[args.cmd](args)


if __name__ == "__main__":
    main()
