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
- **V2, recovery at PSID's population.** ``select`` resamples 10 populations
  of 409 from the SBC set's untrained draws, weighted to PSID's fitted g. Fitted against
  realised spreads; gate G2 trusts a parameter's spread if its median
  relative bias is within 25% and the Hessian's 90% interval covers the
  realised value in at least 8 of 10.
- **V3, pooled households** (on V2): coverage of the true theta by the pooled
  90% intervals (gate G3: 0.85-0.95) against the unpooled; MSE of the pooled
  means relative to the unpooled; the variance ratio.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/hierarchical_validation.py null-sim
    PYTHONPATH=. .venv/bin/python scripts/hierarchical_validation.py select
    PYTHONPATH=. .venv/bin/python scripts/hierarchical_validation.py validate
    PYTHONPATH=. .venv/bin/python scripts/hierarchical_validation.py report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from hh_npe.data.windows import cut_windows
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
CARD_MIX = (0.5, 0.85)
SEED = 45_000_000           # household streams: no training run, SBC set or earlier check
AGE = FEATURES_TWOASSET_AGE.index("age")
NAMES = ("beta", "delta", "crra", "R_gamma")


def theta_stars() -> dict[str, np.ndarray]:
    s = json.loads((PSID_RUN / "summary.json").read_text())["uncorrected"]["median_of_means"]
    return {"psid_median": np.array([s[k] for k in NAMES]), "laibson": hp.LAIBSON}


def cmd_null_sim(args) -> None:
    from hh_npe.simulator.dispatch import simulate_batch_twoasset_gpu
    from scripts.generate_dataset import _load_pool

    cfg = json.loads((MAIN / "solver_config.json").read_text())
    pool, digest = _load_pool(POOL)
    if digest != cfg["init_pool_sha256"]:
        raise SystemExit(f"{POOL}: not the seed pool the training run used")
    spec = oa.spec_of(json.loads((hp.RUNS[0] / "results.json").read_text())["_config"])
    ages = torch.load(PSID_X, weights_only=False)["x"].numpy()[:, 0, AGE].astype(int)
    lo_start, hi_start = oa.WINDOW["start_low"], oa.WINDOW["start_high"]
    args.out.mkdir(parents=True, exist_ok=True)
    for j, (name, th) in enumerate(theta_stars().items()):
        _x, _a, panel = simulate_batch_twoasset_gpu(
            np.array([th[:3], th[:3]]), SEED + 1000 * j, cfg["start_age"], cfg["n_waves"],
            cfg["wave_years"], cfg["grid"], cfg["theta_batch"], cfg["chunk"],
            return_panels=True, n_households=PER_CARD, card=np.array([0, 1]),
            r_gamma=np.array([th[3], th[3]]), init_pool=pool)
        n = 2 * PER_CARD
        card = np.repeat([0, 1], PER_CARD)
        rng = np.random.default_rng(SEED + j)
        # Start so the window's age column matches a PSID couple's first wave.
        starts = np.clip(rng.choice(ages, n) - 1, lo_start, hi_start)
        _t, x, ids = cut_windows(panel, np.zeros((n, 1)), np.arange(n), starts[:, None],
                                 oa.WINDOW["n_waves"], oa.WINDOW["wave_years"],
                                 FEATURES_TWOASSET_AGE)
        if len(ids) != n:
            raise SystemExit(f"{name}: {n - len(ids)} windows not alive")
        order = np.argsort(ids)
        x = oa.add_obs_noise(torch.from_numpy(x[order].astype(np.float32)), spec.obs_noise,
                             seed=SEED + 77 + j).numpy()
        np.savez(args.out / f"null_{name}.npz", x=x, card=card, theta=th, start=starts)
        print(f"{name} {np.round(th, 4).tolist()}: {n} households; wrote "
              f"{args.out / f'null_{name}.npz'}")


def untouched(net: hp.Network) -> dict:
    """The SBC set's untrained half: never seen by the summary likelihood."""
    sim = hs.simulated(net)
    keep = ~hs.split(sim["draw"], sim["sbc"])
    return {"theta": sim["theta"][keep], "x": sim["x"][keep].numpy(), "draw": sim["draw"][keep]}


def psid_population(args):
    s = json.loads((hp.OUT / args.likelihood / "uncorrected" / "summary.json").read_text())
    p = {k: torch.tensor(v, dtype=torch.float64) for k, v in s["families"]["M1"]["params"].items()}
    return hp.FAMILIES["M1"], p


def cmd_select(args) -> None:
    net = hp.Network(hp.RUNS)
    pool = untouched(net)
    spec_g, p_g = psid_population(args)
    th = pool["theta"]
    with torch.no_grad():
        u = torch.from_numpy(pop.box_logit(np.clip(th, net.lo + 1e-9, net.hi - 1e-9),
                                           net.lo, net.hi))
        log_g = pop.log_density(spec_g, p_g, u).numpy() + pop.box_logit_log_jacobian(
            th, net.lo, net.hi)
    # The pool is uniform on the box, so g is the resampling weight.
    w = np.exp(log_g - log_g.max())
    w /= w.sum()
    rng = np.random.default_rng(SEED + 5)
    reps = [rng.choice(len(th), N_HH, replace=False, p=w) for _ in range(N_REP)]
    idx = np.concatenate(reps)
    out = args.out / args.likelihood
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "v2_populations.npz", x=pool["x"][idx], theta=th[idx],
             replicate=np.repeat(np.arange(N_REP), N_HH), draw=pool["draw"][idx])
    print(f"pool: {len(th)} untouched SBC households from {len(np.unique(pool['draw']))} draws; "
          f"ESS of the weights to PSID's g {1 / (w ** 2).sum():.0f}; distinct draws per "
          f"population {np.mean([len(np.unique(pool['draw'][r])) for r in reps]):.0f}")


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
    net = hp.Network(hp.RUNS)
    mdn = hp.load_mdn(args.mdn, args.n_draws) if args.likelihood == "summary" else None
    out = args.out / args.likelihood
    out.mkdir(parents=True, exist_ok=True)
    path = out / "validation.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["settings"] = {"likelihood": args.likelihood, "n_draws": args.n_draws,
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
    if "null" in args.parts and "null" not in res:
        res["null"] = {}
        for name in theta_stars():
            d = np.load(args.out / f"null_{name}.npz")
            for mix in CARD_MIX:
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
        v2 = np.load(out / "v2_populations.npz")
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
        psid = hp.OUT / args.likelihood / "uncorrected" / "summary.json"
        if psid.exists():
            der = json.loads(psid.read_text())["families"]["M1"]["derived"]
            out["psid_sd_u"] = [float(np.exp(der[f"log_sd_u {n}"])) for n in NAMES]
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
            print("  heterogeneity: " + str(s["heterogeneity"]))


def cmd_report(args) -> None:
    res = json.loads((args.out / args.likelihood / "validation.json").read_text())
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
    fig.suptitle(f"Population layer on simulated couples ({res['settings']['likelihood']} "
                 "likelihood, RESULTS 45)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("null-sim", "select", "validate", "report"))
    ap.add_argument("--likelihood", choices=("summary", "npe"), default="summary")
    ap.add_argument("--mdn", type=Path, default=Path("outputs/hier_summary/mdn.pt"))
    ap.add_argument("--parts", nargs="+", default=["uniform", "null", "v2"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fig", type=Path, default=FIG)
    ap.add_argument("--n_draws", type=int, default=hp.N_DRAWS)
    ap.add_argument("--n_ref", type=int, default=hp.N_REF)
    ap.add_argument("--rounds", type=int, default=hp.ROUNDS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {"null-sim": cmd_null_sim, "select": cmd_select, "validate": cmd_validate,
     "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
