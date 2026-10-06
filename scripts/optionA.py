"""Option A: train, evaluate and apply the four-parameter model (RESULTS.md 35).

Data: ``data/processed/optionA_card_dataset_shards`` -- 32,768 comphs draws of
(beta, delta, rho, R_gamma) from the switched edge-mixture proposal, card
access 50/50, 16 households per draw (RESULTS 30, 32). Configuration: the
adopted one (RESULTS 29) -- anchor + level inputs, ``log(1 - delta)`` target,
wide embedder, static standardisation on age and log mean income only -- with
R_gamma as a fourth target. ``train --beta_transform log1m|logit`` swaps beta's
flow target (RESULTS 37); evaluation and PSID read it from the checkpoints'
config.

Three things differ from every earlier training script, and each would be
silent if got wrong:

* **theta comes from the shards**, never from a sampler, and is checked against
  the proposal the generation recorded: the importance weights are only right
  for draws from that proposal.
* **Every posterior summary is importance-weighted** back to the uniform prior
  (``prior.proposal_log_weight``). The checkpoint records its proposal, and
  ``load_posterior`` refuses it to any caller that does not weight.
* **Card type is marginalised**: the network never sees it, so its posterior
  integrates over cardholder / no card at the 50/50 generation share. PSID has
  no card-access variable. Its proxy (any reported card debt) adds nothing the
  network cannot already see -- a household that borrows shows negative liquid
  wealth, which no no-card household can -- and it misclassifies cardholders
  who never borrow, so conditioning on it would only add bias.

``evaluate``:

* **SBC** on the 1,000 uniform-prior simulations, with weighted ranks:
  coverage, rank uniformity and recovery of the corrected posterior. Uniform
  draws, so recovery is comparable to earlier models' held-out numbers.
* **Coverage by posterior-mean region**: the valid conditional check. Coverage
  by TRUE-theta region (RESULTS 30.1) is printed for continuity but is not a
  calibration test: an exact posterior also under-covers truths at a prior
  edge whenever the likelihood is wide. In a uniform-prior Gaussian toy at
  beta's width an exact posterior covers 0.56-0.76 of truths above 0.95, and
  0.90 in every bin of its own posterior mean.
* **Held-out proposal draws** (the last shards, one household per draw so
  start ages are not clumped): the network's unweighted ``q``, comparable to
  the R_gamma pilot (RESULTS 32.1), and held-out ``log q``.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/optionA.py verify
    PYTHONPATH=. .venv/bin/python scripts/optionA.py train --seed 0
    PYTHONPATH=. .venv/bin/python scripts/optionA.py evaluate \
        --run_dirs outputs/optionA/s0 outputs/optionA/s1 ...
    PYTHONPATH=. .venv/bin/python scripts/optionA.py psid \
        --run_dirs outputs/optionA/s0 outputs/optionA/s1 ...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from hh_npe.data.windows import build_windowed, window_panel
from hh_npe.evaluation.weighted import sample_weighted, sbc_from_u
from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import (
    BETA_TRANSFORMS,
    PHASE3,
    PHASE3_RGAMMA,
    SWITCHED_A2,
    EdgeMixture,
    SwitchedProposal,
    beta_flow_bounds,
    beta_from_flow,
    beta_log_jacobian,
    beta_to_flow,
    from_log1m,
    log1m_box,
    proposal_log_weight,
    sample_sobol,
    to_log1m,
)
from hh_npe.npe.train import load_posterior, save_posterior, train_npe
from scripts.compare_windows import EMBEDDER, TRAINING, anchor_log, mean_income_channel

SHARDS = Path("data/processed/optionA_card_dataset_shards")
SBC_CACHE = Path("outputs/optionA/sbc_sims.pt")
PSID_X = Path("data/processed/psid_x_comphs_optionA.pt")
OUT = Path("outputs/optionA")
BOX = PHASE3_RGAMMA
N_TOTAL = 32768
N_HELDOUT_SHARDS = 2
WINDOW = dict(n_waves=7, start_low=24, start_high=45, wave_years=2, with_age=True)
ARCH = dict(d_model=128, n_layers=3, output_dim=64)
STATIC = ("age", "log_mean_income")
#: Window seeds: SBC as in ensemble_eval (4242), held-out as everywhere (999).
SBC_SEED, HELD_SEED = 4242, 999
PRIOR_SD = (BOX.high - BOX.low) / np.sqrt(12)
#: Region cut points for the coverage tables: RESULTS 30.1's, plus R_gamma thirds.
CUTS = {"beta": (0.30, 0.80, 0.95, 1.00), "delta": (0.85, 0.95, 0.98, 1.00),
        "crra": (0.5, 3.5, 4.5, 5.0), "R_gamma": (1.025, 1.0417, 1.0583, 1.075)}


# --------------------------------------------------------------------------- data

def load_shards(shards: Path, train_shards: int | None = None,
                need_complete: bool = False):
    """theta and card per draw from the shards, the train / held-out split, and
    the proposal the training draws came from.

    Held out: the last ``N_HELDOUT_SHARDS`` shards. Training: every shard before
    them, or only the first ``train_shards`` (smoke tests). Training draws are
    always a prefix ``[0, n_train)`` of the generation sequence, which is what
    ``SwitchedProposal.log_weight(theta, n_train)`` assumes.
    """
    cfg = json.loads((shards / "solver_config.json").read_text())
    if tuple(cfg.get("rgamma_range") or ()) != (BOX.rgamma_low, BOX.rgamma_high):
        raise SystemExit(f"{shards}: rgamma_range {cfg.get('rgamma_range')} does not "
                         f"match the box {BOX.rgamma_low}-{BOX.rgamma_high}")
    if not cfg.get("card_types") or cfg.get("educ") != "comphs":
        raise SystemExit(f"{shards}: expected a comphs card-types run, got {cfg}")
    files = sorted(shards.glob("shard_*.npz"))
    meta = []
    for f in files:
        d = np.load(f)
        meta.append((int(d["lo"]), int(d["hi"]), d["theta"], d["card"], f))
    meta.sort(key=lambda t: t[0])
    if [m[0] for m in meta] != [0] + [m[1] for m in meta[:-1]]:
        raise SystemExit(f"{shards}: shards do not tile a contiguous range from 0")
    n = meta[-1][1]
    if need_complete and n != N_TOTAL:
        raise SystemExit(f"{shards}: {n} of {N_TOTAL} draws generated")
    theta_all = np.concatenate([m[2] for m in meta])
    card_all = np.concatenate([m[3] for m in meta])

    # The weights are only right for these exact draws: regenerate the
    # proposal's sequence and compare.
    name = cfg["proposal"]
    # "uniform": the option A2 pilot (RESULTS 43). generate_dataset draws the
    # whole run's Sobol sequence and the pilot is its prefix; Sobol points are
    # sequential, so the prefix is sample_sobol(n).
    ref = {"edge_mixture": lambda: EdgeMixture().sample(n, seed=cfg["seed"]),
           "edge_mixture_switched": lambda: SwitchedProposal().sample(n, seed=cfg["seed"]),
           "uniform": lambda: sample_sobol(n, PHASE3, seed=cfg["seed"]),
           # The rest of the option A2 run: the uniform pilot, then the mixture
           # re-aimed at PSID couples (RESULTS 43.3).
           "a2_switched": lambda: SWITCHED_A2.sample(n, seed=cfg["seed"])}
    if name not in ref:
        raise SystemExit(f"proposal {name!r} not handled here")
    if not np.allclose(theta_all[:, :3], ref[name](), rtol=0, atol=1e-12):
        raise SystemExit(f"{shards}: stored theta is not the {name} sequence")

    if len(meta) <= N_HELDOUT_SHARDS:
        raise SystemExit("not enough shards for a held-out set")
    held = meta[-N_HELDOUT_SHARDS:]
    train = meta[:-N_HELDOUT_SHARDS]
    if train_shards is not None:
        train = train[:train_shards]
    n_train = train[-1][1]
    proposal = {"name": name, "n_train": int(n_train)}
    return (theta_all, card_all, [m[4] for m in train], [m[4] for m in held],
            proposal, cfg)


def transform(x: torch.Tensor):
    """The adopted input chain: level channel from dollar levels, then anchor."""
    x, feats = mean_income_channel(x, FEATURES_TWOASSET_AGE)
    return anchor_log(x, feats), feats


def windows(files, theta_all, seed, n_waves=WINDOW["n_waves"]):
    th, x, pid = build_windowed(files, theta_all, k=1, seed=seed,
                                **{**WINDOW, "n_waves": n_waves})
    x, feats = transform(x)
    return th, x, pid, feats


def one_per_draw(pid: torch.Tensor, seed: int) -> np.ndarray:
    """One random household row per draw. Rows come out grouped by window start
    age, so a prefix of them is NOT a random sample of start ages."""
    perm = np.random.default_rng(seed).permutation(len(pid))
    _, first = np.unique(pid.numpy()[perm], return_index=True)
    return np.sort(perm[first])


# --------------------------------------------------------------- flow targets

def flow_box(beta_transform: str = "linear"):
    """The box the flow lives in: ``log(1 - delta)`` always, beta's column as
    ``beta_transform`` (RESULTS 37)."""
    lo, hi = beta_flow_bounds(BOX, beta_transform)
    return dataclasses.replace(log1m_box(BOX), beta_low=lo, beta_high=hi)


def to_flow(theta, beta_transform: str = "linear"):
    return beta_to_flow(to_log1m(theta, BOX), BOX, beta_transform)


def from_flow(t, beta_transform: str = "linear"):
    return from_log1m(beta_from_flow(t, BOX, beta_transform), BOX)


# --------------------------------------------------------------------------- train

def train(args) -> None:
    theta_all, card_all, train_f, held_f, proposal, _cfg = load_shards(
        args.shards, args.train_shards)
    th, x, pid, feats = windows(train_f, theta_all, seed=0, n_waves=args.n_waves)
    drawn = np.unique(pid.numpy())
    assert drawn.min() == 0 and drawn.max() < proposal["n_train"]
    print(f"train: {len(th)} rows from {len(drawn)} draws (n_train "
          f"{proposal['n_train']}, proposal {proposal['name']}); cardholder share "
          f"{card_all[drawn].mean():.3f}; channels {list(feats)}", flush=True)
    keep = torch.tensor([f in STATIC for f in feats])
    f_mean = torch.where(keep, x.mean((0, 1)), torch.zeros(len(feats)))
    f_std = torch.where(keep, x.std((0, 1)), torch.ones(len(feats)))
    emb = TrajectoryTransformer(n_features=len(feats), seq_len=args.n_waves,
                                feature_mean=f_mean, feature_std=f_std,
                                per_sequence=False, **{**EMBEDDER, **ARCH})
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    over = {"max_num_epochs": args.max_epochs} if args.max_epochs else {}
    torch.manual_seed(args.seed)
    bt = args.beta_transform
    post, _d, _i = train_npe(to_flow(th, bt).float(), x, embedder=emb,
                             box=flow_box(bt), device=dev, group_ids=pid,
                             **{**TRAINING, "batch_size": 1024,
                                "learning_rate": 1e-3, **over})
    out = args.out / f"s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_posterior(post, emb, flow_box(bt), out / f"posterior_{args.n_waves}w.pt",
                   proposal=proposal)
    (out / "results.json").write_text(json.dumps({"_config": {
        "shards": str(args.shards), "train_shards": args.train_shards,
        "proposal": proposal, "box": list(BOX.names), "delta_transform": True,
        "beta_transform": bt,
        "window": {**WINDOW, "n_waves": args.n_waves}, "features": list(feats),
        "anchor_log": True,
        "mean_income_channel": True, "static_norm_channels": list(STATIC),
        "card": "marginalised", "arch": ARCH, "max_epochs": args.max_epochs,
        "train_seed": args.seed}}, indent=2))
    print(f"saved {out}")


# ------------------------------------------------------------------------ evaluate

def load_members(run_dirs, device=None, n_waves=WINDOW["n_waves"]):
    """Members of one ensemble: same configuration except the seed. Refuses
    members trained on another window length (the 5-wave models of the
    out-of-sample test, ``train --n_waves 5``)."""
    posts, cfgs, props = [], [], []
    for d in run_dirs:
        cfg = json.loads((Path(d) / "results.json").read_text())["_config"]
        if cfg["window"]["n_waves"] != n_waves:
            raise SystemExit(f"{d}: trained on {cfg['window']['n_waves']} waves, "
                             f"expected {n_waves}")
        ck = load_posterior(Path(d) / f"posterior_{n_waves}w.pt", map_location=device,
                            weighted_ok=True)
        p = ck["posterior"]
        if device:
            p.posterior_estimator.to(device)
            p._device = device
        posts.append(p)
        props.append(ck.get("proposal"))
        cfg.setdefault("beta_transform", "linear")     # runs before RESULTS 37
        cfgs.append({k: v for k, v in cfg.items() if k != "train_seed"})
    if any(c != cfgs[0] for c in cfgs) or any(p != props[0] for p in props):
        raise SystemExit("ensemble members differ in configuration or proposal")
    if props[0] is None:
        raise SystemExit("checkpoints record no proposal; not an option A run")
    return posts, props[0], cfgs[0]


def _weighting(proposal, beta_transform):
    inv = lambda t: from_flow(t, beta_transform)
    lw = lambda th: proposal_log_weight(proposal, th)
    return inv, lw


def _recovery(r: dict, truth: np.ndarray) -> dict:
    ok = np.isfinite(r["mean"][:, 0])
    m, s, t = r["mean"][ok], r["sd"][ok], truth[ok]
    cov = (r["lo"][ok] <= t) & (t <= r["hi"][ok])
    out = {}
    for j, n in enumerate(BOX.names):
        e = np.abs(m[:, j] - t[:, j]).mean()
        out[n] = {"corr": float(np.corrcoef(m[:, j], t[:, j])[0, 1]),
                  "mae": float(e), "mae_over_prior_sd": float(e / PRIOR_SD[j]),
                  "contraction": float(1 - (s[:, j] ** 2).mean() / PRIOR_SD[j] ** 2),
                  "coverage_90_interval": float(cov[:, j].mean())}
    out["n"] = int(ok.sum())
    out["median_in_box"] = float(np.median(r["in_box_frac"]))
    out["median_ess_frac"] = float(np.nanmedian(r["ess"] / np.maximum(r["n_kept"], 1)))
    return out


def _by_region(v: np.ndarray, u: np.ndarray) -> dict:
    """Coverage within each region of ``v`` (one column per parameter), and
    which side the misses fall on: ``above_95`` is the share of truths above
    the 95th percentile (an upper tail too short), ``below_5`` below the 5th."""
    out = {}
    for j, n in enumerate(BOX.names):
        c = CUTS[n]
        out[n] = []
        for a, b in zip(c[:-1], c[1:]):
            m = (v[:, j] >= a) & ((v[:, j] < b) if b < c[-1] else (v[:, j] <= b))
            uj = u[m, j]
            out[n].append({"lo": a, "hi": b, "n": int(m.sum()),
                           "coverage_90": float(((uj >= 0.05) & (uj <= 0.95)).mean())
                           if m.any() else None,
                           "below_5": float((uj < 0.05).mean()) if m.any() else None,
                           "above_95": float((uj > 0.95).mean()) if m.any() else None})
    return out


def _sbc(r: dict, truth: np.ndarray) -> dict:
    ok = np.isfinite(r["u"]).all(1)
    cov, ks = sbc_from_u(r["u"])
    u = r["u"][ok]
    return {"coverage_90": dict(zip(BOX.names, map(float, cov))),
            "ks_p": dict(zip(BOX.names, map(float, ks))),
            "by_posterior_mean": _by_region(r["mean"][ok], u),
            "by_true_theta": _by_region(truth[ok], u),
            "recovery": _recovery(r, truth)}


def _log_q(members, theta: torch.Tensor, x: torch.Tensor, beta_transform: str,
           batch=1024) -> float:
    """Mean held-out log density of the equal-weight mixture, with delta as
    ``log(1 - delta)`` and beta on its own scale (the beta transform's Jacobian
    added back, so arms with different beta targets compare), and without the
    box-truncation renormalisation sbi's ``log_prob`` adds -- comparable between
    option A runs, not to earlier models."""
    dev = torch.device(getattr(members[0], "_device", "cpu"))
    th_flow = to_flow(theta, beta_transform)
    jac = torch.from_numpy(beta_log_jacobian(theta[:, 0].numpy(), BOX, beta_transform))
    out = []
    with torch.no_grad():
        for b0 in range(0, len(x), batch):
            tb, xb = th_flow[b0:b0 + batch].to(dev), x[b0:b0 + batch].to(dev)
            lp = torch.stack([m.posterior_estimator.log_prob(tb[None], condition=xb)[0]
                              for m in members])
            out.append((torch.logsumexp(lp, 0) - np.log(len(members))).cpu())
    return float((torch.cat(out).double() + jac).mean())


def evaluate(args) -> None:
    members, proposal, cfg = load_members(args.run_dirs, args.device)
    bt = cfg["beta_transform"]
    inv, lw = _weighting(proposal, bt)
    fb = flow_box(bt)
    print(f"{len(members)} members; proposal {proposal}; beta transform {bt}")

    cache = torch.load(args.sbc_cache, weights_only=False)
    if (tuple(cache["rgamma_range"]) != (BOX.rgamma_low, BOX.rgamma_high)
            or not cache["card_types"] or cache["educ"] != "comphs"):
        raise SystemExit(f"{args.sbc_cache} was not simulated for option A")
    th_sbc, x_sbc, _ids = window_panel(cache["panels"], cache["thetas"].numpy(),
                                       k=1, seed=SBC_SEED, **WINDOW)
    x_sbc, _ = transform(x_sbc)
    th_sbc = th_sbc.numpy().astype(float)
    if args.n_sbc:
        th_sbc, x_sbc = th_sbc[:args.n_sbc], x_sbc[:args.n_sbc]

    theta_all, _card, _tr, held_f, _p, _cfg = load_shards(args.shards)
    th_ho, x_ho, pid_ho, _f = windows(held_f, theta_all, seed=HELD_SEED)
    pick = one_per_draw(pid_ho, seed=0)[:args.n_heldout]
    th_ho, x_ho = th_ho[pick], x_ho[pick]
    print(f"SBC: {len(th_sbc)} uniform-prior draws; held-out: {len(th_ho)} proposal "
          f"draws, one household each")

    sets = {"ensemble": members} if len(members) > 1 else {}
    sets.update({f"member_{i}": [m] for i, m in enumerate(members)})
    res = {"proposal": proposal, "config": cfg, "run_dirs": list(map(str, args.run_dirs))}
    torch.manual_seed(0)
    for name, ms in sets.items():
        r = sample_weighted(ms, x_sbc, args.n_draws, fb.low, fb.high,
                            invert=inv, log_weight=lw, truth=th_sbc)
        q = sample_weighted(ms, x_ho, args.n_draws, fb.low, fb.high,
                            invert=inv, log_weight=None, truth=th_ho.numpy())
        res[name] = {"sbc": _sbc(r, th_sbc),
                     "heldout_q": {**_recovery(q, th_ho.numpy()),
                                   "log_q": _log_q(ms, th_ho, x_ho, bt)}}
        if name == "ensemble" or len(sets) == 1:
            np.savez(args.out / f"sbc_{name}.npz", u=r["u"], mean=r["mean"],
                     sd=r["sd"], truth=th_sbc)
        print(f"  {name} done", flush=True)
    (args.out / "evaluation.json").write_text(json.dumps(res, indent=2))
    report(res)


def report(res: dict) -> None:
    names = BOX.names
    print("\n=== SBC (uniform prior, importance-weighted) ===")
    print(f"{'':12s}" + "".join(f"{n:>22s}" for n in names))
    for k in [k for k in res if k.startswith(("ensemble", "member"))]:
        s = res[k]["sbc"]
        print(f"{k:12s}" + "".join(
            f"   cov {s['coverage_90'][n]:.3f} ks {s['ks_p'][n]:.3f}" for n in names))
    key = "ensemble" if "ensemble" in res else "member_0"
    s = res[key]["sbc"]
    print(f"\n--- {key}: recovery on the SBC draws (corrected posterior) ---")
    print(f"{'':10s}{'corr':>8s}{'mae':>10s}{'mae/prior sd':>14s}{'contraction':>13s}")
    for n in names:
        v = s["recovery"][n]
        print(f"{n:10s}{v['corr']:8.3f}{v['mae']:10.4f}{v['mae_over_prior_sd']:14.3f}"
              f"{v['contraction']:13.3f}")
    print(f"in-box median {s['recovery']['median_in_box']:.3f}; median ESS / kept "
          f"{s['recovery']['median_ess_frac']:.3f}")
    for title, k in (("by POSTERIOR-MEAN region (the calibration check)", "by_posterior_mean"),
                     ("by TRUE-theta region (continuity with RESULTS 30.1; an exact "
                      "posterior also under-covers at the edges)", "by_true_theta")):
        print(f"\n--- {key}: 90% coverage {title} ---")
        for n in names:
            print(f"{n:8s}" + "".join(
                f"  [{b['lo']:g}, {b['hi']:g}) "
                + (f"{b['coverage_90']:.3f}" if b["coverage_90"] is not None else "  -  ")
                + f" (n={b['n']})" for b in s[k][n]))
    print(f"\n--- {key}: which side the misses fall on, by POSTERIOR-MEAN region "
          f"(5% each if calibrated) ---")
    for n in names:
        print(f"{n:8s}" + "".join(
            f"  [{b['lo']:g}, {b['hi']:g}) "
            + (f"below {b['below_5']:.3f} above {b['above_95']:.3f}"
               if b["coverage_90"] is not None else "  -  ")
            for b in s["by_posterior_mean"][n]))
    h = res[key]["heldout_q"]
    print(f"\n--- {key}: held-out proposal draws, unweighted q (cf. RESULTS 32.1) ---")
    print(f"{'':10s}{'corr':>8s}{'mae':>10s}{'mae/prior sd':>14s}{'cov90':>8s}")
    for n in names:
        v = h[n]
        print(f"{n:10s}{v['corr']:8.3f}{v['mae']:10.4f}{v['mae_over_prior_sd']:14.3f}"
              f"{v['coverage_90_interval']:8.3f}")
    print(f"held-out log q (beta linear, delta as log(1-delta)): {h['log_q']:.3f}"
          + "".join(f"   {k} {res[k]['heldout_q']['log_q']:.3f}"
                    for k in res if k.startswith("member")))


# ---------------------------------------------------------------------------- PSID

def psid(args) -> None:
    from scripts.literature_ranges import LAIBSON, META
    from scripts.psid_posterior import graded_correction

    members, proposal, cfg = load_members(args.run_dirs, args.device)
    bt = cfg["beta_transform"]
    inv, lw = _weighting(proposal, bt)
    fb = flow_box(bt)
    d = torch.load(args.x, weights_only=False)
    if list(d["features"]) != list(FEATURES_TWOASSET_AGE):
        raise SystemExit(f"{args.x}: features {d['features']} are not "
                         f"{list(FEATURES_TWOASSET_AGE)}")
    from hh_npe.simulator.laibson_calibration import EDUC_GROUPS
    keep = np.flatnonzero(d["educ"].numpy() == EDUC_GROUPS.index("comphs"))
    x_raw = d["x"].numpy()[keep]
    print(f"{args.x}: {len(keep)} comphs households; {len(members)} members; "
          f"proposal {proposal}; beta transform {bt}")
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "household_index.npy", keep)

    ci = FEATURES_TWOASSET_AGE.index("consumption")
    arms = {"uncorrected": x_raw, "corrected": graded_correction(x_raw, cons_idx=ci)}
    names = list(BOX.names)
    lit_lo = np.append(META[0], BOX.rgamma_low)      # R_gamma: no literature band
    lit_hi = np.append(META[1], BOX.rgamma_high)
    ref = np.append(LAIBSON, 1.05)                   # their calibrated R_gamma
    summary = {"proposal": proposal, "beta_transform": bt, "n_households": int(len(keep))}
    for arm, xa in arms.items():
        torch.manual_seed(0)
        xt, _ = transform(torch.from_numpy(xa).float())
        r = sample_weighted(members, xt, args.n_draws, fb.low, fb.high,
                            invert=inv, log_weight=lw)
        np.savez(args.out / f"posterior_{arm}.npz", mean=r["mean"], sd=r["sd"],
                 lo=r["lo"], hi=r["hi"], in_box_frac=r["in_box_frac"],
                 ess=r["ess"], n_kept=r["n_kept"], names=np.array(names))
        ok = np.isfinite(r["mean"][:, 0])
        m, lo, hi = r["mean"][ok], r["lo"][ok], r["hi"][ok]
        inside = (m >= lit_lo) & (m <= lit_hi)
        covers = (lo <= ref) & (ref <= hi)
        summary[arm] = {
            "median_of_means": dict(zip(names, np.median(m, 0).tolist())),
            "sd_across_households": dict(zip(names, m.std(0).tolist())),
            "median_posterior_sd": dict(zip(names, np.median(r["sd"][ok], 0).tolist())),
            "median_90_width": dict(zip(names, np.median(hi - lo, 0).tolist())),
            "share_ci_covers_reference": dict(zip(names, covers.mean(0).tolist())),
            "share_mean_in_meta_range": dict(zip(names[:3], inside[:, :3].mean(0).tolist())),
            "share_all_three_in_meta_range": float(inside[:, :3].all(1).mean()),
            "in_box_median": float(np.median(r["in_box_frac"])),
            "in_box_p10": float(np.percentile(r["in_box_frac"], 10)),
            "median_ess_frac": float(np.median(r["ess"][ok] / r["n_kept"][ok])),
            # Weights vary most for posteriors near delta = 1, where the
            # proposal is densest; the low tail of the ESS is the one to watch.
            "ess_p10": float(np.percentile(r["ess"][ok], 10)),
            "unrepresentable": int((~ok).sum()),
        }
        s = summary[arm]
        print(f"\n--- {arm} ---")
        print(f"{'':24s}" + "".join(f"{n:>11s}" for n in names))
        for k in ("median_of_means", "sd_across_households", "median_posterior_sd",
                  "median_90_width", "share_ci_covers_reference"):
            print(f"{k:24s}" + "".join(f"{s[k][n]:11.4f}" for n in names))
        print(f"{'reference':24s}" + "".join(f"{v:11.4f}" for v in ref)
              + "   (Laibson et al.; R_gamma their calibrated 1.05)")
        print("households whose mean is in the meta-analytic range: "
              + "  ".join(f"{n} {s['share_mean_in_meta_range'][n]:.1%}" for n in names[:3])
              + f"   all three {s['share_all_three_in_meta_range']:.1%}")
        print(f"in-box median {s['in_box_median']:.3f} (p10 {s['in_box_p10']:.3f}); "
              f"median ESS / kept {s['median_ess_frac']:.3f}, ESS p10 {s['ess_p10']:.0f}; "
              f"unrepresentable "
              f"{s['unrepresentable']}")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    # Standing requirement: every estimation result ships a contour figure.
    from hh_npe.evaluation.plots import contour_corner
    m = np.load(args.out / "posterior_uncorrected.npz")["mean"]
    m = m[np.isfinite(m[:, 0])]
    contour_corner(
        {f"household posterior means, option A  N={len(m)}": m}, BOX,
        truth={"Laibson et al. MSM (R_gamma: calibrated 1.05)": ref},
        truth_markers=("*",), truth_colors=("#d62728",),
        bands={"meta-analytic range (lit.; R_gamma none)": (lit_lo, lit_hi)},
        reflect_axes=("delta",),
        axis_limits=(BOX.low, np.where(np.array(names) == "delta", 1.04, BOX.high)),
        path=args.out / "psid_household_means.png",
        title="Per-household posterior means, PSID comphs, option A (4 parameters)"
              + ("" if bt == "linear" else f", beta target {bt}") + "\n"
              "importance-weighted to the uniform prior; card type marginalised")
    print(f"\nwrote {args.out}")


# ---------------------------------------------------------------------------- verify

def verify(args) -> None:
    """Every check the reboot audit ran, before anything trains on the shards."""
    theta_all, card_all, train_f, held_f, proposal, cfg = load_shards(
        args.shards, need_complete=not args.partial)
    tb = int(cfg["theta_batch"])
    # Per-block R_gamma and card, as generated (RESULTS 32).
    blocks = theta_all[:, 3].reshape(-1, tb) if len(theta_all) % tb == 0 else None
    if blocks is None or not (blocks == blocks[:, :1]).all():
        raise SystemExit("R_gamma is not constant within blocks")
    if not (card_all.reshape(-1, tb) == card_all.reshape(-1, tb)[:, :1]).all():
        raise SystemExit("card type is not constant within blocks")
    lo, hi = BOX.low, BOX.high
    if not ((theta_all >= lo) & (theta_all <= hi)).all():
        raise SystemExit("theta outside the prior box")
    bad = []
    for f in train_f + held_f:
        d = np.load(f)
        for k in d.files:
            if (k.startswith("panel_") or k == "x") and not np.isfinite(d[k]).all():
                bad.append(f"{f.name}:{k}")
        if "init_pool_sha256" in cfg:
            # RESULTS 43: every household's age-20 start, from the pool.
            w = d["init_wealth"] if "init_wealth" in d.files else None
            if w is None or w.shape != (len(d["x"]), 2) or (w < 0).any():
                bad.append(f"{f.name}:init_wealth")
            if "init_state" in d.files and not np.array_equal(
                    d["init_state"], d["panel_income_state"][:, 0]):
                bad.append(f"{f.name}:init_state")
    if bad:
        raise SystemExit(f"non-finite values or bad initial conditions: {bad}")
    print(f"OK: {len(theta_all)} draws in {len(train_f) + len(held_f)} shards, "
          f"theta matches {proposal['name']}, R_gamma and card per block of {tb}, "
          f"all panels finite. Training {proposal['n_train']} draws "
          f"({len(train_f)} shards), held out {len(held_f)} shards. Cardholder share "
          f"{card_all.mean():.3f}; R_gamma {theta_all[:, 3].min():.4f}-"
          f"{theta_all[:, 3].max():.4f}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="Check the shards before training.")
    v.add_argument("--shards", type=Path, default=SHARDS)
    v.add_argument("--partial", action="store_true",
                   help="Accept an unfinished run (checks the shards so far).")
    t = sub.add_parser("train", help="Train one ensemble member.")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--shards", type=Path, default=SHARDS)
    t.add_argument("--train_shards", type=int, default=None,
                   help="Use only the first N training shards (smoke tests).")
    t.add_argument("--max_epochs", type=int, default=None,
                   help="Cap epochs (smoke tests); default TRAINING's.")
    t.add_argument("--device", default=None)
    t.add_argument("--beta_transform", choices=BETA_TRANSFORMS, default="linear",
                   help="beta's flow target (RESULTS 37); delta is always log(1 - delta).")
    t.add_argument("--n_waves", type=int, default=WINDOW["n_waves"],
                   help="Window length; 5 for the out-of-sample test (oos_optionA.py).")
    t.add_argument("--out", type=Path, default=OUT)
    e = sub.add_parser("evaluate", help="SBC and held-out scores, ensemble and members.")
    e.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    e.add_argument("--shards", type=Path, default=SHARDS)
    e.add_argument("--sbc_cache", type=Path, default=SBC_CACHE)
    e.add_argument("--n_draws", type=int, default=2000,
                   help="Raw posterior draws per household, split across members.")
    e.add_argument("--n_sbc", type=int, default=None, help="Use only the first N SBC draws.")
    e.add_argument("--n_heldout", type=int, default=1024)
    e.add_argument("--device", default=None)
    e.add_argument("--out", type=Path, default=OUT / "evaluation")
    p = sub.add_parser("psid", help="Per-household posteriors for PSID comphs.")
    p.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    p.add_argument("--x", type=Path, default=PSID_X)
    p.add_argument("--n_draws", type=int, default=40000,
                   help="Raw draws per household; the weights cost effective "
                        "sample size, most for posteriors near delta = 1 "
                        "(at 4,000 the ESS p10 was 106; RESULTS 36).")
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=Path("outputs/psid_optionA"))
    args = ap.parse_args()
    if args.cmd == "evaluate":
        args.out.mkdir(parents=True, exist_ok=True)
    {"verify": verify, "train": train, "evaluate": evaluate, "psid": psid}[args.cmd](args)


if __name__ == "__main__":
    main()
