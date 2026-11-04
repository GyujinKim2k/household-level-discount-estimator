"""Observation noise as an unknown (RESULTS 62).

Every earlier arm was told its noise range (v2, typical, gross; RESULTS 58),
so RESULTS 61's three population answers are three assumptions, not three
readings of the data. Here each simulated household draws its own noise sd
per dollar feature uniformly on ``[0, SIGMA_HIGH]``, 1.3 times the gross
arm's upper end (RESULTS 57.4), and the network estimates it: the targets are
(beta, log(1 - delta), rho, R_gamma) as v2's, then the four sigmas, each as
``logit(sigma / SIGMA_HIGH)``. Everything else is v2's: data, windows,
architecture, training settings and five members.

Whether the panel separates noise from preferences is then measurable:
- ``sbc``: on v2's SBC set (uniform theta, sigma from its prior), coverage and
  recovery for all eight targets, and how far each theta posterior trades
  against the sigmas (within-posterior correlations).
- ``psid``: PSID couples' and all 889's posteriors for theta and sigma.
- ``population``: RESULTS 61's indirect inference with eight statistics (the
  mean posterior means of theta and sigma) for eight unknowns (theta and one
  noise sd per feature for the population). Its Jacobian also says how far
  the network's theta estimate moves when the true noise changes: zero if it
  tells noise from signal.

The noise is the same kind as every arm's (``optionA.add_obs_noise``): each
wave multiplied by ``exp(sigma z)``, z iid N(0, 1); zero stays zero and signs
are kept. PSID itself never gets any.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/noise_unknown.py train --seed 0
    PYTHONPATH=. .venv/bin/python scripts/noise_unknown.py sbc
    PYTHONPATH=. .venv/bin/python scripts/noise_unknown.py psid
    PYTHONPATH=. .venv/bin/python scripts/noise_unknown.py population
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from hh_npe.npe.prior import BETA_RANGE_A2, LOG1M_EPS, proposal_log_weight
from scripts import optionA as oa

MAIN = Path("data/processed/couples_v2_dataset_shards")
STRIP = Path("data/processed/couples_v2_strip_dataset_shards")
SBC = Path("data/processed/optionA2v2_sbc16_shards")
RANGES = Path("outputs/psid_iid_error/psid_iid_error.json")
OUT = Path("outputs/optionA2v2_noise_unknown")
NOISY = oa.NOISY
SIGMA_NAMES = tuple(f"sigma_{f}" for f in NOISY)
NAMES = oa.BOX.names + SIGMA_NAMES
#: The gross arm's upper ends (RESULTS 57.4, ``RANGES``), and the prior's: 1.3 times them.
GROSS_HIGH = np.array([0.194, 0.201, 0.631, 0.591])
SIGMA_HIGH = np.round(1.3 * GROSS_HIGH, 3)
#: Added to a window seed for the noise stream; no other arm uses it.
SIGMA_SEED = 620_000
_E = float(np.log(LOG1M_EPS) - np.log1p(-LOG1M_EPS))     # logit's cap, as beta's


def add_noise(x: torch.Tensor, seed: int, sigma=None):
    """Raw (dollar) windows times ``exp(sigma z)`` per dollar feature, and the
    sigmas used, ``(rows, 4)``. Each row draws its sigma uniformly on
    ``[0, SIGMA_HIGH]``; ``sigma`` (4,) or (rows, 4) sets it instead. The
    stream (z, and the uniforms drawn either way) depends on ``seed`` only, so
    the same seed at another sigma is a common-random-numbers comparison."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(x.shape, generator=g, dtype=x.dtype)
    u = torch.rand((x.shape[0], len(NOISY)), generator=g, dtype=x.dtype)
    hi = torch.as_tensor(SIGMA_HIGH, dtype=x.dtype)
    s = hi * u if sigma is None else torch.as_tensor(
        np.asarray(sigma, float), dtype=x.dtype).expand(x.shape[0], len(NOISY)).clone()
    out = x.clone()
    for k, f in enumerate(NOISY):
        j = FEATURES_TWOASSET_AGE.index(f)
        out[..., j] = x[..., j] * torch.exp(s[:, k, None] * z[..., j])
    return out, s


def to_flow(theta8, spec: oa.Spec):
    """(theta, sigma) to the flow's targets: theta as ``spec``'s, sigma as
    ``logit(sigma / SIGMA_HIGH)``, capped as beta's logit is."""
    t = spec.to_flow(theta8[:, :4])
    if torch.is_tensor(theta8):
        hi = torch.as_tensor(SIGMA_HIGH, dtype=theta8.dtype)
        s = torch.clamp(theta8[:, 4:] / hi, LOG1M_EPS, 1 - LOG1M_EPS)
        return torch.cat([t, torch.log(s) - torch.log1p(-s)], 1)
    s = np.clip(np.asarray(theta8, float)[:, 4:] / SIGMA_HIGH, LOG1M_EPS, 1 - LOG1M_EPS)
    return np.concatenate([t, np.log(s) - np.log1p(-s)], 1)


def from_flow(t8: np.ndarray, spec: oa.Spec) -> np.ndarray:
    from scipy.special import expit

    t8 = np.asarray(t8, float)
    return np.concatenate([spec.from_flow(t8[..., :4]), SIGMA_HIGH * expit(t8[..., 4:])], -1)


def flow_box(spec: oa.Spec) -> SimpleNamespace:
    fb = spec.flow_box()
    return SimpleNamespace(low=np.concatenate([fb.low, np.full(4, _E)]),
                           high=np.concatenate([fb.high, np.full(4, -_E)]))


def prior_sd(spec: oa.Spec) -> np.ndarray:
    return np.concatenate([spec.prior_sd, SIGMA_HIGH / np.sqrt(12)])


def check_ranges() -> None:
    """``GROSS_HIGH`` must be RESULTS 57.4's gross upper ends (when the
    git-ignored file is present)."""
    if RANGES.exists():
        d = json.loads(RANGES.read_text())["sigma_range_network"]["with_gross_errors"]
        if not np.allclose([d[f][1] for f in NOISY], GROSS_HIGH):
            raise SystemExit(f"{RANGES}: the gross upper ends are not {GROSS_HIGH}")


# --------------------------------------------------------------------------- train

def train(args) -> None:
    from hh_npe.npe.embedder import TrajectoryTransformer
    from hh_npe.npe.train import save_posterior, train_npe
    from scripts.compare_windows import EMBEDDER, TRAINING

    check_ranges()
    theta_all, card_all, train_f, _held, proposal, cfg = oa.load_shards(
        args.shards, args.train_shards)
    spec = oa.Spec(box=oa.box_of(cfg), beta_transform="logit", rgamma_transform="logit")
    th, x, pid = oa.windows_raw(train_f, theta_all, seed=0)
    x, sig = add_noise(x, seed=SIGMA_SEED)
    x, feats = oa.transform(x)
    target = to_flow(torch.cat([th.double(), sig.double()], 1), spec)
    print(f"train: {len(th)} rows from {len(np.unique(pid.numpy()))} draws; cardholder share "
          f"{card_all[np.unique(pid.numpy())].mean():.3f}; {spec.describe()}; noise sd per "
          f"household uniform on [0, {SIGMA_HIGH.tolist()}] and estimated", flush=True)
    keep = torch.tensor([f in oa.STATIC for f in feats])
    f_mean = torch.where(keep, x.mean((0, 1)), torch.zeros(len(feats)))
    f_std = torch.where(keep, x.std((0, 1)), torch.ones(len(feats)))
    emb = TrajectoryTransformer(n_features=len(feats), seq_len=oa.WINDOW["n_waves"],
                                feature_mean=f_mean, feature_std=f_std,
                                per_sequence=False, **{**EMBEDDER, **oa.ARCH})
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    over = {"max_num_epochs": args.max_epochs} if args.max_epochs else {}
    torch.manual_seed(args.seed)
    box = flow_box(spec)
    post, _d, _i = train_npe(target.float(), x, embedder=emb, box=box, device=dev,
                             group_ids=pid, **{**TRAINING, "batch_size": 1024,
                                               "learning_rate": 1e-3, **over})
    out = args.out / f"s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_posterior(post, emb, box, out / f"posterior_{oa.WINDOW['n_waves']}w.pt",
                   proposal=proposal)
    lag = oa.flow_lag_of([f.path for f in train_f])
    (out / "results.json").write_text(json.dumps({"_config": {
        "shards": list(map(str, args.shards)), "train_shards": args.train_shards,
        "proposal": proposal, "box": list(oa.BOX.names), "delta_transform": True,
        "beta_transform": "logit", "rgamma_range": cfg["rgamma_range"],
        # RESULTS 63: the box with the beta strip below 0.3
        **({"beta_range": cfg["beta_range"]} if "beta_range" in cfg else {}),
        "rgamma_transform": "logit",
        "noise_targets": {"features": list(NOISY), "high": SIGMA_HIGH.tolist(),
                          "transform": "logit", "seed": SIGMA_SEED},
        "window": {**oa.WINDOW, "flow_lag": lag}, "features": list(feats),
        "anchor_log": True, "mean_income_channel": True,
        "static_norm_channels": list(oa.STATIC), "card": "marginalised", "arch": oa.ARCH,
        "max_epochs": args.max_epochs, "train_seed": args.seed}}, indent=2))
    print(f"saved {out}")


# ---------------------------------------------------------------------- members

class Members:
    """The noise-unknown ensemble: v2's θ part, plus the four sigma targets."""

    def __init__(self, run=OUT, device=None):
        dirs = sorted(Path(run).glob("s[0-9]*"))
        self.nets, self.proposal, self.cfg = oa.load_members(dirs, device)
        nt = self.cfg.get("noise_targets")
        if not nt or not np.allclose(nt["high"], SIGMA_HIGH) or nt["features"] != list(NOISY):
            raise SystemExit(f"{run}: not a noise-unknown ensemble with sigma up to {SIGMA_HIGH}")
        self.spec = oa.spec_of(self.cfg)
        self.n = len(dirs)

    def posterior(self, x: torch.Tensor, n_draws: int, truth=None, draws: bool = False):
        """Importance-weighted posterior summaries for (theta, sigma); the
        weights are theta's (sigma's training draws are its prior). ``draws``
        returns the weighted draws themselves (``sample_draws``)."""
        from hh_npe.evaluation.weighted import sample_draws, sample_weighted

        fb = flow_box(self.spec)

        def inv(t):
            return from_flow(t, self.spec)

        def lw(th):
            return proposal_log_weight(self.proposal, th[:, :4])

        torch.manual_seed(0)
        if draws:
            return sample_draws(self.nets, x, n_draws, fb.low, fb.high, invert=inv,
                                log_weight=lw)
        return sample_weighted(self.nets, x, n_draws, fb.low, fb.high, invert=inv,
                               log_weight=lw, truth=truth)


def weighted_corr(theta: np.ndarray, log_w: np.ndarray, n_kept: np.ndarray) -> np.ndarray:
    """Each row's weighted within-posterior correlation matrix, ``(rows, 8, 8)``
    (NaN for a row with fewer than 20 kept draws)."""
    out = np.full((len(theta), theta.shape[2], theta.shape[2]), np.nan)
    for i, n in enumerate(n_kept):
        if n < 20:
            continue
        t, lw = theta[i, :n].astype(float), log_w[i, :n]
        w = np.exp(lw - lw.max())
        w /= w.sum()
        c = t - w @ t
        cov = (w[:, None] * c).T @ c
        sd = np.sqrt(np.diag(cov))
        out[i] = cov / np.outer(sd, sd)
    return out


def _cross(corr: np.ndarray) -> dict:
    """Median within-posterior correlation of each theta with each sigma."""
    med = np.nanmedian(corr, 0)
    return {a: {b: float(med[i, 4 + k]) for k, b in enumerate(SIGMA_NAMES)}
            for i, a in enumerate(oa.BOX.names)}


# ------------------------------------------------------------------------ SBC

def sbc(args) -> None:
    """v2's SBC set (1,024 uniform-prior draws, 16 households each), every
    household with its own sigma from the prior: calibration and recovery of
    all eight targets."""
    from scipy import stats

    from scripts.sbc_optionA2 import load_sets
    from scripts.tail_diagnostic import BLOCK

    m = Members(args.run, args.device)
    theta_all, files, _sets, _ranges = load_sets(args.sbc, m.spec.box, train=MAIN)
    oa.check_flow_lag(m.cfg, files)
    th, x, pid = oa.windows_raw(files, theta_all, seed=oa.SBC_SEED)
    x, sig = add_noise(x, seed=SIGMA_SEED + oa.SBC_SEED)
    x, _f = oa.transform(x)
    truth = np.concatenate([th.numpy(), sig.numpy()], 1).astype(float)
    pid = pid.numpy()
    one = np.zeros(len(pid), bool)
    one[oa.one_per_draw(torch.from_numpy(pid), seed=0)] = True
    print(f"{m.n} members; {len(th)} households from {len(np.unique(pid))} draws", flush=True)
    r = m.posterior(x, args.n_draws, truth=truth)
    d = m.posterior(x[np.flatnonzero(one)], args.n_draws, draws=True)
    corr = weighted_corr(d["theta"], d["log_w"], d["n_kept"])

    ok = np.isfinite(r["u"]).all(1)
    inside = (r["u"] >= 0.05) & (r["u"] <= 0.95)
    sd0 = prior_sd(m.spec)
    blk = pid // BLOCK
    rows = np.flatnonzero(ok)
    ub, inv = np.unique(blk[rows], return_inverse=True)
    of = [rows[inv == b] for b in range(len(ub))]
    rng = np.random.default_rng(0)
    boot = [np.concatenate([of[b] for b in rng.integers(0, len(ub), len(ub))])
            for _ in range(200)]
    res = {"n_households": int(ok.sum()), "sigma_high": SIGMA_HIGH.tolist(), "targets": {}}
    for j, n in enumerate(NAMES):
        mean, t = r["mean"][ok, j], truth[ok, j]
        pick = ok & one & ((pid % BLOCK == 0) if n == "R_gamma" else True)
        cov_b = [inside[i, j].mean() for i in boot]
        res["targets"][n] = {
            "coverage_90": float(inside[ok, j].mean()),
            "coverage_90_ci": np.percentile(cov_b, [5, 95]).tolist(),
            "ks_p_one_per_draw": float(stats.kstest(r["u"][pick, j], "uniform").pvalue),
            "corr": float(np.corrcoef(mean, t)[0, 1]),
            "mae_over_prior_sd": float(np.abs(mean - t).mean() / sd0[j]),
            "contraction": float(1 - (r["sd"][ok, j] ** 2).mean() / sd0[j] ** 2)}
    # sigma recovered by its true value: can the network tell low from high noise?
    for k, n in enumerate(SIGMA_NAMES):
        edges = np.linspace(0, SIGMA_HIGH[k], 6)
        b = np.clip(np.digitize(truth[ok, 4 + k], edges) - 1, 0, 4)
        res["targets"][n]["by_true_quintile"] = [
            {"lo": float(edges[q]), "hi": float(edges[q + 1]),
             "mean_of_means": float(r["mean"][ok, 4 + k][b == q].mean()),
             "median_sd": float(np.median(r["sd"][ok, 4 + k][b == q]))} for q in range(5)]
    # theta's calibration where the network reads the data as noisy or clean
    # (posterior-mean sigma terciles: conditioning on the data, a valid check)
    res["theta_coverage_by_posterior_sigma"] = {}
    for k, n in enumerate(SIGMA_NAMES):
        q = np.quantile(r["mean"][ok, 4 + k], [1 / 3, 2 / 3])
        b = np.digitize(r["mean"][ok, 4 + k], q)
        res["theta_coverage_by_posterior_sigma"][n] = {
            nm: [float(inside[ok, j][b == t].mean()) for t in range(3)]
            for j, nm in enumerate(oa.BOX.names)}
    res["within_posterior_corr_theta_sigma"] = _cross(corr)
    res["within_posterior_corr_median"] = np.nanmedian(corr, 0).tolist()
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "sbc16.npz", truth=truth, pid=pid, one_per_draw=one,
             **{k: r[k] for k in ("mean", "sd", "lo", "hi", "u")}, names=np.array(NAMES))
    (args.out / "sbc16.json").write_text(json.dumps(res, indent=2))
    show_sbc(res)
    sbc_figure(r["u"][ok & one], truth[ok], r["mean"][ok], args.figure, sbc_where(m.spec, pid))
    print(f"wrote {args.out} and {args.figure}")


def show_sbc(res: dict) -> None:
    print(f"\n{'':20s}{'cov90':>8s}{'(90% CI)':>16s}{'KS p':>8s}{'corr':>8s}"
          f"{'MAE/sd':>8s}{'contr':>8s}")
    for n, v in res["targets"].items():
        print(f"{n:20s}{v['coverage_90']:8.3f}   {v['coverage_90_ci'][0]:.3f}-"
              f"{v['coverage_90_ci'][1]:.3f}{v['ks_p_one_per_draw']:8.3f}{v['corr']:8.3f}"
              f"{v['mae_over_prior_sd']:8.3f}{v['contraction']:8.3f}")
    for n in SIGMA_NAMES:
        print(f"{n}: true quintile -> mean of means " + "  ".join(
            f"[{b['lo']:.3f},{b['hi']:.3f}) {b['mean_of_means']:.3f}"
            for b in res["targets"][n]["by_true_quintile"]))
    print("theta coverage by posterior-mean sigma tercile (low / mid / high):")
    for n, v in res["theta_coverage_by_posterior_sigma"].items():
        print(f"  {n:24s}" + "  ".join(f"{k} " + "/".join(f"{c:.3f}" for c in cs)
                                       for k, cs in v.items()))
    print("median within-posterior correlation, theta (rows) with sigma (columns):")
    for a, v in res["within_posterior_corr_theta_sigma"].items():
        print(f"  {a:8s}" + "".join(f"{c:+9.3f}" for c in v.values()))


def arm_head(box) -> str:
    """A figure title's head: the arm, and its beta range when it is not the A box's."""
    lo, hi = box.beta_low, box.beta_high
    if (lo, hi) == tuple(BETA_RANGE_A2):
        return "Noise unknown (RESULTS 62)"
    return f"Noise unknown, beta {lo:g}-{hi:g} (RESULTS 63)"


def sbc_where(spec, pid: np.ndarray) -> str:
    """The SBC figure's title head: the arm's box and how many draws."""
    if (spec.box.beta_low, spec.box.beta_high) == tuple(BETA_RANGE_A2):
        return f"{arm_head(spec.box)}: SBC on v2's set"
    return (f"{arm_head(spec.box)}: SBC on v2's set and the beta strip, "
            f"{len(np.unique(pid)):,} draws")


def sbc_figure(u: np.ndarray, truth: np.ndarray, mean: np.ndarray, path: Path,
               where: str = "Noise unknown (RESULTS 62): SBC on v2's set") -> None:
    """Top: rank ECDF minus uniform, eight targets, with the 95% KS band.
    Bottom: each sigma's posterior mean against its truth (2D 68/95% contours)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hh_npe.evaluation.plots import _hpd_levels, _kde_grid

    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    grid = np.linspace(0, 1, 201)
    band = 1.358 / np.sqrt(len(u))
    for j, n in enumerate(NAMES):
        a = ax[0, 0] if j < 4 else ax[0, 1]
        e = (u[:, j][:, None] <= grid).mean(0) - grid
        a.plot(grid, e, label=n)
    for a, t in zip(ax[0, :2], ("theta", "sigma"), strict=True):
        a.fill_between(grid, -band, band, color="0.85", label="95% KS band")
        a.set(title=f"SBC rank ECDF − uniform: {t}", xlabel="rank", ylim=(-0.08, 0.08))
        a.legend(fontsize=7)
    for a in ax[0, 2:]:
        a.axis("off")
    for k, n in enumerate(SIGMA_NAMES):
        a = ax[1, k]
        X, Y, Z = _kde_grid(truth[:, 4 + k], mean[:, 4 + k], (0, 0), (SIGMA_HIGH[k],) * 2)
        a.contour(X, Y, Z, levels=_hpd_levels(Z), colors="#D08A1E", linestyles=["--", "-"])
        a.plot([0, SIGMA_HIGH[k]], [0, SIGMA_HIGH[k]], "k:", lw=0.8)
        a.set(xlabel=f"true {n}", ylabel="posterior mean", xlim=(0, SIGMA_HIGH[k]),
              ylim=(0, SIGMA_HIGH[k]), title=n)
    fig.suptitle(textwrap.fill(f"{where}, each household's noise sd "
                               f"uniform on [0, {SIGMA_HIGH.tolist()}]; bottom: posterior mean "
                               "against truth, 68% (solid) / 95% (dashed)", 150), fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------- PSID

#: Where an arm's PSID posteriors go: ``<root>/<PSID_NAME>_{couples,all}``.
PSID_NAME = "psid_optionA2v2_noise_unknown"


def psid_dirs(root: Path = Path("outputs"), name: str = PSID_NAME) -> dict:
    """PSID inputs and where their posteriors go, under ``root``."""
    return {"couples": (Path("data/processed/psid_x_comphs_couples.pt"),
                        root / f"{name}_couples"),
            "all": (Path("data/processed/psid_x_comphs_optionA.pt"),
                    root / f"{name}_all")}


def psid_x(path: Path) -> torch.Tensor:
    """PSID comphs windows, transformed (uncorrected consumption), as ``optionA.psid``."""
    from hh_npe.simulator.laibson_calibration import EDUC_GROUPS

    d = torch.load(path, weights_only=False)
    if list(d["features"]) != list(FEATURES_TWOASSET_AGE):
        raise SystemExit(f"{path}: features {d['features']}")
    keep = np.flatnonzero(d["educ"].numpy() == EDUC_GROUPS.index("comphs"))
    return oa.transform(torch.from_numpy(d["x"].numpy()[keep]).float())[0], keep


def psid(args) -> None:
    from scripts.literature_ranges import LAIBSON

    m = Members(args.run, args.device)
    ref = np.append(LAIBSON, 1.05)
    summary = {}
    dirs = psid_dirs(args.root, args.psid_name)
    for name, (xf, out) in dirs.items():
        x, keep = psid_x(xf)
        r = m.posterior(x, args.n_draws)
        d = m.posterior(x, min(args.n_draws, 8000), draws=True)
        corr = weighted_corr(d["theta"], d["log_w"], d["n_kept"])
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "household_index.npy", keep)
        np.savez(out / "posterior_uncorrected.npz", **{k: r[k] for k in (
            "mean", "sd", "lo", "hi", "in_box_frac", "ess", "n_kept")}, names=np.array(NAMES))
        ok = np.isfinite(r["mean"]).all(1)
        mm = r["mean"][ok]
        summary[name] = {
            "n": int(ok.sum()), "unrepresentable": int((~ok).sum()),
            "median_of_means": dict(zip(NAMES, np.median(mm, 0).tolist(), strict=True)),
            "mean_of_means": dict(zip(NAMES, mm.mean(0).tolist(), strict=True)),
            "p10_of_means": dict(zip(NAMES, np.percentile(mm, 10, 0).tolist(), strict=True)),
            "p90_of_means": dict(zip(NAMES, np.percentile(mm, 90, 0).tolist(), strict=True)),
            "sd_across_households": dict(zip(NAMES, mm.std(0, ddof=1).tolist(), strict=True)),
            "median_posterior_sd": dict(zip(NAMES, np.median(r["sd"][ok], 0).tolist(),
                                            strict=True)),
            "share_ci_covers_laibson": dict(zip(oa.BOX.names, (
                (r["lo"][ok, :4] <= ref) & (ref <= r["hi"][ok, :4])).mean(0).tolist(),
                strict=True)),
            "within_posterior_corr_theta_sigma": _cross(corr),
            "in_box_median": float(np.median(r["in_box_frac"])),
            "median_ess_frac": float(np.median(r["ess"][ok] / r["n_kept"][ok]))}
        s = summary[name]
        print(f"\n=== PSID {name}: {s['n']} households "
              f"({s['unrepresentable']} unrepresentable) ===")
        for k in ("median_of_means", "mean_of_means", "p10_of_means", "p90_of_means",
                  "sd_across_households", "median_posterior_sd"):
            print(f"{k:22s}" + "".join(f"{s[k][n]:9.4f}" for n in NAMES))
        print("Laibson's value inside the 90% interval: " + "  ".join(
            f"{n} {v:.2f}" for n, v in s["share_ci_covers_laibson"].items()))
        print("median within-posterior corr theta x sigma:")
        for a, v in s["within_posterior_corr_theta_sigma"].items():
            print(f"  {a:8s}" + "".join(f"{c:+9.3f}" for c in v.values()))
        if name == "couples":
            psid_figure(mm, d, ok, args.figure, m.spec.box)
    (dirs["couples"][1] / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {dirs['couples'][1]}, {dirs['all'][1]} and {args.figure}")


def psid_figure(means: np.ndarray, d: dict, ok: np.ndarray, path: Path, tbox=oa.BOX) -> None:
    """Couples' posterior means and the posterior of the couple nearest their
    median, all eight targets, 68/95%; the arms' assumed noise ranges shaded."""
    from hh_npe.evaluation.plots import contour_corner

    box = SimpleNamespace(names=NAMES, low=np.concatenate([tbox.low, np.zeros(4)]),
                          high=np.concatenate([tbox.high, SIGMA_HIGH]))
    z = (means - means.mean(0)) / means.std(0)
    i = np.flatnonzero(ok)[np.argmin(np.abs(z - np.median(z, 0)).sum(1))]
    n = d["n_kept"][i]
    w = np.exp(d["log_w"][i, :n] - d["log_w"][i, :n].max())
    rs = np.random.default_rng(0).choice(n, 4000, p=w / w.sum())
    rngs = json.loads(RANGES.read_text())["sigma_range_network"] if RANGES.exists() else {}
    bands = {}
    for lab, key in (("v2's noise range", None), ("gross arm's range", "with_gross_errors")):
        lo, hi = box.low.copy(), box.high.copy()
        v = ({"income": [0, 0], "consumption": [0, 0.15], "liquid_assets": [0, 0.10],
              "illiquid_assets": [0.175, 0.47]} if key is None else rngs.get(key))
        if v:
            lo[4:] = [v[f][0] for f in NOISY]
            hi[4:] = [max(v[f][1], v[f][0] + 0.003) for f in NOISY]
            bands[lab] = (lo, hi)
    contour_corner({f"couples' posterior means, N={len(means)}": means,
                    f"one couple's posterior (nearest the median, #{i})": d["theta"][i, rs]},
                   box, bands=bands, reflect_axes=("delta",), path=path,
                   axis_limits=(box.low, np.where(np.array(NAMES) == "delta", 1.0, box.high)),
                   title=f"{arm_head(tbox)}: PSID couples, θ and each feature's noise "
                         "sd estimated together; importance-weighted, card type marginalised")


# ----------------------------------------------------------------- population

POP_OUT = Path("outputs/noise_unknown")
_POP = {"out": POP_OUT}


def _pop_out() -> Path:
    return _POP["out"]
POP_SEED = SIGMA_SEED + 61          # the population sims' noise stream
#: central-difference half-steps in each sigma
SIGMA_STEP = np.array([0.02, 0.02, 0.06, 0.06])


def _score(m: Members, sim: dict, sigma: np.ndarray, name: str, n_draws: int) -> dict:
    """Posterior means for simulated households with noise sd ``sigma`` on
    every household (common random numbers: ``POP_SEED`` everywhere)."""
    f = _pop_out() / f"post_unknown_{name}.npz"
    if f.exists():
        d = dict(np.load(f))
        if not np.allclose(d["sigma"], sigma) or not np.allclose(d["theta"], sim["theta"]):
            raise SystemExit(f"{f} was scored at another point")
        return d
    x, _s = add_noise(torch.from_numpy(sim["x"]), seed=POP_SEED, sigma=sigma)
    r = m.posterior(oa.transform(x)[0], n_draws)
    d = {"mean": r["mean"], "sd": r["sd"], "card": sim["card"], "ids": sim["ids"],
         "sigma": np.asarray(sigma, float), "theta": sim["theta"]}
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, **d)
    print(f"scored {name}", flush=True)
    return d


def _score_known(arm, sim: dict, sigma: np.ndarray, name: str, n_draws: int) -> np.ndarray:
    """A noise-known arm's posterior means for the same households and noise:
    the network is not told the noise changed."""
    from scripts.sbc_optionA2 import posterior

    f = _pop_out() / f"post_{arm.name}_{name}.npz"
    if f.exists():
        d = dict(np.load(f))
        if not np.allclose(d["sigma"], sigma):
            raise SystemExit(f"{f} was scored at another sigma")
        return d["mean"]
    x, _s = add_noise(torch.from_numpy(sim["x"]), seed=POP_SEED, sigma=sigma)
    r = posterior(arm.members, oa.transform(x)[0], None, arm.spec, n_draws)
    np.savez(f, mean=r["mean"], sigma=np.asarray(sigma, float))
    print(f"scored {name} with {arm.name}", flush=True)
    return r["mean"]


def population(args) -> None:
    from scripts import laibson_comparison as lc

    _POP["out"] = args.out
    lc.OUT = args.sim_cache          # where the simulated households are cached
    m = Members(args.run, args.device)
    lai = lc.laibson()
    th_l = np.append(lai["theta"], lc.R_GAMMA)
    u_l = lc.to_u(th_l)
    couples = psid_dirs(args.root, args.psid_name)["couples"][1]
    b_lo, b_hi = lc.box_bounds(m.spec.box)
    z = np.load(couples / "posterior_uncorrected.npz")
    pm = z["mean"][np.isfinite(z["mean"]).all(1)]
    target = pm.mean(0)
    se_t = pm.std(0, ddof=1) / np.sqrt(len(pm))
    # the noise starts at couples' own estimate, away from its box's ends
    s0 = np.clip(target[4:], SIGMA_STEP + 0.005, SIGMA_HIGH - SIGMA_STEP - 0.005)
    sims = {"laibson": lc.cached_sim("laibson", th_l)}
    for k in range(4):
        for sign, tag in ((1, "plus"), (-1, "minus")):
            sims[f"{tag}{k}"] = lc.cached_sim(f"{tag}{k}",
                                              lc.from_u(u_l + sign * lc.STEP[k] * np.eye(4)[k]))
    card = sims["laibson"]["card"]
    sh = lc.CARD_SHARE
    nd = args.n_draws
    base = _score(m, sims["laibson"], s0, "laibson", nd)
    fd = {"plus": [_score(m, sims[f"plus{k}"], s0, f"plus{k}", nd) for k in range(4)],
          "minus": [_score(m, sims[f"minus{k}"], s0, f"minus{k}", nd) for k in range(4)]}
    for k in range(4):
        e = SIGMA_STEP[k] * np.eye(4)[k]
        fd["plus"].append(_score(m, sims["laibson"], s0 + e, f"laibson_sigma_plus{k}", nd))
        fd["minus"].append(_score(m, sims["laibson"], s0 - e, f"laibson_sigma_minus{k}", nd))
    step = np.concatenate([lc.STEP, SIGMA_STEP])
    J = jacobian8(fd, card, sh, step)

    lo_v = np.concatenate([np.minimum(lc.to_u(b_lo), lc.to_u(b_hi)), np.full(4, 0.003)])
    hi_v = np.concatenate([np.maximum(lc.to_u(b_lo), lc.to_u(b_hi)), SIGMA_HIGH - 0.003])
    v = np.concatenate([u_l, s0])
    at_v, post_hat = lc.stat(base["mean"], card, sh), base
    path = [{"theta": th_l.tolist(), "sigma": s0.tolist(), "stat": at_v.tolist()}]
    for it in range(1, lc.MAX_NEWTON + 1):
        if (np.abs(target - at_v) < 0.25 * se_t).all():
            break
        v = np.clip(lc.newton(v, J, target, at_v), lo_v, hi_v)
        th = lc.from_u(v[:4])
        sim = lc.cached_sim(f"ii_{args.tag}_{it}", th)
        post_hat = _score(m, sim, v[4:], f"ii_{args.tag}_{it}", nd)
        at_v = lc.stat(post_hat["mean"], card, sh)
        path.append({"theta": th.tolist(), "sigma": v[4:].tolist(), "stat": at_v.tolist()})
    resid = (target - at_v) / se_t
    v_raw = lc.newton(v, J, target, at_v)
    v_hat = np.clip(v_raw, lo_v, hi_v)
    rng = np.random.default_rng(0)
    boot_raw = boot8(v, fd, post_hat, pm, sh, step, rng)
    boot_raw = np.concatenate([lc.from_u(boot_raw[:, :4]), boot_raw[:, 4:]], 1)
    lo8 = np.concatenate([b_lo, np.full(4, 0.003)])
    hi8 = np.concatenate([b_hi, SIGMA_HIGH - 0.003])
    beyond = ((boot_raw < lo8) | (boot_raw > hi8)).mean(0)
    boot = np.clip(boot_raw, lo8, hi8)
    rep = np.concatenate([lc.from_u(v_hat[:4]), v_hat[4:]])

    # how far each network's theta estimate moves when the true noise moves
    sens = {args.tag: J[:4, 4:].tolist()}
    for name in args.known:
        arm = lc.Arm(name, args.device)
        plus = [_score_known(arm, sims["laibson"], s0 + SIGMA_STEP[k] * np.eye(4)[k],
                             f"laibson_sigma_plus{k}", nd) for k in range(4)]
        minus = [_score_known(arm, sims["laibson"], s0 - SIGMA_STEP[k] * np.eye(4)[k],
                              f"laibson_sigma_minus{k}", nd) for k in range(4)]
        sens[name] = lc.jacobian(np.array([lc.stat(p, card, sh) for p in plus]),
                                 np.array([lc.stat(p, card, sh) for p in minus]),
                                 SIGMA_STEP)[:4].tolist()
    res = {"tag": args.tag, "box_floor_beta": float(m.spec.box.beta_low),
           "sigma_high": SIGMA_HIGH.tolist(), "card_share": sh, "theta_laibson": th_l.tolist(),
           "psid_mean_of_means": target.tolist(), "psid_se": se_t.tolist(),
           "sigma_start": s0.tolist(), "jacobian": J.tolist(), "newton_path": path,
           "last_simulated": np.concatenate([lc.from_u(v[:4]), v[4:]]).tolist(),
           "last_simulated_misfit_over_psid_se": resid.tolist(),
           "converged": bool((np.abs(resid) < 0.25).all()),
           "representative": rep.tolist(),
           "representative_unclipped": np.concatenate([lc.from_u(v_raw[:4]), v_raw[4:]]).tolist(),
           "at_box_edge": (v_raw != v_hat).tolist(), "boot_share_beyond_box": beyond.tolist(),
           "representative_se": boot.std(0, ddof=1).tolist(),
           "representative_ci90": np.percentile(boot, [5, 95], axis=0).T.tolist(),
           "representative_corr": np.corrcoef(boot.T).tolist(),
           "at_laibson_stat": lc.stat(base["mean"], card, sh).tolist(),
           "theta_estimate_per_unit_true_sigma": sens, "boot": boot.tolist()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "population.json").write_text(json.dumps(res, indent=2))
    show_population(res)
    if args.with_arms and args.with_arms.exists():
        population_figure(res, json.loads(args.with_arms.read_text()), couples, args.figure,
                          args.title)
        print(f"wrote {args.out / 'population.json'} and {args.figure}")


def jacobian8(fd: dict, card, share, step) -> np.ndarray:
    from scripts import laibson_comparison as lc

    return lc.jacobian(np.array([lc.stat(p["mean"], card, share) for p in fd["plus"]]),
                       np.array([lc.stat(p["mean"], card, share) for p in fd["minus"]]), step)


def boot8(v, fd, post_hat, psid_means, share, step, rng, n_boot=1000) -> np.ndarray:
    """``laibson_comparison.boot_theta`` with eight unknowns: resample PSID
    couples and the simulated households (paired across points), recompute
    the Jacobian and the statistic, one Newton step from ``v``."""
    from scripts import laibson_comparison as lc

    card = post_hat["card"]
    by_card = [np.flatnonzero(card == c) for c in (0, 1)]
    out = np.empty((n_boot, len(v)))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(i, len(i)) for i in by_card])
        ip = rng.integers(0, len(psid_means), len(psid_means))
        J = lc.jacobian(np.array([lc.stat(p["mean"], card, share, idx) for p in fd["plus"]]),
                        np.array([lc.stat(p["mean"], card, share, idx) for p in fd["minus"]]),
                        step)
        out[b] = lc.newton(v, J, psid_means[ip].mean(0), lc.stat(post_hat["mean"], card,
                                                                share, idx))
    return out


def show_population(res: dict) -> None:
    def f(v, d=4):
        return " ".join(f"{x:9.{d}f}" for x in v)

    short = ("beta", "delta", "rho", "R_gamma", "s_inc", "s_cons", "s_liq", "s_illiq")
    print(f"\n{'':40s}" + "".join(f"{n:>10s}" for n in short))
    for lab, k in (("PSID couples: mean of means", "psid_mean_of_means"),
                   ("  its se", "psid_se"),
                   ("at their theta, sigma start: stat", "at_laibson_stat"),
                   ("last simulated point", "last_simulated"),
                   ("  misfit / PSID se", "last_simulated_misfit_over_psid_se"),
                   ("representative (theta, sigma)", "representative"),
                   ("  bootstrap se", "representative_se"),
                   ("  unclipped", "representative_unclipped"),
                   ("  bootstrap share beyond the box", "boot_share_beyond_box")):
        print(f"{lab:40s} {f(res[k])}")
    ci = zip(NAMES, res["representative_ci90"], strict=True)
    print("  90%: " + "  ".join(f"{n} {lo:.4f}-{hi:.4f}" for n, (lo, hi) in ci))
    print("\nd(mean theta estimate) / d(true sigma), at their theta (rows beta, delta, rho, "
          "R_gamma; columns sigma income, consumption, liquid, illiquid):")
    for name, J in res["theta_estimate_per_unit_true_sigma"].items():
        print(f"  {name}")
        for n, row in zip(oa.BOX.names, J, strict=True):
            print(f"    {n:8s}" + "".join(f"{x:+9.4f}" for x in row))


def population_figure(res: dict, arms: dict, couples: Path, path: Path,
                      title: str | None = None) -> None:
    """RESULTS 61's figure with the noise-unknown arm's representative theta
    added: its bootstrap (68/95%), from ``res``; the other arms from
    ``laibson_comparison --out``."""
    from scripts import laibson_comparison as lc

    rep = np.array(res["representative"])
    tag = res.get("tag", "unknown")
    arms = {**arms, "arms": {**arms["arms"], tag: {
        "boot": np.array(res["boot"])[:, :4].tolist(),
        "representative_theta": rep[:4].tolist(),
        "last_simulated_theta": res["last_simulated"][:4],
        "last_simulated_misfit_over_psid_se": res["last_simulated_misfit_over_psid_se"],
        "at_box_edge": res["at_box_edge"][:4],
        "box_floor_beta": res.get("box_floor_beta", 0.3),
        "at_laibson": {"mean_of_means": res["at_laibson_stat"][:4]},
        "psid_mean_of_means": res["psid_mean_of_means"][:4],
        "representative_se": res["representative_se"][:4],
        "psid_median_posterior_sd": json.loads(
            (couples / "summary.json").read_text())["couples"]["median_posterior_sd"]}}}
    mps = arms["arms"][tag]["psid_median_posterior_sd"]
    arms["arms"][tag]["psid_median_posterior_sd"] = [mps[n] for n in oa.BOX.names]
    lc.plot(arms, path, title=title or (
        "Laibson et al.'s MSM point against PSID couples' representative θ under each noise "
        "arm (RESULTS 62; unknown: θ and noise fitted together)"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="Train one member.")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--train_shards", type=int, default=None, help="Smoke tests.")
    t.add_argument("--max_epochs", type=int, default=None, help="Smoke tests.")
    t.add_argument("--device", default=None)
    t.add_argument("--out", type=Path, default=OUT)
    t.add_argument("--shards", type=Path, nargs="+", default=[MAIN, STRIP],
                   help="Training runs, combined in order (RESULTS 63: with the beta strip).")
    dev = "cuda" if torch.cuda.is_available() else None
    s = sub.add_parser("sbc", help="Calibration and recovery on v2's SBC set.")
    s.add_argument("--run", type=Path, default=OUT)
    s.add_argument("--n_draws", type=int, default=2000)
    s.add_argument("--device", default=dev)
    s.add_argument("--out", type=Path, default=OUT / "sbc")
    s.add_argument("--figure", type=Path, default=Path("figures/85_sbc_noise_unknown.png"))
    s.add_argument("--sbc", type=Path, nargs="+", default=[SBC],
                   help="SBC sets tiling the ensemble's beta range (RESULTS 63).")
    p = sub.add_parser("psid", help="PSID couples' and all 889's posteriors.")
    p.add_argument("--run", type=Path, default=OUT)
    p.add_argument("--n_draws", type=int, default=40000)
    p.add_argument("--device", default=dev)
    p.add_argument("--figure", type=Path, default=Path("figures/86_psid_noise_unknown.png"))
    p.add_argument("--root", type=Path, default=Path("outputs"),
                   help="Where the PSID posterior directories go.")
    p.add_argument("--psid_name", default=PSID_NAME,
                   help="Their names, <psid_name>_couples and _all.")
    q = sub.add_parser("population", help="Representative theta and noise, indirect inference.")
    q.add_argument("--run", type=Path, default=OUT)
    q.add_argument("--n_draws", type=int, default=4000)
    q.add_argument("--device", default=dev)
    q.add_argument("--known", nargs="*", default=["v2", "gross", "none"],
                   help="Noise-known arms whose theta estimates are scored at the same noise.")
    q.add_argument("--with_arms", type=Path,
                   default=Path("outputs/laibson_comparison/laibson_comparison_with_none.json"))
    q.add_argument("--figure", type=Path, default=Path("figures/87_population_noise_arms.png"))
    q.add_argument("--root", type=Path, default=Path("outputs"),
                   help="Where psid wrote the couples' posteriors.")
    q.add_argument("--psid_name", default=PSID_NAME)
    q.add_argument("--tag", default="unknown",
                   help="The arm's name: its Newton points' simulations and scores.")
    q.add_argument("--title", default=None, help="the figure's title")
    q.add_argument("--out", type=Path, default=POP_OUT)
    q.add_argument("--sim_cache", type=Path, default=Path("outputs/laibson_comparison"),
                   help="laibson_comparison's simulated households (reused, and added to).")
    args = ap.parse_args()
    {"train": train, "sbc": sbc, "psid": psid, "population": population}[args.cmd](args)


if __name__ == "__main__":
    main()
