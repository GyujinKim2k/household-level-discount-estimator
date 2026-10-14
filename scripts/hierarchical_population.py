"""A population (hierarchical) layer over A2n's PSID couples (RESULTS 45).

One couple's A2n posterior for beta or rho is about as wide as the spread of
all couples' posterior means, so the spread alone cannot say whether couples
differ or whether the estimates only reproduce a single representative theta.
This fits the population distribution g of (beta, delta, rho, R_gamma) across
the 409 couples from their posteriors (``hh_npe.evaluation.population``), and
uses it as each couple's prior (empirical Bayes):

1. **Draws.** ``n_draws`` per couple from the five-member ensemble,
   importance-weighted to the uniform prior exactly as ``optionA.psid``
   (``weighted.sample_draws``), and the ensemble's density at each draw.
2. **Fit** (``population.fit_adaptive``). A first fit recycles the draws; then
   ``rounds`` times, ``n_ref`` reference points per couple are drawn around
   the current fit, the likelihood and the ensemble's density are evaluated
   at them, and the population is refitted on both blocks (multiple
   importance sampling). Recycling alone leaves a floor of ~0.4 of a
   posterior's width on a spread that is truly zero; the reference points
   remove it.
3. **Families.** M1, one normal in each parameter's box-logit with full
   covariance (headline); M1d, diagonal; M2, two normals. All on M1's final
   points.
4. **Uncertainty.** ``n_boot`` bootstrap refits of M1 over couples, and the
   observed information (Hessian).
5. **Households.** Each couple's posterior unpooled (uniform prior) and pooled
   (prior g), with the ESS of its weights.

Whether a fitted spread is real is decided against simulated couples who all
share one theta, run through this same pipeline
(``scripts/hierarchical_validation.py``): a spread is heterogeneity only if it
exceeds that null's 95th percentile.

The ensemble's density is the flow's own (sbi's ``log_prob`` re-embeds each
couple's data once per point, which overflows the GPU at these sizes, so each
couple is embedded once), in theta: the flow's density in its coordinates
(logit beta and R_gamma, log(1 - delta), rho) times their Jacobian.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/hierarchical_population.py fit --arm uncorrected
    PYTHONPATH=. .venv/bin/python scripts/hierarchical_population.py plot

``--variant targeted`` (RESULTS 46) uses the summary likelihood retrained with
the targeted couples (``hierarchical_summary.py targeted``) and writes under
``summary_targeted``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from hh_npe.evaluation import population as pop
from hh_npe.evaluation.weighted import resample, sample_draws
from scripts import optionA as oa
from scripts.noise_calibration import PSID_X

RUNS = [Path(f"outputs/optionA2n/s{i}") for i in range(5)]
OUT = Path("outputs/hier_optionA2n_couples")
FIG = Path("figures/52_psid_population_optionA2n.png")
N_DRAWS, N_REF, ROUNDS, N_BOOT = 4000, 4096, 2, 200
LAIBSON = np.array([0.5305, 0.9891, 1.9355, 1.05])
FAMILIES = {"M1": pop.Spec(d=4), "M1d": pop.Spec(d=4, diagonal=True), "M2": pop.Spec(d=4, k=2)}
#: The summary likelihood's density model by variant (RESULTS 45, 46).
MDN = {"base": Path("outputs/hier_summary/mdn.pt"),
       "targeted": Path("outputs/hier_summary/targeted/mdn.pt")}


def run_name(likelihood: str, variant: str = "base") -> str:
    """Where fits on a likelihood (and summary-likelihood variant) go, under
    ``OUT``."""
    return likelihood if variant == "base" else f"{likelihood}_{variant}"


def families(lo, hi) -> dict[str, pop.Spec]:
    """The families, with TN (independent truncated normals in theta) on the
    box ``lo``-``hi``."""
    return {**FAMILIES, "TN": pop.Spec(d=4, diagonal=True,
                                       theta_box=(tuple(map(float, lo)), tuple(map(float, hi))))}


class Network:
    """The A2n ensemble as the population layer needs it: weighted draws for a
    set of households, and its density at any theta for each of them."""

    def __init__(self, run_dirs=RUNS, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.members, self.proposal, self.cfg = oa.load_members(run_dirs, self.device)
        self.spec = oa.spec_of(self.cfg)
        self.box = self.spec.box
        self.lo, self.hi = np.asarray(self.box.low, float), np.asarray(self.box.high, float)
        self.inv, self.log_weight = oa._weighting(self.proposal, self.spec)
        self.fb = self.spec.flow_box()
        self.di = list(self.box.names).index("delta")

    def log_jacobian(self, theta: np.ndarray) -> np.ndarray:
        """``log |dt / dtheta|`` from theta to the flow's coordinates."""
        return self.spec.log_jacobian(theta) - np.log1p(-theta[:, self.di])

    @torch.no_grad()
    def log_density(self, xt: torch.Tensor, theta: np.ndarray, chunk: int = 256,
                    hh_batch: int = 512) -> np.ndarray:
        """The ensemble's density in theta, ``(B, M)``, at ``theta`` of shape
        ``(B, M, d)`` (per household) or ``(M, d)`` (shared). NaN rows (padding)
        give NaN. Untruncated: the flow's own density, not renormalised to the
        box."""
        th = np.asarray(theta, dtype=float)
        shared = th.ndim == 2
        B, d = len(xt), th.shape[-1]
        M = th.shape[0] if shared else th.shape[1]
        out = np.full((B, M), np.nan)
        centre = (self.lo + self.hi) / 2
        for h0 in range(0, B, hh_batch):
            xb = xt[h0:h0 + hh_batch].to(self.device)
            nb = len(xb)
            embs = [m.posterior_estimator.net._embedding_net(xb) for m in self.members]
            for m0 in range(0, M, chunk):
                tc = th[m0:m0 + chunk] if shared else th[h0:h0 + nb, m0:m0 + chunk]
                tc = np.broadcast_to(tc, (nb,) + tc.shape) if shared else tc   # (nb, mc, d)
                ok = np.isfinite(tc).all(-1)
                tc = np.where(ok[..., None], tc, centre)
                flat = tc.reshape(-1, d)
                jac = self.log_jacobian(flat).reshape(ok.shape)
                t = torch.as_tensor(self.spec.to_flow(flat), dtype=torch.float32,
                                    device=self.device).reshape(nb, -1, d)
                t = t.transpose(0, 1).reshape(-1, d)             # (mc * nb, d), m-major
                lps = []
                for m, e in zip(self.members, embs, strict=True):
                    net = m.posterior_estimator.net
                    ctx = e.repeat(len(t) // nb, 1)
                    noise, lad = net._transform(t, context=ctx)
                    lps.append(net._distribution.log_prob(noise, context=ctx) + lad)
                lp = (torch.logsumexp(torch.stack(lps), 0) - np.log(len(lps)))
                lp = lp.reshape(-1, nb).T.double().cpu().numpy() + jac
                out[h0:h0 + nb, m0:m0 + lp.shape[1]] = np.where(ok, lp, np.nan)
        return out

    def points(self, xt: torch.Tensor, n_draws: int, seed: int) -> dict:
        """Each household's weighted draws and the ensemble's density at them."""
        torch.manual_seed(seed)
        d = sample_draws(self.members, xt, n_draws, self.fb.low, self.fb.high,
                         invert=self.inv, log_weight=self.log_weight)
        d["log_qf"] = self.log_density(xt, d["theta"])
        return d

    def evaluator(self, xt: torch.Tensor, log_z: np.ndarray):
        """``evaluate(theta_ref) -> (log_l, log_q)`` for ``population.fit_adaptive``
        at each household's own points ``theta_ref`` (N, R, d): the likelihood,
        ``q / p~`` (``log_weight`` is ``log p / p~``, p constant), and the
        density the household's draws came from, ``q`` on the box (``log_z``:
        log in-box fraction)."""
        def evaluate(th_ref):
            lq = self.log_density(xt, th_ref)
            plw = self.log_weight(th_ref.reshape(-1, th_ref.shape[-1])).reshape(lq.shape)
            return lq + plw, lq - log_z[:, None]
        return evaluate


@torch.no_grad()
def summary_log_l(mdn, s: np.ndarray, theta: np.ndarray, lo, hi, device,
                  chunk: int = 64) -> np.ndarray:
    """``log p(s_i | theta)`` (N, M) at each household's points ``theta``
    (N, M, d); NaN rows give NaN."""
    th = np.asarray(theta, dtype=float)
    ok = np.isfinite(th).all(-1)
    span = hi - lo
    thc = np.clip(np.where(ok[..., None], th, (lo + hi) / 2), lo + 1e-7 * span, hi - 1e-7 * span)
    u = pop.box_logit(thc, lo, hi)
    mdn = mdn.to(device)
    out = np.empty(ok.shape)
    for c0 in range(0, len(th), chunk):
        uu = torch.from_numpy(u[c0:c0 + chunk]).to(device)
        ss = torch.from_numpy(np.asarray(s[c0:c0 + chunk], float)).to(device)[:, None, :]
        out[c0:c0 + chunk] = mdn.log_prob(ss, uu).cpu().numpy()
    return np.where(ok, out, np.nan)


def fit_population(net: Network, xt: torch.Tensor, n_draws=N_DRAWS, n_ref=N_REF,
                   rounds=ROUNDS, seed: int = 0, families_=("M1",), mdn=None) -> dict:
    """The pipeline of the module docstring on households ``xt``; returns the
    points, M1's parameters and the families' fits (no bootstrap).

    ``mdn``: the summary likelihood ``p(s | theta)`` (``hierarchical_summary``),
    with ``s`` each household's posterior mean in ``u`` from these same draws.
    ``None``: the network's own ``q / p~`` (RESULTS 45: fails the uniform
    test for beta). Either way the network's draws, with its density, are the
    importance-sampling proposal alongside the reference points."""
    t0 = time.time()
    d = net.points(xt, n_draws, seed)
    ok = d["n_kept"] > 0
    if not ok.all():
        raise SystemExit(f"{int((~ok).sum())} households have no draws in the box")
    log_z = np.log(d["in_box_frac"])
    log_q = d["log_qf"] - log_z[:, None]
    if mdn is None:
        log_l = d["log_qf"] + d["log_w"]
        evaluate, broad = net.evaluator(xt, log_z), None
    else:
        s = pop.posterior_mean_u(d["theta"], d["log_w"], net.lo, net.hi)
        d["summary"] = s
        log_l = summary_log_l(mdn, s, d["theta"], net.lo, net.hi, net.device)

        def evaluate(th_ref):
            return (summary_log_l(mdn, s, th_ref, net.lo, net.hi, net.device),
                    net.log_density(xt, th_ref) - log_z[:, None])
        # The summary likelihood sits further out than the shrunk posterior,
        # so the reference points always keep a component over the whole box.
        broad = 2.0
    t1 = time.time()
    p, ll, hd, hist = pop.fit_adaptive(d["theta"], log_l, log_q, evaluate,
                                       net.lo, net.hi, FAMILIES["M1"], n_ref=n_ref,
                                       rounds=rounds, seed=seed, device=net.device,
                                       broad_sd=broad)
    fits = {"M1": (p, ll)}
    specs = families(net.lo, net.hi)
    for f in families_:
        if f != "M1":
            fits[f] = pop.fit(hd, specs[f])
    print(f"  {len(xt)} households: draws {t1 - t0:.0f}s, fit {time.time() - t1:.0f}s",
          flush=True)
    return {"draws": d, "hd": hd, "fits": fits, "history": hist}


def natural(spec: pop.Spec, p: dict, lo, hi, n: int = 200_000, seed: int = 0) -> dict:
    """The population in natural units, by sampling it."""
    th = pop.from_box_logit(pop.sample_u(spec, p, n, seed), lo, hi)
    return {"q05_q25_q50_q75_q95": np.percentile(th, [5, 25, 50, 75, 95], axis=0).T.tolist(),
            "mean": th.mean(0).tolist(), "sd": th.std(0).tolist()}


def household_table(r0: dict, r1: dict, spec_prior_sd, names) -> dict:
    """Unpooled against pooled: the numbers RESULTS 45 opens with."""
    out = {}
    for j, n in enumerate(names):
        m0, m1 = r0["mean"][:, j], r1["mean"][:, j]
        v0, v1 = r0["sd"][:, j] ** 2, r1["sd"][:, j] ** 2
        out[n] = {"median_sd": [float(np.median(r0["sd"][:, j])), float(np.median(r1["sd"][:, j]))],
                  "width90_over_range_of_means": [
                      float(np.median(r["hi"][:, j] - r["lo"][:, j])
                            / np.subtract(*np.percentile(r["mean"][:, j], [95, 5])))
                      for r in (r0, r1)],
                  "contraction": [float(1 - v.mean() / spec_prior_sd[j] ** 2) for v in (v0, v1)],
                  "reliability": [float(m.var() / (m.var() + v.mean()))
                                  for m, v in ((m0, v0), (m1, v1))]}
    return out


def psid_x(path: Path, arm: str) -> tuple[torch.Tensor, np.ndarray]:
    """PSID couples' input as ``optionA.psid`` builds it, for one arm."""
    from hh_npe.simulator.laibson_calibration import EDUC_GROUPS
    from scripts.psid_posterior import graded_correction

    d = torch.load(path, weights_only=False)
    keep = np.flatnonzero(d["educ"].numpy() == EDUC_GROUPS.index("comphs"))
    x = d["x"].numpy()[keep]
    if arm == "corrected":
        x = graded_correction(x, cons_idx=FEATURES_TWOASSET_AGE.index("consumption"))
    xt, _ = oa.transform(torch.from_numpy(x).float())
    return xt, keep


def load_mdn(path: Path, n_draws: int):
    from hh_npe.evaluation import summary_likelihood as sl

    mdn, info = sl.load(path)
    if info["n_draws"] != n_draws:
        raise SystemExit(f"{path} was trained on summaries from {info['n_draws']} draws; "
                         f"these use {n_draws}")
    return mdn


def cmd_fit(args) -> None:
    net = Network(args.run_dirs)
    xt, keep = psid_x(args.x, args.arm)
    if args.max_households:                     # smoke runs only
        xt, keep = xt[:args.max_households], keep[:args.max_households]
    out = args.out / run_name(args.likelihood, args.variant) / args.arm
    out.mkdir(parents=True, exist_ok=True)
    args.mdn = args.mdn or MDN[args.variant]
    mdn = load_mdn(args.mdn, args.n_draws) if args.likelihood == "summary" else None
    print(f"{args.x} ({args.arm}, {args.likelihood} likelihood): {len(keep)} couples; "
          f"{net.spec.describe()}")
    specs = families(net.lo, net.hi)
    res = fit_population(net, xt, args.n_draws, args.n_ref, args.rounds, args.seed,
                         families_=tuple(specs), mdn=mdn)
    hd, names = res["hd"], list(net.box.names)
    dn = pop.derived_names(names)
    summary = {"x": str(args.x), "arm": args.arm, "likelihood": args.likelihood,
               "variant": args.variant, "mdn": str(args.mdn) if mdn is not None else None,
               "run_dirs": list(map(str, args.run_dirs)),
               "n_households": int(len(keep)), "n_draws": args.n_draws, "n_ref": args.n_ref,
               "rounds": args.rounds, "box": [net.lo.tolist(), net.hi.tolist()],
               "history": res["history"], "families": {}}
    for f, (p, ll) in res["fits"].items():
        spec = specs[f]
        k = sum(p[n].numel() for n in spec.free())
        summary["families"][f] = {
            "log_marginal": ll, "n_params": int(k), "bic": float(-2 * ll + k * np.log(len(hd))),
            "derived": dict(zip(dn, pop.derived(spec, p).numpy().tolist(), strict=True)),
            "natural": natural(spec, p, net.lo, net.hi),
            "params": {n: v.numpy().tolist() for n, v in p.items()}}
        if spec.k > 1:
            summary["families"][f]["weights"] = torch.softmax(p["logit_pi"], 0).numpy().tolist()
    m1, spec1 = res["fits"]["M1"][0], FAMILIES["M1"]
    cov = pop.hessian_cov(spec1, m1, hd)
    summary["families"]["M1"]["hessian_se"] = dict(zip(dn, np.sqrt(np.clip(np.diag(cov), 0, None))
                                                       .tolist(), strict=True))
    t0 = time.time()
    boot = pop.bootstrap(hd, spec1, m1, args.n_boot, seed=args.seed)
    sd_nat = np.array([natural(spec1, _params_from_derived(spec1, m1, b), net.lo, net.hi,
                               n=20_000)["sd"] for b in boot]) if args.n_boot else None
    print(f"  bootstrap {args.n_boot}: {time.time() - t0:.0f}s")
    if args.n_boot:
        summary["families"]["M1"]["bootstrap"] = {
            "n": args.n_boot,
            "derived_q05_q95": dict(zip(dn, np.percentile(boot, [5, 95], 0).T.tolist(),
                                        strict=True)),
            "natural_sd_q05_q95": dict(zip(names, np.percentile(sd_nat, [5, 95], 0).T.tolist(),
                                           strict=True))}
        np.save(out / "bootstrap_derived.npy", boot)
    r0, r1 = pop.pooled(hd), pop.pooled(hd, spec1, m1)
    summary["households"] = household_table(r0, r1, net.spec.prior_sd, names)
    summary["pooled_ess"] = {"min": float(r1["ess"].min()),
                             "p10": float(np.percentile(r1["ess"], 10)),
                             "median": float(np.median(r1["ess"])),
                             "below_50": int((r1["ess"] < 50).sum())}
    np.savez(out / "households.npz", household_index=keep, names=np.array(names),
             summary=res["draws"].get("summary", np.full((len(keep), 4), np.nan)),
             **{f"unpooled_{k}": v for k, v in r0.items()},
             **{f"pooled_{k}": v for k, v in r1.items()})
    torch.save({"theta": hd.theta, "log_w": hd.log_w, "b": hd.b.cpu(), "u": hd.u.cpu(),
                "lo": hd.lo, "hi": hd.hi}, out / "points.pt")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    report(summary)
    print(f"\nwrote {out}")


def _params_from_derived(spec: pop.Spec, base: dict, der: np.ndarray) -> dict:
    """M1 parameters with the mean, sds and correlations of ``der``."""
    d = spec.d
    m, logsd = der[:d], der[d:2 * d]
    corr = np.eye(d)
    i, j = np.tril_indices(d, -1)
    corr[i, j] = corr[j, i] = der[2 * d:]
    cov = corr * np.exp(logsd)[:, None] * np.exp(logsd)[None, :]
    w, v = np.linalg.eigh(cov)
    L = np.linalg.cholesky(v @ np.diag(np.clip(w, 1e-12, None)) @ v.T)
    return {**base, "mu": torch.tensor(m[None]), "log_diag": torch.tensor(np.log(np.diag(L))[None]),
            "off": torch.tensor(L[np.tril_indices(d, -1)][None])}


def report(s: dict) -> None:
    names = ["beta", "delta", "crra", "R_gamma"]
    for f, r in s["families"].items():
        print(f"\n--- {f}: sum log m {r['log_marginal']:.2f}, BIC {r['bic']:.1f}"
              + (f", weights {np.round(r['weights'], 3).tolist()}" if "weights" in r else ""))
        nat = r["natural"]
        for j, n in enumerate(names):
            q = nat["q05_q25_q50_q75_q95"][j]
            line = (f"  {n:8s} mean_u {r['derived'][f'mean_u {n}']:+.3f}  sd_u "
                    f"{np.exp(r['derived'][f'log_sd_u {n}']):.3f}  natural: median {q[2]:.4f}"
                    f" (5-95% {q[0]:.4f}-{q[4]:.4f}), sd {nat['sd'][j]:.4f}")
            if f == "M1" and "bootstrap" in r:
                lo_, hi_ = r["bootstrap"]["natural_sd_q05_q95"][n]
                line += f" [boot 90% {lo_:.4f}-{hi_:.4f}]"
            print(line)
        if f == "M1":
            print("  correlations: " + "  ".join(
                f"{k[5:]} {v:+.2f}" for k, v in r["derived"].items() if k.startswith("corr")))
    print("\nhouseholds, unpooled -> pooled:")
    for n, h in s["households"].items():
        print(f"  {n:8s} median sd {h['median_sd'][0]:.4f} -> {h['median_sd'][1]:.4f}; 90% width "
              f"/ range of means {h['width90_over_range_of_means'][0]:.2f} -> "
              f"{h['width90_over_range_of_means'][1]:.2f}; contraction "
              f"{h['contraction'][0]:.2f} -> {h['contraction'][1]:.2f}; reliability "
              f"{h['reliability'][0]:.2f} -> {h['reliability'][1]:.2f}")
    print(f"pooled ESS: {s['pooled_ess']}")
    print("adaptive rounds (M1 sd_u; share of reference points clipped): " + "; ".join(
        f"round {h['round']}: " + " ".join(f"{np.exp(v):.3f}" for v in h["derived"][4:8])
        + f" ({h.get('ref_clipped', 0.0):.3f})" for h in s["history"]))


def cmd_ppc(args) -> None:
    """Posterior predictive check of the fitted population (summary
    likelihood): theta ~ g, s ~ p(s | theta), against PSID couples' own
    summaries, per coordinate (PIT under the predictive marginal, KS), and
    the share of PSID summaries beyond what simulated households ever reach
    (``hierarchical_summary`` pairs)."""
    from scipy import stats

    from scripts import hierarchical_summary as hs

    out = args.out / run_name("summary", args.variant) / args.arm
    s = json.loads((out / "summary.json").read_text())
    h = np.load(out / "households.npz")
    mdn = load_mdn(args.mdn or MDN[args.variant], s["n_draws"])
    spec = FAMILIES["M1"]
    p = {k: torch.tensor(v, dtype=torch.float64) for k, v in s["families"]["M1"]["params"].items()}
    u = torch.from_numpy(pop.sample_u(spec, p, args.n_sim, seed=1))
    sim = mdn.sample(u, generator=torch.Generator().manual_seed(2)).numpy()
    obs = h["summary"]
    pairs_s = hs.training_pairs(hs.OUT, args.variant)[0]
    lo_s, hi_s = np.percentile(pairs_s, [0.1, 99.9], axis=0)
    names = list(h["names"])
    res = {}
    for j, n in enumerate(names):
        pit = (sim[:, j][None] <= obs[:, j][:, None]).mean(1)
        res[n] = {"ks_p": float(stats.ks_2samp(obs[:, j], sim[:, j]).pvalue),
                  "pit_deciles": np.histogram(pit, bins=10, range=(0, 1))[0].tolist(),
                  "psid_q05_q50_q95": np.percentile(obs[:, j], [5, 50, 95]).tolist(),
                  "predicted_q05_q50_q95": np.percentile(sim[:, j], [5, 50, 95]).tolist(),
                  "share_beyond_simulated": float(((obs[:, j] < lo_s[j])
                                                   | (obs[:, j] > hi_s[j])).mean())}
    (out / "ppc.json").write_text(json.dumps(res, indent=2))
    print(f"posterior predictive check ({args.arm}): PSID summaries s (posterior means in u) "
          "against theta ~ g, s ~ p(s | theta)")
    for n, r in res.items():
        print(f"  {n:8s} KS p {r['ks_p']:.3f}; PSID 5/50/95 "
              + " ".join(f"{v:+.2f}" for v in r["psid_q05_q50_q95"]) + "; predicted "
              + " ".join(f"{v:+.2f}" for v in r["predicted_q05_q50_q95"])
              + f"; PIT deciles {r['pit_deciles']}; beyond simulated "
              f"{r['share_beyond_simulated']:.3f}")


def cmd_plot(args) -> None:
    from hh_npe.evaluation.plots import contour_corner

    out = args.out / run_name(args.likelihood, args.variant) / args.arm
    s = json.loads((out / "summary.json").read_text())
    h = np.load(out / "households.npz")
    pts = torch.load(out / "points.pt", weights_only=False)
    lo, hi = np.asarray(pts["lo"]), np.asarray(pts["hi"])
    spec = FAMILIES["M1"]
    p = {k: torch.tensor(v, dtype=torch.float64) for k, v in s["families"]["M1"]["params"].items()}
    g = pop.from_box_logit(pop.sample_u(spec, p, 20_000), lo, hi)
    # Example couple: the one whose unpooled beta sd is the median.
    sd = h["unpooled_sd"][:, 0]
    i = int(np.argsort(sd)[len(sd) // 2])
    th = pts["theta"][i].astype(float)
    with torch.no_grad():
        lw_pooled = (pts["b"][i] + pop.log_density(spec, p, pts["u"][i])).numpy()
    lw_un = pts["log_w"][i]
    ok_u, ok_p = np.isfinite(lw_un), np.isfinite(lw_pooled)
    names = list(h["names"])
    box = oa.box_of({"rgamma_range": [lo[3], hi[3]]})
    contour_corner(
        {"population g (M1)": g,
         f"couples' unpooled posterior means (N={len(h['unpooled_mean'])})": h["unpooled_mean"],
         "one couple, unpooled": resample(th[ok_u], lw_un[ok_u], 20_000),
         "the same couple, pooled (prior g)": resample(th[ok_p], lw_pooled[ok_p], 20_000)},
        box, truth={"Laibson et al. MSM (R_gamma: calibrated 1.05)": LAIBSON},
        truth_markers=("*",), truth_colors=("#d62728",), reflect_axes=("delta",),
        path=args.fig,
        title=(f"PSID couples under A2n ({args.arm}"
               + ("" if args.variant == "base" else f", {args.variant} likelihood")
               + "): the fitted population against one couple"
               f" ({names[0]} sd at the median), unpooled and pooled"))
    print(f"wrote {args.fig}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--run_dirs", type=Path, nargs="+", default=RUNS)
    f.add_argument("--x", type=Path, default=PSID_X)
    f.add_argument("--arm", choices=("uncorrected", "corrected"), default="uncorrected")
    f.add_argument("--likelihood", choices=("summary", "npe"), default="summary",
                   help="summary: p(posterior mean | theta) learned from simulation "
                        "(hierarchical_summary.py); npe: the network's own q / p~.")
    f.add_argument("--variant", choices=tuple(MDN), default="base",
                   help="The summary likelihood: RESULTS 45's (base) or retrained with the "
                        "targeted couples (RESULTS 46).")
    f.add_argument("--mdn", type=Path, default=None, help="Default: the variant's.")
    f.add_argument("--n_draws", type=int, default=N_DRAWS)
    f.add_argument("--n_ref", type=int, default=N_REF)
    f.add_argument("--rounds", type=int, default=ROUNDS)
    f.add_argument("--n_boot", type=int, default=N_BOOT)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--max_households", type=int, default=None, help="Smoke runs only.")
    f.add_argument("--out", type=Path, default=OUT)
    pc = sub.add_parser("ppc")
    pc.add_argument("--arm", choices=("uncorrected", "corrected"), default="uncorrected")
    pc.add_argument("--variant", choices=tuple(MDN), default="base")
    pc.add_argument("--mdn", type=Path, default=None, help="Default: the variant's.")
    pc.add_argument("--n_sim", type=int, default=200_000)
    pc.add_argument("--out", type=Path, default=OUT)
    pl = sub.add_parser("plot")
    pl.add_argument("--arm", choices=("uncorrected", "corrected"), default="uncorrected")
    pl.add_argument("--likelihood", choices=("summary", "npe"), default="summary")
    pl.add_argument("--variant", choices=tuple(MDN), default="base")
    pl.add_argument("--out", type=Path, default=OUT)
    pl.add_argument("--fig", type=Path, default=FIG)
    args = ap.parse_args()
    {"fit": cmd_fit, "ppc": cmd_ppc, "plot": cmd_plot}[args.cmd](args)


if __name__ == "__main__":
    main()
