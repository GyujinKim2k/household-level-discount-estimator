"""Held-out calibration of an option A2 ensemble, and the rho / R_gamma floors
on PSID (RESULTS 43.4).

**Held-out.** The two held-out shards (draws 31,744-32,767) come from the
re-aimed mixture q2 = ``EdgeMixture(**EDGE_A2)``, not the prior. Posteriors are
importance-weighted to the uniform prior as in ``optionA.psid``. Each held-out
case is weighted by p/q2, so coverage and recovery estimate prior-averaged
quantities: the same estimand as the pilot's uniform held-out table (43.2). It
is not a full SBC.

**Floors.** For PSID posteriors already written by ``optionA.py psid``, this
reports the share of households whose mean or 5th percentile sits near the
lower edges of rho (0.5) and R_gamma (1.025). It also reports held-out coverage
by posterior-mean region near those edges.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/check_optionA2_heldout.py \\
        --run_dirs outputs/optionA2/s{0..4}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.evaluation.weighted import sample_weighted
from hh_npe.npe.prior import EDGE_A2, EdgeMixture
from scripts import optionA as oa

N = oa.BOX.names
PSID = {"couples, A2": "outputs/psid_optionA2_couples", "all 889, A2": "outputs/psid_optionA2_all",
        "couples, pilot": "outputs/psid_optionA2_pilot_couples",
        "all 889, pilot": "outputs/psid_optionA2_pilot_all"}
CUTS = {"crra": [0.5, 0.75, 1.0, 1.5, 2.5, 5.0],
        "R_gamma": [1.025, 1.03, 1.04, 1.05, 1.06, 1.075]}


def per_row_log_q(ms, th, x, bt) -> np.ndarray:
    """``optionA._log_q`` per household instead of averaged."""
    dev = torch.device(getattr(ms[0], "_device", "cpu"))
    thf = oa.to_flow(th, bt)
    jac = torch.from_numpy(oa.beta_log_jacobian(th[:, 0].numpy(), oa.BOX, bt))
    with torch.no_grad():
        lp = torch.stack([m.posterior_estimator.log_prob(thf[None].to(dev),
                                                         condition=x.to(dev))[0] for m in ms])
    return ((torch.logsumexp(lp, 0) - np.log(len(ms))).cpu().double() + jac).numpy()


def heldout(r: dict, t: np.ndarray, w: np.ndarray, log_q: np.ndarray) -> dict:
    ok = np.isfinite(r["u"]).all(1) & np.isfinite(r["mean"]).all(1)
    ww = w[ok] / w[ok].sum()
    u, m, sd, tt = r["u"][ok], r["mean"][ok], r["sd"][ok], t[ok]
    s = {"n_ok": int(ok.sum()), "log_q_weighted": float((ww * log_q[ok]).sum()),
         "median_ess_frac": float(np.nanmedian(r["ess"] / np.maximum(r["n_kept"], 1)))}
    for j, n in enumerate(N):
        dm, dt = m[:, j] - (ww * m[:, j]).sum(), tt[:, j] - (ww * tt[:, j]).sum()
        s[n] = {"cov90": float((ww * ((u[:, j] >= 0.05) & (u[:, j] <= 0.95))).sum()),
                "below5": float((ww * (u[:, j] < 0.05)).sum()),
                "above95": float((ww * (u[:, j] > 0.95)).sum()),
                "corr": float((ww * dm * dt).sum()
                              / np.sqrt((ww * dm ** 2).sum() * (ww * dt ** 2).sum())),
                "mae_over_prior_sd": float((ww * np.abs(m[:, j] - tt[:, j])).sum()
                                           / oa.PRIOR_SD[j]),
                "contraction": float(1 - (ww * sd[:, j] ** 2).sum() / oa.PRIOR_SD[j] ** 2)}
    return s


def by_region(r: dict, w: np.ndarray) -> dict:
    out = {}
    for n, cuts in CUTS.items():
        j = N.index(n)
        out[n] = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            k = (r["mean"][:, j] >= a) & (r["mean"][:, j] < b)
            ww, uj = w[k] / w[k].sum(), r["u"][k, j]
            out[n].append({"lo": a, "hi": b, "n": int(k.sum()),
                           "cov90": float((ww * ((uj >= .05) & (uj <= .95))).sum()),
                           "below5": float((ww * (uj < .05)).sum()),
                           "above95": float((ww * (uj > .95)).sum())})
    return out


def floors() -> dict:
    out = {}
    for nm, p in PSID.items():
        a = np.load(f"{p}/posterior_uncorrected.npz")
        m, lo = a["mean"], a["lo"]
        rho, rg = lo[:, 2] < 0.55, lo[:, 3] < 1.0275
        out[nm] = {k: float(v) for k, v in {
            "rho_mean_lt_1": (m[:, 2] < 1.0).mean(), "rho_lo_lt_0.55": rho.mean(),
            "R_mean_lt_1.030": (m[:, 3] < 1.030).mean(), "R_lo_lt_1.0275": rg.mean(),
            "either": (rho | rg).mean(), "both": (rho & rg).mean(),
            "corr_rho_R_means": np.corrcoef(m[:, 2], m[:, 3])[0, 1]}.items()}
    a = np.load(f"{PSID['couples, A2']}/posterior_uncorrected.npz")["mean"]
    b = np.load(f"{PSID['couples, pilot']}/posterior_uncorrected.npz")["mean"]
    out["corr_couples_A2_vs_pilot"] = {n: float(np.corrcoef(a[:, j], b[:, j])[0, 1])
                                       for j, n in enumerate(N)}
    out["couples_A2_share_in_region"] = float(((a[:, 0] >= EDGE_A2["beta_lo"])
                                               & (a[:, 1] >= 1 - EDGE_A2["one_minus_delta_hi"]))
                                              .mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--shards", type=Path, default=Path("data/processed/couples_dataset_shards"))
    ap.add_argument("--n_draws", type=int, default=4000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("outputs/optionA2/heldout_check.json"))
    args = ap.parse_args()

    members, proposal, cfg = oa.load_members(args.run_dirs, args.device)
    bt = cfg["beta_transform"]
    inv, lw = oa._weighting(proposal, bt)
    fb = oa.flow_box(bt)
    theta_all, _card, _tr, held_f, _p, _cfg = oa.load_shards(args.shards)
    th, x, pid, _f = oa.windows(held_f, theta_all, seed=oa.HELD_SEED)
    pick = oa.one_per_draw(pid, seed=0)
    th, x = th[pick], x[pick]
    t = th.numpy().astype(float)
    w = np.exp(EdgeMixture(**EDGE_A2).log_weight(t))
    print(f"{len(members)} members; held-out {len(t)} draws; weights p/q2 in "
          f"[{w.min():.3f}, {w.max():.3f}], ESS {w.sum() ** 2 / (w ** 2).sum():.0f}")

    res = {"run_dirs": list(map(str, args.run_dirs)), "n_draws": args.n_draws}
    torch.manual_seed(0)
    sets = [("ensemble", members)] + [(f"member_{i}", [m]) for i, m in enumerate(members)]
    for name, ms in sets:
        r = sample_weighted(ms, x, args.n_draws, fb.low, fb.high, invert=inv, log_weight=lw,
                            truth=t)
        res[name] = heldout(r, t, w, per_row_log_q(ms, th, x, bt))
        if name == "ensemble":
            res["ensemble_by_posterior_mean"] = by_region(r, w)
        s = res[name]
        print(f"\n=== {name}: weighted held-out log q {s['log_q_weighted']:.3f} ===")
        print(f"{'':9s}{'cov90':>7s}{'<5%':>7s}{'>95%':>7s}{'corr':>7s}{'mae/sd':>8s}{'contr':>7s}")
        for n in N:
            v = s[n]
            print(f"{n:9s}{v['cov90']:7.3f}{v['below5']:7.3f}{v['above95']:7.3f}{v['corr']:7.3f}"
                  f"{v['mae_over_prior_sd']:8.3f}{v['contraction']:7.3f}")
    for n, rows in res["ensemble_by_posterior_mean"].items():
        print(f"\nensemble, held-out 90% coverage by posterior-mean {n}:")
        for b in rows:
            print(f"  [{b['lo']}, {b['hi']}) n={b['n']:4d}  cov90 {b['cov90']:.3f}  "
                  f"<5% {b['below5']:.3f}  >95% {b['above95']:.3f}")
    res["psid_floors"] = floors()
    print("\nPSID floors:")
    for k, v in res["psid_floors"].items():
        print(f"  {k}: " + (json.dumps({a: round(b, 3) for a, b in v.items()})
                            if isinstance(v, dict) else f"{v:.3f}"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
