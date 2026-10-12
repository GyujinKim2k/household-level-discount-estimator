"""Held-out calibration of an option A2 ensemble, and the rho / R_gamma floors
on PSID (RESULTS 43.4, 43.5).

**Held-out.** The held-out shards are the main run's last two (draws
31,744-32,767) and, for an ensemble trained with the R_gamma strip, the
strip's last one. All of them come from the re-aimed mixture q2 =
``EdgeMixture(**EDGE_A2)``, with R_gamma uniform on their run's range, not
from the prior. Posteriors are importance-weighted to the uniform prior as in
``optionA.psid``. Each held-out case is weighted by p / q_held, where q_held is
the held-out set's own mixture, so coverage and recovery estimate
prior-averaged quantities. That is the same estimand as the pilot's uniform
held-out table (43.2). It is not a full SBC.

**Floors.** For PSID posteriors already written by ``optionA.py psid``, this
reports the share of households whose mean or 5th percentile sits near the
lower edges of rho (0.5) and R_gamma (each run's own floor, from its
summary.json). It also reports held-out coverage by posterior-mean region.

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
from hh_npe.npe.prior import proposal_log_weight
from scripts import optionA as oa

N = oa.BOX.names
PSID = ["couples, A2=outputs/psid_optionA2_couples", "all 889, A2=outputs/psid_optionA2_all",
        "couples, pilot=outputs/psid_optionA2_pilot_couples",
        "all 889, pilot=outputs/psid_optionA2_pilot_all"]
#: Held-out draws of these runs all come from the A2 mixture (a2_switched's
#: draws past its 4,096-draw pilot).
HELD_DENSITY = {"a2_switched": "a2_strip", "a2_strip": "a2_strip"}
RHO_NEAR, R_NEAR = 0.55, 0.0025


def heldout_log_weight(cfg: dict, pid: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """log p / q_held for held-out rows (draw ids ``pid``)."""
    runs = cfg.get("runs") or [{"proposal": cfg["proposal"], "rgamma_range": cfg["rgamma_range"],
                                "n_draws": int(pid.max()) + 1}]
    sizes = [r["n_draws"] for r in runs]
    offsets = np.cumsum([0] + sizes[:-1])
    parts = []
    for r, off, size in zip(runs, offsets, sizes):
        if r["proposal"] not in HELD_DENSITY:
            raise SystemExit(f"held-out draws of {r['proposal']!r} not described here")
        k = int(((pid >= off) & (pid < off + size)).sum())
        if k:
            parts.append({"name": HELD_DENSITY[r["proposal"]], "n_train": k,
                          "rgamma_range": r["rgamma_range"]})
    if [p["rgamma_range"] for p in parts] != [r["rgamma_range"] for r in runs]:
        raise SystemExit("a run has no held-out rows")
    return proposal_log_weight({"parts": parts}, theta)


def heldout(r: dict, t: np.ndarray, w: np.ndarray, log_q: np.ndarray, spec) -> dict:
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
                                           / spec.prior_sd[j]),
                "contraction": float(1 - (ww * sd[:, j] ** 2).sum() / spec.prior_sd[j] ** 2)}
    return s


def by_region(r: dict, w: np.ndarray, spec) -> dict:
    lo = spec.box.rgamma_low
    cuts = {"crra": [0.5, 0.75, 1.0, 1.5, 2.5, 5.0],
            "R_gamma": sorted({lo, *[c for c in (1.025, 1.03, 1.04, 1.05, 1.06, 1.075)
                                     if c > lo]})}
    out = {}
    for n, c in cuts.items():
        j = N.index(n)
        out[n] = []
        for a, b in zip(c[:-1], c[1:]):
            k = (r["mean"][:, j] >= a) & (r["mean"][:, j] < b)
            if not k.any():
                out[n].append({"lo": a, "hi": b, "n": 0})
                continue
            ww, uj = w[k] / w[k].sum(), r["u"][k, j]
            out[n].append({"lo": a, "hi": b, "n": int(k.sum()),
                           "cov90": float((ww * ((uj >= .05) & (uj <= .95))).sum()),
                           "below5": float((ww * (uj < .05)).sum()),
                           "above95": float((ww * (uj > .95)).sum())})
    return out


def floors(entries: list[str]) -> dict:
    out, means = {}, {}
    for e in entries:
        nm, p = e.split("=", 1)
        a = np.load(f"{p}/posterior_uncorrected.npz")
        summ = json.loads((Path(p) / "summary.json").read_text())
        r_floor = (summ.get("rgamma_range") or [oa.BOX.rgamma_low])[0]
        m, lo = a["mean"], a["lo"]
        rho, rg = lo[:, 2] < RHO_NEAR, lo[:, 3] < r_floor + R_NEAR
        means[nm] = m
        out[nm] = {k: float(v) for k, v in {
            "R_gamma_floor": r_floor,
            "rho_mean_lt_1": (m[:, 2] < 1.0).mean(), f"rho_lo_lt_{RHO_NEAR}": rho.mean(),
            "R_mean_lt_floor+0.005": (m[:, 3] < r_floor + 0.005).mean(),
            "R_lo_lt_floor+0.0025": rg.mean(),
            "R_mean_lt_1.025": (m[:, 3] < 1.025).mean(),
            "either": (rho | rg).mean(), "both": (rho & rg).mean(),
            "corr_rho_R_means": np.corrcoef(m[:, 2], m[:, 3])[0, 1]}.items()}
    first = next(iter(means))
    out["corr_with_" + first] = {
        nm: {n: float(np.corrcoef(means[first][:, j], m[:, j])[0, 1]) for j, n in enumerate(N)}
        for nm, m in means.items() if nm != first and len(m) == len(means[first])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--shards", type=Path, nargs="+",
                    default=[Path("data/processed/couples_dataset_shards")],
                    help="The ensemble's shard directories, in training order.")
    ap.add_argument("--psid", nargs="+", default=PSID,
                    help="NAME=DIR of `optionA.py psid` outputs; the first is the "
                         "reference for the correlations.")
    ap.add_argument("--n_draws", type=int, default=4000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("outputs/optionA2/heldout_check.json"))
    args = ap.parse_args()

    members, proposal, cfg = oa.load_members(args.run_dirs, args.device)
    spec = oa.spec_of(cfg)
    inv, lw = oa._weighting(proposal, spec)
    fb = spec.flow_box()
    theta_all, _card, _tr, held_f, _p, sh_cfg = oa.load_shards(args.shards)
    if oa.box_of(sh_cfg) != spec.box:
        raise SystemExit(f"{args.shards} cover R_gamma {sh_cfg['rgamma_range']}, not the "
                         f"ensemble's box")
    th, x, pid, _f = oa.windows(held_f, theta_all, seed=oa.HELD_SEED, noise=spec.obs_noise)
    pick = oa.one_per_draw(pid, seed=0)
    th, x, pid = th[pick], x[pick], pid[pick].numpy()
    t = th.numpy().astype(float)
    w = np.exp(heldout_log_weight(sh_cfg, pid, t))
    print(f"{len(members)} members; {spec.describe()}; held-out {len(t)} draws; weights "
          f"p/q_held in [{w.min():.3f}, {w.max():.3f}], ESS {w.sum() ** 2 / (w ** 2).sum():.0f}")

    res = {"run_dirs": list(map(str, args.run_dirs)), "n_draws": args.n_draws,
           "spec": spec.describe()}
    torch.manual_seed(0)
    sets = [("ensemble", members)] + [(f"member_{i}", [m]) for i, m in enumerate(members)]
    for name, ms in sets:
        r = sample_weighted(ms, x, args.n_draws, fb.low, fb.high, invert=inv, log_weight=lw,
                            truth=t)
        res[name] = heldout(r, t, w, oa._log_q(ms, th, x, spec, per_row=True), spec)
        if name == "ensemble":
            res["ensemble_by_posterior_mean"] = by_region(r, w, spec)
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
            print(f"  [{b['lo']}, {b['hi']}) n={b['n']:4d}" + (
                f"  cov90 {b['cov90']:.3f}  <5% {b['below5']:.3f}  >95% {b['above95']:.3f}"
                if b["n"] else ""))
    res["psid_floors"] = floors(args.psid)
    print("\nPSID floors:")
    for k, v in res["psid_floors"].items():
        print(f"  {k}: " + json.dumps(v if not isinstance(next(iter(v.values()), 0), float)
                                      else {a: round(b, 3) for a, b in v.items()}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
