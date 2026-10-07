"""Single-household posteriors for an option A2 ensemble (RESULTS 43): the
counterparts of figures 10 (simulated) and 15 (PSID), on the 4-parameter box.

Same sampling path as ``optionA.psid`` and ``evaluate``: draws are rejected in
the box the flow lives in, inverted (logit beta, log(1 - delta)), importance-
weighted back to the uniform prior (``proposal_log_weight``), and then
resampled to equally weighted draws for the contour plot. Card type is
marginalised, as in training.

**Which households.** Picked by rule, not by eye:

- *Simulated*: from a random ``--pool`` of held-out windows (one per draw),
  the household whose prior-standardised posterior-mean error is the median
  among those whose true theta is 10% in from every edge of the box -- as for
  figure 10. Its true theta is on the figure.
- *PSID*: the couple whose posterior mean is nearest, in prior-standardised
  distance, the couples' median posterior mean (``optionA.py psid`` output) --
  as for figure 15. No true theta exists; Laibson et al.'s population estimate
  and the meta-analytic ranges are shown for reference.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/plot_optionA2_households.py \\
        --run_dirs outputs/optionA2/s{0..4} --shards data/processed/couples_dataset_shards \\
        --psid_out outputs/psid_optionA2_couples --x data/processed/psid_x_comphs_couples.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.windows import build_windowed
from hh_npe.evaluation.plots import contour_corner
from hh_npe.evaluation.weighted import resample, sample_weighted, summarize
from hh_npe.evaluation.sbc import posterior_device
from scripts import optionA as oa
from scripts.literature_ranges import LAIBSON, META

BOX = oa.BOX
FEATS = ("income", "consumption", "liquid_assets", "illiquid_assets")


def weighted_draws(members, x1: torch.Tensor, n_raw: int, fb, inv, lw):
    """Raw ensemble draws for one household, in the order sample_weighted uses:
    reject in the flow box, invert, weight. Returns (theta draws, log weights)."""
    dev = posterior_device(members[0])
    per = int(np.ceil(n_raw / len(members)))
    lo, hi = np.asarray(fb.low), np.asarray(fb.high)
    with torch.no_grad():
        s = np.concatenate([m.posterior_estimator.sample((per,), condition=x1.to(dev))
                            .squeeze(1).cpu().numpy().astype(float) for m in members])
    th = inv(s[((s >= lo) & (s <= hi)).all(1)])
    return th, lw(th)


def describe(x_raw: np.ndarray) -> str:
    """The household's waves, from the untransformed features."""
    ages = x_raw[:, -1].astype(int)
    rows = [f"  ages {ages.tolist()}"]
    for j, f in enumerate(FEATS):
        rows.append(f"  {f:16s} " + " ".join(f"{int(round(v)):>9,d}" for v in x_raw[:, j]))
    return "\n".join(rows)


def report(d: np.ndarray, lw: np.ndarray, truth=None, ref=None) -> np.ndarray:
    s = summarize(d, lw, None)
    print(f"  {len(d):,} draws kept, ESS {s['ess']:,.0f}")
    print(f"  {'':8s}" + (f"{'true':>9s}" if truth is not None else "")
          + f"{'estimate':>10s}{'90% interval':>24s}"
          + (f"{'reference':>11s}" if ref is not None else ""))
    for j, n in enumerate(BOX.names):
        cov = "" if truth is None else ("  covered" if s["lo"][j] <= truth[j] <= s["hi"][j]
                                        else "  NOT covered")
        print(f"  {n:8s}" + (f"{truth[j]:9.4f}" if truth is not None else "")
              + f"{s['mean'][j]:10.4f}   [{s['lo'][j]:.4f}, {s['hi'][j]:.4f}]"
              + (f"{ref[j]:11.4f}" if ref is not None else "") + cov)
    return s["mean"]


def simulated(args, members, fb, inv, lw, label) -> None:
    theta_all, card_all, _tr, held_f, _p, _cfg = oa.load_shards(args.shards)
    th, x_raw, pid = build_windowed(held_f, theta_all, k=1, seed=oa.HELD_SEED, **oa.WINDOW)
    pick1 = oa.one_per_draw(pid, seed=0)
    # one_per_draw's rows are grouped by window start age; sample, not a prefix.
    pool = np.sort(np.random.default_rng(1).choice(pick1, min(args.pool, len(pick1)),
                                                   replace=False))
    th, x_raw, pid = th[pool].numpy().astype(float), x_raw[pool], pid[pool].numpy()
    x, _f = oa.transform(x_raw)
    torch.manual_seed(0)
    r = sample_weighted(members, x, args.pool_draws, fb.low, fb.high, invert=inv,
                        log_weight=lw)
    err = np.sqrt((((r["mean"] - th) / oa.PRIOR_SD) ** 2).mean(1))
    span = BOX.high - BOX.low
    interior = ((th > BOX.low + 0.1 * span) & (th < BOX.high - 0.1 * span)).all(1)
    cand = np.flatnonzero(interior & np.isfinite(err))
    pick = cand[np.argsort(err[cand])[len(cand) // 2]]
    print(f"\n=== simulated: pool {len(th)} held-out households, interior {len(cand)}; "
          f"standardised error p25 {np.nanpercentile(err, 25):.3f} median "
          f"{np.nanmedian(err):.3f} p75 {np.nanpercentile(err, 75):.3f}; picked "
          f"#{pick} at {err[pick]:.3f} (draw {pid[pick]}, card {int(card_all[pid[pick]])}) ===")
    print(describe(x_raw[pick].numpy()))
    torch.manual_seed(0)
    d, w = weighted_draws(members, x[pick:pick + 1], args.n_raw, fb, inv, lw)
    truth = th[pick]
    est = report(d, w, truth=truth)
    plain = resample(d, w, args.n_plot, seed=0)
    age = x_raw[pick, :, -1].int().tolist()
    out = args.fig_dir / args.sim_name
    contour_corner(
        {f"posterior ({len(members)}-member ensemble, importance-weighted)": plain}, BOX,
        truth={"true θ (generated the data)": truth, "posterior mean (point estimate)": est},
        truth_markers=("*", "X"), truth_colors=("#d62728", "#1f1f1f"),
        reflect_axes=("delta",), path=out,
        title=f"One held-out simulated comphs household, ages {age[0]}-{age[-1]}, "
              f"{'cardholder' if card_all[pid[pick]] else 'no card'}: posterior vs. truth\n"
              f"{label}; 7 waves; card type marginalised")
    np.savez(out.with_suffix(".npz"), draws=d, log_w=w, truth=truth, estimate=est,
             pool_error=err, pick=pick, draw=pid[pick])
    print(f"wrote {out}")


def psid(args, members, fb, inv, lw, label) -> None:
    m = np.load(args.psid_out / "posterior_uncorrected.npz")["mean"]
    keep = np.load(args.psid_out / "household_index.npy")
    ok = np.isfinite(m[:, 0])
    med = np.median(m[ok], axis=0)
    dist = np.sqrt((((m - med) / oa.PRIOR_SD) ** 2).sum(1))
    dist[~ok] = np.inf
    pick = int(np.argmin(dist))
    dd = torch.load(args.x, weights_only=False)
    x_raw = dd["x"].numpy()[keep]
    assert len(x_raw) == len(m), "tensor and posterior file disagree on households"
    print(f"\n=== PSID: household #{pick} of {len(m)} (nearest the median posterior "
          f"mean; standardised distance {dist[pick]:.3f}) ===")
    print(describe(x_raw[pick]))
    x, _f = oa.transform(torch.from_numpy(x_raw[pick:pick + 1]).float())
    torch.manual_seed(0)
    d, w = weighted_draws(members, x, args.n_raw, fb, inv, lw)
    ref = np.append(LAIBSON, 1.05)
    est = report(d, w, ref=ref)
    plain = resample(d, w, args.n_plot, seed=0)
    age = x_raw[pick, :, -1].astype(int)
    out = args.fig_dir / args.psid_name
    contour_corner(
        {f"posterior ({len(members)}-member ensemble, importance-weighted)": plain}, BOX,
        truth={"posterior mean (point estimate)": est,
               "Laibson et al. MSM (population; R_gamma calibrated 1.05)": ref},
        truth_markers=("X", "*"), truth_colors=("#1f1f1f", "#d62728"),
        bands={"meta-analytic range (lit.; R_gamma none)":
               (np.append(META[0], BOX.rgamma_low), np.append(META[1], BOX.rgamma_high))},
        reflect_axes=("delta",), path=out,
        title=f"One real PSID couple (#{pick} of {len(m)}), ages {age[0]}-{age[-1]}: "
              f"typical, nearest the couples' median\n{label}; no true θ exists for "
              "real data; card type marginalised")
    np.savez(out.with_suffix(".npz"), draws=d, log_w=w, estimate=est, pick=pick)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--shards", type=Path, default=Path("data/processed/couples_dataset_shards"))
    ap.add_argument("--psid_out", type=Path, required=True)
    ap.add_argument("--x", type=Path, default=Path("data/processed/psid_x_comphs_couples.pt"))
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--pool_draws", type=int, default=4000,
                    help="Raw draws per pool household, for picking.")
    ap.add_argument("--n_raw", type=int, default=40000,
                    help="Raw draws for the plotted household (as optionA psid).")
    ap.add_argument("--n_plot", type=int, default=20000,
                    help="Equally weighted draws resampled for the contours.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--label", default="option A2 (couples regeneration, PSID seed pool)")
    ap.add_argument("--fig_dir", type=Path, default=Path("figures"))
    ap.add_argument("--sim_name", default="36_simulated_household_optionA2.png")
    ap.add_argument("--psid_name", default="37_psid_household_optionA2.png")
    args = ap.parse_args()

    members, proposal, cfg = oa.load_members(args.run_dirs, args.device)
    bt = cfg["beta_transform"]
    inv, lw = oa._weighting(proposal, bt)
    fb = oa.flow_box(bt)
    print(f"{len(members)} members; proposal {proposal}; beta transform {bt}")
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    simulated(args, members, fb, inv, lw, args.label)
    psid(args, members, fb, inv, lw, args.label)


if __name__ == "__main__":
    main()
