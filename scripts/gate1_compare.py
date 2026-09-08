"""Gate 1's verdict: does observation noise account for the misfit?

Adding measurement noise to the simulated ``x`` is post-hoc -- it needs no new
data and no regeneration. If it accounts for what is wrong with the estimates,
the ~9.5 GPU-days the regeneration would cost are saved outright.

The two things it has to move, from ``RESULTS.md`` sections 1 and 7.2, are the
two findings that motivated the regeneration in the first place:

* **The rho ceiling pileup.** 9.3% of households have a posterior-mean rho
  within 5% of the prior's 5.0 ceiling. High rho is the only channel this model
  has for precautionary saving, so missing precautionary motives are routed
  through it and pile up against the boundary.
* **The beta between/within ratio.** 0.79 at baseline -- households differ by
  *less* than any one of them is uncertain, so the apparent heterogeneity is
  estimation noise and the project's own premise is not yet demonstrated.

**Both have to move, and in the right direction.** Noise widens every posterior
mechanically: a wider ``p(x|theta)`` means less information per household, so
within-household sd rises and the between/within ratio falls even when nothing
real has changed. A ratio that *drops* is therefore not evidence of anything --
it is the null result dressed up. The pass condition is the pileup falling
*and* the beta ratio rising.

Usage::

    uv run python scripts/gate1_compare.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.npe.prior import PHASE3

#: A posterior mean this close to the prior's upper edge is pinned rather than
#: estimated -- the same 5% band RESULTS.md section 7.2 reports.
CEILING_BAND = 0.05


def load(run: Path):
    z = np.load(run / "posterior_uncorrected.npz")
    m = z["mean"]
    ok = np.isfinite(m[:, 0])
    return m[ok], z["sd"][ok], z["in_box_frac"][ok], ok


def ensemble_and_x(run_dirs, x_path, ok, waves):
    """The ensemble members and the empirical x, on the posterior's device."""
    from scripts.ensemble_eval import _ensemble
    from hh_npe.evaluation.sbc import posterior_device
    from hh_npe.npe.train import load_posterior

    posts = [load_posterior(d / f"posterior_{waves}w.pt")["posterior"]
             for d in run_dirs]
    dev = posterior_device(_ensemble(posts))
    return posts, torch.load(x_path, weights_only=False)["x"][ok].to(dev)


def median_household(means):
    """The household closest to the median in every coordinate.

    Same rule as ``plot_psid_population.py``, so the within-household sd here is
    the same quantity RESULTS.md section 1 reports rather than a tail draw.
    """
    d2 = np.abs((means - np.median(means, axis=0)) / means.std(axis=0)).sum(axis=1)
    return int(np.argmin(d2))


def draws_for(posts, x, i, n_post):
    lo, hi = np.asarray(PHASE3.low), np.asarray(PHASE3.high)
    with torch.no_grad():
        d = torch.cat([m.posterior_estimator.sample((n_post,),
                                                    condition=x[i:i + 1])
                       for m in posts], dim=0).squeeze(1).cpu().numpy()
    return d[((d >= lo) & (d <= hi)).all(axis=1)]


def report(tag, means, sd, in_box, draws):
    hi = np.asarray(PHASE3.high)
    lo = np.asarray(PHASE3.low)
    span = hi - lo
    pile_hi = (means > hi - CEILING_BAND * span).mean(axis=0)
    pile_lo = (means < lo + CEILING_BAND * span).mean(axis=0)
    out = {"n": int(len(means)), "in_box_median": float(np.median(in_box))}
    print(f"\n=== {tag}  (N={len(means)}) ===")
    print(f"{'':10s}{'median mean':>13s}{'between sd':>12s}{'within sd':>11s}"
          f"{'ratio':>8s}{'at ceiling':>12s}{'at floor':>10s}")
    for j, n in enumerate(PHASE3.names):
        b, w = means[:, j].std(), draws[:, j].std()
        print(f"{n:10s}{np.median(means[:, j]):13.4f}{b:12.4f}{w:11.4f}"
              f"{b / w:8.2f}{pile_hi[j]:12.1%}{pile_lo[j]:10.1%}")
        out[n] = {"median": float(np.median(means[:, j])),
                  "between_sd": float(b), "within_sd": float(w),
                  "ratio": float(b / w), "at_ceiling": float(pile_hi[j]),
                  "at_floor": float(pile_lo[j])}
    print(f"in-box posterior mass: median {np.median(in_box):.3f}, "
          f"p10 {np.percentile(in_box, 10):.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path("outputs/psid_rental"))
    ap.add_argument("--noisy", type=Path, default=Path("outputs/gate1_psid"))
    ap.add_argument("--base_runs", type=Path, nargs="+",
                    default=[Path(f"outputs/flow_fix/w7_s{s}") for s in range(5)])
    ap.add_argument("--noisy_runs", type=Path, nargs="+",
                    default=[Path(f"outputs/gate1_noise/w7_s{s}") for s in range(5)])
    ap.add_argument("--x", type=Path,
                    default=Path("data/processed/psid_x_rental.pt"))
    ap.add_argument("--waves", type=int, default=7)
    ap.add_argument("--n_post", type=int, default=6000)
    ap.add_argument("--out", type=Path, default=Path("outputs/gate1_psid"))
    args = ap.parse_args()

    res = {}
    for tag, run, runs in (("baseline (no noise)", args.base, args.base_runs),
                           ("gate 1 (sigma=0.30)", args.noisy, args.noisy_runs)):
        means, sd, in_box, ok = load(run)
        posts, x = ensemble_and_x(runs, args.x, ok, args.waves)
        i = median_household(means)
        res[tag] = report(tag, means, sd, in_box,
                          draws_for(posts, x, i, args.n_post))

    a, b = res["baseline (no noise)"], res["gate 1 (sigma=0.30)"]
    print("\n=== the two numbers the gate turns on ===")
    print(f"{'':22s}{'baseline':>10s}{'noisy':>10s}{'change':>10s}")
    print(f"{'rho at ceiling':22s}{a['crra']['at_ceiling']:10.1%}"
          f"{b['crra']['at_ceiling']:10.1%}"
          f"{b['crra']['at_ceiling'] - a['crra']['at_ceiling']:+10.1%}")
    print(f"{'beta between/within':22s}{a['beta']['ratio']:10.2f}"
          f"{b['beta']['ratio']:10.2f}"
          f"{b['beta']['ratio'] - a['beta']['ratio']:+10.2f}")

    pile_fell = b["crra"]["at_ceiling"] < a["crra"]["at_ceiling"] - 0.02
    ratio_rose = b["beta"]["ratio"] > a["beta"]["ratio"] + 0.05
    print(f"\npileup fell materially: {pile_fell}")
    print(f"beta ratio rose:        {ratio_rose}")
    print("\nPASS (skip the regeneration) only if BOTH. A falling beta ratio is\n"
          "the null result, not a finding: noise widens every posterior, so the\n"
          "within-household sd rises mechanically whether or not anything real\n"
          "changed.")
    print(f"\nVERDICT: {'PASS' if (pile_fell and ratio_rose) else 'FAIL'}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gate1_verdict.json").write_text(json.dumps(
        {**res, "pass": bool(pile_fell and ratio_rose)}, indent=2))
    print(f"wrote {args.out}/gate1_verdict.json")


if __name__ == "__main__":
    main()
