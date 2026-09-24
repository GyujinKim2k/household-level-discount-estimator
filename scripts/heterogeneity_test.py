"""Is there any true between-household heterogeneity, and does it survive?

The project's headline claim is a posterior *per household* rather than one
population point estimate, so the question that decides whether that buys
anything is whether the households actually differ. Spread in posterior means
is not evidence: each mean carries its own estimation error, and averaging noisy
estimates of one common parameter produces a spread too.

The decomposition (RESULTS.md 11.3):

    Var(posterior means) = Var(true theta) + E[posterior variance]

so ``Var(true theta) = Var(means) - mean(posterior var)``. A **negative** value
means the observed spread is smaller than estimation error alone would produce
— no detectable heterogeneity, and the per-household posteriors are consistent
with every household sharing one parameter.

The bootstrap resamples households, so the reported P is
``P(Var(true theta) <= 0)`` under resampling: large P means the data cannot
distinguish the population from homogeneous.

**Read the ratio with care.** ``RESULTS.md`` §1 reports a between/within ratio
built on the *median* posterior sd, while the variance identity above needs the
*root-mean-square* posterior sd. The two differ whenever posterior widths are
skewed across households, which they are here. Both are printed, labelled.

Usage::

    uv run python scripts/heterogeneity_test.py \
        --runs outputs/psid_phase4_comphs outputs/psid_adopted_arch \
        --labels baseline adopted
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hh_npe.npe.prior import PHASE3


def decompose(mean: np.ndarray, sd: np.ndarray, n_boot: int, seed: int) -> dict:
    """Per-parameter variance decomposition with a household bootstrap."""
    ok = np.isfinite(mean[:, 0])
    mean, sd = mean[ok], sd[ok]
    n = len(mean)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    out = {"n": int(n)}
    for j, name in enumerate(PHASE3.names):
        m, s = mean[:, j], sd[:, j]
        between = m.var(ddof=1)
        within = (s ** 2).mean()
        # Resample households, recomputing both terms on each replicate: the
        # two are estimated on the same households and move together, so
        # bootstrapping the difference is not the same as differencing two
        # separate bootstraps.
        bm, bs = m[idx], s[idx]
        boot = bm.var(axis=1, ddof=1) - (bs ** 2).mean(axis=1)
        out[name] = {
            "median_mean": float(np.median(m)),
            "between_sd": float(np.sqrt(between)),
            "within_sd_rms": float(np.sqrt(within)),
            "within_sd_median": float(np.median(s)),
            "ratio_rms": float(np.sqrt(between / within)),
            "ratio_median_sd": float(np.sqrt(between) / np.median(s)),
            "var_true": float(between - within),
            "var_true_ci": [float(np.percentile(boot, 2.5)),
                            float(np.percentile(boot, 97.5))],
            "p_le_zero": float((boot <= 0).mean()),
        }
    return out


def show(label: str, r: dict) -> None:
    print(f"\n=== {label}  (N={r['n']}) ===")
    print(f"{'param':8s}{'median':>9s}{'betw sd':>9s}{'within':>9s}"
          f"{'ratio':>7s}{'Var(true)':>11s}{'95% CI':>22s}{'P<=0':>7s}")
    for name in PHASE3.names:
        d = r[name]
        ci = f"[{d['var_true_ci'][0]:+.5f}, {d['var_true_ci'][1]:+.5f}]"
        print(f"{name:8s}{d['median_mean']:9.4f}{d['between_sd']:9.4f}"
              f"{d['within_sd_rms']:9.4f}{d['ratio_rms']:7.2f}"
              f"{d['var_true']:+11.5f}{ci:>22s}{d['p_le_zero']:7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True,
                    help="psid_posterior output directories.")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Names for the runs; defaults to the directory names.")
    ap.add_argument("--arm", default="uncorrected",
                    choices=["uncorrected", "corrected"],
                    help="Which consumption arm to read.")
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON destination.")
    args = ap.parse_args()

    labels = args.labels or [d.name for d in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit(f"{len(labels)} labels for {len(args.runs)} runs")

    res = {}
    for lab, d in zip(labels, args.runs):
        f = d / f"posterior_{args.arm}.npz"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        z = np.load(f)
        res[lab] = decompose(z["mean"], z["sd"], args.n_boot, args.seed)
        show(lab, res[lab])

    if len(labels) == 2:
        a, b = (res[k] for k in labels)
        print(f"\n=== {labels[1]} minus {labels[0]} ===")
        print(f"{'param':8s}{'d median':>10s}{'d betw sd':>11s}"
              f"{'d within':>10s}{'d Var(true)':>13s}{'sign change':>13s}")
        for n in PHASE3.names:
            flip = ("yes" if (a[n]["var_true"] > 0) != (b[n]["var_true"] > 0)
                    else "-")
            print(f"{n:8s}{b[n]['median_mean'] - a[n]['median_mean']:+10.4f}"
                  f"{b[n]['between_sd'] - a[n]['between_sd']:+11.4f}"
                  f"{b[n]['within_sd_rms'] - a[n]['within_sd_rms']:+10.4f}"
                  f"{b[n]['var_true'] - a[n]['var_true']:+13.5f}{flip:>13s}")

    print("\nA negative Var(true) means the spread across households is smaller "
          "than\nestimation error alone would produce: no heterogeneity "
          "detectable. P<=0 is\nthe bootstrap share of replicates at or below "
          "zero, so large P means the\ndata cannot tell this population from a "
          "homogeneous one.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
