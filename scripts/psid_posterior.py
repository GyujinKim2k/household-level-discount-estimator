"""Per-household posteriors for the PSID panel — the project's headline output.

Applies the trained amortized posterior to the empirical ``x`` tensor, giving
each of ~2,100 households its own distribution over (beta, delta, rho). That is
the contribution: Laibson et al. report one MSM point estimate for the whole
population (beta=0.5305, delta=0.9891, rho=1.9355); this reports a posterior per
household.

Two arms, per the consumption-measurement question (see `engel_correction.py`):

**uncorrected** — the tensor as built. The Engel diagnostic found PSID's food
elasticity at 0.575, inside the literature's 0.50-0.60, which means differential
under-reporting can account for at most a ~1.20x spread across the consumption
distribution against the ~2.2x consumption/income gradient actually observed.
Most of that gradient is behaviour, not measurement, so this is the headline.

**corrected** — the most correction the food equation permits: a graded factor
``exp(phi' * (log C - log C_p10))`` with ``phi' = 0.130``, the value implied if
the true food elasticity is 0.50 (the end of the literature range most
favourable to correcting). Anchored at the 10th percentile. This is deliberately
the *upper bound* on the correction, so agreement between arms is a strong
robustness statement rather than a weak one.

The correction is applied in 2010 dollars, before standardisation. The
embedder's mean/std buffers are simulator-derived and must not be re-estimated
on PSID: they are the scale the network was trained on, and the level
information they preserve is what identifies the model.

Usage::

    uv run python scripts/psid_posterior.py --out outputs/psid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.evaluation.plots import contour_corner
from hh_npe.evaluation.sbc import posterior_device
from hh_npe.npe.prior import PHASE3
from hh_npe.npe.train import load_posterior

#: Laibson et al., benchmark naive quasi-hyperbolic (SIMULATOR_SPEC 7).
LAIBSON = {"beta": 0.5305, "delta": 0.9891, "crra": 1.9355}
LAIBSON_SE = {"beta": 0.114, "delta": 0.0051, "crra": 0.435}

PHI_PRIME = 0.130   # upper bound from the food equation; see module docstring


def graded_correction(x: np.ndarray, cons_idx: int = 1) -> np.ndarray:
    """Scale consumption by exp(phi' * (log C - log C_p10)), anchored at p10."""
    out = x.copy()
    c = out[:, :, cons_idx]
    pos = c > 0
    anchor = np.log(np.percentile(c[pos], 10))
    f = np.ones_like(c)
    f[pos] = np.exp(PHI_PRIME * (np.log(c[pos]) - anchor))
    out[:, :, cons_idx] = c * f
    return out


def sample_all(post, x: torch.Tensor, n_post: int, batch: int = 512):
    """Posterior summaries per household, plus the share of mass in the prior box.

    Deliberately does NOT use ``posterior.sample``. That rejection-samples
    against the prior box until it collects ``n_post`` accepted draws, and on
    real PSID data some households have *zero* flow mass inside the box -- the
    model cannot represent them at any admissible (beta, delta, rho). Those
    households make the loop run forever: a first attempt completed under 250 of
    2119 households in nine hours. `sample_batched` inherits the same loop, so
    the batch is held hostage by its worst member.

    Instead: draw a fixed budget from the flow, keep what lands in the box, and
    report the in-box fraction. For a well-behaved household that is exactly
    what rejection sampling returns. For an off-support household it returns a
    diagnostic rather than hanging, and the fraction is itself a result -- it
    says what share of real households the model can represent at all.
    """
    dev = posterior_device(post)
    x = x.to(dev)
    lo_b = torch.as_tensor(PHASE3.low, dtype=torch.float32, device=dev)
    hi_b = torch.as_tensor(PHASE3.high, dtype=torch.float32, device=dev)
    members = getattr(post, "posteriors", [post])
    per = max(1, int(np.ceil(n_post * 4 / len(members))))   # 4x for rejection

    means, sds, los, his, frac = [], [], [], [], []
    with torch.no_grad():
        for lo in range(0, len(x), batch):
            xb = x[lo:lo + batch]
            draws = torch.cat([m.posterior_estimator.sample((per,), condition=xb)
                               for m in members], dim=0)      # (S, B, d)
            inb = ((draws >= lo_b) & (draws <= hi_b)).all(-1)  # (S, B)
            frac.append(inb.float().mean(0).cpu().numpy())
            d = draws.cpu().numpy()
            k = inb.cpu().numpy()
            for j in range(d.shape[1]):
                keep = d[k[:, j], j, :]
                if len(keep) < 20:
                    # Too little admissible mass to summarise honestly.
                    nan = np.full(d.shape[-1], np.nan)
                    means.append(nan); sds.append(nan); los.append(nan); his.append(nan)
                    continue
                means.append(keep.mean(0)); sds.append(keep.std(0))
                los.append(np.percentile(keep, 5, axis=0))
                his.append(np.percentile(keep, 95, axis=0))
            print(f"    {min(lo + batch, len(x))}/{len(x)}", flush=True)
    return (np.array(means), np.array(sds), np.array(los), np.array(his),
            np.concatenate(frac))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dirs", type=Path, nargs="+",
                    default=[Path(f"outputs/flow_fix/w7_s{s}") for s in range(5)])
    ap.add_argument("--waves", type=int, default=7)
    ap.add_argument("--x", type=Path, default=Path("data/processed/psid_x.pt"))
    ap.add_argument("--n_post", type=int, default=500)
    ap.add_argument("--condition_educ", action="store_true",
                   help="Append the one-hot education block, for a posterior "
                        "trained with --condition_educ. Education comes from "
                        "the x file's own `educ` field, which is PSID's "
                        "observed value -- the whole point of conditioning is "
                        "that we do observe it.")
    ap.add_argument("--educ_group", choices=["comphs", "somehs", "compco"],
                   default=None,
                   help="Restrict to one education group, for a per-group "
                        "posterior. Scoring a per-group model on households "
                        "from other groups asks it about people it was never "
                        "trained on.")
    ap.add_argument("--figure_only", action="store_true",
                   help="Reuse the saved posterior_*.npz and regenerate only "
                        "the contour figure. For a run whose sampling "
                        "succeeded but whose figure did not -- redoing ~50 min "
                        "of sampling to redraw a plot is pure waste.")
    ap.add_argument("--out", type=Path, default=Path("outputs/psid"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from scripts.ensemble_eval import _ensemble
    posts = [load_posterior(d / f"posterior_{args.waves}w.pt")["posterior"]
             for d in args.run_dirs]
    post = _ensemble(posts) if len(posts) > 1 else posts[0]
    print(f"posterior: {len(posts)}-member ensemble, {args.waves} waves")

    d = torch.load(args.x, weights_only=False)
    x_raw = d["x"].numpy()
    print(f"empirical x: {x_raw.shape}  features {d['features']}")

    educ = d["educ"].numpy() if "educ" in d else None
    if (args.condition_educ or args.educ_group) and educ is None:
        raise SystemExit(
            f"{args.x} carries no `educ` field. Rebuild it with "
            f"build_psid_tensor.py --educ_groups."
        )
    keep = np.arange(len(x_raw))
    if args.educ_group:
        from hh_npe.simulator.laibson_calibration import EDUC_GROUPS
        keep = np.flatnonzero(educ == EDUC_GROUPS.index(args.educ_group))
        x_raw, educ = x_raw[keep], educ[keep]
        print(f"education filter {args.educ_group}: {len(keep)} households")
    np.save(args.out / "household_index.npy", keep)

    arms = {"uncorrected": x_raw, "corrected": graded_correction(x_raw)}
    cf = arms["corrected"][:, :, 1] / np.maximum(arms["uncorrected"][:, :, 1], 1)
    print(f"correction factor: median {np.median(cf):.3f}, "
          f"p10 {np.percentile(cf, 10):.3f}, p90 {np.percentile(cf, 90):.3f}")

    res = {}
    if args.figure_only:
        for name in arms:
            z = np.load(args.out / f"posterior_{name}.npz")
            res[name] = {"mean": z["mean"], "sd": z["sd"], "lo": z["lo"],
                         "hi": z["hi"], "frac": z["in_box_frac"]}
        print(f"reusing saved posteriors from {args.out}")
        _figure(args, post, arms, x_raw, educ)
        return
    for name, xa in arms.items():
        print(f"\n--- {name} ---", flush=True)
        torch.manual_seed(0)
        xt = torch.from_numpy(xa).float()
        if args.condition_educ:
            from scripts.compare_windows import one_hot_educ
            # The correction only touches consumption, so the education block
            # is appended after it -- appending first would leave the one-hot
            # columns to be rescaled as if they were dollars.
            xt = one_hot_educ(xt, educ)
        m, sdv, lo, hi, fr = sample_all(post, xt, args.n_post)
        res[name] = {"mean": m, "sd": sdv, "lo": lo, "hi": hi, "frac": fr}
        np.savez(args.out / f"posterior_{name}.npz", mean=m, sd=sdv, lo=lo,
                 hi=hi, in_box_frac=fr)
        ok = np.isfinite(m[:, 0])
        print(f"  in-box mass: median {np.median(fr):.3f}  p10 "
              f"{np.percentile(fr, 10):.3f}")
        print(f"  households the model cannot represent (<20 admissible "
              f"draws): {int((~ok).sum())} of {len(m)} "
              f"({(~ok).mean():.2%})")

    names = list(PHASE3.names)
    # ONE mask across both arms. Filtering each arm by its own would leave
    # arrays of different length, and the corrected-minus-uncorrected shift
    # below would then subtract different households from each other.
    ok = np.logical_and.reduce([np.isfinite(res[a]["mean"][:, 0]) for a in arms])
    print(f"\nsummaries on {int(ok.sum())} households represented in BOTH arms "
          f"({ok.mean():.2%} of {len(ok)})")
    res = {a: {k: (v[ok] if k != "frac" else v) for k, v in res[a].items()}
           for a in arms}
    print(f"\n{'':24s}" + "".join(n.rjust(12) for n in names))
    for arm in arms:
        m = res[arm]["mean"]
        for stat, fn in (("median of means", lambda a: np.median(a, 0)),
                         ("mean of means", lambda a: a.mean(0)),
                         ("sd across households", lambda a: a.std(0)),
                         ("median posterior sd", lambda a: None)):
            if stat == "median posterior sd":
                v = np.median(res[arm]["sd"], 0)
            else:
                v = fn(m)
            print(f"{arm[:10]:11s}{stat:13s}" + "".join(f"{q:12.4f}" for q in v))
        print()

    print(f"{'Laibson et al. MSM':24s}"
          + "".join(f"{LAIBSON[n]:12.4f}" for n in names))
    print(f"{'  their std error':24s}"
          + "".join(f"{LAIBSON_SE[n]:12.4f}" for n in names))

    # Share of households whose 90% credible interval contains their estimate.
    print()
    for arm in arms:
        lo, hi = res[arm]["lo"], res[arm]["hi"]
        cov = [(float(np.mean((lo[:, j] <= LAIBSON[n]) & (hi[:, j] >= LAIBSON[n]))))
               for j, n in enumerate(names)]
        print(f"{arm[:11]:12s}share of households whose 90% CI covers "
              f"Laibson: " + "  ".join(f"{n}={c:.3f}" for n, c in zip(names, cov)))

    # Robustness: how far does the correction move each household?
    dm = res["corrected"]["mean"] - res["uncorrected"]["mean"]
    print(f"\n{'shift from correction':24s}"
          + "".join(f"{v:12.4f}" for v in np.median(dm, 0))
          + "   (median, corrected - uncorrected)")
    print(f"{'  as fraction of their SE':24s}"
          + "".join(f"{np.median(dm[:, j]) / LAIBSON_SE[n]:12.3f}"
                    for j, n in enumerate(names)))

    json.dump({"laibson": LAIBSON, "phi_prime": PHI_PRIME,
               "n_households": int(len(x_raw)),
               "median_posterior_mean": {a: {n: float(np.median(res[a]['mean'][:, j]))
                                             for j, n in enumerate(names)}
                                         for a in arms}},
              open(args.out / "summary.json", "w"), indent=2)

    _figure(args, post, arms, x_raw, educ)


def _figure(args, post, arms, x_raw, educ):
    """Standing requirement: a contour figure with every estimation result."""
    rng = np.random.default_rng(0)
    pick = rng.choice(len(x_raw), size=3, replace=False)
    series = {}
    dev = posterior_device(post)
    for r, i in enumerate(pick):
        xt = torch.from_numpy(arms["uncorrected"][i : i + 1]).float()
        if args.condition_educ:
            # The figure re-derives x from the raw array, so it needs the same
            # education block the scored path got. Forgetting it here is how
            # the first conditioned run died *after* writing its posteriors.
            from scripts.compare_windows import one_hot_educ
            xt = one_hot_educ(xt, educ[i : i + 1])
        s = post.sample((4000,), x=xt[0].to(dev), show_progress_bars=False)
        series[f"PSID household {r + 1}"] = s.detach().cpu().numpy()
    # Title from what actually ran. It previously read "Phase 3 posterior ...
    # 2,119-household panel" on every figure regardless of the variant or the
    # sample, which is wrong on both counts for anything after Phase 3.
    what = []
    if args.educ_group:
        what.append(args.educ_group)
    if args.condition_educ:
        what.append("education-conditioned")
    label = ", ".join(what) if what else "pooled"
    contour_corner(series, PHASE3, path=args.out / "psid_households.png",
                   title=(f"PSID households — {label}, {args.waves} waves, "
                          f"{len(args.run_dirs)}-member ensemble\nthree "
                          f"households drawn at random from "
                          f"{len(x_raw):,}"))
    print(f"\nwrote {args.out}/psid_households.png")


if __name__ == "__main__":
    main()
