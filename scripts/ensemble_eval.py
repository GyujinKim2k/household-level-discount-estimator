"""Does ensembling fix the undercoverage?

Every arm of the wave matrix undercovers reproducibly -- coverage_90 lands at
0.86-0.89 against a nominal 0.900, with seed spreads of 0.004-0.012, so the
shortfall sits far outside noise. The credible intervals are systematically too
narrow, independent of wave count, which makes it a defect in the approximation
rather than in the observation window.

The seed replication says what kind of defect. Holding the data, the panel
split and the SBC draws fixed and varying only network initialisation moved the
KS p-values over two to three orders of magnitude. That is approximation
*variance*: each fitted flow lands somewhere different, and each individual one
is over-confident about its own landing spot. Averaging several is the standard
remedy, and it is the one the evidence actually points at.

This costs no training. The matrix already saved five independently initialised
posteriors per arm, so the ensemble is those five as a mixture, scored the same
way the individuals were -- same held-out draws, same cached SBC simulations,
same window seeds. Any difference is the ensemble and nothing else.

Usage::

    uv run python scripts/ensemble_eval.py --waves 7 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from hh_npe.data.waves import FEATURE_SETS
from hh_npe.data.windows import build_windowed, window_panel
from hh_npe.evaluation.scoring import calibration_scores, estimation_scores
from hh_npe.npe.prior import PHASE3, sample_sobol
from hh_npe.npe.train import load_posterior

log = logging.getLogger("ensemble_eval")

WAVE_YEARS = 2
# Paired with the wave count so both arms occupy ages 25-59; see
# scripts/run_wave_matrix.sh.
START_HIGH = {7: 46, 10: 40}


def _ensemble(posteriors: list):
    """Uniform mixture of independently trained posteriors."""
    from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior

    ens = EnsemblePosterior(posteriors)
    dev = getattr(posteriors[0], "_device", "cpu")
    ens._device = dev
    # sbi builds the mixture weights on CPU unconditionally, then in log_prob
    # does `log_weights.expand_as(log_probs) + log_probs` against CUDA member
    # log-probs (ensemble_posterior.py:243) and raises. Members carry their own
    # device; the wrapper does not follow them.
    ens._weights = ens._weights.to(dev)
    return ens


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--waves", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--matrix", type=Path, default=Path("outputs/wave_matrix"))
    p.add_argument("--run_dirs", type=Path, nargs="+", default=None,
                   help="Explicit run directories to ensemble, instead of "
                        "<matrix>/w<waves>_s<seed>. The evaluation sets are cut "
                        "with k=1 regardless of the k each member trained on, "
                        "so runs differing only in windows-per-panel are "
                        "scored on identical draws and stay comparable.")
    p.add_argument("--tag", type=str, default=None,
                   help="Output subdirectory name; defaults to w<waves>.")
    p.add_argument("--shards", type=Path,
                   default=Path("data/processed/phase3_dataset_shards"))
    p.add_argument("--sbc_cache", type=Path,
                   default=Path("outputs/window_comparison/sbc_sims.pt"))
    p.add_argument("--train_n", type=int, default=57344)
    p.add_argument("--n_total", type=int, default=65536)
    p.add_argument("--n_heldout_eval", type=int, default=2048)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--start_low", type=int, default=25)
    p.add_argument("--features", type=str, default=None,
                   choices=sorted(FEATURE_SETS),
                   help="Must match what the members were TRAINED on. The "
                        "evaluation windows are rebuilt here, so a mismatch "
                        "means scoring a posterior on features it never saw. "
                        "A different feature *count* aborts on shape; the same "
                        "count cut differently would not, so pass it "
                        "explicitly rather than relying on the default.")
    p.add_argument("--educ_group", choices=["comphs", "somehs", "compco"],
                   default=None,
                   help="Must match what the members were TRAINED on, for the "
                        "same reason as --features. A per-group model scored "
                        "against a three-group evaluation set is being tested "
                        "on households it was never built for, and unlike a "
                        "feature-count mismatch that does NOT abort on shape.")
    p.add_argument("--condition_educ", action="store_true",
                   help="Must match training. A feature-count mismatch does "
                        "abort here, which is how the first phase 4 run was "
                        "caught.")
    p.add_argument("--out", type=Path, default=Path("outputs/ensemble"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    w = args.waves
    start_high = START_HIGH[w]
    out = args.out / (args.tag or f"w{w}")
    out.mkdir(parents=True, exist_ok=True)

    run_dirs = args.run_dirs or [args.matrix / f"w{w}_s{s}" for s in args.seeds]
    posts = []
    for d in run_dirs:
        f = Path(d) / f"posterior_{w}w.pt"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        posts.append(load_posterior(f)["posterior"])
    log.info(f"Loaded {len(posts)} posteriors for {w} waves from "
             f"{[str(d) for d in run_dirs]}")

    # Rebuild the *same* evaluation sets the individual runs were scored on.
    # These seeds are fixed in compare_windows.py; changing one here would make
    # the ensemble incomparable to the members it is built from.
    shard_files = sorted(args.shards.glob("shard_*.npz"))
    held = [f for f in shard_files if int(np.load(f)["lo"]) >= args.train_n]
    theta_all = sample_sobol(args.n_total, PHASE3, seed=0)
    win = dict(start_low=args.start_low, start_high=start_high,
               wave_years=WAVE_YEARS, with_age=True,
               features=FEATURE_SETS[args.features] if args.features else None)

    th_ho, x_ho, pid_ho = build_windowed(held, theta_all, k=1, n_waves=w,
                                         seed=999, **win)
    cache = torch.load(args.sbc_cache, weights_only=False)
    th_sbc, x_sbc, ids = window_panel(cache["panels"], cache["thetas"].numpy(),
                                      k=1, n_waves=w, seed=4242, **win)

    # The evaluation windows are rebuilt here, so every transformation the
    # members saw in training has to be reapplied. Omitting them does not
    # abort -- it silently scores each posterior on a distribution it was not
    # built for.
    if args.educ_group or args.condition_educ:
        from scripts.compare_windows import educ_by_draw, one_hot_educ
        by_draw = educ_by_draw(shard_files)
        sbc_educ = cache.get("educ_idx")
        if by_draw is None or sbc_educ is None:
            raise SystemExit(
                f"--educ_group/--condition_educ need shards and an SBC cache "
                f"from a --educ mixed run; {args.shards} or {args.sbc_cache} "
                f"carries no per-draw education."
            )
        e_ho = by_draw[pid_ho.numpy()]
        e_sbc = np.asarray(sbc_educ)[ids.numpy()]
        if args.educ_group:
            from hh_npe.simulator.laibson_calibration import EDUC_GROUPS
            g = EDUC_GROUPS.index(args.educ_group)
            m_ho, m_sbc = e_ho == g, e_sbc == g
            th_ho, x_ho, e_ho = th_ho[m_ho], x_ho[m_ho], e_ho[m_ho]
            th_sbc, x_sbc, e_sbc = th_sbc[m_sbc], x_sbc[m_sbc], e_sbc[m_sbc]
            log.info(f"education filter {args.educ_group}: {len(th_ho)} "
                     f"held-out, {len(th_sbc)} SBC")
        if args.condition_educ:
            x_ho, x_sbc = one_hot_educ(x_ho, e_ho), one_hot_educ(x_sbc, e_sbc)
            log.info(f"conditioning on education: {x_ho.shape[-1]} features")
    th_ho, x_ho = th_ho[: args.n_heldout_eval], x_ho[: args.n_heldout_eval]
    log.info(f"Scoring on {len(th_ho)} held-out and {len(th_sbc)} SBC draws")

    ens = _ensemble(posts)
    torch.manual_seed(0)
    per_param, log_q = estimation_scores(ens, PHASE3, th_ho, x_ho, n_post=400)
    cal = calibration_scores(ens, PHASE3, th_sbc, x_sbc, n_post=args.n_post,
                             out_dir=out, tag=f"{w}w_ensemble")

    # Members' own numbers, for the only comparison that matters: is the
    # ensemble better than the average member, and is it better than the *best*
    # member? Beating the average is expected; beating the best is the claim.
    members = {}
    for s, d_ in zip(args.seeds, run_dirs):
        # A member may have trained and saved its posterior but died before
        # writing its own scores. The ensemble is still perfectly scoreable
        # from the saved posteriors -- only the ensemble-vs-members comparison
        # is lost -- so skip rather than discard the whole run.
        rj, rn = Path(d_) / "results.json", Path(d_) / f"sbc_ranks_{w}w.npz"
        if not (rj.exists() and rn.exists()):
            log.warning(f"{d_} has no member scores; excluded from the "
                        f"ensemble-vs-members comparison")
            continue
        j = json.load(open(rj))
        d = np.load(rn)
        ks = [stats.kstest((d["ranks"][:, i] + 0.5) / (args.n_post + 1),
                           "uniform").pvalue for i in range(3)]
        members[s] = {"log_q": j[str(w)]["held_out_log_q"],
                      "coverage_90": d["coverage_90"].tolist(), "ks_p": ks}

    res = {"waves": w, "seeds": args.seeds, "start_high": start_high,
           "ensemble": {"estimation": per_param, "held_out_log_q": log_q,
                        "calibration": cal},
           "members": members}
    (out / "results.json").write_text(json.dumps(res, indent=2))

    names = PHASE3.names
    seeds_ok = [s for s in args.seeds if s in members]
    if not seeds_ok:
        print(f"\n=== {w} waves: ensemble of {len(posts)} "
              f"(no member scores available) ===")
        print(f"log q {log_q:.3f}")
        for i, n in enumerate(names):
            print(f"{n:8s} coverage_90 {cal[n]['coverage_90']:.3f}  "
                  f"ks_p {cal[n]['ks_p']:.4f}")
        return
    cov_m = np.array([members[s]["coverage_90"] for s in seeds_ok])
    print(f"\n=== {w} waves: ensemble of {len(posts)} vs its members ===")
    print(f"{'':8s}{'members mean':>15s}{'members best':>15s}{'ensemble':>12s}")
    print(f"{'log q':8s}"
          f"{np.mean([members[s]['log_q'] for s in seeds_ok]):15.3f}"
          f"{max(members[s]['log_q'] for s in seeds_ok):15.3f}"
          f"{log_q:12.3f}")
    print("-- coverage_90 (target 0.900) --")
    for i, nm in enumerate(names):
        best = cov_m[:, i][np.argmin(np.abs(cov_m[:, i] - 0.9))]
        print(f"{nm:8s}{cov_m[:, i].mean():15.3f}{best:15.3f}"
              f"{cal[nm]['coverage_90']:12.3f}")
    print("-- ks_p --")
    for i, nm in enumerate(names):
        ks_m = np.array([members[s]["ks_p"][i] for s in seeds_ok])
        print(f"{nm:8s}{ks_m.mean():15.4f}{ks_m.max():15.4f}"
              f"{cal[nm]['ks_p']:12.4f}")
    print("\nEnsembling reduces approximation variance; it does not add "
          "information.\nIf coverage is still short of 0.900 the residual is "
          "bias, not variance,\nand needs capacity or a different "
          "architecture rather than more members.")


if __name__ == "__main__":
    main()
