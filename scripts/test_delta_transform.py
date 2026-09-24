"""Does estimating log(1 - delta) instead of delta fix the truncation?

RESULTS.md §12.6: ~24% of comphs households have their delta 95th percentile at
the 1.0 bound, and sampling the flow unrestricted shows 23.6% of their mass lies
above 1.0. The posteriors are *tight* (sd 0.0043) and centred at 0.998 — a spike
pressed against a hard wall, which is a known failure mode for normalising
flows.

Under ``d' = log(1 - delta)`` that wall moves to minus infinity:

    delta 0.9891 -> -4.52     delta 0.9979 -> -6.17
    delta 0.9922 -> -4.85     delta 0.9999 -> -9.21

The spike occupies ~11.5% of the linear [0.85, 1.0] range and essentially all of
the log range, and **truncation becomes structurally impossible**.

Two things this does NOT do, stated up front so the result is read correctly:

* It does not let the data say ``delta > 1``. It makes that unrepresentable —
  which is economically right, but it imposes the restriction rather than
  testing it.
* It adds no training signal. Only 1.33% of draws have ``delta > 0.998``, where
  a quarter of households want their posterior. That is prior coverage, and no
  reparameterisation fixes it.

Metrics are computed **after inverting back to delta**, so they are directly
comparable to a untransformed run.

Usage::

    uv run python scripts/test_delta_transform.py --train_seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from hh_npe.data.windows import build_windowed
from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import (
    PHASE3,
    from_log1m,
    log1m_box,
    sample_sobol,
    to_log1m,
)
from hh_npe.npe.train import train_npe
from scripts.compare_windows import EMBEDDER, TRAINING, educ_by_draw, split_shards

log = logging.getLogger("delta_transform")

def score(post, box, th_true, x, n_post: int, invert: bool) -> dict:
    """corr / mae / 90% coverage, always in delta space."""
    lo, hi = np.asarray(box.low), np.asarray(box.high)
    means, cov = [], []
    truth = th_true.numpy()
    with torch.no_grad():
        for i in range(len(x)):
            s = post.posterior_estimator.sample((n_post,),
                                                condition=x[i:i + 1])
            s = s.squeeze(1).cpu().numpy()
            s = s[((s >= lo) & (s <= hi)).all(axis=1)]
            if len(s) < 20:
                means.append(np.full(3, np.nan)); cov.append(np.zeros(3, bool))
                continue
            if invert:
                s = from_log1m(s)
            means.append(s.mean(axis=0))
            q = np.percentile(s, [5, 95], axis=0)
            cov.append((q[0] <= truth[i]) & (truth[i] <= q[1]))
    m = np.array(means); c = np.array(cov)
    ok = np.isfinite(m[:, 0])
    out = {}
    for j, n in enumerate(PHASE3.names):
        out[n] = {"corr": float(np.corrcoef(m[ok, j], truth[ok, j])[0, 1]),
                  "mae": float(np.abs(m[ok, j] - truth[ok, j]).mean()),
                  "coverage_90": float(c[ok, j].mean())}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=Path,
                    default=Path("data/processed/phase4_educ_dataset_shards"))
    ap.add_argument("--educ_group", default="comphs")
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--train_n", type=int, default=57344)
    ap.add_argument("--n_heldout", type=int, default=1024)
    ap.add_argument("--n_post", type=int, default=600)
    ap.add_argument("--transform", choices=["log1m", "none"], default="log1m")
    ap.add_argument("--save_posterior", action="store_true",
                   help="Persist the trained posterior. Needed to evaluate on "
                        "PSID, which is the only place the delta boundary "
                        "actually bites -- held-out theta is uniform, so only "
                        "~1.3%% of it sits where truncation happens.")
    ap.add_argument("--d_model", type=int, default=EMBEDDER["d_model"])
    ap.add_argument("--n_layers", type=int, default=EMBEDDER["n_layers"])
    ap.add_argument("--n_heads", type=int, default=EMBEDDER["n_heads"])
    ap.add_argument("--embed_dim", type=int, default=EMBEDDER["output_dim"],
                   help="Embedder output width. RESULTS 19.6 adopts 64 with "
                        "--d_model 128 --n_layers 3.")
    ap.add_argument("--out", type=Path, default=Path("outputs/test_delta"))
    args = ap.parse_args()

    from hh_npe.simulator.laibson_calibration import EDUC_GROUPS
    shard_files = sorted(args.shards.glob("shard_*.npz"))
    train_sh, held_sh = split_shards(shard_files, args.train_n)
    theta_all = sample_sobol(65536, PHASE3, seed=0)
    win = dict(start_low=24, start_high=45, wave_years=2, with_age=True)

    th_tr, x_tr, pid_tr = build_windowed(train_sh, theta_all, k=1, n_waves=7,
                                         seed=0, **win)
    th_ho, x_ho, pid_ho = build_windowed(held_sh, theta_all, k=1, n_waves=7,
                                         seed=999, **win)
    by = educ_by_draw(shard_files)
    g = EDUC_GROUPS.index(args.educ_group)
    m_tr, m_ho = by[pid_tr.numpy()] == g, by[pid_ho.numpy()] == g
    th_tr, x_tr, pid_tr = th_tr[m_tr], x_tr[m_tr], pid_tr[m_tr]
    th_ho, x_ho = th_ho[m_ho][: args.n_heldout], x_ho[m_ho][: args.n_heldout]
    log.info(f"{args.educ_group}: {len(th_tr)} train rows, {len(th_ho)} held out")

    use = args.transform == "log1m"
    box = log1m_box(PHASE3) if use else PHASE3
    th_fit = to_log1m(th_tr) if use else th_tr
    log.info(f"transform={args.transform}  delta range in fit space: "
             f"[{th_fit[:,1].min():.3f}, {th_fit[:,1].max():.3f}]")

    skip = tuple(i for i, f in enumerate(FEATURES_TWOASSET_AGE) if f == "age")
    emb = TrajectoryTransformer(
        n_features=x_tr.shape[-1], seq_len=7,
        feature_mean=x_tr.mean(dim=(0, 1)), feature_std=x_tr.std(dim=(0, 1)),
        per_sequence=False, per_sequence_skip=skip,
        **{**EMBEDDER, "d_model": args.d_model, "n_heads": args.n_heads,
           "n_layers": args.n_layers, "output_dim": args.embed_dim})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.train_seed)
    post, _de, _inf = train_npe(th_fit, x_tr, embedder=emb, box=box, device=dev,
                                group_ids=pid_tr,
                                **{**TRAINING, "batch_size": 1024,
                                   "learning_rate": 1e-3})

    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_posterior:
        from hh_npe.npe.train import save_posterior
        save_posterior(post, emb, box,
                       args.out / f"posterior_7w.pt")
        log.info(f"saved posterior for {args.transform}")
    res = score(post, box, th_ho, x_ho.to(dev), args.n_post, invert=use)
    (args.out / f"{args.transform}_s{args.train_seed}.json").write_text(
        json.dumps(res, indent=2))
    print(f"\n=== {args.transform} (seed {args.train_seed}) — metrics in DELTA space ===")
    print(f"{'param':8s}{'corr':>9s}{'mae':>10s}{'coverage_90':>13s}")
    for n in PHASE3.names:
        r = res[n]
        print(f"{n:8s}{r['corr']:9.3f}{r['mae']:10.4f}{r['coverage_90']:13.3f}")


if __name__ == "__main__":
    main()
