"""The summary likelihood for the population layer (RESULTS 45).

The network's own implied likelihood (``q / p~``) failed the uniform test:
on simulated households drawn uniformly over the box, the fitted population
of beta came out a third as wide in logit as the truth. So each household is
summarised by ``s``, its A2n posterior mean in ``u`` (each parameter's logit on
the box), and ``p(s | theta)`` is learned from simulation
(``hh_npe.evaluation.summary_likelihood``).

- ``pairs``: ``s`` for every simulated household of the SBC set (1,024
  uniform draws x 16) and the held-out shards (1,536 draws x 16), windows and
  observation noise exactly as in their checks, ``n_draws`` posterior draws
  each, importance-weighted to the uniform prior as ``optionA.psid``.
- ``train``: the mixture density network on the held-out pairs and a seeded
  random half of the SBC set's draws (``split``; validation split by draw).
  The **other half is never trained on**: it is the uniform test and V2's
  pool (``hierarchical_validation.py``). Its calibration is reported on them:
  per-coordinate PIT and 90% coverage of ``s`` given the true theta.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/hierarchical_summary.py pairs
    PYTHONPATH=. .venv/bin/python scripts/hierarchical_summary.py train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from hh_npe.evaluation import population as pop
from hh_npe.evaluation import summary_likelihood as sl
from hh_npe.evaluation.weighted import sample_draws
from scripts import hierarchical_population as hp
from scripts import optionA as oa
from scripts.noise_calibration import MAIN, STRIP

OUT = Path("outputs/hier_summary")
SBC = Path("data/processed/optionA2_sbc16_shards")
#: SBC draw ids are offset past every held-out id in the pairs file.
SBC_OFFSET = 10_000_000


def summaries(net: hp.Network, xt: torch.Tensor, n_draws: int, seed: int,
              chunk: int = 2048) -> np.ndarray:
    """``summary`` for many households, in chunks (the draws are not kept)."""
    torch.manual_seed(seed)
    out = []
    for c0 in range(0, len(xt), chunk):
        d = sample_draws(net.members, xt[c0:c0 + chunk], n_draws, net.fb.low, net.fb.high,
                         invert=net.inv, log_weight=net.log_weight)
        out.append(pop.posterior_mean_u(d["theta"], d["log_w"], net.lo, net.hi))
    return np.concatenate(out)


def simulated(net: hp.Network):
    """The SBC set's and held-out households' windows (with the training
    observation noise), theta and draw ids."""
    from scripts.sbc_optionA2 import load_set

    theta_all, _card, _train, held, _prop, cfg = oa.load_shards([MAIN, STRIP])
    th_h, x_h, pid_h = oa.windows_raw(held, theta_all, seed=oa.HELD_SEED,
                                      noise=net.spec.obs_noise)
    theta_s, files_s, _c = load_set(SBC, net.box)
    th_s, x_s, pid_s = oa.windows_raw(files_s, theta_s, seed=oa.SBC_SEED,
                                      noise=net.spec.obs_noise)
    return {"theta": torch.cat([th_h, th_s]).numpy().astype(float),
            "x": torch.cat([x_h, x_s]),
            "draw": np.concatenate([pid_h.numpy(), SBC_OFFSET + pid_s.numpy()]),
            "sbc": np.r_[np.zeros(len(th_h), bool), np.ones(len(th_s), bool)]}


#: Seed of the SBC set's random half that may be trained on.
SPLIT_SEED = 45_001


def split(draw: np.ndarray, sbc: np.ndarray) -> np.ndarray:
    """True where a household may be trained on: every held-out household and
    a seeded random half of the SBC set's draws. Not alternate draws: the SBC
    set is a Sobol sequence, whose consecutive points alternate halves of the
    delta axis, so odd and even draws each cover half of delta's range."""
    ids = np.unique(draw[sbc])
    train = ids[np.random.default_rng(SPLIT_SEED).permutation(len(ids))[:len(ids) // 2]]
    return ~sbc | np.isin(draw, train)


def cmd_pairs(args) -> None:
    net = hp.Network(args.run_dirs)
    sim = simulated(net)
    xt, _ = oa.transform(sim["x"])
    print(f"{len(xt)} simulated households ({sim['sbc'].sum()} SBC); {args.n_draws} draws each",
          flush=True)
    s = summaries(net, xt, args.n_draws, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "pairs.npz", s=s, theta=sim["theta"], draw=sim["draw"], sbc=sim["sbc"],
             n_draws=args.n_draws, lo=net.lo, hi=net.hi)
    print(f"wrote {args.out / 'pairs.npz'}")


def marginal_cdf(mdn: sl.SummaryMDN, s: torch.Tensor, u: torch.Tensor) -> np.ndarray:
    """Per-coordinate CDF of ``s`` under ``p(s | theta)`` (each coordinate's
    marginal of the mixture), for PIT."""
    with torch.no_grad():
        logit, mu, log_diag, off = mdn._params(u)
        ds = mdn.d_s
        L = torch.diag_embed(torch.exp(log_diag))
        i, j = torch.tril_indices(ds, ds, -1)
        L[..., i, j] = off
        var = (L @ L.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1)      # (n, k, ds)
        z = (s - mdn.s_mean) / mdn.s_sd
        cdf = stats.norm.cdf((z[:, None, :] - mu).numpy() / np.sqrt(var.numpy()))
        return (torch.softmax(logit, -1).numpy()[..., None] * cdf).sum(1)


def cmd_train(args) -> None:
    d = np.load(args.out / "pairs.npz")
    lo, hi = d["lo"], d["hi"]
    u = pop.box_logit(np.clip(d["theta"], lo + 1e-9, hi - 1e-9), lo, hi)
    s, draw, sbc = d["s"], d["draw"], d["sbc"]
    tr = split(draw, sbc)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdn, info = sl.train_mdn(s[tr], u[tr], draw[tr], k=args.k, seed=args.seed, device=dev)
    test = ~tr
    with torch.no_grad():
        nll = float(-mdn.log_prob(torch.from_numpy(s[test]), torch.from_numpy(u[test])).mean())
    pit = marginal_cdf(mdn, torch.from_numpy(s[test]), torch.from_numpy(u[test]))
    names = ["beta", "delta", "crra", "R_gamma"]
    calib = {n: {"ks_p": float(stats.kstest(pit[:, j], "uniform").pvalue),
                 "cover90": float(((pit[:, j] >= 0.05) & (pit[:, j] <= 0.95)).mean()),
                 "pit_deciles": np.histogram(pit[:, j], bins=10, range=(0, 1))[0].tolist()}
             for j, n in enumerate(names)}
    info.update({"k": args.k, "n_untouched": int(test.sum()), "untouched_nll": nll,
                 "calibration_untouched": calib, "n_draws": int(d["n_draws"])})
    sl.save(mdn, args.out / "mdn.pt", info)
    (args.out / "mdn.json").write_text(json.dumps(info, indent=2))
    print(f"MDN k={args.k}: {info['epochs']} epochs; validation NLL {info['val_nll']:.4f}, "
          f"untouched NLL {nll:.4f} ({test.sum()} households)")
    for n, c in calib.items():
        print(f"  {n:8s} PIT KS p {c['ks_p']:.3f}, 90% coverage {c['cover90']:.3f}, "
              f"deciles {c['pit_deciles']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("pairs", "train"))
    ap.add_argument("--run_dirs", type=Path, nargs="+", default=hp.RUNS)
    ap.add_argument("--n_draws", type=int, default=hp.N_DRAWS)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    {"pairs": cmd_pairs, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
