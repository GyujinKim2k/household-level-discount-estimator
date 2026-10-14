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
- ``targeted`` (RESULTS 46): new simulated couples where PSID's population
  sits. Of the 2,048 trainable thetas above, 3 have beta's logit below -2.5
  and delta's above 3, where 40% of PSID's fitted population lies: the
  generation's proposal was half uniform and half on beta >= 0.40. There the
  learned ``p(s | theta)`` put ``s`` for beta 0.3 sd too low, and V2's fitted
  population came out pulled off beta's floor. ``TARGET_BLOCKS`` blocks of 16
  thetas from PSID's fitted g (RESULTS 45, M1) with its covariance widened
  ``TARGET_WIDEN``-fold in sd, simulated as V2's (``hierarchical_validation
  .simulate``: the training solver and seed pool, PSID couples' start ages,
  A2n's noise), R_gamma per block, card type alternating by block (the
  generation's half and half), ``TARGET_PER`` households each. A likelihood
  is conditional on theta, so where its training thetas come from changes
  only where it is accurate. ``train --variant targeted`` adds them all to
  the trainable pairs.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/hierarchical_summary.py pairs
    PYTHONPATH=. .venv/bin/python scripts/hierarchical_summary.py train
    PYTHONPATH=. .venv/bin/python scripts/hierarchical_summary.py targeted
    PYTHONPATH=. .venv/bin/python scripts/hierarchical_summary.py train --variant targeted
"""

from __future__ import annotations

import argparse
import json
import time
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
#: SBC draw ids are offset past every held-out id in the pairs file, and
#: targeted ones past those.
SBC_OFFSET, TARGET_OFFSET = 10_000_000, 20_000_000
TARGET_BLOCKS, TARGET_PER, TARGET_WIDEN = 24, 32, 1.5
#: Blocks simulated per saved part (the run resumes by part).
TARGET_PART = 4


def summaries(net: hp.Network, xt: torch.Tensor, n_draws: int, seed: int,
              chunk: int = 2048, retries: int = 3) -> np.ndarray:
    """``summary`` for many households, in chunks (the draws are not kept)."""
    torch.manual_seed(seed)
    out = []
    for c0 in range(0, len(xt), chunk):
        for attempt in range(1 + retries):
            try:
                d = sample_draws(net.members, xt[c0:c0 + chunk], n_draws, net.fb.low,
                                 net.fb.high, invert=net.inv, log_weight=net.log_weight)
                break
            except AssertionError:
                # nflows' spline inverse occasionally asserts in float32 on one
                # noise draw (a null set at one seed, RESULTS 46); the chunk is
                # redrawn with the generator's next noise.
                if attempt == retries:
                    raise
                print(f"  summaries: rows {c0}-{c0 + chunk}: spline inverse asserted; "
                      f"redrawing ({attempt + 1} of {retries})", flush=True)
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


def training_pairs(out: Path, variant: str):
    """``s``, theta, draw ids and the trainable mask: the pairs file, plus
    every targeted household for ``variant`` 'targeted'."""
    d = np.load(out / "pairs.npz")
    s, theta, draw, train = d["s"], d["theta"], d["draw"], split(d["draw"], d["sbc"])
    if variant == "targeted":
        t = np.load(out / "targeted.npz")
        if int(t["n_draws"]) != int(d["n_draws"]):
            raise SystemExit("targeted.npz and pairs.npz use different draw counts")
        s, theta = np.concatenate([s, t["s"]]), np.concatenate([theta, t["theta"]])
        draw = np.concatenate([draw, t["draw"]])
        train = np.r_[train, np.ones(len(t["s"]), bool)]
    return s, theta, draw, train, d["lo"], d["hi"], int(d["n_draws"])


def mdn_path(out: Path, variant: str) -> Path:
    return out / "mdn.pt" if variant == "base" else out / variant / "mdn.pt"


def cmd_targeted(args) -> None:
    from scripts import hierarchical_validation as hv

    fit = json.loads((hp.OUT / "summary" / "uncorrected" / "summary.json").read_text())
    p = {k: torch.tensor(v, dtype=torch.float64)
         for k, v in fit["families"]["M1"]["params"].items()}
    lo, hi = np.array(fit["box"][0]), np.array(fit["box"][1])
    ref = pop.Reference(hp.FAMILIES["M1"], p, scales=(TARGET_WIDEN,))
    n = args.blocks * 16
    th = pop.from_box_logit(ref.sample_u(n, seed=hv.SEED + 31), lo, hi).reshape(args.blocks, 16, -1)
    th[:, :, 3] = th[:, :1, 3]
    th = th.reshape(-1, 4)
    card = np.repeat(np.arange(args.blocks) % 2, 16)
    parts = args.out / "targeted_sims"
    parts.mkdir(parents=True, exist_ok=True)
    xs = []
    for k, b0 in enumerate(range(0, args.blocks, TARGET_PART)):
        f = parts / f"part{k}.npz"
        rows = slice(16 * b0, 16 * min(b0 + TARGET_PART, args.blocks))
        if not f.exists():
            t0 = time.time()
            x = hv.simulate(th[rows], card[rows], args.per, hv.SEED + 3000 + k)
            np.savez(f, x=x, theta=th[rows], card=card[rows], per=args.per)
            print(f"part {k}: {len(x)} households at {len(th[rows])} thetas in "
                  f"{time.time() - t0:.0f}s", flush=True)
        d = np.load(f)
        if not (np.array_equal(d["theta"], th[rows]) and int(d["per"]) == args.per):
            raise SystemExit(f"{f} was simulated with other settings")
        xs.append(d["x"])
    x = np.concatenate(xs)
    net = hp.Network(args.run_dirs)
    xt, _ = oa.transform(torch.from_numpy(x))
    s = summaries(net, xt, args.n_draws, args.seed)
    idx = np.repeat(np.arange(n), args.per)
    np.savez(args.out / "targeted.npz", s=s, theta=th[idx], draw=TARGET_OFFSET + idx,
             card=card[idx], n_draws=args.n_draws, lo=net.lo, hi=net.hi)
    u = pop.box_logit(np.clip(th, lo + 1e-9, hi - 1e-9), lo, hi)
    print(f"wrote {args.out / 'targeted.npz'}: {len(s)} households at {n} thetas; "
          f"{int(((u[:, 0] < -2.5) & (u[:, 1] > 3)).sum())} thetas with beta's logit < -2.5 "
          "and delta's > 3")


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
    s, theta, draw, tr, lo, hi, n_draws = training_pairs(args.out, args.variant)
    u = pop.box_logit(np.clip(theta, lo + 1e-9, hi - 1e-9), lo, hi)
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
    info.update({"k": args.k, "variant": args.variant, "n_untouched": int(test.sum()),
                 "untouched_nll": nll, "calibration_untouched": calib, "n_draws": n_draws})
    path = mdn_path(args.out, args.variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    sl.save(mdn, path, info)
    path.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"MDN k={args.k}: {info['epochs']} epochs; validation NLL {info['val_nll']:.4f}, "
          f"untouched NLL {nll:.4f} ({test.sum()} households)")
    for n, c in calib.items():
        print(f"  {n:8s} PIT KS p {c['ks_p']:.3f}, 90% coverage {c['cover90']:.3f}, "
              f"deciles {c['pit_deciles']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("pairs", "train", "targeted"))
    ap.add_argument("--variant", choices=("base", "targeted"), default="base",
                    help="train: the pairs file alone (RESULTS 45) or with the targeted "
                         "couples (RESULTS 46).")
    ap.add_argument("--run_dirs", type=Path, nargs="+", default=hp.RUNS)
    ap.add_argument("--n_draws", type=int, default=hp.N_DRAWS)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=TARGET_BLOCKS)
    ap.add_argument("--per", type=int, default=TARGET_PER)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    {"pairs": cmd_pairs, "train": cmd_train, "targeted": cmd_targeted}[args.cmd](args)


if __name__ == "__main__":
    main()
