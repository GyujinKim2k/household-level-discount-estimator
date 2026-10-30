"""RESULTS 58: every noise arm's ensemble on held-out windows carrying every
arm's noise.

``check_optionA2_heldout.py`` scores an ensemble on windows carrying its own
training noise, so each arm is graded on a different test. Here the held-out
draws, households and noise draws (z and the per-household u, seeded as in
``optionA.windows``) are the same in every cell; only the noise range differs,
sigma = lo + (hi - lo) u. A column is one candidate truth for PSID's error:
- coverage says whether each network is calibrated if that is the truth;
- log q compares the networks on the same windows (a proper score);
- corr, MAE and contraction are recovery, as in the held-out check.

Arms must share everything but the noise (box, transforms, mask, windows).

Usage::

    PYTHONPATH=. .venv/bin/python scripts/cross_noise_heldout.py \\
        --arm v2 outputs/optionA2v2 --arm typical outputs/optionA2v2_noise_typical \\
        --arm gross outputs/optionA2v2_noise_gross \\
        --shards data/processed/couples_v2_dataset_shards \\
                 data/processed/couples_v2_strip_dataset_shards \\
        --out outputs/v2_noise_arms/cross_noise_heldout.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.evaluation.weighted import sample_weighted
from scripts import optionA as oa
from scripts.check_optionA2_heldout import heldout, heldout_log_weight

N = oa.BOX.names
SHOW = ("cov90", "below5", "above95", "corr", "mae_over_prior_sd", "contraction")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs=2, action="append", required=True,
                    metavar=("NAME", "RUN_DIR"), help="RUN_DIR holds the members s*/.")
    ap.add_argument("--shards", type=Path, nargs="+", required=True,
                    help="The ensembles' shard directories, in training order.")
    ap.add_argument("--n_draws", type=int, default=4000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    arms = {}
    for name, d in args.arm:
        dirs = sorted(p for p in Path(d).glob("s[0-9]*") if (p / "results.json").exists())
        members, proposal, cfg = oa.load_members(dirs, args.device)
        arms[name] = {"dirs": dirs, "members": members, "proposal": proposal, "cfg": cfg,
                      "spec": oa.spec_of(cfg)}
    first = next(iter(arms.values()))
    base = dataclasses.replace(first["spec"], obs_noise=())
    for name, a in arms.items():
        if (dataclasses.replace(a["spec"], obs_noise=()) != base
                or a["proposal"] != first["proposal"]):
            raise SystemExit(f"{name} differs from {args.arm[0][0]} in more than the noise")
        if {k: v for k, v in a["cfg"].items() if k != "obs_noise"} != \
                {k: v for k, v in first["cfg"].items() if k != "obs_noise"}:
            raise SystemExit(f"{name}'s training config differs from {args.arm[0][0]}'s "
                             f"in more than the noise")

    theta_all, _card, _tr, held_f, _p, sh_cfg = oa.load_shards(args.shards)
    if oa.box_of(sh_cfg) != base.box:
        raise SystemExit(f"{args.shards} do not cover the ensembles' box")
    oa.check_flow_lag(first["cfg"], held_f)
    inv, lw = oa._weighting(first["proposal"], base)
    fb = base.flow_box()

    res = {"arms": {n: {"run_dirs": list(map(str, a["dirs"])),
                        "noise": {f: [lo, hi] for f, lo, hi in a["spec"].obs_noise}}
                    for n, a in arms.items()},
           "n_draws": args.n_draws, "cells": {}}
    pick = None
    for wname, wa in arms.items():                      # the windows' noise
        th, x, pid, _f = oa.windows(held_f, theta_all, seed=oa.HELD_SEED,
                                    noise=wa["spec"].obs_noise, mask=base.mask_patterns())
        if pick is None:
            pick = oa.one_per_draw(pid, seed=0)
            pid0 = pid[pick]
        th, x, pid = th[pick], x[pick], pid[pick]
        assert torch.equal(pid, pid0)                   # the same households in every column
        t = th.numpy().astype(float)
        w = np.exp(heldout_log_weight(sh_cfg, pid.numpy(), t))
        res["n_heldout"], res["ess"] = len(t), float(w.sum() ** 2 / (w ** 2).sum())
        for nname, na in arms.items():                  # the network
            torch.manual_seed(0)
            r = sample_weighted(na["members"], x, args.n_draws, fb.low, fb.high, invert=inv,
                                log_weight=lw, truth=t)
            s = heldout(r, t, w, oa._log_q(na["members"], th, x, base, per_row=True), base)
            res["cells"][f"{nname} on {wname} noise"] = s
            print(f"network {nname:8s} windows {wname:8s} log q {s['log_q_weighted']:7.3f}  "
                  + "  ".join(f"{n} cov {s[n]['cov90']:.3f} contr {s[n]['contraction']:.3f}"
                              for n in N), flush=True)

    names = list(arms)
    print(f"\n{res['n_heldout']} held-out draws, ESS {res['ess']:.0f}; rows: network, "
          "columns: the windows' noise")
    for k in ("log_q_weighted",) + SHOW:
        for n in [None] if k == "log_q_weighted" else N:
            print(f"\n{k}" + (f", {n}" if n else "") + "".join(f"{c:>10s}" for c in names))
            for a in names:
                vals = [res["cells"][f"{a} on {c} noise"] for c in names]
                print(f"  {a:8s}" + "".join(
                    f"{(v[k] if n is None else v[n][k]):10.3f}" for v in vals))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
