"""RESULTS 66: why the noise-free 7-wave network reads PSID couples' rho so low.

Under none_w07 PSID couples' median posterior mean of rho is 0.76, against
2.5-2.7 under every noisy arm, and its population fit finds no theta in the box
(RESULTS 62.1 saw the same on the old box). Two checks of whether that is
PSID's noise read as preference:

``households``
    Couples simulated at the theta the noisy w07 network's population fit
    settles on (``laibson_comparison``'s last Newton point: under w07 with
    v2's noise these couples' mean posterior means are PSID couples' within
    0.2 se), and at Laibson et al.'s point, scored by none_w07 with no noise
    and with v2's, typical's and gross's noise, each in full and on one
    feature at a time. w07 scores the same windows, with no noise and v2's,
    for reference. Noise seeded as ``laibson_comparison.Arm.score``. Figure
    124: the couples' posterior means under none_w07, PSID against the
    simulated couples with no noise, v2's and gross's, 68/95%, and rho's
    distribution. The means go to ``noise_free_rho_means.npz`` (PSID-derived,
    never committed); ``figure`` redraws 124 from them.

``recovery``
    Held-out households over the whole box (one per draw, the held-out
    check's draws), with v2's noise and without: each network's median
    posterior mean of rho and R_gamma by true rho, so the pull is read at
    every rho, not at one theta.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/noise_free_rho.py households
    PYTHONPATH=. .venv/bin/python scripts/noise_free_rho.py recovery
    PYTHONPATH=. .venv/bin/python scripts/noise_free_rho.py figure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts import household_resolution as hr
from scripts import laibson_comparison as lc
from scripts import optionA as oa
from scripts.sbc_optionA2 import Members, posterior

NAMES = hr.NAMES
RUNS = {"none_w07": Path("outputs/waves_none/w07"), "w07": Path("outputs/waves/w07")}
PSID = {"none_w07": Path("outputs/waves_none/psid_w07_couples"),
        "w07": Path("outputs/waves/psid_w07_couples")}
FIT = Path("outputs/waves_none/laibson_comparison_noise_free.json")
NOISE_ARMS = {"typical": Path("outputs/optionA2v2_b010_noise_typical/s0"),
              "gross": Path("outputs/optionA2v2_b010_noise_gross/s0")}
OUT = Path("outputs/waves_none/noise_free_rho.json")
MEANS = Path("outputs/waves_none/noise_free_rho_means.npz")     # PSID-derived: never committed
RHO_BANDS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5)
FIGURE_SETS = {"psid": ("#B03A2E", "PSID couples"),
               "none": ("#4A9A5B", "simulated at w07's fit, no noise"),
               "v2": ("#1F4E79", "the same couples, v2's noise"),
               "gross": ("#D68910", "the same couples, gross's noise")}
HIST_ONLY = {"typical": ("#8E6BB8", "the same couples, typical's noise")}
LIMITS = {**hr.LIMITS, "beta": (0.10, 1.00), "R_gamma": (1.021, 1.075)}       # the box


def noise_of(member: Path) -> tuple:
    return oa.spec_of(json.loads((member / "results.json").read_text())["_config"]).obs_noise


def conditions(v2: tuple) -> dict:
    """Noise conditions: none, then v2's, typical's and gross's noise, each in
    full and on each noisy feature alone (Spec ``obs_noise`` tuples)."""
    out = {"none": ()}
    for arm, noise in (("v2", v2), *((k, noise_of(p)) for k, p in NOISE_ARMS.items())):
        out[arm] = noise
        out |= {f"{arm}, {f} only": ((f, lo, hi),) for f, lo, hi in noise if hi > 0}
    return out


def summary(mean: np.ndarray) -> dict:
    return {"median_of_means": np.median(mean, 0).tolist(), "mean_of_means": mean.mean(0).tolist(),
            "share_rho_below_1": float((mean[:, 2] < 1).mean())}


def score(members: Members, spec, x_raw: np.ndarray, noise: tuple) -> np.ndarray:
    x = oa.add_obs_noise(torch.from_numpy(x_raw), noise, seed=lc.SEED + 77)
    x, _f = oa.transform(x, masked=spec.masked)
    return posterior(members, x, None, spec, lc.N_DRAWS)["mean"]


def households(args) -> dict:
    fit = json.loads(FIT.read_text())
    w07 = fit["arms"]["w07"]
    points = {"w07 fit": ("ii_w07_3", np.array(w07["last_simulated_theta"])),
              "Laibson et al.": ("laibson", np.array(fit["theta_laibson"]))}
    nets = {k: Members(sorted(d.glob("s[0-9]*")), args.device) for k, d in RUNS.items()}
    specs = {k: oa.spec_of(m.cfg) for k, m in nets.items()}
    v2 = specs["w07"].obs_noise
    if specs["none_w07"].obs_noise:
        raise SystemExit("none_w07 was trained with noise")
    res = {"psid": {k: summary(hr.psid(str(d))["mean"]) for k, d in PSID.items()},
           "noise": {k: [list(e) for e in v] for k, v in conditions(v2).items()},
           "points": {}, "means": {}}
    for pname, (cache, theta) in points.items():
        sim = np.load(lc.OUT / f"sim_{cache}.npz")
        if not np.allclose(sim["theta"], theta):
            raise SystemExit(f"sim_{cache}: simulated at {sim['theta']}, not {theta}")
        row = {"theta": theta.tolist(), "n": len(sim["x"]), "scored": {}}
        for cname, noise in conditions(v2).items():
            for net in ("none_w07", "w07") if cname in ("none", "v2") else ("none_w07",):
                mean = score(nets[net], specs[net], sim["x"], noise)
                row["scored"][f"{net}, {cname}"] = summary(mean)
                if pname == "w07 fit" and net == "none_w07" and cname in {
                        **FIGURE_SETS, **HIST_ONLY}:
                    res["means"][cname] = mean
                print(f"{pname:15s} {net:9s} {cname:28s} median "
                      + " ".join(f"{v:.4f}" for v in np.median(mean, 0)), flush=True)
        res["points"][pname] = row
    res["means"]["psid"] = hr.psid(str(PSID["none_w07"]))["mean"]
    return res


def report_households(res: dict) -> None:
    print("\nmedian of couples' posterior means (beta delta rho R_gamma) / share rho < 1")
    for k, s in res["psid"].items():
        print(f"  PSID couples under {k:9s}             "
              + " ".join(f"{v:.4f}" for v in s["median_of_means"])
              + f"  {s['share_rho_below_1']:.2f}")
    for pname, row in res["points"].items():
        print(f"{pname} {np.round(row['theta'], 4).tolist()} ({row['n']} couples)")
        for k, s in row["scored"].items():
            print(f"  {k:41s}" + " ".join(f"{v:.4f}" for v in s["median_of_means"])
                  + f"  {s['share_rho_below_1']:.2f}")


def figure(res: dict, means: dict, path: Path) -> None:
    """Couples' posterior means under none_w07: PSID, and couples simulated at
    the noisy fit's theta with no noise, v2's and gross's (68/95%), and rho's
    distribution (typical's noise too)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = res["points"]["w07 fit"]["theta"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (i, j) in zip(axes[:2], ((0, 2), (3, 2)), strict=True):
        for key, (color, label) in FIGURE_SETS.items():
            hr._contour(ax, np.asarray(means[key]), i, j, color, label, limits=LIMITS)
        ax.plot(theta[i], theta[j], "k*", ms=12, label="w07's fit (the simulated θ)")
        ax.set_xlabel(hr.LABEL[NAMES[i]])
        ax.set_ylabel(hr.LABEL[NAMES[j]])
    axes[0].legend(fontsize=8, loc="upper right")
    ax = axes[2]
    bins = np.linspace(*LIMITS["crra"], 46)
    for key, (color, label) in {**FIGURE_SETS, **HIST_ONLY}.items():
        rho = np.asarray(means[key])[:, 2]
        ax.hist(rho, bins, density=True, histtype="step", lw=1.6, color=color,
                label=f"{label}: median {np.median(rho):.2f}")
    ax.axvline(theta[2], color="k", lw=1, ls=":", label=f"the simulated ρ {theta[2]:.2f}")
    ax.set_xlabel("posterior mean of " + hr.LABEL["crra"])
    ax.set_ylabel("density")
    ax.legend(fontsize=7.5)
    fig.suptitle("The noise-free 7-wave network (none_w07): couples' posterior means, 68% (solid) "
                 "and 95% (dashed). Noise read as preference: with\nPSID-level noise (typical, "
                 "gross) the simulated couples' ρ falls to PSID couples'; v2's lighter noise goes "
                 "part of the way (RESULTS 66)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def recovery(args) -> dict:
    """Median posterior mean of rho and R_gamma by true rho on held-out
    households (one per draw), with v2's noise and without, under both nets."""
    nets = {k: Members(sorted(d.glob("s[0-9]*")), args.device) for k, d in RUNS.items()}
    specs = {k: oa.spec_of(m.cfg) for k, m in nets.items()}
    cfg = nets["w07"].cfg
    theta_all, _card, _tr, held_f, _p, _sh = oa.load_shards([Path(s) for s in cfg["shards"]])
    window = oa.window_of(cfg)
    res = {"bands": list(RHO_BANDS), "cells": {}}
    pick = None
    for cname, noise in (("none", ()), ("v2", specs["w07"].obs_noise)):
        th, x, pid, _f = oa.windows(held_f, theta_all, seed=oa.HELD_SEED, noise=noise,
                                    window=window)
        if pick is None:
            pick = oa.one_per_draw(pid, seed=0)
        th, x = th[pick].numpy().astype(float), x[pick]
        band = np.digitize(th[:, 2], RHO_BANDS[1:-1])
        for net in RUNS:
            mean = posterior(nets[net], x, None, specs[net], lc.N_DRAWS)["mean"]
            rows = [{"n": int((band == b).sum()),
                     "true_median": np.median(th[band == b], 0).tolist(),
                     "median_of_means": np.median(mean[band == b], 0).tolist()}
                    for b in range(len(RHO_BANDS) - 1)]
            res["cells"][f"{net} on {cname} noise"] = {
                "by_true_rho": rows, "median_error": np.median(mean - th, 0).tolist()}
    res["n"] = int(len(pick))
    return res


def report_recovery(res: dict) -> None:
    print(f"\n{res['n']} held-out households; median posterior mean of rho (R_gamma) "
          "by true rho")
    edges = res["bands"]
    bands = zip(edges[:-1], edges[1:], strict=True)
    print(" " * 24 + "".join(f"{f'{a:g}-{b:g}':>16s}" for a, b in bands))
    first = next(iter(res["cells"].values()))["by_true_rho"]
    print(f"  {'true median':22s}" + "".join(
        f"{r['true_median'][2]:7.2f} ({r['true_median'][3]:.3f})" for r in first))
    for k, c in res["cells"].items():
        print(f"  {k:22s}" + "".join(
            f"{r['median_of_means'][2]:7.2f} ({r['median_of_means'][3]:.3f})"
            for r in c["by_true_rho"]))
    print("median error (mean - truth), beta delta rho R_gamma")
    for k, c in res["cells"].items():
        print(f"  {k:22s} " + " ".join(f"{v:+.4f}" for v in c["median_error"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("households", "recovery", "figure"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fig", type=Path, default=Path("figures/124_noise_free_rho.png"))
    args = ap.parse_args()
    res = json.loads(args.out.read_text()) if args.out.exists() else {}
    if args.command == "figure":
        figure(res["households"], dict(np.load(MEANS)), args.fig)
        return
    if args.command == "households":
        r = households(args)
        report_households(r)
        np.savez(MEANS, **r.pop("means"))
        figure(r, dict(np.load(MEANS)), args.fig)
    else:
        r = recovery(args)
        report_recovery(r)
    res[args.command] = r
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
