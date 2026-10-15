"""Posterior widths of the A2, A2w and A2n ensembles on the same simulated
households (RESULTS 48).

Figures 36, 39 and 42 each show one held-out household, but each picks its own
household by rule, so they differ in age, theta and card. This puts the same
households through all three ensembles.

**Households.** The main run's two held-out shards (draws 31,744-32,767),
which all three ensembles hold out. Windows are cut once (``windows_raw``,
``HELD_SEED``), one per draw (``one_per_draw``, seed 0): 1,024 households, the
same rows ``check_optionA2_heldout`` uses for A2. A2 and A2w see them
noise-free, as they were trained. A2n sees them with its observation noise,
seeded as in its own pipeline (the main run's group, offset 0), so it sees
exactly the windows it was checked on; this is verified against
``windows_raw`` over A2n's own shards. Each household's drawn noise sigma per
feature is recorded.

**Figure households.** Each figure's household is rebuilt by the plotting
script's own rule (``plot_optionA2_households.simulated``: pool of 300, the
saved pick) and checked against the figure's saved draw id. All three lie in
the main run's group, so every ensemble sees the same window.

**Reported.**
- each figure household's 90% widths under each ensemble, and figure 56: each
  one's posteriors under the three ensembles, overlaid
  (``56_same_household_fig{36,39,42}.png``);
- on the 1,024: median 90% widths, and per-household width ratios A2w / A2 and
  A2n / A2w; widths by window start age;
- A2n: log 90% width regressed on each feature's drawn sigma, scaled to its
  range, so a coefficient is the change from the bottom of the range to the top.

**Signal against noise** (``--signal_only``, CPU, no ensembles). On all 16,384
windows of the main run's held-out group, the IQR-based variance of 1-wave and
6-wave changes, ((p75 - p25) / 1.349)^2, on the network's input scale (log
over mean income for consumption; asinh over mean income for wealth), without
and with A2n's noise: for every window, and for illiquid holders (illiquid
>= 0.5 x mean income in every wave, noise-free, as in RESULTS 44.2). Income
carries no noise, so the mean-income anchor is the same in both.

**Noise-free models on noisy windows** (``--noisy_inputs``). A2 and A2w were
trained and checked on exact windows. Real windows carry noise. On the same
1,024 households this scores A2 and A2w on their exact windows and on A2n's
noisy ones, and A2n on its noisy ones: 90% coverage of the true theta per
parameter and for all four jointly, unweighted and weighted by p / q_held as in
``check_optionA2_heldout``, the median 90% width, and the median |mean - truth|
in posterior sds.

**Illiquid noise off** (``--illiquid_off DIR``, RESULTS 52). A2w, A2n with
only the illiquid noise off (DIR) and A2n, each at members 0 and 1 and on its
own windows of the same households. The arm's consumption and liquid noise is
A2n's draw for draw (checked), so A2w -> arm is what the consumption and
liquid noise cost and arm -> A2n what the illiquid noise costs: median 90%
widths and per-household ratios, for every household and split into illiquid
holders and the rest, illiquid's share of the mean log width gap, 90%
coverage, and figure 61 (``61_illiquid_off_fig{36,39,42}.png``): each figure
household's three posteriors overlaid. The arm is also scored on A2n's windows,
which carry the illiquid noise as PSID's data do: its coverage there says
whether its precision would hold on PSID.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/compare_optionA2_widths.py
    PYTHONPATH=. .venv/bin/python scripts/compare_optionA2_widths.py --signal_only
    PYTHONPATH=. .venv/bin/python scripts/compare_optionA2_widths.py --noisy_inputs
    PYTHONPATH=. .venv/bin/python scripts/compare_optionA2_widths.py \
        --illiquid_off outputs/optionA2n_sens/illoff
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.evaluation.plots import contour_corner
from hh_npe.evaluation.weighted import resample, sample_weighted
from scripts import optionA as oa
from scripts.plot_optionA2_households import weighted_draws

MAIN = Path("data/processed/couples_dataset_shards")
STRIP = Path("data/processed/couples_strip_dataset_shards")
MODELS = {"A2": ("outputs/optionA2", [MAIN]),
          "A2w": ("outputs/optionA2w", [MAIN, STRIP]),
          "A2n": ("outputs/optionA2n", [MAIN, STRIP])}
FIGURES = {"A2": "figures/36_simulated_household_optionA2.npz",
           "A2w": "figures/39_simulated_household_optionA2w.npz",
           "A2n": "figures/42_simulated_household_optionA2n.npz"}
POOL = 300
N = oa.BOX.names
NOISY_ILL = "illiquid off, A2n windows"


def noise_sigmas(x: torch.Tensor, noise, seed: int) -> dict[str, np.ndarray]:
    """Each row's sigma per noisy feature, drawn exactly as ``add_obs_noise``
    draws it (same generator, same order)."""
    g = torch.Generator().manual_seed(seed)
    torch.randn(x.shape, generator=g, dtype=x.dtype)
    u = torch.rand((x.shape[0], x.shape[-1]), generator=g, dtype=x.dtype)
    out = {}
    for f, lo, hi in noise:
        j = oa.FEATURES_TWOASSET_AGE.index(f)
        out[f] = (lo + (hi - lo) * u[:, j]).numpy().astype(float)
    return out


def figure_row(files, theta_all, noise, fig: dict) -> int:
    """The plotting script's pick, as a row of ``windows_raw(files)``."""
    _th, _x, pid = oa.windows_raw(files, theta_all, seed=oa.HELD_SEED, noise=noise)
    pick1 = oa.one_per_draw(pid, seed=0)
    pool = np.sort(np.random.default_rng(1).choice(pick1, min(POOL, len(pick1)),
                                                   replace=False))
    row = int(pool[int(fig["pick"])])
    if int(pid[row]) != int(fig["draw"]):
        raise SystemExit(f"rebuilt pick is draw {int(pid[row])}, the figure's is "
                         f"{int(fig['draw'])}")
    return row


def widths(r: dict) -> np.ndarray:
    return r["hi"] - r["lo"]


def iqr_var(v: np.ndarray) -> float:
    q75, q25 = np.percentile(v, [75, 25])
    return float(((q75 - q25) / 1.349) ** 2)


def signal_table(x0: torch.Tensor, xn: torch.Tensor) -> dict:
    """IQR-based variance of 1- and 6-wave changes per noisy feature, on the
    network's input scale, noise-free and with A2n's noise."""
    t0, feats = oa.transform(x0)
    tn, _f = oa.transform(xn)
    t0, tn = t0.numpy(), tn.numpy()
    j_ill = oa.FEATURES_TWOASSET_AGE.index("illiquid_assets")
    ybar = x0[..., oa.FEATURES_TWOASSET_AGE.index("income")].mean(1, keepdim=True)
    holders = (x0[..., j_ill] >= 0.5 * ybar).all(1).numpy()
    out = {"holders_share": float(holders.mean())}
    print(f"\nchanges on the network's scale, IQR-based variance; illiquid holders "
          f"{holders.mean():.3f} of {len(holders)} windows")
    print(f"  {'feature':16s}{'windows':>9s}{'V1 free':>10s}{'V1 noisy':>10s}"
          f"{'V6 free':>10s}{'V6 noisy':>10s}")
    for f in ("consumption", "liquid_assets", "illiquid_assets"):
        j = feats.index(f)
        for grp, k in (("all", np.ones(len(holders), bool)), ("holders", holders)):
            if grp == "holders" and f != "illiquid_assets":
                continue
            row = {}
            for lag in (1, 6):
                row[f"V{lag}_free"] = iqr_var((t0[k, lag:, j] - t0[k, :-lag, j]).ravel())
                row[f"V{lag}_noisy"] = iqr_var((tn[k, lag:, j] - tn[k, :-lag, j]).ravel())
            out[f"{f}, {grp}"] = row
            print(f"  {f[:12] + ', ' + grp:16s}{int(k.sum()):9d}" + "".join(
                f"{row[c]:10.4f}" for c in ("V1_free", "V1_noisy", "V6_free", "V6_noisy")))
    return out


def noisy_inputs(args, ens, x0, xn, rows, figs, t_all, pid) -> None:
    """A2 and A2w on exact and on noisy windows, A2n on noisy ones. ``t_all``
    holds the 1,024 households' theta, then the figure households'."""
    from scripts.check_optionA2_heldout import heldout_log_weight
    t = t_all[:len(rows)]
    *_, cfg_main = oa.load_shards([MAIN])
    w = np.exp(heldout_log_weight(cfg_main, pid[rows].numpy(), t))
    w /= w.sum()
    fig_rows = list(figs.values())
    runs = [("A2", "exact"), ("A2", "noisy"), ("A2w", "exact"), ("A2w", "noisy"),
            ("A2n", "noisy")]
    out = {"n": len(rows), "ess": float(1 / (w ** 2).sum())}
    print(f"\n=== noise-free models on noisy windows: {len(rows)} households "
          f"(weighted ESS {out['ess']:.0f}) ===")
    print(f"{'':14s}{'90% coverage: unweighted / weighted':>48s}{'joint':>13s}"
          f"{'median width':>36s}{'median |err|/sd':>30s}")
    for m, kind in runs:
        members, spec, inv, lw = ens[m]
        xr = xn if kind == "noisy" else x0
        x, _f = oa.transform(xr[rows + fig_rows])
        fb = spec.flow_box()
        torch.manual_seed(0)
        r = sample_weighted(members, x, args.n_draws, fb.low, fb.high, invert=inv,
                            log_weight=lw)
        n = len(rows)
        lo, hi, mean, sd = (r[k][:n] for k in ("lo", "hi", "mean", "sd"))
        inside = (lo <= t) & (t <= hi)
        ok = np.isfinite(mean).all(1)
        ww = w[ok] / w[ok].sum()
        cov_u, cov_w = inside[ok].mean(0), ww @ inside[ok]
        joint_u, joint_w = inside[ok].all(1).mean(), ww @ inside[ok].all(1)
        width = np.median((hi - lo)[ok], 0)
        zerr = np.median(np.abs(mean - t)[ok] / sd[ok], 0)
        out[f"{m} on {kind}"] = {"cov90_unweighted": cov_u.tolist(),
                                 "cov90_weighted": cov_w.tolist(),
                                 "joint_unweighted": float(joint_u),
                                 "joint_weighted": float(joint_w),
                                 "median_width": width.tolist(),
                                 "median_abs_err_over_sd": zerr.tolist(), "n_ok": int(ok.sum())}
        print(f"{m + ' on ' + kind:14s}" + " ".join(f"{a:.3f}/{b:.3f}" for a, b in
                                                    zip(cov_u, cov_w, strict=True))
              + f"   {joint_u:.3f}/{joint_w:.3f}   " + " ".join(f"{v:.4g}" for v in width)
              + "   " + " ".join(f"{v:.2f}" for v in zerr))
        for fname, k in zip(figs, range(n, n + len(fig_rows)), strict=True):
            out[f"{m} on {kind}"][f"figure_{fname}_inside90"] = (
                (r["lo"][k] <= t_all[k]) & (t_all[k] <= r["hi"][k])).tolist()
            out[f"{m} on {kind}"][f"figure_{fname}_abs_err_over_sd"] = (
                np.abs(r["mean"][k] - t_all[k]) / r["sd"][k]).tolist()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "noisy_inputs.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}/noisy_inputs.json")


def illiquid_off(args, x0, xn, rows, figs, t_all, pid, held, theta_all, card_all) -> None:
    """A2w, the illiquid-noise-off arm and A2n at members 0 and 1, each on its
    own windows of the same households (RESULTS 52). The arm's consumption and
    liquid noise is A2n's draw for draw, so A2w -> arm is the consumption and
    liquid noise and arm -> A2n is the illiquid noise."""
    from scripts.check_optionA2_heldout import heldout_log_weight
    dirs = {"A2w": Path(MODELS["A2w"][0]), "illiquid off": args.illiquid_off,
            "A2n": Path(MODELS["A2n"][0])}
    ens = {}
    for name, d in dirs.items():
        members, proposal, cfg = oa.load_members([d / "s0", d / "s1"], args.device)
        spec = oa.spec_of(cfg)
        ens[name] = (members, spec, *oa._weighting(proposal, spec))
        print(f"{name}: {len(members)} members; {spec.describe()}")
    off, a2n = ({f: (lo, hi) for f, lo, hi in ens[m][1].obs_noise} for m in ("illiquid off", "A2n"))
    if off.pop("illiquid_assets", (0, 0)) != (0, 0) or a2n.pop("illiquid_assets")[1] == 0 \
            or off != a2n:
        raise SystemExit(f"the arm's noise is not A2n's with the illiquid noise off: {off}")

    seed_n = oa.NOISE_SEED + oa.HELD_SEED
    xo = oa.add_obs_noise(x0, ens["illiquid off"][1].obs_noise, seed=seed_n)
    _t, xo_own, _p = oa.windows_raw(held, theta_all, seed=oa.HELD_SEED,
                                    noise=ens["illiquid off"][1].obs_noise)
    j_ill = oa.FEATURES_TWOASSET_AGE.index("illiquid_assets")
    other = [j for j in range(x0.shape[-1]) if j != j_ill]
    if not (torch.equal(xo_own[:len(xo)], xo) and torch.equal(xo[..., other], xn[..., other])
            and torch.equal(xo[..., j_ill], x0[..., j_ill])):
        raise SystemExit("the arm's windows are not A2n's with the illiquid noise removed")
    print("the arm's held-out windows reproduced: A2n's, illiquid noise-free")
    windows = {"A2w": x0, "illiquid off": xo, "A2n": xn}

    n = len(rows)
    rows_all = rows + list(figs.values())
    t = t_all[:n]
    *_, cfg_main = oa.load_shards([MAIN])
    w = np.exp(heldout_log_weight(cfg_main, pid[rows].numpy(), t))
    w /= w.sum()
    ybar = x0[rows, :, oa.FEATURES_TWOASSET_AGE.index("income")].mean(1, keepdim=True)
    holders = (x0[rows, :, j_ill] >= 0.5 * ybar).all(1).numpy()
    groups = {"all": np.ones(n, bool), "illiquid holders": holders, "non-holders": ~holders}
    res, W = {}, {}
    for m, (members, spec, inv, lw) in ens.items():
        x, _f = oa.transform(windows[m][rows_all])
        fb = spec.flow_box()
        torch.manual_seed(0)
        res[m] = sample_weighted(members, x, args.n_draws, fb.low, fb.high, invert=inv,
                                 log_weight=lw, truth=t_all)
        W[m] = widths(res[m])
    # The arm on windows that carry A2n's illiquid noise, as PSID's do: is its
    # precision kept, or is it overconfident (RESULTS 49.1's test for A2w)?
    members, spec, inv, lw = ens["illiquid off"]
    x, _f = oa.transform(xn[rows_all])
    fb = spec.flow_box()
    torch.manual_seed(0)
    res[NOISY_ILL] = sample_weighted(members, x, args.n_draws, fb.low, fb.high, invert=inv,
                                     log_weight=lw, truth=t_all)
    W[NOISY_ILL] = widths(res[NOISY_ILL])

    out = {"n": n, "holders": int(holders.sum()), "ess": float(1 / (w ** 2).sum()),
           "median_width": {}, "ratio": {}, "illiquid_share_of_log_gap": {}, "cov90": {},
           "figure_households": {}}
    print(f"\n=== {n} common households, two members each; {holders.sum()} illiquid holders ===")
    print(f"{'median 90% width':34s}" + "".join(f"{v:>10s}" for v in N))
    for g, k in groups.items():
        out["median_width"][g] = {}
        for m in W:
            med = np.nanmedian(W[m][:n][k], 0)
            out["median_width"][g][m] = med.tolist()
            print(f"{m + ', ' + g:34s}" + "".join(f"{v:10.4g}" for v in med))
    print(f"\n{'median ratio (share wider)':34s}" + "".join(f"{v:>14s}" for v in N))
    for a, b in (("illiquid off", "A2w"), ("A2n", "illiquid off"), ("A2n", "A2w")):
        out["ratio"][f"{a}/{b}"] = {}
        for g, k in groups.items():
            rat = W[a][:n][k] / W[b][:n][k]
            med, wider = np.nanmedian(rat, 0), np.nanmean(rat > 1, 0)
            out["ratio"][f"{a}/{b}"][g] = {"median": med.tolist(), "share_wider": wider.tolist()}
            print(f"{a + '/' + b + ', ' + g:34s}" + "".join(
                f"{med[j]:7.2f} ({wider[j]:.2f})" for j in range(4)))
    print(f"\n{'illiquid share of the log gap':34s}" + "".join(f"{v:>10s}" for v in N))
    for g, k in groups.items():
        gap = np.nanmean(np.log(W["A2n"][:n][k] / W["A2w"][:n][k]), 0)
        ill = np.nanmean(np.log(W["A2n"][:n][k] / W["illiquid off"][:n][k]), 0)
        out["illiquid_share_of_log_gap"][g] = (ill / gap).tolist()
        print(f"{g:34s}" + "".join(f"{v:10.2f}" for v in ill / gap))
    print(f"\n90% coverage, unweighted / weighted (ESS {out['ess']:.0f})")
    for m, r in res.items():
        inside = (r["lo"][:n] <= t) & (t <= r["hi"][:n])
        ok = np.isfinite(r["mean"][:n]).all(1)
        ww = w[ok] / w[ok].sum()
        out["cov90"][m] = {"unweighted": inside[ok].mean(0).tolist(),
                           "weighted": (ww @ inside[ok]).tolist(),
                           **{f"unweighted, {g}": inside[ok & k].mean(0).tolist()
                              for g, k in groups.items() if g != "all"}}
        print(f"  {m:28s}" + " ".join(f"{a:.3f}/{b:.3f}" for a, b in zip(
            inside[ok].mean(0), ww @ inside[ok], strict=True)))
        for g, k in groups.items():
            if g != "all":
                print(f"    {g + ', unweighted':26s}" + " ".join(
                    f"{v:11.3f}" for v in inside[ok & k].mean(0)))
    print("\nfigure households: 90% widths")
    for i, fname in enumerate(figs, start=n):
        num = Path(FIGURES[fname]).name[:2]
        out["figure_households"][num] = {m: W[m][i].tolist() for m in W}
        print(f"  figure {num}'s household: " + "; ".join(
            f"{m} " + " ".join(f"{v:.3g}" for v in W[m][i]) for m in W))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "illiquid_off.json").write_text(json.dumps(out, indent=1))
    np.savez(args.out / "illiquid_off.npz", rows=np.array(rows_all), theta=t_all, holders=holders,
             **{f"{m.replace(',', '').replace(' ', '_')}_{k}": res[m][k]
                for m in res for k in ("mean", "lo", "hi")})
    print(f"wrote {args.out}/illiquid_off.json, illiquid_off.npz")

    box = ens["A2w"][1].box
    for k, (fname, row) in enumerate(figs.items()):
        series = {}
        for m, (members, spec, inv, lw) in ens.items():
            x, _f = oa.transform(windows[m][row:row + 1])
            torch.manual_seed(0)
            d, wt = weighted_draws(members, x, args.n_raw, spec.flow_box(), inv, lw)
            series[m] = resample(d, wt, args.n_plot, seed=0)
        num = Path(FIGURES[fname]).name[:2]
        path = args.fig.with_name(f"61_illiquid_off_fig{num}.png")
        a0 = int(x0[row, 0, -1])
        contour_corner(series, box, truth={"true θ": t_all[n + k]}, truth_colors=("#d62728",),
                       reflect_axes=("delta",), path=path,
                       title=f"Figure {num}'s household (draw {int(pid[row])}, ages {a0}-"
                             f"{a0 + 12}, {'cardholder' if card_all[pid[row]] else 'no card'}), "
                             "members 0 and 1\nA2w (no noise), A2n with the illiquid noise off, "
                             "and A2n (RESULTS 52)")
        print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_draws", type=int, default=4000)
    ap.add_argument("--n_raw", type=int, default=40000)
    ap.add_argument("--n_plot", type=int, default=20000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("outputs/optionA2_widths"))
    ap.add_argument("--fig", type=Path, default=Path("figures/56_same_household"))
    ap.add_argument("--signal_only", action="store_true",
                    help="Only the signal-against-noise table (CPU).")
    ap.add_argument("--noisy_inputs", action="store_true",
                    help="Only A2 and A2w scored on A2n's noisy windows.")
    ap.add_argument("--illiquid_off", type=Path, metavar="DIR",
                    help="Only A2w, the illiquid-noise-off arm in DIR (s0, s1) and A2n, "
                         "members 0 and 1, each on its own windows (RESULTS 52).")
    args = ap.parse_args()

    if args.signal_only:
        cfg = json.loads((Path(MODELS["A2n"][0]) / "s0" / "results.json").read_text())["_config"]
        noise_n = oa.spec_of(cfg).obs_noise
        theta_all, _c, _tr, held, _p, _cfg = oa.load_shards([MAIN, STRIP])
        _th, x0, _pid = oa.windows_raw([s for s in held if s.offset == 0], theta_all,
                                       seed=oa.HELD_SEED)
        xn = oa.add_obs_noise(x0, noise_n, seed=oa.NOISE_SEED + oa.HELD_SEED)
        res = signal_table(x0, xn)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "signal.json").write_text(json.dumps(res, indent=1))
        print(f"wrote {args.out}/signal.json")
        return

    ens = {}
    for name, (d, _shards) in MODELS.items():
        members, proposal, cfg = oa.load_members([Path(d) / f"s{i}" for i in range(5)],
                                                 args.device)
        spec = oa.spec_of(cfg)
        ens[name] = (members, spec, *oa._weighting(proposal, spec))
        print(f"{name}: {len(members)} members; {spec.describe()}")
    noise_n = ens["A2n"][1].obs_noise
    if ens["A2"][1].obs_noise or ens["A2w"][1].obs_noise:
        raise SystemExit("A2 / A2w carry observation noise; this script assumes not")

    theta_all, card_all, _tr, held, _p, _cfg = oa.load_shards([MAIN, STRIP])
    main_held = [s for s in held if s.offset == 0]
    th, x0, pid = oa.windows_raw(main_held, theta_all, seed=oa.HELD_SEED)
    seed_n = oa.NOISE_SEED + oa.HELD_SEED
    xn = oa.add_obs_noise(x0, noise_n, seed=seed_n)
    sig = noise_sigmas(x0, noise_n, seed_n)
    _t, xn_own, pid_own = oa.windows_raw(held, theta_all, seed=oa.HELD_SEED, noise=noise_n)
    if not (torch.equal(xn_own[:len(xn)], xn) and torch.equal(pid_own[:len(pid)], pid)):
        raise SystemExit("A2n's own held-out windows differ from the rebuilt ones")
    print(f"main run's held-out group: {len(pid)} windows; A2n's noise reproduced")

    rows = list(oa.one_per_draw(pid, seed=0))
    figs = {}
    for name, (_d, shards) in MODELS.items():
        files = main_held if shards == [MAIN] else held
        fig = np.load(FIGURES[name])
        row = figure_row(files, theta_all, noise_n if name == "A2n" else (), fig)
        if row >= len(pid):
            raise SystemExit(f"figure {name}'s household is not in the main run's group")
        figs[name] = row
        print(f"figure {name}: draw {int(fig['draw'])}, row {row}, ages "
              f"{int(x0[row, 0, -1])}-{int(x0[row, -1, -1])}, card {int(card_all[pid[row]])}")
    rows_all = rows + list(figs.values())
    t = th[rows_all].numpy().astype(float)
    if args.noisy_inputs:
        noisy_inputs(args, ens, x0, xn, rows, figs, t, pid)
        return
    if args.illiquid_off:
        illiquid_off(args, x0, xn, rows, figs, t, pid, held, theta_all, card_all)
        return

    res = {}
    for name, (members, spec, inv, lw) in ens.items():
        xr = xn if name == "A2n" else x0
        x, _f = oa.transform(xr[rows_all])
        fb = spec.flow_box()
        torch.manual_seed(0)
        res[name] = sample_weighted(members, x, args.n_draws, fb.low, fb.high, invert=inv,
                                    log_weight=lw, truth=t)
    n = len(rows)
    W = {m: widths(r) for m, r in res.items()}
    age = x0[rows_all, 0, -1].numpy()
    summary = {"n": n, "figure_households": {}, "median_width": {}, "ratio": {},
               "by_start_age": {}, "a2n_noise_regression": {}}

    print("\n=== figure households: 90% widths (beta, delta, rho, R_gamma) ===")
    for k, (fname, row) in enumerate(figs.items()):
        i = n + k
        tr = t[i]
        summary["figure_households"][fname] = {
            "draw": int(pid[row]), "truth": tr.tolist(), "start_age": int(age[i]),
            **{m: W[m][i].tolist() for m in W}}
        print(f"  figure {fname}'s household (draw {int(pid[row])}, ages {int(age[i])}-"
              f"{int(age[i]) + 12}, truth {np.round(tr, 4).tolist()})")
        for m in W:
            print(f"    {m:4s} " + "  ".join(f"{N[j]} {W[m][i, j]:.4f}" for j in range(4)))

    print(f"\n=== {n} common households ===")
    print(f"{'':10s}" + "".join(f"{v:>10s}" for v in N))
    for m in W:
        med = np.nanmedian(W[m][:n], 0)
        summary["median_width"][m] = med.tolist()
        print(f"{'width ' + m:10s}" + "".join(f"{v:10.4f}" for v in med))
    for a, b in (("A2w", "A2"), ("A2n", "A2w"), ("A2n", "A2")):
        rat = W[a][:n] / W[b][:n]
        q = np.nanpercentile(rat, [25, 50, 75], 0)
        wider = np.nanmean(rat > 1, 0)
        summary["ratio"][f"{a}/{b}"] = {"p25": q[0].tolist(), "median": q[1].tolist(),
                                        "p75": q[2].tolist(), "share_wider": wider.tolist()}
        print(f"{a + '/' + b:10s}" + "".join(f"{q[1, j]:6.2f} ({wider[j]:.2f})"
                                             for j in range(4))
              + "   median ratio (share wider)")

    print("\nmedian 90% width of rho by window start age:")
    bins = [(20, 35), (35, 45), (45, 70)]
    for lo, hi in bins:
        k = (age[:n] >= lo) & (age[:n] < hi)
        row = {m: np.nanmedian(W[m][:n][k], 0).tolist() for m in W}
        summary["by_start_age"][f"{lo}-{hi - 1}"] = {"n": int(k.sum()), **row}
        print(f"  {lo}-{hi - 1} (n {k.sum():4d}): " + "  ".join(
            f"{m} {row[m][2]:.2f}" for m in W))

    print("\nA2n: log 90% width on drawn sigma, scaled to its range (bottom -> top):")
    feats = [f for f, lo, hi in noise_n if hi > lo]
    rng = {f: (lo, hi) for f, lo, hi in noise_n}
    Xs = np.column_stack([np.ones(n)] + [(sig[f][rows] - rng[f][0]) / (rng[f][1] - rng[f][0])
                                         for f in feats])
    for j, p in enumerate(N):
        y = np.log(W["A2n"][:n, j])
        ok = np.isfinite(y)
        b, *_ = np.linalg.lstsq(Xs[ok], y[ok], rcond=None)
        resid = y[ok] - Xs[ok] @ b
        se = np.sqrt(np.diag(np.linalg.inv(Xs[ok].T @ Xs[ok])) * resid.var(ddof=Xs.shape[1]))
        summary["a2n_noise_regression"][p] = {f: {"factor": float(np.exp(b[i + 1])),
                                                  "se_log": float(se[i + 1])}
                                              for i, f in enumerate(feats)}
        print(f"  {p:8s} " + "  ".join(f"{f} x{np.exp(b[i + 1]):.2f} (se {se[i + 1]:.2f})"
                                       for i, f in enumerate(feats)))

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "widths.npz", rows=np.array(rows_all), draw=pid[rows_all].numpy(),
             theta=t, start_age=age, **{f"sigma_{f}": sig[f][rows_all] for f in sig},
             **{f"{m}_{k}": res[m][k] for m in res for k in ("mean", "sd", "lo", "hi")})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {args.out}/widths.npz, summary.json")

    box = ens["A2w"][1].box
    for k, (fname, row) in enumerate(figs.items()):
        series = {}
        for m, (members, spec, inv, lw) in ens.items():
            xr = xn if m == "A2n" else x0
            x, _f = oa.transform(xr[row:row + 1])
            torch.manual_seed(0)
            d, w = weighted_draws(members, x, args.n_raw, spec.flow_box(), inv, lw)
            series[f"{m}" + (" (observation noise)" if m == "A2n" else "")] = resample(
                d, w, args.n_plot, seed=0)
        num = Path(FIGURES[fname]).name[:2]
        path = args.fig.with_name(f"{args.fig.name}_fig{num}.png")
        a0 = int(age[n + k])
        contour_corner(series, box, truth={"true θ": t[n + k]}, truth_colors=("#d62728",),
                       reflect_axes=("delta",), path=path,
                       title=f"Figure {num}'s household (draw {int(pid[row])}, ages {a0}-"
                             f"{a0 + 12}, {'cardholder' if card_all[pid[row]] else 'no card'})"
                             "\nthe same window under A2, A2w and A2n (RESULTS 48)")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
