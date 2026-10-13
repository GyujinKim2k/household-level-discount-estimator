"""Illiquid wealth by age: the model at PSID couples' own estimates against
PSID couples (RESULTS 44.5).

RESULTS 44.4 left one illiquid misfit, who holds illiquid wealth: a third of
PSID couples go from below half a year's income to above it within their 12
years, while the model's couples in the same region mostly hold it throughout
or never do. This asks whether that is timing -- the model's couples building
illiquid wealth too early or too late -- by comparing levels by age.

**Model side: each couple's posterior predictive.** Each PSID couple gets model
windows over its own ages (started so the window's age column matches its
first wave), simulated near its A2n posterior (the uncorrected arm, A2n's
headline; ``--arm corrected`` for the other). They are cut from the annual
panels stored with A2's main run and strip, so nothing is re-solved:

- ``mean``: the ``K`` stored draws nearest the couple's posterior mean, and all
  16 simulated couples of each;
- ``posterior``: ``S`` draws from the couple's posterior, approximated as
  independent normals (its mean and sd) truncated to the box, each matched to
  its nearest stored draw, and one of that draw's couples at random.

Distances are in units of the couples' median posterior sd per parameter.
Card access is not observed: a couple seen with negative liquid wealth is
matched to cardholder draws only, the rest to draws of either type (the
generation's 50/50). A2n's observation noise is added to the model's windows
as in training (illiquid sigma per couple uniform on 0.175-0.455), since
PSID's carry their own. Every couple has the same number of model windows, so
both sides weight couples equally.

**Statistics.** Per household-wave, the ratio r of illiquid wealth to the
window's mean income, by age bin: quantiles, the share at r >= 0.5 (holding)
and r < 0.1 (next to none), and median illiquid wealth and income. Per window:
holders, never-holders and the rest (RESULTS 44.1), and the rest split by
direction: ``up`` (r below 0.5 in the first wave, at or above it in the last),
``down`` (the reverse) and ``other``. Then, couple by couple, what the model
predicts for the PSID couples of each kind. PSID is shown on both illiquid
definitions: net of all debt (the estimation input) and net of secured debt
only (RESULTS 44.4).

**The model's zeros.** From the same couples' annual panels (posterior
variant, ages 20-60): the share at exactly zero illiquid wealth by age, how
often a couple drains a positive balance to zero (``drain``: Z > 0 one year, 0
the next), the balance drained, card debt and income growth in the year of a
drain against other years with a balance, whether the couples at zero by 45
held some earlier, and how long a zero spell after a drain lasts (censored at
60).

Usage::

    PYTHONPATH=. .venv/bin/python scripts/illiquid_by_age.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from hh_npe.data.windows import _panel_of, cut_windows
from scripts import optionA as oa
from scripts.illiquid_shock_calibration import HOLD, INC, groups, ratio
from scripts.noise_calibration import MAIN, PSID_X, STRIP

RUN = Path("outputs/psid_optionA2n_couples")
PSID_SECURED = Path("data/processed/psid_x_comphs_couples_illsecured.pt")
OUT = Path("outputs/optionA3/illiquid_by_age.json")
FIG = Path("figures/50_illiquid_by_age.png")
ILL = FEATURES_TWOASSET_AGE.index("illiquid_assets")
LIQ = FEATURES_TWOASSET_AGE.index("liquid_assets")
AGE = FEATURES_TWOASSET_AGE.index("age")
AGE_EDGES = (25, 30, 35, 40, 45, 50, 55, 60)
#: A window's age column is start + 1, start + 3, ... (``aggregate_waves``).
AGE_OFFSET = 1
SEED = 50
KINDS = ("holders", "never", "up", "down", "other")


def kinds(x: np.ndarray) -> dict:
    """Holders, never-holders, and the rest by direction (module docstring)."""
    g = groups(x)
    r = ratio(x)
    up = g["rest"] & (r[:, 0] < HOLD) & (r[:, -1] >= HOLD)
    down = g["rest"] & (r[:, 0] >= HOLD) & (r[:, -1] < HOLD)
    return {"holders": g["holders"], "never": g["never"], "up": up, "down": down,
            "other": g["rest"] & ~up & ~down}


def by_age(x: np.ndarray) -> list[dict]:
    """Household-wave statistics by age bin."""
    r, age = ratio(x), x[..., AGE]
    out = []
    for lo, hi in zip(AGE_EDGES[:-1], AGE_EDGES[1:], strict=True):
        m = (age >= lo) & (age < hi)
        if not m.any():
            continue
        rr = r[m]
        out.append({"ages": [lo, hi - 1], "n": int(m.sum()),
                    "r_p10_p25_p50_p75_p90": np.percentile(rr, [10, 25, 50, 75, 90]).tolist(),
                    "hold": float((rr >= HOLD).mean()), "none": float((rr < 0.1).mean()),
                    "zero": float((x[..., ILL][m] <= 0).mean()),
                    "illiquid_p50": float(np.median(x[..., ILL][m])),
                    "income_p50": float(np.median(x[..., INC][m]))})
    return out


def describe(x: np.ndarray) -> dict:
    return {"by_age": by_age(x), "kinds": {k: float(v.mean()) for k, v in kinds(x).items()}}


def neighbours(theta, card, box, post, borrower, k, s, rng):
    """Stored draws for each couple (module docstring): ``{variant: (couple,
    draw, household or -1 for all)}``."""
    scale = np.median(post["sd"], axis=0)
    trees = {c: (np.flatnonzero(m), cKDTree(theta[m] / scale))
             for c, m in (("card", card == 1), ("any", np.ones(len(card), bool)))}
    lo, hi = np.asarray(box.low), np.asarray(box.high)
    picks, dist = {}, {}
    for variant in ("mean", "posterior"):
        rows = []
        for i, (mu, sd) in enumerate(zip(post["mean"], post["sd"], strict=True)):
            idx, tree = trees["card" if borrower[i] else "any"]
            if variant == "mean":
                q = mu[None]
            else:
                q = rng.normal(mu, sd, (s * 4, len(mu)))
                q = q[((q >= lo) & (q <= hi)).all(1)][:s]
                if len(q) < s:     # a posterior mostly outside the box
                    q = np.clip(rng.normal(mu, sd, (s, len(mu))), lo, hi)
            d, j = tree.query(q / scale, k=k if variant == "mean" else 1)
            d, j = np.ravel(d), idx[np.ravel(j)]
            rows += [(i, int(t), -1 if variant == "mean" else int(rng.integers(16)))
                     for t in j]
            dist.setdefault(variant, []).append(np.abs(theta[j] - (q.mean(0) if variant == "mean"
                                                                   else q)) / scale)
        picks[variant] = np.array(rows)
    return picks, {v: np.median(np.concatenate(d), axis=0).tolist() for v, d in dist.items()}


def model_windows(shards, picks, start, noise, seed):
    """The picked couples' windows, raw (dollars), with A2n's noise, and the
    couple each belongs to; and their annual panels (no noise), ages 20-90."""
    xs, owner, annual = [], [], []
    for sh in shards:
        d = np.load(sh.path)
        lo, hi = sh.offset + int(d["lo"]), sh.offset + int(d["hi"])
        m = int(d["n_households"])
        sel = picks[(picks[:, 1] >= lo) & (picks[:, 1] < hi)]
        if not len(sel):
            continue
        rows, who = [], []
        for i, t, h in sel:
            hs = range(m) if h < 0 else (h,)
            rows += [(t - lo) * m + j for j in hs]
            who += [i] * len(hs)
        rows, who = np.array(rows), np.array(who)
        panel = {key: v[rows] for key, v in _panel_of(d).items()}
        annual.append({k: panel[k] for k in ("illiquid_assets", "liquid_assets", "income")})
        # cut_windows regroups rows by start age; the id column carries the row.
        _th, x, ids = cut_windows(panel, np.zeros((len(rows), 1)), np.arange(len(rows)),
                                  start[who][:, None], 7, 2, FEATURES_TWOASSET_AGE)
        if len(ids) != len(rows):
            raise SystemExit(f"{sh.path.name}: {len(rows) - len(ids)} windows not alive")
        xs.append(x)
        owner.append(who[ids])
    x = torch.from_numpy(np.concatenate(xs).astype(np.float64))
    x = oa.add_obs_noise(x, noise, seed=seed)
    annual = {k: np.concatenate([a[k] for a in annual]) for k in annual[0]}
    return x.numpy(), np.concatenate(owner), annual


def zeros(annual: dict) -> dict:
    """The model's zeros (module docstring), ages 20-60."""
    z, y, liq = (annual[k][:, :41] for k in ("illiquid_assets", "income", "liquid_assets"))
    held = z[:, :-1] > 0
    drain = held & (z[:, 1:] == 0)
    other = held & ~drain
    gy = np.log(y[:, 1:] / y[:, :-1])
    spell = []
    for i, t in zip(*np.nonzero(drain), strict=True):
        back = np.flatnonzero(z[i, t + 1:] > 0)
        spell.append(back[0] if len(back) else z.shape[1] - 1 - t)
    at45 = z[:, 25] == 0
    return {"zero_by_age": {a: float((z[:, a - 20] == 0).mean()) for a in range(25, 61, 5)},
            "drains_per_couple_year": float(drain.mean()),
            "share_ever_drained": float(drain.any(1).mean()),
            "drained_balance_p10_p50_p90": np.percentile(z[:, :-1][drain], [10, 50, 90]).tolist(),
            "drained_over_income_p50": float(np.median(z[:, :-1][drain] / y[:, :-1][drain])),
            "card_debt": {"drain": float((liq[:, :-1][drain] < 0).mean()),
                          "other": float((liq[:, :-1][other] < 0).mean())},
            "income_growth_mean": {"drain": float(gy[drain].mean()),
                                   "other": float(gy[other].mean())},
            "zero_at_45": float(at45.mean()),
            "zero_at_45_held_before": float((z[at45, :25] > 0).any(1).mean()),
            "zero_spell_years_p25_p50_p75": np.percentile(spell, [25, 50, 75]).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=RUN, help="A2n's PSID couples run.")
    ap.add_argument("--psid_x", type=Path, default=PSID_X)
    ap.add_argument("--psid_secured", type=Path, default=PSID_SECURED)
    ap.add_argument("--arm", choices=("uncorrected", "corrected"), default="uncorrected")
    ap.add_argument("--k", type=int, default=4, help="Nearest draws to the posterior mean.")
    ap.add_argument("--s", type=int, default=64, help="Posterior draws per couple.")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fig", type=Path, default=FIG)
    args = ap.parse_args()

    summ = json.loads((args.run / "summary.json").read_text())
    noise = summ["training_obs_noise"]
    post = dict(np.load(args.run / f"posterior_{args.arm}.npz"))
    hidx = np.load(args.run / "household_index.npy")
    psid = {"PSID, net of all debt": torch.load(args.psid_x, weights_only=False)}
    if args.psid_secured.exists():
        psid["PSID, net of secured debt"] = torch.load(args.psid_secured, weights_only=False)
    for p in psid.values():
        if list(p["features"]) != list(FEATURES_TWOASSET_AGE):
            raise SystemExit(f"PSID features {p['features']} are not {FEATURES_TWOASSET_AGE}")
    pxs = {k: p["x"].numpy().astype(np.float64)[hidx] for k, p in psid.items()}
    px = pxs["PSID, net of all debt"]
    start = (px[:, 0, AGE] - AGE_OFFSET).astype(int)
    borrower = (px[..., LIQ] < 0).any(1)

    theta, card, train, held, _p, cfg = oa.load_shards([MAIN, STRIP])
    box = oa.box_of(cfg)
    if list(post["names"]) != list(box.names):
        raise SystemExit(f"posterior names {post['names']} are not {box.names}")
    rng = np.random.default_rng(SEED)
    picks, dist = neighbours(theta, card, box, post, borrower, args.k, args.s, rng)
    print(f"{len(px)} couples ({borrower.sum()} seen borrowing); {len(theta)} stored draws. "
          "Median |matched - target| in posterior sds (beta, delta, rho, R_gamma): "
          + "; ".join(f"{v} {np.round(d, 2).tolist()}" for v, d in dist.items()))

    res = {"run": str(args.run), "arm": args.arm, "obs_noise": noise, "hold": HOLD,
           "k": args.k, "s": args.s, "match_distance_in_posterior_sd": dist, "sides": {}}
    for k, x in pxs.items():
        res["sides"][k] = describe(x)
    owners, annual = {}, {}
    for variant, pk in picks.items():
        x, owner, annual[variant] = model_windows(train + held, pk, start, noise,
                                                  seed=SEED + (variant == "posterior"))
        owners[variant] = (x, owner)
        res["sides"][f"model, posterior {variant}"] = describe(x)
    res["model_zeros"] = zeros(annual["posterior"])

    # Couple by couple: what the model predicts for each kind of PSID couple.
    x, owner = owners["posterior"]
    mk = kinds(x)
    cross = {}
    for name, p in pxs.items():
        pk = kinds(p)
        cross[name] = {}
        for kind in KINDS:
            who = np.flatnonzero(pk[kind])
            m = np.isin(owner, who)
            cross[name][kind] = {"n": int(len(who)),
                                 **{k: float(v[m].mean()) if m.any() else None
                                    for k, v in mk.items()}}
    res["predicted_kind_by_psid_kind"] = cross
    report(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    plot(res, args.fig)
    print(f"\nwrote {args.out} and {args.fig}")


def report(res: dict) -> None:
    for name, side in res["sides"].items():
        print(f"\n===== {name}")
        print("  ages    n      r p10 / p25 / p50 / p75 / p90       hold  <0.1  <=0   "
              "illiquid p50  income p50")
        for b in side["by_age"]:
            q = " / ".join(f"{v:5.2f}" for v in b["r_p10_p25_p50_p75_p90"])
            print(f"  {b['ages'][0]}-{b['ages'][1]} {b['n']:6d}  {q}  {b['hold']:.2f}  "
                  f"{b['none']:.2f}  {b['zero']:.2f}  {b['illiquid_p50']:12,.0f}  "
                  f"{b['income_p50']:10,.0f}")
        print("  windows: " + "  ".join(f"{k} {v:.2f}" for k, v in side["kinds"].items()))
    print("\nthe model's zeros (posterior variant, annual, ages 20-60):")
    for k, v in res["model_zeros"].items():
        print(f"  {k}: {np.round(v, 3).tolist() if isinstance(v, list) else v}")
    for name, tab in res["predicted_kind_by_psid_kind"].items():
        print(f"\nmodel (posterior) kinds for each kind of couple, {name}:")
        for kind, row in tab.items():
            print(f"  {kind:8s} n {row['n']:3d}: " + "  ".join(
                f"{k} {row[k]:.2f}" for k in KINDS if row[k] is not None))


def plot(res: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    style = {"PSID, net of all debt": ("k", "-"), "PSID, net of secured debt": ("0.5", "--"),
             "model, posterior mean": ("tab:orange", ":"),
             "model, posterior posterior": ("tab:blue", "-")}
    label = {"model, posterior posterior": "model, over each couple's posterior",
             "model, posterior mean": "model, at each couple's posterior mean"}
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    for name, side in res["sides"].items():
        c, ls = style[name]
        mid = [np.mean(b["ages"]) for b in side["by_age"]]
        q = np.array([b["r_p10_p25_p50_p75_p90"] for b in side["by_age"]])
        lab = label.get(name, name)
        ax[0].plot(mid, q[:, 2], color=c, ls=ls, label=lab)
        if name in ("PSID, net of all debt", "model, posterior posterior"):
            ax[0].fill_between(mid, q[:, 1], q[:, 3], color=c, alpha=0.12)
        ax[1].plot(mid, [b["hold"] for b in side["by_age"]], color=c, ls=ls, label=lab)
        ax[2].plot(mid, [b["none"] for b in side["by_age"]], color=c, ls=ls, label=lab)
    ax[0].set(title="illiquid / window mean income: median, p25-p75", xlabel="age",
              yscale="symlog")
    ax[1].set(title=f"share at >= {HOLD} x income", xlabel="age", ylim=(0, 1))
    ax[2].set(title="share below 0.1 x income", xlabel="age", ylim=(0, 1))
    ax[0].legend(fontsize=7)
    names = list(res["sides"])
    w = 0.8 / len(names)
    for j, name in enumerate(names):
        v = [res["sides"][name]["kinds"][k] for k in KINDS]
        ax[3].bar(np.arange(len(KINDS)) + (j - (len(names) - 1) / 2) * w, v, w,
                  color=style[name][0], label=label.get(name, name))
    ax[3].set_xticks(range(len(KINDS)), KINDS)
    ax[3].set(title="couples' 12 years: kind of window", ylim=(0, 1))
    fig.suptitle("Illiquid wealth by age: PSID couples against the model at their own "
                 "A2n estimates (RESULTS 44.5)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
