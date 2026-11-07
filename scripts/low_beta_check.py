"""RESULTS 64.12: is one household's beta below 0.3 worse resolved than the
rest of the box?

RESULTS 63.4 and 64.6 read beta below 0.3 as unresolved at any panel length
and traced it to behaviour at the credit limit. This checks that reading on
the waves arms' SBC households (truth known, uniform prior on the box):

* bands: the same measures in every 0.2-wide band of beta, so the strip below
  0.3 is compared like for like (a correlation inside a narrow band is low
  anywhere), and the edge yardstick applied at both ends;
* behaviour: the SBC sets' stored annual panels (noise-free), ages 24-64, by
  true beta: does behaviour stop moving below 0.3?
* summaries: a classifier on simple per-household summaries of the very
  windows each arm scored (same rows, same noise; and noise-free), against
  the network's posterior means;
* the limit: cardholders' posterior sd by the share of their window's waves
  spent at the credit limit (the reading of why extra waves help less there).

CPU only. Writes ``outputs/waves/low_beta_check.json`` and figure 111.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/low_beta_check.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from hh_npe.data.waves import FEATURES_TWOASSET_AGE as F
from hh_npe.simulator import grids
from scripts import waves_experiment as we

DATA = Path("data/processed")
SETS = (DATA / "optionA2v2_sbc16_shards", DATA / "optionA2v2_sbc16_betastrip_shards")
MAIN = DATA / "couples_v2_dataset_shards"
XJUMP = 1000.0                 # the full grid's liquid step, so the limit lands on it
BANDS = ((0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.8, 1.0))
#: bands 0.1 wide, 0.2 apart: the network's ranking of one against the other
PAIRS = (((0.1, 0.2), (0.3, 0.4)), ((0.3, 0.4), (0.5, 0.6)),
         ((0.5, 0.6), (0.7, 0.8)), ((0.7, 0.8), (0.9, 1.0)))
AGES = np.arange(we.START_LOW, we.LAST_WAVE_START + 2)          # 24..64
BEHAVIOUR = ("at_limit", "in_debt", "no_wealth", "debt_over_limit",
             "illiquid_over_mean_income", "consumption_over_income")
SUMMARIES = ("waves_at_limit", "waves_in_debt", "liquid_mean", "liquid_min",
             "illiquid_mean", "illiquid_slope", "c_over_y_mean", "c_over_y_sd", "income_mean")
LIMIT_CUTS = (0.0, 1 / 3, 2 / 3, 1.0 + 1e-9)
COLORS_N = ("#222222", "#EF8636", "#3B75AF")


def band(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """``lo <= t < hi``, with the box's top (1.0) inside the last band."""
    return (t >= lo) & ((t < hi) if hi < 1.0 else (t <= hi))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """P(score of a y household > score of a not-y one), ties counted half."""
    y = np.asarray(y, bool)
    r = rankdata(score)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bands(s: dict) -> dict:
    """Per 0.2-wide band of true beta: corr(mean, truth), median sd, median
    90% width, coverage; the ranking of bands 0.2 apart; the edge yardstick
    at both ends (95th percentile below 0.3, 5th above 0.8)."""
    t, m, sd, lo, hi = s["truth"][:, 0], *(s[k][:, 0] for k in ("mean", "sd", "lo", "hi"))
    out = {"bands": [], "pairs": []}
    for a, b in BANDS:
        k = band(t, a, b)
        out["bands"].append({"lo": a, "hi": b, "n": int(k.sum()),
                             "corr": float(np.corrcoef(m[k], t[k])[0, 1]),
                             "median_sd": float(np.median(sd[k])),
                             "median_width_90": float(np.median(hi[k] - lo[k])),
                             "coverage_90": float(((lo[k] <= t[k]) & (t[k] <= hi[k])).mean())})
    for (a0, a1), (b0, b1) in PAIRS:
        k = band(t, a0, a1) | band(t, b0, b1)
        out["pairs"].append({"low": [a0, a1], "high": [b0, b1],
                             "auc_beta_mean": auc(m[k], t[k] >= b0)})
    out["p95_below_0.3_at_0.1_0.2"] = float((hi[band(t, 0.1, 0.2)] < 0.3).mean())
    out["p5_above_0.8_at_0.9_1.0"] = float((lo[band(t, 0.9, 1.0)] > 0.8).mean())
    return out


def sbc_sets(box):
    """The SBC sets as sbc_optionA2 loads them, with each draw's card type."""
    from scripts import sbc_optionA2 as sb

    theta_all, files, _sets, _r = sb.load_sets(SETS, box, train=MAIN)
    card = np.full(len(theta_all), -1)
    for f in files:
        d = np.load(f.path)
        lo, hi = int(d["lo"]), int(d["hi"])
        assert np.array_equal(d["theta"], theta_all[f.offset + lo: f.offset + hi])
        card[f.offset + lo: f.offset + hi] = d["card"]
    assert (card >= 0).all()
    return theta_all, files, card


def _mean_alive(v: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Each household's mean over the years it is alive (the original head)."""
    return np.where(alive, v, 0).sum(1) / np.maximum(alive.sum(1), 1)


def behaviour(files, theta_all, card) -> dict:
    """The SBC sets' annual panels, ages 24-64 alive, by true beta
    (``we.BETA_BINS``), card type, all delta and delta >= 0.95: each
    household's share of years at the limit (within one liquid step of it),
    in debt and with no wealth (liquid and illiquid both under one step),
    and its mean debt / limit, illiquid / mean income, consumption / income."""
    cols = AGES - 20
    lim = grids.credit_limit(AGES.astype(float), XJUMP)
    ym = grids.mean_income(AGES.astype(float))
    rows, who = [], []
    for f in files:
        d = np.load(f.path)
        m, lo = int(d["n_households"]), int(d["lo"])
        alive = d["panel_t_age"][:, cols] == cols[None, :]
        liq, ill = d["panel_liquid_assets"][:, cols], d["panel_illiquid_assets"][:, cols]
        inc, con = d["panel_income"][:, cols], d["panel_consumption"][:, cols]
        rows.append(np.column_stack([_mean_alive(v, alive) for v in (
            liq <= -lim + XJUMP, liq < 0, (liq <= XJUMP) & (ill <= XJUMP),
            -np.minimum(liq, 0) / lim, ill / ym, con / np.maximum(inc, 1.0))]))
        who.append(f.offset + lo + np.arange(len(liq)) // m)
    r, draw = np.concatenate(rows), np.concatenate(who)
    beta, delta, c = theta_all[draw, 0], theta_all[draw, 1], card[draw]
    out = {}
    for dlo, tag in ((0.85, "all_delta"), (0.95, "delta_0.95")):
        for ct, cname in ((1, "card"), (0, "no_card")):
            out[f"{cname}_{tag}"] = [
                {"lo": a, "hi": b, "n": int(k.sum()),
                 **{n: float(r[k, j].mean()) for j, n in enumerate(BEHAVIOUR)}}
                for a, b in we.BETA_BINS
                for k in [band(beta, a, b) & (c == ct) & (delta >= dlo)]]
    out["no_card_years_in_debt"] = float(r[c == 0, 1].mean())   # grid rounding, < one step
    return out


def summaries(x) -> np.ndarray:
    """Per household, from its raw windows (dollars; NaN-safe): shares of waves
    at the limit (90% of it) and in debt, liquid / mean income (mean, min),
    illiquid / mean income (mean, slope per wave), consumption / income (mean,
    sd), income / mean income."""
    x = np.asarray(x, float)
    i = {f: F.index(f) for f in F}
    age = x[..., i["age"]]
    lim, ym = grids.credit_limit(age, XJUMP), grids.mean_income(age)
    liq, ill = x[..., i["liquid_assets"]], x[..., i["illiquid_assets"]]
    inc, con = x[..., i["income"]], x[..., i["consumption"]]
    t = np.arange(x.shape[1]) - (x.shape[1] - 1) / 2
    ill_y = ill / ym
    slope = np.nansum(t * (ill_y - np.nanmean(ill_y, 1, keepdims=True)), 1) / (t ** 2).sum()
    cy = con / np.maximum(inc, 1.0)
    def share(cond):                       # a missing wave counts neither way
        return np.nanmean(np.where(np.isnan(liq), np.nan, cond), 1)

    return np.column_stack([
        share(liq <= -0.9 * lim), share(liq < 0), np.nanmean(liq / ym, 1),
        np.nanmin(liq / ym, 1), np.nanmean(ill_y, 1), slope, np.nanmean(cy, 1),
        np.nanstd(cy, 1), np.nanmean(inc / ym, 1)])


def arm_windows(n: int, theta_all, files, root: Path):
    """The arm's SBC windows exactly as sbc_optionA2 cut them (same rows, same
    noise), raw, and the same rows noise-free; checked against its sbc16.npz."""
    from scripts import optionA as oa

    cfg = json.loads((we.members_of(we.arm_dir(n, root))[0] / "results.json")
                     .read_text())["_config"]
    spec, win = oa.spec_of(cfg), oa.window_of(cfg)
    th, x, pid = oa.windows_raw(files, theta_all, seed=oa.SBC_SEED, noise=spec.obs_noise,
                                mask=spec.mask_patterns(), window=win)
    _t0, x0, pid0 = oa.windows_raw(files, theta_all, seed=oa.SBC_SEED, window=win)
    s = dict(np.load(we.arm_dir(n, root) / "sbc" / "sbc16.npz"))
    if not (np.array_equal(pid.numpy(), s["pid"]) and np.array_equal(pid0.numpy(), s["pid"])
            and np.allclose(th.numpy(), s["truth"])):
        raise SystemExit(f"{n} waves: the rebuilt windows are not the SBC run's rows")
    return s, x.numpy(), x0.numpy()


def cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str = "logit") -> float:
    """Out-of-fold AUC, folds by draw (the 16 households of a draw share
    theta). ``logit``: logistic regression on standardised features (low
    variance; it loses nothing on the network's beta mean); ``spline``: the
    same on cubic splines of each feature; ``gb``: gradient boosting, the
    flexible check (it loses up to 0.12 on the network's own means here)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    model = {"logit": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
             "spline": lambda: make_pipeline(StandardScaler(), SplineTransformer(n_knots=5),
                                             LogisticRegression(max_iter=2000)),
             "gb": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                          random_state=0)}[kind]()
    p = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(5),
                          method="predict_proba")[:, 1]
    return auc(p, y)


#: (key, classifier, features) of :func:`against_summaries`
CLASSIFIERS = (("logit_network", "logit", "net"), ("logit_summaries", "logit", "sum"),
               ("spline_summaries", "spline", "sum"), ("spline_noise_free", "spline", "sum0"),
               ("logit_network_and_summaries", "logit", "net+sum"),
               ("logit_network_and_noise_free", "logit", "net+sum0"),
               ("gb_network", "gb", "net"), ("gb_summaries", "gb", "sum"),
               ("gb_network_and_summaries", "gb", "net+sum"))


def against_summaries(s: dict, x: np.ndarray, x0: np.ndarray) -> list:
    """Per pair of bands: the network's beta mean (no fitting), and out-of-fold
    classifiers (:data:`CLASSIFIERS`) on its four posterior means, on the
    summaries (as scored, and noise-free), and on both."""
    t, mean, pid = s["truth"][:, 0], s["mean"], s["pid"]
    S, S0 = summaries(x), summaries(x0)
    out = []
    for (a0, a1), (b0, b1) in PAIRS:
        k = band(t, a0, a1) | band(t, b0, b1)
        y, g = t[k] >= b0, pid[k]
        feats = {"net": mean[k], "sum": S[k], "sum0": S0[k],
                 "net+sum": np.column_stack([mean[k], S[k]]),
                 "net+sum0": np.column_stack([mean[k], S0[k]])}
        out.append({"low": [a0, a1], "high": [b0, b1], "n": int(k.sum()),
                    "network_beta_mean": auc(mean[k, 0], y),
                    **{key: cv_auc(feats[f], y, g, kind) for key, kind, f in CLASSIFIERS}})
    return out


def at_limit(s: dict, x0: np.ndarray, card: np.ndarray) -> list:
    """Cardholders in two bands of true beta: median posterior sd of beta by
    the share of their (noise-free) window's waves at the limit."""
    t, sd = s["truth"][:, 0], s["sd"][:, 0]
    share = summaries(x0)[:, 0]
    holder = card[s["pid"]] == 1
    out = []
    for a, b in ((0.1, 0.3), (0.3, 0.5)):
        for c0, c1 in zip(LIMIT_CUTS[:-1], LIMIT_CUTS[1:], strict=True):
            k = band(t, a, b) & holder & (share >= c0) & (share < c1)
            out.append({"lo": a, "hi": b, "share_lo": c0, "share_hi": min(c1, 1.0),
                        "n": int(k.sum()), "median_sd": float(np.median(sd[k])) if k.any()
                        else None})
    return out


def run(args) -> dict:
    from scripts import optionA as oa

    cfg7 = json.loads((we.members_of(we.arm_dir(7, args.root))[0] / "results.json")
                      .read_text())["_config"]
    theta_all, files, card = sbc_sets(oa.spec_of(cfg7).box)
    res = {"behaviour": behaviour(files, theta_all, card), "arms": {}}
    for n in args.waves:
        s, x, x0 = arm_windows(n, theta_all, files, args.root)
        res["arms"][n] = {**bands(s), "summaries": against_summaries(s, x, x0),
                          "at_limit": at_limit(s, x0, card)}
        print(f"{n} waves done", flush=True)
    out = args.root / "low_beta_check.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")
    return res


def report(res: dict) -> None:
    arms = res["arms"]
    ns = sorted(arms, key=int)
    print("\nBands of true beta, 0.2 wide: corr(mean, truth) / median sd / median 90% width")
    print("| band | " + " | ".join(f"{n} waves" for n in ns) + " |")
    for j, (a, b) in enumerate(BANDS):
        print(f"| {a}-{b} | " + " | ".join(
            "{corr:.3f} / {median_sd:.3f} / {median_width_90:.3f}".format(**arms[n]["bands"][j])
            for n in ns) + " |")
    print("\nNetwork's ranking (AUC of the beta mean), bands 0.2 apart")
    for j, (lo_, hi_) in enumerate(PAIRS):
        print(f"| {lo_[0]}-{lo_[1]} vs {hi_[0]}-{hi_[1]} | " + " | ".join(
            f"{arms[n]['pairs'][j]['auc_beta_mean']:.3f}" for n in ns) + " |")
    print("\nEdge yardstick: 95th pct below 0.3 at true 0.1-0.2 / 5th pct above 0.8 at 0.9-1.0")
    print(" | ".join(f"{n}: {arms[n]['p95_below_0.3_at_0.1_0.2']:.3f} / "
                     f"{arms[n]['p5_above_0.8_at_0.9_1.0']:.3f}" for n in ns))
    print("\nBehaviour by true beta (SBC panels, ages 24-64): " + ", ".join(BEHAVIOUR))
    for key, rows in res["behaviour"].items():
        if not isinstance(rows, list):
            print(f"{key}: {rows:.5f}")
            continue
        print(key)
        for r in rows:
            print(f"| {r['lo']}-{r['hi']} | {r['n']} | " +
                  " | ".join(f"{r[k]:.3f}" for k in BEHAVIOUR) + " |")
    keys = ("network_beta_mean",) + tuple(k for k, _c, _f in CLASSIFIERS)
    print("\nAgainst summaries (AUC, out of fold): " + " | ".join(keys))
    for n in ns:
        for r in arms[n]["summaries"]:
            print(f"| {n} | {r['low'][0]}-{r['low'][1]} vs {r['high'][0]}-{r['high'][1]} | " +
                  " | ".join(f"{r[k]:.3f}" for k in keys) + " |")
    print("\nCardholders: median sd of beta by share of window waves at the limit")
    for n in ns:
        print(f"{n} waves: " + "  ".join(
            f"[{r['lo']}-{r['hi']}) {r['share_lo']:.2f}-{r['share_hi']:.2f}: "
            f"{r['median_sd']:.3f} (n {r['n']})" if r["median_sd"] is not None else "-"
            for r in arms[n]["at_limit"]))


def figure(res: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    arms = res["arms"]
    ns = sorted(arms, key=int)
    xs = [int(n) for n in ns]
    shades = ("#C53A32", "#EF8636", "#519E3E", "#3B75AF", "#7B3294")
    fig, axs = plt.subplots(2, 3, figsize=(19, 9.5))
    ax = axs[0, 0]
    for j, ((a0, a1), (b0, b1)) in enumerate(PAIRS):
        ax.plot(xs, [arms[n]["pairs"][j]["auc_beta_mean"] for n in ns], "o-", color=shades[j],
                label=f"β {a0}–{a1} vs {b0}–{b1}")
    ax.set(xlabel="waves", ylabel="P(ranked the right way round)", xticks=xs,
           title="(a) The network's ranking of households in bands 0.2 apart")
    ax.axvline(7, color="grey", ls=":", lw=1)
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    for j, (a, b) in enumerate(BANDS):
        ax.plot(xs, [arms[n]["bands"][j]["median_width_90"] for n in ns], "o-",
                color=shades[j], label=f"true β {a}–{b}")
    ax.set(xlabel="waves", ylabel="median width of the 90% interval", xticks=xs,
           title="(b) One household's β interval, by band of true β")
    ax.axvline(7, color="grey", ls=":", lw=1)
    ax.legend(fontsize=8)
    ax = axs[0, 2]
    ax.plot(xs, [arms[n]["p95_below_0.3_at_0.1_0.2"] for n in ns], "o-", color=shades[0],
            label="true β 0.1–0.2: 95th percentile below 0.3")
    ax.plot(xs, [arms[n]["p5_above_0.8_at_0.9_1.0"] for n in ns], "o-", color=shades[4],
            label="true β 0.9–1.0: 5th percentile above 0.8")
    ax.set(xlabel="waves", ylabel="share of households", xticks=xs,
           title="(c) The edge yardstick at both ends of the box")
    ax.axvline(7, color="grey", ls=":", lw=1)
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    beh = res["behaviour"]
    mid = [(r["lo"] + r["hi"]) / 2 for r in beh["card_all_delta"]]
    for key, lab, col in (("card", "cardholders: years at the limit", "#C53A32"),
                          ("no_card", "no card: years with no wealth", "#3B75AF")):
        for tag, ls in (("all_delta", "-"), ("delta_0.95", "--")):
            k = "at_limit" if key == "card" else "no_wealth"
            ax.plot(mid, [r[k] for r in beh[f"{key}_{tag}"]], "o" + ls, color=col,
                    label=f"{lab} ({'δ ≥ 0.95' if tag == 'delta_0.95' else 'all δ'})")
    ax.axvline(0.3, color="grey", ls=":", lw=1)
    ax.set(xlabel="true β", ylabel="share of years, ages 24–64", ylim=(0, 1),
           title="(d) Behaviour keeps moving below 0.3 (SBC panels, noise-free)")
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    w = 0.13
    for i, n in enumerate((ns[0], ns[-1])):
        rows = arms[n]["summaries"]
        pos = np.arange(len(rows)) + (i - 0.5) * 3 * w
        for k, (key, lab, col) in enumerate((
                ("network_beta_mean", "network's β mean", COLORS_N[0]),
                ("spline_summaries", "summaries, as scored", COLORS_N[1]),
                ("spline_noise_free", "summaries, noise-free", COLORS_N[2]))):
            ax.bar(pos + k * w, [r[key] for r in rows], w, color=col,
                   alpha=1.0 if i else 0.5, label=f"{lab}, {n} waves")
    ax.set_xticks(np.arange(len(PAIRS)) - 0.5 * w)
    ax.set_xticklabels([f"{a0}–{a1}\nvs {b0}–{b1}" for (a0, a1), (b0, b1) in PAIRS], fontsize=8)
    ax.set(ylim=(0.5, 0.9), ylabel="P(ranked the right way round)",
           title=f"(e) The network against simple summaries ({ns[0]} pale, {ns[-1]} waves)")
    ax.legend(fontsize=7, ncol=2)
    ax = axs[1, 2]
    for (a, b), ls in (((0.1, 0.3), "-"), ((0.3, 0.5), "--")):
        for c0, col in zip(LIMIT_CUTS[:-1], ("#3B75AF", "#EF8636", "#C53A32"), strict=True):
            rows = [[r for r in arms[n]["at_limit"] if r["lo"] == a and r["share_lo"] == c0][0]
                    for n in ns]
            ax.plot(xs, [r["median_sd"] for r in rows], "o" + ls, color=col,
                    label=f"β {a}–{b}, {rows[0]['share_lo']:.2f}–{rows[0]['share_hi']:.2f} "
                          "of waves at the limit")
    ax.set(xlabel="waves", ylabel="median posterior sd of β", xticks=xs,
           title="(f) Cardholders: households mostly at the limit barely sharpen")
    ax.axvline(7, color="grey", ls=":", lw=1)
    ax.legend(fontsize=7)
    fig.suptitle("RESULTS 64.12: one household's β below 0.3 is resolved like the rest of the "
                 "box at 7 waves, and gains less from more waves (SBC households, truth known)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=we.ROOT)
    ap.add_argument("--waves", type=int, nargs="+", default=list(we.WAVES))
    ap.add_argument("--figure", type=Path, default=we.FIG / "111_low_beta_check.png")
    ap.add_argument("--from_json", action="store_true",
                    help="report and draw from the saved json, without recomputing")
    args = ap.parse_args()
    out = args.root / "low_beta_check.json"
    res = json.loads(out.read_text()) if args.from_json else run(args)
    res["arms"] = {str(k): v for k, v in res["arms"].items()}
    report(res)
    figure(res, args.figure)


if __name__ == "__main__":
    main()
