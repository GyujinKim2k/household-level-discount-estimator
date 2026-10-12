"""Collect the A2n noise-range arms into one table (RESULTS 43.10).

Reads what ``run_optionA2n_noise_sensitivity.sh`` and
``run_optionA2n_member_pairs.sh`` wrote and prints, per arm: PSID couples'
medians of the per-household posterior means (uncorrected arm), rho's median
90% width, the floors, held-out calibration, couples' rho read on the
known-rho scale, and how far couples' ranking moves (per-couple correlation of
means with the reference). The pair rows give what two A2n members show with
the noise unchanged.

Known-rho scale: ``rho_floor_diagnostic.py recovery`` groups held-out simulated
households in couples' A2n region by their true rho and records each group's
true median and median posterior mean. Couples' median mean is mapped to a
true rho by linear interpolation between those points, each arm on its own
recovery, since the mapping moves with the noise.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/optionA2n_noise_sensitivity.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path("outputs/optionA2n_sens")
ARMS = ["a2n_2m", "cons010", "cons020", "all067", "all133"]
PARAMS = ["beta", "delta", "crra", "R_gamma"]


def known_rho(rec: dict) -> float:
    """Couples' median posterior mean of rho, as a true rho (see module doc)."""
    rows = rec["by_true_rho"]
    true = np.array([r["true_median"] for r in rows])
    means = np.array([r["median_of_means"] for r in rows])
    if np.any(np.diff(means) <= 0):
        raise ValueError("recovery means are not increasing in the true rho")
    return float(np.interp(rec["psid"]["median_of_means"], means, true))


def row(psid_dir: Path, rec_path: Path, heldout_path: Path | None) -> dict:
    s = json.loads((psid_dir / "summary.json").read_text())
    u = s["uncorrected"]
    rec = json.loads(rec_path.read_text())["recovery_a2n"]
    out = {"noise": s.get("training_obs_noise"),
           "median": {p: u["median_of_means"][p] for p in PARAMS},
           "corrected_median": {p: s["corrected"]["median_of_means"][p] for p in PARAMS},
           "rho_median_90_width": u["median_90_width"]["crra"],
           "rho_median_sd": u["median_posterior_sd"]["crra"],
           "known_rho": known_rho(rec),
           "recovery_psid_rho_median": rec["psid"]["median_of_means"],
           "recovery_contraction_psid": rec["psid"]["contraction"],
           "recovery_15_25_mean": rec["by_true_rho"][3]["median_of_means"]}
    if heldout_path is not None:
        h = json.loads(heldout_path.read_text())
        e = h["ensemble"]
        out["heldout"] = {"log_q": e["log_q_weighted"],
                          **{p: {k: e[p][k] for k in ("cov90", "below5", "above95", "corr",
                                                        "contraction")} for p in PARAMS}}
        out["floors"] = next(iter(h["psid_floors"].values()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "summary.json")
    args = ap.parse_args()

    arms = {a: row(Path(f"outputs/psid_optionA2n_sens_{a}_couples"),
                   ROOT / a / "rho_floor_diagnostic.json", ROOT / a / "heldout_check.json")
            for a in ARMS if (ROOT / a / "rho_floor_diagnostic.json").exists()}
    pairs = {d.name: row(d / "psid", d / "rho_floor_diagnostic.json", None)
             for d in sorted((ROOT / "pairs").glob("s*"))
             if (d / "rho_floor_diagnostic.json").exists()}

    keys = [("beta", lambda r: r["median"]["beta"], 3),
            ("delta", lambda r: r["median"]["delta"], 4),
            ("rho", lambda r: r["median"]["crra"], 2),
            ("R_gamma", lambda r: r["median"]["R_gamma"], 4),
            ("rho known", lambda r: r["known_rho"], 2),
            ("rho 90% width", lambda r: r["rho_median_90_width"], 2)]
    print("PSID couples (409), uncorrected arm: medians of per-household means")
    print(f"{'':10s}" + "".join(f"{k:>14s}" for k, _, _ in keys))
    for a, r in arms.items():
        print(f"{a:10s}" + "".join(f"{f(r):14.{d}f}" for _, f, d in keys))
    if pairs:
        print(f"A2n member pairs (n = {len(pairs)}): min - max")
        lo = [min(f(r) for r in pairs.values()) for _, f, _ in keys]
        hi = [max(f(r) for r in pairs.values()) for _, f, _ in keys]
        print(f"{'':10s}" + "".join(f"{a:>6.{d}f}-{b:<7.{d}f}"
                                    for a, b, (_, _, d) in zip(lo, hi, keys, strict=True)))

    print("\nCouples' floors: rho mean < 1, rho lower 5% < 0.55, either floor, both")
    for a, r in arms.items():
        f = r["floors"]
        print(f"{a:10s} {f['rho_mean_lt_1']:.3f} {f['rho_lo_lt_0.55']:.3f} "
              f"{f['either']:.3f} {f['both']:.3f}")

    print("\nHeld-out ensemble: 90% coverage / contraction per parameter; weighted log q")
    for a, r in arms.items():
        h = r["heldout"]
        print(f"{a:10s} " + "  ".join(f"{p} {h[p]['cov90']:.3f}/{h[p]['contraction']:.3f}"
                                       for p in PARAMS) + f"  log q {h['log_q']:.3f}")

    # Ranking: per-couple correlation of means with the reference, against
    # what two pairs of A2n members sharing no member give.
    def means(d: Path) -> np.ndarray:
        return np.load(d / "posterior_uncorrected.npz")["mean"]

    def corr(a: np.ndarray, b: np.ndarray) -> list[float]:
        return [float(np.corrcoef(a[:, i], b[:, i])[0, 1]) for i in range(len(PARAMS))]

    ref = means(Path("outputs/psid_optionA2n_sens_a2n_2m_couples"))
    ranking = {a: corr(ref, means(Path(f"outputs/psid_optionA2n_sens_{a}_couples")))
               for a in arms if a != "a2n_2m"}
    pm = {k: means(ROOT / "pairs" / k / "psid") for k in pairs}
    disjoint = np.array([corr(pm[a], pm[b]) for a, b in itertools.combinations(pm, 2)
                         if not set(a[1::2]) & set(b[1::2])])
    print("\nPer-couple correlation of posterior means with a2n_2m: " + " ".join(PARAMS))
    for a, c in ranking.items():
        print(f"{a:10s} " + " ".join(f"{v:.2f}" for v in c))
    if len(disjoint):
        print(f"{len(disjoint)} disjoint member pairs: min " + " ".join(
            f"{v:.2f}" for v in disjoint.min(0)) + ", max " + " ".join(
            f"{v:.2f}" for v in disjoint.max(0)))

    args.out.write_text(json.dumps(
        {"arms": arms, "pairs": pairs, "ranking_corr_with_a2n_2m": ranking,
         "disjoint_pairs_corr": {"min": disjoint.min(0).tolist(), "max": disjoint.max(0).tolist()}
         if len(disjoint) else None}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
