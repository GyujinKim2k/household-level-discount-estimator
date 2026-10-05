"""Option A2 pilot: where PSID couples sit, and the re-aimed proposal (RESULTS 43).

Reads the pilot model's PSID posteriors (``scripts/run_optionA2_pilot_model.sh``)
and answers the three questions the pilot exists for:

1. **R_gamma edge.** Do couples' R_gamma posteriors pile at the 1.025 floor?
   If so, widen it toward R = 1.0203 before the rest of the run (below R the
   illiquid asset is dominated and the floor is economic, not arbitrary).
2. **The concentrated region.** The edge mixture's concentrated half was aimed
   at where PSID sat under the broken solver and the SCF seed (RESULTS 41.5).
   Candidate regions from couples' posterior means -- beta from its p10,
   1 - delta to its p90, rho from its p10 -- with the share of household means
   each covers, for couples and for all 889, against the current
   ``EDGE_WIDENED`` region.
3. **Levels.** At couples' median posterior theta, does the model (PSID seed
   pool, 80% cardholders -- couples' lower bound is 79.5%, RESULTS 42.5) hold
   PSID couples' illiquid wealth and debt at 35-44?

Usage::

    PYTHONPATH=. .venv/bin/python scripts/optionA2_pilot_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.npe.prior import EDGE_WIDENED, PHASE3_RGAMMA

ARMS = {"couples": Path("outputs/psid_optionA2_pilot_couples"),
        "all 889": Path("outputs/psid_optionA2_pilot_all")}
X = {"couples": Path("data/processed/psid_x_comphs_couples.pt"),
     "all 889": Path("data/processed/psid_x_comphs_optionA.pt")}
POOL = Path("data/processed/seed_pool_couples.npz")
OUT = Path("outputs/optionA2_pilot/report.json")
CARD_SHARE = 0.80
BOX = PHASE3_RGAMMA


def region_cover(m: np.ndarray, beta_lo, omd_hi, crra_lo, omd_lo=1e-4) -> float:
    omd = 1 - m[:, 1]
    return float(((m[:, 0] >= beta_lo) & (omd >= omd_lo) & (omd <= omd_hi)
                  & (m[:, 2] >= crra_lo)).mean())


def main() -> None:
    out: dict = {}
    means = {}
    for arm, d in ARMS.items():
        r = np.load(d / "posterior_uncorrected.npz")
        ok = np.isfinite(r["mean"][:, 0])
        m, lo, hi = r["mean"][ok], r["lo"][ok], r["hi"][ok]
        means[arm] = m
        q = np.percentile(m, [10, 50, 90], axis=0)
        rg = m[:, 3]
        floor_band = BOX.rgamma_low + 0.1 * (BOX.rgamma_high - BOX.rgamma_low)
        o = {"n": int(ok.sum()),
             "p10_p50_p90": dict(zip(BOX.names, q.T.tolist())),
             "rgamma_mean_in_bottom_decile": float((rg <= floor_band).mean()),
             "rgamma_upper90_below_1.035": float((hi[:, 3] < 1.035).mean()),
             "rho_mean_ge_4.2": float((m[:, 2] >= 4.2).mean()),
             "median_posterior_sd": dict(zip(BOX.names, np.median(r["sd"][ok], 0).tolist()))}
        out[arm] = o
        print(f"\n=== {arm}: {o['n']} households (posterior means, importance-weighted) ===")
        print(f"{'':10s}{'p10':>10s}{'p50':>10s}{'p90':>10s}{'post sd':>10s}")
        for j, n in enumerate(BOX.names):
            print(f"{n:10s}" + "".join(f"{v:10.4f}" for v in q[:, j])
                  + f"{o['median_posterior_sd'][n]:10.4f}")
        print(f"R_gamma: mean in bottom decile of [1.025, 1.075] "
              f"{o['rgamma_mean_in_bottom_decile']:.1%} (uniform: 10%); upper 90% "
              f"bound below 1.035 {o['rgamma_upper90_below_1.035']:.1%}")
        print(f"rho posterior mean >= 4.2 (the float64 exposure zone, now fixed): "
              f"{o['rho_mean_ge_4.2']:.1%}")

    # --- candidate regions -----------------------------------------------------
    mc = means["couples"]
    b10 = np.floor(np.percentile(mc[:, 0], 10) / 0.05) * 0.05
    o90 = np.ceil(np.percentile(1 - mc[:, 1], 90) / 0.01) * 0.01
    r10 = np.floor(np.percentile(mc[:, 2], 10) / 0.25) * 0.25
    cands = {"current EDGE_WIDENED": (EDGE_WIDENED["beta_lo"],
                                      EDGE_WIDENED["one_minus_delta_hi"],
                                      EDGE_WIDENED["crra_lo"]),
             "couples p10/p90": (b10, o90, r10),
             "couples p10/p90, rho widened 0.5": (b10, o90, max(0.5, r10 - 0.5))}
    print("\n=== candidate concentrated regions: share of household means covered ===")
    out["regions"] = {}
    for k, (bl, oh, cl) in cands.items():
        cov = {arm: region_cover(means[arm], bl, oh, cl) for arm in means}
        vol = ((BOX.beta_high - bl) / (BOX.beta_high - BOX.beta_low)
               * (BOX.crra_high - cl) / (BOX.crra_high - BOX.crra_low))
        out["regions"][k] = {"beta_lo": bl, "one_minus_delta_hi": oh, "crra_lo": cl,
                             "cover": cov, "beta_rho_box_share": vol}
        print(f"{k:34s} beta >= {bl:.2f}, 1-delta <= {oh:.2f}, rho >= {cl:.2f}:  "
              + "  ".join(f"{a} {v:.0%}" for a, v in cov.items())
              + f"   (beta x rho share of box {vol:.2f})")

    # --- levels at couples' median theta --------------------------------------
    from hh_npe.simulator.dispatch import simulate_batch_twoasset_gpu

    th = np.median(mc, axis=0)
    pool = np.load(POOL)["pool"]
    n = 4000
    _x, _a, p = simulate_batch_twoasset_gpu(
        np.array([th[:3], th[:3]]), seed_base=777, start_age=30, n_waves=7,
        wave_years=2, grid="full", theta_batch=1, chunk=16, return_panels=True,
        n_households=n, card=np.array([1, 0]), r_gamma=np.array([th[3], th[3]]),
        init_pool=pool)
    ages = slice(35 - 20, 44 - 20 + 1)
    k1 = int(CARD_SHARE * n)
    ill = np.concatenate([p["illiquid_assets"][:k1, ages], p["illiquid_assets"][n:n + n - k1, ages]]).ravel()
    liq = np.concatenate([p["liquid_assets"][:k1, ages], p["liquid_assets"][n:n + n - k1, ages]]).ravel()
    d = torch.load(X["couples"], weights_only=False)["x"].numpy()
    a = d[:, :, 4]
    sel = (a >= 35) & (a <= 44)
    pill, pliq = d[:, :, 3][sel], d[:, :, 2][sel]
    lv = {"theta": th.tolist(),
          "model": {"illiquid_q": np.percentile(ill, [25, 50, 75]).tolist(),
                    "zero_illiquid": float((ill <= 0).mean()), "debt": float((liq < 0).mean())},
          "psid_couples": {"illiquid_q": np.percentile(pill, [25, 50, 75]).tolist(),
                           "zero_illiquid": float((pill <= 0).mean()),
                           "debt": float((pliq < 0).mean())}}
    out["levels_35_44"] = lv
    print(f"\n=== ages 35-44 at couples' median theta {np.round(th, 4)}, "
          f"{CARD_SHARE:.0%} cardholders ===")
    for k in ("model", "psid_couples"):
        v = lv[k]
        print(f"{k:13s} illiquid p25/50/75 " + " ".join(f"{q:9,.0f}" for q in v["illiquid_q"])
              + f"   zero {v['zero_illiquid']:.0%}   in debt {v['debt']:.0%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
