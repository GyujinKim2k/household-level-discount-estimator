"""Did correcting the window start-age alignment change anything?

``RESULTS.md`` §10.8: ``aggregate_waves`` reports each wave at the *last* year of
its window, so ``--start_low 25`` produces a first wave at age 26. Simulated
wave-0 ages ran 26-47 against PSID's 25-46, leaving ~5% of households -- those
aged 25 at wave 0 -- outside the training support in a feature the network
otherwise anchors on.

This compares the baseline ensemble against one retrained with
``--start_low 24 --start_high 45``, identical in every other argument.

**The split by wave-0 age is the point.** A population-level null is weak
evidence here: the households that were extrapolating are 5% of the sample, so a
large shift confined to them would barely move any aggregate. The affected
subgroup is reported separately, and against the unaffected one as a control --
if the corrected run moves the age-25 households *and* leaves the rest alone,
that is the alignment; if it moves both equally, it is retraining noise.

Usage::

    uv run python scripts/startage_compare.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.npe.prior import PHASE3

CEILING_BAND = 0.05
LAIBSON = np.array([0.5305, 0.9891, 1.9355])


def recovery(base: Path, fixed: Path) -> None:
    """Held-out recovery on simulated data, where the truth is known."""
    print("\n=== held-out recovery (simulated, truth known) ===")
    print(f"{'':10s}{'corr base':>11s}{'corr fixed':>12s}"
          f"{'mae base':>11s}{'mae fixed':>11s}{'mae change':>12s}")
    a = json.loads((base / "results.json").read_text())["ensemble"]
    b = json.loads((fixed / "results.json").read_text())["ensemble"]
    for k in PHASE3.names:
        e0, e1 = a["estimation"][k], b["estimation"][k]
        print(f"{k:10s}{e0['corr']:11.3f}{e1['corr']:12.3f}"
              f"{e0['mae']:11.4f}{e1['mae']:11.4f}"
              f"{e1['mae'] / e0['mae'] - 1:+12.1%}")


def summarise(tag: str, means: np.ndarray, in_box: np.ndarray) -> dict:
    hi, lo = np.asarray(PHASE3.high), np.asarray(PHASE3.low)
    pile = (means > hi - CEILING_BAND * (hi - lo)).mean(axis=0)
    print(f"\n--- {tag}  (N={len(means)}) ---")
    print(f"{'':8s}{'median':>10s}{'mean':>10s}{'sd':>10s}{'at ceiling':>12s}")
    out = {}
    for j, n in enumerate(PHASE3.names):
        print(f"{n:8s}{np.median(means[:, j]):10.4f}{means[:, j].mean():10.4f}"
              f"{means[:, j].std():10.4f}{pile[j]:12.1%}")
        out[n] = {"median": float(np.median(means[:, j])),
                  "sd": float(means[:, j].std()),
                  "at_ceiling": float(pile[j])}
    out["in_box_median"] = float(np.median(in_box))
    print(f"in-box posterior mass: median {np.median(in_box):.3f}, "
          f"p10 {np.percentile(in_box, 10):.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_ens", type=Path,
                    default=Path("outputs/ensemble/flowfix_w7"))
    ap.add_argument("--fixed_ens", type=Path,
                    default=Path("outputs/ensemble/startage_w7"))
    ap.add_argument("--base_psid", type=Path,
                    default=Path("outputs/psid_rental"))
    ap.add_argument("--fixed_psid", type=Path,
                    default=Path("outputs/startage_psid"))
    ap.add_argument("--x", type=Path,
                    default=Path("data/processed/psid_x_rental.pt"))
    ap.add_argument("--out", type=Path, default=Path("outputs/startage_psid"))
    args = ap.parse_args()

    recovery(args.base_ens, args.fixed_ens)

    age0 = torch.load(args.x, weights_only=False)["x"][:, 0, 4].numpy()
    res = {}
    for tag, run in (("baseline (start 25-46)", args.base_psid),
                     ("corrected (start 24-45)", args.fixed_psid)):
        z = np.load(run / "posterior_uncorrected.npz")
        m, ib = z["mean"], z["in_box_frac"]
        ok = np.isfinite(m[:, 0])
        print(f"\n{'=' * 62}\n{tag}\n{'=' * 62}")
        res[tag] = {"all": summarise("all households", m[ok], ib[ok])}
        a = age0[ok]
        # The affected 5%, and the rest as a control.
        res[tag]["age25"] = summarise("wave-0 age 25 (was extrapolating)",
                                      m[ok][a == 25], ib[ok][a == 25])
        res[tag]["rest"] = summarise("wave-0 age >25 (control)",
                                     m[ok][a > 25], ib[ok][a > 25])

    b, f = res["baseline (start 25-46)"], res["corrected (start 24-45)"]
    print(f"\n{'=' * 62}\nchange, corrected minus baseline\n{'=' * 62}")
    print(f"{'group':22s}{'d beta':>10s}{'d delta':>10s}{'d crra':>10s}"
          f"{'d in-box':>11s}")
    for g, label in (("age25", "age 25 (affected)"),
                     ("rest", "age >25 (control)"),
                     ("all", "all households")):
        d = [f[g][n]["median"] - b[g][n]["median"] for n in PHASE3.names]
        print(f"{label:22s}{d[0]:+10.4f}{d[1]:+10.4f}{d[2]:+10.4f}"
              f"{f[g]['in_box_median'] - b[g]['in_box_median']:+11.4f}")

    print("\nRead the affected row against the control. A shift in both is\n"
          "retraining noise; a shift confined to the age-25 households is the\n"
          "alignment. A null in both means the support gap did not matter, and\n"
          "the fix is still correct to keep -- it just changes nothing.")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "startage_comparison.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}/startage_comparison.json")


if __name__ == "__main__":
    main()
