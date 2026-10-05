"""RESULTS.md §40: does the solver lose float64 resolution at high rho?

Utility is ``h * ((c / h)**(1 - rho) - 1) / (1 - rho)`` in dollars. The "-1"
makes each period's utility a constant plus a variable part that, at rho ~ 4.6
and typical consumption, is below float64's resolution of that constant, so
choices tie and break toward the lowest index (most borrowing, least illiquid
wealth). The check compares the current GPU solver with a patched copy that
drops the constant from period and bequest utility -- exact in real arithmetic,
since a per-period constant never changes a choice:

1. ``batch``: each rho solved alone and in one batch with the others; the share
   of policy entries that agree, and the model's own households at 40-44.
2. ``sweep``: debt share and illiquid median at 40-44 across rho, for several
   (beta, delta) and both card types, current against patched.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/solver_precision_check.py batch
    PYTHONPATH=. .venv/bin/python scripts/solver_precision_check.py sweep
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

import hh_npe.simulator.twoasset_gpu as gpu
from hh_npe.simulator.twoasset import simulate
from scripts.oos_optionA import spec_for


def patched_solver():
    """``twoasset_gpu`` with the CRRA constant dropped (period and bequest)."""
    src = Path(gpu.__file__).read_text()
    edits = (("torch.exp(omr5 * logc[None]) - 1.0)", "torch.exp(omr5 * logc[None]))"),
             ("(mean_hhy / mean_hhs) ** omr3 - 1.0)", "(mean_hhy / mean_hhs) ** omr3)"),
             ("ratio ** omr3 - 1.0)", "ratio ** omr3)"))
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    mod = types.ModuleType("twoasset_gpu_patched")
    mod.__file__ = gpu.__file__
    exec(compile(src, "twoasset_gpu_patched", "exec"), mod.__dict__)
    return mod


def own(sol, n=3000):
    """Debt share, liquid and illiquid medians of the model's own households, 40-44."""
    p = simulate(sol, n_households=n, seed=1)
    li, il = p["liquid_assets"][:, 20:25], p["illiquid_assets"][:, 20:25]
    return float(np.mean(li < 0)), float(np.median(li)), float(np.median(il))


def batch() -> None:
    pat = patched_solver()
    rhos = [1.9355, 3.0, 4.0, 4.6]
    th = np.array([[0.7612, 0.9759, r] for r in rhos])
    spec = spec_for(1, 1.0482)
    for name, mod in (("current", gpu), ("patched", pat)):
        together = mod.solve_batch(th, spec, theta_batch=16, chunk=16)
        for i, r in enumerate(rhos):
            alone = mod.solve_batch(th[i:i + 1], spec, theta_batch=16, chunk=16)[0]
            ax = float((alone.next_x == together[i].next_x).mean())
            az = float((alone.next_z == together[i].next_z).mean())
            d, lm, im = own(alone)
            print(f"{name} rho {r:.2f}: alone vs batch agree x {ax:.3f} z {az:.3f}; "
                  f"debt {d:.0%}, liquid median {lm:,.0f}, illiquid median {im:,.0f}", flush=True)
            del alone
            torch.cuda.empty_cache()
        del together
        torch.cuda.empty_cache()


def sweep() -> None:
    pat = patched_solver()
    rhos = [3.5, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]
    for card in (1, 0):
        for b, d in ((0.77, 0.988), (0.95, 0.95), (0.5, 0.99)):
            th = np.array([[b, d, r] for r in rhos])
            spec = spec_for(card, 1.05)
            rows = []
            for mod in (gpu, pat):
                sols = mod.solve_batch(th, spec, theta_batch=16, chunk=16)
                rows.append([own(s) for s in sols])
                del sols
                torch.cuda.empty_cache()
            print(f"card {card} beta {b} delta {d}  (debt share / illiquid median at 40-44, "
                  f"current -> patched)")
            print("  " + "  ".join(f"rho {r}: {x[0]:.0%}/{x[2] / 1e3:.0f}k -> "
                                   f"{y[0]:.0%}/{y[2] / 1e3:.0f}k"
                                   for r, x, y in zip(rhos, *rows)), flush=True)


if __name__ == "__main__":
    {"batch": batch, "sweep": sweep}[sys.argv[1] if len(sys.argv) > 1 else "batch"]()
