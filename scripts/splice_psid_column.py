"""Splice one feature column of a PSID tensor into another (RESULTS 44.4).

The PSID input tensors are built from column splices on identical households
(RESULTS 33), so a change to one feature's definition is tested by replacing
that column only. Two rebuilds are needed: ``--check``, built with the current
builder and the base's own settings, and ``--source``, the same with the one
definition changed. The splice is refused unless

- ``--check``'s column equals the base's on every base household, so the
  builder still reproduces the base, and
- ``--source`` differs from ``--check`` in that column only,

so the spliced tensor differs from the base by the definition change alone.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/splice_psid_column.py \\
        --base data/processed/psid_x_comphs_couples.pt \\
        --check data/processed/psid_x_pension_illdebts_all.pt \\
        --source data/processed/psid_x_pension_illdebts_secured.pt \\
        --feature illiquid_assets --out data/processed/psid_x_comphs_couples_illsecured.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def rows(base: dict, other: dict) -> np.ndarray:
    """Index into ``other`` of each base household (by ``psid_row``)."""
    pos = {int(r): i for i, r in enumerate(np.asarray(other["psid_row"]))}
    missing = [int(r) for r in np.asarray(base["psid_row"]) if int(r) not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} base households missing, e.g. {missing[:5]}")
    return np.array([pos[int(r)] for r in np.asarray(base["psid_row"])])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--check", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--feature", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base, check, src = (torch.load(p, weights_only=False)
                        for p in (args.base, args.check, args.source))
    for d, p in ((check, args.check), (src, args.source)):
        if list(d["features"]) != list(base["features"]):
            raise SystemExit(f"{p}: features {d['features']} differ from the base's")
    j = list(base["features"]).index(args.feature)
    if not np.array_equal(np.asarray(check["psid_row"]), np.asarray(src["psid_row"])):
        raise SystemExit("--check and --source cover different households")
    xc, xs = check["x"].numpy(), src["x"].numpy()
    others = [k for k in range(xc.shape[-1]) if k != j]
    if not np.array_equal(xc[..., others], xs[..., others]):
        raise SystemExit("--source differs from --check outside the spliced column")
    i = rows(base, check)
    xb = base["x"].numpy()
    if not np.array_equal(xc[i][..., j], xb[..., j]):
        n = int((xc[i][..., j] != xb[..., j]).sum())
        raise SystemExit(f"--check does not reproduce the base's {args.feature} ({n} cells "
                         "differ): the builder or its settings are not the base's")
    x = xb.copy()
    x[..., j] = xs[i][..., j]
    changed = (x[..., j] != xb[..., j])
    out = dict(base)
    out["x"] = torch.from_numpy(x)
    out["note"] = (str(base.get("note", "")) + f" Spliced {args.feature} from "
                   f"{args.source.name} ({changed.mean():.1%} of household-waves changed).")
    out["spliced"] = {"feature": args.feature, "source": str(args.source),
                      "check": str(args.check)}
    torch.save(out, args.out)
    print(f"{args.feature}: {changed.mean():.1%} of household-waves and "
          f"{changed.any(1).mean():.1%} of households changed; wrote {args.out}")


if __name__ == "__main__":
    main()
