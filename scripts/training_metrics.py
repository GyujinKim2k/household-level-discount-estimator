"""Per-epoch training metrics as CSV files (RESULTS 67).

Every ``optionA.py train`` since RESULTS 67 writes one row per epoch while it
trains (``hh_npe.npe.train.EpochCSV``): ``metrics.csv`` beside the member's
checkpoint and ``training_metrics/<run>/s<seed>.csv`` in this repository,
``<run>`` being the output path after ``outputs/``. Columns:
``hh_npe.npe.train.METRIC_COLUMNS``. ``passes`` counts passes over the clean
windows (an epoch over K noisy copies is K passes), so runs with and without
copies line up on one axis.

``backfill``
    The same layout for members trained before RESULTS 67, from their
    ``training_summary.json`` (training and validation loss and epoch times;
    the columns sbi did not store are left empty). The best-loss columns
    follow sbi's rule: a new best is any strictly lower validation loss.

Usage::

    PYTHONPATH=.:src .venv/bin/python scripts/training_metrics.py backfill \\
        outputs/waves/w07 outputs/waves_none/w07
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hh_npe.npe.train import METRIC_COLUMNS

METRICS_DIR = Path("training_metrics")


def run_name(run: Path) -> Path:
    """A run's name under ``training_metrics/``: its path after ``outputs/``."""
    parts = Path(run).parts
    if "outputs" not in parts:
        return Path(Path(run).name)
    return Path(*parts[len(parts) - parts[::-1].index("outputs"):])


def rows_from_summary(summary: dict, passes_per_epoch: int = 1) -> list[dict]:
    """CSV rows from a ``training_summary.json``."""
    tr, va = summary["training_loss"], summary["validation_loss"]
    sec = summary["epoch_durations_sec"]
    best, since, rows, elapsed = np.inf, 0, [], 0.0
    for e, (t, v, s) in enumerate(zip(tr, va, sec, strict=True), start=1):
        if v < best:
            best, since = v, 0
        else:
            since += 1
        elapsed += s
        rows.append({"epoch": e, "passes": e * passes_per_epoch, "train_loss": t,
                     "val_loss": v, "best_val_loss": best, "is_best": int(since == 0),
                     "epochs_since_best": since, "epoch_sec": s, "elapsed_min": elapsed / 60})
    return rows


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def read(path: Path) -> dict:
    """A metrics CSV as columns of floats (empty cells as NaN)."""
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
            for k in (rows[0].keys() if rows else METRIC_COLUMNS)}


def backfill(args) -> None:
    for run in args.runs:
        members = sorted(Path(run).glob("s[0-9]*"))
        if not members:
            raise SystemExit(f"{run}: no members")
        for m in members:
            cfg = json.loads((m / "results.json").read_text())["_config"]
            rows = rows_from_summary(json.loads((m / "training_summary.json").read_text()),
                                     cfg.get("noise_copies", 1))
            out = args.metrics_dir / run_name(run) / f"{m.name}.csv"
            write(out, rows)
            print(f"{m}: {len(rows)} epochs -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill")
    b.add_argument("runs", type=Path, nargs="+", help="Run directories holding s*/.")
    b.add_argument("--metrics_dir", type=Path, default=METRICS_DIR)
    args = ap.parse_args()
    {"backfill": backfill}[args.cmd](args)


if __name__ == "__main__":
    main()
