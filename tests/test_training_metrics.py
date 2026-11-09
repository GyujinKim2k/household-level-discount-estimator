"""The per-epoch training CSV (RESULTS 67).

One row per epoch trained, carrying exactly the losses sbi records, written
while training runs, and changing nothing in training: the same seed gives
bit-identical losses with and without the logger. The backfill of older
runs follows sbi's best-epoch rule.
"""

import csv

import numpy as np
import pytest
import torch

from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import PriorBox, sample_sobol
from hh_npe.npe.train import METRIC_COLUMNS, train_npe
from scripts import training_metrics as tm


def _data(n=64, per=2):
    torch.manual_seed(0)
    box = PriorBox()
    theta = torch.from_numpy(sample_sobol(n, box, seed=0)).float().repeat_interleave(per, 0)
    x = torch.randn(len(theta), 5, 3)
    x[:, :, 0] += 5 * theta[:, :1]
    return theta, x, torch.arange(n).repeat_interleave(per), box


def _train(metrics_csv=None, epochs=4):
    theta, x, pid, box = _data()
    torch.manual_seed(1)
    return train_npe(theta, x, embedder=TrajectoryTransformer(
        d_model=16, n_heads=2, n_layers=1, output_dim=8), hidden_features=16,
        num_transforms=2, max_num_epochs=epochs, stop_after_epochs=50, batch_size=16,
        group_ids=pid, box=box, metrics_csv=metrics_csv, passes_per_epoch=3)


def test_one_row_per_epoch_with_sbi_s_losses(tmp_path):
    paths = [tmp_path / "a" / "metrics.csv", tmp_path / "b" / "s0.csv"]
    _p, _d, inf = _train(paths)
    s = inf._summary
    for p in paths:
        with p.open() as f:
            rows = list(csv.DictReader(f))
        assert tuple(rows[0]) == METRIC_COLUMNS
        assert len(rows) == len(s["training_loss"]) == 5          # sbi trains max + 1
        for e, r in enumerate(rows, start=1):
            assert int(r["epoch"]) == e and int(r["passes"]) == 3 * e
            assert float(r["train_loss"]) == pytest.approx(s["training_loss"][e - 1], rel=1e-5)
            assert float(r["val_loss"]) == pytest.approx(s["validation_loss"][e - 1], rel=1e-5)
            assert float(r["grad_norm_max"]) >= float(r["grad_norm_mean"]) > 0
            assert int(r["n_train_rows"]) + int(r["n_val_rows"]) == 128
        best = np.minimum.accumulate(s["validation_loss"])
        assert [float(r["best_val_loss"]) for r in rows] == pytest.approx(best.tolist(), rel=1e-5)
        assert [int(r["is_best"]) for r in rows] == \
            [int(v == b) for v, b in zip(s["validation_loss"], best, strict=True)]
    m = tm.read(paths[0])
    assert len(m["epoch"]) == 5 and np.isfinite(m["elapsed_min"]).all()


def test_logging_changes_nothing_in_training(tmp_path):
    a = _train(None)[2]._summary
    b = _train(tmp_path / "m.csv")[2]._summary
    assert a["training_loss"] == b["training_loss"]
    assert a["validation_loss"] == b["validation_loss"]


def test_the_gradient_clip_is_restored_after_training(tmp_path):
    from sbi.inference.trainers.npe import npe_base
    orig = npe_base.clip_grad_norm_
    _train(tmp_path / "m.csv", epochs=1)
    assert npe_base.clip_grad_norm_ is orig


def test_backfill_follows_sbi_s_best_epoch_rule(tmp_path):
    summ = {"training_loss": [3.0, 2.9, 2.8, 2.7], "validation_loss": [3.3, 3.2, 3.2, 3.25],
            "epoch_durations_sec": [30.0, 30.0, 30.0, 30.0]}
    rows = tm.rows_from_summary(summ)
    assert [r["is_best"] for r in rows] == [1, 1, 0, 0]           # equal is not better
    assert [r["epochs_since_best"] for r in rows] == [0, 0, 1, 2]
    assert [r["elapsed_min"] for r in rows] == [0.5, 1.0, 1.5, 2.0]
    tm.write(tmp_path / "s0.csv", rows)
    m = tm.read(tmp_path / "s0.csv")
    assert m["val_loss"].tolist() == summ["validation_loss"]
    assert np.isnan(m["grad_norm_mean"]).all()                    # not stored before 67


def test_run_names():
    from pathlib import Path
    assert tm.run_name(Path("outputs/waves/w07")) == Path("waves/w07")
    assert tm.run_name(Path("/x/smoke/outputs/noise_copies/v2_k8/w07")) == \
        Path("noise_copies/v2_k8/w07")
