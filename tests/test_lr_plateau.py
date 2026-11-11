"""RESULTS 69: reduce-on-plateau and a harder early stop.

Patience, the cap and the plateau rule are set in passes and must become the
epochs RESULTS 67 used (and the runs before must keep theirs); the learning
rate each epoch trained at, as the CSV logs it, must be the rate rebuilt from
the validation losses alone; a plateau never reached must change nothing; at
the cap the best weights must come back without a redraw too; and the arms'
draw seeds up to 600 must stay clear of every other seed. Synthetic arrays
only (the configs of the runs before when they are on disk).
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.train import METRIC_COLUMNS, train_npe
from scripts import lr_plateau as lp
from scripts import optionA as oa
from scripts import training_metrics as tm
from tests.test_fresh_noise import _data

PLATEAU = {"factor": 0.5, "patience": 4, "min_lr": 1e-5}


def test_the_schedule_in_passes_becomes_epochs():
    assert oa.schedule(1, 20, 200) == (20, None, None)             # every run before
    assert oa.schedule(8, 20, 200) == (3, 25, None)                # RESULTS 67's copies
    assert oa.schedule(1, 60, 600, 5) == (60, 600, PLATEAU)
    assert oa.schedule(8, 60, 600, 5) == (8, 75, {**PLATEAU, "patience": 0})
    assert oa.schedule(8, 60, 600, 9)[2]["patience"] == 1          # 9 passes: the 2nd epoch


@pytest.mark.parametrize("kw", [dict(patience=0, cap=600), dict(patience=60, cap=0),
                                dict(patience=60, cap=600, plateau=0),
                                dict(patience=60, cap=600, plateau=5, factor=1.0),
                                dict(patience=60, cap=600, plateau=5, floor=1e-3)])
def test_a_schedule_that_makes_no_sense_is_refused(kw):
    with pytest.raises(SystemExit):
        oa.schedule(1, **kw)


@pytest.mark.parametrize("run,copies", [("outputs/waves/w07/s0", 1),
                                        ("outputs/noise_copies/v2_k8/w07/s0", 8),
                                        ("outputs/fresh_noise/v2_fresh/w07/s0", 1)])
def test_the_runs_before_keep_their_schedule(run, copies):
    f = Path(run) / "results.json"
    if not f.exists():
        pytest.skip(f"{f} is not on disk")
    cfg = json.loads(f.read_text())["_config"]
    stop, cap, plateau = oa.schedule(copies, oa.TRAINING["stop_after_epochs"],
                                     oa.TRAINING["max_num_epochs"])
    assert cfg.get("stop_after_epochs", oa.TRAINING["stop_after_epochs"]) == stop
    assert ("stop_after_epochs" in cfg) == (copies > 1)
    assert cfg["max_epochs"] == cap and plateau is None and "lr_plateau" not in cfg


def test_member_outcomes_are_not_the_ensemble_s_setting():
    assert set(oa.MEMBER_KEYS) == {"train_seed", "fresh_noise_restored_best_at_cap"}
    assert "lr_plateau" in oa.TRAINING_KEYS and "lr_plateau" not in oa.MEMBER_KEYS


def test_draw_seeds_up_to_600_stay_clear_of_every_other_window_seed():
    others = {oa.HELD_SEED, oa.SBC_SEED, 0}
    offsets = (0, 32768, 35840)                     # the three runs' draw offsets
    draws = {oa.NOISE_SEED + off + e * oa.COPY_STRIDE for off in offsets for e in range(1, 601)}
    base = {oa.NOISE_SEED + s + off for s in others for off in offsets}
    assert not draws & base and len(draws) == 3 * 600
    top = max(draws)                                # 6.0e9: past int32, which both take
    assert top > 2**32 and torch.Generator().manual_seed(top).initial_seed() == top
    assert np.random.default_rng(top).integers(10) >= 0


# ------------------------------------------------------------------ training

def _train(tmp_path=None, plateau=None, epochs=30, patience=100, lr=1e-3):
    theta, x, pid, box = _data()
    torch.manual_seed(1)
    return train_npe(theta, x, embedder=TrajectoryTransformer(
        d_model=16, n_heads=2, n_layers=1, output_dim=8), hidden_features=16,
        num_transforms=2, max_num_epochs=epochs, stop_after_epochs=patience, batch_size=16,
        group_ids=pid, box=box, learning_rate=lr, lr_plateau=plateau,
        metrics_csv=None if tmp_path is None else tmp_path / "s0.csv")


@pytest.mark.parametrize("plateau", [{"factor": 0.5, "patience": 1, "min_lr": 1e-3 / 16},
                                     {"factor": 0.3, "patience": 0, "min_lr": 1e-4}])
def test_the_logged_rate_is_the_rule_rebuilt_from_the_validation_losses(tmp_path, plateau):
    _p, _d, inf = _train(tmp_path, plateau)
    m = tm.read(tmp_path / "s0.csv")
    lr, cuts = lp.expected_lr(m["val_loss"], 1e-3, **{
        "factor": plateau["factor"], "patience": plateau["patience"], "floor": plateau["min_lr"]})
    assert len(cuts) >= 2 and cuts == inf.lr_cuts             # the rate did fall, as the rule says
    assert np.allclose(m["lr"], lr, rtol=1e-5, atol=0)         # each epoch at its own rate
    assert m["lr"].min() >= plateau["min_lr"] * (1 - 1e-6)
    assert np.isclose(m["lr"][-1], plateau["min_lr"])          # 30 epochs reach the floor
    assert all(c < len(m["epoch"]) for c in cuts)              # no step after the last epoch


def test_a_plateau_never_reached_changes_nothing():
    a = _train()[2]
    b = _train(plateau={"factor": 0.5, "patience": 10_000, "min_lr": 1e-5})[2]
    assert a._summary["training_loss"] == b._summary["training_loss"]
    assert a._summary["validation_loss"] == b._summary["validation_loss"]
    assert b.lr_cuts == [] and not hasattr(a, "lr_cuts")


def test_the_best_weights_come_back_at_the_cap_without_a_redraw():
    # a high rate, so the last epoch is not the best
    _p, d, inf = _train(plateau={"factor": 0.5, "patience": 100, "min_lr": 1e-5}, epochs=6,
                        lr=3e-2)
    assert len(inf._summary["training_loss"]) == 7                    # the cap, not patience
    assert inf._summary["validation_loss"][-1] > inf._best_val_loss
    assert inf.restored_best_at_cap
    best = inf._best_model_state_dict
    assert all(torch.equal(d.state_dict()[k], v) for k, v in best.items())
    assert all(torch.equal(inf._neural_net.state_dict()[k], v) for k, v in best.items())


def test_patience_still_stops_training_under_the_plateau_rule():
    _p, _d, inf = _train(plateau={"factor": 0.5, "patience": 0, "min_lr": 1e-5}, epochs=200,
                         patience=3)
    n = len(inf._summary["training_loss"])
    assert n < 201 and not inf.restored_best_at_cap      # sbi had loaded the best itself
    assert all(c < n for c in inf.lr_cuts)


# ------------------------------------------------------------- the measures

def test_each_arm_lives_where_the_chain_puts_it():
    assert lp.run_dir("v2_fresh_rop") == Path("outputs/lr_plateau/v2_fresh_rop/w07")
    assert lp.run_dir("file_k8") == Path("outputs/noise_copies/file_k8/w07")
    assert lp.run_dir("v2_fresh") == Path("outputs/fresh_noise/v2_fresh/w07")
    assert lp.arm_root("w07") == Path("outputs/waves")
    with pytest.raises(ValueError):
        lp.run_dir("gross")
    for a, b in lp.PAIRS:                     # every pair on one noise: the same windows
        assert lp.NOISE_OF[a] == lp.NOISE_OF[b]
    assert {lp.NOISE_OF[a] for a in lp.ARMS} == {"w07", "file_k1"}
    assert len(set(lp.COLOR[a] for a in lp.ARMS)) == len(lp.ARMS)


def test_the_schedule_check_catches_a_rate_off_the_rule(tmp_path):
    # bad at 3, 4: cut after 4, the count restarts (5 is the first bad), a gain at 6,
    # bad at 7, 8: cut after 8
    val = [3.0, 2.9, 2.95, 2.96, 2.97, 2.8, 2.85, 2.86, 2.87]
    lr, cuts = lp.expected_lr(val, 1e-3, 0.5, 1, 1e-5)
    assert cuts == [4, 8] and np.allclose(lr, [1e-3] * 4 + [5e-4] * 4 + [2.5e-4])
    run = tmp_path / "run"
    (run / "s0").mkdir(parents=True)
    (run / "s0" / "results.json").write_text(json.dumps(
        {"_config": {"lr_plateau": {"factor": 0.5, "patience": 1, "min_lr": 1e-5}}}))
    (run / "s0" / "training_summary.json").write_text(json.dumps({"lr_cut_after_epochs": cuts}))

    def write(rates):
        with (tmp_path / "s0.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(METRIC_COLUMNS)
            for e, (v, r) in enumerate(zip(val, rates, strict=True), start=1):
                w.writerow([dict(epoch=e, passes=e, val_loss=v, lr=f"{r:.6g}").get(c, 0)
                            for c in METRIC_COLUMNS])

    write(lr)
    assert lp.schedule_check(tmp_path / "s0.csv", run)["ok"]
    write(np.r_[lr[:5], [1e-3] * 4])                  # a cut that did not happen
    assert not lp.schedule_check(tmp_path / "s0.csv", run)["ok"]
