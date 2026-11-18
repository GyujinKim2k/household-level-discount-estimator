"""RESULTS 72: the learning-rate floor of 1e-7, four models, annual against
biennial in the same years.

The low-floor command must be RESULTS 71's recipe with only the floor changed,
the floor must be one torch's scheduler reaches cut by cut, the four arms must
be the two same-years pairs inside ages 24-64, the paired rows must come from
identical windows, and the training measures must read what happened below
1e-5.
"""

import csv
import json
import shlex

import numpy as np
import pytest

from hh_npe.npe.train import METRIC_COLUMNS
from scripts import low_floor as lf
from scripts import lr_plateau as lp
from scripts import optionA as oa
from scripts import waves_default as wd
from scripts import waves_experiment as we
from tests.test_waves_default import EXPLICIT, NOISE

SH = oa.Path("scripts/run_low_floor.sh")


def low_flags() -> list[str]:
    """The chain's low-floor flags, as it passes them."""
    line = next(s for s in SH.read_text().splitlines() if s.startswith("LOW="))
    floor = next(s for s in SH.read_text().splitlines() if s.startswith("FLOOR="))
    return shlex.split(line.split("=", 1)[1].strip('"').replace("$FLOOR", floor.split("=")[1]))


def test_the_low_floor_is_the_recipe_with_only_the_floor_changed():
    a = vars(oa.parse(["train", "--recipe", "plateau_fresh", *NOISE, "--n_waves", "14"]))
    b = vars(oa.parse(["train", *low_flags(), *NOISE, "--n_waves", "14"]))
    assert a.pop("recipe") == "plateau_fresh" and b.pop("recipe") is None
    assert (a.pop("lr_floor"), b.pop("lr_floor")) == (1e-5, lf.FLOOR)
    assert a == b
    assert low_flags() == [*EXPLICIT[:6], "1e-7", *EXPLICIT[7:]]
    assert oa.schedule(1, b["patience"], b["cap"], b["lr_plateau"], b["lr_factor"],
                       lf.FLOOR) == (20, 600, {"factor": 0.5, "patience": 4, "min_lr": 1e-7})


def test_every_cut_down_to_the_floor_counts():
    """Halving from 1e-3 reaches 1e-7 at the 14th cut. torch's scheduler skips a
    cut smaller than its eps of 1e-8; the smallest here, 1.22e-7 to 1e-7, is
    2.2e-8, and expected_lr (eps 1e-8, as torch) counts all 14."""
    val = [1.0] + [2.0] * 100                       # never a gain after the first epoch
    lr, cuts = lp.expected_lr(val, 1e-3, 0.5, 4, lf.FLOOR)
    assert len(cuts) == 14 and np.isclose(lr[-1], lf.FLOOR) and lr.min() >= lf.FLOOR
    assert 1e-3 * 0.5 ** 13 - lf.FLOOR > 1e-8


def test_the_four_arms_are_the_two_same_years_pairs_inside_ages_24_to_64():
    assert set(lf.ARMS) == {x for pair in lf.SAME_YEARS for x in pair}
    for an, bi in lf.SAME_YEARS:   # annual against biennial, never the same count
        assert an[0] == "a" and bi[0] == "w"
        assert we.parse_arm(an)[0] != we.parse_arm(bi)[0]
        assert 0 <= wd.aw.years_read(an) - wd.aw.years_read(bi) <= 1
    for a in lf.ARMS:
        n, wy = we.parse_arm(a)
        assert we.start_high(n, wy) + wy * n - 1 <= we.LAST_AGE
    sh = SH.read_text()
    floor = next(s for s in sh.splitlines() if s.startswith("FLOOR="))
    assert f'ARMS="{" ".join(lf.ARMS)}"' in sh and float(floor.split("=")[1]) == lf.FLOOR


def test_the_paired_rows_are_one_set_of_windows(tmp_path):
    rows = np.arange(6.0)
    (tmp_path / "floor").mkdir()
    np.savez(tmp_path / "floor" / "cross_noise_a14.npz", low__low=rows + 1, low__floor5=rows + 1,
             floor5__low=rows, floor5__floor5=rows, w=np.ones(6), pid=np.arange(6))
    d, b, w = lf.floor_rows(tmp_path, "a14")
    assert np.array_equal(d - b, np.ones(6)) and np.array_equal(w, np.ones(6))
    np.savez(tmp_path / "floor" / "cross_noise_a20.npz", low__low=rows + 1, low__floor5=rows,
             floor5__low=rows, floor5__floor5=rows, w=np.ones(6), pid=np.arange(6))
    with pytest.raises(SystemExit):
        lf.floor_rows(tmp_path, "a20")


def test_the_gate_is_beta_delta_and_rho():
    assert wd.GATED == ("beta", "delta", "crra") and wd.BAND == (0.88, 0.92)


def test_what_training_below_1e5_did(tmp_path):
    """Epochs 1-4 at 1e-5 or above, 5-8 below it; the record's best comes at
    epoch 6, below 1e-5, by a gain the CSV's 6 digits would hide."""
    lr = [1e-3, 1e-4, 1e-5, 1e-5, 5e-6, 1e-6, 1e-7, 1e-7]
    val = [3.0, 2.0, 1.5, 1.4, 1.39, 1.39 - 1e-8, 1.395, 1.396]
    run, mdir = tmp_path / "run", tmp_path / "metrics"
    (run / "s0").mkdir(parents=True)
    mdir.mkdir()
    (run / "s0" / "results.json").write_text(json.dumps(
        {"_config": {"lr_plateau": {"factor": 0.5, "patience": 0, "min_lr": 1e-7}}}))
    (run / "s0" / "training_summary.json").write_text(json.dumps({"validation_loss": val}))
    with (mdir / "s0.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(METRIC_COLUMNS)
        for e, (v, r) in enumerate(zip(val, lr, strict=True), start=1):
            w.writerow([dict(epoch=e, passes=e, val_loss=f"{v:.6g}", lr=r).get(c, 0)
                        for c in METRIC_COLUMNS])
    b = lf.below(mdir, run)["s0"]
    assert b["floor"] == 1e-7 and b["lowest_rate"] == 1e-7 and b["reached_floor"]
    assert b["passes_below"] == 4 and b["best_below"]
    best = min(val)
    assert b["gain_from_rate"] == pytest.approx(2.0 - best)     # best before epoch 3: 2.0
    assert b["gain_below_rate"] == pytest.approx(1.4 - best)    # best before epoch 5: 1.4
