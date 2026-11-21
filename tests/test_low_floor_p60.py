"""RESULTS 73: a longer patience with the low floor, four models.

The patience-60 command must be RESULTS 72's with only the stop changed, the
arms must be RESULTS 72's, the three settings' paired rows must come from
identical windows, and the shorter stops must be read correctly off a
patience-60 trajectory.
"""

import shlex

import numpy as np
import pytest

from scripts import low_floor as lf
from scripts import low_floor_p60 as lp6
from scripts import optionA as oa
from scripts import waves_default as wd
from tests.test_low_floor import low_flags
from tests.test_waves_default import EXPLICIT, NOISE

SH = oa.Path("scripts/run_low_floor_p60.sh")


def flags(name: str) -> list[str]:
    """One of the chain's flag sets (P60 or P20), as it passes them."""
    lines = SH.read_text().splitlines()
    var = {k: next(s for s in lines if s.startswith(f"{k}=")).split("=", 1)[1]
           for k in ("FLOOR", "PATIENCE")}
    line = next(s for s in lines if s.startswith(f"{name}=")).split("=", 1)[1].strip('"')
    for k, v in var.items():
        line = line.replace(f"${k}", v)
    return shlex.split(line)


def test_patience_60_is_results_72_with_only_the_stop_changed():
    a = vars(oa.parse(["train", *low_flags(), *NOISE, "--n_waves", "14"]))
    b = vars(oa.parse(["train", *flags("P60"), *NOISE, "--n_waves", "14"]))
    assert (a.pop("patience"), b.pop("patience")) == (20, lp6.PATIENCE)
    assert a == b
    assert flags("P60") == [*EXPLICIT[:6], "1e-7", "--patience", "60", *EXPLICIT[9:]]
    assert flags("P20") == low_flags()          # the smoke's stand-ins: RESULTS 72's command
    assert oa.schedule(1, lp6.PATIENCE, b["cap"], b["lr_plateau"], b["lr_factor"],
                       b["lr_floor"]) == (60, 600, {"factor": 0.5, "patience": 4,
                                                    "min_lr": 1e-7})


def test_the_arms_are_results_72s():
    assert lp6.ARMS == lf.ARMS and lp6.SAME_YEARS == lf.SAME_YEARS
    assert f'ARMS="{" ".join(lf.ARMS)}"' in SH.read_text()
    assert lp6.SETTING[lp6.P20] == (lf.FLOOR, 20) and lp6.SETTING[lp6.P60][0] == lf.FLOOR


def test_the_stop_is_read_from_the_config():
    assert lp6.stop_of({"stop_after_epochs": 60}) == 60
    assert lp6.stop_of({}) == lp6.stop_of({"stop_after_epochs": None}) == 20
    cfg = {"lr_plateau": {"min_lr": 1e-7}, "fresh_noise": True}
    lp6.check_setting(lp6.P20, "a14", cfg)
    with pytest.raises(SystemExit):
        lp6.check_setting(lp6.P60, "a14", cfg)
    lp6.check_setting(lp6.P60, "a14", {**cfg, "stop_after_epochs": 60})
    with pytest.raises(SystemExit):
        lp6.check_setting(lp6.FIVE, "a14", {**cfg, "stop_after_epochs": 60})


def test_the_three_settings_rows_are_one_set_of_windows(tmp_path):
    rows = np.arange(6.0)
    (tmp_path / "compare").mkdir()
    cols = {f"{net}__{noise}": rows + k for k, net in enumerate(lp6.TAGS) for noise in lp6.TAGS}
    np.savez(tmp_path / "compare" / "cross_noise_a14.npz", **cols, w=np.ones(6), pid=np.arange(6))
    q, w = lp6.compare_rows(tmp_path, "a14")
    assert [q[t][0] for t in lp6.TAGS] == [0.0, 1.0, 2.0] and np.array_equal(w, np.ones(6))
    cols[f"{lp6.P20}__{lp6.FIVE}"] = rows
    np.savez(tmp_path / "compare" / "cross_noise_a20.npz", **cols, w=np.ones(6), pid=np.arange(6))
    with pytest.raises(SystemExit):
        lp6.compare_rows(tmp_path, "a20")


def test_the_gate_is_beta_delta_and_rho():
    assert wd.GATED == ("beta", "delta", "crra") and wd.BAND == (0.88, 0.92)


def test_the_shorter_stops_off_a_patience_60_trajectory():
    """Gains to epoch 10, then 25 epochs without one, then a lower best at
    epoch 36: a stop after 20 fires at epoch 30 and keeps epoch 10's best; a
    stop after 40 never fires before the end and keeps the final best."""
    v = list(np.linspace(3.0, 2.0, 10)) + [2.1] * 25 + [1.9] + [2.0] * 30
    s = lp6.stops(v)
    assert s["20"]["stop_epoch"] == 30 and not s["20"]["same_best"]
    assert s["20"]["best_then"] == pytest.approx(2.0)
    assert s["20"]["fall_after"] == pytest.approx(0.1)
    assert s["40"]["stop_epoch"] is None and s["40"]["same_best"] and s["40"]["fall_after"] == 0
