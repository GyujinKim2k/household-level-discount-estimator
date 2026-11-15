"""RESULTS 71: the default recipe, and the waves experiment run under it.

``train --recipe plateau_fresh`` must be exactly its six flags (so it trains
and writes the config they do), refuse what contradicts it, and leave a bare
``train`` as before. The new arms' windows must lie inside ages 24-64, the
PSID projection must read the reference's covariance, and the paired rows
must come from identical windows.
"""

import json

import numpy as np
import pytest

from scripts import optionA as oa
from scripts import waves_default as wd
from scripts import waves_experiment as we

NOISE = ["--obs_noise", "income=0.1:0.3", "consumption=0.1:0.2"]
EXPLICIT = ["--fresh_noise", "--lr_plateau", "5", "--lr_factor", "0.5", "--lr_floor", "1e-5",
            "--patience", "20", "--cap", "600"]


def test_the_recipe_is_its_six_flags():
    a = vars(oa.parse(["train", "--recipe", "plateau_fresh", *NOISE, "--n_waves", "20"]))
    b = vars(oa.parse(["train", *EXPLICIT, *NOISE, "--n_waves", "20"]))
    assert a.pop("recipe") == "plateau_fresh" and b.pop("recipe") is None
    assert a == b
    # so the schedule, the stop and the cap are RESULTS 69's rule at patience 20
    assert oa.schedule(1, a["patience"], a["cap"], a["lr_plateau"], a["lr_factor"],
                       a["lr_floor"]) == (20, 600, {"factor": 0.5, "patience": 4, "min_lr": 1e-5})


def test_a_bare_train_keeps_the_old_settings():
    a = oa.parse(["train", *NOISE])
    assert (a.fresh_noise, a.lr_plateau, a.recipe) == (False, None, None)
    assert (a.patience, a.cap) == (oa.TRAINING["stop_after_epochs"], oa.TRAINING["max_num_epochs"])
    assert oa.schedule(1, a.patience, a.cap, a.lr_plateau) == (a.patience, None, None)


def test_flags_that_agree_with_the_recipe_are_accepted():
    a = oa.parse(["train", "--recipe", "plateau_fresh", "--fresh_noise", "--patience", "20",
                  "--max_epochs", "3", *NOISE])
    assert (a.fresh_noise, a.patience, a.cap, a.max_epochs) == (True, 20, 600, 3)


@pytest.mark.parametrize("extra", [["--patience", "60"], ["--lr_plateau", "3"],
                                   ["--lr_factor", "0.3"], ["--cap", "300"],
                                   ["--lr_floor", "1e-6"], ["--noise_copies", "8"]])
def test_the_recipe_refuses_what_contradicts_it(extra):
    with pytest.raises(SystemExit):
        oa.parse(["train", "--recipe", "plateau_fresh", *NOISE, *extra])


def test_the_recipe_needs_noise_to_redraw():
    with pytest.raises(SystemExit):
        oa.parse(["train", "--recipe", "plateau_fresh"])
    a = oa.parse(["train", "--recipe", "plateau_fresh", "--noise_model", "noise.json"])
    assert a.fresh_noise


def test_cross_noise_compares_a_recipe_arm_with_an_old_one():
    """What the recipe writes beyond an old arm's config is training, not
    windows (cross_noise_heldout.py skips these keys)."""
    for key in ("fresh_noise", "fresh_noise_restored_best_at_cap", "lr_plateau", "max_epochs",
                "stop_after_epochs"):
        assert key in oa.TRAINING_KEYS


def test_the_annual_ladder_lies_inside_ages_24_to_64():
    assert [(we.start_low(1), we.start_high(n, 1)) for n in (7, 10, 14, 20)] == \
        [(25, 58), (25, 55), (25, 51), (25, 45)]
    assert [we.start_high(n) for n in (7, 10, 14, 20)] == [51, 45, 37, 25]
    for a in wd.ARMS:
        n, wy = we.parse_arm(a)
        we.check_window({"n_waves": n, "start_low": we.start_low(wy),
                         "start_high": we.start_high(n, wy), "wave_years": wy}, n, wy)
        assert we.start_high(n, wy) + wy * n - 1 <= we.LAST_AGE


def test_the_arms_and_comparisons_hang_together():
    assert wd.TRAINED == tuple(a for a in wd.ARMS if a != "w07")
    assert set(wd.OLD) <= set(wd.ARMS)
    for an, bi in wd.SAME_COUNT + wd.SAME_YEARS:
        assert an in wd.ARMS and bi in wd.ARMS and an[0] == "a" and bi[0] == "w"
    assert all(we.parse_arm(an)[0] == we.parse_arm(bi)[0] for an, bi in wd.SAME_COUNT)
    sh = (oa.Path("scripts/run_waves_default.sh")).read_text()
    assert f'TRAINED="{" ".join(wd.TRAINED)}"' in sh and f'OLD="{" ".join(wd.OLD)}"' in sh
    assert 'ALL="' + " ".join(wd.ARMS) + '"' in sh and "--recipe plateau_fresh" in sh


def test_the_psid_projection_reads_the_reference_and_keeps_its_default():
    for ref, arm in ((we.REFERENCE, we.PSID_ARM), (wd.PSID_REFERENCE, wd.PSID_ARM)):
        if not ref.exists():
            pytest.skip(f"{ref} is not on disk")
        a = json.loads(ref.read_text())["arms"][arm]
        cov, se = we.psid_cov(ref, arm)
        assert np.allclose(se, a["representative_se"])
        assert np.allclose(np.sqrt(np.diag(cov)), se)
    assert np.array_equal(we.psid_cov()[0], we.psid_cov(we.REFERENCE, "v2_b010")[0])
    assert np.round(we.psid_cov(wd.PSID_REFERENCE, wd.PSID_ARM)[1][0], 4) == 0.0269


def test_the_paired_rows_are_one_set_of_windows(tmp_path):
    rows = np.arange(6.0)
    wd.recipe_rows(tmp_path, "w10").parent.mkdir(parents=True)
    np.savez(wd.recipe_rows(tmp_path, "w10"), default__default=rows + 1, default__old=rows + 1,
             old__default=rows, old__old=rows, w=np.ones(6), pid=np.arange(6))
    d, b, w = wd.pair_rows(tmp_path, "w10")
    assert np.array_equal(d - b, np.ones(6)) and np.array_equal(w, np.ones(6))
    np.savez(wd.recipe_rows(tmp_path, "w14"), default__default=rows + 1, default__old=rows,
             old__default=rows, old__old=rows, w=np.ones(6), pid=np.arange(6))
    with pytest.raises(SystemExit):
        wd.pair_rows(tmp_path, "w14")
    f = tmp_path / "rows69.npz"
    np.savez(f, v2_fresh_rop__w07=rows + 2, w07__w07=rows, w=np.ones(6), pid=np.arange(6))
    d, b, _w = wd.pair_rows(tmp_path, "w07", f)
    assert np.array_equal(d - b, 2 * np.ones(6))


def test_the_gate_covers_beta_delta_and_rho_only():
    ok = {"heldout": {"cov90": [0.90, 0.90, 0.90, 0.80]}, "sbc_coverage_90": [0.9, 0.9, 0.9, 0.8]}
    assert wd.calibrated(ok)
    bad = {"heldout": {"cov90": [0.93, 0.90, 0.90, 0.90]}, "sbc_coverage_90": [0.9] * 4}
    assert not wd.calibrated(bad)


@pytest.mark.skipif(not wd.W07_RUN.exists(), reason="v2_fresh_rop is not on disk")
def test_w07_under_the_default_is_v2_fresh_rop_with_patience_20s_weights():
    """RESULTS 69.4: in every member a stop after 20 passes keeps the weights
    the stop after 60 kept, so v2_fresh_rop stands for w07 under the default."""
    from scripts import lr_plateau as lp
    from scripts import training_metrics as tm

    for f in sorted(wd.W07_METRICS.glob("s[0-9].csv")):
        v = np.asarray(tm.read(f)["val_loss"])
        kept = lp.patience_kept(v, 20)
        assert kept["stop_epoch"] is not None and kept["best_then"] == v.min()
    cfg = json.loads((wd.W07_RUN / "s0" / "results.json").read_text())["_config"]
    assert cfg["fresh_noise"] and cfg["lr_plateau"]["passes"] == 5
    we.check_window(cfg["window"], 7, 2)
