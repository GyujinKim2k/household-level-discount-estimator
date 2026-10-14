"""The summary likelihood's training pairs (``scripts/hierarchical_summary``,
RESULTS 45-46): the SBC set's untouched half stays out of training in every
variant, and the targeted couples join only the targeted one."""

import numpy as np
import pytest

from scripts import hierarchical_summary as hs


@pytest.fixture
def out(tmp_path):
    draw = np.r_[np.repeat(np.arange(10), 2), hs.SBC_OFFSET + np.repeat(np.arange(20), 2)]
    sbc = draw >= hs.SBC_OFFSET
    n = len(draw)
    np.savez(tmp_path / "pairs.npz", s=np.zeros((n, 4)), theta=np.ones((n, 4)), draw=draw,
             sbc=sbc, n_draws=4000, lo=np.zeros(4), hi=np.ones(4) * 2)
    np.savez(tmp_path / "targeted.npz", s=np.ones((6, 4)), theta=np.full((6, 4), 0.5),
             draw=hs.TARGET_OFFSET + np.repeat(np.arange(3), 2), card=np.zeros(6),
             n_draws=4000, lo=np.zeros(4), hi=np.ones(4) * 2)
    return tmp_path


def test_base_is_the_pairs_file_and_keeps_half_the_sbc_draws_out(out):
    s, theta, draw, train, *_ = hs.training_pairs(out, "base")
    assert len(s) == 60
    assert train[draw < hs.SBC_OFFSET].all()
    sbc_ids = np.unique(draw[draw >= hs.SBC_OFFSET])
    trained = np.unique(draw[(draw >= hs.SBC_OFFSET) & train])
    assert len(trained) == len(sbc_ids) // 2
    # households of one draw stay on one side
    for g in sbc_ids:
        assert len(np.unique(train[draw == g])) == 1


def test_targeted_adds_every_targeted_household_and_nothing_else(out):
    base = hs.training_pairs(out, "base")
    s, theta, draw, train, *_ = hs.training_pairs(out, "targeted")
    assert len(s) == len(base[0]) + 6
    np.testing.assert_array_equal(train[:len(base[0])], base[3])
    assert train[len(base[0]):].all()
    assert (draw[len(base[0]):] >= hs.TARGET_OFFSET).all()
    np.testing.assert_array_equal(theta[len(base[0]):], np.full((6, 4), 0.5))


def test_targeted_pairs_must_share_the_draw_count(out):
    d = dict(np.load(out / "targeted.npz"))
    d["n_draws"] = 8000
    np.savez(out / "targeted.npz", **d)
    with pytest.raises(SystemExit):
        hs.training_pairs(out, "targeted")
