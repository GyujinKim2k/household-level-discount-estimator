"""M households per solve, and the education index that must survive to disk.

Two things here would ruin a 9.5-day run without failing anything:

* **``panel_id`` keyed on the household instead of the draw.** With M>1 the
  shard holds M rows per theta. If the split keys on the row, two households
  sharing a theta land on opposite sides and validation leaks -- the same
  failure as splitting windows of one panel, and just as invisible in the loss.
* **The education index not reaching the shard.** It cannot be recovered
  afterwards, and without it the dataset cannot be split per group, conditioned
  on, or reweighted to population shares, which is the entire reason for
  drawing it.

These use synthetic shards rather than the solver: the concern is the
bookkeeping around it, and a real solve would make the test minutes long.
"""

import numpy as np
import pytest
import torch

from hh_npe.data.windows import build_windowed
from hh_npe.simulator.dispatch import AGE_START_SIM
from hh_npe.data.waves import FEATURES_TWOASSET_AGE

T = 71
N_DRAWS = 6
PANEL_KEYS = ("income", "consumption", "liquid_assets", "illiquid_assets",
              "cash_on_hand", "income_state", "t_age", "alive")


def _write_shard(path, lo, hi, m, educ=None, seed=0):
    """A shard with M households per draw, shaped like the real thing."""
    rng = np.random.default_rng(seed)
    n = (hi - lo) * m
    panel = {}
    for k in PANEL_KEYS:
        if k == "t_age":
            panel[k] = np.tile(np.arange(T, dtype=float), (n, 1))
        elif k == "alive":
            panel[k] = np.ones((n, T))
        elif k == "income_state":
            panel[k] = np.zeros((n, T))
        else:
            # Distinct per row, so a mixed-up row ordering is detectable.
            panel[k] = (rng.random((n, T)) + np.arange(n)[:, None]) * 1000.0
    extra = {} if educ is None else {"educ": educ}
    np.savez(path, x=np.zeros((n, 7, 5)), alive=np.ones((n, 7)),
             lo=lo, hi=hi, n_households=m, **extra,
             **{f"panel_{k}": v for k, v in panel.items()})


@pytest.fixture
def shard_m4(tmp_path):
    p = tmp_path / "shard_00000.npz"
    _write_shard(p, 0, N_DRAWS, 4)
    return p


def test_panel_id_is_the_draw_not_the_household(shard_m4):
    """The load-bearing one. M=4 households per draw must share a panel_id."""
    theta = np.random.default_rng(0).random((N_DRAWS, 3))
    _th, _x, pid = build_windowed([shard_m4], theta, k=1, n_waves=7,
                                  fixed_start=30, features=FEATURES_TWOASSET_AGE)
    assert len(pid) == N_DRAWS * 4
    assert len(pid.unique()) == N_DRAWS
    counts = torch.bincount(pid)
    assert (counts[counts > 0] == 4).all()


def test_theta_repeats_across_the_households_of_one_draw(shard_m4):
    theta = np.random.default_rng(1).random((N_DRAWS, 3))
    th, _x, pid = build_windowed([shard_m4], theta, k=1, n_waves=7,
                                 fixed_start=30, features=FEATURES_TWOASSET_AGE)
    for d in pid.unique():
        rows = th[pid == d]
        assert torch.allclose(rows, rows[0].expand_as(rows))
        np.testing.assert_allclose(rows[0].numpy(), theta[int(d)], rtol=1e-6)


def test_households_of_one_draw_are_not_identical(shard_m4):
    """If M households collapsed to one trajectory, M would buy nothing."""
    theta = np.random.default_rng(2).random((N_DRAWS, 3))
    _th, x, pid = build_windowed([shard_m4], theta, k=1, n_waves=7,
                                 fixed_start=30, features=FEATURES_TWOASSET_AGE)
    first = x[pid == pid[0]]
    assert not torch.allclose(first[0], first[1])


def test_m_equals_one_shards_still_work(tmp_path):
    """Every existing shard has M=1 and no `n_households` field."""
    p = tmp_path / "shard_00000.npz"
    _write_shard(p, 0, N_DRAWS, 1)
    theta = np.random.default_rng(3).random((N_DRAWS, 3))
    _th, x, pid = build_windowed([p], theta, k=1, n_waves=7, fixed_start=30,
                                 features=FEATURES_TWOASSET_AGE)
    assert len(pid) == N_DRAWS and len(pid.unique()) == N_DRAWS
    assert x.shape[0] == N_DRAWS


def test_k_windows_and_m_households_compose(shard_m4):
    """k windows per household on top of M households: k*M rows per draw, all
    still one group."""
    theta = np.random.default_rng(4).random((N_DRAWS, 3))
    _th, _x, pid = build_windowed([shard_m4], theta, k=3, n_waves=7,
                                  start_low=25, start_high=40, seed=0,
                                  features=FEATURES_TWOASSET_AGE)
    assert len(pid) == N_DRAWS * 4 * 3
    assert len(pid.unique()) == N_DRAWS


def test_ragged_panel_is_rejected(tmp_path):
    """A row count that is not a whole number of households per draw means the
    shard and its header disagree; guessing M would silently mispair theta."""
    p = tmp_path / "shard_00000.npz"
    _write_shard(p, 0, N_DRAWS, 4)
    d = dict(np.load(p))
    for k in d:
        if k.startswith("panel_"):
            d[k] = d[k][:-3]           # 21 rows for 6 draws
    np.savez(p, **d)
    with pytest.raises(SystemExit, match="not a whole number"):
        build_windowed([p], np.zeros((N_DRAWS, 3)), k=1, n_waves=7,
                       fixed_start=30, features=FEATURES_TWOASSET_AGE)


def test_education_index_is_stored_per_draw(tmp_path):
    """Unrecoverable afterwards: the whole point of drawing it is the split."""
    p = tmp_path / "shard_00000.npz"
    educ = np.array([0, 1, 2, 0, 1, 2])
    _write_shard(p, 0, N_DRAWS, 4, educ=educ)
    d = np.load(p)
    assert "educ" in d.files
    np.testing.assert_array_equal(d["educ"], educ)
    assert len(d["educ"]) == int(d["hi"]) - int(d["lo"])
    assert int(d["n_households"]) == 4
