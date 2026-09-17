"""Per-group filtering and education conditioning.

`panel_id` from `build_windowed` is the Sobol draw index, and `educ` is stored
per draw, so `educ_by_draw(shards)[panel_id]` recovers each row's group with no
change to the windowing code. These tests pin that relationship, because it is
an implicit contract between three files and nothing would fail loudly if it
broke -- the rows would simply be labelled with the wrong group.
"""

import numpy as np
import pytest
import torch

from scripts.compare_windows import educ_by_draw, one_hot_educ
from tests.test_multi_household_shards import _write_shard


def test_educ_by_draw_indexes_by_global_draw_number(tmp_path):
    a, b = tmp_path / "shard_00000.npz", tmp_path / "shard_00001.npz"
    _write_shard(a, 0, 4, 2, educ=np.array([0, 1, 2, 0]))
    _write_shard(b, 4, 8, 2, educ=np.array([2, 2, 1, 0]))
    got = educ_by_draw([a, b])
    assert list(got) == [0, 1, 2, 0, 2, 2, 1, 0]


def test_educ_by_draw_returns_none_for_comphs_only_shards(tmp_path):
    """A comphs-only dataset has no `educ`; the caller must be told rather than
    silently given zeros, which would read as 'every draw is comphs'."""
    p = tmp_path / "shard_00000.npz"
    _write_shard(p, 0, 4, 2)
    assert educ_by_draw([p]) is None


def test_educ_survives_the_round_trip_through_panel_id(tmp_path):
    """The contract the per-group split depends on."""
    from hh_npe.data.waves import FEATURES_TWOASSET_AGE
    from hh_npe.data.windows import build_windowed

    p = tmp_path / "shard_00000.npz"
    educ = np.array([0, 1, 2, 0, 1, 2])
    _write_shard(p, 0, 6, 4, educ=educ)
    _th, _x, pid = build_windowed([p], np.zeros((6, 3)), k=1, n_waves=7,
                                  fixed_start=30,
                                  features=FEATURES_TWOASSET_AGE)
    rows = educ_by_draw([p])[pid.numpy()]
    assert len(rows) == 6 * 4
    # Every household of a draw carries that draw's group.
    for d in pid.unique():
        assert len(set(rows[pid.numpy() == int(d)])) == 1
        assert rows[pid.numpy() == int(d)][0] == educ[int(d)]


@pytest.mark.parametrize("g", [0, 1, 2])
def test_one_hot_is_constant_within_a_sequence(g):
    x = torch.rand(5, 7, 5)
    y = one_hot_educ(x, np.full(5, g))
    assert y.shape == (5, 7, 8)
    assert torch.equal(y[..., :5], x)          # original untouched
    block = y[..., 5:]
    assert torch.equal(block[:, 0], block[:, -1])   # constant across waves
    assert block[0, 0, g] == 1.0 and block[0, 0].sum() == 1.0


def test_one_hot_varies_across_rows_so_global_normalisation_is_defined():
    """The embedder normalises per feature over (rows, waves). A block that were
    constant everywhere would have zero std and produce NaNs."""
    y = one_hot_educ(torch.rand(30, 7, 5), np.arange(30) % 3)
    assert (y[..., 5:].std(dim=(0, 1)) > 0).all()


def test_unclassified_education_does_not_produce_an_invalid_row():
    """-1 marks a household we cannot classify. It must still yield a valid
    one-hot rather than an out-of-range index."""
    y = one_hot_educ(torch.rand(2, 7, 5), np.array([-1, 2]))
    assert torch.isfinite(y).all()
    assert y[0, 0, 5:].sum() == 1.0
