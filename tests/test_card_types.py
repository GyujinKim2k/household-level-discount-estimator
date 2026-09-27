"""Card-access types in data generation (RESULTS.md 24.1, 26).

A no-card draw must be solved with a zero credit line; a cardholder draw must be
exactly what it was before card types existed. Both are silent if wrong: a
no-card draw solved with credit just looks like another cardholder, and a
perturbed cardholder draw shifts the dataset without failing anything.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

from hh_npe.simulator.dispatch import simulate_batch_twoasset_gpu  # noqa: E402

THETAS = np.array([[0.7, 0.98, 2.0], [0.9, 0.99, 3.5], [0.6, 0.97, 1.5]])
KW = dict(seed_base=5, start_age=30, n_waves=5, wave_years=2, grid="coarse",
          theta_batch=4, chunk=8, return_panels=True, n_households=20)


def test_no_card_draws_never_borrow():
    _x, _a, p = simulate_batch_twoasset_gpu(THETAS, card=np.array([0, 0, 0]), **KW)
    # liquid_assets is post-income cash minus income: grid rounding can dip it
    # below zero by at most one xjump (4000 on the coarse grid).
    assert p["liquid_assets"].min() > -4000


def test_cardholder_draws_are_unchanged_by_the_card_machinery():
    _x0, _a0, p0 = simulate_batch_twoasset_gpu(THETAS, **KW)
    _x1, _a1, p1 = simulate_batch_twoasset_gpu(THETAS, card=np.array([1, 1, 1]), **KW)
    for k in p0:
        np.testing.assert_array_equal(p0[k], p1[k])


def test_mixed_types_keep_draw_order_and_each_draw_its_own_type():
    card = np.array([1, 0, 1])
    _x, _a, p = simulate_batch_twoasset_gpu(THETAS, card=card, **KW)
    _x0, _a0, only1 = simulate_batch_twoasset_gpu(THETAS, card=np.ones(3, int), **KW)
    rows = lambda j: slice(j * 20, (j + 1) * 20)  # noqa: E731
    np.testing.assert_array_equal(p["liquid_assets"][rows(0)], only1["liquid_assets"][rows(0)])
    np.testing.assert_array_equal(p["liquid_assets"][rows(2)], only1["liquid_assets"][rows(2)])
    assert p["liquid_assets"][rows(1)].min() > -4000
