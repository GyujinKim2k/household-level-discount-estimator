"""The liquidation-penalty scale in generation (RESULTS 43.11).

Scale 1 must be exactly the generator as before, so every existing dataset
and its resume check are untouched; another scale must reach the solve and be
recorded with the shards, or a misspecification set would silently be one
more copy of the SBC set.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.generate_dataset import _solver_config

torch = pytest.importorskip("torch")

THETAS = np.array([[0.7, 0.98, 2.0], [0.6, 0.97, 4.0]])
KW = dict(seed_base=5, start_age=30, n_waves=5, wave_years=2, grid="coarse",
          theta_batch=4, chunk=8, return_panels=True, n_households=20)


def _args(**kw):
    a = dict(simulator="twoasset", grid="full", device="cpu", start_age=30, n_waves=7,
             wave_years=2, seed=40_000_000, no_panel=False, n_households=16,
             educ="comphs", card_types=True, proposal="uniform",
             rgamma_range=[1.021, 1.075], init_pool=None, liqpen_scale=1.0)
    return SimpleNamespace(**(a | kw))


def test_scale_one_is_not_recorded_and_another_is():
    assert "liqpen_scale" not in _solver_config(_args())
    assert _solver_config(_args(liqpen_scale=0.5))["liqpen_scale"] == 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_scale_one_is_the_generator_as_before_and_another_changes_the_solve():
    from hh_npe.simulator.dispatch import simulate_batch_twoasset_gpu

    _x0, _a0, p0 = simulate_batch_twoasset_gpu(THETAS, **KW)
    _x1, _a1, p1 = simulate_batch_twoasset_gpu(THETAS, liqpen_scale=1.0, **KW)
    for k in p0:
        np.testing.assert_array_equal(p0[k], p1[k])
    _x2, _a2, p2 = simulate_batch_twoasset_gpu(THETAS, liqpen_scale=0.25, **KW)
    assert any(not np.array_equal(p0[k], p2[k]) for k in ("consumption", "illiquid_assets"))
    # Same draws, same shocks: income does not depend on the penalty.
    np.testing.assert_array_equal(p0["income"], p2["income"])
