"""`simulate_from` must continue a household exactly as `simulate` would.

It is the engine of the out-of-sample test: each PSID household is dropped into
the solved model at its own observed wave-5 state and run forward. If it read
the policy arrays at a different index than `simulate` -- the wrong period, the
pre- instead of post-income liquid position -- every prediction would be off by
a systematic amount that no comparison between parameter values would reveal,
because all arms would share it.

From an identical state the policy is deterministic, so the first period's
consumption and the next period's wealth must match `simulate` to the dollar.
Only the future income shocks differ.
"""

import numpy as np
import pytest

from hh_npe.simulator.twoasset import COARSE, simulate, simulate_from, solve


@pytest.fixture(scope="module")
def sol():
    return solve(0.85, 0.97, 2.0, spec=COARSE)


@pytest.fixture(scope="module")
def panel(sol):
    return simulate(sol, n_households=400, seed=3)


@pytest.mark.parametrize("t0", [5, 15, 25])
def test_first_step_matches_simulate(sol, panel, t0):
    liq = panel["liquid_assets"][:, t0]
    ill = panel["illiquid_assets"][:, t0]
    inc = panel["income"][:, t0]
    out = simulate_from(sol, np.full(len(liq), t0), liq, ill, inc,
                        horizon=1, n_sims=1, seed=0)
    # The persistent state is inferred from one income observation, which is
    # only an approximation; restrict to households where it is recovered, so
    # this tests the indexing rather than the inference.
    ymean = np.log(np.maximum(inc, 1.0))
    from hh_npe.simulator import grids
    s_hat = np.argmin(np.abs((ymean - grids.mean_log_income(sol.age, sol.spec.calib)[t0])
                             [:, None] - sol.states[None, :]), axis=1)
    ok = s_hat == panel["income_state"][:, t0]
    assert ok.sum() > 50, "too few households to test on"
    np.testing.assert_array_equal(out["consumption"][ok, 0, 0],
                                  panel["consumption"][ok, t0])
    np.testing.assert_array_equal(out["illiquid_assets"][ok, 0, 1],
                                  panel["illiquid_assets"][ok, t0 + 1])
    # Liquid wealth is defined as simulate defines it (post-income cash on the
    # grid, minus income), so at t0 it reproduces the observed input exactly.
    np.testing.assert_array_equal(out["liquid_assets"][ok, 0, 0],
                                  panel["liquid_assets"][ok, t0])


def test_shocks_do_not_depend_on_preferences(sol):
    """Common random numbers: same seed, different preferences, same income."""
    other = solve(0.6, 0.99, 3.0, spec=COARSE)
    args = (np.array([10, 20]), np.array([1000.0, -500.0]),
            np.array([0.0, 20000.0]), np.array([30000.0, 50000.0]))
    a = simulate_from(sol, *args, horizon=4, n_sims=50, seed=7)
    b = simulate_from(other, *args, horizon=4, n_sims=50, seed=7)
    np.testing.assert_array_equal(a["income"], b["income"])
    assert not np.array_equal(a["consumption"], b["consumption"])


def test_shapes_and_horizon_guard(sol):
    out = simulate_from(sol, np.array([10, 12, 14]), np.zeros(3), np.zeros(3),
                        np.full(3, 40000.0), horizon=4, n_sims=7, seed=0)
    assert out["consumption"].shape == (3, 7, 5)
    with pytest.raises(ValueError, match="horizon"):
        simulate_from(sol, np.array([len(sol.age) - 2]), np.zeros(1),
                      np.zeros(1), np.full(1, 4e4), horizon=4)
