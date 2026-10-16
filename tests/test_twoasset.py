"""Tests for the two-asset lifecycle port of Laibson et al.

The headline check is :func:`test_budget_constraint_holds`, which is their own
``assert(abs(check) < 1e-5)`` from ``LifecycleSim_BackwardInduct.m:169``
re-derived independently from the stored policy indices. It catches sign errors
in the liquid/illiquid budget split, the borrowing rate kink, and the
liquidation penalty.
"""

import dataclasses

import numpy as np
import pytest

from hh_npe.simulator import grids
from hh_npe.simulator import laibson_calibration as cal
from hh_npe.simulator.twoasset import ModelSpec, NEG, simulate, solve

# Small grid so the whole suite stays fast; the model structure is unchanged.
TINY = ModelSpec(xjump=20000.0, x_cells_per_step=4, zjump=200000.0, z_cells_per_step=3)
BENCHMARK = cal.BENCHMARK_PREFS  # (beta, delta, rho) = naive quasi-hyperbolic


@pytest.fixture(scope="module")
def sol():
    return solve(*BENCHMARK, TINY)


@pytest.fixture(scope="module")
def sol64():
    """float64 solve, so their exact ``abs(check) < 1e-5`` bound is testable."""
    import dataclasses
    return solve(*BENCHMARK, dataclasses.replace(TINY, dtype=np.float64))


def test_calibration_matches_replication_package():
    """Frozen constants are the comphs (completed-high-school) benchmark."""
    assert cal.EDUC == "comphs"
    assert cal.DEATH_PROB.shape == (71,)
    assert cal.TARGET_MOMENTS.shape == (16,)
    assert np.all(np.diff(cal.survival_share()) < 0)
    assert cal.survival_share()[0] == 1.0


def test_tauchen_rows_sum_to_one():
    states, P = grids.tauchen()
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    # Symmetric grid centred on zero.
    np.testing.assert_allclose(states, -states[::-1], atol=1e-12)


def test_stationary_is_invariant():
    _, P = grids.tauchen()
    pi = grids.stationary(P)
    np.testing.assert_allclose(pi @ P, pi, atol=1e-12)
    assert np.all(pi > 0)


def test_transitory_probs_normalized():
    age = grids.ages()
    p, y = grids.discretize_transitory(float(grids.mean_log_income(age)[10]))
    np.testing.assert_allclose(p.sum(), 1.0)
    assert np.all(y > 0)
    assert len(p) == len(y)


def test_credit_limit_is_on_the_grid():
    """Every age's borrowing limit must land exactly on a liquid grid point."""
    age = grids.ages()
    X, feasible = grids.liquid_grid(age, TINY.xjump, TINY.xmax, TINY.x_cells_per_step)
    xmin = grids.credit_limit(age, TINY.xjump)
    for t in range(len(age)):
        assert np.isclose(X[feasible[t]][0], -xmin[t])


def test_liquidation_penalty_declines_with_age():
    pen = grids.liquidation_penalty(grids.ages())
    assert np.all(np.diff(pen) < 0)
    assert 0.0 < pen[-1] < pen[0] < 0.5


def _budget_residual(sol):
    """Max absolute violation of their dynamic budget identity over the grid.

    ``R * a + (R_CC - R) * min(a, 0) + Z' == X' + Z'`` where ``a`` is what is
    left after consumption and the illiquid transfer.
    """
    X, Z, spec = sol.X, sol.Z, sol.spec
    T, nX, nZ, nS = sol.next_x.shape
    zliqpen = grids.liquidation_penalty(sol.age)
    worst = 0.0
    for t in range(T):
        ok = sol.solvable[t] & sol.feasible[t][:, None, None]
        if not ok.any():
            continue
        ix, iz, s = np.nonzero(ok)
        jx, jz = sol.next_x[t][ok], sol.next_z[t][ok]
        move = Z[jz] - Z[iz]
        after = X[ix] - sol.cons[t][ok] - move + zliqpen[t] * np.minimum(move, 0.0)
        grew = spec.R * after + (spec.R_CC - spec.R) * np.minimum(after, 0.0)
        worst = max(worst, float(np.abs(grew - X[jx]).max()))
    return worst


def test_budget_constraint_holds_float64(sol64):
    """Their own assertion, ``abs(check) < 1e-5`` (BackwardInduct.m:169)."""
    assert _budget_residual(sol64) < 1e-5


def test_budget_constraint_holds_float32(sol):
    """float32 keeps the identity to ~1e-9 relative -- ample for an argmax."""
    residual = _budget_residual(sol)
    assert residual / max(np.abs(sol.X).max(), 1.0) < 1e-6


def test_policies_stay_within_credit_limit(sol):
    """No chosen next-period liquid position may breach the next age's limit."""
    T = sol.next_x.shape[0]
    for t in range(T - 1):
        ok = sol.solvable[t] & sol.feasible[t][:, None, None]
        chosen = sol.next_x[t][ok]
        assert sol.feasible[t + 1][chosen].all(), f"limit breached at age index {t}"


def test_terminal_no_ponzi(sol):
    """Nobody dies in debt: the final-period policy never picks negative X."""
    t = sol.next_x.shape[0] - 1
    ok = sol.solvable[t] & sol.feasible[t][:, None, None]
    assert (sol.X[sol.next_x[t][ok]] >= 0).all()


def test_consumption_positive(sol):
    """Consumption is non-negative wherever an affordable action exists."""
    ok = sol.solvable & sol.feasible[:, :, None, None]
    assert np.all(sol.cons[ok] >= -1e-6)
    assert np.isfinite(sol.cons).all()


def test_unsolvable_states_are_deep_in_debt(sol):
    """Only heavily indebted, illiquid-poor states can be unsolvable.

    These are households that cannot repay from any affordable action. They
    exist in their MATLAB too (every payoff is -1e100); the forward pass never
    reaches them, but the masks must be honest about it.
    """
    bad = ~sol.solvable
    assert bad.any(), "expected some unaffordable corner states"
    # Every unsolvable state has a negative liquid position -- or, with
    # MATLAB's infinities (RESULTS 55), none at all: from zero cash every
    # affordable plan can force a year at zero consumption, worth -inf. Cash
    # on hand includes the year's income, so households never hold zero.
    ix = np.nonzero(bad.any(axis=(0, 2, 3)))[0]
    assert (sol.X[ix] < 0).all() if not TINY.matlab_infinities else (sol.X[ix] <= 0).all()
    # The forward pass must never land on one.
    panel = simulate(sol, n_households=32, seed=0)
    assert np.isfinite(panel["consumption"]).all()


def test_illiquid_grid_is_never_negative(sol):
    assert sol.Z[0] == 0.0
    assert np.all(sol.Z >= 0)


def test_simulate_panel_shape_and_keys(sol):
    panel = simulate(sol, n_households=16, seed=0)
    T = len(sol.age)
    for key in ("income", "consumption", "liquid_assets", "illiquid_assets",
                "cash_on_hand", "income_state", "t_age"):
        assert panel[key].shape == (16, T), key
    assert np.all(panel["income"] > 0)
    assert np.all(panel["illiquid_assets"] >= 0)
    assert np.all(np.diff(panel["t_age"], axis=1) == 1)


def test_simulate_is_deterministic_given_seed(sol):
    a = simulate(sol, n_households=8, seed=3)
    b = simulate(sol, n_households=8, seed=3)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_liquid_assets_can_go_negative(sol):
    """Credit cards exist: some household must borrow at some point.

    This is the whole point of Phase 3 -- the Phase 1-2 MVP had
    ``BoroCnstArt = 0`` and could never produce a negative liquid position.
    """
    panel = simulate(sol, n_households=64, seed=0)
    assert (panel["liquid_assets"] < 0).any()


def test_present_bias_lowers_wealth():
    """A naive present-biased agent accumulates less than an exponential one."""
    beta, delta, rho = BENCHMARK
    biased = simulate(solve(beta, delta, rho, TINY), n_households=48, seed=1)
    patient = simulate(solve(1.0, delta, rho, TINY), n_households=48, seed=1)
    mid = slice(20, 45)  # ages 40-64
    assert (biased["illiquid_assets"][:, mid].mean()
            < patient["illiquid_assets"][:, mid].mean())


def test_works_with_wave_aggregation(sol):
    """The panel plugs into the pre-registered biennial aggregation unchanged."""
    from hh_npe.data.waves import aggregate_waves

    panel = simulate(sol, n_households=8, seed=0)
    x, alive = aggregate_waves(panel, age_start_sim=cal.AGE_START, start_age=30)
    assert x.shape == (8, 5, 3)
    assert alive.all()  # no mortality in the forward pass, so no rebirth


def test_solve_rejects_bad_parameters():
    with pytest.raises(ValueError, match="delta"):
        solve(0.5, 1.0, 2.0, TINY)
    with pytest.raises(ValueError, match="positive"):
        solve(-0.1, 0.95, 2.0, TINY)


def test_expectation_step_preserves_constants():
    """Integrating a state-independent value must return that same value.

    Rows of the transition matrix sum to 1, so if V does not vary with the
    persistent income state the expectation is a no-op. This catches a
    transposed transition matrix (``P.T`` is not row-stochastic) and any
    normalization error in the transitory-shock discretization.
    """
    from hh_npe.simulator.twoasset import _expectation_step

    age = grids.ages()
    X, _ = grids.liquid_grid(age, TINY.xjump, TINY.xmax, TINY.x_cells_per_step)
    states, P = grids.tauchen(n_states=TINY.n_income_states)
    nX, nZ, nS = len(X), 4, TINY.n_income_states

    # V constant everywhere: shifting along X cannot change it either.
    V = np.full((nX, nZ, nS), 3.25, dtype=np.float64)
    ev = _expectation_step(
        V, X, float(grids.mean_log_income(age)[10]), states, P,
        dataclasses.replace(TINY, dtype=np.float64),
        float(grids.credit_limit(age, TINY.xjump)[10]), nX, nZ, nS,
    )
    np.testing.assert_allclose(ev, 3.25, rtol=1e-6)


def test_transition_matrix_is_row_stochastic_not_column():
    """Guard the orientation explicitly: P.T is not a transition matrix."""
    _, P = grids.tauchen()
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    assert not np.allclose(P.T.sum(axis=1), 1.0), "P is symmetric; test is vacuous"


def test_ev_centring_is_exact_under_shift():
    """Adding a constant to EV must not change the chosen policy.

    This is what licenses centring EV before it enters the float32 tensor. In
    float64 the invariance is essentially exact; the test guards the algebra,
    not the arithmetic.
    """
    from hh_npe.simulator.twoasset import _age_step

    spec = dataclasses.replace(TINY, dtype=np.float64)
    rng = np.random.default_rng(0)
    nX, nZ, nS = 12, 5, 3
    A = rng.normal(0, 1e4, (nX, nX))
    B = rng.normal(0, 1e4, (nZ, nZ))
    Z = np.linspace(0.0, 1e5, nZ)
    EV = rng.normal(0, 1.0, (nX, nZ, nS))

    args = (A, B, Z, 0.6, 0.97, 2.0, 1.0, 2.3, 1.05, spec)
    nx0, nz0, *_ = _age_step(EV, *args)
    nx1, nz1, *_ = _age_step(EV + 137.0, *args)
    np.testing.assert_array_equal(nx0, nx1)
    np.testing.assert_array_equal(nz0, nz1)


def test_float32_and_float64_broadly_agree(sol64):
    """Centred float32 must track float64 -- a canary for precision regressions.

    Without EV centring this test fails badly: uncentred float32 inflated
    simulated credit-card borrowing by roughly 50%.
    """
    sol32 = solve(*BENCHMARK, dataclasses.replace(TINY, dtype=np.float32))
    p32 = simulate(sol32, n_households=2000, seed=0)
    p64 = simulate(sol64, n_households=2000, seed=0)
    frac32 = (p32["liquid_assets"] < 0).mean()
    frac64 = (p64["liquid_assets"] < 0).mean()
    assert abs(frac32 - frac64) < 0.05, f"float32={frac32:.3f} float64={frac64:.3f}"


def test_nearest_index_rounds_ties_up():
    """Exact midpoints must resolve upward, as MATLAB's 'nearest' does.

    Rounding ties down leaks liquid wealth wherever the asset grid is coarser
    than the income lattice, biasing the simulation toward borrowing.
    """
    from hh_npe.simulator.twoasset import _nearest_index

    g = np.array([0.0, 2000.0, 4000.0])
    np.testing.assert_array_equal(_nearest_index(g, np.array([1000.0, 3000.0])), [1, 2])
    np.testing.assert_array_equal(
        _nearest_index(g, np.array([100.0, 1900.0, 3900.0])), [0, 1, 2]
    )


def test_snapping_does_not_lose_wealth_on_average():
    """Snapping cash-on-hand to the grid must not be systematically downward."""
    from hh_npe.simulator.twoasset import ModelSpec, _nearest_index

    spec = ModelSpec()
    age = grids.ages()
    X, _ = grids.liquid_grid(age, spec.xjump, spec.xmax, spec.x_cells_per_step)
    q = np.arange(0.0, 400001.0, spec.xjump)  # values the income lattice can take
    err = X[_nearest_index(X, q)] - q
    assert err.mean() >= 0.0, f"snapping loses ${-err.mean():,.0f} on average"
