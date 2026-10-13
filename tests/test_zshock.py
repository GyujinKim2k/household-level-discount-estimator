"""The illiquid valuation shock (``ModelSpec.z_return_sd``, RESULTS 44).

Zero must be the model as before, since every existing dataset and resume
check rests on it. Above zero, the shock must add risk and no drift (so
R_gamma keeps its meaning), households must face exactly the risk they plan
for, and it must leave the income shocks alone, so a draw with the shock is
paired with the same draw without it.
"""

import dataclasses

import numpy as np
import pytest

from hh_npe.simulator import grids, twoasset
from hh_npe.simulator import laibson_calibration as cal
from hh_npe.simulator.twoasset import (
    ModelSpec,
    _z_shock_draw,
    simulate,
    simulate_from,
    solve,
    z_shock_matrix,
)

TINY = ModelSpec(xjump=20000.0, x_cells_per_step=4, zjump=200000.0, z_cells_per_step=3)
SHOCK = dataclasses.replace(TINY, z_return_sd=0.2)
BENCHMARK = cal.BENCHMARK_PREFS


@pytest.mark.parametrize("sd", [0.05, 0.2, 0.5])
def test_matrix_is_a_mean_preserving_spread(sd):
    Z = grids.illiquid_grid()
    M = z_shock_matrix(Z, sd)
    assert (M >= 0).all()
    np.testing.assert_allclose(M.sum(1), 1.0, atol=1e-12)
    assert M[0, 0] == pytest.approx(1.0, abs=1e-12) and not M[0, 1:].any()   # zero stays zero
    # At the top, gains are capped, so the shock drifts down there only.
    assert (M @ Z)[-1] < Z[-1]
    # Below the top, no drift: E[Z_next | Z'] = Z'.
    x, _ = np.polynomial.hermite_e.hermegauss(twoasset.Z_SHOCK_NODES)
    inside = Z * np.exp(sd * x.max()) * 1.01 < Z[-1]
    np.testing.assert_allclose((M @ Z)[inside], Z[inside], rtol=1e-12, atol=1e-6)
    # And it is a spread: some mass leaves the diagonal wherever Z > 0.
    assert (np.diag(M)[1:-1] < 1).all()


def test_households_face_the_risk_they_plan_for():
    """The simulation's transitions are draws from the solve's matrix."""
    Z = grids.illiquid_grid()
    M = z_shock_matrix(Z, 0.2)
    rng = np.random.default_rng(0)
    a = 40
    got = _z_shock_draw(np.cumsum(M, axis=1), np.full(200_000, a), rng)
    freq = np.bincount(got, minlength=len(Z)) / len(got)
    np.testing.assert_allclose(freq, M[a], atol=0.005)


def test_zero_never_builds_the_shock(monkeypatch):
    """sd = 0 takes none of the new code: solve and simulate are as before."""
    def boom(*a, **k):
        raise AssertionError("z_shock_matrix called at sd = 0")
    monkeypatch.setattr(twoasset, "z_shock_matrix", boom)
    sol = solve(*BENCHMARK, TINY)
    simulate(sol, n_households=8, seed=0)
    simulate_from(sol, np.array([20]), np.array([0.0]), np.array([1e5]),
                  np.array([4e4]), horizon=3, n_sims=2, seed=0)


@pytest.fixture(scope="module")
def cpu_pair():
    return solve(*BENCHMARK, TINY), solve(*BENCHMARK, SHOCK)


def test_shock_changes_the_solve_and_the_paths_not_the_income(cpu_pair):
    s0, s1 = cpu_pair
    assert (s0.next_z != s1.next_z).any()
    p0 = simulate(s0, n_households=400, seed=3)
    p1 = simulate(s1, n_households=400, seed=3)
    for k in ("income", "income_state"):
        np.testing.assert_array_equal(p0[k], p1[k])
    falls = [(np.diff(p["illiquid_assets"], axis=1) < 0).mean() for p in (p0, p1)]
    assert falls[1] > falls[0] + 0.02


def test_simulate_from_draws_the_shock(cpu_pair):
    s0, s1 = cpu_pair
    kw = dict(t0=np.array([20, 25]), liquid0=np.array([5e3, 0.0]),
              illiquid0=np.array([4e5, 8e5]), income0=np.array([5e4, 7e4]),
              horizon=6, n_sims=200, seed=1)
    f0, f1 = simulate_from(s0, **kw), simulate_from(s1, **kw)
    np.testing.assert_array_equal(f0["income"], f1["income"])
    assert not np.array_equal(f0["illiquid_assets"], f1["illiquid_assets"])
    # Column 0 is the observed start, untouched by the shock.
    np.testing.assert_array_equal(f0["illiquid_assets"][..., 0], f1["illiquid_assets"][..., 0])


torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


@cuda
def test_gpu_reproduces_the_cpu_with_the_shock(cpu_pair):
    from hh_npe.simulator.twoasset_gpu import solve_batch

    _s0, s1 = cpu_pair
    g = solve_batch(np.array([BENCHMARK]), SHOCK, theta_batch=1)[0]
    ok = s1.solvable & s1.feasible[:, :, None, None]
    assert (s1.next_z[ok] == g.next_z[ok]).mean() > 0.999
    pc = simulate(s1, n_households=64, seed=0)
    pg = simulate(g, n_households=64, seed=0)
    for key in pc:
        np.testing.assert_array_equal(pc[key], pg[key], err_msg=key)


@cuda
def test_dispatch_zero_is_as_before_and_a_shock_moves_illiquid_only():
    from hh_npe.simulator.dispatch import simulate_batch_twoasset_gpu

    thetas = np.array([[0.7, 0.98, 2.0], [0.6, 0.97, 4.0]])
    kw = dict(seed_base=5, start_age=30, n_waves=5, wave_years=2, grid="coarse",
              theta_batch=4, chunk=8, return_panels=True, n_households=20)
    _x0, _a0, p0 = simulate_batch_twoasset_gpu(thetas, **kw)
    _x1, _a1, p1 = simulate_batch_twoasset_gpu(thetas, z_sd=np.zeros(2), **kw)
    for k in p0:
        np.testing.assert_array_equal(p0[k], p1[k])
    _x2, _a2, p2 = simulate_batch_twoasset_gpu(thetas, z_sd=np.full(2, 0.2), **kw)
    np.testing.assert_array_equal(p0["income"], p2["income"])
    assert not np.array_equal(p0["illiquid_assets"], p2["illiquid_assets"])
