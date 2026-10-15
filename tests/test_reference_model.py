"""The independent reference model (RESULTS 54), checked by routes other than
the ones it computes with: MATLAB's own budget assertion, a Monte Carlo of the
lifetime payoff against the value function, and the naive-agent value algebra.
"""

from __future__ import annotations

import numpy as np
import pytest

from hh_npe.simulator import reference as ref

pytestmark = pytest.mark.skipif(not ref.PACKAGE.exists(),
                                reason="needs the replication package")

#: A tiny grid: about 15 liquid and 4 illiquid points, every age. xjump is
#: also the income lattice's step, so it must stay fine enough for several
#: income points to survive the purge (their discretizeIID fails on one).
TINY = dict(xjump=5000.0, xmax=1e5, x_cells=4, xjump_max=20000.0,
            zjump=50000.0, zmax=2e5, z_cells=2, zjump_max=100000.0)


@pytest.fixture(scope="module")
def model():
    return ref.Model(ref.Config(**TINY))


@pytest.fixture(scope="module")
def sol_naive(model):
    return ref.solve(0.6, 0.97, 2.0, model)


@pytest.fixture(scope="module")
def sol_exp(model):
    return ref.solve(1.0, 0.95, 3.0, model)


def test_grids_and_income_process(model):
    assert all(np.all(np.diff(x) > 0) for x in model.X)
    assert all(x[0] == -xm for x, xm in zip(model.X, model.xmin, strict=True))
    np.testing.assert_allclose(model.P.sum(axis=1), 1.0)
    pi = model.stationary()
    np.testing.assert_allclose(pi @ model.P, pi, atol=1e-14)
    for t in (0, 35, 70):
        for s in range(3):
            p, y = model.transitory(t, s)
            assert abs(p.sum() - 1) < 1e-12
            # On the lattice only up to exp(log(.)) error -- RESULTS 54's H7.
            assert np.all(np.round(y) % model.cfg.xjump == 0)
            np.testing.assert_allclose(y, np.round(y), rtol=1e-14)


def test_budget_identity_matlab_form(model, sol_naive):
    """LifecycleSim_BackwardInduct.m:170-173, applied to every chosen policy:
    what is left after consumption and the illiquid move, grown at R (or
    R_CC if negative), plus the illiquid balance, is next period's wealth."""
    cfg = model.cfg
    for t in range(model.T):
        Xt = model.X[t]
        Xn = model.X[t + 1] if t < model.T - 1 else Xt
        ix, iz, c = sol_naive.ix[t], sol_naive.iz[t], sol_naive.c[t]
        cur_x = Xt[:, None, None]
        cur_z = model.Z[None, :, None]
        dz = model.Z[iz] - cur_z
        cur = cur_x - c - dz + model.pen[t] * np.minimum(dz, 0)
        nxt = Xn[ix] + model.Z[iz]
        check = cfg.R * cur + (cfg.R_CC - cfg.R) * np.minimum(cur, 0) + (cur_z + dz) - nxt
        feasible = c >= 0
        assert np.abs(check[feasible]).max() < 1e-5


def test_naive_value_is_the_believed_selfs(model, sol_naive):
    """With betahat = 1 the value handed back is max(u + delta EV): the
    exponential self's own optimum, not the present-biased choice."""
    beta, delta, rho = sol_naive.prefs
    t = 40
    Xt, Xn, EV = model.X[t], model.X[t + 1], sol_naive.EV[t]
    cost = np.where(Xn >= 0, Xn / model.cfg.R, Xn / model.cfg.R_CC)
    for i in range(len(Xt)):
        for a in range(len(model.Z)):
            dz = model.Z - model.Z[a]
            C = (Xt[i] - cost)[:, None] + (-dz + model.pen[t] * np.minimum(dz, 0))[None]
            u = model.utility(C + (model.cfg.R_gamma - 1) * model.Z[a], model.hhs[t], rho) \
                - 1e100 * (C < 0)
            for s in range(3):
                assert np.isclose(sol_naive.V[t][i, a, s], (u + delta * EV[:, :, s]).max(),
                                  rtol=1e-12, atol=1e-9)


def test_value_function_by_monte_carlo(model, sol_exp):
    """An exponential agent's V at age index t is the expected discounted
    utility of following the policy: each year either die (probability d,
    collect the bequest of the choice) or draw the next state and income."""
    _beta, delta, rho = sol_exp.prefs
    rng = np.random.default_rng(0)
    t0, n = model.T - 6, 40_000
    i0 = int(np.flatnonzero(model.X[t0] == 0)[0]) + 2
    a0, s0 = 1, 1
    ix = np.full(n, i0)
    iz = np.full(n, a0)
    s = np.full(n, s0)
    alive = np.ones(n, bool)
    total = np.zeros(n)
    disc = 1.0
    cdf_p = np.cumsum(model.P, axis=1)
    for t in range(t0, model.T):
        c = sol_exp.c[t][ix, iz, s]
        u = model.utility(c + (model.cfg.R_gamma - 1) * model.Z[iz], model.hhs[t], rho)
        total[alive] += disc * u[alive]
        jx, jz = sol_exp.ix[t][ix, iz, s], sol_exp.iz[t][ix, iz, s]
        if t == model.T - 1:
            Xn = model.X[t]
            beq = model.bequest(t, Xn, delta, rho)[jx, jz]
            total[alive] += disc * delta * beq[alive]
            break
        Xn = model.X[t + 1]
        beq = model.bequest(t + 1, Xn, delta, rho)[jx, jz]
        die = rng.random(n) < model.death[t]
        total[alive & die] += disc * delta * beq[alive & die]
        alive &= ~die
        s = (rng.random(n)[:, None] > cdf_p[s]).sum(axis=1).clip(0, 2)
        y = np.empty(n)
        for k in range(3):
            p, lv = model.transitory(t + 1, k)
            r = s == k
            y[r] = lv[np.searchsorted(np.cumsum(p), rng.random(r.sum())).clip(0, len(lv) - 1)]
        ix, iz = model.snap(Xn, Xn[jx] + y), jz
        disc *= delta
    v = sol_exp.V[t0][i0, a0, s0]
    se = total.std() / np.sqrt(n)
    assert abs(total.mean() - v) < 4 * se + 1e-9 * abs(v), (total.mean(), v, se)


def test_tie_rules():
    pay = np.array([[0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]])
    assert ref._argmax(pay, "x_first") == (0, 1)    # lowest x', then lowest z'
    assert ref._argmax(pay, "z_first") == (1, 0)    # lowest z', then lowest x'
    with pytest.raises(FloatingPointError):
        ref._argmax(np.array([[np.nan, 0.0]]), "x_first")


def test_centring_changes_no_choice(model):
    """In real arithmetic the centred utility differs by a constant per period,
    so on a grid where nothing is near a tie the policies agree exactly."""
    a = ref.solve(0.8, 0.96, 2.0, model)
    b = ref.solve(0.8, 0.96, 2.0, ref.Model(ref.Config(**TINY, centred=True), model.fs))
    same = np.mean([np.mean((a.ix[t] == b.ix[t]) & (a.iz[t] == b.iz[t]))
                    for t in range(model.T)])
    assert same > 0.999
