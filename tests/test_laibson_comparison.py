"""Laibson et al.'s point against PSID couples (RESULTS 61): the indirect
inference arithmetic, and their standard errors from their own matrices."""

import numpy as np
import pytest

from scripts import laibson_comparison as lc


def test_u_round_trips():
    th = np.array([[0.53, 0.989, 1.94, 1.05], [0.3, 0.85, 5.0, 1.021]])
    assert np.allclose(lc.from_u(lc.to_u(th)), th)


@pytest.mark.skipif(not (lc.PKG / "input/comphs/est_secondstage.mat").exists(),
                    reason="replication package not present (git-ignored)")
def test_their_standard_errors_reproduce_from_their_matrices():
    r = lc.laibson()            # raises if any of the four variants does not reproduce
    assert r["theta"] == pytest.approx([0.53046, 0.98906, 1.93553], abs=1e-5)
    assert r["se"]["reported"] == pytest.approx([0.11429, 0.00511, 0.43498], rel=1e-3)
    assert r["se"]["sampling only"] == pytest.approx([0.050286, 0.002043, 0.204467], rel=1e-3)
    assert r["n_obs"] == 5710
    c = r["cov"]["reported"]
    assert c[0, 1] / np.sqrt(c[0, 0] * c[1, 1]) < -0.9          # beta and delta trade off


def test_stat_mixes_card_types():
    mean = np.array([[1.0, 0, 0, 0], [3.0, 0, 0, 0], [10.0, 0, 0, 0]])
    card = np.array([0, 0, 1])
    assert lc.stat(mean, card, 0.85)[0] == pytest.approx(0.15 * 2 + 0.85 * 10)


def _linear(u, a, B):
    return a + u @ B.T


def test_central_differences_and_newton_recover_a_linear_map():
    rng = np.random.default_rng(0)
    a, B = rng.normal(size=4), rng.normal(size=(4, 4)) + 3 * np.eye(4)
    u0 = np.array([0.5, -4.5, 2.0, 1.05])
    plus = np.array([_linear(u0 + lc.STEP[k] * np.eye(4)[k], a, B) for k in range(4)])
    minus = np.array([_linear(u0 - lc.STEP[k] * np.eye(4)[k], a, B) for k in range(4)])
    J = lc.jacobian(plus, minus, lc.STEP)
    assert np.allclose(J, B)
    u_star = np.array([0.4, -4.8, 2.3, 1.055])
    assert np.allclose(lc.newton(u0, J, _linear(u_star, a, B), _linear(u0, a, B)), u_star)


def test_bootstrap_matches_the_delta_method():
    """Simulated households: posterior means = B u + noise, the same noise
    at every point (common random numbers). PSID: 400 households. The
    bootstrap sd of the representative theta matches J^-1 (Sigma_psid / n +
    Sigma_sim / m) J^-T."""
    rng = np.random.default_rng(1)
    B = np.diag([0.4, -0.01, 0.6, 0.5]) + 0.05
    u0 = np.array([0.53, -4.5, 1.94, 1.05])
    m, n, noise = 4000, 400, np.array([0.06, 0.002, 0.4, 0.006])
    card = np.repeat([0, 1], m // 2)
    e = noise * rng.standard_normal((m, 4))

    def post(u):
        return {"mean": u @ B.T + e, "card": card}

    fd = {"plus": [post(u0 + lc.STEP[k] * np.eye(4)[k]) for k in range(4)],
          "minus": [post(u0 - lc.STEP[k] * np.eye(4)[k]) for k in range(4)]}
    psid = u0 @ B.T + noise * 1.5 * rng.standard_normal((n, 4))
    boot = lc.boot_theta(u0, fd, post(u0), psid, 0.5, lc.STEP, rng, n_boot=400)
    Jinv = np.linalg.inv(B)
    want = np.sqrt(np.diag(Jinv @ np.diag((1.5 * noise) ** 2 / n + noise ** 2 / m) @ Jinv.T))
    assert boot.std(0) == pytest.approx(want, rel=0.15)


def test_a_fit_that_misses_psid_is_not_drawn_as_one():
    """RESULTS 62: a last simulated point beyond FIT_TOL PSID standard errors
    gets no region or bars; an edge alone is flagged; a close fit is plain."""
    edge = [False] * 4
    assert lc.fit_note({"last_simulated_misfit_over_psid_se": [0.3, -0.8, 0.85, 0],
                        "at_box_edge": edge}) == ""
    assert lc.fit_note({"last_simulated_misfit_over_psid_se": [-1.8, 0, 0, 0],
                        "at_box_edge": [True] + edge[1:]}) == "β at the box floor"
    assert lc.fit_note({"last_simulated_misfit_over_psid_se": [-13.6, 0.5, -7.8, -8.8],
                        "at_box_edge": edge}).startswith("no fit (misses PSID by 14 se")


# RESULTS 63.7: the Jacobian recomputed at each point


def _fd(f, step, lo, hi):
    def fd_at(v, _r):
        c = lc.stencil_centre(v, step, lo, hi)
        e = np.eye(len(v))
        return {"plus": [f(c + step[k] * e[k]) for k in range(len(v))],
                "minus": [f(c - step[k] * e[k]) for k in range(len(v))]}
    return fd_at


def test_a_refreshed_jacobian_converges_where_the_fixed_one_stalls():
    # the second statistic flattens away from the start, as delta's does near 1
    def f(v):
        return np.array([v[0] + 0.3 * v[1], 5 * np.tanh(v[1])])
    target, se = np.array([1.0, 4.9]), np.full(2, 0.01)
    lo, hi, step = np.full(2, -10.0), np.full(2, 10.0), np.full(2, 1e-4)
    v0 = np.zeros(2)
    J0 = lc.jacobian(np.array(_fd(f, step, lo, hi)(v0, 0)["plus"]),
                     np.array(_fd(f, step, lo, hi)(v0, 0)["minus"]), step)
    v = v0
    for _ in range(3):                                  # RESULTS 61's three fixed steps
        v = lc.newton(v, J0, target, f(v))
    assert np.abs((target - f(v)) / se).max() > 3
    v, post, _fd_end, J, path = lc.refine(v, f(v), target, se, lo, hi, step,
                                          lambda w, _t: f(w), _fd(f, step, lo, hi),
                                          lambda p: p, n_iter=8)
    assert np.abs((target - post) / se).max() < 0.25
    assert np.allclose(post, f(v))
    # the Jacobian returned is the one at the point returned
    fd = _fd(f, step, lo, hi)(v, 0)
    assert np.allclose(J, lc.jacobian(np.array(fd["plus"]), np.array(fd["minus"]), step))
    assert all(q["accepted"] for q in path)


def test_a_step_that_overshoots_is_halved():
    # Newton on arctan from 2 overshoots to -3.5, further from 0 than it started
    def f(v):
        return np.arctan(v)
    lo, hi, step = np.array([-10.0]), np.array([10.0]), np.array([1e-5])
    _v, _p, _fd_end, _J, path = lc.refine(np.array([2.0]), f(np.array([2.0])), np.zeros(1),
                                          np.ones(1), lo, hi, step, lambda w, _t: f(w),
                                          _fd(f, step, lo, hi), lambda p: p, n_iter=1)
    assert [(q["step_fraction"], q["accepted"]) for q in path] == [(1.0, False), (0.5, True)]


def test_a_component_at_its_bound_is_held_and_the_rest_solve_least_squares():
    J = np.array([[1.0, 0.5], [0.2, 1.0]])
    v, lo, hi = np.array([1.0, 0.0]), np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    resid, se = np.array([2.0, 0.5]), np.array([1.0, 0.5])
    d = lc.bounded_step(v, J, resid, se, lo, hi)          # the full step pushes v[0] past 1
    assert np.linalg.solve(J, resid)[0] > 0 and d[0] == 0
    w = 1 / se
    assert d[1] == pytest.approx(np.linalg.lstsq((J[:, 1] * w)[:, None], resid * w,
                                                 rcond=None)[0][0])
    assert np.allclose(lc.bounded_step(np.zeros(2), J, resid, se, lo, hi),
                       np.linalg.solve(J, resid))         # inside the box: plain Newton


def test_only_the_held_component_is_flagged_at_the_edge():
    J = np.array([[1.0, 0.5], [0.2, 1.0]])
    v, lo, hi = np.array([1.0, 0.0]), np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    resid, se = np.array([2.0, 0.5]), np.array([1.0, 0.5])
    v_hat = np.clip(v + lc.bounded_step(v, J, resid, se, lo, hi), lo, hi)
    assert (lc.newton(v, J, resid, 0) != v_hat).all()     # the old flag: both components
    assert lc.at_edge(v_hat, lo, hi).tolist() == [True, False]
    assert lc.at_edge(np.clip(lc.newton(v, J, resid, 0), lo, hi), lo, hi).tolist() == [True, False]


def test_the_stencil_stays_inside_the_box():
    lo, hi, step = np.array([0.1, -11.5]), np.array([0.999, -1.9]), np.array([0.03, 0.15])
    c = lc.stencil_centre(np.array([0.11, -6.0]), step, lo, hi)
    assert c[0] == pytest.approx(0.13) and c[1] == -6.0
    assert (c - step >= lo).all() and (c + step <= hi).all()
