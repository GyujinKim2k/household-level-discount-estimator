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
