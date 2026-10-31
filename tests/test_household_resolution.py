"""Household resolution (RESULTS 60).

Synthetic SBC-like sets: draws uniform on the box, 16 households per draw
sharing the draw's theta and card, each with a posterior mean that is a
shrunk function of theta plus its own estimation noise.
"""

import numpy as np
import pytest

from scripts import household_resolution as hr

LO = np.array([0.30, 0.85, 0.50, 1.021])
HI = np.array([1.00, 0.999, 5.00, 1.075])
MID = (LO + HI) / 2
SLOPE = np.array([0.3, 0.5, 0.5, 0.5])
NOISE = np.array([0.05, 0.008, 0.35, 0.005])
SD = np.array([0.17, 0.03, 0.85, 0.011])


def _f(theta, cross=0.0, card=None, card_shift=0.0):
    """Posterior means' centre: each parameter shrunk toward the box's middle;
    beta's also moves with rho (``cross`` per unit of rho), and with card."""
    m = MID + SLOPE * (theta - MID)
    m[..., 0] += cross * (theta[..., 2] - MID[2])
    if card is not None:
        m[..., 0] += card_shift * card
    return m


def _sim(n=2048, seed=0, cross=0.0, card_shift=0.0):
    rng = np.random.default_rng(seed)
    truth = LO + (HI - LO) * rng.random((n, 4))
    card = rng.integers(0, 2, n)
    centre = _f(truth, cross, card, card_shift)
    mean = centre[:, None] + NOISE * rng.standard_normal((n, hr.PER_DRAW, 4))
    sd = np.broadcast_to(SD, mean.shape).copy()
    return {"truth": truth, "card": card, "mean": mean, "sd": sd,
            "lo": mean - 1.645 * sd, "hi": mean + 1.645 * sd}


def _households(theta, seed, cross=0.0):
    """'PSID' households: one posterior mean each, from the same map."""
    rng = np.random.default_rng(seed)
    mean = _f(theta, cross) + NOISE * rng.standard_normal(theta.shape)
    sd = np.broadcast_to(SD, mean.shape).copy()
    return {"mean": mean, "sd": sd, "lo": mean - 1.645 * sd, "hi": mean + 1.645 * sd}


def test_others_spread_recovers_what_the_other_parameters_add():
    cross = 0.04
    est, boot = hr.others_spread(_sim(cross=cross), n_boot=200)
    want = cross ** 2 * (HI[2] - LO[2]) ** 2 / 12
    assert est[0] == pytest.approx(want, rel=0.2)
    lo, hi = np.percentile(boot[:, 0], [5, 95])
    assert lo < est[0] < hi and hi - lo < 0.5 * want
    assert est[3] < 0.1 * NOISE[3] ** 2          # R_gamma's mean depends on itself only


def test_card_shift_is_found():
    sim = _sim(card_shift=0.03)
    w = np.full(len(sim["truth"]), 1 / len(sim["truth"]))
    shift = hr.card_shift(sim, w)
    assert shift[0] == pytest.approx(0.03, abs=0.005)
    assert abs(shift[2]) < 0.03


def test_the_identity_adds_for_a_bayes_posterior():
    """A conjugate normal population: Var(truth) = Var(means) + E[posterior
    var], and RESULTS 1's difference is negative although the truth varies."""
    rng = np.random.default_rng(1)
    n = 4096
    mu, tau, s = np.array([0.65, 0.95, 2.75, 1.048]), np.array([0.12, 0.01, 0.8, 0.008]), \
        np.array([0.15, 0.012, 0.9, 0.01])
    k = tau ** 2 / (tau ** 2 + s ** 2)
    truth = mu + tau * rng.standard_normal((n, 4))
    x = truth[:, None] + s * rng.standard_normal((n, hr.PER_DRAW, 4))
    mean = mu + k * (x - mu)
    sd = np.broadcast_to(np.sqrt(k) * s, mean.shape).copy()
    sim = {"truth": truth, "card": np.zeros(n, int), "mean": mean, "sd": sd,
           "lo": mean - 1.645 * sd, "hi": mean + 1.645 * sd}
    r = hr.simulated(sim)
    for name in hr.NAMES:
        assert r[name]["var_means_plus_post"] == pytest.approx(r[name]["var_truth"], rel=0.05)
        assert r[name]["var_means_minus_post"] < 0 < r[name]["var_truth"]


def test_the_shared_theta_null_finds_no_spread_when_households_share_theta():
    sim = _sim()
    theta = np.tile([0.6, 0.95, 2.5, 1.05], (400, 1))
    r = hr.null_vs_psid(sim, _households(theta, seed=2), seed=0)
    # four 90% intervals miss zero together often enough; across seeds the
    # share stays within +-0.23, where real spread reads above 0.4 (next test)
    for name in hr.NAMES:
        assert abs(r[name]["share_not_noise"]) < 0.3


def test_the_shared_theta_null_finds_real_spread_and_the_second_null_attributes_it():
    """Households differ in beta only: beta's spread is real under both nulls
    (the second adds what rho does to beta's mean over the whole box, here
    a quarter of beta's own signal). Then they differ in rho only, which
    beta's mean follows: beta's spread is real under the first null and not
    under the second."""
    rng = np.random.default_rng(3)
    base = np.tile([0.6, 0.95, 2.5, 1.05], (400, 1))
    in_beta = base.copy()
    in_beta[:, 0] = 0.35 + 0.6 * rng.random(400)
    r = hr.null_vs_psid(_sim(cross=0.02), _households(in_beta, 4, cross=0.02), seed=0)
    assert r["beta"]["share_not_noise_90"][0] > 0.3
    assert r["beta"]["share_own_90"][0] > 0.0
    in_rho = base.copy()
    in_rho[:, 2] = 0.6 + 4.3 * rng.random(400)
    r = hr.null_vs_psid(_sim(cross=0.04), _households(in_rho, 5, cross=0.04), seed=0)
    assert r["beta"]["share_not_noise_90"][0] > 0.2
    assert r["beta"]["share_own_90"][0] < 0.0


def test_pairs_count_only_separated_households():
    mean = np.array([[0.40, 0.95, 2.0, 1.04], [0.90, 0.95, 2.1, 1.04], [0.45, 0.95, 2.05, 1.04]])
    sd = np.full((3, 4), 0.05)
    r = hr.pairs(mean, sd, mean - 1.645 * sd, mean + 1.645 * sd, truth=mean)
    assert r["beta"]["intervals_apart"] == pytest.approx(2 / 3)
    assert r["beta"]["ordered_p90"] == pytest.approx(2 / 3)
    assert r["beta"]["right_when_ordered"] == 1.0
    assert r["delta"]["ordered_p90"] == 0.0
