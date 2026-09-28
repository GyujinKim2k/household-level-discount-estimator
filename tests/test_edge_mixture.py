"""The edge-concentrated training proposal and the weights that undo it.

Training on EdgeMixture draws and then reweighting by p/p~ must give back the
uniform-prior posterior. The ways this goes wrong silently: a density that is
off by a constant or a Jacobian (the reweighted posterior is then biased with
nothing failing), weights that are unbounded (a few draws dominate), and draws
that do not follow the stated density.
"""

import numpy as np
import pytest

from hh_npe.npe.prior import PHASE3, EdgeMixture, sample_sobol

MIX = EdgeMixture()


def test_half_the_draws_are_concentrated_and_interleaved():
    th = MIX.sample(1024, seed=0)
    assert th.shape == (1024, 3)
    inside = MIX._concentrated_density(th) > 0
    # Uniform draws can also fall in the region, so >= half.
    assert inside.mean() >= 0.5
    # Every prefix of even length is exactly half from each component: the
    # concentrated rows are the odd ones.
    th2 = MIX.sample(8, seed=0)
    assert (MIX._concentrated_density(th2[1::2]) > 0).all()


def test_all_draws_lie_in_the_prior_box():
    th = MIX.sample(4096, seed=3)
    assert ((th >= PHASE3.low) & (th <= PHASE3.high)).all()
    assert (th[:, 1] < 1.0).all()


def test_weights_are_bounded_by_two():
    th = np.vstack([MIX.sample(4096, seed=1), sample_sobol(4096, PHASE3, seed=2)])
    w = np.exp(MIX.log_weight(th))
    assert w.max() <= 2.0 + 1e-12
    assert w.min() > 0.0
    # Outside the concentrated region the proposal is half the prior: weight 2.
    outside = MIX._concentrated_density(th) == 0
    np.testing.assert_allclose(w[outside], 2.0)


def test_concentrated_density_integrates_to_one():
    """Monte Carlo over the region's bounding box, in (beta, u = log(1-delta),
    rho) where the density is flat -- so the Jacobian is checked too."""
    rng = np.random.default_rng(0)
    n = 400_000
    b = rng.uniform(MIX.beta_lo, 1.0, n)
    u = rng.uniform(np.log(MIX.one_minus_delta_lo), np.log(MIX.one_minus_delta_hi), n)
    r = rng.uniform(MIX.crra_lo, 5.0, n)
    th = np.column_stack([b, 1 - np.exp(u), r])
    # E_uniform-in-(b,u,r)[f(theta) * |d delta / d u|] * volume = 1
    jac = np.exp(u)
    vol = (1 - MIX.beta_lo) * (np.log(MIX.one_minus_delta_hi / MIX.one_minus_delta_lo)) * (5 - MIX.crra_lo)
    est = (MIX._concentrated_density(th) * jac).mean() * vol
    assert est == pytest.approx(1.0, rel=0.01)


def test_reweighting_recovers_the_uniform_prior():
    """The load-bearing one: proposal draws, importance-weighted, must have the
    uniform prior's moments. Checked on delta, where the proposal differs most."""
    th = MIX.sample(2 ** 16, seed=5)
    w = np.exp(MIX.log_weight(th)); w /= w.sum()
    for j in range(3):
        lo, hi = PHASE3.low[j], PHASE3.high[j]
        assert (w * th[:, j]).sum() == pytest.approx((lo + hi) / 2, rel=2e-3)
    # and the share above 0.99, which the uniform prior puts at 6.67%
    assert (w * (th[:, 1] > 0.99)).sum() == pytest.approx(0.0667, abs=0.003)


def test_many_more_training_draws_near_delta_one():
    th = MIX.sample(2 ** 15, seed=0)
    uni = sample_sobol(2 ** 15, PHASE3, seed=0)
    assert (th[:, 1] > 0.998).mean() > 5 * (uni[:, 1] > 0.998).mean()
