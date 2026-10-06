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


# --- RESULTS 32.2: the switched proposal (pilot kept, widened region after) ---

from hh_npe.npe.prior import EDGE_WIDENED, SwitchedProposal  # noqa: E402

SW = SwitchedProposal()


def test_switched_reproduces_the_pilot_draws_exactly():
    """The load-bearing one: the pilot's 4,096 stored draws must be what the
    switched sampler generates at those indices."""
    np.testing.assert_array_equal(SW.sample(32768, seed=0)[:4096],
                                  EdgeMixture().sample(4096, seed=0))
    np.testing.assert_array_equal(SW.sample(4096, seed=0),
                                  EdgeMixture().sample(4096, seed=0))


def test_switched_is_prefix_stable_beyond_the_switch():
    np.testing.assert_array_equal(SW.sample(49152, seed=0)[:32768],
                                  SW.sample(32768, seed=0))


def test_second_part_is_the_widened_mixture():
    th = SW.sample(8192, seed=0)[4096:]
    np.testing.assert_array_equal(th, EdgeMixture(**EDGE_WIDENED).sample(4096, seed=10))
    w = EdgeMixture(**EDGE_WIDENED)
    assert (w._concentrated_density(th[1::2]) > 0).all()


@pytest.mark.parametrize("n", [4096, 20000, 32768])
def test_switched_weights_are_bounded_and_recover_the_prior(n):
    th = SW.sample(n, seed=0)
    lw = SW.log_weight(th, n)
    assert np.exp(lw).max() <= 2.0 + 1e-12
    w = np.exp(lw); w /= w.sum()
    for j in range(3):
        lo, hi = PHASE3.low[j], PHASE3.high[j]
        assert (w * th[:, j]).sum() == pytest.approx((lo + hi) / 2, rel=4e-3)
    assert (w * (th[:, 1] > 0.99)).sum() == pytest.approx(0.0667, abs=0.006)


def test_switched_weight_with_only_pilot_draws_equals_the_first_mixture():
    th = SW.sample(2048, seed=0)
    np.testing.assert_allclose(SW.log_weight(th, 2048), EdgeMixture().log_weight(th))


# --- RESULTS 43.3: the option A2 run (uniform pilot kept, re-aimed region after) ---

from hh_npe.npe.prior import (  # noqa: E402
    EDGE_A2,
    SWITCHED_A2,
    UniformProposal,
    proposal_log_weight,
)

A2 = EdgeMixture(**EDGE_A2)


def test_a2_reproduces_the_uniform_pilot_draws_exactly():
    """The pilot was generated as the prefix of sample_sobol(32768)."""
    full_uniform = sample_sobol(32768, PHASE3, seed=0)
    np.testing.assert_array_equal(SWITCHED_A2.sample(32768, seed=0)[:4096], full_uniform[:4096])
    np.testing.assert_array_equal(SWITCHED_A2.sample(4096, seed=0), full_uniform[:4096])
    np.testing.assert_array_equal(SWITCHED_A2.sample(1024, seed=0), full_uniform[:1024])


def test_a2_is_prefix_stable_beyond_the_switch():
    np.testing.assert_array_equal(SWITCHED_A2.sample(49152, seed=0)[:32768],
                                  SWITCHED_A2.sample(32768, seed=0))


def test_a2_second_part_is_the_reaimed_mixture_with_uniform_delta():
    th = SWITCHED_A2.sample(32768, seed=0)[4096:]
    np.testing.assert_array_equal(th, A2.sample(28672, seed=10))
    con = th[1::2]
    assert (A2._concentrated_density(con) > 0).all()
    assert con[:, 0].min() >= 0.40 and con[:, 1].min() >= 0.95
    # rho unrestricted in the region; delta uniform on [0.95, 1], not log-spaced
    assert con[:, 2].min() < 0.6 and con[:, 2].max() > 4.9
    assert (con[:, 1] > 0.998).mean() == pytest.approx(0.04, abs=0.005)
    assert np.median(con[:, 1]) == pytest.approx(0.975, abs=0.002)


def test_uniform_delta_region_density_is_flat_and_integrates_to_one():
    f = 1.0 / ((1.0 - 0.40) * 0.05 * (5.0 - 0.5))
    th = np.array([[0.41, 0.951, 0.51], [0.99, 0.9999999, 4.99], [0.7, 0.975, 2.0]])
    np.testing.assert_allclose(A2._concentrated_density(th), f)
    out = np.array([[0.39, 0.97, 2.0], [0.7, 0.949, 2.0]])
    np.testing.assert_array_equal(A2._concentrated_density(out), 0.0)


def test_default_edge_mixture_is_unchanged_by_the_delta_option():
    """delta_log defaults to True: the 32.2 run's proposal and weights must not move."""
    assert EdgeMixture().delta_log and EdgeMixture(**EDGE_WIDENED).delta_log
    assert SW.first == EdgeMixture() and SW.second == EdgeMixture(**EDGE_WIDENED)


@pytest.mark.parametrize("n", [4096, 20000, 32768])
def test_a2_weights_are_bounded_and_recover_the_prior(n):
    th = SWITCHED_A2.sample(n, seed=0)
    lw = SWITCHED_A2.log_weight(th, n)
    assert np.exp(lw).max() <= 2.0 + 1e-12
    w = np.exp(lw); w /= w.sum()
    for j in range(3):
        lo, hi = PHASE3.low[j], PHASE3.high[j]
        assert (w * th[:, j]).sum() == pytest.approx((lo + hi) / 2, rel=4e-3)
    np.testing.assert_allclose(
        proposal_log_weight({"name": "a2_switched", "n_train": n}, np.column_stack(
            [th, np.full(n, 1.05)])), lw)


def test_a2_weights_with_only_pilot_draws_are_one():
    th = SWITCHED_A2.sample(2048, seed=0)
    np.testing.assert_array_equal(SWITCHED_A2.log_weight(th, 2048), 0.0)
    np.testing.assert_array_equal(UniformProposal().log_weight(th), 0.0)


def test_a2_full_run_density_inside_and_outside_the_region():
    """What RESULTS 43.2 quoted: 2.1x the uniform density inside, 0.56x outside."""
    th = np.array([[0.7, 0.98, 1.5], [0.35, 0.98, 1.5], [0.7, 0.90, 1.5]])
    q_over_p = np.exp(-SWITCHED_A2.log_weight(th, 32768))
    np.testing.assert_allclose(q_over_p, [4096 / 32768 + 28672 / 32768 * (0.5 + 0.5 / (0.6 / 0.7 * 0.05 / 0.15)),
                                          4096 / 32768 + 28672 / 32768 * 0.5,
                                          4096 / 32768 + 28672 / 32768 * 0.5])
    assert q_over_p[0] == pytest.approx(2.09, abs=0.01)
    assert q_over_p[1] == pytest.approx(0.5625)
