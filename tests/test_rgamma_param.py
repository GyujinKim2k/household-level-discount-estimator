"""R_gamma as a fourth estimated parameter (RESULTS.md 32).

Three things must hold, all silent if wrong:

* The column layout. beta 0, delta 1, rho 2 are hard-coded across the codebase;
  R_gamma must be appended, never inserted.
* A draw's R_gamma must reach its solve. Draws are grouped by R_gamma value
  before solving; a draw solved at the default 1.05 would look perfectly normal.
* The pilot must be the start of the full run. The pilot's 4,096 draws are only
  reusable if generating 32,768 draws reproduces them exactly, R_gamma and card
  type included.
"""

import argparse

import numpy as np
import pytest

from hh_npe.npe.prior import PHASE3, PHASE3_RGAMMA, EdgeMixture, log1m_box


def test_rgamma_is_appended_last():
    assert PHASE3_RGAMMA.names == ("beta", "delta", "crra", "R_gamma")
    assert PHASE3_RGAMMA.names[:3] == PHASE3.names
    np.testing.assert_array_equal(PHASE3_RGAMMA.low[:3], PHASE3.low)
    assert (PHASE3_RGAMMA.low[3], PHASE3_RGAMMA.high[3]) == (1.025, 1.075)


def test_log1m_box_keeps_rgamma():
    b = log1m_box(PHASE3_RGAMMA)
    assert b.names == PHASE3_RGAMMA.names
    assert b.low[3] == 1.025 and b.high[3] == 1.075


def test_edge_mixture_refuses_the_four_parameter_box():
    with pytest.raises(ValueError, match="per block"):
        EdgeMixture(box=PHASE3_RGAMMA)


def test_edge_mixture_weight_ignores_the_rgamma_column():
    th = EdgeMixture().sample(64, seed=0)
    th4 = np.column_stack([th, np.full(64, 1.03)])
    np.testing.assert_array_equal(EdgeMixture().log_weight(th4),
                                  EdgeMixture().log_weight(th))


def _gen(n):
    from scripts.generate_dataset import _per_block_rgamma
    args = argparse.Namespace(theta_batch=16, block=512, n_samples=n, seed=0,
                              rgamma_range=[1.025, 1.075])
    th = EdgeMixture().sample(n, seed=0)
    card = np.random.default_rng(1).integers(0, 2, size=n)
    return _per_block_rgamma(th, card, args)


def test_rgamma_and_card_are_constant_within_each_block_of_16():
    th, card = _gen(4096)
    rg = th[:, 3].reshape(-1, 16)
    assert (rg == rg[:, :1]).all()
    assert (card.reshape(-1, 16) == card.reshape(-1, 16)[:, :1]).all()
    assert len(np.unique(rg[:, 0])) == 256
    assert ((th[:, 3] >= 1.025) & (th[:, 3] <= 1.075)).all()


def test_pilot_is_exactly_the_start_of_the_full_run():
    """The load-bearing one: the pilot's shards are reused by the full run."""
    th_p, card_p = _gen(4096)
    th_f, card_f = _gen(32768)
    np.testing.assert_array_equal(th_p, th_f[:4096])
    np.testing.assert_array_equal(card_p, card_f[:4096])


def test_rgamma_marginal_is_uniform():
    th, _ = _gen(32768)
    u = (np.unique(th[:, 3]) - 1.025) / 0.05
    assert abs(u.mean() - 0.5) < 0.01
    assert np.histogram(u, bins=8, range=(0, 1))[0].min() > 0.9 * len(u) / 8


torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_each_draw_is_solved_at_its_own_rgamma():
    from hh_npe.simulator.dispatch import simulate_batch_twoasset_gpu
    th = np.array([[0.8, 0.98, 2.0]] * 4)
    kw = dict(seed_base=3, start_age=30, n_waves=5, wave_years=2, grid="coarse",
              theta_batch=4, chunk=8, return_panels=True, n_households=30)
    _x, _a, base = simulate_batch_twoasset_gpu(th, **kw)
    _x, _a, same = simulate_batch_twoasset_gpu(th, r_gamma=np.full(4, 1.05), **kw)
    for k in base:                    # 1.05 is the default: bit-identical
        np.testing.assert_array_equal(base[k], same[k])
    _x, _a, mixed = simulate_batch_twoasset_gpu(
        th, r_gamma=np.array([1.03, 1.03, 1.07, 1.07]), **kw)
    ill = mixed["illiquid_assets"].reshape(4, 30, -1)
    assert np.array_equal(ill[0], ill[0])
    # A higher illiquid return must raise illiquid holdings.
    assert ill[2:].mean() > ill[:2].mean()
