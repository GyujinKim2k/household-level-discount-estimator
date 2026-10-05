"""Utility centred at mean income (RESULTS.md 40), and the penalty scale.

Centred at $1, CRRA utility is a constant ``-1 / (1 - rho)`` plus a variable
part that, at rho >~ 4.2 and typical consumption, falls below float64's
resolution of that constant: choices tie and break toward the most borrowing
and the least illiquid wealth. Centring at mean income per adult changes
utility by a per-period constant only -- no choice can move in real
arithmetic -- and keeps the variable part resolvable.
"""

import dataclasses

import numpy as np
import pytest

from hh_npe.simulator import grids
from hh_npe.simulator.twoasset import COARSE, ModelSpec, _crra, solve

C = np.array([5e3, 1e4, 2e4, 2.1e4, 5e4, 1e5])
REF = 15_000.0


def test_centring_is_a_constant_shift():
    """At rho = 2, where both forms resolve, they differ by a constant only."""
    d = _crra(C, 2.0, 2.0, REF) - _crra(C, 2.0, 2.0)
    np.testing.assert_allclose(d, d[0], rtol=0, atol=1e-12)


def test_centred_form_keeps_the_log_limit():
    for eps in (1e-5, -1e-5):
        np.testing.assert_allclose(_crra(C, 2.0, 1.0 + eps, REF),
                                   _crra(C, 2.0, 1.0, REF), rtol=1e-4, atol=2e-3)


def test_centred_form_resolves_high_rho():
    """$20,000 against $21,000 per adult at rho = 4.6: the textbook form cannot
    tell them apart in float64; the centred form can, with the right sign."""
    legacy = _crra(C, 1.0, 4.6)
    centred = _crra(C, 1.0, 4.6, REF)
    assert legacy[3] - legacy[2] < 1e-15            # at most an ulp or so
    assert centred[3] > centred[2]
    np.testing.assert_allclose(
        (centred[3] - centred[2]) / (2.1e4 ** -3.6 - 2e4 ** -3.6) * -3.6, 1.0, rtol=1e-6)


def test_defaults():
    assert ModelSpec().crra_centred is True
    assert ModelSpec().liqpen_scale == 1.0


def test_penalty_scale_one_is_the_schedule():
    """scale 1 is bit-identical to the unscaled solve; another scale is not."""
    spec = ModelSpec(xjump=20000.0, x_cells_per_step=4, zjump=200000.0, z_cells_per_step=3)
    a = solve(0.8, 0.98, 2.0, spec)
    b = solve(0.8, 0.98, 2.0, dataclasses.replace(spec, liqpen_scale=1.0))
    c = solve(0.8, 0.98, 2.0, dataclasses.replace(spec, liqpen_scale=0.5))
    np.testing.assert_array_equal(a.next_z, b.next_z)
    assert (a.next_z != c.next_z).any()
    assert grids.liquidation_penalty(np.array([45.0]))[0] == pytest.approx(0.3112, abs=1e-4)


torch = pytest.importorskip("torch")
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


@gpu
def test_gpu_matches_cpu_centred():
    from hh_npe.simulator.twoasset_gpu import solve_batch

    tiny = ModelSpec(xjump=20000.0, x_cells_per_step=4, zjump=200000.0, z_cells_per_step=3)
    cpu = solve(0.5305, 0.9891, 1.9355, tiny)
    g = solve_batch(np.array([[0.5305, 0.9891, 1.9355]]), tiny, theta_batch=1)[0]
    ok = cpu.solvable & cpu.feasible[:, :, None, None]
    np.testing.assert_array_equal(cpu.next_x[ok], g.next_x[ok])
    np.testing.assert_array_equal(cpu.next_z[ok], g.next_z[ok])


@gpu
@pytest.mark.slow
def test_gpu_high_rho_is_batch_invariant_when_centred():
    """The symptom RESULTS 40 found: at rho = 4.6 the legacy solve depends on
    what else is in its batch. Centred, it must not."""
    from hh_npe.simulator.twoasset_gpu import solve_batch

    th = np.array([[0.77, 0.988, 4.6], [0.77, 0.988, 2.0], [0.95, 0.95, 3.0]])
    alone = solve_batch(th[:1], COARSE, theta_batch=16, chunk=16)[0]
    batch = solve_batch(th, COARSE, theta_batch=16, chunk=16)[0]
    np.testing.assert_array_equal(alone.next_x, batch.next_x)
    np.testing.assert_array_equal(alone.next_z, batch.next_z)
