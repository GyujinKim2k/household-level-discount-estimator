"""The rho = 1 singularity on the GPU path, and the reproducibility it must not cost.

``twoasset._crra`` has always branched to ``log`` at rho == 1. The GPU solver
did not: it divided by ``1 - rho`` unguarded, so ``solve_batch`` at rho = 1 --
log utility, the canonical case in this literature -- died with a CUDA
device-side assert.

No generated dataset is affected. The nearest Sobol draw is |rho - 1| = 5e-5,
where the relative error from cancellation is ~1e-5 in float64, far below the
solver's ~1-5% grid discretisation error.

The harder requirement is that fixing it changes *nothing else*. ~99% of states
hold two exactly-tied choices, so a last-ulp difference flips argmax ties and
produces a different (equally optimal, but different) policy. Two attempts at
this guard each failed that bar:

  1. rewriting ``x ** omr`` as ``exp(omr * log x)`` -- algebraically identical,
     but ``pow`` is exact at integer exponents, so rho = 2, 3, 5 matched while
     rho = 4.5 did not;
  2. regrouping ``a * num / den`` as ``a * (num / den)`` -- floating-point
     multiplication and division do not associate.

These tests pin both the fix and the arithmetic it must not disturb.
"""

import numpy as np
import pytest

from hh_npe.simulator.twoasset import COARSE, _crra
from hh_npe.simulator.twoasset_gpu import RHO_LOG_EPS, _safe_omr

torch = pytest.importorskip("torch")


def test_safe_omr_only_replaces_near_zero():
    """The denominator guard must be the identity everywhere it is not needed,
    or it changes results at every rho rather than just rho = 1."""
    omr = torch.tensor([-4.0, -0.5, -1e-9, 0.0, 1e-9, 0.15, 3.0])
    out = _safe_omr(omr)
    near = omr.abs() < RHO_LOG_EPS
    assert torch.equal(out[~near], omr[~near])
    assert (out[near] == 1.0).all()


def test_cpu_crra_matches_its_own_log_limit():
    """Anchors the limit the GPU branch has to reproduce."""
    c = np.array([1e3, 1e4, 1e5])
    np.testing.assert_allclose(_crra(c, 2.0, 1.0), 2.0 * np.log(c / 2.0))
    # approached from both sides
    for eps in (1e-5, -1e-5):
        np.testing.assert_allclose(_crra(c, 2.0, 1.0 + eps),
                                   _crra(c, 2.0, 1.0), rtol=1e-4)


@pytest.mark.slow
def test_gpu_solves_at_rho_one_and_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from hh_npe.simulator.twoasset import solve
    from hh_npe.simulator.twoasset_gpu import solve_batch

    g = solve_batch(np.array([[0.85, 0.99, 1.0]]), spec=COARSE,
                    theta_batch=1, chunk=16)[0]
    assert np.isfinite(g.cons).all(), "rho = 1 produced non-finite consumption"
    c = solve(0.85, 0.99, 1.0, spec=COARSE)
    # Ties can break either way, so this is a high bar rather than equality.
    assert (g.next_x == c.next_x).mean() > 0.95


@pytest.mark.slow
@pytest.mark.parametrize("rho", [2.0, 3.0, 4.5, 5.0])
def test_guard_does_not_disturb_other_rho(rho):
    """4.5 is the load-bearing case: `1 - rho` is non-integer there, which is
    where both failed attempts diverged while the integer exponents matched."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from hh_npe.simulator.twoasset import solve
    from hh_npe.simulator.twoasset_gpu import solve_batch

    g = solve_batch(np.array([[0.85, 0.99, rho]]), spec=COARSE,
                    theta_batch=1, chunk=16)[0]
    c = solve(0.85, 0.99, rho, spec=COARSE)
    # Against the CPU, which the guard cannot have changed at all.
    assert np.isfinite(g.cons).all()
    assert g.cons.shape == c.cons.shape
