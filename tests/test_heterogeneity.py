"""Per-household initial wealth and observation noise.

Both are inputs to the regeneration gates (RESULTS.md section 9), and both fail
silently if wrong: a broken `initial_wealth` still returns a valid panel, and
broken noise still returns a valid tensor. The gate numbers would just be wrong.
"""

import numpy as np
import pytest
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from hh_npe.simulator.twoasset import COARSE, simulate, solve
from scripts.compare_windows import add_obs_noise


@pytest.fixture(scope="module")
def sol():
    return solve(0.85, 0.99, 2.0, spec=COARSE)


def test_initial_wealth_gives_households_different_starts(sol):
    """The default seeds every household identically -- the whole point of the
    override is that it stops doing that."""
    w = np.array([[0.0, 0.0], [0.05, 1.0], [0.15, 2.5], [-0.10, 0.0]])
    p = simulate(sol, n_households=4, seed=0, initial_wealth=w)
    assert len(np.unique(p["illiquid_assets"][:, 0])) > 1
    d = simulate(sol, n_households=4, seed=0)
    assert len(np.unique(d["illiquid_assets"][:, 0])) == 1


def test_initial_wealth_is_ordered_in_the_illiquid_argument(sol):
    """Monotone in, monotone out -- catches a transposed (liquid, illiquid)."""
    w = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 3.0]])
    z = simulate(sol, n_households=3, seed=0, initial_wealth=w)["illiquid_assets"][:, 0]
    assert z[0] <= z[1] <= z[2] and z[0] < z[2]


def test_initial_wealth_allows_starting_in_credit_card_debt(sol):
    """Clamped to the age-20 credit limit, NOT to zero: somehs's own SCF median
    liquid ratio is -0.037, so a non-negative clamp would erase a real group.

    `grids.credit_limit` returns a POSITIVE magnitude, so the floor is -xmin[0];
    clamping at +xmin[0] instead put every drawn household at or above +4,000.
    The ratio here is sized to the COARSE grid's 4,000 spacing -- -0.037 itself
    snaps to zero there and would pass against the bug.
    """
    w = np.array([[-0.14, 0.0], [0.14, 0.0]])
    liq = simulate(sol, n_households=2, seed=0,
                   initial_wealth=w)["liquid_assets"][:, 0]
    assert liq[0] < 0 < liq[1]


def test_initial_wealth_shape_is_checked(sol):
    with pytest.raises(ValueError):
        simulate(sol, n_households=3, seed=0, initial_wealth=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        simulate(sol, n_households=2, seed=0, initial_wealth=np.zeros((2, 3)))


def test_obs_noise_leaves_age_exact():
    """Age is exact in PSID. Perturbing it would model an error that is not
    there, and it is the one feature the network can anchor windows on."""
    x = torch.full((64, 7, 5), 1000.0)
    y = add_obs_noise(x, 0.3, FEATURES_TWOASSET_AGE, seed=1)
    i = FEATURES_TWOASSET_AGE.index("age")
    assert torch.equal(x[..., i], y[..., i])
    assert not torch.equal(x[..., 0], y[..., 0])


def test_obs_noise_is_median_preserving_at_the_stated_sigma():
    x = torch.full((4000, 7, 5), 1000.0)
    r = (add_obs_noise(x, 0.3, FEATURES_TWOASSET_AGE, seed=2)[..., 0] / 1000.0)
    assert r.log().std().item() == pytest.approx(0.3, abs=0.02)
    assert r.mean().item() == pytest.approx(1.0, abs=0.02)


def test_obs_noise_zero_sigma_is_a_no_op():
    """The default. If this ever perturbs, every past run silently changes."""
    x = torch.rand(16, 7, 5)
    assert torch.equal(x, add_obs_noise(x, 0.0, FEATURES_TWOASSET_AGE, seed=3))
