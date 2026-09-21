"""The opt-in transitory income disruption (RESULTS.md §16).

The load-bearing property is that it is **off by default and bit-identical when
off**. Every result in this project predates it, and the faithful port's
table-3 fidelity check (§11.6) depends on the income process being exactly
Laibson et al.'s. A disruption that leaked in at `disrupt_p = 0` would
invalidate all of it silently -- the outputs would still look plausible.

The feature itself exists because the unmodified income process is symmetric
and thin-tailed (skew 0.00, excess kurtosis -0.6) where PSID's is left-skewed
and fat-tailed (-1.83, +6.88), so a third of households have at least one wave
the model cannot generate (§12.4).
"""

import numpy as np
import pytest

from hh_npe.simulator import grids

Y0 = float(np.log(50_000.0))


def test_off_by_default_is_bit_identical():
    """The guarantee every pre-existing result rests on."""
    a_p, a_l = grids.discretize_transitory(Y0)
    b_p, b_l = grids.discretize_transitory(Y0, disrupt_p=0.0)
    assert np.array_equal(a_p, b_p)
    assert np.array_equal(a_l, b_l)


@pytest.mark.parametrize("dp", [0.01, 0.068, 0.10, 0.25])
def test_probabilities_still_sum_to_one(dp):
    p, _ = grids.discretize_transitory(Y0, disrupt_p=dp)
    assert p.sum() == pytest.approx(1.0)
    assert (p >= 0).all()


@pytest.mark.parametrize("dp", [0.068, 0.10])
def test_disruption_mass_is_what_was_asked_for(dp):
    """The added point must carry exactly `disrupt_p`, or the calibration
    measured from PSID is not the calibration being simulated."""
    p, l = grids.discretize_transitory(Y0, disrupt_p=dp, disrupt_mult=0.27)
    target = np.exp(Y0) * 0.27
    i = int(np.argmin(np.abs(l - target)))
    assert p[i] == pytest.approx(dp, abs=1e-12)


def test_support_stays_on_the_income_lattice():
    """Realized income must land on a representable cash-on-hand shift; an
    off-lattice point would be silently snapped elsewhere downstream."""
    xjump = 1000.0
    _, l = grids.discretize_transitory(Y0, disrupt_p=0.1, disrupt_mult=0.27,
                                       xjump=xjump)
    np.testing.assert_allclose(l / xjump, np.round(l / xjump), atol=1e-9)


def test_disruption_lowers_the_left_tail_without_touching_the_right():
    """The whole point: PSID's upside already matches, only the downside is
    missing (§12.5). A symmetric widening would be the wrong fix."""
    p0, l0 = grids.discretize_transitory(Y0)
    p1, l1 = grids.discretize_transitory(Y0, disrupt_p=0.10, disrupt_mult=0.20)
    assert l1.min() < l0.min()
    assert l1.max() == l0.max()
    # mean of the top decile is essentially unchanged
    top = lambda p, l: (l[l >= np.percentile(l, 90)]).mean()
    assert top(p1, l1) == pytest.approx(top(p0, l0), rel=1e-9)


def test_monotone_in_probability():
    """More disruption must mean more mass at the bottom, or the parameter is
    not doing what its name says."""
    below = []
    for dp in (0.0, 0.05, 0.15):
        p, l = grids.discretize_transitory(Y0, disrupt_p=dp, disrupt_mult=0.2)
        below.append(p[l < 0.5 * np.exp(Y0)].sum())
    assert below[0] < below[1] < below[2]


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_invalid_probability_raises(bad):
    with pytest.raises(ValueError, match="disrupt_p"):
        grids.discretize_transitory(Y0, disrupt_p=bad)


def test_modelspec_defaults_to_off():
    """A ModelSpec built any of the usual ways must carry no disruption."""
    import dataclasses

    from hh_npe.simulator.twoasset import COARSE, GRIDS, MID, ModelSpec

    for spec in (ModelSpec(), COARSE, MID, *GRIDS.values()):
        assert spec.disrupt_p == 0.0
    on = dataclasses.replace(ModelSpec(), disrupt_p=0.07)
    assert on.disrupt_p == 0.07 and ModelSpec().disrupt_p == 0.0
