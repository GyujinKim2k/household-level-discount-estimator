"""The delta reparameterisation, and the property it exists for.

`log(1 - delta)` is adopted because it makes `delta >= 1` unrepresentable rather
than merely unlikely (RESULTS.md 13.1, 19.1). Two things could break silently:

* **The wrong column.** delta is index 1 with beta estimated and index 0
  without. Transforming beta instead of delta would train a perfectly good flow
  on the wrong axis and report plausible numbers.
* **Bounds not swapping.** The map is decreasing, so `delta_high` becomes the
  *low* end in log space. Getting that backwards gives an inverted box, which
  `PriorBox` rejects -- but only if the swap is attempted at all.
"""

import numpy as np
import pytest
import torch

from hh_npe.npe.prior import (
    LOG1M_EPS,
    PHASE3,
    PriorBox,
    delta_index,
    from_log1m,
    log1m_box,
    sample_sobol,
    to_log1m,
)


def test_delta_is_column_one_with_beta_and_zero_without():
    assert delta_index(PHASE3) == 1
    assert delta_index(PriorBox()) == 0


def test_round_trip_is_the_identity():
    th = sample_sobol(512, PHASE3, seed=0)
    back = from_log1m(to_log1m(th))
    np.testing.assert_allclose(back, th, rtol=0, atol=1e-9)


def test_torch_and_numpy_agree():
    th = sample_sobol(64, PHASE3, seed=1)
    np.testing.assert_allclose(to_log1m(torch.from_numpy(th)).numpy(),
                               to_log1m(th), rtol=1e-6, atol=1e-9)


def test_beta_and_crra_are_untouched():
    th = sample_sobol(128, PHASE3, seed=2)
    t = to_log1m(th)
    np.testing.assert_array_equal(t[:, 0], th[:, 0])
    np.testing.assert_array_equal(t[:, 2], th[:, 2])


def test_delta_stays_below_one_for_every_in_box_draw():
    """The whole point, but only inside the box -- see the next test."""
    b = log1m_box(PHASE3)
    wild = np.zeros((4, 3))
    wild[:, 1] = [b.delta_low, -10.0, -5.0, b.delta_high]
    d = from_log1m(wild)[:, 1]
    assert (d < 1.0).all()
    assert (d >= PHASE3.delta_low).all()


def test_underflow_reaches_exactly_one_so_the_box_test_must_come_first():
    """`truncation is structurally impossible` holds in real arithmetic, NOT in
    floating point: `exp(-800)` underflows to 0 and `1 - 0` is exactly 1.0.

    Consequence for every consumer: reject against the TRANSFORMED box, whose
    low end is `log(1e-6) = -13.8`, and invert only what survives. Inverting
    first and then testing `delta <= 1` lets an underflowed draw through and
    reports delta = 1.0 -- reinstating the truncation the transform was adopted
    to remove, in the one place it matters (PSID, RESULTS.md 13.1).
    """
    assert from_log1m(np.array([[0.9, -800.0, 2.0]]))[0, 1] == 1.0
    # and the box is what stops it
    assert -800.0 < log1m_box(PHASE3).delta_low


def test_the_box_swaps_ends_and_stays_valid():
    b = log1m_box(PHASE3)
    assert b.delta_low == pytest.approx(np.log(LOG1M_EPS))
    assert b.delta_high == pytest.approx(np.log(1.0 - PHASE3.delta_low))
    assert b.delta_low < b.delta_high          # PriorBox would have raised
    assert (b.beta_low, b.beta_high) == (PHASE3.beta_low, PHASE3.beta_high)


def test_no_sobol_draw_is_clipped_by_the_epsilon():
    """If EPS bit, the transform would silently pile draws at one value."""
    th = sample_sobol(65536, PHASE3, seed=0)
    assert (1.0 - th[:, 1]).min() > LOG1M_EPS


def test_every_transformed_draw_lands_inside_the_transformed_box():
    th = sample_sobol(4096, PHASE3, seed=3)
    t, b = to_log1m(th), log1m_box(PHASE3)
    assert ((t >= b.low) & (t <= b.high)).all()
