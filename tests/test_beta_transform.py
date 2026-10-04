"""beta's flow targets for the transform screen (RESULTS.md 37).

The failure modes are the same as for delta's ``log(1 - delta)``: a draw that
does not invert to the beta it came from, a training point outside the box the
flow is truncated to, or a held-out ``log q`` compared across arms in different
spaces (a missing Jacobian).
"""

import numpy as np
import pytest
import torch

from hh_npe.npe.prior import (
    BETA_TRANSFORMS,
    PHASE3_RGAMMA,
    SwitchedProposal,
    beta_flow_bounds,
    beta_from_flow,
    beta_log_jacobian,
    beta_to_flow,
)

BOX = PHASE3_RGAMMA


def _draws():
    th = SwitchedProposal().sample(4096, seed=0)
    return np.column_stack([th, np.full(len(th), 1.05)])


@pytest.mark.parametrize("kind", BETA_TRANSFORMS)
def test_round_trip_numpy_and_torch(kind):
    th = _draws()
    np.testing.assert_allclose(beta_from_flow(beta_to_flow(th, BOX, kind), BOX, kind),
                               th, rtol=0, atol=1e-12)
    tt = torch.from_numpy(th)
    back = beta_from_flow(beta_to_flow(tt, BOX, kind), BOX, kind)
    assert torch.allclose(back, tt, rtol=0, atol=1e-12)
    # Only beta's column moves.
    np.testing.assert_array_equal(beta_to_flow(th, BOX, kind)[:, 1:], th[:, 1:])


@pytest.mark.parametrize("kind", BETA_TRANSFORMS)
def test_training_draws_lie_inside_the_flow_box(kind):
    t = beta_to_flow(_draws(), BOX, kind)[:, 0]
    lo, hi = beta_flow_bounds(BOX, kind)
    assert lo < hi and (t >= lo).all() and (t <= hi).all()


def test_bounds_swap_for_the_decreasing_map():
    lo, hi = beta_flow_bounds(BOX, "log1m")
    assert beta_from_flow(np.array([[lo, 0, 0, 0]]), BOX, "log1m")[0, 0] > 0.999
    assert beta_from_flow(np.array([[hi, 0, 0, 0]]), BOX, "log1m")[0, 0] == pytest.approx(0.3)
    lo, hi = beta_flow_bounds(BOX, "logit")
    assert lo == -hi


@pytest.mark.parametrize("kind", BETA_TRANSFORMS)
def test_log_jacobian_matches_a_numerical_derivative(kind):
    b, h = np.array([0.31, 0.5, 0.8, 0.95, 0.999]), 1e-7
    pad = np.zeros((len(b), 3))
    up = beta_to_flow(np.c_[b + h, pad], BOX, kind)[:, 0]
    dn = beta_to_flow(np.c_[b - h, pad], BOX, kind)[:, 0]
    np.testing.assert_allclose(beta_log_jacobian(b, BOX, kind),
                               np.log(np.abs((up - dn) / (2 * h))), atol=1e-6)


def test_unknown_transform_is_refused():
    with pytest.raises(ValueError):
        beta_to_flow(_draws(), BOX, "probit")


def test_option_a_flow_box_changes_only_beta():
    from scripts.optionA import flow_box, from_flow, to_flow

    lin, lg = flow_box("linear"), flow_box("logit")
    np.testing.assert_array_equal(lin.low[1:], lg.low[1:])
    np.testing.assert_array_equal(lin.high[1:], lg.high[1:])
    th = _draws()
    np.testing.assert_allclose(from_flow(to_flow(th, "log1m"), "log1m"), th, atol=1e-12)
