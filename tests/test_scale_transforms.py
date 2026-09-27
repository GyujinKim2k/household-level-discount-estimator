"""anchor_log and household_ratios: scale-free inputs that keep feature ratios.

The point of both (RESULTS 27) is to remove the absolute dollar scale while
KEEPING the ratios between features, which per-feature z-scoring destroys. So
the load-bearing property is scale invariance: multiplying every dollar value
of a household by a constant must leave the transformed input unchanged.
"""

import pytest
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE as FEATS
from scripts.compare_windows import RATIO_FEATURES, anchor_log, household_ratios


@pytest.fixture
def x():
    g = torch.Generator().manual_seed(0)
    inc = 30_000 + 50_000 * torch.rand(4, 7, generator=g)
    cons = inc * (0.6 + 0.3 * torch.rand(4, 7, generator=g))
    liq = 20_000 * (torch.rand(4, 7, generator=g) - 0.5)       # signed
    ill = 100_000 * torch.rand(4, 7, generator=g)
    age = torch.arange(30, 44, 2).float().expand(4, 7)
    return torch.stack([inc, cons, liq, ill, age], -1)


def _scale(x, k):
    y = x.clone(); y[..., :4] *= k
    return y


def test_anchor_log_is_invariant_to_the_dollar_scale(x):
    a, b = anchor_log(x, FEATS), anchor_log(_scale(x, 3.7), FEATS)
    torch.testing.assert_close(a, b)


def test_anchor_log_keeps_ratios_between_features(x):
    """A household consuming more of its income must look different even
    after anchoring -- the property per-feature z-scoring loses."""
    lean = x.clone(); lean[..., 1] *= 0.7
    assert not torch.allclose(anchor_log(x, FEATS)[..., 1],
                              anchor_log(lean, FEATS)[..., 1])


def test_anchor_log_handles_signed_liquid_and_leaves_age(x):
    out = anchor_log(x, FEATS)
    assert torch.isfinite(out).all()
    assert torch.equal(torch.sign(out[..., 2]), torch.sign(x[..., 2]))
    assert torch.equal(out[..., 4], x[..., 4])


def test_household_ratios_are_scale_invariant_and_constant_in_time(x):
    a, names = household_ratios(x, FEATS)
    b, _ = household_ratios(_scale(x, 0.25), FEATS)
    assert names == tuple(FEATS) + RATIO_FEATURES
    r = slice(len(FEATS), None)
    torch.testing.assert_close(a[..., r], b[..., r])
    assert torch.equal(a[..., r], a[:, :1, r].expand_as(a[..., r]))
    assert torch.equal(a[..., :len(FEATS)], x)


def test_ratios_are_skipped_by_per_sequence_normalisation(x):
    """Constant within a household: per-sequence z-scoring would divide by a
    zero spread. compare_windows skips any `ratio_` channel."""
    assert all(n.startswith("ratio_") for n in RATIO_FEATURES)


def test_zero_income_household_does_not_blow_up():
    z = torch.zeros(1, 7, 5); z[..., 2] = -500.0
    assert torch.isfinite(anchor_log(z, FEATS)).all()
    assert torch.isfinite(household_ratios(z, FEATS)[0]).all()
