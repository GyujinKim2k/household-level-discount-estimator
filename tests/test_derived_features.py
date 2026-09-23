"""Derived channels are appended without disturbing what was already there.

These are exact functions of columns the network already sees, so the risk is
not that they are wrong arithmetic -- it is that they are wired to the wrong
column, or that a later transform treats them as dollars. Both would be silent:
the shape is right either way and the loss would simply be a little worse.
"""

import pytest
import torch

from scripts.compare_windows import (
    DERIVED_FEATURES,
    derived_features,
    log_features,
)
from hh_npe.data.waves import FEATURES_TWOASSET_AGE as FEATS


@pytest.fixture
def x():
    # income, consumption, liquid, illiquid, age -- liquid deliberately signed.
    return torch.tensor([[[50_000.0, 30_000.0, -2_000.0, 10_000.0, 30.0],
                          [60_000.0, 35_000.0,  4_000.0, 20_000.0, 32.0]]])


def test_original_columns_are_untouched(x):
    out, _ = derived_features(x, FEATS)
    assert torch.equal(out[..., : len(FEATS)], x)


def test_names_grow_with_the_tensor(x):
    out, names = derived_features(x, FEATS)
    assert names == tuple(FEATS) + DERIVED_FEATURES
    assert out.shape[-1] == len(names)


def test_indicator_is_the_sign_of_liquid_not_its_size(x):
    out, names = derived_features(x, FEATS)
    col = out[..., names.index("card_debt")]
    assert col.tolist() == [[1.0, 0.0]]


def test_ratios_divide_by_this_household_s_income(x):
    out, names = derived_features(x, FEATS)
    liq = out[..., names.index("liq_to_inc")]
    illiq = out[..., names.index("illiq_to_inc")]
    assert liq[0, 0] == pytest.approx(-2_000.0 / 50_000.0)
    assert illiq[0, 1] == pytest.approx(20_000.0 / 60_000.0)


def test_zero_income_does_not_produce_an_infinite_ratio():
    z = torch.zeros(1, 1, len(FEATS))
    z[0, 0, 2] = -500.0
    out, _ = derived_features(z, FEATS)
    assert torch.isfinite(out).all()


def test_log_features_leaves_the_derived_channels_alone(x):
    """The load-bearing one. An indicator and two ratios are not dollars; a
    signed log1p applied to them would be silent and would destroy the
    indicator's only property, that it is exactly 0 or 1."""
    out, names = derived_features(x, FEATS)
    logged = log_features(out, names)
    d = slice(len(FEATS), None)
    assert torch.equal(logged[..., d], out[..., d])
    # and the dollar columns really were transformed
    assert not torch.equal(logged[..., :4], out[..., :4])


def test_log_features_still_skips_age_and_education(x):
    names = tuple(FEATS) + ("educ_comphs", "educ_somehs", "educ_compco")
    wide = torch.cat([x, torch.ones(1, 2, 3)], dim=-1)
    logged = log_features(wide, names)
    assert torch.equal(logged[..., 4:], wide[..., 4:])
