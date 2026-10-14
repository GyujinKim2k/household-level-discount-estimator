"""PSID's imputed values treated as missing (RESULTS 50).

Simulated windows must take masks with PSID couples' patterns, independent of
theta and of the noise; the network's input must carry a masked value only
as a 0/1 missing flag, never as a number; an unmasked run's input must be
exactly as before; and the masking rule must be monotone in its threshold.
"""

import numpy as np
import pytest
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE as FEATS
from scripts import optionA as oa
from scripts import psid_accuracy_codes as pac
from tests.test_obs_noise import _windows

MASKED_J = [FEATS.index(f) for f in oa.MASKABLE]


def _patterns(n=200, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.random((n, 7, len(oa.MASKABLE))) < np.array([0.02, 0.06, 0.05])
    m[:, 2, 0] |= rng.random(n) < 0.9            # a wave where one feature is mostly masked
    return m


def test_mask_marks_only_the_maskable_features_with_psid_rates():
    x = torch.from_numpy(_windows(20000)).double()
    pat = _patterns()
    y = oa.add_imputation_mask(x, pat, seed=5)
    nan = torch.isnan(y)
    assert not nan[..., FEATS.index("income")].any() and not nan[..., FEATS.index("age")].any()
    keep = ~nan
    assert torch.equal(y[keep], x[keep])                       # nothing else changes
    rate = nan[..., MASKED_J].double().mean((0,)).numpy()      # (waves, features)
    assert np.allclose(rate, pat.mean(0), atol=0.02)
    assert torch.equal(torch.isnan(oa.add_imputation_mask(x, pat, seed=5)), nan)


def test_masked_households_take_whole_psid_patterns():
    x = torch.from_numpy(_windows(3000)).double()
    pat = _patterns(50)
    y = oa.add_imputation_mask(x, pat, seed=2)
    got = torch.isnan(y[..., MASKED_J]).numpy()
    seen = {p.tobytes() for p in pat}
    assert all(g.tobytes() in seen for g in got)


def test_transform_masked_flags_and_zeroes_missing_values():
    x = torch.from_numpy(_windows(500)).float()
    y = oa.add_imputation_mask(x, _patterns(), seed=1)
    plain, f0 = oa.transform(x)
    out, f1 = oa.transform(y, masked=True)
    assert f1 == tuple(f0) + tuple(f"missing_{f}" for f in oa.MASKABLE)
    assert not torch.isnan(out).any()
    miss = out[..., len(f0):].bool()
    assert torch.equal(miss, torch.isnan(y[..., MASKED_J]))
    for k, f in enumerate(oa.MASKABLE):
        j = f0.index(f)
        assert (out[..., j][miss[..., k]] == 0).all()
        assert torch.equal(out[..., j][~miss[..., k]], plain[..., j][~miss[..., k]])
    other = [i for i, f in enumerate(f0) if f not in oa.MASKABLE]
    assert torch.equal(out[..., other], plain[..., other])


def test_unmasked_transform_is_unchanged_and_rejects_stray_missing_values():
    x = torch.from_numpy(_windows(100)).float()
    a, fa = oa.transform(x)
    b, fb = oa.transform(x, masked=False)
    assert torch.equal(a, b) and fa == fb
    bad = x.clone()
    bad[0, 0, FEATS.index("income")] = torch.nan
    with pytest.raises(ValueError):
        oa.transform(bad, masked=True)


def test_mask_stream_leaves_the_noise_alone():
    x = torch.from_numpy(_windows(2000)).double()
    noise = (("consumption", 0.0, 0.15), ("illiquid_assets", 0.175, 0.455))
    n = oa.add_obs_noise(x, noise, seed=oa.NOISE_SEED + 3)
    m = oa.add_imputation_mask(n, _patterns(), seed=oa.MASK_SEED + 3)
    keep = ~torch.isnan(m)
    assert torch.equal(m[keep], n[keep])


def test_spec_round_trips_the_mask_setting():
    cfg = {"rgamma_range": [1.021, 1.075], "beta_transform": "logit",
           "rgamma_transform": "logit", "obs_noise": {"consumption": [0, 0.15]},
           "imputation_mask": {"patterns": "p.pt", "tau": 0.25, "sha256": "ab"}}
    s = oa.spec_of(cfg)
    assert s.masked and s.imputation_mask == ("p.pt", 0.25, "ab")
    assert "tau 0.25" in s.describe()
    assert not oa.spec_of({k: v for k, v in cfg.items() if k != "imputation_mask"}).masked


def test_psid_values_are_set_missing_where_masked():
    x = _windows(30)
    miss = _patterns(30)
    y = oa.set_missing(x, miss)
    for k, j in enumerate(MASKED_J):
        assert np.isnan(y[..., j][miss[..., k]]).all()
        assert np.array_equal(y[..., j][~miss[..., k]], x[..., j][~miss[..., k]])
    assert oa.set_missing(x, None) is x


def test_masking_rule_is_monotone_in_tau():
    rng = np.random.default_rng(0)
    n, w = 300, 7
    c = {"imputed": {f: rng.random((n, w)) < 0.2 for f in ("liquid", "illiquid", "consumption")},
         "amount": {f: rng.random((n, w)) * 10 + 1 for f in ("liquid", "illiquid", "consumption")},
         "equity": rng.random((n, w)) * 5, "pension": rng.random((n, w)),
         "equity_imputed": rng.random((n, w)) < 0.05}
    c["amount_imp"] = {f: np.where(c["imputed"][f], rng.random((n, w)) * c["amount"][f], 0.0)
                       for f in c["imputed"]}
    prev = None
    for tau in (0.0, 0.1, 0.25, 0.5, 1.0):
        m = pac.masks(c, tau)
        if prev is not None:
            assert all((m[f] <= prev[f]).all() for f in m)
        prev = m
    any_, _ = pac.imputed_share(c, "illiquid")
    assert np.array_equal(pac.masks(c, 0.0)["illiquid"], any_)
