"""The methodology file's noise model (RESULTS 67, ``scripts/psid_noise.py``).

It must be what the file says: multiplicative on the magnitude (zeros and
signs kept, age untouched); income with mean reversion toward a fixed
reference and AR(1) errors across waves; consumption iid normal; wealth
unit-variance t clipped at |u| <= 3 with sigma^2 = s V / 2; every parameter
drawn per household from the file's priors; reproducible in the seed. And it
must reach every window through ``optionA.add_obs_noise`` and a run's config.
Synthetic arrays only.
"""

import dataclasses
import json

import numpy as np
import pytest
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE as FEATS
from scripts import optionA as oa
from scripts import psid_noise as pn

AGE, INC, CONS = FEATS.index("age"), FEATS.index("income"), FEATS.index("consumption")
LIQ, ILL = FEATS.index("liquid_assets"), FEATS.index("illiquid_assets")
REF = (10.7,) * 7


def _windows(n=20000, waves=7, seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty((n, waves, len(FEATS)))
    x[..., INC] = np.exp(10.7 + rng.normal(0, 0.7, (n, 1)) + rng.normal(0, 0.1, (n, waves)))
    x[..., CONS] = 30_000 * np.exp(rng.normal(0, 0.1, (n, waves)))
    x[..., LIQ] = 5_000 * np.exp(rng.normal(0, 0.5, (n, waves))) \
        * np.where(rng.random((n, 1)) < 0.3, -1, 1) * (rng.random((n, 1)) > 0.1)
    x[..., ILL] = 80_000 * np.exp(rng.normal(0, 0.3, (n, waves))) * (rng.random((n, 1)) > 0.3)
    x[..., AGE] = rng.integers(24, 52, (n, 1)) + 2 * np.arange(waves)
    return x


def _model(**fixed):
    """The file's model, with any prior pinned to one value."""
    pri = dict(pn.PRIORS) | {k: (v, v) for k, v in fixed.items()}
    return pn.FileNoise(var_liquid=1.0, var_illiquid=0.5, income_ref=REF,
                        priors=tuple(sorted(pri.items())))


def test_magnitudes_only_zeros_signs_and_age_kept():
    x = torch.from_numpy(_windows(3000))
    y = pn.add_psid_noise(x, _model(), seed=1)
    assert torch.equal(y[..., AGE], x[..., AGE])
    for j in (INC, CONS, LIQ, ILL):
        assert torch.equal(y[..., j][x[..., j] == 0], x[..., j][x[..., j] == 0])
        assert torch.equal(torch.sign(y[..., j]), torch.sign(x[..., j]))
        assert not torch.equal(y[..., j], x[..., j])


def test_the_same_seed_is_the_same_draw():
    x = torch.from_numpy(_windows(500)).float()
    a, b = pn.add_psid_noise(x, _model(), 7), pn.add_psid_noise(x, _model(), 7)
    assert torch.equal(a, b) and a.dtype == x.dtype
    assert not torch.equal(a, pn.add_psid_noise(x, _model(), 8))


def test_parameters_are_drawn_per_household_inside_the_priors():
    p = pn.sample_noise_params(np.random.default_rng(0), 50000, _model())
    for name in ("sigma_v", "b", "rho", "sigma_c"):
        lo, hi = pn.PRIORS[name]
        assert p[name].min() >= lo and p[name].max() <= hi
        assert p[name].std() > 0.25 * (hi - lo)                  # spread, not one value
    for f, v in (("liquid_assets", 1.0), ("illiquid_assets", 0.5)):
        s = p[f"share_{f}"]
        assert s.min() >= 0.3 and s.max() <= 0.7
        assert np.allclose(p[f"sigma_{f}"] ** 2, s * v / 2)
        assert p[f"nu_{f}"].min() >= 3 and p[f"nu_{f}"].max() <= 8


def test_income_error_reverts_to_the_reference_and_is_ar1_across_waves():
    """With the priors pinned: u = b (log y - ref) + sigma_v z, corr(z_t, z_t-1)
    = rho, and Var(u) = b^2 Var(log y) + sigma_v^2 (the file's 2.5)."""
    x = _windows(40000)
    e = pn.errors(x, _model(sigma_v=0.11, b=-0.10, rho=0.15), np.random.default_rng(3))
    u = e["u"]["income"]
    dev = np.log(x[..., INC]) - np.array(REF)
    v = u - (-0.10) * dev
    assert v.std() == pytest.approx(0.11, rel=0.02)
    assert np.corrcoef(v[:, 1:].ravel(), v[:, :-1].ravel())[0, 1] == pytest.approx(0.15, abs=0.01)
    assert np.corrcoef(v[:, 2:].ravel(), v[:, :-2].ravel())[0, 1] == pytest.approx(0.0225,
                                                                                  abs=0.01)
    slope = np.polyfit(dev.ravel(), u.ravel(), 1)[0]
    assert slope == pytest.approx(-0.10, abs=0.005)
    assert u.var() == pytest.approx(0.01 * dev.var() + 0.11 ** 2, rel=0.03)


def test_consumption_error_is_iid_normal():
    e = pn.errors(_windows(40000), _model(sigma_c=0.18), np.random.default_rng(4))
    u = e["u"]["consumption"]
    assert u.std() == pytest.approx(0.18, rel=0.01)
    assert abs(np.corrcoef(u[:, 1:].ravel(), u[:, :-1].ravel())[0, 1]) < 0.01


def test_wealth_error_is_unit_variance_t_scaled_and_clipped():
    rng = np.random.default_rng(5)
    t = pn.unit_t(rng, np.full(200000, 5.0), (200000, 1))
    assert t.var() == pytest.approx(1.0, rel=0.03)
    kurt = ((t - t.mean()) ** 4).mean() / t.var() ** 2
    assert kurt > 5                                               # t5: 9 in theory; normal 3
    # a huge V: sigma ~ 4, so the clip binds
    m = dataclasses.replace(_model(share=0.5), var_liquid=64.0)
    u = pn.errors(_windows(5000), m, np.random.default_rng(6))["u"]["liquid_assets"]
    assert np.abs(u).max() == pytest.approx(3.0) and (np.abs(u) == 3.0).mean() > 0.2
    u = pn.errors(_windows(40000), _model(share=0.5, nu=4.0),
                  np.random.default_rng(7))["u"]["illiquid_assets"]
    assert u.std() == pytest.approx(np.sqrt(0.5 * 0.5 / 2), rel=0.05)


def test_calibrate_asset_sigma_is_the_file_s_rule():
    rng = np.random.default_rng(8)
    a = np.exp(np.cumsum(rng.normal(0, 0.4, (500, 7)), 1))
    a[:100, 3] = 0                                                # pairs touching it drop
    c = pn.calibrate_asset_sigma(a, observed=True, share=0.5)
    d = np.diff(np.log(np.where(a > 0, a, 1.0)), axis=1)[(a[:, 1:] > 0) & (a[:, :-1] > 0)]
    q25, q75 = np.percentile(d, [25, 75])
    assert c["n_pairs"] == 500 * 6 - 200
    assert c["var"] == pytest.approx(((q75 - q25) / 1.349) ** 2)
    assert c["var"] == pytest.approx(0.16, rel=0.08)
    assert c["sigma2"] == pytest.approx(0.5 * c["var"] / 2)
    s = pn.calibrate_asset_sigma(a, observed=False, share=0.5)["sigma2"]
    assert s == pytest.approx(0.5 / 0.5 * c["var"] / 2)
    with pytest.raises(ValueError, match="need 50"):
        pn.calibrate_asset_sigma(a[:5])


def test_income_reference_is_the_mean_log_income_by_position():
    x = torch.from_numpy(_windows(1000))
    x[0, 2, INC] = 0.0                                            # skipped, not log(0)
    ref = pn.income_reference(x)
    y = x[..., INC].numpy()
    want = [np.log(y[y[:, t] > 0, t]).mean() for t in range(7)]
    assert ref == pytest.approx(want)


def test_the_model_round_trips_through_a_record_and_a_file(tmp_path):
    m = _model()
    assert pn.FileNoise.from_record(m.record()) == m
    f = tmp_path / "noise.json"
    f.write_text(json.dumps({**m.record(), "calibration": {"anything": 1}}))
    loaded = pn.FileNoise.load(f)
    assert loaded.sha256 and dataclasses.replace(loaded, sha256="") == m
    with pytest.raises(ValueError):
        pn.FileNoise.from_record({"model": "other"})


def test_optionA_dispatches_and_a_config_restores_it():
    x = torch.from_numpy(_windows(300)).float()
    m = _model()
    assert torch.equal(oa.add_obs_noise(x, m, seed=5), pn.add_psid_noise(x, m, seed=5))
    spec = oa.spec_of({"noise_model": {**m.record(), "path": "x.json"},
                       "beta_transform": "logit", "rgamma_transform": "logit",
                       "beta_range": [0.1, 1.0], "rgamma_range": [1.021, 1.075]})
    assert spec.obs_noise == m and "methodology file" in spec.describe()
    assert oa.noise_record(spec.obs_noise)["model"] == "psid_file"
    assert oa.noise_record((("consumption", 0.0, 0.15),)) == {"consumption": [0.0, 0.15]}
    # a config with lognormal ranges is read as before
    old = oa.spec_of({"obs_noise": {"consumption": [0.0, 0.15]}})
    assert old.obs_noise == (("consumption", 0.0, 0.15),)


def test_the_income_reference_must_match_the_window_length():
    with pytest.raises(ValueError, match="waves"):
        pn.add_psid_noise(torch.from_numpy(_windows(10, waves=5)), _model(), 0)
