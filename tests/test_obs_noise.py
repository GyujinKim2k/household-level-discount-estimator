"""Per-feature observation noise on the simulated windows (RESULTS 43.7).

The noise must reach the windows the network trains on and every simulated
window it is scored against, never PSID, never age, and it must be the
classical kind the calibration measures: iid per household and wave,
mean zero on the log scale, zeros and signs kept. The calibration must
recover a noise level it was given.
"""

import numpy as np
import pytest
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE as FEATS
from scripts import noise_calibration as nc
from scripts import optionA as oa

AGE = FEATS.index("age")
CONS = FEATS.index("consumption")
INC = FEATS.index("income")
LIQ = FEATS.index("liquid_assets")
ILL = FEATS.index("illiquid_assets")


def _windows(n=4000, waves=7, seed=0, sd=0.08):
    """Dollar windows with random-walk income, consumption and wealth, debt
    for some households, zero illiquid wealth for others, and exact ages."""
    rng = np.random.default_rng(seed)
    walk = lambda: np.exp(np.cumsum(rng.normal(0, sd, (n, waves)), 1))
    x = np.empty((n, waves, len(FEATS)))
    x[..., INC] = 50_000 * walk()
    x[..., CONS] = 40_000 * walk()
    x[..., LIQ] = 5_000 * walk() * np.where(rng.random((n, 1)) < 0.3, -1, 1)
    x[..., ILL] = 60_000 * walk() * (rng.random((n, 1)) > 0.2)
    x[..., AGE] = rng.integers(24, 46, (n, 1)) + 2 * np.arange(waves)
    return x


def test_no_noise_is_the_identity():
    x = torch.from_numpy(_windows(50)).float()
    assert oa.add_obs_noise(x, (), seed=1) is x
    assert torch.equal(oa.add_obs_noise(x, {"consumption": 0.0}, seed=1), x)


def test_noise_is_classical_per_feature_and_leaves_age_alone():
    x = torch.from_numpy(_windows()).double()
    sig = {"income": 0.0, "consumption": 0.12, "liquid_assets": 0.05, "illiquid_assets": 0.2}
    y = oa.add_obs_noise(x, sig, seed=3)
    assert torch.equal(y[..., AGE], x[..., AGE])
    assert torch.equal(y[..., INC], x[..., INC])                  # sigma 0: untouched
    for f, s in sig.items():
        j = FEATS.index(f)
        nz = x[..., j] != 0
        assert torch.equal(y[..., j][~nz], x[..., j][~nz])        # zero stays zero
        assert torch.equal(torch.sign(y[..., j]), torch.sign(x[..., j]))
        if s:
            e = torch.log(y[..., j][nz] / x[..., j][nz])
            assert abs(e.std().item() - s) < 0.03 * s
            assert abs(e.mean().item()) < 0.02 * s + 1e-3        # mean zero in logs
    # Deterministic in the seed, and a different seed is different noise.
    assert torch.equal(oa.add_obs_noise(x, sig, seed=3), y)
    assert not torch.equal(oa.add_obs_noise(x, sig, seed=4), y)


def test_each_household_draws_its_sigma_from_the_range():
    """Long rows, so each household's own sigma can be read off its noise."""
    x = torch.ones(3000, 400, len(FEATS), dtype=torch.float64)
    y = oa.add_obs_noise(x, {"consumption": (0.05, 0.25), "illiquid_assets": 0.1}, seed=0)
    s = torch.log(y[..., CONS]).std(1)
    assert s.min() > 0.05 - 0.02 and s.max() < 0.25 + 0.03
    assert s.mean().item() == pytest.approx(0.15, abs=0.01)     # uniform: midpoint
    assert np.histogram(s.numpy(), bins=4, range=(0.05, 0.25))[0].min() > 500
    f = torch.log(y[..., ILL]).std(1)                              # fixed sigma
    assert f.std() < 0.01 and f.mean().item() == pytest.approx(0.1, abs=0.005)
    # Same seed, same households' sigmas: the held-out windows are reproducible.
    assert torch.equal(oa.add_obs_noise(x, {"consumption": (0.05, 0.25)}, seed=0)[..., CONS],
                       y[..., CONS])


def test_noise_on_age_an_unknown_feature_or_a_bad_range_is_refused():
    x = torch.zeros(2, 7, len(FEATS))
    for bad in ("age", "wealth"):
        with pytest.raises(ValueError, match="no observation noise"):
            oa.add_obs_noise(x, {bad: 0.1}, seed=0)
        with pytest.raises(ValueError, match="no observation noise"):
            oa.Spec(obs_noise=((bad, 0.1, 0.1),))
    for r in ((0.2, 0.1), (-0.1, 0.1)):
        with pytest.raises(ValueError, match="low <= high"):
            oa.add_obs_noise(x, {"consumption": r}, seed=0)
        with pytest.raises(ValueError, match="low <= high"):
            oa.Spec(obs_noise=(("consumption", *r),))


def test_spec_records_and_reads_back_the_noise():
    noise = (("income", 0.0, 0.0), ("consumption", 0.0, 0.15), ("liquid_assets", 0.0, 0.08),
             ("illiquid_assets", 0.175, 0.455))
    spec = oa.Spec(box=oa.box_of({"rgamma_range": [1.021, 1.075]}), beta_transform="logit",
                   rgamma_transform="logit", obs_noise=noise)
    cfg = {"rgamma_range": [1.021, 1.075], "beta_transform": "logit",   # what train writes
           "rgamma_transform": "logit", "obs_noise": {f: [lo, hi] for f, lo, hi in noise}}
    assert oa.spec_of(cfg) == spec
    assert "consumption 0-0.15" in spec.describe() and "income 0," in spec.describe()
    # Runs before RESULTS 43.7 record none, and read back exactly as before.
    legacy = oa.spec_of({"beta_transform": "logit"})
    assert legacy.obs_noise == () and "noise" not in legacy.describe()


def test_windows_carry_the_noise_per_run_and_nothing_without_it(monkeypatch):
    base = {0: _windows(40, seed=1), 320: _windows(24, seed=2)}

    def fake_build(paths, theta, k, seed, **kw):
        off = 0 if "main" in str(paths[0]) else 320
        x = torch.from_numpy(base[off]).float()
        return torch.zeros(len(x), 4), x, torch.arange(len(x))

    monkeypatch.setattr(oa, "build_windowed", fake_build)
    files = [oa.Shard(oa.Path("main/s0.npz"), 0), oa.Shard(oa.Path("strip/s0.npz"), 320)]
    _t, x0, pid = oa.windows_raw(files, np.zeros((400, 4)), seed=999)
    assert torch.equal(x0, torch.cat([torch.from_numpy(base[o]).float() for o in (0, 320)]))
    assert torch.equal(pid[40:], torch.arange(24) + 320)
    sig = (("consumption", 0.0, 0.15), ("illiquid_assets", 0.175, 0.455))
    _t, x1, _p = oa.windows_raw(files, np.zeros((400, 4)), seed=999, noise=sig)
    want = torch.cat([oa.add_obs_noise(torch.from_numpy(base[o]).float(), sig,
                                       seed=oa.NOISE_SEED + 999 + o) for o in (0, 320)])
    assert torch.equal(x1, want)
    assert torch.equal(x1[..., AGE], x0[..., AGE]) and torch.equal(x1[..., INC], x0[..., INC])


def test_the_signature_sees_noise_and_not_random_walks():
    x = _windows(20_000)
    a0 = nc.signature(nc.input_scale(x, "consumption"))[0]
    assert abs(a0) < 0.002                                       # random walk: no signature
    y = oa.add_obs_noise(torch.from_numpy(x), {"consumption": 0.15}, seed=0).numpy()
    a1 = nc.signature(nc.input_scale(y, "consumption"))[0]
    assert a1 == pytest.approx(2 * 0.15 ** 2, rel=0.15)


@pytest.mark.parametrize("f, s0", [("consumption", 0.15), ("illiquid_assets", 0.25),
                                   ("liquid_assets", 0.10)])
def test_the_calibration_recovers_a_known_noise(f, s0):
    model = _windows(6000, seed=5)
    psid = oa.add_obs_noise(torch.from_numpy(_windows(6000, seed=6)), {f: s0}, seed=1).numpy()
    target = nc.signature(nc.input_scale(psid, f))[0]
    c = nc.curve(model, f, np.random.default_rng(2), stop=target)
    s, cell = nc.sigma_min({7: c}, target)
    assert cell == 7 and s == pytest.approx(s0, abs=0.03)
    # A model that already shows the signature needs none.
    assert nc.sigma_min({7: nc.curve(psid, f, np.random.default_rng(2), stop=target)},
                        target)[0] == 0.0
