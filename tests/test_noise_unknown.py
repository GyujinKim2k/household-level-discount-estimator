"""The noise-unknown arm (RESULTS 62): its noise, targets and summaries."""

import numpy as np
import pytest
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from scripts import noise_unknown as nu
from scripts import optionA as oa


def _x(n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.exp(torch.randn((n, 7, len(FEATURES_TWOASSET_AGE)), generator=g)) * 1e4


def test_noise_sd_is_drawn_on_its_box_and_applied_per_feature():
    x = _x()
    y, s = nu.add_noise(x, seed=3)
    assert s.shape == (len(x), 4)
    assert (s >= 0).all() and (s <= torch.as_tensor(nu.SIGMA_HIGH, dtype=s.dtype)).all()
    assert torch.allclose(s.mean(0).double(), torch.as_tensor(nu.SIGMA_HIGH / 2), atol=0.02)
    for k, f in enumerate(nu.NOISY):
        j = FEATURES_TWOASSET_AGE.index(f)
        z = torch.log(y[..., j] / x[..., j]) / s[:, k, None]     # each row's own sigma
        assert abs(float(z.std()) - 1) < 0.02 and abs(float(z.mean())) < 0.02
    age = FEATURES_TWOASSET_AGE.index("age")
    assert torch.equal(y[..., age], x[..., age])


def test_a_given_sigma_reuses_the_stream():
    """Common random numbers: the same seed at another sigma scales the same z."""
    x = _x(500)
    a, _ = nu.add_noise(x, seed=7, sigma=[0.1, 0.1, 0.3, 0.3])
    b, _ = nu.add_noise(x, seed=7, sigma=[0.2, 0.2, 0.6, 0.6])
    j = FEATURES_TWOASSET_AGE.index("consumption")
    assert torch.allclose(2 * torch.log(a[..., j] / x[..., j]), torch.log(b[..., j] / x[..., j]),
                          atol=1e-4)


def test_targets_round_trip_and_stay_in_the_flow_box():
    spec = oa.Spec(box=oa.box_of({"rgamma_range": [1.021, 1.075]}),
                   beta_transform="logit", rgamma_transform="logit")
    rng = np.random.default_rng(0)
    th = np.column_stack([rng.uniform(0.31, 0.99, 50), rng.uniform(0.86, 0.995, 50),
                          rng.uniform(0.6, 4.9, 50), rng.uniform(1.022, 1.074, 50),
                          rng.uniform(0, 1, (50, 4)) * nu.SIGMA_HIGH])
    t = nu.to_flow(th, spec)
    assert np.allclose(nu.from_flow(t, spec), th, atol=1e-7)
    tt = nu.to_flow(torch.from_numpy(th), spec)
    assert np.allclose(tt.numpy(), t)
    fb = nu.flow_box(spec)
    assert (t >= fb.low).all() and (t <= fb.high).all() and len(fb.low) == 8


def test_weighted_correlation_matches_numpy():
    rng = np.random.default_rng(1)
    c = np.eye(8)
    c[0, 4] = c[4, 0] = 0.6
    d = rng.multivariate_normal(np.zeros(8), c, size=(2, 3000)).astype(np.float32)
    lw = np.zeros((2, 3000))
    lw[1, 1500:] = -np.inf        # the second row keeps half its draws
    r = nu.weighted_corr(d, lw, np.array([3000, 1500]))
    assert r[0] == pytest.approx(np.corrcoef(d[0].T), abs=1e-6)
    assert r[1] == pytest.approx(np.corrcoef(d[1, :1500].T), abs=1e-6)
    assert nu._cross(r[:1])["beta"]["sigma_income"] == pytest.approx(0.6, abs=0.05)
