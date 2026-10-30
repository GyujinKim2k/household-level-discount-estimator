"""PSID-only iid error (RESULTS 57.2) and the illiquid reversal statistics (57.1).

The estimator must recover the training noise's sigma when that noise is
added exactly as training adds it (``optionA.add_obs_noise``), whatever the
true process's random walk, household drift, zeros, sign changes or common
year effects, and must find none where there is none.
"""

import numpy as np
import pytest
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE as FEATS
from scripts import illiquid_reversals as ir
from scripts import optionA as oa
from scripts import psid_iid_error as pe

ILL = FEATS.index("illiquid_assets")
LIQ = FEATS.index("liquid_assets")


def _panel(n=4000, waves=7, seed=0, eta=0.10, drift_sd=0.05, year_sd=0.0):
    """Log random walk with household drift and a common year effect, in
    dollars, on the illiquid column of (n, waves, 5) windows."""
    rng = np.random.default_rng(seed)
    g = rng.normal(0, drift_sd, (n, 1))
    year = rng.normal(0, year_sd, (1, waves))
    logv = np.log(50_000) + np.cumsum(rng.normal(0, eta, (n, waves)) + g + year, axis=1)
    x = np.ones((n, waves, len(FEATS)))
    x[..., ILL] = np.exp(logv)
    return x


def _noisy(x, sigma, seed=1):
    return oa.add_obs_noise(torch.from_numpy(x), {"illiquid_assets": sigma}, seed=seed).numpy()


def test_in_run_needs_one_sign_and_no_zero_over_the_whole_gap():
    v = np.array([[1.0, 2.0, 0.0, 3.0, -1.0, -2.0, -3.0]])
    assert pe.in_run(v, 1).tolist() == [[True, False, False, False, True, True]]
    assert pe.in_run(v, 2).tolist() == [[False, False, False, False, True]]
    clean = np.array([[True, True, True, True, True, False, True]])
    assert pe.in_run(v, 1, clean).tolist() == [[True, False, False, False, False, False]]


@pytest.mark.parametrize("sigma", [0.1, 0.25])
def test_the_drift_spec_recovers_the_training_noise(sigma):
    x = _noisy(_panel(), sigma)
    e = pe.estimate(x[..., ILL])
    assert e["iqr"]["rw+drift"]["sigma"] == pytest.approx(sigma, abs=0.02)
    assert e["winsorised"]["rw+drift"]["sigma"] == pytest.approx(sigma, abs=0.02)


def test_without_drift_every_spec_recovers_it():
    x = _noisy(_panel(drift_sd=0.0), 0.15)
    e = pe.estimate(x[..., ILL])
    for spec in pe.SPECS:
        assert e["iqr"][spec]["sigma"] == pytest.approx(0.15, abs=0.02)


def test_no_noise_reads_as_none():
    e = pe.estimate(_panel()[..., ILL])
    assert e["iqr"]["rw+drift"]["sigma"] < 0.03
    assert e["winsorised"]["rw+drift"]["sigma"] < 0.03


def test_a_common_year_effect_is_not_read_as_noise():
    e = pe.estimate(_panel(year_sd=0.3)[..., ILL])
    assert e["iqr"]["rw+drift"]["sigma"] < 0.03


def test_zeros_and_sign_changes_do_not_bias_it():
    """A fifth of household-waves at zero and a third of households switching
    sign mid-window: only changes inside runs are used, and the noise, which
    keeps zeros and signs, is still recovered."""
    rng = np.random.default_rng(3)
    x = _panel(drift_sd=0.0)
    v = x[..., ILL]
    v[rng.random(v.shape) < 0.2] = 0.0
    flip = rng.random(len(v)) < 1 / 3
    v[flip, 4:] *= -1
    x[..., LIQ] = v
    y = oa.add_obs_noise(torch.from_numpy(x), {"liquid_assets": 0.2}, seed=2).numpy()
    e = pe.estimate(y[..., LIQ])
    assert e["iqr"]["rw+drift"]["sigma"] == pytest.approx(0.2, abs=0.025)


def test_the_bootstrap_interval_covers_the_truth():
    x = _noisy(_panel(n=1500), 0.15)
    b = pe.bootstrap(x[..., ILL], None, n_boot=40)
    lo, hi = b["iqr"]["rw+drift"]
    assert lo < 0.15 < hi


def test_reversals_iid_noise_random_walk_and_steady_saving():
    rng = np.random.default_rng(0)
    ybar = np.ones((20_000, 1))
    flat = 2.0 + 0.1 * rng.standard_normal((20_000, 7))              # noise around a level
    walk = 2.0 + np.cumsum(0.1 * rng.standard_normal((20_000, 7)), 1)
    steady = 2.0 + np.cumsum(0.1 + 0.05 * np.abs(rng.standard_normal((20_000, 1)))
                             * np.ones((1, 7)), 1) + 0.01 * rng.standard_normal((20_000, 7))
    assert ir.lag1_corr(flat, ybar)[0] == pytest.approx(-0.5, abs=0.03)
    assert ir.lag1_corr(walk, ybar)[0] == pytest.approx(0.0, abs=0.03)
    assert ir.lag1_corr(steady, ybar)[0] > 0.5


def test_freezing_a_component_that_never_moves_removes_nothing():
    rng = np.random.default_rng(1)
    move = np.abs(1e4 * (5 + np.cumsum(rng.standard_normal((400, 7)), 1)))
    comp = {k: np.zeros((400, 7)) for k in ir.COMPONENTS}
    comp["home equity"] = move
    comp["vehicles"] = np.full((400, 7), 8_000.0)
    f = ir.freezes(comp, np.full((400, 1), 50_000.0), n_boot=10)
    assert f["vehicles"]["V1_removed"] == pytest.approx(0.0, abs=1e-12)
    assert f["home equity"]["V1_removed"] > 0.9


def _ar_panel(n=4000, waves=7, seed=4, phi=0.6, sd=0.3):
    """A stationary AR(1) per wave in logs around each household's level."""
    rng = np.random.default_rng(seed)
    e = np.empty((n, waves))
    e[:, 0] = rng.normal(0, sd, n)
    for t in range(1, waves):
        e[:, t] = phi * e[:, t - 1] + rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    x = np.ones((n, waves, len(FEATS)))
    x[..., ILL] = np.exp(rng.normal(np.log(50_000), 0.5, (n, 1)) + e)
    return x


@pytest.mark.parametrize("make", [_panel, _ar_panel])
def test_the_range_brackets_the_truth(make):
    """Low (rw+drift+ar, IQR) at or below the true sigma, high (rw+drift,
    winsorised) at or above it, for a random walk with drift and for a
    mean-reverting true process."""
    e = pe.estimate(_noisy(make(), 0.15)[..., ILL])
    assert e["iqr"]["rw+drift+ar"]["sigma"] <= 0.15 + 0.02
    assert e["winsorised"]["rw+drift"]["sigma"] >= 0.15 - 0.02


def test_iqr_reads_the_typical_error_and_winsorising_counts_gross_ones():
    """Noise that is N(0, 0.1) for 90% of records and N(0, 0.6) for 10% (sd
    0.21): the IQR reads near the typical record's 0.1, the winsorised
    variance near the overall sd."""
    rng = np.random.default_rng(5)
    x = _panel(drift_sd=0.0)
    z = np.where(rng.random(x.shape[:2]) < 0.1, rng.normal(0, 0.6, x.shape[:2]),
                 rng.normal(0, 0.1, x.shape[:2]))
    e = pe.estimate(x[..., ILL] * np.exp(z))
    assert e["iqr"]["rw+drift"]["sigma"] < 0.14
    assert e["winsorised"]["rw+drift"]["sigma"] == pytest.approx(0.21, abs=0.03)
