"""The variance decomposition behind the project's headline claim.

``Var(true theta) = Var(posterior means) - E[posterior variance]``. Everything
reported about between-household heterogeneity rests on it, and two mistakes in
it would be invisible in the output:

* **Forgetting to subtract the within term.** Spread in noisy estimates of one
  common parameter looks like heterogeneity. The whole point of the correction
  is that it must come out negative when there is none.
* **Bootstrapping the two terms separately.** They are estimated on the same
  households and move together, so the difference has to be resampled as one
  quantity or its interval is wrong.

Synthetic data, where the true heterogeneity is set by construction.
"""

import numpy as np
import pytest

from hh_npe.npe.prior import PHASE3
from scripts.heterogeneity_test import decompose

N = 4000


def _fake(true_sd: float, post_sd: float, seed: int = 0):
    """Posterior means/sds for N households with a known true spread.

    Each household's posterior mean is its true value plus independent
    estimation error of size `post_sd`, which is exactly the model the
    decomposition assumes.
    """
    rng = np.random.default_rng(seed)
    truth = rng.normal(0.8, true_sd, N)
    mean = np.stack([truth + rng.normal(0, post_sd, N)] * 3, axis=1)
    sd = np.full((N, 3), post_sd)
    return mean, sd


def test_recovers_a_known_positive_variance():
    mean, sd = _fake(true_sd=0.10, post_sd=0.05)
    r = decompose(mean, sd, n_boot=500, seed=0)
    assert r["beta"]["var_true"] == pytest.approx(0.01, rel=0.15)
    assert r["beta"]["p_le_zero"] < 0.01


def test_homogeneous_population_gives_a_negative_or_zero_estimate():
    """The load-bearing one: no true spread must not read as heterogeneity."""
    mean, sd = _fake(true_sd=0.0, post_sd=0.05, seed=1)
    r = decompose(mean, sd, n_boot=500, seed=0)
    assert r["beta"]["var_true"] < 0.002        # ~0 up to sampling error
    assert r["beta"]["p_le_zero"] > 0.2


def test_inflating_posterior_widths_destroys_apparent_heterogeneity():
    """Why the rho finding flipped: over-confident posteriors inflate
    Var(true), and widening them to honest widths removes it."""
    rng = np.random.default_rng(2)
    truth = rng.normal(0.8, 0.0, N)             # homogeneous in truth
    err = rng.normal(0, 0.05, N)
    mean = np.stack([truth + err] * 3, axis=1)
    over = decompose(mean, np.full((N, 3), 0.02), n_boot=500, seed=0)
    honest = decompose(mean, np.full((N, 3), 0.05), n_boot=500, seed=0)
    assert over["beta"]["var_true"] > 0         # spurious heterogeneity
    assert honest["beta"]["var_true"] < over["beta"]["var_true"]
    assert honest["beta"]["p_le_zero"] > over["beta"]["p_le_zero"]


def test_nan_households_are_dropped_not_propagated():
    mean, sd = _fake(true_sd=0.1, post_sd=0.05, seed=3)
    mean[:50] = np.nan
    sd[:50] = np.nan
    r = decompose(mean, sd, n_boot=200, seed=0)
    assert r["n"] == N - 50
    assert np.isfinite(r["beta"]["var_true"])


def test_both_ratio_conventions_are_reported_and_differ_when_widths_are_skewed():
    """RESULTS 1 uses the median posterior sd, the identity needs the RMS."""
    mean, _ = _fake(true_sd=0.1, post_sd=0.05, seed=4)
    rng = np.random.default_rng(5)
    sd = np.abs(rng.lognormal(np.log(0.05), 0.8, (N, 3)))   # skewed widths
    r = decompose(mean, sd, n_boot=200, seed=0)
    assert r["beta"]["within_sd_rms"] > r["beta"]["within_sd_median"]
    assert r["beta"]["ratio_median_sd"] > r["beta"]["ratio_rms"]


def test_every_parameter_is_reported():
    mean, sd = _fake(true_sd=0.1, post_sd=0.05, seed=6)
    r = decompose(mean, sd, n_boot=100, seed=0)
    assert set(PHASE3.names) <= set(r)
