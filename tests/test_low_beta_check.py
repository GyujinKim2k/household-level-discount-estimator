"""RESULTS 64.12's check of beta below 0.3: the ranking measure, the bands,
the per-household summaries and the alive mean. Synthetic arrays only."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from hh_npe.data.waves import FEATURES_TWOASSET_AGE as F
from hh_npe.simulator import grids
from scripts import low_beta_check as lb


def test_auc_is_sklearns_with_ties_counted_half():
    rng = np.random.default_rng(0)
    score = rng.integers(0, 5, 500).astype(float)          # many ties
    y = rng.uniform(size=500) < 0.4
    assert lb.auc(score, y) == pytest.approx(roc_auc_score(y, score))
    assert lb.auc(np.zeros(10), np.arange(10) < 5) == 0.5


def test_the_last_band_holds_the_box_top():
    t = np.array([0.1, 0.3, 0.9, 1.0])
    assert lb.band(t, 0.1, 0.3).tolist() == [True, False, False, False]
    assert lb.band(t, 0.8, 1.0).tolist() == [False, False, True, True]


def _sbc(mean_of):
    t = np.repeat(np.linspace(0.1, 1.0, 901), 4)
    m = mean_of(t)
    truth = np.column_stack([t, np.zeros((len(t), 3))])
    mean = np.column_stack([m, np.zeros((len(t), 3))])
    sd = np.full_like(mean, 0.1)
    return {"truth": truth, "mean": mean, "sd": sd, "lo": mean - 0.15, "hi": mean + 0.15}


def test_bands_rank_perfectly_when_means_are_the_truth_and_not_at_all_when_flat():
    exact = lb.bands(_sbc(lambda t: t))
    assert all(p["auc_beta_mean"] == 1.0 for p in exact["pairs"])
    assert all(b["corr"] == pytest.approx(1.0) and b["coverage_90"] == 1.0
               for b in exact["bands"])
    flat = lb.bands(_sbc(lambda t: np.full_like(t, 0.55)))
    assert all(p["auc_beta_mean"] == 0.5 for p in flat["pairs"])
    assert flat["p95_below_0.3_at_0.1_0.2"] == 0.0 and flat["p5_above_0.8_at_0.9_1.0"] == 0.0
    # intervals of +-0.15 around the truth: below 0.3 only for truths under 0.15,
    # above 0.8 only for truths over 0.95, so half of each end band
    assert exact["p95_below_0.3_at_0.1_0.2"] == pytest.approx(0.5, abs=0.011)
    assert exact["p5_above_0.8_at_0.9_1.0"] == pytest.approx(0.5, abs=0.011)


def test_summaries_read_the_limit_and_the_ratios_from_raw_windows():
    n_w = 3
    age = np.array([30.0, 32.0, 34.0])
    lim, ym = grids.credit_limit(age, lb.XJUMP), grids.mean_income(age)
    x = np.zeros((2, n_w, len(F)))
    i = {f: F.index(f) for f in F}
    x[..., i["age"]] = age
    x[..., i["income"]] = ym
    x[..., i["consumption"]] = 0.9 * ym
    x[0, :, i["liquid_assets"]] = -lim                       # at the limit every wave
    x[1, :, i["liquid_assets"]] = [ym[0], -0.5 * lim[1], ym[2]]
    x[1, :, i["illiquid_assets"]] = ym * [1.0, 2.0, 3.0]      # one mean income a wave
    s = dict(zip(lb.SUMMARIES, lb.summaries(x).T, strict=True))
    assert s["waves_at_limit"].tolist() == [1.0, 0.0]
    assert s["waves_in_debt"] == pytest.approx([1.0, 1 / 3])
    assert s["illiquid_mean"] == pytest.approx([0.0, 2.0])
    assert s["illiquid_slope"] == pytest.approx([0.0, 1.0])
    assert s["c_over_y_mean"] == pytest.approx([0.9, 0.9])
    assert s["c_over_y_sd"] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert s["income_mean"] == pytest.approx([1.0, 1.0])
    x[0, 0, i["liquid_assets"]] = np.nan                     # missing waves are skipped
    x[1, 1, i["liquid_assets"]] = np.nan
    s = dict(zip(lb.SUMMARIES, lb.summaries(x).T, strict=True))
    assert s["waves_at_limit"].tolist() == [1.0, 0.0]
    assert s["waves_in_debt"].tolist() == [1.0, 0.0]


def test_the_alive_mean_skips_years_after_the_head_is_replaced():
    v = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    alive = np.array([[True, True, False, False], [True, True, True, True]])
    assert lb._mean_alive(v, alive).tolist() == [1.0, 0.5]


@pytest.mark.parametrize("kind", ["logit", "spline"])
def test_a_low_variance_classifier_loses_nothing_on_one_monotone_score(kind):
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(300), 16)                         # draws of 16 households
    y = np.repeat(rng.uniform(size=300) < 0.5, 16)
    score = y + rng.normal(0, 1.2, len(y))
    assert lb.cv_auc(score[:, None], y, g, kind) == pytest.approx(lb.auc(score, y), abs=0.01)
