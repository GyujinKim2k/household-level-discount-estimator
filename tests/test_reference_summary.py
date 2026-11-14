"""RESULTS 70: the numbers RESULTS 1 quotes that no other script reports.

The low-beta shares must count the right ends of the interval, the
comparison with Laibson et al. must use the combined standard error, and on
v2_b010's saved fit (when it is on disk) it must give back the numbers
RESULTS 1 and 63.10 quoted for it.
"""

import json

import numpy as np
import pytest

from scripts import reference_summary as rs


def test_low_beta_counts_the_means_and_the_interval_ends():
    mean = np.array([[0.20, 0], [0.35, 0], [0.50, 0], [0.90, 0]])
    lo = np.array([[0.10, 0], [0.12, 0], [0.29, 0], [0.60, 0]])
    hi = np.array([[0.29, 0], [0.60, 0], [0.80, 0], [1.00, 0]])
    assert rs.low_beta(mean, lo, hi) == {"n": 4, "mean_below_0.3": 0.25, "p5_below_0.3": 0.75,
                                         "p5_below_0.11": 0.25, "p95_below_0.3": 0.25}


def test_the_difference_is_over_the_combined_standard_error():
    se = {"reported": [0.4, 0.004, 4.0], "sampling only": [0.3, 0.003, 3.0]}
    r = rs.against_laibson([0.5, 0.95, 2.0, 1.04], [0.3, 0.003, 3.0, 9.9], [0.0, 1.0, 2.0], se)
    assert r["difference"] == pytest.approx([0.5, -0.05, 0.0])
    assert r["z_reported"] == pytest.approx([1.0, -10.0, 0.0])      # hypot(0.4, 0.3) = 0.5
    assert r["z_sampling_only"] == pytest.approx([0.5 / 0.3 / 2**0.5, -0.05 / 0.003 / 2**0.5, 0])
    assert r["their_sampling_only_over_ours"] == pytest.approx([1.0, 1.0, 1.0])
    assert r["their_reported_over_ours"] == pytest.approx([4 / 3, 4 / 3, 4 / 3])


def test_v2_b010_gives_back_what_results_1_quoted():
    if not rs.LAIBSON_JSON.exists() or not all(
            (rs.Path(d) / "posterior_uncorrected.npz").exists() for d in rs.ARMS["v2_b010"]):
        pytest.skip("v2_b010's outputs are not on disk")
    r = rs.arm_summary("v2_b010", *map(rs.Path, rs.ARMS["v2_b010"]),
                       json.loads(rs.LAIBSON_JSON.read_text()))
    p = r["representative"]
    assert round(max(map(abs, p["z_reported"])), 2) == 0.80                # "|z| <= 0.80"
    assert [round(abs(z), 1) for z in p["z_sampling_only"]] == [1.6, 1.7, 1.6]
    assert [round(v, 1) for v in p["their_sampling_only_over_ours"]] == [1.9, 2.4, 2.4]
    assert [round(v, 1) for v in p["their_reported_over_ours"]] == [4.3, 5.9, 5.1]  # "4.3-5.9"
    assert round(p["max_abs_misfit_over_psid_se"], 2) == 0.02
    assert round(r["couples"]["mean_below_0.3"], 2) == 0.14                  # RESULTS 63.5
    assert round(r["couples"]["p5_below_0.3"], 2) == 0.75
    assert round(r["couples"]["p5_below_0.11"], 2) == 0.08
    assert r["couples"]["p95_below_0.3"] == r["all 889"]["p95_below_0.3"] == 0
