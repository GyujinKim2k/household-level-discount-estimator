"""RESULTS 65's annual arms: the years each arm reads, and which biennial arm
each is compared with."""

import numpy as np
import pytest

from scripts import annual_waves as aw
from scripts import waves_experiment as we


def test_years_read():
    assert {a: aw.years_read(a) for a in aw.ARMS} == {
        "w07": 14, "a14": 15, "w14": 28, "w10": 20, "a20": 21, "w20": 40}


def test_each_annual_arm_has_its_two_comparisons():
    for an, bi in aw.SAME_COUNT:
        assert we.parse_arm(an)[0] == we.parse_arm(bi)[0] and we.parse_arm(an)[1] == 1
    for an, bi in aw.SAME_YEARS:                    # the annual arm reads one more year
        assert aw.years_read(an) == aw.years_read(bi) + 1


def test_every_annual_window_reads_inside_24_to_64():
    for a in aw.ANNUAL:
        n, wy = we.parse_arm(a)
        lo, hi = we.start_low(wy), we.start_high(n, wy)
        assert lo - we.FLOW_LAG == we.FIRST_AGE              # first flow year
        assert hi + wy * n - 1 == we.LAST_AGE                # last stock year
        assert aw.years_read(a) == (hi + n - 1) - (hi - we.FLOW_LAG) + 1


def test_beta_through_splits_betas_variance_into_its_parts():
    rng = np.random.default_rng(0)
    J = np.eye(4) + 0.3 * rng.normal(size=(4, 4))
    a = rng.normal(size=(4, 4))
    v = a @ a.T
    b = aw.beta_through(J, v)
    r = np.linalg.inv(J)[0]
    assert np.allclose(b["j_inv_beta_row"], r)
    assert sum(b["own_share"]) + b["cross_share"] == pytest.approx(1.0)
    assert b["own_share"][1] == pytest.approx(r[1] ** 2 * v[1, 1] / (r @ v @ r))
    d = aw.beta_through(np.eye(4), np.diag([1.0, 2.0, 3.0, 4.0]))    # beta on its own
    assert d["own_share"] == [1.0, 0.0, 0.0, 0.0] and d["cross_share"] == 0.0


def test_by_start_groups_cardholders_by_their_window_start():
    starts = np.array([25] * 40 + [33] * 40 + [40] * 40 + [50] * 20)
    card = np.ones(len(starts), int)
    card[:5] = 0                                       # not cardholders: left out
    mean = np.zeros((len(starts), 4))
    mean[:40, 0] = np.r_[np.zeros(5), np.tile([0.0, 1.0], 17), [0.0]]
    sd = np.full((len(starts), 4), 0.2)
    rows = aw.by_start(mean, sd, starts, card)
    assert [r["start"] for r in rows] == [[24, 31], [32, 37], [38, 44]]   # 45-51 has 20 < 30
    assert rows[0]["n"] == 35
    assert rows[0]["mean_sd_across"][0] == pytest.approx(np.std(mean[5:40, 0]))
    assert rows[1]["mean_sd_across"] == [0.0, 0.0]
    assert rows[2]["median_posterior_sd"] == [0.2, 0.2]
