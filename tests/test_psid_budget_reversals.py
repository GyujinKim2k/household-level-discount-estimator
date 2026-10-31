"""The budget decomposition of reversals (RESULTS 59).

Synthetic households live year by year: income and consumption each year,
wealth carrying every year's saving. As in PSID, a wave observes the flows of
one year only (the year before the interview) and wealth at the interview,
two years apart. The decomposition must read reporting errors as error and
real transitory movements as real, whatever the persistent dynamics, capital
gains and wealth's own valuation errors.
"""

import numpy as np
import pytest

from scripts import psid_budget_reversals as br


def _panel(n=20_000, waves=7, seed=0, real_c=0.0, err_c=0.0, real_y=0.0, err_y=0.0,
           mpc=0.0, eta_y=0.02, eta_c=0.0, gains=0.0, val_err=0.0, spend_next=False):
    """(y, c, W) observed, (n, waves), in units close to mean income."""
    rng = np.random.default_rng(seed)
    years = 2 * waves
    perm_y = 1.0 + np.cumsum(rng.normal(0, eta_y, (n, years)), 1)
    t_y = rng.normal(0, real_y, (n, years))
    y = perm_y + t_y
    c = (0.8 * perm_y + np.cumsum(rng.normal(0, eta_c, (n, years)), 1)
         + rng.normal(0, real_c, (n, years)))
    if spend_next:      # a windfall saved in its year and spent the next
        c[:, 1:] += t_y[:, :-1]
    else:
        c += mpc * t_y
    w = 3.0 + np.cumsum(y - c + rng.normal(0, gains, (n, years)), 1)
    obs = 2 * np.arange(waves) + 1     # the flow year; wealth at its end
    return (y[:, obs] + rng.normal(0, err_y, (n, waves)),
            c[:, obs] + rng.normal(0, err_c, (n, waves)),
            w[:, obs] + rng.normal(0, val_err, (n, waves)))


def _run(**kw):
    y, c, w = _panel(**kw)
    return br.decompose(y, c, w, y.mean(1, keepdims=True))


def test_intercept_recovers_the_constant_of_a_quadratic():
    k = np.arange(1, 7, dtype=float)
    assert br.intercept(0.3 - 0.1 * k + 0.02 * k ** 2, np.full(6, 1000)) == pytest.approx(0.3)
    assert np.isnan(br.intercept(np.array([1.0, 2, np.nan, np.nan, np.nan, np.nan]),
                                 np.full(6, 1000)))


def test_reporting_errors_read_as_error():
    r = _run(err_c=0.10, err_y=0.08)
    assert r["consumption"]["real_share"] == pytest.approx(0.0, abs=0.06)
    assert r["income"]["real_share"] == pytest.approx(0.0, abs=0.06)
    assert r["consumption"]["error_var"] == pytest.approx(0.10 ** 2, rel=0.12)
    assert r["income"]["error_var"] == pytest.approx(0.08 ** 2, rel=0.12)


def test_real_transitory_movements_read_as_real():
    """Spending paid from wealth, and windfalls 30% spent and 70% saved."""
    r = _run(real_c=0.10, real_y=0.08, mpc=0.3)
    assert r["consumption"]["real_share"] == pytest.approx(1.0, abs=0.07)
    assert r["income"]["real_share"] == pytest.approx(1.0, abs=0.07)
    assert r["saving"]["real_share"] == pytest.approx(1.0, abs=0.07)
    assert r["consumption"]["real_var"] == pytest.approx(0.10 ** 2 + (0.3 * 0.08) ** 2, rel=0.12)


def test_a_mixture_is_split():
    r = _run(real_c=0.08, err_c=0.08, real_y=0.06, err_y=0.08)
    assert r["consumption"]["real_share"] == pytest.approx(0.5, abs=0.07)
    assert r["income"]["real_share"] == pytest.approx(0.36, abs=0.07)


def test_capital_gains_and_valuation_errors_in_wealth_add_noise_not_bias():
    r = _run(n=60_000, real_c=0.08, err_c=0.08, real_y=0.06, err_y=0.08, gains=0.3,
             val_err=0.3)
    assert r["consumption"]["real_share"] == pytest.approx(0.5, abs=0.12)
    assert r["income"]["real_share"] == pytest.approx(0.36, abs=0.12)


def test_a_persistent_consumption_drift_is_not_read_as_real():
    """Consumption wandering away from income, which wealth integrates: k
    and k^2 terms, no constant."""
    r = _run(err_c=0.10, eta_c=0.03)
    assert r["consumption"]["real_share"] == pytest.approx(0.0, abs=0.08)


def test_a_windfall_spent_the_next_year_is_counted_twice():
    """The documented failure: saving that reverses before the next interview
    but one moves wealth at both ends of a change, so the real share reads
    near 2."""
    r = _run(real_y=0.08, spend_next=True)
    assert r["income"]["real_share"] > 1.5
