"""Education bundles reach the solver, and comphs stays bit-identical.

Two failure modes, both silent:

* **The bundle does not reach the solver.** ``grids.py`` reads these values at
  call time, so before this plumbing a ``ModelSpec`` could not express "solve
  this draw as somehs" -- it would return a comphs solution and nothing would
  complain. Two thirds of a regenerated dataset would be mislabelled.
* **The default drifts.** Every result before Phase 4's regeneration is comphs,
  and ``COMPHS`` is a second copy of the module globals. If the two ever
  disagree, past and future runs stop being comparable.

Bit-exactness against the replication package's ``.mat`` files is checked
separately by ``python -m hh_npe.simulator.laibson_calibration``, which needs
the 653 MB package and so cannot live here.
"""

import dataclasses

import numpy as np
import pytest

from hh_npe.simulator import grids
from hh_npe.simulator import laibson_calibration as cal
from hh_npe.simulator.twoasset import COARSE, ModelSpec, simulate, solve

AGE = grids.ages()


def test_default_spec_is_comphs():
    assert ModelSpec().calib is cal.COMPHS
    assert COARSE.calib.educ == "comphs"


@pytest.mark.parametrize("name,attr", [
    ("A0_KIDS", "a0_kids"), ("A1_KIDS", "a1_kids"), ("A2_KIDS", "a2_kids"),
    ("A0_DEPADUL", "a0_depadul"), ("A1_DEPADUL", "a1_depadul"),
    ("A2_DEPADUL", "a2_depadul"),
    ("YWORK_CONS", "ywork_cons"), ("YWORK_AGECOEFF", "ywork_agecoeff"),
    ("YWORK_AGE2COEFF", "ywork_age2coeff"),
    ("YWORK_AGE3COEFF", "ywork_age3coeff"),
    ("YWORK_KIDSCOEFF", "ywork_kidscoeff"),
    ("YWORK_SPOUSECOEFF", "ywork_spousecoeff"),
    ("YWORK_DEPADULCOEFF", "ywork_depadulcoeff"),
    ("YWORK_AUTO", "ywork_auto"), ("YWORK_VAREPS", "ywork_vareps"),
    ("YWORK_VARNU", "ywork_varnu"),
    ("C0_CREDIT", "c0_credit"), ("C1_CREDIT", "c1_credit"),
    ("C2_CREDIT", "c2_credit"),
    ("MED_TOTAL_WEALTH", "med_total_wealth"),
    ("MED_LIQ_WEALTH", "med_liq_wealth"),
])
def test_comphs_bundle_matches_the_module_globals(name, attr):
    """Exact equality, not allclose: these are two copies of one literal."""
    assert getattr(cal.COMPHS, attr) == getattr(cal, name)


def test_derived_sigmas_track_their_variances():
    for g in cal.EDUC_GROUPS:
        b = cal.BUNDLES[g]
        assert b.ywork_sigmaeps == pytest.approx(np.sqrt(b.ywork_vareps))
        assert b.ywork_sigmanu == pytest.approx(np.sqrt(b.ywork_varnu))


def test_educ_group_order_is_fixed():
    """The index is recorded per draw during generation, so a reordering would
    silently relabel every stored nuisance draw."""
    assert cal.EDUC_GROUPS == ("comphs", "somehs", "compco")
    for i, g in enumerate(cal.EDUC_GROUPS):
        assert cal.bundle(i).educ == g


@pytest.mark.parametrize("fn", ["mean_income", "mean_log_income",
                                "effective_hh_size"])
def test_grids_default_equals_explicit_comphs(fn):
    f = getattr(grids, fn)
    assert np.array_equal(f(AGE), f(AGE, cal.COMPHS))


def test_credit_limit_and_tauchen_defaults_are_comphs():
    assert np.array_equal(grids.credit_limit(AGE, 1000.0),
                          grids.credit_limit(AGE, 1000.0, cal.COMPHS))
    s0, p0 = grids.tauchen()
    s1, p1 = grids.tauchen(c=cal.COMPHS)
    assert np.array_equal(s0, s1) and np.array_equal(p0, p1)


def test_explicit_psi_still_overrides_the_bundle():
    """The nuisance-sweep scripts pass psi directly; the bundle must not win."""
    s, _ = grids.tauchen(psi=0.5, sigma_eps=0.2, c=cal.COMPCO)
    ref, _ = grids.tauchen(psi=0.5, sigma_eps=0.2, c=cal.COMPHS)
    assert np.array_equal(s, ref)


@pytest.mark.parametrize("g", ["somehs", "compco"])
def test_groups_produce_different_income_and_credit(g):
    b = cal.BUNDLES[g]
    assert not np.allclose(grids.mean_income(AGE, b),
                           grids.mean_income(AGE, cal.COMPHS))
    assert not np.allclose(grids.credit_limit(AGE, 1000.0, b),
                           grids.credit_limit(AGE, 1000.0, cal.COMPHS))


@pytest.fixture(scope="module")
def comphs_solution():
    return solve(0.85, 0.99, 2.0, spec=COARSE)


def test_bundle_reaches_the_solver(comphs_solution):
    """The load-bearing one: a ModelSpec carrying SOMEHS must not silently
    return a comphs solution."""
    sp = dataclasses.replace(COARSE, calib=cal.SOMEHS)
    assert not np.array_equal(solve(0.85, 0.99, 2.0, spec=sp).next_x,
                              comphs_solution.next_x)


def test_explicit_comphs_is_bit_identical_to_the_default(comphs_solution):
    sp = dataclasses.replace(COARSE, calib=cal.COMPHS)
    other = solve(0.85, 0.99, 2.0, spec=sp)
    assert np.array_equal(other.next_x, comphs_solution.next_x)
    assert np.array_equal(other.cons, comphs_solution.cons)


def test_forward_pass_uses_the_bundle_too(comphs_solution):
    """`simulate` re-derives income and the credit limit from `sol.spec`; if it
    read module globals the panel would mix two groups' calibrations."""
    sp = dataclasses.replace(COARSE, calib=cal.COMPCO)
    p = simulate(solve(0.85, 0.99, 2.0, spec=sp), n_households=200, seed=1)
    q = simulate(comphs_solution, n_households=200, seed=1)
    # compco's income profile is much steeper (agecoeff 0.247 vs 0.135).
    assert np.median(p["income"][:, 15:25]) > np.median(q["income"][:, 15:25])
