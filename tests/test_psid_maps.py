"""PSID variable maps: the failure mode here is silent and wave-specific.

Every empirical result runs through hand-built dictionaries of PSID variable
codes, one entry per wave. A wrong code does not raise -- it returns a
different quantity, and the number that comes out is plausible. Two real bugs
of exactly this kind were found by audit rather than by anything failing:

* 2011 reports other-real-estate and farm/business as COMBINED NET values
  (``W2``, ``W11``); 2013 onward splits them into asset and debt legs
  (``W2A/W2B``, ``W11A/W11B``). Reading the 2011 codes as gross assets mixed
  net and gross across the panel.
* A double space inside ``IMP VAL OTH REAL ESTATE DEBT  (W2B) 2013`` hid that
  variable from a regex, so the 2013 debt leg was silently absent.

These tests need no PSID microdata -- they check the maps themselves, which is
where the bugs live.
"""

import pytest

from scripts.build_psid_tensor import CPI, CPI_BASE, FAM, REF_YEAR, WAVES

# Codes that are legitimately absent in some waves, with the reason.
KNOWN_GAPS = {
    "cd": {2011, 2013, 2015, 2017},      # CD/bonds inside W28 until 2019
    "re_b": {2011},                       # 2011 reports combined net W2
    "fb_b": {2011},                       # 2011 reports combined net W11
    "othdebt": {2011},                    # residual "other debt" added in 2013
}


def test_every_map_covers_every_wave():
    for key, per_wave in FAM.items():
        assert set(per_wave) == set(WAVES), f"{key} does not cover all waves"


def test_only_documented_gaps_are_none():
    """A None that is not in KNOWN_GAPS is a missing code, not a real gap."""
    for key, per_wave in FAM.items():
        missing = {y for y, v in per_wave.items() if v is None}
        assert missing == KNOWN_GAPS.get(key, set()), (
            f"{key}: unexpected missing waves {missing - KNOWN_GAPS.get(key, set())}"
            f" / unexpectedly present {KNOWN_GAPS.get(key, set()) - missing}"
        )


def test_codes_are_never_reused_across_concepts_within_a_wave():
    """The 2011 bug: ER52354 was both `re_a` and the combined net W2.

    Two concepts sharing a code in the same wave means one of them is reading
    the wrong quantity, and nothing downstream would notice.
    """
    for w in WAVES:
        seen = {}
        for key, per_wave in FAM.items():
            v = per_wave[w]
            if v is None:
                continue
            assert v not in seen, (
                f"wave {w}: {key} and {seen[v]} both map to {v}"
            )
            seen[v] = key


def test_codes_are_plausible_psid_identifiers():
    for key, per_wave in FAM.items():
        for w, v in per_wave.items():
            if v is None:
                continue
            assert v.startswith("ER") and v[2:3].isdigit(), f"{key}[{w}] = {v}"


def test_codes_increase_with_wave():
    """PSID assigns ER numbers in ascending survey order.

    A code that goes backwards across waves is almost always a copy-paste from
    the wrong year -- the one class of error that stays plausible downstream.
    """
    for key, per_wave in FAM.items():
        # Only the leading numeric block: codes like ER58212E4 carry a letter
        # suffix, and concatenating digits across it gives 582124.
        def _num(code):
            import itertools
            return int("".join(itertools.takewhile(str.isdigit, code[2:])))

        nums = [(w, _num(per_wave[w])) for w in WAVES if per_wave[w] is not None]
        for (w0, n0), (w1, n1) in zip(nums, nums[1:]):
            assert n1 > n0, f"{key}: {w1} code is below {w0} ({n1} <= {n0})"


def test_reference_year_is_the_year_before_the_interview():
    """PSID wave Y reports income for calendar Y-1 (PSID_DATA.md)."""
    for y in WAVES:
        assert REF_YEAR[y] == y - 1


def test_cpi_covers_every_year_used_and_is_based_on_2010():
    assert CPI_BASE == CPI[2010]
    for y in WAVES:
        assert y in CPI and REF_YEAR[y] in CPI
    # Monotone over our span: a transcription slip would show up as a dip.
    years = sorted(y for y in CPI if 2010 <= y <= 2023)
    vals = [CPI[y] for y in years]
    assert vals == sorted(vals), "CPI series is not monotone over 2010-2023"


@pytest.mark.parametrize("key", ["stud", "med", "legal", "famloan"])
def test_unsecured_debts_present_in_all_waves(key):
    """These net out of illiquid wealth; a gap would silently inflate Z."""
    assert all(FAM[key][w] is not None for w in WAVES)


# --- education groups ------------------------------------------------------
# `2_buildmoments.do:69-74` defines their three groups by SCF EDCL:
#   somehs EDCL 1, comphs EDCL 2-3, compco EDCL 4.
# PSID reports years completed, so the mapping is by year equivalent. A wrong
# cut point would silently assign households the wrong income process, credit
# limit and initial wealth -- and nothing downstream would notice.

def test_educ_index_matches_their_edcl_definitions():
    import numpy as np

    from scripts.build_psid_tensor import EDUC_GROUPS, educ_index

    years = np.array([0.0, 8.0, 11.0, 12.0, 13.0, 15.0, 16.0, 17.0, np.nan])
    got = educ_index(years)
    hs, co, cc = (EDUC_GROUPS.index(g) for g in ("somehs", "comphs", "compco"))
    assert list(got) == [hs, hs, hs, co, co, co, cc, cc, -1]


def test_educ_group_order_matches_the_calibration_bundles():
    """The stored index is read back through `laibson_calibration.EDUC_GROUPS`;
    if the two orderings drift apart every household is relabelled."""
    from hh_npe.simulator.laibson_calibration import EDUC_GROUPS as CAL_GROUPS
    from scripts.build_psid_tensor import EDUC_GROUPS

    assert EDUC_GROUPS == CAL_GROUPS


def test_comphs_cut_points_are_the_ones_the_matched_sample_uses():
    """--educ_groups and --match_laibson must agree on who is comphs, or the
    comphs subset of the all-groups tensor stops reproducing the 889."""
    import numpy as np

    from scripts.build_psid_tensor import (COMPHS_HI, COMPHS_LO, EDUC_GROUPS,
                                           educ_index)

    years = np.arange(0.0, 18.0)
    from_index = educ_index(years) == EDUC_GROUPS.index("comphs")
    from_filter = (years >= COMPHS_LO) & (years < COMPHS_HI)
    assert np.array_equal(from_index, from_filter)
