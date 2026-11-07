"""RESULTS 65's annual arms: the years each arm reads, and which biennial arm
each is compared with."""

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
