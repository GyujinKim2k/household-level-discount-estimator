"""Frozen first-stage calibration from Laibson et al.'s replication package.

Education group ``comphs`` (their benchmark, ``EDFbatch_baseline.m:22``).
Values are inlined so the 653 MB replication package is **not** a runtime
dependency; ``python -m hh_npe.simulator.laibson_calibration`` re-extracts them
from the package and asserts they still match.

Sources (their ``FirstStageParams.m``):
- demographics — IPUMS-USA
- income process — PSID
- credit limit — SCF
- mortality — SSA TR2023 historical death probabilities, male/female average
  over calendar years 2000-2004, ages 20-90
- second-stage target moments — their ``est_secondstage.mat``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EDUC = "comphs"
AGE_START = 20
AGE_END = 90
AGE_RETIRE = 64  # their ``retireage``; only used by the income-split robustness

# --- demographics: effective household size ------------------------------
# kids_(age)    = a0kids * exp(a1kids*age - a2kids*age^2)
# depadul_(age) = a0depadul * exp(a1depadul*age - a2depadul*age^2)
A0_KIDS = np.float64(0.003410572104586154)
A1_KIDS = np.float64(0.35821261723801895)
A2_KIDS = np.float64(0.005081298245188375)
A0_DEPADUL = np.float64(4.585428728590593e-06)
A1_DEPADUL = np.float64(0.45178921907870556)
A2_DEPADUL = np.float64(0.004382787905395041)

# Household-member weights [spouse, dependent adult, kid] (baseline, not sqrt scale).
HH_WEIGHT = (1.0, 1.0, 0.4)

# --- income: deterministic log profile ------------------------------------
# ymean_(age) = cons + agecoeff*age + age2coeff*age^2/100 + age3coeff*age^3/10000
#               + spousecoeff*spouse + kidscoeff*kids + depadulcoeff*depadul
YWORK_KIDSCOEFF = np.float64(0.013492077455287167)
YWORK_SPOUSECOEFF = np.float64(0.31911520596187626)
YWORK_DEPADULCOEFF = np.float64(0.23651044041995328)
YWORK_AGECOEFF = np.float64(0.13502511541998957)
YWORK_AGE2COEFF = np.float64(-0.2221586353815037)
YWORK_AGE3COEFF = np.float64(0.10646200599193933)
YWORK_CONS = np.float64(7.563421249389648)

# --- income: stochastic component ----------------------------------------
# AR(1) persistent component (variance ``vareps``) + iid transitory (``varnu``).
YWORK_AUTO = np.float64(0.8400135500431678)
YWORK_VAREPS = np.float64(0.05707941151455871)
YWORK_VARNU = np.float64(0.04508554448217923)
YWORK_SIGMAEPS = float(np.sqrt(YWORK_VAREPS))
YWORK_SIGMANU = float(np.sqrt(YWORK_VARNU))

# --- credit limit ---------------------------------------------------------
# creditline_(age) = c0 + c1*age + c2*age^2   (as a multiple of mean income)
C0_CREDIT = np.float64(0.16721227922042575)
C1_CREDIT = np.float64(-0.001869365470594639)
C2_CREDIT = np.float64(0.00013566344085291099)

# --- initial wealth (median wealth-to-average-income ratios) --------------
MED_TOTAL_WEALTH = np.float64(1.4695795059204104)
MED_LIQ_WEALTH = np.float64(0.054860156774520885)

# --- rates of return (their 'benchmark' case) -----------------------------
R_FREE = 1.0203      # liquid saving
R_GAMMA = 1.0500     # illiquid asset
R_CC = 1.1059        # credit-card borrowing

# --- other structural constants (their baseline flags) --------------------
ALPHA_BEQUEST = 0.5  # weight on the bequest motive
N_INCOME_STATES = 3  # their ``nS``
AR1_GRID_SPAN = 1.5  # their ``m``: grid half-width in multiples of the sd

# --- mortality: P(die between age i and i+1), ages 20..90 ----------------
DEATH_PROB = np.array([
    np.float64(0.0009253000000000001), np.float64(0.0009497), np.float64(0.0009505), np.float64(0.0009434),
    np.float64(0.0009381000000000001), np.float64(0.0009358), np.float64(0.0009422999999999999), np.float64(0.0009505),
    np.float64(0.0009629), np.float64(0.0009841), np.float64(0.0010098), np.float64(0.001051),
    np.float64(0.0011065000000000003), np.float64(0.0011768), np.float64(0.0012594), np.float64(0.0013555),
    np.float64(0.0014624), np.float64(0.0015855), np.float64(0.0017194), np.float64(0.0018652000000000002),
    np.float64(0.0020253), np.float64(0.0021964000000000003), np.float64(0.002382), np.float64(0.0025872),
    np.float64(0.0028059), np.float64(0.0030424), np.float64(0.0032958999999999996), np.float64(0.0035637000000000004),
    np.float64(0.003844), np.float64(0.0041329999999999995), np.float64(0.0044346), np.float64(0.004750599999999999),
    np.float64(0.0050941), np.float64(0.0054748), np.float64(0.0059254), np.float64(0.0064429),
    np.float64(0.0070162), np.float64(0.007663499999999999), np.float64(0.0083956), np.float64(0.009182599999999999),
    np.float64(0.010024700000000001), np.float64(0.010931199999999999), np.float64(0.011941499999999999), np.float64(0.0130477),
    np.float64(0.014299800000000001), np.float64(0.0157005), np.float64(0.0171709), np.float64(0.0187691),
    np.float64(0.0204987), np.float64(0.0223924), np.float64(0.0244593), np.float64(0.0267129),
    np.float64(0.029215599999999998), np.float64(0.0319821), np.float64(0.035022300000000006), np.float64(0.038369),
    np.float64(0.0420458), np.float64(0.04609339999999999), np.float64(0.050580099999999996), np.float64(0.055610799999999995),
    np.float64(0.06129949999999999), np.float64(0.0677416), np.float64(0.0749908), np.float64(0.0830679),
    np.float64(0.0920392), np.float64(0.10198700000000001), np.float64(0.11297109999999999), np.float64(0.1250501),
    np.float64(0.1382246), np.float64(0.1523771), np.float64(0.1675718),
])

# --- second-stage target moments -----------------------------------------
# 16 rows = 4 moment types x 4 age bands (21-30, 31-40, 41-50, 51-60), in the
# order [%Visa, meanVisa, wealth|debt, wealth|no debt]. Columns: value, se, N.
MOMENT_NAMES = ("pct_visa", "mean_visa", "wealth_debt", "wealth_nodebt")
AGE_BANDS = ((21, 30), (31, 40), (41, 50), (51, 60))
TARGET_MOMENTS = np.array([
    np.float64(0.6395306417155814), np.float64(0.6292221575171006), np.float64(0.5883615818911393), np.float64(0.5026888776318379),
    np.float64(0.11140327191032179), np.float64(0.0963981110101683), np.float64(0.1096009531059808), np.float64(0.10443145018365),
    np.float64(1.221574513866596), np.float64(1.8679500796161603), np.float64(3.377182686700301), np.float64(4.64977713548991),
    np.float64(1.6585798850310285), np.float64(2.8002288033170393), np.float64(4.612995320394674), np.float64(8.070565796020265),
])
TARGET_MOMENT_SE = np.array([
    np.float64(0.02148365664375412), np.float64(0.02589845728101662), np.float64(0.029116165470790097), np.float64(0.03689864131366247),
    np.float64(0.012358889193796187), np.float64(0.013861757011603465), np.float64(0.017146113661819503), np.float64(0.020241294491796045),
    np.float64(0.11657422380175385), np.float64(0.1667293412294971), np.float64(0.2268413032747909), np.float64(0.34046862967675323),
    np.float64(0.12918191063990117), np.float64(0.1542058567916562), np.float64(0.25180196072952105), np.float64(0.3926143586496589),
])
TARGET_MOMENT_N = np.array([
    np.float64(1097.0), np.float64(1504.0), np.float64(1698.0), np.float64(1411.0),
    np.float64(1097.0), np.float64(1504.0), np.float64(1698.0), np.float64(1411.0),
    np.float64(1097.0), np.float64(1504.0), np.float64(1698.0), np.float64(1411.0),
    np.float64(1097.0), np.float64(1504.0), np.float64(1698.0), np.float64(1411.0),
])

# --- Laibson et al. MSM estimates, table 3 (order: beta, delta, rho) ------
BENCHMARK_PREFS = (0.5305, 0.9891, 1.9355)     # naive quasi-hyperbolic
EXPONENTIAL_PREFS = (1.0, 0.9600, 1.4663)      # beta identically 1


def survival_share() -> np.ndarray:
    """``alive_``: fraction of the age-20 cohort still alive at each age."""
    return np.concatenate([[1.0], np.cumprod(1.0 - DEATH_PROB[:-1])])


# ---------------------------------------------------------------------------
# Education-group bundles
# ---------------------------------------------------------------------------
# Their first stage is estimated separately for each education group, and
# education moves FIVE blocks coherently at once: demographics, the income
# profile, the income AR(1), the credit limit and initial wealth. Changing one
# without the others produces a household that exists in no group.
#
# The module-level globals above remain the ``comphs`` values, so every existing
# caller keeps its exact behaviour; ``COMPHS`` below is the same numbers in
# bundle form, and ``__main__`` asserts that the two agree.
#
# Note the plan for this work spoke of *four* blocks. Demographics is a fifth,
# and it is included because excluding it would give ``somehs`` households
# ``comphs`` family structure -- incoherent, since family size enters both the
# income profile and the consumption equivalence scale. See RESULTS.md 9.8 for
# the coupling this creates with ``scripts/typical_household.py``.


@dataclass(frozen=True)
class Calibration:
    """One education group's first-stage estimates, frozen.

    Selected per Sobol draw during dataset generation and carried on
    :class:`~hh_npe.simulator.twoasset.ModelSpec`, because ``grids.py`` reads
    these at call time and would otherwise silently use ``comphs`` for every
    group.
    """

    educ: str
    a0_kids: float
    a1_kids: float
    a2_kids: float
    a0_depadul: float
    a1_depadul: float
    a2_depadul: float
    ywork_kidscoeff: float
    ywork_spousecoeff: float
    ywork_depadulcoeff: float
    ywork_agecoeff: float
    ywork_age2coeff: float
    ywork_age3coeff: float
    ywork_cons: float
    ywork_auto: float
    ywork_vareps: float
    ywork_varnu: float
    c0_credit: float
    c1_credit: float
    c2_credit: float
    med_total_wealth: float
    med_liq_wealth: float

    @property
    def ywork_sigmaeps(self) -> float:
        """Derived, never stored: keeps sigma consistent with its variance."""
        return float(np.sqrt(self.ywork_vareps))

    @property
    def ywork_sigmanu(self) -> float:
        return float(np.sqrt(self.ywork_varnu))

    @property
    def hh_weight(self) -> tuple[float, float, float]:
        """Not education-varying, but read through the bundle for uniformity."""
        return HH_WEIGHT



COMPHS = Calibration(
    educ="comphs",
    a0_kids=0.003410572104586154,
    a1_kids=0.35821261723801895,
    a2_kids=0.005081298245188375,
    a0_depadul=4.585428728590593e-06,
    a1_depadul=0.45178921907870556,
    a2_depadul=0.004382787905395041,
    ywork_kidscoeff=0.013492077455287167,
    ywork_spousecoeff=0.31911520596187626,
    ywork_depadulcoeff=0.23651044041995328,
    ywork_agecoeff=0.13502511541998957,
    ywork_age2coeff=-0.2221586353815037,
    ywork_age3coeff=0.10646200599193933,
    ywork_cons=7.563421249389648,
    ywork_auto=0.8400135500431678,
    ywork_vareps=0.05707941151455871,
    ywork_varnu=0.04508554448217923,
    c0_credit=0.16721227922042575,
    c1_credit=-0.001869365470594639,
    c2_credit=0.00013566344085291099,
    med_total_wealth=1.4695795059204104,
    med_liq_wealth=0.054860156774520885,
)

SOMEHS = Calibration(
    educ="somehs",
    a0_kids=0.024155518703213332,
    a1_kids=0.26164753291855974,
    a2_kids=0.0037738180548852317,
    a0_depadul=0.00022634369133402177,
    a1_depadul=0.31313322639280566,
    a2_depadul=0.0030287279848345123,
    ywork_kidscoeff=0.06424890369822132,
    ywork_spousecoeff=0.24666779320064575,
    ywork_depadulcoeff=0.22939700441126532,
    ywork_agecoeff=0.07935888552454122,
    ywork_age2coeff=-0.13075285174937756,
    ywork_age3coeff=0.059983717141207066,
    ywork_cons=8.209381103515625,
    ywork_auto=0.8103632426445152,
    ywork_vareps=0.05879097729894339,
    ywork_varnu=0.06858134308308815,
    c0_credit=0.0005713016795078744,
    c1_credit=0.0003207173988604436,
    c2_credit=0.00014744408720943427,
    med_total_wealth=1.0999339818954468,
    med_liq_wealth=-0.037107574939727786,
)

COMPCO = Calibration(
    educ="compco",
    a0_kids=1.868271709288051e-05,
    a1_kids=0.5755466044648688,
    a2_kids=0.007195086028678667,
    a0_depadul=2.0411966527600334e-07,
    a1_depadul=0.5353614844871469,
    a2_depadul=0.0049253553181109345,
    ywork_kidscoeff=-0.015060199432439038,
    ywork_spousecoeff=0.2748529245104821,
    ywork_depadulcoeff=0.1583187861103173,
    ywork_agecoeff=0.24669308278336832,
    ywork_age2coeff=-0.37224764208173683,
    ywork_age3coeff=0.15859636729376456,
    ywork_cons=5.817234039306641,
    ywork_auto=0.7623831704307206,
    ywork_vareps=0.04480460221277371,
    ywork_varnu=0.030324849232566606,
    c0_credit=0.4219486064021487,
    c1_credit=-0.006960359652454074,
    c2_credit=0.00019516738546471283,
    med_total_wealth=3.5893962383270264,
    med_liq_wealth=0.1923161268234253,
)

#: Education groups in a fixed order. The index is what gets recorded per draw
#: during generation, so this order must never change: a reordering would
#: silently relabel every stored nuisance draw.
EDUC_GROUPS = ("comphs", "somehs", "compco")
BUNDLES = {"comphs": COMPHS, "somehs": SOMEHS, "compco": COMPCO}


def bundle(i: int) -> Calibration:
    """The bundle for education index ``i``, in ``EDUC_GROUPS`` order."""
    return BUNDLES[EDUC_GROUPS[i]]


if __name__ == "__main__":  # re-extract from the replication package and verify
    from pathlib import Path

    from scipy.io import loadmat

    root = Path(__file__).resolve().parents[3] / "replication-package-LLMRT"
    b = root / "LifecycleSimulation" / "input" / EDUC
    d = root / "ParameterAndMoments" / "DeathProbs"

    dem = loadmat(b / "est_firststage_demographics.mat")["est_demographics"].squeeze()
    inc = loadmat(b / "est_firststage_income.mat")["est_income"].squeeze()
    ar1 = loadmat(b / "est_firststage_income.mat")["est_ar1"].squeeze()
    cred = loadmat(b / "est_firststage_creditlim.mat")["est_creditlim"].squeeze()
    iw = loadmat(b / "est_firststage_initwealth.mat")["est_initwealth"].squeeze()
    ss = loadmat(b / "est_secondstage.mat")["est_secondstage"]

    m = np.genfromtxt(d / "DeathProbsE_M_Hist_TR2023.csv", delimiter=",", skip_header=2)[:, 1:]
    f = np.genfromtxt(d / "DeathProbsE_F_Hist_TR2023.csv", delimiter=",", skip_header=2)[:, 1:]
    death = ((m + f) / 2)[100:105, 20:91].mean(axis=0)

    checks = [
        ("demographics", dem, [A0_KIDS, A1_KIDS, A2_KIDS, A0_DEPADUL, A1_DEPADUL, A2_DEPADUL]),
        ("income", inc, [YWORK_KIDSCOEFF, YWORK_SPOUSECOEFF, YWORK_DEPADULCOEFF,
                         YWORK_AGECOEFF, YWORK_AGE2COEFF, YWORK_AGE3COEFF, YWORK_CONS]),
        ("ar1", ar1, [YWORK_AUTO, YWORK_VAREPS, YWORK_VARNU]),
        ("creditlim", cred, [C0_CREDIT, C1_CREDIT, C2_CREDIT]),
        ("initwealth", iw, [MED_TOTAL_WEALTH, MED_LIQ_WEALTH]),
        ("death", death, DEATH_PROB),
        ("moments", ss[:, 0], TARGET_MOMENTS),
        ("moment_se", ss[:, 1], TARGET_MOMENT_SE),
        ("moment_N", ss[:, 2], TARGET_MOMENT_N),
    ]
    for name, fresh, frozen in checks:
        np.testing.assert_allclose(np.ravel(fresh), np.ravel(frozen), rtol=0, atol=0)
        print(f"{name:14s} ok ({np.size(fresh)} values)")

    # Every bundle, not just comphs. A wrong value in SOMEHS or COMPCO would
    # not disturb any existing result -- it would only corrupt the two thirds
    # of the regenerated dataset that nothing else checks.
    blocks = [
        ("demographics", "est_demographics",
         ["a0_kids", "a1_kids", "a2_kids",
          "a0_depadul", "a1_depadul", "a2_depadul"]),
        ("income", "est_income",
         ["ywork_kidscoeff", "ywork_spousecoeff", "ywork_depadulcoeff",
          "ywork_agecoeff", "ywork_age2coeff", "ywork_age3coeff",
          "ywork_cons"]),
        ("income", "est_ar1", ["ywork_auto", "ywork_vareps", "ywork_varnu"]),
        ("creditlim", "est_creditlim", ["c0_credit", "c1_credit", "c2_credit"]),
        ("initwealth", "est_initwealth",
         ["med_total_wealth", "med_liq_wealth"]),
    ]
    for g in EDUC_GROUPS:
        bd = root / "LifecycleSimulation" / "input" / g
        cb = BUNDLES[g]
        n = 0
        for fname, key, fields in blocks:
            fresh = loadmat(bd / f"est_firststage_{fname}.mat")[key].squeeze()
            np.testing.assert_allclose(
                np.ravel(fresh), [getattr(cb, f) for f in fields],
                rtol=0, atol=0)
            n += len(fields)
        print(f"bundle {g:8s} ok ({n} values)")

    # The comphs bundle and the module globals are two copies of one thing.
    for a, b in (("a0_kids", A0_KIDS), ("ywork_cons", YWORK_CONS),
                 ("ywork_auto", YWORK_AUTO), ("c0_credit", C0_CREDIT),
                 ("med_liq_wealth", MED_LIQ_WEALTH)):
        assert getattr(COMPHS, a) == b, a
    print("COMPHS bundle agrees with the module globals")
    print("all frozen calibration values match the replication package")
