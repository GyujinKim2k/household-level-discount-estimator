"""Annual simulator output → fixed-length observation-wave trajectories.

Aggregation rule (per SIMULATOR_SPEC.md §6, pinned 2026-08-09):
- **Flows** (income, consumption): summed across the ``wave_years`` window.
- **Stocks** (liquid_assets): value at the end of the final year of the window.

``wave_years`` sets the observation frequency. All phases use
``wave_years=2`` (biennial), matching PSID's post-1997 observation schedule --
the empirical target in Phase 4. ``wave_years=1`` (annual) is supported and
tested, since Laibson et al.'s own moments are annual, but is not the
pre-registered choice.

Wave ``w`` spans annual indices ``[t_start + wave_years*w, t_start +
wave_years*(w+1))`` where ``t_start = start_age - age_start_sim``.

The function also returns an ``alive`` mask flagging waves during which HARK
replaced the household with a newborn (``t_age`` non-monotone within the
window). Callers typically discard households whose ``alive`` is False at any
wave.
"""

from __future__ import annotations

import numpy as np

#: Phase 1-2 feature set (HARK MVP: single liquid asset, no credit cards).
FEATURES_MVP: tuple[str, ...] = ("income", "consumption", "liquid_assets")

#: Phase 3 feature set. ``liquid_assets`` now goes negative (credit-card debt)
#: and ``illiquid_assets`` is observed. Eight of Laibson et al.'s sixteen target
#: moments are wealth *conditional on debt status*, so both asset series carry
#: identifying information -- dropping the illiquid balance would discard the
#: signal that present bias rides on.
FEATURES_TWOASSET: tuple[str, ...] = FEATURES_MVP + ("illiquid_assets",)

#: Phase 3 with the household's age attached to each wave. Needed once the
#: observation window stops being fixed at ages 30-39: consumption and asset
#: levels are strongly age-dependent, so a model shown a window without knowing
#: *when* it is looking has to marginalize over age rather than condition on
#: it. PSID always records the head's age, so conditioning is the honest
#: choice. ``age`` is not a flow, so it is read at the end of each wave like
#: any other stock.
FEATURES_TWOASSET_AGE: tuple[str, ...] = FEATURES_TWOASSET + ("age",)

#: Phase 3 without consumption. This is **Laibson et al.'s own information
#: set**: their 16 target moments are
#: ``[%Visa, meanVisa, wealth|debt, wealth|no debt] x 4 age bands``
#: (`scripts/validate_twoasset.py:32`) and their SCF moment code
#: (`SCF/code/2_buildmoments.do`) has no consumption block at all -- SCF is a
#: wealth survey and does not collect it. So (beta, delta, rho) are identified
#: in their design entirely from credit-card borrowing and wealth conditional
#: on debt status. Dropping consumption returns us to that information set,
#: plus income and the panel structure they never had.
#:
#: The empirical motive is that consumption is the one feature with a
#: measurement problem: PSID expenditure sits ~34% below the simulated level,
#: and the shortfall is *not* uniform -- consumption/income runs 1.68 in the
#: bottom income decile to 0.40 in the top, against a flat ~0.99 in
#: simulation, so no single rescaling can reconcile it. About 21 points of
#: that 34 is our own deliberate stripping of mortgage and vehicle principal
#: (saving, not consumption, and correct to remove); ~8 points are categories
#: PSID never collects; the remainder mixes under-reporting with genuine
#: saving behaviour and is not separable. See PSID_DATA.md.
FEATURES_TWOASSET_NOCONS: tuple[str, ...] = (
    "income", "liquid_assets", "illiquid_assets",
)
FEATURES_TWOASSET_NOCONS_AGE: tuple[str, ...] = (
    FEATURES_TWOASSET_NOCONS + ("age",)
)

#: Without illiquid wealth (RESULTS.md 31): tests whether rho ~ 4.5 is driven by
#: the illiquid-wealth level, which §15 found implies high rho on its own.
FEATURES_TWOASSET_NOILLIQ_AGE: tuple[str, ...] = (
    "income", "consumption", "liquid_assets", "age",
)

#: Named sets, for CLI selection. Order within each is load-bearing.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "noilliq_age": FEATURES_TWOASSET_NOILLIQ_AGE,
    "mvp": FEATURES_MVP,
    "twoasset": FEATURES_TWOASSET,
    "twoasset_age": FEATURES_TWOASSET_AGE,
    "nocons": FEATURES_TWOASSET_NOCONS,
    "nocons_age": FEATURES_TWOASSET_NOCONS_AGE,
}

#: Default remains the MVP set so Phase 1-2 behaviour is unchanged.
FEATURES = FEATURES_MVP

#: Variables measured per unit time; everything else is a stock read at the
#: window's end. How flows are collapsed is set by ``flow_agg`` below.
FLOWS = frozenset({"income", "consumption"})

#: How a flow is collapsed to one number per wave.
#:
#: ``"last"`` — the annual rate in the wave's final year. **This is the default
#: and matches PSID**: a biennial interview reports income for the single
#: preceding calendar year, so the observed series is 2010, 2012, 2014, … and
#: never a two-year total. Verified against the extract — the 2011 wave carries
#: ``TOTAL FAMILY INCOME-2010`` (`PSID_DATA.md`).
#:
#: ``"sum"`` — total over the window. What Phases 1–2 used and what
#: `SIMULATOR_SPEC` §6.2 specified until 2026-09-04. At ``wave_years = 2`` it
#: makes simulated income roughly twice the PSID measure, which is not a
#: harmless scale factor: the embedder standardizes per feature, so it survives
#: as a distorted income-to-debt ratio — exactly the margin `beta` is
#: identified from. Retained only to reproduce Phase 1–2 artifacts.
FLOW_AGG = ("last", "sum")


def aggregate_waves(
    panel: dict[str, np.ndarray],
    age_start_sim: int,
    start_age: int,
    n_waves: int = 5,
    wave_years: int = 2,
    features: tuple[str, ...] = FEATURES,
    flow_agg: str = "last",
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse an annual simulation panel to an ``(N, n_waves, n_features)`` tensor.

    Parameters
    ----------
    panel
        Output of :func:`hh_npe.simulator.forward.simulate_households`. Must
        contain keys ``income``, ``consumption``, ``liquid_assets``, ``t_age``.
    age_start_sim
        The simulator's first-period age (e.g. 20).
    start_age
        Age at which the first wave begins.
    n_waves
        Number of consecutive waves to extract.
    wave_years
        Years spanned by each wave. 2 = biennial (all phases, matches PSID);
        1 = annual (supported, not pre-registered).
    features
        Panel keys to extract, in order. :data:`FEATURES_MVP` (Phases 1-2) or
        :data:`FEATURES_TWOASSET` (Phase 3). Order is load-bearing: the
        embedder's input dimension is positional.
    flow_agg
        How :data:`FLOWS` are collapsed: ``"last"`` (the annual rate in the
        wave's final year, matching PSID's single-year income report) or
        ``"sum"`` (window total, the pre-2026-09-04 behaviour). See
        :data:`FLOW_AGG`. Identical at ``wave_years = 1``.

    Returns
    -------
    x : ndarray, shape ``(N, n_waves, len(features))``, dtype float32
        Feature order matches ``features``.
    alive : ndarray, shape ``(N, n_waves)``, dtype bool
        True if no ``t_age`` reset occurred in the window or in the preceding
        annual step (so the wave fully reflects one continuous household).
    """
    if wave_years < 1:
        raise ValueError(f"wave_years must be >= 1; got {wave_years}")
    if flow_agg not in FLOW_AGG:
        raise ValueError(f"flow_agg must be one of {FLOW_AGG}; got {flow_agg!r}")

    missing = [f for f in features if f not in panel]
    if missing:
        raise KeyError(f"panel is missing feature(s) {missing}; has {sorted(panel)}")
    t_age = panel["t_age"]

    N, T = panel[features[0]].shape
    t_start = start_age - age_start_sim
    t_end = t_start + wave_years * n_waves
    if t_end > T:
        raise ValueError(
            f"Observation window ages {start_age}..{start_age + t_end - t_start - 1} "
            f"exceeds simulator range; have {T} annual periods, need {t_end}."
        )

    x = np.zeros((N, n_waves, len(features)), dtype=np.float32)
    alive = np.ones((N, n_waves), dtype=bool)

    for w in range(n_waves):
        t0 = t_start + wave_years * w
        t1 = t0 + wave_years  # exclusive
        for k, name in enumerate(features):
            series = panel[name]
            if name in FLOWS and flow_agg == "sum":
                x[:, w, k] = series[:, t0:t1].sum(axis=1)
            else:
                # Flows under "last", and every stock, are read at the final
                # year of the wave -- the survey's reference year.
                x[:, w, k] = series[:, t1 - 1]

        # Rebirth detection: t_age must be monotonically non-decreasing across
        # the window (and across the boundary from the prior period if any).
        check_start = max(t0 - 1, 0)
        window_t_age = t_age[:, check_start:t1]
        diffs = np.diff(window_t_age, axis=1)
        alive[:, w] = (diffs >= 0).all(axis=1)

    return x, alive
