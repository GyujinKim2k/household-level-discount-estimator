"""Age profiles, asset grids and shock discretization for the two-asset model.

Ported from Laibson et al.'s ``LifecycleSim.m`` (baseline flags only: no
``sqrtscale``, no ``mortalityn``, no ``split_income_at_retirement``, no income
variance rescaling, ``zilliq = 0``). Calibration constants live in
:mod:`hh_npe.simulator.laibson_calibration`.

One deviation from their code, made deliberately for vectorization: they store
a **separate** liquid-asset grid per age (``X__(1:nX_(t), t)``), because the
credit limit is age-varying. We build a single grid wide enough for the
largest credit limit over the lifecycle and expose a per-age feasibility mask
instead. Since the negative part of their grid is uniform at ``xjump`` spacing
and every ``xmin_(t)`` is a multiple of ``xjump``, each age's grid is an exact
subset of the common grid -- no interpolation, no approximation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from hh_npe.simulator import laibson_calibration as cal

# Their hard-coded grid-refinement constants (LifecycleSim.m:147, 175).
XJUMP_MAX = 6000.0
ZJUMP_MAX = 96000.0
X_CELLS_PER_STEP = 50
Z_CELLS_PER_STEP = 10


def ages(age_start: int = cal.AGE_START, age_end: int = cal.AGE_END) -> np.ndarray:
    """Inclusive age grid, e.g. 20..90 → 71 entries."""
    return np.arange(age_start, age_end + 1, dtype=float)


# Every function below takes the education bundle as ``c``, defaulting to
# ``comphs`` so existing callers are unchanged. These used to read the module
# globals at call time, which meant a ModelSpec could not express "solve this
# draw as somehs" -- the solver would silently use comphs for every group.


def household_composition(
    age: np.ndarray, c: cal.Calibration = cal.COMPHS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(spouse, kids, dependent adults)`` at each age."""
    spouse = np.full_like(age, 2.0)
    kids = c.a0_kids * np.exp(c.a1_kids * age - c.a2_kids * age**2)
    depadul = c.a0_depadul * np.exp(c.a1_depadul * age - c.a2_depadul * age**2)
    return spouse, kids, depadul


def effective_hh_size(
    age: np.ndarray, c: cal.Calibration = cal.COMPHS
) -> np.ndarray:
    """``effhhN_``: consumption-equivalent household size (baseline weights)."""
    spouse, kids, depadul = household_composition(age, c)
    w_spouse, w_depadul, w_kids = c.hh_weight
    return w_spouse * spouse + w_depadul * depadul + w_kids * kids


def mean_log_income(
    age: np.ndarray, c: cal.Calibration = cal.COMPHS
) -> np.ndarray:
    """``ymean_``: deterministic component of log household income."""
    spouse, kids, depadul = household_composition(age, c)
    return (
        c.ywork_cons
        + c.ywork_agecoeff * age
        + c.ywork_age2coeff * age**2 / 100.0
        + c.ywork_age3coeff * age**3 / 10000.0
        + c.ywork_spousecoeff * spouse
        + c.ywork_kidscoeff * kids
        + c.ywork_depadulcoeff * depadul
    )


def mean_income(age: np.ndarray, c: cal.Calibration = cal.COMPHS) -> np.ndarray:
    """``Ymean_``: mean income in levels (lognormal correction applied)."""
    var_persistent = c.ywork_vareps / (1.0 - c.ywork_auto**2)
    return np.exp(mean_log_income(age, c) + 0.5 * (var_persistent + c.ywork_varnu))


def liquidation_penalty(age: np.ndarray) -> np.ndarray:
    """``zliqpen_``: proportional cost of withdrawing from the illiquid account.

    Falls from ~0.5 in youth toward 0 in retirement (their ``zilliq = 0`` case,
    a stand-in for early-withdrawal penalties on retirement accounts).
    """
    return 0.5 / (1.0 + np.exp((age - 50.0) / 10.0))


def credit_limit(
    age: np.ndarray, xjump: float, c: cal.Calibration = cal.COMPHS
) -> np.ndarray:
    """``xmin_``: borrowing limit in dollars, rounded to the ``xjump`` lattice.

    Returned as a POSITIVE magnitude: the floor on the liquid position is
    ``-credit_limit(...)``. Reading it as a signed bound is a real bug we have
    already made once (RESULTS.md 9.1).

    The limit is a quadratic in age as a *multiple of mean income*, scaled by
    ``Ymean_`` and snapped to the grid so it lands exactly on a grid point.
    """
    creditline = c.c0_credit + c.c1_credit * age + c.c2_credit * age**2
    return xjump * np.round(creditline * mean_income(age, c) / xjump)


def _nonuniform_positive_grid(
    jump: float, jump_max: float, top: float, cells_per_step: int
) -> np.ndarray:
    """Grid on ``[0, top]`` whose step doubles every ``cells_per_step`` cells.

    Mirrors their ``Xpos_`` / ``Z_`` construction: start at ``jump``, double
    after each block of ``cells_per_step`` cells, and once the step would
    exceed ``jump_max`` switch to a uniform tail at ``jump_max``.
    """
    out = [0.0]
    c = 1
    while out[-1] < top:
        step = jump * c
        if step < jump_max:
            out.extend(out[-1] + step * np.arange(1, cells_per_step + 1))
            c *= 2
        else:
            step = jump_max
            out.extend(np.arange(out[-1] + step, top + step / 2, step))
            break
    grid = np.array(out)
    return grid[grid <= top]


def liquid_grid(
    age: np.ndarray,
    xjump: float = 1000.0,
    xmax: float = 4e5,
    cells_per_step: int = X_CELLS_PER_STEP,
    nonlinear: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Common liquid-asset grid plus a per-age feasibility mask.

    Returns
    -------
    X : ndarray, shape ``(nX,)``
        Ascending grid from ``-max_t xmin_(t)`` to ``xmax``. The negative
        segment is uniform at ``xjump``; the non-negative segment is the
        nonuniform grid above (or uniform if ``nonlinear`` is False).
    feasible : ndarray, shape ``(n_ages, nX)``, dtype bool
        ``feasible[t, i]`` is True when ``X[i] >= -xmin_(t)``, i.e. the point
        is within that age's credit limit.
    """
    xmin = credit_limit(age, xjump)
    if nonlinear:
        pos = _nonuniform_positive_grid(xjump, XJUMP_MAX, xmax, cells_per_step)
    else:
        pos = np.arange(0.0, xmax + xjump / 2, xjump)
    neg = np.arange(-xmin.max(), -xjump / 2, xjump)
    X = np.concatenate([neg, pos])
    feasible = X[None, :] >= -xmin[:, None] - 1e-9
    return X, feasible


def illiquid_grid(
    zjump: float = 2000.0,
    zmax: float = 3.5e6,
    cells_per_step: int = Z_CELLS_PER_STEP,
    nonlinear: bool = True,
) -> np.ndarray:
    """``Z_``: illiquid-asset grid on ``[0, zmax]``, finest near zero."""
    if not nonlinear:
        return np.arange(0.0, zmax + zjump / 2, zjump)
    return _nonuniform_positive_grid(zjump, ZJUMP_MAX, zmax, cells_per_step)


def tauchen(
    psi: float | None = None,
    sigma_eps: float | None = None,
    n_states: int = cal.N_INCOME_STATES,
    m: float = cal.AR1_GRID_SPAN,
    c: cal.Calibration = cal.COMPHS,
) -> tuple[np.ndarray, np.ndarray]:
    """Tauchen (1986) discretization of the persistent AR(1) log-income component.

    Ported from their ``discretizeAR1``. Note this is *not* the Rouwenhorst
    method used by the Phase 1-2 HARK simulator: at ``n_states = 3`` the
    discretization choice materially changes the income process, so we match
    theirs rather than substitute our own.

    Returns ``(states, transition)`` where ``transition[j, k] = P(k | j)``.
    """
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2; got {n_states}")
    # Defaults come from the bundle, not from module globals bound at import
    # time, so passing ``c`` is enough to switch education group. An explicit
    # psi/sigma_eps still wins, which the nuisance-sweep scripts rely on.
    psi = c.ywork_auto if psi is None else psi
    sigma_eps = c.ywork_sigmaeps if sigma_eps is None else sigma_eps
    sd = np.sqrt(sigma_eps**2 / (1.0 - psi**2))
    states = np.linspace(-m * sd, m * sd, n_states)
    step = states[1] - states[0]

    z = (states[None, :] - psi * states[:, None]) / sigma_eps
    half = step / (2.0 * sigma_eps)
    upper = norm.cdf(z + half)
    lower = norm.cdf(z - half)
    P = upper - lower
    P[:, 0] = upper[:, 0]
    P[:, -1] = 1.0 - lower[:, -1]
    return states, P


def stationary(P: np.ndarray) -> np.ndarray:
    """Stationary distribution of a row-stochastic transition matrix."""
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    return v / v.sum()


def discretize_transitory(
    y0: float,
    sigma_nu: float | None = None,
    xjump: float = 1000.0,
    xmax: float = 4e5,
    xmin: float = 0.0,
    purge: float = 1e5,
    c: cal.Calibration = cal.COMPHS,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize the iid transitory shock onto the income lattice.

    Ported from their ``discretizeIID``. Support points are the *liquid-grid
    lattice in levels* (``xjump, 2*xjump, ...``), so realized income always
    lands on a representable cash-on-hand shift. Probabilities come from the
    lognormal density with a middle-Riemann-sum width weighting, after
    discarding points below ``sum(pdf) / purge``.

    Returns ``(probs, income_levels)`` with ``probs`` summing to 1.
    """
    sigma_nu = c.ywork_sigmanu if sigma_nu is None else sigma_nu
    lattice = np.arange(xjump, xmax + xmin + xjump / 2, xjump)
    log_y = np.log(lattice)
    pdf = norm.pdf(log_y, loc=y0, scale=sigma_nu)
    keep = pdf > pdf.sum() / purge
    if keep.sum() < 2:
        # Degenerate: mass concentrated on a single lattice point.
        i = int(np.argmax(pdf))
        return np.array([1.0]), np.array([lattice[i]])

    nu_log = log_y[keep]
    p = pdf[keep]
    dist = np.diff(nu_log)
    width = (
        np.concatenate([[dist[0]], dist]) + np.concatenate([dist, [dist[-1]]])
    ) / 2.0
    weighted = p * width
    return weighted / weighted.sum(), np.exp(nu_log)


if __name__ == "__main__":
    age = ages()
    X, feasible = liquid_grid(age)
    Z = illiquid_grid()
    states, P = tauchen()
    pi = stationary(P)

    xmin = credit_limit(age, 1000.0)
    assert X[0] == -xmin.max(), (X[0], -xmin.max())
    assert np.all(np.diff(X) > 0), "liquid grid must be strictly ascending"
    assert np.all(np.diff(Z) > 0), "illiquid grid must be strictly ascending"
    assert 0.0 in X and 0.0 in Z
    # Every age's own grid is an exact subset of the common grid.
    for t, xm in enumerate(xmin):
        assert feasible[t].sum() > 0
        assert np.isclose(X[feasible[t]][0], -xm), (t, X[feasible[t]][0], -xm)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    np.testing.assert_allclose(pi @ P, pi, atol=1e-12)

    p, y = discretize_transitory(mean_log_income(age)[10] + states[1])
    np.testing.assert_allclose(p.sum(), 1.0)

    print(f"ages          {age[0]:.0f}..{age[-1]:.0f}  ({len(age)})")
    print(f"liquid grid   nX={len(X)}  [{X[0]:,.0f}, {X[-1]:,.0f}]")
    print(f"illiquid grid nZ={len(Z)}  [{Z[0]:,.0f}, {Z[-1]:,.0f}]")
    print(f"credit limit  ${xmin.min():,.0f}..${xmin.max():,.0f}")
    print(f"liq penalty   {liquidation_penalty(age)[0]:.3f} at 20 → "
          f"{liquidation_penalty(age)[-1]:.4f} at 90")
    print(f"income states {np.round(states, 4)}  stationary {np.round(pi, 4)}")
    print(f"mean income   ${mean_income(age).min():,.0f}..${mean_income(age).max():,.0f}")
    print(f"transitory    {len(p)} support points")
    print("all grid self-checks passed")
