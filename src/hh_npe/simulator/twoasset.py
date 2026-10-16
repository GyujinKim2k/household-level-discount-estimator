"""Two-asset lifecycle model with credit cards and naive quasi-hyperbolic discounting.

Port of Laibson, Maxted, Repetto and Tobacman's ``LifecycleSim_BackwardInduct.m``
and ``LifecycleSim_ForwardIter.m``, baseline configuration only.

State per period: liquid assets ``X`` (may be negative down to an age-varying
credit limit, borrowed at ``R_CC``), illiquid assets ``Z >= 0`` (returns
``R_gamma``, withdrawals pay a proportional age-varying penalty), a persistent
income state ``S``, and age. The household picks next-period ``(X', Z')``
jointly by brute-force grid search -- no first-order conditions, no
interpolation of policies.

**Naive quasi-hyperbolic discounting.** The current self discounts the future
by ``beta * delta`` and acts on that (this is the simulated policy), but
believes every *future* self will discount by ``betahat * delta``. Their
benchmark sets ``betahat = 1``: a full naif who expects to behave
exponentially from tomorrow on. The value propagated backwards is therefore
evaluated at the *believed* future policy, not the actual one -- see
``_age_step`` where two separate argmaxes are taken.

Skipped from their code (robustness branches, none active in the benchmark):
``sqrtscale``, ``mortalityn``, ``split_income_at_retirement``,
``income_var_multiplier``, ``income_auto9``, ``zilliq``, the
``R_gamma_expectation`` perceived-return case, and the sophisticated
(``betahat = beta``) agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hh_npe.simulator import grids
from hh_npe.simulator import laibson_calibration as cal

NEG = -1e30  # sentinel for infeasible states; their code uses -1e100 in float64


@dataclass(frozen=True)
class ModelSpec:
    """Numerical and structural configuration.

    Grid granularity defaults reproduce Laibson et al. exactly (nX=190, nZ=84).
    Phase 3 dataset generation coarsens them for throughput -- see
    ``COARSE`` below and the grid-error check in ``scripts/validate_twoasset.py``.
    """

    xjump: float = 1000.0
    xmax: float = 4e5
    x_cells_per_step: int = grids.X_CELLS_PER_STEP
    zjump: float = 2000.0
    zmax: float = 3.5e6
    z_cells_per_step: int = grids.Z_CELLS_PER_STEP
    age_start: int = cal.AGE_START
    age_end: int = cal.AGE_END
    n_income_states: int = cal.N_INCOME_STATES
    R: float = cal.R_FREE
    R_gamma: float = cal.R_GAMMA
    R_CC: float = cal.R_CC
    alpha: float = cal.ALPHA_BEQUEST
    betahat: float = 1.0
    #: Education-group first-stage bundle. Defaults to ``comphs``, their
    #: benchmark and every result before Phase 4's regeneration. Frozen, so it
    #: is safe as a dataclass default. ``grids.py`` reads these at call time, so
    #: this field is the only way a draw can be solved as a different group.
    calib: cal.Calibration = cal.COMPHS
    #: Transitory income-disruption probability per period and the multiplier
    #: applied when it fires. Zero reproduces Laibson et al. exactly; see
    #: `grids.discretize_transitory` and RESULTS.md 12.4-12.5 for why the
    #: unmodified income process cannot generate a third of PSID households.
    disrupt_p: float = 0.0
    disrupt_mult: float = 0.27
    # float64 by default. The argmax must resolve utility gaps of ~1e-5 while
    # |EV| ~ 1e2; float32 cannot, and silently inflates borrowing. ``_age_step``
    # centres EV to buy back most of that headroom, but centred float32 still
    # differs from float64 by up to ~5% on the target moments, so float32 is
    # only for quick iteration -- never for a dataset or a validation run.
    dtype: type = np.float64
    chunk: int = 24  # current-X rows per vectorized block; caps peak memory
    #: Centre CRRA utility at the calibration's mean income per adult equivalent
    #: ``k``, as ``(x**(1-rho) - k**(1-rho)) / (1-rho)``, instead of at $1
    #: (RESULTS.md 40). Exact in real arithmetic -- a per-period constant never
    #: changes a choice -- and the log limit at rho = 1 is kept. Centred at $1,
    #: the constant ``-1 / (1 - rho)`` leaves the variable part below float64's
    #: resolution once rho >~ 4.2: choices tie, and ties break toward the most
    #: borrowing and the least illiquid wealth. False reproduces every dataset
    #: generated before 2026-10-05.
    crra_centred: bool = True
    #: Multiplies the age schedule of the illiquid liquidation penalty
    #: (``grids.liquidation_penalty``); 1 is Laibson et al.'s.
    liqpen_scale: float = 1.0
    #: Annual sd of a valuation shock on the illiquid balance (RESULTS 44). The
    #: balance chosen for next period, Z', arrives as Z' * g with
    #: log g ~ N(-sd^2/2, sd^2): mean one, so R_gamma stays the mean return,
    #: and the shock adds capital gains and losses that households plan for.
    #: On the grid it is the transition matrix of :func:`z_shock_matrix`. 0 is
    #: Laibson et al.'s model, where Z moves only by deposits and withdrawals,
    #: and leaves the solve and the simulation exactly as before.
    z_return_sd: float = 0.0
    #: Income levels in the expectation step rounded to whole dollars, the
    #: values the forward pass realises (RESULTS 54.6, H7). Laibson et al.'s
    #: levels are ``exp(log(k * xjump))``, an ulp off the lattice either way
    #: depending on the maths library; at cash-grid midpoints the plan then
    #: snaps one way and the realised path the other. False reproduces every
    #: dataset generated before 2026-10-16.
    round_income: bool = True
    #: MATLAB's infinities (RESULTS 54.2): u(0) = -inf for rho >= 1 (no
    #: clamp at 1e-9), an impossible choice ``u(max(c + dividend, 0)) - 1e100``,
    #: and -inf for dying in debt or choosing an unaffordable X'. A plan with
    #: any chance of a year at zero consumption is then never worth it, as in
    #: their ``_BackwardInduct.m``. False keeps the earlier finite sentinels.
    matlab_infinities: bool = True
    #: Which of exactly tied (X', Z') choices wins: ``"z_first"``, MATLAB's
    #: ``max`` over X' then over Z' (lowest Z', then lowest X'), or
    #: ``"x_first"``, the flat argmax earlier datasets used (lowest X' first).
    tie: str = "z_first"

    def __post_init__(self) -> None:
        if self.tie not in ("z_first", "x_first"):
            raise ValueError(f"tie must be 'z_first' or 'x_first'; got {self.tie!r}")


#: The solver settings of every dataset generated before 2026-10-16 (option
#: A2's v1 data included): ``dataclasses.replace(spec, **LEGACY_FIXES)``
#: reproduces them bit for bit.
LEGACY_FIXES = {"round_income": False, "matlab_infinities": False, "tie": "x_first"}


#: Gauss-Hermite nodes for the illiquid valuation shock (``z_shock_matrix``).
Z_SHOCK_NODES = 9


def z_shock_matrix(Z: np.ndarray, sd: float, n_nodes: int = Z_SHOCK_NODES) -> np.ndarray:
    """``M[a, b]``: probability that the illiquid balance chosen at grid point
    ``a`` arrives next period at grid point ``b`` (``ModelSpec.z_return_sd``).

    ``n_nodes`` Gauss-Hermite nodes of ``log g``, rescaled so the discrete mean
    of ``g`` is exactly 1; each ``Z[a] * g`` is then split between its two
    neighbouring grid points with linear weights. Below the top of the grid
    ``E[Z_next | a] = Z[a]`` exactly, so the shock adds risk and no drift. Zero
    stays zero (the shock is multiplicative). Gains beyond the last point are
    capped at it, so the mean drifts down near the top of the grid: on the
    full grid (top $3.47M) by 1-4% a year over the last five points at sd 0.1,
    and 8-12% at sd 0.3; below $1M by under 0.3% even at sd 0.5. The solve
    integrates over ``M`` and the simulation draws from it, so households face
    exactly the risk they plan for.
    """
    Z = np.asarray(Z, dtype=np.float64)
    nZ = len(Z)
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / w.sum()
    g = np.exp(sd * x)
    g = g / (w @ g)
    v = Z[:, None] * g[None, :]                                   # (nZ, nodes)
    top = v >= Z[-1]
    j = np.clip(np.searchsorted(Z, v, side="right") - 1, 0, nZ - 2)
    up = np.where(top, 1.0, (v - Z[j]) / (Z[j + 1] - Z[j]))
    j = np.where(top, nZ - 2, j)
    rows = np.broadcast_to(np.arange(nZ)[:, None], v.shape)
    M = np.zeros((nZ, nZ))
    np.add.at(M, (rows, j), w[None, :] * (1.0 - up))
    np.add.at(M, (rows, j + 1), w[None, :] * up)
    return M


def _z_shock_draw(zcdf: np.ndarray, zind: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Each household's illiquid grid index after the valuation shock: one
    draw from row ``zind`` of ``z_shock_matrix``'s cumulative sums ``zcdf``."""
    u = rng.random(len(zind))
    return np.minimum((u[:, None] > zcdf[zind]).sum(axis=1), zcdf.shape[1] - 1)


#: Coarsest grid (81x46). Cheap but carries 3.15 se of discretization error
#: against the full grid -- too much for a dataset meant to be comparable to
#: Laibson et al.'s estimates. Kept for quick smoke runs only.
COARSE = ModelSpec(xjump=4000.0, x_cells_per_step=16, zjump=16000.0, z_cells_per_step=5)

#: Working grid for Phase 3 dataset generation (107x56). Measured at 0.94 se of
#: discretization error against the full grid -- under one standard error of the
#: target moments -- at roughly 1/7 the cost. See scripts/validate_twoasset.py.
MID = ModelSpec(xjump=2000.0, x_cells_per_step=25, zjump=8000.0, z_cells_per_step=7)

#: Named grids accepted by ``--grid``. ``full`` is Laibson et al.'s exact 190x84.
GRIDS = {"coarse": COARSE, "mid": MID, "full": ModelSpec()}


@dataclass
class Solution:
    """Solved policy rules, indexed ``[t, ix, iz, is]``."""

    next_x: np.ndarray  # int32 index into X of chosen next-period liquid assets
    next_z: np.ndarray  # int32 index into Z of chosen next-period illiquid assets
    cons: np.ndarray  # float32 consumption excluding the illiquid dividend
    solvable: np.ndarray  # bool: at least one affordable (X', Z') exists
    X: np.ndarray
    Z: np.ndarray
    feasible: np.ndarray
    states: np.ndarray
    P: np.ndarray
    age: np.ndarray
    spec: ModelSpec
    prefs: tuple[float, float, float] = field(default=(1.0, 1.0, 1.0))


def _crra(c: np.ndarray, hhs: float, rho: float, ref: float = 1.0,
          floor: float = 1e-9) -> np.ndarray:
    """Per-period utility ``hhs * u(c / hhs)`` with CRRA ``rho``, centred at
    consumption ``ref`` per adult equivalent (``ModelSpec.crra_centred``).
    ``ref = 1`` is the textbook form and reproduces it bit for bit.
    ``floor = 0`` gives MATLAB's u(0): -inf for rho >= 1."""
    if floor == 0.0:
        with np.errstate(divide="ignore"):
            return _crra_raw(np.maximum(c, 0.0), hhs, rho, ref)
    return _crra_raw(np.maximum(c, floor), hhs, rho, ref)


def _crra_raw(c: np.ndarray, hhs: float, rho: float, ref: float) -> np.ndarray:
    if ref == 1.0:
        if abs(rho - 1.0) < 1e-9:
            return hhs * np.log(c / hhs)
        return hhs * ((c / hhs) ** (1.0 - rho) - 1.0) / (1.0 - rho)
    if abs(rho - 1.0) < 1e-9:
        return hhs * (np.log(c / hhs) - np.log(ref))
    return hhs * ((c / hhs) ** (1.0 - rho) - ref ** (1.0 - rho)) / (1.0 - rho)


def utility_ref(spec: "ModelSpec", mean_hhs: float, mean_hhy: float) -> float:
    """The consumption level per adult equivalent utility is centred at."""
    return float(mean_hhy / mean_hhs) if spec.crra_centred else 1.0


def _bequest_utility(
    X: np.ndarray, Z: np.ndarray, zliqpen: float, delta: float, rho: float,
    mean_hhs: float, mean_hhy: float, spec: ModelSpec,
) -> np.ndarray:
    """``beqUtil__``: utility from annuitizing the estate, relative to no estate."""
    estate = X[:, None] + Z[None, :] * (1.0 - zliqpen)
    annuity = max(spec.R - 1.0, 0.0) * np.maximum(estate, 0.0)
    ref = utility_ref(spec, mean_hhs, mean_hhy)
    baseline = _crra(np.array(mean_hhy), mean_hhs, rho, ref)
    bequest = _crra(mean_hhy + annuity, mean_hhs, rho, ref)
    return np.asarray(spec.alpha / (1.0 - delta) * (bequest - baseline), dtype=np.float64)


def _age_step(
    EV: np.ndarray, A: np.ndarray, B: np.ndarray, Z: np.ndarray,
    beta: float, delta: float, rho: float, betahat: float,
    hhs: float, R_gamma: float, spec: ModelSpec, ref: float = 1.0,
    first_allowed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One backward-induction step.

    ``A[i, j] = X[i] - cost of next-period liquid grid point j`` and
    ``B[a, b] = -(Z[b] - Z[a]) + zliqpen * min(Z[b] - Z[a], 0)`` are the
    separable liquid and illiquid components of the budget, so consumption is
    ``C[i, a, j, b] = A[i, j] + B[a, b]``.

    Returns ``(next_x, next_z, cons, V)`` where ``V`` is the continuation value
    the *previous* self expects: evaluated at the ``betahat`` policy when the
    agent is naive, at the actual policy when ``betahat == beta``.
    """
    nX, nZ, nS = EV.shape
    naive = abs(betahat - beta) > 1e-12

    next_x = np.empty((nX, nZ, nS), dtype=np.int32)
    next_z = np.empty((nX, nZ, nS), dtype=np.int32)
    cons = np.empty((nX, nZ, nS), dtype=spec.dtype)
    solvable = np.empty((nX, nZ, nS), dtype=bool)
    # The value function is always float64: it accumulates over 71 periods to
    # |V| ~ 1e2 while the differences that decide the argmax are ~1e-5.
    V = np.empty((nX, nZ, nS), dtype=np.float64)

    EV_flat = EV.reshape(nX * nZ, nS).astype(np.float64)
    dividend = ((R_gamma - 1.0) * Z).astype(spec.dtype)

    # Centre the continuation value before it enters the big (float32) tensor.
    # argmax is invariant to adding a constant, so this is *exact*, but it is
    # not optional: |EV| ~ 1e2 and the utility gap between adjacent grid points
    # is ~1e-5, which is below float32's resolution at that magnitude. Adding
    # EV uncentred makes the argmax select quantisation noise -- it inflated
    # simulated credit-card borrowing by ~50%. Centring restores ~7 digits.
    live = EV_flat > NEG / 2
    ev_ref = np.where(live.any(axis=0), np.where(live, EV_flat, -np.inf).max(axis=0), 0.0)

    inf_mode = spec.matlab_infinities
    for lo in range(0, nX, spec.chunk):
        hi = min(lo + spec.chunk, nX)
        # C[i, a, j, b] over this chunk of current liquid-asset rows.
        C = A[lo:hi, None, :, None] + B[None, :, None, :]
        u = _crra(C + dividend[None, :, None, None], hhs, rho, ref,
                  floor=0.0 if inf_mode else 1e-9).astype(spec.dtype)
        if inf_mode:
            u -= 1e100 * (C < 0.0)  # their ``- 1e100*ImpossibleC__``
        else:
            u[C < 0.0] = NEG  # cannot consume negative amounts
        C4 = C.reshape(hi - lo, nZ, nX, nZ)
        u4 = u.reshape(hi - lo, nZ, nX, nZ)
        u = u.reshape(hi - lo, nZ, nX * nZ)

        for s in range(nS):
            ev = EV_flat[:, s]
            ev_c = (ev - ev_ref[s]).astype(spec.dtype)
            payoff = u + (beta * delta) * ev_c[None, None, :]
            jx, jz = _choice(payoff, nX, nZ, spec.tie)
            rows = np.arange(hi - lo)[:, None]
            cols = np.arange(nZ)[None, :]
            best_pay = payoff.reshape(hi - lo, nZ, nX, nZ)[rows, cols, jx, jz]
            if inf_mode:
                # Every choice -inf: their max returns the first point of next
                # age's grid (lowest affordable X', Z' = 0). Unaffordable X'
                # are -inf here too and must not win.
                none = np.isneginf(best_pay)
                jx, jz = np.where(none, first_allowed, jx), np.where(none, 0, jz)
            next_x[lo:hi, :, s] = jx
            next_z[lo:hi, :, s] = jz
            cons[lo:hi, :, s] = C4[rows, cols, jx, jz]
            # A state is unsolvable when every (X', Z') is unaffordable; their
            # code lets ``max`` return an arbitrary index there. Such states are
            # unreachable in the forward pass, but must not be asserted over.
            # With MATLAB's infinities an impossible choice (-1e100) can beat a
            # possible one worth -inf, so affordability is checked directly.
            solvable[lo:hi, :, s] = ((cons[lo:hi, :, s] >= 0.0) & np.isfinite(best_pay)
                                     if inf_mode else best_pay > NEG / 2)

            if naive:
                # Re-optimize under the *believed* future discounting; the value
                # handed to the previous self is evaluated at that policy.
                bhat = u + (betahat * delta) * ev_c[None, None, :]
                jxh, jzh = _choice(bhat, nX, nZ, spec.tie)
                V[lo:hi, :, s] = (
                    u4[rows, cols, jxh, jzh].astype(np.float64)
                    + delta * ev.reshape(nX, nZ)[jxh, jzh]
                )
                del bhat
            else:
                V[lo:hi, :, s] = (
                    u4[rows, cols, jx, jz].astype(np.float64)
                    + delta * ev.reshape(nX, nZ)[jx, jz]
                )
            del payoff

        del C, C4, u, u4

    if not inf_mode:
        np.maximum(V, NEG, out=V)
    return next_x, next_z, cons, solvable, V


def _choice(payoff: np.ndarray, nX: int, nZ: int, tie: str) -> tuple[np.ndarray, np.ndarray]:
    """``(jx, jz)`` of the maximum over the last axis, flat ``jx * nZ + jz``.
    ``np.argmax`` returns the first maximum, so the axis order sets which of
    exactly tied choices wins: lowest X' first (``"x_first"``), or MATLAB's
    ``max`` then ``max``, lowest Z' first (``"z_first"``)."""
    if tie == "x_first":
        return np.divmod(payoff.argmax(axis=-1), nZ)
    lead = payoff.shape[:-1]
    k = payoff.reshape(*lead, nX, nZ).swapaxes(-1, -2).reshape(*lead, nZ * nX).argmax(axis=-1)
    jz, jx = np.divmod(k, nX)
    return jx, jz


def solve(
    beta: float,
    delta: float,
    rho: float,
    spec: ModelSpec = ModelSpec(),
) -> Solution:
    """Backward-induct the two-asset lifecycle model.

    Parameters
    ----------
    beta
        Present-bias parameter. ``1.0`` gives an exponential discounter.
    delta
        Long-run discount factor. Must be strictly below 1 (the bequest term
        divides by ``1 - delta``).
    rho
        Coefficient of relative risk aversion.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1); got {delta}")
    if beta <= 0.0 or rho <= 0.0:
        raise ValueError(f"beta and rho must be positive; got {beta}, {rho}")

    age = grids.ages(spec.age_start, spec.age_end)
    T = len(age)
    X, feasible = grids.liquid_grid(
        age, spec.xjump, spec.xmax, spec.x_cells_per_step, c=spec.calib
    )
    Z = grids.illiquid_grid(spec.zjump, spec.zmax, spec.z_cells_per_step)
    states, P = grids.tauchen(n_states=spec.n_income_states, c=spec.calib)
    nX, nZ, nS = len(X), len(Z), spec.n_income_states

    hhs = grids.effective_hh_size(age, spec.calib)
    ymean = grids.mean_log_income(age, spec.calib)
    ylevel = grids.mean_income(age, spec.calib)
    zliqpen = grids.liquidation_penalty(age) * spec.liqpen_scale
    xmin = grids.credit_limit(age, spec.xjump, spec.calib)
    death = cal.DEATH_PROB
    mean_hhs, mean_hhy = hhs.mean(), ylevel.mean()
    ref = utility_ref(spec, mean_hhs, mean_hhy)
    Mz = z_shock_matrix(Z, spec.z_return_sd) if spec.z_return_sd > 0 else None
    if spec.matlab_infinities and not (P > 0).all():
        raise FloatingPointError("a zero transition probability would make 0 * -inf")

    # Cost today of holding X[j] tomorrow: saved at R, borrowed at R_CC.
    cost_next = np.maximum(X / spec.R, 0.0) + np.minimum(X / spec.R_CC, 0.0)
    A_full = (X[:, None] - cost_next[None, :]).astype(spec.dtype)
    dZ = Z[None, :] - Z[:, None]

    next_x = np.empty((T, nX, nZ, nS), dtype=np.int32)
    next_z = np.empty((T, nX, nZ, nS), dtype=np.int32)
    cons = np.empty((T, nX, nZ, nS), dtype=spec.dtype)
    solvable = np.empty((T, nX, nZ, nS), dtype=bool)

    # Their code has no grid points below the credit limit at all and -inf for
    # dying in debt; the finite sentinel is the earlier datasets' convention.
    off = -np.inf if spec.matlab_infinities else NEG
    EV = None
    for t in range(T - 1, -1, -1):
        beq = _bequest_utility(
            X, Z, zliqpen[t], delta, rho, mean_hhs, mean_hhy, spec
        )
        if t == T - 1:
            # Terminal: continuation value is the bequest, and the no-Ponzi
            # condition forbids dying in debt.
            EV = np.repeat(beq[:, :, None], nS, axis=2)
            EV[X < 0.0, :, :] = off

        B = (-dZ + zliqpen[t] * np.minimum(dZ, 0.0)).astype(spec.dtype)
        A = A_full.copy()
        # Choices are restricted to the *next* period's affordable grid; at the
        # terminal age the grid is its own successor (their ``nextX_ = X_``).
        allowed = feasible[t + 1] if t < T - 1 else feasible[t]
        EV_t = EV.copy()
        EV_t[~allowed, :, :] = off

        nx, nz, c, ok, V = _age_step(
            EV_t, A, B, Z, beta, delta, rho, spec.betahat,
            float(hhs[t]), spec.R_gamma, spec, ref, int(np.flatnonzero(allowed)[0])
        )
        next_x[t], next_z[t], cons[t], solvable[t] = nx, nz, c, ok

        if t > 0:
            V[~feasible[t], :, :] = off
            if spec.matlab_infinities and np.isnan(V).any():
                raise FloatingPointError(f"NaN in V at age index {t}")
            EV = _expectation_step(
                V, X, ymean[t], states, P, spec, xmin[t], nX, nZ, nS, Mz
            )
            EV = (1.0 - death[t - 1]) * EV + death[t - 1] * beq[:, :, None]

    return Solution(
        next_x=next_x, next_z=next_z, cons=cons, solvable=solvable,
        X=X, Z=Z, feasible=feasible,
        states=states, P=P, age=age, spec=spec, prefs=(beta, delta, rho),
    )


def _expectation_step(
    V: np.ndarray, X: np.ndarray, ymean_t: float, states: np.ndarray,
    P: np.ndarray, spec: ModelSpec, xmin_t: float, nX: int, nZ: int, nS: int,
    Mz: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate ``V`` over transitory income and the persistent-state transition.

    Income arrives as a *level* added to liquid assets, so the expectation is a
    probability-weighted average of ``V`` evaluated at grid points shifted up by
    each possible income realization (nearest-neighbour on the grid, as in their
    ``griddedInterpolant(..., 'nearest')``).

    ``Mz`` (:func:`z_shock_matrix`) first integrates over where the chosen
    illiquid balance lands after its valuation shock. It acts on the Z axis and
    income on the X axis, so the order does not matter. ``None`` (no shock)
    leaves ``V`` untouched.
    """
    if Mz is not None:
        if spec.matlab_infinities:
            # The shock matrix has exact zeros and 0 * -inf is NaN: sum the
            # finite part, then -inf wherever a -inf is reached with weight.
            dead = np.isneginf(V)
            reach = np.einsum("ab,xbs->xas", (Mz > 0).astype(float), dead.astype(float))
            V = np.einsum("ab,xbs->xas", Mz, np.where(dead, 0.0, V))
            V[reach > 0] = -np.inf
        else:
            V = np.einsum("ab,xbs->xas", Mz, V)
    # V_shift[s2] = E_nu[ V(X + nu, Z, s2) ]
    V_shift = np.empty((nS, nX, nZ), dtype=np.float64)
    for s2 in range(nS):
        probs, levels = grids.discretize_transitory(
            float(ymean_t + states[s2]),
            xjump=spec.xjump, xmax=spec.xmax, xmin=xmin_t, c=spec.calib,
            disrupt_p=spec.disrupt_p, disrupt_mult=spec.disrupt_mult,
        )
        if spec.round_income:
            levels = np.round(levels)  # what the forward pass realises
        if spec.matlab_infinities and not (probs > 0).all():
            raise FloatingPointError("a zero income weight would make 0 * -inf")
        acc = np.zeros((nX, nZ), dtype=np.float64)
        for p, y in zip(probs, levels):
            acc += p * V[_nearest_index(X, X + y), :, s2]
        V_shift[s2] = acc

    # EV[:, :, s1] = sum_s2 P[s1, s2] * V_shift[s2]  (their EVprev___ loop).
    # P must be passed un-transposed: its *rows* are the conditional
    # distributions, and P.T is not even row-stochastic.
    return np.einsum("jk,kxz->xzj", P, V_shift)


def _nearest_index(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Index of the closest entry of ascending ``grid``, **ties rounding up**.

    Matches MATLAB's ``griddedInterpolant(..., 'nearest')``, which resolves an
    exact midpoint to the higher index. This is not a cosmetic convention.
    Income realizations live on the ``xjump`` lattice, but above $50k the liquid
    grid coarsens to 2000/4000/6000, so **26.9%** of cash-on-hand queries land
    exactly midway between two grid points. Rounding those down discards up to
    $6,000 of liquid wealth per snap -- only for wealthier households, and
    compounding with age. Because this function serves both the forward pass and
    the expectation step, rounding down also made saving look worse inside the
    DP itself. It inflated simulated credit-card borrowing by ~10%.
    """
    idx = np.searchsorted(grid, values)
    idx = np.clip(idx, 1, len(grid) - 1)
    left, right = grid[idx - 1], grid[idx]
    return np.where(values - left < right - values, idx - 1, idx)


def simulate(
    sol: Solution,
    n_households: int = 1,
    seed: int = 0,
    initial_wealth: np.ndarray | None = None,
    initial_state: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Forward-simulate households through the solved policy rules.

    Port of ``LifecycleSim_ForwardIter.m``. Households are seeded at the SCF
    median liquid/illiquid wealth-to-income ratios at age 20 and followed for
    the full lifecycle.

    ``initial_wealth`` overrides that seed with an ``(n_households, 2)`` array of
    ``(liquid, illiquid)`` ratios to age-20 mean income -- one row per household,
    so households no longer all start from the same point. Their default is a
    single SCF median for everybody, which means every simulated household
    begins life identically and all cross-sectional dispersion afterwards comes
    from income shocks alone (RESULTS.md 7.1). Pass the pair jointly: liquid and
    illiquid holdings are strongly dependent, and drawing them independently
    manufactures households that do not exist. Note their forward pass applies **no** mortality: death
    enters only through the backward induction and through the ``alive_``
    weights used when averaging moments, so no household is ever replaced.

    ``initial_state`` sets each household's age-20 persistent income state (an
    index into ``sol.states``) instead of drawing it from the stationary
    distribution. The uniform draw that would have picked it is still consumed,
    so every later shock is the one the household would otherwise have had
    (RESULTS 43).

    With ``spec.z_return_sd > 0`` the illiquid balance chosen each year is
    moved by its valuation shock before the next (:func:`z_shock_matrix`).
    Those draws come from their own stream, ``default_rng([seed, 30])``, so the
    income and state shocks are exactly those of the model without it.

    Returns panel arrays of shape ``(n_households, T)``:

    - ``income``           — realized labour income
    - ``consumption``      — total consumption, including the illiquid dividend
    - ``liquid_assets``    — liquid position *before* the income realization
      (their ``simL__``); negative values are credit-card debt
    - ``illiquid_assets``  — illiquid balance (their ``simZ__``)
    - ``cash_on_hand``     — liquid position after income (their ``simX__``)
    - ``income_state``     — index into ``sol.states``
    - ``t_age``            — 0..T-1, strictly increasing (no rebirth)
    """
    rng = np.random.default_rng(seed)
    spec = sol.spec
    X, Z, P = sol.X, sol.Z, sol.P
    T, nS = len(sol.age), spec.n_income_states
    N = n_households

    ymean = grids.mean_log_income(sol.age, spec.calib)
    ylevel = grids.mean_income(sol.age, spec.calib)
    xmin = grids.credit_limit(sol.age, spec.xjump, spec.calib)

    # --- persistent income state path ------------------------------------
    state_idx = np.empty((N, T), dtype=np.int64)
    cdf0 = np.cumsum(grids.stationary(P))
    u0 = rng.random(N)
    if initial_state is None:
        state_idx[:, 0] = np.searchsorted(cdf0, u0)
    else:
        s0 = np.asarray(initial_state, dtype=np.int64)
        if s0.shape != (N,) or s0.min() < 0 or s0.max() >= nS:
            raise ValueError(f"initial_state must be ({N},) indices into {nS} states")
        state_idx[:, 0] = s0
    cdf = np.cumsum(P, axis=1)
    for t in range(1, T):
        u = rng.random(N)
        state_idx[:, t] = (u[:, None] > cdf[state_idx[:, t - 1]]).sum(axis=1)
    np.clip(state_idx, 0, nS - 1, out=state_idx)

    # --- realized income --------------------------------------------------
    income = np.empty((N, T))
    for t in range(T):
        u = rng.random(N)
        for s in range(nS):
            rows = np.flatnonzero(state_idx[:, t] == s)
            if rows.size == 0:
                continue
            probs, levels = grids.discretize_transitory(
                float(ymean[t] + sol.states[s]),
                xjump=spec.xjump, xmax=spec.xmax, xmin=xmin[t], c=spec.calib,
                disrupt_p=spec.disrupt_p, disrupt_mult=spec.disrupt_mult,
            )
            draw = np.searchsorted(np.cumsum(probs), u[rows])
            income[rows, t] = levels[np.clip(draw, 0, len(levels) - 1)]
    income = np.round(income)

    # --- initial wealth (SCF medians, as multiples of age-20 mean income) --
    y0 = ylevel[0]
    if initial_wealth is not None:
        w = np.asarray(initial_wealth, dtype=float)
        if w.shape != (N, 2):
            raise ValueError(
                f"initial_wealth must have shape ({N}, 2) of (liquid, illiquid) "
                f"ratios to age-20 mean income; got {w.shape}"
            )
        # Clamp to what is feasible at age 20, NOT to zero. The default seed
        # is clamped non-negative below (`max(ix0, argmin|X|)`), but a drawn
        # household may legitimately start in credit-card debt -- `somehs`'s own
        # SCF median liquid ratio is -0.037 -- and forcing those to zero would
        # erase exactly the variation being added.
        #
        # `grids.credit_limit` returns the borrowing limit as a POSITIVE
        # magnitude in dollars (its docstring; the solver passes it to
        # `discretize_transitory` as an offset, never as a signed bound), so the
        # floor on the liquid position is -xmin[0]. Using +xmin[0] here forces
        # every drawn household to start at or above +5,000 -- silently erasing
        # the indebted households this override exists to create.
        x0 = np.clip(w[:, 0] * y0, -xmin[0], X[-1])
        xind = _nearest_index(X, x0).astype(np.int64)
        zind = _nearest_index(Z, np.clip(w[:, 1] * y0, 0.0, Z[-1])).astype(np.int64)
    iz = int(_nearest_index(Z, np.array(
        [(spec.calib.med_total_wealth - spec.calib.med_liq_wealth) * y0]))[0])
    ix0 = int(_nearest_index(X, np.array([spec.calib.med_liq_wealth * y0]))[0])
    ix0 = max(ix0, int(np.argmin(np.abs(X))))

    # --- decisions --------------------------------------------------------
    if initial_wealth is None:
        xind = np.full(N, ix0, dtype=np.int64)
        zind = np.full(N, iz, dtype=np.int64)
    cash = np.empty((N, T))
    illiquid = np.empty((N, T))
    cons = np.empty((N, T))
    zcdf = (np.cumsum(z_shock_matrix(Z, spec.z_return_sd), axis=1)
            if spec.z_return_sd > 0 else None)
    rng_z = np.random.default_rng([seed, 30]) if zcdf is not None else None

    for t in range(T):
        xind = _nearest_index(X, X[xind] + income[:, t])
        s = state_idx[:, t]
        cons[:, t] = sol.cons[t, xind, zind, s]
        cash[:, t] = X[xind]
        illiquid[:, t] = Z[zind]
        if t < T - 1:
            nxt_x = sol.next_x[t, xind, zind, s]
            nxt_z = sol.next_z[t, xind, zind, s]
            xind, zind = nxt_x.astype(np.int64), nxt_z.astype(np.int64)
            if zcdf is not None:
                zind = _z_shock_draw(zcdf, zind, rng_z)

    cons = cons + (spec.R_gamma - 1.0) * illiquid

    return {
        "income": income,
        "consumption": cons,
        "liquid_assets": cash - income,
        "illiquid_assets": illiquid,
        "cash_on_hand": cash,
        "income_state": state_idx,
        "t_age": np.tile(np.arange(T), (N, 1)),
    }


def simulate_from(
    sol: Solution,
    t0: np.ndarray,
    liquid0: np.ndarray,
    illiquid0: np.ndarray,
    income0: np.ndarray,
    horizon: int,
    n_sims: int = 1,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Forward-simulate households from an OBSERVED state, mid-life.

    :func:`simulate` always starts at age 20 from an endowment. Out-of-sample
    prediction needs the opposite: take each household where the data leave it
    -- its own liquid and illiquid wealth and income at period ``t0`` -- and run
    the solved policy forward ``horizon`` years.

    Inputs are one entry per household (length ``H``); each is simulated
    ``n_sims`` times with independent future shocks. ``liquid0`` is the position
    *before* period-``t0`` income, matching ``simulate``'s ``liquid_assets``;
    ``income0`` is period-``t0`` income, which is observed and so is used as is.
    The persistent income state at ``t0`` is the Tauchen state nearest to
    ``log(income0) - mean_log_income[t0]``; the transitory part is not separable
    from one observation, so that is an approximation, and the same one for
    every parameter value being compared.

    Shocks depend only on ``seed`` and the income process, never on ``sol``'s
    preferences, so two solutions simulated with the same seed face identical
    income paths -- which is what makes a paired comparison across parameter
    values possible. The illiquid valuation shock (``spec.z_return_sd > 0``)
    draws its uniforms from its own stream, ``default_rng([seed, 30])``, so
    they are shared across parameter values too.

    Returns arrays of shape ``(H, n_sims, horizon + 1)``, column 0 being period
    ``t0`` itself.
    """
    rng = np.random.default_rng(seed)
    spec = sol.spec
    X, Z, P = sol.X, sol.Z, sol.P
    nS = spec.n_income_states
    t0 = np.asarray(t0, dtype=np.int64)
    H = len(t0)
    if t0.max() + horizon >= len(sol.age):
        raise ValueError("horizon runs past the end of the lifecycle")
    ymean = grids.mean_log_income(sol.age, spec.calib)
    xmin = grids.credit_limit(sol.age, spec.xjump, spec.calib)

    rep = lambda a: np.repeat(np.asarray(a, dtype=float), n_sims)  # noqa: E731
    tt = np.repeat(t0, n_sims)
    N = H * n_sims
    inc0 = rep(income0)
    s = np.argmin(np.abs((np.log(np.maximum(inc0, 1.0)) - ymean[tt])[:, None]
                         - sol.states[None, :]), axis=1)
    xind = _nearest_index(X, np.clip(rep(liquid0), -xmin[tt], X[-1])).astype(np.int64)
    zind = _nearest_index(Z, np.clip(rep(illiquid0), 0.0, Z[-1])).astype(np.int64)

    out = {k: np.empty((N, horizon + 1)) for k in
           ("income", "consumption", "liquid_assets", "illiquid_assets")}
    cdf = np.cumsum(P, axis=1)
    zcdf = (np.cumsum(z_shock_matrix(Z, spec.z_return_sd), axis=1)
            if spec.z_return_sd > 0 else None)
    rng_z = np.random.default_rng([seed, 30]) if zcdf is not None else None
    for h in range(horizon + 1):
        t = tt + h
        if h == 0:
            inc = inc0
        else:
            s = np.clip((rng.random(N)[:, None] > cdf[s]).sum(axis=1), 0, nS - 1)
            inc = np.empty(N)
            u = rng.random(N)
            for key in np.unique(np.stack([t, s], 1), axis=0):
                rows = np.flatnonzero((t == key[0]) & (s == key[1]))
                probs, levels = grids.discretize_transitory(
                    float(ymean[key[0]] + sol.states[key[1]]),
                    xjump=spec.xjump, xmax=spec.xmax, xmin=xmin[key[0]],
                    c=spec.calib, disrupt_p=spec.disrupt_p,
                    disrupt_mult=spec.disrupt_mult)
                d = np.searchsorted(np.cumsum(probs), u[rows])
                inc[rows] = levels[np.clip(d, 0, len(levels) - 1)]
            inc = np.round(inc)
        xind = _nearest_index(X, X[xind] + inc).astype(np.int64)
        # Same definition as `simulate`'s liquid_assets -- post-income cash on
        # the grid, minus income -- which is what the network was trained on.
        # Reporting the pre-income grid point instead differs by up to one
        # xjump ($2,000).
        out["liquid_assets"][:, h] = X[xind] - inc
        out["income"][:, h] = inc
        out["illiquid_assets"][:, h] = Z[zind]
        out["consumption"][:, h] = (sol.cons[t, xind, zind, s]
                                    + (spec.R_gamma - 1.0) * Z[zind])
        nx = sol.next_x[t, xind, zind, s].astype(np.int64)
        nz = sol.next_z[t, xind, zind, s].astype(np.int64)
        xind, zind = nx, nz
        if zcdf is not None and h < horizon:
            zind = _z_shock_draw(zcdf, zind, rng_z)
    return {k: v.reshape(H, n_sims, horizon + 1) for k, v in out.items()}
