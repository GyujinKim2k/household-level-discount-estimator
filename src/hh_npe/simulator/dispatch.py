"""One-household trajectory generation, shared by dataset generation and SBC.

Both ``scripts/generate_dataset.py`` and ``scripts/run_sbc.py`` need the exact
same "theta in, wave tensor out" mapping -- SBC is only meaningful if its
simulator is bit-identical to the one that produced the training set. Keeping
the dispatch here rather than in either script guarantees that.
"""

from __future__ import annotations

import numpy as np

from hh_npe.data.waves import FEATURES_MVP, FEATURES_TWOASSET, aggregate_waves

AGE_START_SIM = 20
AGE_END_SIM = 90

#: Second word of the per-draw stream that picks initial wealth from a seed
#: pool. Its own stream, so income shocks (``seed_base + j``) are untouched.
INIT_POOL_STREAM = 20


def draw_initial_conditions(pool: np.ndarray, seed_base: int, j: int,
                            n_households: int, states: np.ndarray | None = None,
                            P: np.ndarray | None = None,
                            sigma_nu: float | None = None):
    """Draw ``j``'s households' age-20 starting point from a PSID ``pool``.

    Each household is one pool row, with replacement, keyed on the draw index
    like the income shocks, so the same draw gets the same households however
    draws are grouped or batched, and a resumed shard reproduces its seeds.

    A row is ``(liquid, illiquid)`` ratios to mean income (RESULTS 42.2), and
    optionally a third column ``e``: the person's log after-tax income minus
    the model's mean log income at that age (RESULTS 43). With ``e`` the
    household's initial persistent income state is drawn from its posterior
    given ``e`` under the model's own income process -- prior the stationary
    distribution of ``P``, likelihood ``e = state + transitory`` with the
    transitory sd ``sigma_nu`` -- so wealth and income come jointly from the
    same PSID person. Returns ``(wealth (n, 2), state (n,) or None)``.
    """
    rng = np.random.default_rng([seed_base + int(j), INIT_POOL_STREAM])
    rows = pool[rng.integers(0, len(pool), size=n_households)]
    if pool.shape[1] < 3:
        return rows[:, :2], None
    from hh_npe.simulator.grids import stationary

    e = rows[:, 2]
    logp = (np.log(stationary(P))[None, :]
            - 0.5 * ((e[:, None] - np.asarray(states)[None, :]) / sigma_nu) ** 2)
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    cdf = np.cumsum(p / p.sum(axis=1, keepdims=True), axis=1)
    u = rng.random(n_households)
    state = np.minimum((u[:, None] > cdf).sum(axis=1), len(states) - 1)
    return rows[:, :2], state


def draw_initial_wealth(pool: np.ndarray, seed_base: int, j: int,
                        n_households: int) -> np.ndarray:
    """The wealth part of :func:`draw_initial_conditions` (same stream)."""
    return draw_initial_conditions(pool[:, :2], seed_base, j, n_households)[0]


def simulate_one_hark(
    theta: np.ndarray, sim_seed: int, start_age: int, n_waves: int,
    wave_years: int, grid: str = "coarse",
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 1-2 path: HARK ``MarkovConsumerType``, no credit cards, beta = 1."""
    from hh_npe.simulator.forward import simulate_households
    from hh_npe.simulator.lifecycle import build_lifecycle_agent, solve_lifecycle

    delta, crra = float(theta[0]), float(theta[1])
    agent = build_lifecycle_agent(
        delta=delta, crra=crra,
        age_start=AGE_START_SIM, age_end=AGE_END_SIM, n_agents=1,
    )
    solve_lifecycle(agent)
    panel = simulate_households(agent, n_households=1, seed=sim_seed)
    x, alive = aggregate_waves(
        panel, age_start_sim=AGE_START_SIM, start_age=start_age,
        n_waves=n_waves, wave_years=wave_years, features=FEATURES_MVP,
    )
    return x[0], alive[0]


def simulate_one_twoasset(
    theta: np.ndarray, sim_seed: int, start_age: int, n_waves: int,
    wave_years: int, grid: str = "mid", return_panel: bool = False,
) -> tuple[np.ndarray, ...]:
    """Phase 3 path: credit cards, illiquid asset, naive quasi-hyperbolic beta.

    ``return_panel`` additionally returns the annual panel the waves were
    aggregated from. See :func:`simulate_batch_twoasset_gpu` for why keeping it
    is worth the bytes.
    """
    from hh_npe.simulator.twoasset import GRIDS, simulate, solve

    beta, delta, crra = (float(v) for v in theta)
    if grid not in GRIDS:
        raise ValueError(f"unknown grid {grid!r}; expected one of {sorted(GRIDS)}")
    spec = GRIDS[grid]
    sol = solve(beta, delta, crra, spec)
    panel = simulate(sol, n_households=1, seed=sim_seed)
    x, alive = aggregate_waves(
        panel, age_start_sim=AGE_START_SIM, start_age=start_age,
        n_waves=n_waves, wave_years=wave_years, features=FEATURES_TWOASSET,
    )
    if return_panel:
        return x[0], alive[0], panel
    return x[0], alive[0]


def simulate_batch_twoasset_gpu(
    thetas: np.ndarray, seed_base: int, start_age: int, n_waves: int,
    wave_years: int, grid: str = "full", theta_batch: int = 16, chunk: int = 16,
    return_panels: bool = False, n_households: int = 1,
    educ: np.ndarray | None = None, card: np.ndarray | None = None,
    r_gamma: np.ndarray | None = None, init_pool: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Phase 3 on the GPU: many draws per backward induction, one panel each.

    ``n_households`` simulates M households per solve instead of one. The
    forward pass is negligible against the ~12.4 s solve, so this is nearly
    free, and unlike the ``k`` window augmentation -- which reuses a single
    trajectory -- these are genuine independent draws from ``p(x | theta)``.
    Rows come out grouped by draw, ``M`` consecutive rows sharing one theta, so
    the grouped train/validation split must key on the **draw** index and not
    the row: two households of one theta on opposite sides of the split leak
    exactly as same-panel windows would.

    ``educ`` selects a per-draw education bundle by index into
    ``cal.EDUC_GROUPS``. Draws are grouped by bundle before solving, because
    ``solve_batch`` shares one calibration across a whole GPU batch -- mixing
    groups within a batch would silently solve them all as the first one.

    ``return_panels`` additionally returns the annual panels, stacked over
    draws, as a dict of ``(n_draws * n_households, T)`` arrays. Worth storing: the observation
    window (``start_age``, ``n_waves``, ``wave_years``) and the feature set are
    aggregation choices applied *after* the solve, which is ~99% of the cost.
    Keeping the panel makes any of them re-derivable without re-solving; keeping
    only ``x`` means a window change costs the whole run again.

    Lives here, beside the single-draw CPU paths, for the reason in the module
    docstring: SBC is only meaningful when its simulator is the one that made
    the training set. The GPU solve is reproducible at a fixed
    ``(device, theta_batch, chunk)`` but not across them -- ~99% of states hold
    two exactly-tied choices, and cuBLAS picks its summation order by problem
    size -- so those settings are arguments rather than defaults to be guessed,
    and callers should take them from the dataset's recorded configuration.

    ``card`` marks each draw as cardholder (1) or no card (0). A no-card draw
    is solved with its bundle's credit line set to zero, so it cannot borrow
    (RESULTS.md 24.1, 26). Draws are grouped by (education, card) pair, for the
    same reason as by education alone.

    ``r_gamma`` gives each draw its own illiquid return (RESULTS.md 32). The
    solver shares one consumption tensor across a batch, so draws are grouped
    by R_gamma value too; generation assigns it per block of theta_batch draws
    so each group is exactly one full batch. ``thetas`` stays (beta, delta, rho).

    ``init_pool`` is an ``(n, 2)`` array of age-20 (liquid, illiquid) ratios
    to mean income, or ``(n, 3)`` with each person's log income relative to
    the model's mean as well. Each household's start is drawn from it with
    replacement (:func:`draw_initial_conditions`) instead of the single SCF
    seed every household shares by default (RESULTS 42.2), and with three
    columns so is its initial persistent income state (RESULTS 43). ``None``
    leaves the forward pass exactly as before. The draws are a pure function of
    ``seed_base``, the draw index and the calibration, so callers that need
    them (to store per shard) recompute them the same way.

    The forward pass and wave aggregation stay on the CPU and are shared with
    :func:`simulate_one_twoasset` verbatim; only the solver differs.
    """
    import dataclasses

    from hh_npe.simulator import laibson_calibration as cal
    from hh_npe.simulator.twoasset import GRIDS, simulate
    import torch

    from hh_npe.simulator.twoasset_gpu import solve_batch

    if grid not in GRIDS:
        raise ValueError(f"unknown grid {grid!r}; expected one of {sorted(GRIDS)}")
    if n_households < 1:
        raise ValueError(f"n_households must be >= 1; got {n_households}")
    if educ is not None and len(educ) != len(thetas):
        raise ValueError(
            f"educ has {len(educ)} entries for {len(thetas)} thetas")
    if card is not None and len(card) != len(thetas):
        raise ValueError(
            f"card has {len(card)} entries for {len(thetas)} thetas")
    if r_gamma is not None and len(r_gamma) != len(thetas):
        raise ValueError(
            f"r_gamma has {len(r_gamma)} entries for {len(thetas)} thetas")
    if init_pool is not None:
        init_pool = np.asarray(init_pool, dtype=float)
        if (init_pool.ndim != 2 or init_pool.shape[1] not in (2, 3)
                or (init_pool[:, :2] < 0).any() or not np.isfinite(init_pool).all()):
            raise ValueError("init_pool must be (n, 2) non-negative (liquid, "
                             "illiquid) ratios to age-20 mean income, optionally "
                             "with a third column of log income deviations")
    thetas = np.asarray(thetas)[:, :3]

    # Consume each sub-batch before solving the next: a Solution holds ~58 MB of
    # policy arrays, so accumulating a whole large block would exhaust host RAM.
    # The panels are far smaller -- ~4 KB per draw -- so holding those is fine.
    n = len(thetas)
    # Results are written back by draw index, so grouping by education bundle
    # for the solve does not disturb the caller's ordering.
    xs: list[np.ndarray | None] = [None] * n
    alives: list[np.ndarray | None] = [None] * n
    panels: list[dict | None] = [None] * n
    e_arr = np.zeros(n, int) if educ is None else np.asarray(educ, int)
    c_arr = np.ones(n, int) if card is None else np.asarray(card, int)
    r_arr = (np.full(n, np.nan) if r_gamma is None
             else np.asarray(r_gamma, dtype=float))
    groups: dict = {}
    for j in range(n):
        key = (int(e_arr[j]), int(c_arr[j]),
               None if np.isnan(r_arr[j]) else float(r_arr[j]))
        groups.setdefault(key, []).append(j)
    groups = {k: np.array(v) for k, v in groups.items()}

    for (g, c, rg), idx in groups.items():
        # educ=None keeps GRIDS[grid]'s own calibration, exactly as before.
        calib = GRIDS[grid].calib if educ is None else cal.bundle(int(g))
        if c == 0:
            calib = dataclasses.replace(calib, c0_credit=0.0, c1_credit=0.0,
                                        c2_credit=0.0)
        spec = dataclasses.replace(GRIDS[grid], calib=calib)
        if rg is not None:
            spec = dataclasses.replace(spec, R_gamma=rg)
        for s0 in range(0, len(idx), theta_batch):
            sel = idx[s0:s0 + theta_batch]
            sols = solve_batch(thetas[sel], spec,
                               theta_batch=theta_batch, chunk=chunk)
            for j, sol in zip(sel, sols):
                # Seed off the draw index, not a running counter: identical
                # draws must give identical households however they are grouped.
                init, s0 = (None, None) if init_pool is None else \
                    draw_initial_conditions(init_pool, seed_base, j, n_households,
                                            sol.states, sol.P,
                                            spec.calib.ywork_sigmanu)
                panel = simulate(sol, n_households=n_households,
                                 seed=seed_base + int(j), initial_wealth=init,
                                 initial_state=s0)
                x, alive = aggregate_waves(
                    panel, age_start_sim=AGE_START_SIM, start_age=start_age,
                    n_waves=n_waves, wave_years=wave_years,
                    features=FEATURES_TWOASSET,
                )
                xs[j], alives[j] = x, alive
                if return_panels:
                    panels[j] = panel
            del sols
            # Card types and the fixed credit limit give cardholder and no-card
            # draws liquid grids of different sizes, so consecutive batches
            # alternate tensor shapes and PyTorch's cache fragments: an SBC run
            # died with 4.95 GB reserved but unallocated (RESULTS 26). Release
            # it after every batch -- negligible next to a ~200 s solve.
            torch.cuda.empty_cache()

    x_out = np.concatenate(xs) if n_households > 1 else np.stack([v[0] for v in xs])
    a_out = (np.concatenate(alives) if n_households > 1
             else np.stack([v[0] for v in alives]))
    if return_panels:
        merged = {k: np.concatenate([p[k] for p in panels]) for k in panels[0]}
        return x_out, a_out, merged
    return x_out, a_out


SIMULATORS = {"hark": simulate_one_hark, "twoasset": simulate_one_twoasset}

#: Feature set each simulator emits, for embedder sizing and sanity checks.
FEATURES_FOR = {"hark": FEATURES_MVP, "twoasset": FEATURES_TWOASSET}
