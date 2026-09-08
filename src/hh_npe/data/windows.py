"""Random-age observation windows cut from the stored annual panels.

Phase 3 originally trained on one fixed window per household -- 10 biennial
waves, ages 30-49. That is convenient and wrong in two ways.

It does not match PSID, where households appear at whatever ages they happened
to be during the survey years. Requiring a balanced ages-30-49 panel collapses
the eligible birth cohorts (see SIMULATOR_SPEC 6.1), and a posterior trained
only on ages 30-49 is mis-specified for a household observed at any other age.

And it wastes the expensive part. Each panel costs ~12.4 s of GPU backward
induction and spans ages 20-90; one 20-year window uses under a third of it.

Both are fixed here rather than in the simulator, because the window is an
aggregation choice applied *after* the solve. Shards already store the annual
panel, so nothing is re-solved and no dataset is regenerated.

The generative model this implements is::

    theta ~ prior
    panel ~ p(. | theta)          # one GPU solve, ages 20-90
    s     ~ Uniform{start_low..start_high}
    x     = window(panel, s)

Every ``(theta, x)`` pair is a valid draw from that joint, so NPE stays
consistent. Drawing several ``s`` per panel makes the pairs *correlated*, not
invalid -- but it means the effective independent sample size is the number of
panels, not the number of examples. :func:`build_windowed` returns ``panel_id``
so downstream code can respect that; training must split by panel, not by row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hh_npe.data.waves import (
    FEATURES_TWOASSET,
    FEATURES_TWOASSET_AGE,
    aggregate_waves,
)

#: The simulator's first period, mirroring ``dispatch.AGE_START_SIM``. Imported
#: from there would be circular; the two are asserted equal in the tests.
AGE_START_SIM = 20

PANEL_PREFIX = "panel_"


def sample_start_ages(
    n_panels: int, k: int, low: int, high: int, seed: int = 0
) -> np.ndarray:
    """``(n_panels, k)`` start ages, drawn without replacement within a panel.

    Without replacement because a panel contributing the same window twice adds
    an exact duplicate row -- pure weight on one draw, no new information about
    the age mapping, which is the only thing augmentation is here to teach.
    """
    n_choices = high - low + 1
    if k > n_choices:
        raise ValueError(
            f"cannot draw {k} distinct start ages from {low}..{high} "
            f"({n_choices} available); lower windows_per_panel or widen the range"
        )
    rng = np.random.default_rng(seed)
    choices = np.arange(low, high + 1)
    # Argsort of uniforms gives an independent permutation per row, vectorized.
    order = rng.random((n_panels, n_choices)).argsort(axis=1)
    return choices[order[:, :k]]


def add_age(panel: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Attach the ``age`` series, derived from ``t_age``.

    Derived rather than stored: ``t_age`` is already in every shard and in
    every panel the GPU simulator returns, so the age channel costs no bytes on
    disk and no regeneration.
    """
    if "t_age" in panel and "age" not in panel:
        panel = dict(panel, age=AGE_START_SIM + panel["t_age"])
    return panel


def _panel_of(shard) -> dict[str, np.ndarray]:
    panel = {k[len(PANEL_PREFIX):]: shard[k] for k in shard.files
             if k.startswith(PANEL_PREFIX)}
    return add_age(panel) if panel else panel


def cut_windows(
    panel: dict[str, np.ndarray],
    theta_rows: np.ndarray,
    id_rows: np.ndarray,
    starts: np.ndarray,
    n_waves: int,
    wave_years: int,
    features: tuple[str, ...],
    flow_agg: str = "last",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut the windows named by ``starts`` (shape ``(n_rows, k)``) out of a panel.

    Shared by the shard path and by SBC, so a calibration window is cut exactly
    the way a training window was. Rows are grouped by start age so
    ``aggregate_waves`` is called once per distinct age rather than once per
    household.
    """
    thetas, xs, ids = [], [], []
    for j in range(starts.shape[1]):
        for s in np.unique(starts[:, j]):
            rows = np.flatnonzero(starts[:, j] == s)
            sub = {key: v[rows] for key, v in panel.items()}
            x, alive = aggregate_waves(
                sub, age_start_sim=AGE_START_SIM, start_age=int(s),
                n_waves=n_waves, wave_years=wave_years, features=features,
                flow_agg=flow_agg,
            )
            keep = alive.all(axis=1)
            thetas.append(theta_rows[rows][keep])
            xs.append(x[keep])
            ids.append(id_rows[rows][keep])
    return np.concatenate(thetas), np.concatenate(xs), np.concatenate(ids)


def window_panel(
    panel: dict[str, np.ndarray],
    theta: np.ndarray,
    *,
    start_low: int = 25,
    start_high: int = 40,
    k: int = 1,
    n_waves: int = 10,
    wave_years: int = 2,
    seed: int = 0,
    with_age: bool = True,
    flow_agg: str = "last",
    features: tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random-age windows from an in-memory panel (e.g. fresh SBC simulations)."""
    panel = add_age(panel)
    n_rows = len(theta)
    starts = sample_start_ages(n_rows, k, start_low, start_high, seed=seed)
    if features is None:
        features = FEATURES_TWOASSET_AGE if with_age else FEATURES_TWOASSET
    th, x, ids = cut_windows(panel, np.asarray(theta), np.arange(n_rows),
                             starts, n_waves, wave_years, features, flow_agg)
    return (torch.from_numpy(th).float(), torch.from_numpy(x).float(),
            torch.from_numpy(ids).long())


def max_start_age(n_waves: int, wave_years: int, n_periods: int = 71) -> int:
    """Latest start age whose window still fits inside the simulated panel."""
    return AGE_START_SIM + n_periods - wave_years * n_waves


def build_windowed(
    shard_files: list[Path],
    theta_all: np.ndarray,
    *,
    start_low: int = 25,
    start_high: int = 40,
    k: int = 8,
    n_waves: int = 10,
    wave_years: int = 2,
    seed: int = 0,
    with_age: bool = True,
    fixed_start: int | None = None,
    flow_agg: str = "last",
    features: tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cut ``k`` windows from every panel in ``shard_files``.

    ``fixed_start`` reproduces the old behaviour -- one window per panel at that
    age -- so the baseline and the random-age variants come from one code path
    and differ only in the window, which is the point of the comparison.

    Returns ``(theta, x, panel_id)``. ``panel_id`` is the global Sobol draw
    index, so it identifies a household across shards and across windows.
    """
    if features is None:
        features = FEATURES_TWOASSET_AGE if with_age else FEATURES_TWOASSET
    limit = max_start_age(n_waves, wave_years)
    if fixed_start is None and start_high > limit:
        raise ValueError(
            f"start_age {start_high} with {n_waves} waves of {wave_years}y runs "
            f"past the simulated panel; the latest usable start age is {limit}"
        )

    thetas, xs, ids = [], [], []
    for sf in shard_files:
        d = np.load(sf)
        lo, hi = int(d["lo"]), int(d["hi"])
        panel = _panel_of(d)
        if not panel:
            raise SystemExit(
                f"{sf.name} stores no annual panel, so windows cannot be cut "
                f"from it. Only runs generated after the panel change qualify."
            )
        n_draws = hi - lo
        # M households per solve: the panel holds n_draws * M rows, M
        # consecutive rows sharing one theta. Recovered from the row count
        # rather than a stored field so pre-M shards, which have neither, keep
        # working unchanged.
        rows = len(next(iter(panel.values())))
        m = rows // n_draws
        if rows != n_draws * m:
            raise SystemExit(
                f"{sf.name} holds {rows} panel rows for {n_draws} draws, which "
                f"is not a whole number of households per draw."
            )
        n_panels = rows
        if fixed_start is not None:
            starts = np.full((n_panels, 1), fixed_start)
        else:
            # Seed by shard so a shard's windows do not depend on how many
            # shards precede it -- a partial run and a full run agree.
            starts = sample_start_ages(n_panels, k, start_low, start_high,
                                       seed=seed + lo)

        # theta repeats per household, but the id stays the DRAW index. This is
        # load-bearing: `_use_grouped_split` keys the train/validation split on
        # this id, and two households sharing a theta on opposite sides of the
        # split leak exactly as two windows of one panel would.
        th_draw = np.repeat(theta_all[lo:hi], m, axis=0)
        ids_draw = np.repeat(lo + np.arange(n_draws), m)
        th, x, ids_ = cut_windows(panel, th_draw, ids_draw,
                                  starts, n_waves, wave_years, features, flow_agg)
        thetas.append(th)
        xs.append(x)
        ids.append(ids_)

    return (
        torch.from_numpy(np.concatenate(thetas)).float(),
        torch.from_numpy(np.concatenate(xs)).float(),
        torch.from_numpy(np.concatenate(ids)).long(),
    )


def save_windowed(
    theta: torch.Tensor, x: torch.Tensor, panel_id: torch.Tensor,
    meta: dict, path: str | Path,
) -> None:
    """Persist a windowed dataset.

    Deliberately not :func:`hh_npe.data.dataset.save_dataset`: that function's
    two-tensor contract is relied on by ``run_sbc.py`` and the Phase 2
    artifacts, and a windowed dataset carries ``panel_id`` that callers must
    not silently drop -- doing so is what produces a leaked validation split.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"theta": theta.float(), "x": x.float(),
                "panel_id": panel_id.long(), "meta": meta}, p)


def load_windowed(path: str | Path) -> dict:
    """Load a windowed dataset saved by :func:`save_windowed`."""
    return torch.load(Path(path), weights_only=False)
