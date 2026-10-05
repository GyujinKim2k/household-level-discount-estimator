"""Age-20 starting wealth drawn from a PSID seed pool (RESULTS 42-43).

Each failure here would be silent in a multi-day run:

* the pool consuming the income stream, so turning it on also changes every
  household's income path and not only its start;
* households starting in card debt, which the pool is built to exclude;
* a draw's households depending on how draws were grouped into batches, so a
  resumed or regrouped run describes different households;
* the starts not reaching the shard, or a resume under a different pool
  mixing two simulators.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from hh_npe.simulator import grids
from hh_npe.simulator.dispatch import draw_initial_wealth

POOL = np.array([[0.0, 0.0], [0.1, 0.2], [0.0, 1.3], [0.3, 0.05], [0.02, 0.7]])


def test_draws_are_keyed_on_the_draw_and_come_from_the_pool():
    a = draw_initial_wealth(POOL, 100, 3, 50)
    np.testing.assert_array_equal(a, draw_initial_wealth(POOL, 100, 3, 50))
    assert not np.array_equal(a, draw_initial_wealth(POOL, 100, 4, 50))
    # seed_base + j is the key, so (100, 3) and (101, 2) are the same draw.
    np.testing.assert_array_equal(a, draw_initial_wealth(POOL, 101, 2, 50))
    assert {tuple(r) for r in a} <= {tuple(r) for r in POOL}


torch = pytest.importorskip("torch")
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

THETAS = np.array([[0.7, 0.98, 2.0], [0.9, 0.99, 3.5], [0.6, 0.97, 1.5]])
M = 20
KW = dict(seed_base=5, start_age=30, n_waves=5, wave_years=2, grid="coarse",
          theta_batch=4, chunk=8, return_panels=True, n_households=M)


@gpu
def test_no_pool_is_the_default_path():
    from hh_npe.simulator.dispatch import simulate_batch_twoasset_gpu

    _x0, _a0, p0 = simulate_batch_twoasset_gpu(THETAS, **KW)
    _x1, _a1, p1 = simulate_batch_twoasset_gpu(THETAS, init_pool=None, **KW)
    for k in p0:
        np.testing.assert_array_equal(p0[k], p1[k])


@gpu
def test_pool_moves_the_start_and_leaves_income_alone():
    from hh_npe.simulator.dispatch import simulate_batch_twoasset_gpu
    from hh_npe.simulator.twoasset import GRIDS

    _x0, _a0, p0 = simulate_batch_twoasset_gpu(THETAS, **KW)
    _x1, _a1, p1 = simulate_batch_twoasset_gpu(THETAS, init_pool=POOL, **KW)
    np.testing.assert_array_equal(p0["income"], p1["income"])
    np.testing.assert_array_equal(p0["income_state"], p1["income_state"])
    assert not np.array_equal(p0["illiquid_assets"], p1["illiquid_assets"])

    spec = GRIDS["coarse"]
    y0 = grids.mean_income(np.array([20.0]))[0]
    Z = grids.illiquid_grid(spec.zjump, spec.zmax, spec.z_cells_per_step)
    for j in range(len(THETAS)):
        w = draw_initial_wealth(POOL, 5, j, M)
        rows = slice(j * M, (j + 1) * M)
        # Liquid at age 20 is the start on the lattice: never debt, and within
        # one grid step of the drawn ratio.
        liq0 = p1["liquid_assets"][rows, 0]
        assert (liq0 >= 0).all()
        assert np.abs(liq0 - w[:, 0] * y0).max() <= spec.xjump
        z_expected = Z[np.abs(Z[None, :] - (w[:, 1] * y0)[:, None]).argmin(1)]
        np.testing.assert_array_equal(p1["illiquid_assets"][rows, 0], z_expected)


@gpu
def test_a_draws_households_do_not_depend_on_its_batch():
    from hh_npe.simulator.dispatch import simulate_batch_twoasset_gpu

    # Card types regroup the solves: draw 1 moves to its own batch.
    _x, _a, mixed = simulate_batch_twoasset_gpu(
        THETAS, card=np.array([1, 0, 1]), init_pool=POOL, **KW)
    _x, _a, same = simulate_batch_twoasset_gpu(
        THETAS, card=np.ones(3, int), init_pool=POOL, **KW)
    for j in (0, 2):
        rows = slice(j * M, (j + 1) * M)
        for k in ("liquid_assets", "illiquid_assets", "consumption"):
            np.testing.assert_array_equal(mixed[k][rows], same[k][rows])


def test_pool_must_be_non_negative_pairs():
    from hh_npe.simulator.dispatch import simulate_batch_twoasset_gpu

    with pytest.raises(ValueError, match="non-negative"):
        simulate_batch_twoasset_gpu(THETAS, init_pool=np.array([[-0.1, 0.2]]), **KW)


@gpu
def test_generation_stores_starts_and_refuses_another_pool(tmp_path):
    """End to end through generate_dataset on the coarse grid."""
    pool_a = tmp_path / "a.npz"
    pool_b = tmp_path / "b.npz"
    np.savez(pool_a, pool=POOL)
    np.savez(pool_b, pool=POOL[::-1].copy())
    out = tmp_path / "d.pt"
    base = [sys.executable, "scripts/generate_dataset.py", "--simulator", "twoasset",
            "--grid", "coarse", "--device", "cuda", "--n_samples", "32",
            "--block", "16", "--theta_batch", "16", "--chunk", "8",
            "--n_households", "4", "--card_types", "--rgamma_range", "1.025", "1.075",
            "--seed", "7", "--out", str(out)]
    env = {**os.environ, "PYTHONPATH": "."}
    pool_c = tmp_path / "c.npz"
    np.savez(pool_c, pool=np.column_stack([POOL, [-1.5, 0.0, 1.5, 0.3, -0.2]]))
    out_c = tmp_path / "c.pt"
    r = subprocess.run(base[:-1] + [str(out_c), "--init_pool", str(pool_c),
                                    "--max_draws", "16"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    sc = np.load(sorted((tmp_path / "c_shards").glob("shard_*.npz"))[0])
    assert sc["init_state"].shape == (16 * 4,)
    np.testing.assert_array_equal(sc["init_state"], sc["panel_income_state"][:, 0])

    r = subprocess.run(base + ["--init_pool", str(pool_a), "--max_draws", "16"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    shards = sorted((tmp_path / "d_shards").glob("shard_*.npz"))
    assert len(shards) == 1                      # --max_draws stopped after one
    s = np.load(shards[0])
    assert s["init_wealth"].shape == (16 * 4, 2)
    expected = np.concatenate([draw_initial_wealth(POOL, 7 + 0 + 1, j, 4)
                               for j in range(16)])
    np.testing.assert_array_equal(s["init_wealth"], expected)
    cfg = json.loads((tmp_path / "d_shards" / "solver_config.json").read_text())
    assert cfg["init_pool_n"] == len(POOL) and len(cfg["init_pool_sha256"]) == 64

    r = subprocess.run(base + ["--init_pool", str(pool_b)], capture_output=True,
                       text=True, env=env)
    assert r.returncode != 0 and "init_pool_sha256" in (r.stdout + r.stderr)
    r = subprocess.run(base, capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "init_pool" in (r.stdout + r.stderr)


# --- initial income state (RESULTS 43) ----------------------------------------

def _income_process():
    from hh_npe.simulator import laibson_calibration as cal

    states, P = grids.tauchen(c=cal.COMPHS)
    return states, P, cal.COMPHS.ywork_sigmanu


def test_income_column_sets_the_state_and_keeps_the_wealth_stream():
    from hh_npe.simulator.dispatch import draw_initial_conditions

    states, P, sig = _income_process()
    pool3 = np.column_stack([POOL, np.linspace(-2.0, 2.0, len(POOL))])
    w, s = draw_initial_conditions(pool3, 100, 3, 500, states, P, sig)
    # The wealth part is the same stream as the wealth-only pool.
    np.testing.assert_array_equal(w, draw_initial_wealth(POOL, 100, 3, 500))
    w2, s2 = draw_initial_conditions(pool3, 100, 3, 500, states, P, sig)
    np.testing.assert_array_equal(s, s2)
    # Far below / above the mean picks the bottom / top state.
    rows = np.random.default_rng([103, 20]).integers(0, len(POOL), size=500)
    e = pool3[rows, 2]
    assert (s[e <= -2.0] == 0).all() and (s[e >= 2.0] == len(states) - 1).all()
    assert draw_initial_conditions(POOL, 100, 3, 5)[1] is None


def test_initial_state_only_replaces_the_age_20_state():
    """Passing the state the default draw would have picked changes nothing,
    so every later shock is untouched."""
    from hh_npe.simulator.twoasset import ModelSpec, simulate, solve

    tiny = ModelSpec(xjump=20000.0, x_cells_per_step=4, zjump=200000.0, z_cells_per_step=3)
    sol = solve(0.8, 0.98, 2.0, tiny)
    a = simulate(sol, n_households=50, seed=3)
    b = simulate(sol, n_households=50, seed=3, initial_state=a["income_state"][:, 0])
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    c = simulate(sol, n_households=50, seed=3, initial_state=np.full(50, 2))
    assert (c["income_state"][:, 0] == 2).all()
    with pytest.raises(ValueError, match="initial_state"):
        simulate(sol, n_households=50, seed=3, initial_state=np.full(50, 3))


@gpu
def test_dispatch_uses_the_pool_income_state():
    from hh_npe.simulator.dispatch import (draw_initial_conditions,
                                           simulate_batch_twoasset_gpu)

    states, P, sig = _income_process()
    pool3 = np.column_stack([POOL, [-1.5, 0.0, 1.5, 0.3, -0.2]])
    _x, _a, p = simulate_batch_twoasset_gpu(THETAS, init_pool=pool3, **KW)
    for j in range(len(THETAS)):
        _w, s = draw_initial_conditions(pool3, 5, j, M, states, P, sig)
        np.testing.assert_array_equal(p["income_state"][j * M:(j + 1) * M, 0], s)
