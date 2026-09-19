"""The train/held-out split must be non-empty, and must be checked early.

Both properties come from one incident. ``--train_n`` defaulted to 65536 while
generation was still running, when shards written after the snapshot supplied
free held-out draws. Once the dataset finished at exactly 65536 every shard
fell on the training side, ``split_shards`` raised -- but only after
``simulate_sbc_once`` had already spent 3.46 h of GPU on panels held in memory,
so the run died with nothing to show.

So: the default must leave shards on both sides of a complete dataset, and the
split must happen before the simulations.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from scripts.compare_windows import split_shards

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_windows.py"
N_TOTAL = 65536
SHARD = 256


def _shards(tmp_path, n_total=N_TOTAL, per=SHARD):
    d = tmp_path / "shards"
    d.mkdir()
    files = []
    for i, lo in enumerate(range(0, n_total, per)):
        f = d / f"shard_{i:05d}.npz"
        np.savez(f, lo=lo, hi=lo + per)
        files.append(f)
    return files


def _default_of(flag):
    """Read a parser default straight from the source, without running main."""
    tree = ast.parse(SCRIPT.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"
                and node.args and getattr(node.args[0], "value", None) == flag):
            for kw in node.keywords:
                if kw.arg == "default":
                    return ast.literal_eval(kw.value)
    raise AssertionError(f"no {flag} argument found")


def test_default_train_n_leaves_shards_on_both_sides(tmp_path):
    """The regression: a complete 65536-draw dataset must still split."""
    train, held = split_shards(_shards(tmp_path), _default_of("--train_n"))
    assert train and held, "the default must hold something out"
    assert len(train) + len(held) == N_TOTAL // SHARD


def test_default_train_n_splits_on_a_sobol_block_boundary():
    """Both sides stay balanced only if the cut is a multiple of a power of 2.

    57344 = 32768 + 16384 + 8192, and the 8192 held out is one block, so each
    side is a union of exactly-balanced Sobol blocks.
    """
    train_n = _default_of("--train_n")
    held = N_TOTAL - train_n
    assert held > 0
    assert held & (held - 1) == 0, f"held-out {held} is not a power of 2"
    assert train_n % held == 0, f"{train_n} is not a whole number of {held}-blocks"


def test_empty_side_is_refused(tmp_path):
    files = _shards(tmp_path)
    with pytest.raises(SystemExit, match="both sides"):
        split_shards(files, N_TOTAL)
    with pytest.raises(SystemExit, match="both sides"):
        split_shards(files, 0)


def test_split_is_checked_before_the_gpu_simulations():
    """Ordering, read off the source: the cheap guard must come first.

    Without this the failure mode is invisible in review -- both calls are
    present and the script looks correct until it has burned the GPU time.
    """
    src = SCRIPT.read_text()
    main = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n.func.id for n in ast.walk(main)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in ("split_shards", "simulate_sbc_once")]
    assert calls[:2] == ["split_shards", "simulate_sbc_once"], (
        f"expected split_shards before simulate_sbc_once, got {calls}"
    )


def test_sbc_cache_round_trips(tmp_path, monkeypatch):
    """A cache hit must skip the solver entirely."""
    import torch

    from scripts import compare_windows as cw

    cache = tmp_path / "sbc.pt"
    thetas = torch.randn(4, 3)
    panels = {"income": np.zeros((4, 71))}
    torch.save({"thetas": thetas, "panels": panels, "n_sbc": 4, "seed": 7,
                "solver_config": {}}, cache)

    def boom(*a, **k):
        raise AssertionError("solver ran despite a valid cache")

    monkeypatch.setattr(cw, "simulate_batch_twoasset_gpu", boom)
    th, pn, ed = cw.simulate_sbc_once(4, 7, {"theta_batch": 16, "chunk": 16},
                                      cache=cache)
    torch.testing.assert_close(th, thetas)
    assert pn.keys() == panels.keys()
    # A pre-education cache has no `educ_idx`; it must read back as absent
    # rather than as an array of zeros, which would mean "all comphs".
    assert ed is None


def test_sbc_cache_from_a_different_education_mode_is_not_reused(
        tmp_path, monkeypatch):
    """The trap this guards: a comphs-only cache silently reused under `mixed`.

    SBC asks whether a posterior is calibrated for its OWN generative process.
    Scoring a marginalised posterior against comphs-only simulations measures
    the mismatch instead, and looks like miscalibration no training could fix.
    A cache written before education existed reads back as comphs, so it must
    be rejected -- not reused -- when `mixed` is asked for.
    """
    import torch

    from scripts import compare_windows as cw

    cache = tmp_path / "sbc.pt"
    torch.save({"thetas": torch.randn(4, 3), "panels": {"income": np.zeros((4, 71))},
                "n_sbc": 4, "seed": 7, "solver_config": {}}, cache)

    calls = []

    def fake(*a, **k):
        calls.append(k.get("educ"))
        return np.zeros((4, 7, 4)), np.ones((4, 7)), {"income": np.zeros((4, 71))}

    monkeypatch.setattr(cw, "simulate_batch_twoasset_gpu", fake)
    cw.simulate_sbc_once(4, 7, {"theta_batch": 16, "chunk": 16},
                         cache=cache, educ="mixed")
    assert calls, "comphs cache was reused under educ=mixed"


def test_sbc_cache_for_a_different_n_is_not_reused(tmp_path, monkeypatch):
    """Silently scoring 4 draws when 1000 were asked for would be worse."""
    import torch

    from scripts import compare_windows as cw

    cache = tmp_path / "sbc.pt"
    torch.save({"thetas": torch.randn(4, 3), "panels": {}, "n_sbc": 4,
                "seed": 7, "solver_config": {}}, cache)

    called = []

    def fake(theta_np, *a, **k):
        called.append(len(theta_np))
        return None, None, {"income": np.zeros((len(theta_np), 71))}

    monkeypatch.setattr(cw, "simulate_batch_twoasset_gpu", fake)
    cw.simulate_sbc_once(8, 7, {"theta_batch": 16, "chunk": 16}, cache=cache)
    assert called == [8], "a stale cache must be re-simulated, not reused"
