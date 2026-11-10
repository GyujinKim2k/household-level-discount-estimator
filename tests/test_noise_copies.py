"""RESULTS 67's noisy copies: K noise draws on the same clean windows.

Copy 0 must be exactly today's windows (so K = 1 changes nothing), every copy
must share the clean signal, theta and draw id with copy 0 and differ only in
noise, and a grouped split must keep all copies of a household on one side.
Also the measures' helpers. Synthetic shards and arrays only.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hh_npe.data.waves import FEATURES_TWOASSET
from hh_npe.data.waves import FEATURES_TWOASSET_AGE as FEATS
from hh_npe.data.windows import PANEL_PREFIX, build_windowed
from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import PriorBox, sample_sobol
from hh_npe.npe.train import train_npe
from scripts import noise_copies as nc
from scripts import optionA as oa
from scripts import psid_noise as pn
from scripts import training_metrics as tm

T, N = 71, 24
V2 = (("income", 0.0, 0.0), ("consumption", 0.0, 0.15), ("liquid_assets", 0.0, 0.1),
      ("illiquid_assets", 0.175, 0.47))


def _shards(tmp_path, n_shards=2):
    d = tmp_path / "run_shards"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n_shards):
        p = {k: rng.normal(1e5, 3e4, (N, T)) for k in FEATURES_TWOASSET}
        p["t_age"] = np.tile(np.arange(T), (N, 1))
        np.savez(d / f"shard_{i:05d}.npz", lo=i * N, hi=(i + 1) * N,
                 **{PANEL_PREFIX + k: v for k, v in p.items()})
    (d / "solver_config.json").write_text(json.dumps({"flow_lag": 0}))
    theta = rng.uniform(0, 1, (n_shards * N, 4))
    return sorted(d.glob("shard_*.npz")), theta


def _before(files, theta_all, seed, noise, window):
    """``optionA.windows_raw`` as it was before RESULTS 67 (one run, offset 0)."""
    win = {**oa.WINDOW, **window}
    th, x, pid = build_windowed(files, theta_all, k=1, seed=seed, **win, flow_lag=0)
    return th, oa.add_obs_noise(x, noise, seed=oa.NOISE_SEED + seed), pid


WIN = {"n_waves": 7, "start_low": 24, "start_high": 51}
FILE = pn.FileNoise(var_liquid=1.5, var_illiquid=0.55, income_ref=(11.5,) * 7)


@pytest.mark.parametrize("noise", [V2, FILE], ids=["v2", "file"])
def test_one_copy_is_exactly_the_windows_of_before(tmp_path, noise):
    files, theta = _shards(tmp_path)
    old = _before(files, theta, 999, noise, WIN)
    for new in (oa.windows_raw(files, theta, 999, noise=noise, window=WIN),
                oa.windows_raw(files, theta, 999, noise=noise, window=WIN, copies=1)):
        for a, b in zip(old, new, strict=True):
            assert torch.equal(a, b)


@pytest.mark.parametrize("noise", [V2, FILE], ids=["v2", "file"])
def test_copies_share_the_signal_and_differ_in_noise(tmp_path, noise):
    files, theta = _shards(tmp_path)
    one = oa.windows_raw(files, theta, 0, noise=noise, window=WIN)
    clean = oa.windows_raw(files, theta, 0, window=WIN)[1]
    th, x, pid = oa.windows_raw(files, theta, 0, noise=noise, window=WIN, copies=3)
    n = len(one[0])
    assert len(th) == 3 * n
    assert torch.equal(x[:n], one[1]) and torch.equal(th[:n], one[0])
    age = FEATS.index("age")
    for k in range(3):
        blk = slice(k * n, (k + 1) * n)
        assert torch.equal(th[blk], one[0]) and torch.equal(pid[blk], one[2])
        assert torch.equal(x[blk][..., age], clean[..., age])
        if k:
            j = FEATS.index("illiquid_assets")
            assert not torch.equal(x[blk][..., j], one[1][..., j])
            # the same clean value under each copy's own noise
            r = torch.log(x[blk][..., j] / clean[..., j])
            assert torch.isfinite(r).all() and r.abs().max() > 0
    with pytest.raises(ValueError):
        oa.windows_raw(files, theta, 0, noise=noise, window=WIN, copies=0)


def test_copies_of_a_household_stay_on_one_side_of_the_split():
    torch.manual_seed(0)
    n, k = 48, 4
    box = PriorBox()
    theta = torch.from_numpy(sample_sobol(n, box, seed=0)).float().repeat(k, 1)
    pid = torch.arange(n).repeat(k)
    x = torch.randn(n * k, 5, 3) + 5 * theta[:, :1, None]
    _p, _d, inf = train_npe(theta, x, embedder=TrajectoryTransformer(
        d_model=16, n_heads=2, n_layers=1, output_dim=8), hidden_features=16,
        num_transforms=2, max_num_epochs=1, stop_after_epochs=1, batch_size=32,
        group_ids=pid, box=box)
    tr, va = set(pid[inf.train_indices].tolist()), set(pid[inf.val_indices].tolist())
    assert not tr & va and tr | va == set(range(n))
    assert len(inf.val_indices) == k * len(va)


def test_copy_seeds_stay_clear_of_every_other_window_seed():
    others = {oa.HELD_SEED, oa.SBC_SEED, 0}
    offsets = (0, 32768, 35840)                     # the three runs' draw offsets
    copy = {oa.NOISE_SEED + 0 + off + k * oa.COPY_STRIDE for off in offsets for k in range(1, 8)}
    base = {oa.NOISE_SEED + s + off for s in others for off in offsets}
    assert not copy & base


def test_metrics_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    out = Path("outputs/noise_copies/v2_k8/w07")
    assert oa.metrics_paths(out, 3) == [out / "s3" / "metrics.csv",
                                        Path("training_metrics/noise_copies/v2_k8/w07/s3.csv")]
    smoke = tmp_path / "smoke" / "outputs" / "noise_copies" / "v2_k8" / "w07"
    assert oa.metrics_paths(smoke, 0) == [smoke / "s0" / "metrics.csv"]
    assert oa.metrics_paths(smoke, 0, tmp_path / "tm")[1] == \
        tmp_path / "tm" / "noise_copies" / "v2_k8" / "w07" / "s0.csv"
    assert oa.metrics_paths(tmp_path / "elsewhere", 1) == [tmp_path / "elsewhere" / "s1" /
                                                           "metrics.csv"]


def test_cross_noise_may_compare_runs_trained_differently():
    assert {"obs_noise", "noise_model"} == set(oa.NOISE_KEYS)
    assert {"noise_copies", "max_epochs", "stop_after_epochs"} == set(oa.TRAINING_KEYS)
    assert "window" not in oa.NOISE_KEYS + oa.TRAINING_KEYS


# ------------------------------------------------------------- the measures

def test_each_arm_lives_where_the_chain_puts_it():
    assert nc.run_dir("w07") == Path("outputs/waves/w07")
    assert nc.run_dir("file_k8") == Path("outputs/noise_copies/file_k8/w07")
    assert nc.arm_root("w07") == Path("outputs/waves")
    assert nc.arm_root("v2_k8") == Path("outputs/noise_copies/v2_k8")
    with pytest.raises(ValueError):
        nc.run_dir("gross")
    # every pair is on one noise: the second arm's held-out windows are the first's
    for a, b in nc.PAIRS:
        assert nc.NOISE_OF[a] == nc.NOISE_OF[b]


def test_paired_log_q():
    rng = np.random.default_rng(0)
    a = rng.normal(3, 1, 2000)
    w = rng.uniform(0.5, 2, 2000)
    same = nc.paired_log_q({"a": a, "b": a}, w, "a", "b", 200, 0)
    assert same["diff"] == 0 and same["ci95"] == [0, 0]
    up = nc.paired_log_q({"a": a + 0.1 + rng.normal(0, 0.05, 2000), "b": a}, w, "a", "b",
                         500, 0)
    assert up["diff"] == pytest.approx(0.1, abs=0.01)
    assert up["ci95"][0] > 0.09 and up["ci95"][1] < 0.11


def test_training_reads_the_best_epoch_from_the_csvs(tmp_path):
    rows = tm.rows_from_summary({"training_loss": [3.0, 2.8, 2.6, 2.5],
                                 "validation_loss": [3.2, 3.1, 3.15, 3.12],
                                 "epoch_durations_sec": [60, 60, 60, 60]}, passes_per_epoch=8)
    tm.write(tmp_path / "s0.csv", rows)
    t = nc.training(tmp_path)["s0"]
    assert t == {"epochs": 4, "passes_at_best": 16.0, "best_val": 3.1,
                 "train_minus_val_at_best": pytest.approx(-0.3), "minutes": 4.0}


def test_sbc_title_names_the_arm_not_the_run_directory():
    from types import SimpleNamespace

    from scripts import sbc_optionA2 as sbc

    spec = SimpleNamespace(masked=False, imputation_mask=None,
                           box=SimpleNamespace(beta_low=0.1, beta_high=1.0))
    dirs = [nc.run_dir("file_k8") / "s0"]
    # every arm's run directory is w07, so the default title cannot tell them apart
    assert sbc.title(dirs, spec).startswith("w07:")
    assert sbc.title(dirs, spec, name="file_k8").startswith("file_k8:")
