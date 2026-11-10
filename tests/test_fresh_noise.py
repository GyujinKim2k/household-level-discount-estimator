"""RESULTS 68: a new noise draw on the training windows every epoch.

The window code split into clean windows and noisy copies must leave every
earlier window unchanged; draw e of the redraw must be RESULTS 67's copy e
(draw 0 the one-draw windows); in training the training rows must take a new
draw before each epoch while the validation rows keep draw 0, the hook must
not change training when the draw does not change, and at the epoch cap the
best weights must come back. Synthetic shards and arrays only.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hh_npe.data.windows import build_windowed
from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import PriorBox, sample_sobol
from hh_npe.npe.train import train_npe
from scripts import optionA as oa
from tests.test_noise_copies import FILE, V2, WIN, _shards


def _two_runs(tmp_path):
    """Two runs of shards, the second offset as RESULTS 43.5's strips are."""
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
    fa, theta_a = _shards(tmp_path / "a")
    fb, theta_b = _shards(tmp_path / "b")
    files = [oa.Shard(p, 0) for p in fa] + [oa.Shard(p, len(theta_a)) for p in fb]
    return files, np.concatenate([theta_a, theta_b])


def _before(files, theta_all, seed, noise, window, copies):
    """``optionA.windows_raw`` as it was in RESULTS 67, before the split."""
    win = {**oa.WINDOW, **window}
    groups = {}
    for f in files:
        groups.setdefault(f.offset, []).append(f.path)
    clean = [(off, *build_windowed(paths, theta_all[off:], k=1, seed=seed + off, **win,
                                   flow_lag=0))
             for off, paths in groups.items()]
    out = []
    for c in range(copies):
        for off, th, x, pid in clean:
            x = oa.add_obs_noise(x, noise, seed=oa.NOISE_SEED + seed + off + c * oa.COPY_STRIDE)
            out.append((th, x, pid + off))
    return tuple(torch.cat(t) for t in zip(*out, strict=True))


@pytest.mark.parametrize("noise", [V2, FILE], ids=["v2", "file"])
@pytest.mark.parametrize("copies", [1, 3])
def test_the_split_window_code_gives_the_windows_of_before(tmp_path, noise, copies):
    files, theta = _two_runs(tmp_path)
    old = _before(files, theta, 0, noise, WIN, copies)
    new = oa.windows_raw(files, theta, 0, noise=noise, window=WIN, copies=copies)
    for a, b in zip(old, new, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("noise", [V2, FILE], ids=["v2", "file"])
def test_draw_e_is_copy_e(tmp_path, noise):
    files, theta = _two_runs(tmp_path)
    k = 4
    _th, x8, _pid, _f = oa.windows(files, theta, 0, noise=noise, window=WIN, copies=k)
    _th, x1, _pid, _f = oa.windows(files, theta, 0, noise=noise, window=WIN)
    redraw = oa.fresh_redraw(oa.clean_windows(files, theta, 0, window=WIN), 0, noise)
    n = len(x1)
    assert torch.equal(redraw(0), x1)
    for e in range(k):
        assert torch.equal(redraw(e), x8[e * n:(e + 1) * n])
    assert not torch.equal(redraw(1), redraw(2))


def test_fresh_noise_takes_no_copies():
    with pytest.raises(SystemExit, match="fresh_noise"):
        oa.train(SimpleNamespace(fresh_noise=True, noise_copies=8))


def test_draw_seeds_stay_clear_of_every_other_window_seed():
    others = {oa.HELD_SEED, oa.SBC_SEED, 0}
    offsets = (0, 32768, 35840)                     # the three runs' draw offsets
    draws = {oa.NOISE_SEED + off + e * oa.COPY_STRIDE for off in offsets for e in range(1, 201)}
    base = {oa.NOISE_SEED + s + off for s in others for off in offsets}
    assert not draws & base
    assert len(draws) == 3 * 200 and max(draws) < 2**31


# ------------------------------------------------------------------ training

def _data(n=64, per=2):
    torch.manual_seed(0)
    box = PriorBox()
    theta = torch.from_numpy(sample_sobol(n, box, seed=0)).float().repeat_interleave(per, 0)
    x = torch.randn(len(theta), 5, 3)
    x[:, :, 0] += 5 * theta[:, :1]
    return theta, x, torch.arange(n).repeat_interleave(per), box


def _train(redraw=None, epochs=3, patience=50):
    theta, x, pid, box = _data()
    torch.manual_seed(1)
    return train_npe(theta, x, embedder=TrajectoryTransformer(
        d_model=16, n_heads=2, n_layers=1, output_dim=8), hidden_features=16,
        num_transforms=2, max_num_epochs=epochs, stop_after_epochs=patience, batch_size=16,
        group_ids=pid, box=box, redraw_x=redraw)


class Draws:
    """Draw e: the data plus e on every value, so the rows show their draw."""

    def __init__(self, x):
        self.x, self.calls = x, []

    def __call__(self, e):
        self.calls.append(e)
        return self.x + float(e)


def test_training_rows_take_a_new_draw_every_epoch_and_validation_keeps_draw_0():
    _theta, x, _pid, _box = _data()
    draws = Draws(x)
    _p, _d, inf = _train(draws, epochs=3)
    assert len(inf._summary["training_loss"]) == 4                # sbi trains max + 1
    assert draws.calls == [1, 2, 3]                                 # before epochs 2, 3, 4
    data = inf.get_simulations()[1]   # sbi's stored rows: untouched, draw 0
    assert torch.equal(data, x)
    # the loaders' tensor: training rows on the last draw, validation on draw 0
    loader_x = inf.fresh_noise.dataset_x
    assert torch.equal(loader_x[inf.train_indices], x[inf.train_indices] + 3.0)
    assert torch.equal(loader_x[inf.val_indices], x[inf.val_indices])


def test_an_unchanged_draw_changes_nothing_in_training():
    _theta, x, _pid, _box = _data()
    a = _train(None)[2]._summary
    b = _train(lambda e: x.clone())[2]._summary
    assert a["training_loss"] == b["training_loss"]
    assert a["validation_loss"] == b["validation_loss"]


def test_the_best_weights_come_back_at_the_cap():
    _theta, x, _pid, _box = _data()
    gen = torch.Generator().manual_seed(5)

    def ruin_the_last(e):       # the last epoch trains on noise, so it is not the best
        return x.clone() if e < 3 else 50 * torch.randn(x.shape, generator=gen)

    _p, d, inf = _train(ruin_the_last, epochs=3)
    assert inf._summary["validation_loss"][-1] > inf._best_val_loss
    assert inf.restored_best_at_cap
    best = inf._best_model_state_dict
    assert all(torch.equal(d.state_dict()[k], v) for k, v in best.items())
    assert all(torch.equal(inf._neural_net.state_dict()[k], v) for k, v in best.items())


def test_nothing_is_restored_when_patience_stops_training():
    _theta, x, _pid, _box = _data()
    _p, _d, inf = _train(lambda e: x.clone(), epochs=40, patience=1)
    assert len(inf._summary["training_loss"]) < 41      # stopped by patience, not the cap
    assert not inf.restored_best_at_cap                 # sbi had loaded the best itself
