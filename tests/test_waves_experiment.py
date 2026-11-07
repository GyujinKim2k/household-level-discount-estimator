"""RESULTS 64's waves experiment: per-run windows, members that know their
window, the matched age range, and the population standard error.

No GPU and no solve: panels are synthetic, as in test_windows.py.
"""

import json

import numpy as np
import pytest
import torch

from hh_npe.data.waves import FEATURES_TWOASSET, FEATURES_TWOASSET_AGE
from hh_npe.data.windows import PANEL_PREFIX
from scripts import laibson_comparison as lc
from scripts import optionA as oa
from scripts import waves_experiment as we

T, N = 71, 64                               # ages 20..90
AGE = FEATURES_TWOASSET_AGE.index("age")


def _shard(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    p = {k: rng.normal(1e5, 3e4, (N, T)) for k in FEATURES_TWOASSET}
    p["t_age"] = np.tile(np.arange(T), (N, 1))
    d = tmp_path / "s_shards"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "shard_00000.npz"
    np.savez(f, x=np.zeros((N, 5, 4), np.float32), alive=np.ones((N, 5), bool), lo=0, hi=N,
             **{PANEL_PREFIX + k: v for k, v in p.items()})
    return f


@pytest.fixture
def shard(tmp_path, monkeypatch):
    monkeypatch.setattr(oa, "flow_lag_of", lambda paths: 1)
    return _shard(tmp_path)


def test_the_arms_fit_the_matched_age_range():
    assert {n: we.start_high(n) for n in we.WAVES} == {7: 51, 10: 45, 14: 37, 20: 25}
    for n in we.WAVES:
        assert we.start_high(n) + 2 * (n - 1) == we.LAST_WAVE_START
    with pytest.raises(ValueError):
        we.start_high(21)                    # would have to start at 23


def test_the_annual_arms_read_only_ages_24_to_64():
    """RESULTS 65: an annual wave at age s reads its stock at s and its flows
    at s - 1, so annual windows start at 25; biennial ranges are unchanged."""
    assert we.start_low() == we.START_LOW == 24 and we.start_low(1) == 25
    assert {n: we.start_high(n, 1) for n in (14, 20)} == {14: 51, 20: 45}
    assert all(we.start_high(n) == we.start_high(n, 2) == 65 - 2 * n for n in we.WAVES)
    with pytest.raises(ValueError):
        we.start_high(41, 1)                 # would have to start at 24, its flows at 23


def test_arm_labels_round_trip():
    assert we.arm_label(14) == "w14" and we.arm_label(14, 1) == "a14"
    assert we.parse_arm("a20") == (20, 1) and we.parse_arm(we.arm_label(7)) == (7, 2)
    assert we.arm_dir(14) == we.ROOT / "w14" and we.arm_dir("a14") == we.ROOT / "a14"
    for bad in ("x14", "a", "w1x"):
        with pytest.raises(ValueError):
            we.parse_arm(bad)


def test_an_annual_window_is_checked_as_annual():
    a14 = {**oa.WINDOW, "n_waves": 14, "start_low": 25, "start_high": 51, "wave_years": 1}
    we.check_window(a14, 14, 1)
    with pytest.raises(SystemExit):
        we.check_window(a14, 14)                                # not the biennial w14
    with pytest.raises(SystemExit):
        we.check_window({**a14, "start_low": 24}, 14, 1)        # flows at 23


@pytest.mark.parametrize("n", (14, 20))
def test_annual_windows_lie_inside_ages_24_to_64(shard, n):
    win = {**oa.WINDOW, "n_waves": n, "start_low": we.start_low(1),
           "start_high": we.start_high(n, 1), "wave_years": 1}
    _t, x, _p = oa.windows_raw([shard], np.zeros((N, 4)), seed=999, window=win)
    assert x.shape[1] == n
    age = x[..., AGE]                        # an annual wave's age is its own year
    assert age[:, 0].min() == 25 and age[:, -1].max() == 64
    assert torch.equal(age[:, 1:] - age[:, :-1], torch.ones_like(age[:, 1:]))


def _sim(seed=0):
    rng = np.random.default_rng(seed)
    sim = {"panel_" + k: rng.normal(1e5, 3e4, (N, T)) for k in FEATURES_TWOASSET}
    sim["panel_t_age"] = np.tile(np.arange(T), (N, 1))
    sim["panel_age"] = 20 + sim["panel_t_age"]        # as laibson_comparison.simulate stores
    sim["ids"] = np.arange(N)
    return sim


@pytest.mark.parametrize("label", ["w14", "a14", "a20"])
def test_the_populations_starts_keep_the_biennial_seeds(label):
    n, wy = we.parse_arm(label)
    win = {**oa.WINDOW, "n_waves": n, "start_low": we.start_low(wy),
           "start_high": we.start_high(n, wy), "wave_years": wy}
    x = we.cut(_sim(), win, flow_lag=1)
    seed = we.POP_SEED + n + (0 if wy == 2 else 1000)
    s = np.random.default_rng(seed).integers(win["start_low"], win["start_high"] + 1, N)
    assert np.array_equal(x[:, 0, AGE], s + wy - 1)        # the first wave's stock year


def test_a_members_window_is_read_back_and_old_configs_read_as_before():
    assert oa.window_of({}) == oa.WINDOW
    assert oa.window_of({"window": {"n_waves": 7, "flow_lag": 1}}) == oa.WINDOW
    w = oa.window_of({"window": {**oa.WINDOW, "n_waves": 14, "start_high": 37, "flow_lag": 1}})
    assert w == {**oa.WINDOW, "n_waves": 14, "start_high": 37}


def test_default_windows_are_unchanged(shard):
    a = oa.windows_raw([shard], np.zeros((N, 4)), seed=999)
    b = oa.windows_raw([shard], np.zeros((N, 4)), seed=999, window=oa.WINDOW)
    c = oa.windows_raw([shard], np.zeros((N, 4)), seed=999, n_waves=7)
    for x, y, z in zip(a, b, c, strict=True):
        assert torch.equal(x, y) and torch.equal(x, z)


@pytest.mark.parametrize("n", we.WAVES)
def test_each_arms_windows_lie_inside_ages_24_to_64(shard, n):
    win = {**oa.WINDOW, "n_waves": n, "start_high": we.start_high(n)}
    _t, x, _p = oa.windows_raw([shard], np.zeros((N, 4)), seed=999, window=win)
    assert x.shape[1] == n
    age = x[..., AGE]                        # a wave's age is its second year
    assert age[:, 0].min() >= we.START_LOW + 1 and age[:, -1].max() <= we.LAST_WAVE_START + 1
    assert torch.equal(age[:, 1:] - age[:, :-1], torch.full_like(age[:, 1:], 2))


def test_the_20_wave_arm_uses_both_start_ages(shard):
    win = {**oa.WINDOW, "n_waves": 20, "start_high": we.start_high(20)}
    _t, x, _p = oa.windows_raw([shard], np.zeros((N, 4)), seed=999, window=win)
    assert set(x[:, 0, AGE].tolist()) == {25.0, 26.0}


def test_a_window_outside_the_range_is_refused():
    we.check_window({**oa.WINDOW, "n_waves": 14, "start_high": 37}, 14)
    with pytest.raises(SystemExit):
        we.check_window({**oa.WINDOW, "n_waves": 14}, 14)     # start_high 45 runs to 72


def _member(d, window):
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"_config": {"window": window, "seed": 0}}))


def test_members_report_their_window_length_and_refuse_a_mix(tmp_path, monkeypatch):
    class Fake:
        posterior_estimator = None

    monkeypatch.setattr(oa, "load_posterior", lambda path, **kw: (
        {"posterior": Fake(), "proposal": {"parts": []}} if path.name == "posterior_14w.pt"
        else pytest.fail(f"loaded {path.name}")))
    w14 = {**oa.WINDOW, "n_waves": 14, "start_high": 37}
    _member(tmp_path / "s0", w14)
    _member(tmp_path / "s1", w14)
    _posts, _prop, cfg = oa.load_members([tmp_path / "s0", tmp_path / "s1"])
    assert oa.window_of(cfg)["n_waves"] == 14 and oa.window_of(cfg)["start_high"] == 37
    _member(tmp_path / "s2", {**w14, "start_high": 45})
    with pytest.raises(SystemExit):
        oa.load_members([tmp_path / "s0", tmp_path / "s2"])
    with pytest.raises(SystemExit):
        oa.load_members([tmp_path / "s0"], n_waves=7)


def test_the_population_se_is_exact_on_a_linear_map():
    rng = np.random.default_rng(0)
    h, n_pop, share = 4000, 409, 0.85
    card = (np.arange(h) % 2).astype(int)
    th0 = np.array([0.437, 0.9928, 1.58, 1.042])
    A = np.array([[2.0, 0.1, 0.0, 0.3], [0.2, -1.5, 0.1, 0.0],
                  [0.0, 0.3, 0.8, -0.2], [0.1, 0.0, 0.05, 1.2]])
    centre = th0 + rng.normal(0, 1, (h, 4)) * [0.1, 0.01, 0.5, 0.01] + 0.05 * card[:, None]
    means = {"centre": centre}
    for k in range(4):
        du = lc.STEP[k] * np.eye(4)[k]
        means[f"plus{k}"] = centre + du @ A.T
        means[f"minus{k}"] = centre - du @ A.T
    r = we.population_se(means, card, th0, n_pop, share)
    assert np.allclose(r["jacobian"], A)
    w = np.where(card == 1, share / (card == 1).sum(), (1 - share) / (card == 0).sum())
    mu = w @ centre
    v = sum(wi * np.outer(c - mu, c - mu) for wi, c in zip(w, centre, strict=True)) / n_pop
    ai = np.linalg.inv(A)
    D = np.diag([1, -(1 - th0[1]), 1, 1])
    assert np.allclose(r["cov"], D @ ai @ v @ ai.T @ D.T)
    assert np.allclose(r["se"], np.sqrt(np.diag(r["cov"])))


def test_psid_scale_holds_the_remainder_only_in_case_b():
    rng = np.random.default_rng(1)
    L = rng.normal(size=(4, 4))
    v_hom = L @ L.T * 1e-4
    M = rng.normal(size=(4, 4))
    v_psid = v_hom + M @ M.T * 1e-4
    same = we.psid_scale(v_psid, v_hom, np.ones(4))
    assert np.allclose(same["a"], v_psid) and np.allclose(same["b"], v_psid)
    none = we.psid_scale(v_psid, v_hom, np.zeros(4))
    assert np.allclose(none["a"], 0) and np.allclose(none["b"], v_psid - v_hom)
    half = we.psid_scale(v_psid, v_hom, np.full(4, 0.5))
    assert np.allclose(half["a"], v_psid / 4)
    assert np.allclose(half["b"], v_psid - v_hom + v_hom / 4)
    assert (np.diag(half["b"]) >= np.diag(half["a"])).all()


def test_outside_95_counts_parameters_per_household():
    u = np.array([[0.5, 0.5, 0.5, 0.5], [0.01, 0.5, 0.5, 0.5], [0.99, 0.02, 0.5, 0.5],
                  [0.025, 0.975, 0.5, 0.5]])
    assert we.outside_95(u) == {"at_least_1": 0.5, "at_least_2": 0.25}
    calibrated = np.random.default_rng(0).uniform(size=(200_000, 4))
    r = we.outside_95(calibrated)
    assert abs(r["at_least_1"] - (1 - 0.95**4)) < 0.003
    assert abs(r["at_least_2"] - (1 - 0.95**4 - 4 * 0.05 * 0.95**3)) < 0.001


SHARDS = [oa.Path(p) for p in ("data/processed/couples_v2_dataset_shards",
                               "data/processed/couples_v2_strip_dataset_shards",
                               "data/processed/couples_v2_betastrip_dataset_shards")]


@pytest.mark.skipif(not all(p.exists() for p in SHARDS), reason="v2's shards are not here")
@pytest.mark.parametrize("which", [0, 2, 3])        # a held-out shard of each run
def test_every_arms_windows_are_the_stored_panels_read_by_hand(which):
    """RESULTS 64.1 gate 2 (and 65's annual arms): a wave's stocks are the
    panel's at the start of its last year, its flows the year before
    (flow_lag 1); every start inside the arm's range, every year read inside
    24-64."""
    from hh_npe.data.waves import FLOWS
    from hh_npe.data.windows import sample_start_ages

    theta_all, _c, _tr, held_f, _p, _cfg = oa.load_shards(SHARDS)
    f = held_f[which]
    d = np.load(f.path)
    lo, hi = int(d["lo"]), int(d["hi"])
    lag = oa.flow_lag_of([f.path])
    assert lag == 1
    panel = {k[len(PANEL_PREFIX):]: d[k] for k in d.files if k.startswith(PANEL_PREFIX)}
    m = len(panel["t_age"]) // (hi - lo)
    for label in [we.arm_label(n) for n in we.WAVES] + ["a14", "a20"]:
        n, wy = we.parse_arm(label)
        win = {**oa.WINDOW, "n_waves": n, "start_low": we.start_low(wy),
               "start_high": we.start_high(n, wy), "wave_years": wy}
        th, x, pid = oa.windows_raw([oa.Shard(f.path, f.offset)], theta_all, seed=999, window=win)
        x, pid = x.numpy(), pid.numpy()
        # cut_windows groups rows by start age: which panel row each one is
        st = sample_start_ages((hi - lo) * m, 1, win["start_low"], win["start_high"],
                               seed=999 + f.offset + lo)[:, 0]
        order = np.concatenate([np.flatnonzero(st == v) for v in np.unique(st)])
        assert np.array_equal(pid - f.offset, lo + order // m)
        assert np.allclose(th.numpy(), theta_all[f.offset + lo + order // m])
        assert set(np.unique(st)) == set(range(win["start_low"], win["start_high"] + 1))
        for i in range(0, len(x), 97):
            r, s = order[i], st[order[i]]
            assert x[i, 0, AGE] == s + wy - 1 and x[i, -1, AGE] <= we.LAST_AGE
            assert x[i, 0, AGE] - lag >= we.FIRST_AGE   # the first flow year
            for w in range(n):
                t = s - 20 + wy * w + wy - 1            # the wave's last year
                for k, name in enumerate(FEATURES_TWOASSET_AGE):
                    want = (20 + panel["t_age"][r, t] if name == "age"
                            else panel[name][r, t - (lag if name in FLOWS else 0)])
                    assert np.isclose(x[i, w, k], want, rtol=1e-6), (label, i, w, name)
