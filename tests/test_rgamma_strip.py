"""The R_gamma strip below the A2 box and the logit R_gamma target (RESULTS 43.5).

Training on two runs with different R_gamma ranges and weighting back to the
uniform prior on their union. It goes wrong silently if a run's R_gamma
density is left out of the weight (it no longer cancels once the ranges
differ), if the two runs' draw ids collide (the grouped validation split would
leak), if runs from different simulators are combined, or if the R_gamma map
or its Jacobian is off.
"""

import json

import numpy as np
import pytest
import torch

from hh_npe.npe.prior import (
    A2_STRIP,
    EDGE_A2,
    PHASE3_RGAMMA,
    SWITCHED_A2,
    EdgeMixture,
    beta_to_flow,
    check_rgamma_tiling,
    proposal_log_weight,
    rgamma_flow_bounds,
    rgamma_from_flow,
    rgamma_log_jacobian,
    rgamma_to_flow,
    to_log1m,
)
from scripts import optionA as oa

WIDE = oa.box_of({"rgamma_range": [1.021, 1.075]})
MAIN = [1.025, 1.075]
STRIP = [1.021, 1.025]


def _theta(n, seed=0, r=(1.021, 1.075)):
    rng = np.random.default_rng(seed)
    th = np.column_stack([0.3 + 0.7 * rng.random(n), 0.85 + 0.15 * rng.random(n),
                          0.5 + 4.5 * rng.random(n), r[0] + (r[1] - r[0]) * rng.random(n)])
    return th


# --------------------------------------------------------------- R_gamma map

@pytest.mark.parametrize("box", [PHASE3_RGAMMA, WIDE])
def test_rgamma_logit_round_trips_numpy_and_torch(box):
    th = _theta(1000, r=(box.rgamma_low, box.rgamma_high))
    t = rgamma_to_flow(th, box, "logit")
    assert np.allclose(t[:, :3], th[:, :3]) and np.isfinite(t[:, 3]).all()
    assert np.allclose(rgamma_from_flow(t, box, "logit"), th, rtol=0, atol=1e-12)
    tt = rgamma_to_flow(torch.from_numpy(th), box, "logit")
    assert np.allclose(tt.numpy(), t)
    assert np.allclose(rgamma_from_flow(tt, box, "logit").numpy(), th, rtol=0, atol=1e-12)
    lo, hi = rgamma_flow_bounds(box, "logit")
    assert lo <= t[:, 3].min() and t[:, 3].max() <= hi


def test_rgamma_linear_is_the_identity():
    th = _theta(10)
    assert rgamma_to_flow(th, WIDE, "linear") is th
    assert rgamma_flow_bounds(WIDE, "linear") == (1.021, 1.075)
    assert (rgamma_log_jacobian(th[:, 3], WIDE, "linear") == 0).all()


def test_rgamma_jacobian_matches_a_finite_difference():
    r = np.linspace(1.0215, 1.0745, 50)
    h = 1e-7
    f = lambda v: rgamma_to_flow(np.column_stack([np.zeros((len(v), 3)), v]), WIDE, "logit")[:, 3]
    num = np.log((f(r + h) - f(r - h)) / (2 * h))
    assert np.allclose(rgamma_log_jacobian(r, WIDE, "logit"), num, atol=1e-5)


# ---------------------------------------------------------------------- Spec

@pytest.mark.parametrize("bt", ["linear", "log1m", "logit"])
def test_default_spec_is_the_pre_strip_map(bt):
    """Every ensemble before RESULTS 43.5 must map exactly as it did."""
    th = _theta(500, r=MAIN)
    s = oa.Spec(beta_transform=bt)
    old = beta_to_flow(to_log1m(th, oa.BOX), oa.BOX, bt)
    assert np.array_equal(s.to_flow(th), old)
    assert np.array_equal(oa.to_flow(th, bt), old)
    assert s.flow_box() == oa.flow_box(bt)
    assert np.allclose(s.from_flow(old), th, rtol=0, atol=1e-9)
    assert np.allclose(s.prior_sd, oa.PRIOR_SD)
    assert s == oa.spec_of({"beta_transform": bt})


def test_wide_logit_spec_round_trips_and_bounds_r_gamma():
    s = oa.Spec(box=WIDE, beta_transform="logit", rgamma_transform="logit")
    th = _theta(2000)
    t = s.to_flow(torch.from_numpy(th))
    fb = s.flow_box()
    assert (t.numpy() >= fb.low - 1e-9).all() and (t.numpy() <= fb.high + 1e-9).all()
    back = s.from_flow(t).numpy()
    assert np.allclose(back, th, rtol=0, atol=1e-6)
    # Any flow-space point maps inside the box: no mass past the floor.
    z = np.random.default_rng(0).normal(0, 5, (1000, 4))
    z[:, 1] = -np.abs(z[:, 1])
    assert (s.from_flow(z)[:, 3] > 1.021).all()
    assert s.cuts["R_gamma"] == (1.021, 1.039, 1.057, 1.075)
    assert s == oa.spec_of({"beta_transform": "logit", "rgamma_transform": "logit",
                            "rgamma_range": [1.021, 1.075]})


# ------------------------------------------------------------------- weights

def _two_run_draws(n_main, n_strip, seed=0):
    """Draws as the two runs make them: (beta, delta, rho) from their
    proposals, R_gamma uniform on each run's range."""
    rng = np.random.default_rng(seed)
    a = SWITCHED_A2.sample(n_main, seed=seed)
    b = A2_STRIP.sample(n_strip, seed=seed + 100)
    ra = MAIN[0] + (MAIN[1] - MAIN[0]) * rng.random(n_main)
    rb = STRIP[0] + (STRIP[1] - STRIP[0]) * rng.random(n_strip)
    return np.vstack([np.column_stack([a, ra]), np.column_stack([b, rb])])


PARTS = lambda n_main, n_strip: {"parts": [
    {"name": "a2_switched", "n_train": n_main, "rgamma_range": MAIN},
    {"name": "a2_strip", "n_train": n_strip, "rgamma_range": STRIP}]}


@pytest.mark.parametrize("n_main,n_strip", [(31744, 2560), (8192, 4096)])
def test_two_run_weights_recover_the_uniform_prior_on_the_union(n_main, n_strip):
    th = _two_run_draws(n_main, n_strip)
    lw = proposal_log_weight(PARTS(n_main, n_strip), th)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    lo, hi = WIDE.low, WIDE.high
    mean = (w[:, None] * th).sum(0)
    assert np.allclose(mean, (lo + hi) / 2, atol=0.012 * (hi - lo))
    # R_gamma below the old floor gets its prior share, 0.004 / 0.054.
    assert (w * (th[:, 3] < 1.025)).sum() == pytest.approx(0.004 / 0.054, abs=0.01)
    # E_q[p/q] = 1 with the weights unnormalised.
    assert np.exp(lw).mean() == pytest.approx(1.0, abs=0.03)


def test_two_run_weights_at_the_designed_density_are_bounded_and_continuous():
    lw = proposal_log_weight(PARTS(31744, 2560), _two_run_draws(31744, 2560))
    assert np.exp(lw).max() < 2.1
    # Same (beta, delta, rho) either side of 1.025: the strip was sized to the
    # main run's density per unit of R_gamma, so the weights nearly agree.
    b = np.array([[0.7, 0.98, 1.0, 1.0249], [0.7, 0.98, 1.0, 1.0251],
                  [0.35, 0.90, 1.0, 1.0249], [0.35, 0.90, 1.0, 1.0251]])
    w = np.exp(proposal_log_weight(PARTS(31744, 2560), b))
    assert w[0] == pytest.approx(w[1], rel=0.10) and w[2] == pytest.approx(w[3], rel=0.15)


def test_one_part_is_the_single_run_weight():
    """R_gamma's density cancels within one run's range."""
    th = _theta(1000, r=MAIN)
    one = {"parts": [{"name": "a2_switched", "n_train": 31744, "rgamma_range": MAIN}]}
    assert np.allclose(proposal_log_weight(one, th),
                       proposal_log_weight({"name": "a2_switched", "n_train": 31744}, th))
    held = {"parts": [{"name": "a2_strip", "n_train": 1024, "rgamma_range": MAIN}]}
    assert np.allclose(proposal_log_weight(held, th), EdgeMixture(**EDGE_A2).log_weight(th))


def test_switched_density_matches_its_log_weight():
    th = _theta(1000, r=MAIN)
    for n in (4096, 20000, 31744):
        assert np.allclose(SWITCHED_A2.log_weight(th, n),
                           np.log(1 / (0.7 * 0.15 * 4.5)) - np.log(SWITCHED_A2.density(th, n)))


def test_r_gamma_ranges_must_tile():
    assert check_rgamma_tiling([MAIN, STRIP]) == (1.021, 1.075)
    with pytest.raises(ValueError):
        check_rgamma_tiling([MAIN, [1.021, 1.024]])
    with pytest.raises(ValueError):
        check_rgamma_tiling([MAIN, [1.021, 1.03]])


# ---------------------------------------------------------- combining shards

def _fake_run(d, proposal, seed, rrange, n, block, **cfg_over):
    """Shards holding what load_shards reads: lo, hi, theta (from the run's
    proposal, R_gamma per block of 16) and card."""
    d.mkdir()
    sampler = {"a2_switched": SWITCHED_A2.sample, "a2_strip": A2_STRIP.sample}[proposal]
    th3 = sampler(n, seed=seed)
    r = np.repeat(np.linspace(*rrange, n // 16, endpoint=False), 16)
    th = np.column_stack([th3, r])
    for b in range(n // block):
        lo, hi = b * block, (b + 1) * block
        np.savez(d / f"shard_{b:05d}.npz", lo=lo, hi=hi, theta=th[lo:hi],
                 card=np.repeat(np.arange(lo // 16, hi // 16) % 2, 16))
    cfg = {"card_types": True, "educ": "comphs", "grid": "full", "proposal": proposal,
           "seed": seed, "rgamma_range": rrange, "theta_batch": 16, "n_households": 16,
           **cfg_over}
    (d / "solver_config.json").write_text(json.dumps(cfg))
    return th


def test_two_runs_combine_with_offsets_and_parts(tmp_path):
    a = _fake_run(tmp_path / "main", "a2_switched", 0, MAIN, 5 * 64, 64)
    b = _fake_run(tmp_path / "strip", "a2_strip", 100000, STRIP, 3 * 64, 64)
    theta, card, train, held, prop, cfg = oa.load_shards([tmp_path / "main", tmp_path / "strip"])
    assert np.array_equal(theta, np.vstack([a, b]))
    # Main holds out 2 shards, the strip 1 (RUN_SIZE).
    assert [s.offset for s in train] == [0, 0, 0, 320, 320]
    assert [s.offset for s in held] == [0, 0, 320]
    assert prop == {"parts": [{"name": "a2_switched", "n_train": 192, "rgamma_range": MAIN},
                              {"name": "a2_strip", "n_train": 128, "rgamma_range": STRIP}]}
    assert cfg["rgamma_range"] == [1.021, 1.075] and oa.box_of(cfg) == WIDE
    assert [r["n_draws"] for r in cfg["runs"]] == [320, 192]
    # One run: exactly what it always returned.
    t1, _c, tr1, _h, p1, cfg1 = oa.load_shards(tmp_path / "main")
    assert np.array_equal(t1, a) and p1 == {"name": "a2_switched", "n_train": 192}
    assert all(s.offset == 0 for s in tr1) and "runs" not in cfg1


def test_runs_from_different_simulators_are_refused(tmp_path):
    _fake_run(tmp_path / "main", "a2_switched", 0, MAIN, 5 * 64, 64)
    _fake_run(tmp_path / "strip", "a2_strip", 100000, STRIP, 3 * 64, 64, grid="mid")
    with pytest.raises(SystemExit, match="different settings"):
        oa.load_shards([tmp_path / "main", tmp_path / "strip"])


def test_a_gap_between_runs_is_refused(tmp_path):
    _fake_run(tmp_path / "main", "a2_switched", 0, MAIN, 5 * 64, 64)
    _fake_run(tmp_path / "strip", "a2_strip", 100000, [1.021, 1.024], 3 * 64, 64)
    with pytest.raises(ValueError, match="tile"):
        oa.load_shards([tmp_path / "main", tmp_path / "strip"])


def test_legacy_weighting_refuses_a_two_run_ensemble():
    with pytest.raises(SystemExit, match="spec_of"):
        oa._weighting(PARTS(10, 10), "logit")
