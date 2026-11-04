"""The beta strip below the A2 box (RESULTS 63).

Training on three runs -- the v2 main run, the R_gamma strip and the beta
strip on [0.1, 0.3) -- and weighting back to the uniform prior on their union.
It goes wrong silently if a run's density is not zero outside its own
(beta, R_gamma) rectangle, if the union's prior is the A box's, if the strip's
draws are not the sequence its weights describe, or if the change moves any
earlier (two-run) ensemble's weights or box: every RESULTS 56-62 number rests
on those.
"""

import json

import numpy as np
import pytest

from hh_npe.npe.prior import (
    A2_STRIP,
    BETA_RANGE_A2,
    SWITCHED_A2,
    UniformProposal,
    _part_density,
    a2_beta_strip,
    beta_range_box,
    check_tiling,
    proposal_log_weight,
    proposal_sampler,
    sample_sobol,
)
from scripts import laibson_comparison as lc
from scripts import optionA as oa
from scripts import sbc_optionA2 as so

MAIN, STRIP, ALL = [1.025, 1.075], [1.021, 1.025], [1.021, 1.075]
STRIP_B = [0.1, 0.3]


def _old_parts_log_weight(parts, th):
    """RESULTS 43.5's weight as it was before RESULTS 63, verbatim."""
    from hh_npe.npe.prior import check_rgamma_tiling
    lo, hi = check_rgamma_tiling([p["rgamma_range"] for p in parts])
    n = sum(int(p["n_train"]) for p in parts)
    r = th[:, 3]
    dens = np.zeros(len(th))
    for p in parts:
        a, b = map(float, p["rgamma_range"])
        inside = (r >= a) & ((r < b) | ((b == hi) & (r <= b)))
        dens += np.where(inside, int(p["n_train"]) / n * _part_density(p, th[:, :3])
                         / (b - a), 0.0)
    with np.errstate(divide="ignore"):
        return np.log(UniformProposal()._uniform_density() / (hi - lo)) - np.log(dens)


def _three_run_draws(n_main, n_strip, n_beta, seed=0):
    """Draws as the three runs make them: (beta, delta, rho) from their
    proposals, R_gamma uniform on each run's range."""
    rng = np.random.default_rng(seed)
    a = SWITCHED_A2.sample(n_main, seed=seed)
    b = A2_STRIP.sample(n_strip, seed=seed + 100)
    c = a2_beta_strip(*STRIP_B).sample(n_beta, seed=seed + 200)
    r = [lo + (hi - lo) * rng.random(n) for (lo, hi), n in
         ((MAIN, n_main), (STRIP, n_strip), (ALL, n_beta))]
    return np.vstack([np.column_stack([t, x]) for t, x in zip((a, b, c), r, strict=True)])


def PARTS3(n_main, n_strip, n_beta):
    return {"parts": [{"name": "a2_switched", "n_train": n_main, "rgamma_range": MAIN},
                      {"name": "a2_strip", "n_train": n_strip, "rgamma_range": STRIP},
                      {"name": "a2_beta_strip", "n_train": n_beta, "rgamma_range": ALL,
                       "beta_range": STRIP_B}]}


def _uniform_theta(n, beta=(0.3, 1.0), r=(1.021, 1.075), seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([beta[0] + (beta[1] - beta[0]) * rng.random(n),
                            0.85 + 0.15 * rng.random(n), 0.5 + 4.5 * rng.random(n),
                            r[0] + (r[1] - r[0]) * rng.random(n)])


# ----------------------------------------------------------------- the strip

def test_strip_draws_lie_on_the_strip_and_follow_its_density():
    th = a2_beta_strip(*STRIP_B).sample(8192, seed=200000)
    assert th[:, 0].min() >= 0.1 and th[:, 0].max() < 0.3
    assert th[:, 1].min() >= 0.85 and th[:, 1].max() < 1.0
    assert th[:, 2].min() >= 0.5 and th[:, 2].max() <= 5.0
    # Half uniform on the strip's box, half with delta on [0.95, 1).
    assert (th[:, 1] >= 0.95).mean() == pytest.approx(0.5 + 0.5 / 3, abs=0.01)
    d = a2_beta_strip(*STRIP_B).density(np.array([[0.2, 0.9, 2.0], [0.2, 0.97, 2.0]]))
    vol = 0.2 * 0.15 * 4.5
    assert d == pytest.approx([0.5 / vol, 0.5 / vol + 0.5 / (0.2 * 0.05 * 4.5)])


def test_proposal_sampler_follows_a_run_s_beta_range():
    assert np.array_equal(proposal_sampler("uniform", STRIP_B)(64, 3),
                          sample_sobol(64, beta_range_box(STRIP_B), seed=3))
    assert np.array_equal(proposal_sampler("a2_beta_strip", STRIP_B)(64, 3),
                          a2_beta_strip(*STRIP_B).sample(64, seed=3))
    # Runs that recorded no range, or the A box's, are sampled as before.
    assert proposal_sampler("a2_switched", None) is not None
    assert np.array_equal(proposal_sampler("uniform", BETA_RANGE_A2)(64, 3),
                          sample_sobol(64, beta_range_box(BETA_RANGE_A2), seed=3))
    with pytest.raises(ValueError):
        proposal_sampler("a2_switched", STRIP_B)


# ------------------------------------------------------------------- weights

def test_two_run_weights_are_bit_identical_to_before():
    """v2 and every RESULTS 56-62 arm: their parts carry no beta_range."""
    th = _uniform_theta(5000)
    parts = [{"name": "a2_switched", "n_train": 31744, "rgamma_range": MAIN},
             {"name": "a2_strip", "n_train": 2560, "rgamma_range": STRIP}]
    old = _old_parts_log_weight(parts, th)
    assert np.array_equal(proposal_log_weight({"parts": parts}, th), old)
    # The A box's range written out changes nothing either.
    explicit = [{**p, "beta_range": [0.3, 1.0]} for p in parts]
    assert np.array_equal(proposal_log_weight({"parts": explicit}, th), old)


@pytest.mark.parametrize("n", [(31744, 2560, 5632), (8192, 2048, 4096)])
def test_three_run_weights_recover_the_uniform_prior_on_the_union(n):
    th = _three_run_draws(*n)
    lw = proposal_log_weight(PARTS3(*n), th)
    assert np.isfinite(lw).all()
    w = np.exp(lw - lw.max())
    w /= w.sum()
    lo = np.array([0.1, 0.85, 0.5, 1.021])
    hi = np.array([1.0, 1.0, 5.0, 1.075])
    assert np.allclose((w[:, None] * th).sum(0), (lo + hi) / 2, atol=0.012 * (hi - lo))
    # beta below the old floor gets its prior share, 0.2 / 0.9 ...
    assert (w * (th[:, 0] < 0.3)).sum() == pytest.approx(0.2 / 0.9, abs=0.01)
    # ... and R_gamma below 1.025 still its share, 0.004 / 0.054.
    assert (w * (th[:, 3] < 1.025)).sum() == pytest.approx(0.004 / 0.054, abs=0.01)
    # E_q[p/q] = 1 with the weights unnormalised, and every part is at least
    # half uniform, so no weight blows up.
    assert np.exp(lw).mean() == pytest.approx(1.0, abs=0.03)
    assert np.exp(lw).max() < 4


def test_a_run_has_no_density_outside_its_rectangle():
    """beta 0.29 with R_gamma 1.023 is the beta strip's alone: neither the
    main run nor the R_gamma strip may count there."""
    th = np.array([[0.29, 0.97, 2.0, 1.023], [0.31, 0.97, 2.0, 1.023]])
    lw = proposal_log_weight(PARTS3(31744, 2560, 5632), th)
    n = 31744 + 2560 + 5632
    want0 = 5632 / n * a2_beta_strip(*STRIP_B).density(th[:1, :3])[0] / 0.054
    want1 = 2560 / n * A2_STRIP.density(th[1:, :3])[0] / 0.004
    prior = 1 / (0.9 * 0.15 * 4.5 * 0.054)
    assert lw == pytest.approx(np.log(prior / np.array([want0, want1])))


def test_rectangles_must_tile():
    assert check_tiling([([0.3, 1.0], MAIN), ([0.3, 1.0], STRIP), (STRIP_B, ALL)]) == \
        ((0.1, 1.0), (1.021, 1.075))
    with pytest.raises(ValueError, match="gap"):
        check_tiling([([0.3, 1.0], MAIN), ([0.3, 1.0], STRIP), ([0.1, 0.29], ALL)])
    with pytest.raises(ValueError, match="overlap"):
        check_tiling([([0.3, 1.0], MAIN), ([0.3, 1.0], STRIP), ([0.1, 0.31], ALL)])
    with pytest.raises(ValueError, match="gap"):      # the strip under main only
        check_tiling([([0.3, 1.0], MAIN), ([0.3, 1.0], STRIP), (STRIP_B, MAIN)])


# ---------------------------------------------------------- combining shards

def _run(d, proposal, seed, rrange, n, block, beta_range=None, theta=None):
    """Shards holding what load_shards reads, R_gamma per block of 16."""
    d.mkdir()
    th3 = proposal_sampler(proposal, beta_range)(n, seed)
    r = np.repeat(np.linspace(*rrange, n // 16, endpoint=False), 16)
    th = np.column_stack([th3, r]) if theta is None else theta
    for b in range(n // block):
        lo, hi = b * block, (b + 1) * block
        np.savez(d / f"shard_{b:05d}.npz", lo=lo, hi=hi, theta=th[lo:hi],
                 card=np.repeat(np.arange(lo // 16, hi // 16) % 2, 16))
    cfg = {"card_types": True, "educ": "comphs", "grid": "full", "proposal": proposal,
           "seed": seed, "rgamma_range": rrange, "theta_batch": 16, "n_households": 16,
           **({"beta_range": beta_range} if beta_range else {})}
    (d / "solver_config.json").write_text(json.dumps(cfg))
    return th


def _three(tmp_path):
    a = _run(tmp_path / "main", "a2_switched", 0, MAIN, 5 * 64, 64)
    b = _run(tmp_path / "strip", "a2_strip", 100000, STRIP, 3 * 64, 64)
    c = _run(tmp_path / "beta", "a2_beta_strip", 200000, ALL, 4 * 64, 64, beta_range=STRIP_B)
    return [tmp_path / "main", tmp_path / "strip", tmp_path / "beta"], (a, b, c)


def test_three_runs_combine_with_their_rectangles(tmp_path):
    dirs, (a, b, c) = _three(tmp_path)
    theta, _card, train, held, prop, cfg = oa.load_shards(dirs)
    assert np.array_equal(theta, np.vstack([a, b, c]))
    # Main holds out 2 shards, each strip 1 (RUN_SIZE).
    assert [s.offset for s in held] == [0, 0, 320, 512]
    assert prop == {"parts": [
        {"name": "a2_switched", "n_train": 192, "rgamma_range": MAIN},
        {"name": "a2_strip", "n_train": 128, "rgamma_range": STRIP},
        {"name": "a2_beta_strip", "n_train": 192, "rgamma_range": ALL, "beta_range": STRIP_B}]}
    assert cfg["beta_range"] == [0.1, 1.0] and cfg["rgamma_range"] == [1.021, 1.075]
    box = oa.box_of(cfg)
    assert (box.beta_low, box.beta_high, box.rgamma_low) == (0.1, 1.0, 1.021)
    assert cfg["runs"][2]["beta_range"] == STRIP_B and "beta_range" not in cfg["runs"][0]
    # The two earlier runs alone: exactly what they always returned.
    _t, _c, _tr, _h, p2, cfg2 = oa.load_shards(dirs[:2])
    assert "beta_range" not in cfg2 and all("beta_range" not in p for p in p2["parts"])
    assert oa.box_of(cfg2) == oa.box_of({"rgamma_range": [1.021, 1.075]})


def test_train_frac_takes_a_share_of_every_run_in_whole_shards(tmp_path):
    dirs, _ = _three(tmp_path)
    *_x, train, _held, prop, _cfg = oa.load_shards(dirs, train_frac=0.5)
    # 3, 2 and 3 training shards of 64 draws: halves rounded, at least one.
    assert [p["n_train"] for p in prop["parts"]] == [128, 64, 128]
    assert [s.offset for s in train] == [0, 0, 320, 512, 512]


def test_strip_draws_outside_their_range_are_refused(tmp_path):
    th = np.column_stack([a2_beta_strip(*STRIP_B).sample(128, seed=200000),
                          np.repeat(np.linspace(1.021, 1.075, 8, endpoint=False), 16)])
    _run(tmp_path / "beta", "a2_beta_strip", 200000, ALL, 128, 64, beta_range=[0.1, 0.25],
         theta=th)
    with pytest.raises(SystemExit):
        oa.load_shards(tmp_path / "beta")


def test_spec_cuts_start_at_the_box_floor():
    assert oa.Spec().cuts["beta"] == oa.CUTS["beta"]
    wide = oa.box_of({"rgamma_range": ALL, "beta_range": [0.1, 1.0]})
    assert oa.Spec(box=wide).cuts["beta"] == (0.1, 0.3, 0.8, 0.95, 1.0)


def test_population_bounds_follow_the_arm_s_box():
    lo, hi = lc.box_bounds(oa.box_of({"rgamma_range": ALL}))
    assert np.array_equal(lo, lc.BOX_LO) and np.array_equal(hi, lc.BOX_HI)
    lo, hi = lc.box_bounds(oa.box_of({"rgamma_range": ALL, "beta_range": [0.1, 1.0]}))
    assert lo[0] == 0.101 and np.array_equal(lo[1:], lc.BOX_LO[1:])


# ---------------------------------------------------------------- SBC sets

SAME_CFG = {k: 1 for k in so.SAME}


def _sbc_set(d, seed, beta_range, n=64):
    d.mkdir()
    th3 = sample_sobol(n, beta_range_box(beta_range or BETA_RANGE_A2), seed=seed)
    th = np.column_stack([th3, np.repeat(np.linspace(1.021, 1.075, n // 16, endpoint=False), 16)])
    np.savez(d / "shard_00000.npz", lo=0, hi=n, theta=th)
    cfg = {**SAME_CFG, "proposal": "uniform", "seed": seed, "rgamma_range": ALL,
           **({"beta_range": beta_range} if beta_range else {})}
    (d / "solver_config.json").write_text(json.dumps(cfg))
    return th


def test_sbc_sets_must_tile_the_box(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    (train / "solver_config.json").write_text(json.dumps(SAME_CFG))
    a = _sbc_set(tmp_path / "sbc", 40000000, None)
    b = _sbc_set(tmp_path / "sbc_strip", 41000000, STRIP_B, n=32)
    wide = oa.box_of({"rgamma_range": ALL, "beta_range": [0.1, 1.0]})
    theta, files, sets, ranges = so.load_sets([tmp_path / "sbc", tmp_path / "sbc_strip"],
                                              wide, train=train)
    assert np.array_equal(theta, np.vstack([a, b]))
    assert [f.offset for f in files] == [0, 64] and ranges == [(0.3, 1.0), (0.1, 0.3)]
    with pytest.raises(SystemExit, match="tile"):
        so.load_sets([tmp_path / "sbc"], wide, train=train)
    # The A box with the A-box set alone: as before.
    narrow = oa.box_of({"rgamma_range": ALL})
    assert len(so.load_sets([tmp_path / "sbc"], narrow, train=train)[0]) == 64
    assert so.cuts_for(narrow)["beta"] == so.CUTS["beta"]
    assert so.cuts_for(wide)["beta"] == (0.1, 0.3, 0.5, 0.8, 0.95, 1.0)
