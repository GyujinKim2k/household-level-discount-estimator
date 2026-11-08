"""RESULTS 66's noise-free arms: where each arm lives, the comparisons, the
paired bootstrap, and the readers of cross-noise and PSID outputs. Synthetic
arrays only."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import annual_waves as aw
from scripts import laibson_comparison as lc
from scripts import noise_free_waves as nf
from scripts import optionA as oa
from scripts import waves_experiment as we


def test_each_arm_lives_under_its_noise():
    assert nf.run_dir("none", "a14") == Path("outputs/waves_none/a14")
    assert nf.run_dir("v2", "w07") == Path("outputs/waves/w07")
    with pytest.raises(ValueError):
        nf.run_dir("gross", "w07")
    with pytest.raises(ValueError):
        nf.run_dir("none", "x07")


def test_the_annual_pairs_are_about_the_same_years_and_the_same_count():
    (an, years), (an2, count) = nf.ANNUAL_PAIRS
    assert aw.years_read(an) == aw.years_read(years) + 1
    assert we.parse_arm(an2)[0] == we.parse_arm(count)[0]
    assert {an, years, count} == set(nf.ARMS)


def test_the_noise_free_windows_are_the_noisy_ones():
    """The chain passes every window key; the noisy biennial arms were
    trained with the defaults. Cross-noise needs the configs equal."""
    for label in nf.ARMS:
        n, wy = we.parse_arm(label)
        explicit = {**oa.WINDOW, "n_waves": n, "start_low": we.start_low(wy),
                    "start_high": we.start_high(n, wy), "wave_years": wy}
        if wy == 2:
            assert explicit == {**oa.WINDOW, "n_waves": n, "start_high": we.start_high(n)}
        else:
            assert explicit == {**oa.WINDOW, "n_waves": n, "wave_years": 1, "start_low": 25,
                                "start_high": 51}


def test_info_ratio():
    assert nf.info_ratio([0.5, 2 / 3], [0.5, 0.5]) == pytest.approx([1.0, 2.0])


def _means(rng, m=400):
    """Posterior means at theta0's stencil: each point moves the means by its
    step in u, plus household noise shared across points."""
    base = rng.normal(0, 0.05, (m, 4)) + np.array([0.44, 0.99, 1.6, 1.04])
    out = {"centre": base}
    for k in range(4):
        e = np.eye(4)[k] * lc.STEP[k]
        out[f"plus{k}"], out[f"minus{k}"] = base + e, base - e
    return out


def test_identical_arms_have_a_ratio_of_one_with_no_spread():
    rng = np.random.default_rng(0)
    m = _means(rng)
    card = (rng.uniform(size=400) < 0.8).astype(int)
    th0 = np.array([0.44, 0.99, 1.6, 1.04])
    boot = nf.boot_se({"a": m, "b": {t: v.copy() for t, v in m.items()}}, card, th0, 5, 0)
    assert boot["a"].shape == (5, 4)
    assert np.allclose(nf.ratio_ci(boot, "a", "b"), 1.0)
    noisier = {t: v + (m["centre"] - m["centre"].mean(0)) for t, v in m.items()}
    boot = nf.boot_se({"a": m, "b": noisier}, card, th0, 20, 1)
    lo, hi = np.array(nf.ratio_ci(boot, "b", "a")).T
    assert (lo > 1.5).all()                  # twice the scatter, the same Jacobian


def test_cross_noise_reads_the_cells(tmp_path):
    cell = {"log_q_weighted": 1.5, **{p: {"cov90": 0.9, "contraction": 0.6} for p in we.NAMES}}
    (tmp_path / "c.json").write_text(json.dumps({"cells": {"none on v2 noise": cell}}))
    out = nf.cross_noise(tmp_path / "c.json")
    assert out == {"none on v2 noise": {"log_q": 1.5, "cov90": [0.9] * 4,
                                        "contraction": [0.6] * 4}}


def test_psid_summary_gives_aggregates_and_the_couple_by_couple_correlation(tmp_path):
    rng = np.random.default_rng(2)
    mean = rng.normal(size=(30, 4))
    dirs = {}
    for name, m in (("w07", mean), ("none_w07", 2 * mean + 1)):
        d = tmp_path / name
        d.mkdir()
        np.savez(d / "posterior_uncorrected.npz", names=np.array(we.NAMES), mean=m,
                 sd=np.full_like(m, 0.1), lo=m - 1, hi=m + 1)
        dirs[name] = d
    dirs["missing"] = tmp_path / "missing"
    out = nf.psid_summary(dirs)
    assert set(out) == {"w07", "none_w07", "corr_none_w07_w07"}
    assert out["corr_none_w07_w07"] == pytest.approx([1.0] * 4)
    assert out["none_w07"]["median_of_means"] == pytest.approx(
        (2 * np.median(mean, 0) + 1).tolist())
    assert out["w07"]["n"] == 30


def test_noise_conditions_split_every_noise_by_feature(monkeypatch):
    from scripts import noise_free_rho as nr

    v2 = (("income", 0.0, 0.0), ("consumption", 0.0, 0.15), ("liquid_assets", 0.0, 0.1))
    gross = (("income", 0.19, 0.19), ("consumption", 0.2, 0.2))
    monkeypatch.setattr(nr, "NOISE_ARMS", {"gross": "g"})
    monkeypatch.setattr(nr, "noise_of", lambda p: gross)
    c = nr.conditions(v2)
    assert c["none"] == () and c["v2"] == v2 and c["gross"] == gross
    assert c["v2, consumption only"] == (("consumption", 0.0, 0.15),)
    assert "v2, income only" not in c                  # no noise there: nothing to split
    assert c["gross, income only"] == (("income", 0.19, 0.19),)
    assert len(c) == 1 + 3 + 3


def test_rho_summary():
    from scripts import noise_free_rho as nr

    mean = np.array([[0.4, 0.99, 0.8, 1.04], [0.5, 0.98, 2.0, 1.05], [0.6, 0.97, 3.0, 1.06]])
    s = nr.summary(mean)
    assert s["median_of_means"] == pytest.approx([0.5, 0.98, 2.0, 1.05])
    assert s["share_rho_below_1"] == pytest.approx(1 / 3)
