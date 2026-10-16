"""RESULTS 55: the regeneration's fixes, each on a case with a known answer.

- The v2 port (income rounded when planning, MATLAB's infinities and tie
  order) equals the independent reference model run with the same options,
  choice for choice and household for household; the GPU solver equals the
  CPU one, a present-biased batch holding a beta = 1 draw included.
- ``LEGACY_FIXES`` reproduces the solver every earlier dataset came from.
- ``flow_lag`` reads flows a year before stocks; 0 is the old window.
- ``--rgamma_permute`` keeps the R_gamma values and decouples them from the
  block index.
"""

from __future__ import annotations

import dataclasses
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hh_npe.data.waves import aggregate_waves
from hh_npe.simulator import reference as ref
from hh_npe.simulator import twoasset
from scripts import audit_simulation as audit
from scripts.generate_dataset import _per_block_rgamma, _solver_config

needs_package = pytest.mark.skipif(not ref.PACKAGE.exists(), reason="needs the replication package")


@needs_package
@pytest.mark.parametrize("theta, card", [((0.8, 0.98, 4.6), 1), ((0.5, 0.99, 5.0), 0)])
def test_v2_port_equals_the_reference_with_the_same_options(theta, card):
    m = ref.Model(ref.Config(**ref.COARSE, R_gamma=1.05, card=bool(card), **audit.R_V2))
    r = ref.solve(*theta, m)
    p = audit.reexpress(twoasset.solve(*theta, audit.port_spec(m, True, {})), m, theta)
    a = audit.agree(r, p, m)
    assert a["same_x"] == 1.0 and a["same_z"] == 1.0 and a["max_c_diff_where_same"] == 0.0
    u = ref.uniforms(500, m.T, seed=5)
    init = ref.initial_indices(m, 500)
    pr, pp = ref.simulate(r, u, init=init), ref.simulate(p, u, init=init)
    assert np.array_equal(pr["cash"], pp["cash"]) and np.array_equal(pr["illiquid"], pp["illiquid"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_v2_gpu_equals_cpu_with_beta_one_in_a_naive_batch():
    from hh_npe.simulator.twoasset_gpu import solve_batch

    spec = dataclasses.replace(twoasset.COARSE, R_gamma=1.05)
    th = np.array([[0.6, 0.97, 2.0], [1.0, 0.99, 1.0], [0.8, 0.98, 4.6]])
    for t, g in zip(th, solve_batch(th, spec, theta_batch=3, chunk=16), strict=True):
        c = twoasset.solve(*t, spec)
        assert np.array_equal(c.solvable, g.solvable)
        assert np.array_equal(c.next_x, g.next_x) and np.array_equal(c.next_z, g.next_z)
        assert np.abs(c.cons - g.cons).max() < 1e-6


#: sha256 prefixes of (next_x, next_z, cons, solvable) on COARSE at R_gamma
#: 1.05, recorded from the solver at commit 988cba2, before the fixes.
LEGACY_FINGERPRINT = {(0.6, 0.97, 2.0): "62264c40e73bd2ee", (0.8, 0.98, 4.6): "6076ac67c991d7ab"}


@pytest.mark.parametrize("theta", sorted(LEGACY_FINGERPRINT))
def test_legacy_fixes_reproduce_the_earlier_solver(theta):
    spec = dataclasses.replace(twoasset.COARSE, R_gamma=1.05, **twoasset.LEGACY_FIXES)
    s = twoasset.solve(*theta, spec)
    h = hashlib.sha256()
    for a in (s.next_x, s.next_z, s.cons, s.solvable):
        h.update(np.ascontiguousarray(a).tobytes())
    assert h.hexdigest()[:16] == LEGACY_FINGERPRINT[theta]


def test_flow_lag_reads_flows_a_year_before_stocks():
    T = 71
    t = np.arange(T, dtype=float)[None, :].repeat(3, 0)
    panel = {"income": 1000 + t, "consumption": 2000 + t, "liquid_assets": 3000 + t,
             "illiquid_assets": 4000 + t, "t_age": t.astype(int)}
    feats = ("income", "consumption", "liquid_assets", "illiquid_assets")
    x0, a0 = aggregate_waves(panel, 20, 30, n_waves=7, wave_years=2, features=feats)
    x1, a1 = aggregate_waves(panel, 20, 30, n_waves=7, wave_years=2, features=feats, flow_lag=1)
    t1 = 10 + 2 * np.arange(1, 8)                 # end of each wave, exclusive
    assert np.array_equal(x0[0, :, 0], 1000 + t1 - 1)
    assert np.array_equal(x1[0, :, 0], 1000 + t1 - 2)          # flows a year earlier
    assert np.array_equal(x1[0, :, 1], 2000 + t1 - 2)
    assert np.array_equal(x1[:, :, 2:], x0[:, :, 2:])           # stocks unchanged
    assert a0.all() and a1.all()
    with pytest.raises(ValueError):
        aggregate_waves(panel, 20, 20, n_waves=7, wave_years=2, features=feats, flow_lag=1)


def test_rgamma_permutation_keeps_values_and_decouples_them_from_the_block():
    n, tb = 4096, 16
    args = SimpleNamespace(theta_batch=tb, block=512, n_samples=n, rgamma_range=(1.025, 1.075),
                           seed=0, rgamma_permute=False)
    th = np.zeros((n, 3))
    base, _ = _per_block_rgamma(th, None, args)
    on = SimpleNamespace(**{**vars(args), "rgamma_permute": True})
    perm, _ = _per_block_rgamma(th, None, on)
    b, p = base[:, 3].reshape(-1, tb), perm[:, 3].reshape(-1, tb)
    assert (p == p[:, :1]).all()                                # one value per block
    assert np.array_equal(np.sort(b[:, 0]), np.sort(p[:, 0]))   # the same values
    assert not np.array_equal(b[:, 0], p[:, 0])
    rank = np.argsort(np.argsort(p[:, 0]))
    assert abs(np.corrcoef(rank, np.arange(len(rank)))[0, 1]) < 0.15
    again, _ = _per_block_rgamma(th, None, on)
    assert np.array_equal(again, perm)                          # resumable


def test_solver_config_records_the_fixes():
    a = dict(simulator="twoasset", grid="full", device="cpu", start_age=30, n_waves=7,
             wave_years=2, seed=0, no_panel=False, n_households=16, educ="comphs",
             card_types=True, proposal="a2_switched", rgamma_range=[1.025, 1.075],
             init_pool=None, liqpen_scale=1.0)
    old = _solver_config(SimpleNamespace(**a))
    new = _solver_config(SimpleNamespace(**a, rgamma_permute=True, flow_lag=1))
    assert old["round_income"] and old["matlab_infinities"] and old["tie"] == "z_first"
    assert "rgamma_permute" not in old and "flow_lag" not in old
    assert new["rgamma_permute"] is True and new["flow_lag"] == 1
