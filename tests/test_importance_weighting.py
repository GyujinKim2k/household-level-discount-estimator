"""Importance weighting at inference time, for networks trained on the edge
mixtures (RESULTS.md 30.2, 35).

Every way this fails is silent: a sampler that forgets the weights reports the
proposal's posterior, which looks like a perfectly ordinary posterior pushed
toward the edges; weights evaluated in the wrong space (``log(1 - delta)``
instead of ``delta``) are wrong by a Jacobian; an SBC rank computed without
them tests the wrong posterior. The load-bearing test is the toy SBC: a
posterior that is exact under a proposal must come out calibrated under the
uniform prior once weighted, and must not without the weights.
"""

import numpy as np
import pytest
import torch
from scipy import stats

from hh_npe.evaluation.weighted import (
    resample,
    sample_weighted,
    sbc_from_u,
    summarize,
    weighted_quantile,
)
from hh_npe.npe.prior import (
    PHASE3_RGAMMA,
    EdgeMixture,
    SwitchedProposal,
    proposal_log_weight,
)


# --- the weight function, by the name the generation run recorded ----------

def test_proposal_log_weight_dispatches_by_name():
    th = SwitchedProposal().sample(8192, seed=0)
    th4 = np.column_stack([th, np.full(len(th), 1.05)])
    assert (proposal_log_weight({"name": "uniform"}, th4) == 0).all()
    np.testing.assert_array_equal(proposal_log_weight({"name": "edge_mixture"}, th4),
                                  EdgeMixture().log_weight(th))
    np.testing.assert_array_equal(
        proposal_log_weight({"name": "edge_mixture_switched", "n_train": 8192}, th4),
        SwitchedProposal().log_weight(th, 8192))
    with pytest.raises(ValueError):
        proposal_log_weight({"name": "edge"}, th4)


def test_switched_weights_depend_on_the_training_size():
    th = SwitchedProposal().sample(8192, seed=0)
    a = proposal_log_weight({"name": "edge_mixture_switched", "n_train": 4096}, th)
    b = proposal_log_weight({"name": "edge_mixture_switched", "n_train": 31744}, th)
    assert not np.allclose(a, b)


# --- the checkpoint guard ----------------------------------------------------

def test_load_refuses_a_proposal_checkpoint_to_unweighted_callers(tmp_path):
    from hh_npe.npe.train import load_posterior, save_posterior

    emb = torch.nn.Linear(2, 2)
    f = tmp_path / "p.pt"
    save_posterior({"stub": 1}, emb, PHASE3_RGAMMA, f,
                   proposal={"name": "edge_mixture_switched", "n_train": 31744})
    with pytest.raises(SystemExit, match="importance weights"):
        load_posterior(f)
    ck = load_posterior(f, weighted_ok=True)
    assert ck["proposal"] == {"name": "edge_mixture_switched", "n_train": 31744}
    # Uniform-prior and pre-existing checkpoints load as before.
    save_posterior({"stub": 1}, emb, PHASE3_RGAMMA, f, proposal={"name": "uniform"})
    load_posterior(f)
    save_posterior({"stub": 1}, emb, PHASE3_RGAMMA, f)
    assert "proposal" not in load_posterior(f)


# --- weighted summaries --------------------------------------------------------

def test_weighted_quantile_with_equal_weights_is_hazen():
    v = np.random.default_rng(0).normal(size=501)
    for q in (0.05, 0.5, 0.95):
        assert weighted_quantile(v, np.ones_like(v), q) == pytest.approx(
            np.quantile(v, q, method="hazen"))


def _proposal_density(th):
    """Half uniform on [0, 1], half uniform on [0.8, 1]: 0.5 below 0.8, 3.0 above."""
    return np.where(th >= 0.8, 3.0, 0.5)


def test_weights_recover_the_prior_moments():
    rng = np.random.default_rng(1)
    n = 200_000
    th = np.where(rng.random(n) < 0.5, rng.random(n), 0.8 + 0.2 * rng.random(n))
    s = summarize(th[:, None], -np.log(_proposal_density(th)))
    assert s["mean"][0] == pytest.approx(0.5, abs=3e-3)
    assert s["sd"][0] == pytest.approx(np.sqrt(1 / 12), abs=3e-3)
    assert s["lo"][0] == pytest.approx(0.05, abs=5e-3)
    assert s["hi"][0] == pytest.approx(0.95, abs=5e-3)
    r = resample(th[:, None], -np.log(_proposal_density(th)), 50_000)
    assert r.mean() == pytest.approx(0.5, abs=5e-3)


class _ExactUnderProposal:
    """Stands in for an sbi posterior: draws from the EXACT posterior under the
    proposal, x ~ N(theta, SIGMA), by inverse CDF on a fine grid."""

    SIGMA = 0.15
    GRID = np.linspace(0.0, 1.0, 4001)

    class _Est:
        def __init__(self, rng):
            self.rng = rng

        def sample(self, shape, condition):
            g = _ExactUnderProposal.GRID
            x = condition.numpy()[:, 0]
            dens = (stats.norm.pdf(x[:, None], g[None], _ExactUnderProposal.SIGMA)
                    * _proposal_density(g)[None])
            c = np.cumsum(dens, 1)
            c /= c[:, -1:]
            n = shape[0]
            u = self.rng.random((len(x), n))
            idx = np.array([np.searchsorted(c[i], u[i]) for i in range(len(x))])
            jitter = (self.rng.random(idx.shape) - 0.5) * (g[1] - g[0])
            th = np.clip(g[np.minimum(idx, len(g) - 1)] + jitter, 0.0, 1.0)
            return torch.from_numpy(th.T[..., None]).float()     # (n, B, 1)

    def __init__(self, seed):
        self.posterior_estimator = self._Est(np.random.default_rng(seed))


def test_weighted_sbc_is_calibrated_and_unweighted_is_not():
    rng = np.random.default_rng(2)
    n = 1500
    truth = rng.random(n)                                    # uniform prior
    x = torch.from_numpy(truth + _ExactUnderProposal.SIGMA * rng.standard_normal(n)
                         ).float()[:, None]
    members = [_ExactUnderProposal(3), _ExactUnderProposal(4)]   # an "ensemble"
    lw = lambda th: -np.log(_proposal_density(th[:, 0]))
    kw = dict(n_draws=1000, low=[0.0], high=[1.0], truth=truth[:, None], batch=500)

    r = sample_weighted(members, x, log_weight=lw, **kw)
    cov, ks = sbc_from_u(r["u"])
    assert cov[0] == pytest.approx(0.90, abs=0.025)
    assert ks[0] > 0.01
    # The interval and the rank agree on what covers.
    inside = (r["lo"][:, 0] <= truth) & (truth <= r["hi"][:, 0])
    assert (inside == ((r["u"][:, 0] >= 0.05) & (r["u"][:, 0] <= 0.95))).mean() > 0.98

    q = sample_weighted(members, x, log_weight=None, **kw)
    _cov_q, ks_q = sbc_from_u(q["u"])
    assert ks_q[0] < 1e-6, "the test has no power: the proposal's own posterior passed"


def test_rows_without_admissible_draws_are_nan_not_a_hang():
    class Off:
        class _E:
            def sample(self, shape, condition):
                return torch.full((shape[0], len(condition), 1), 2.0)
        posterior_estimator = _E()

    r = sample_weighted([Off()], torch.zeros(3, 1), 100, [0.0], [1.0],
                        truth=np.zeros((3, 1)))
    assert np.isnan(r["mean"]).all() and (r["in_box_frac"] == 0).all()


def test_weighting_happens_after_inversion():
    """Draws arrive in log(1 - delta) space; the weight must see delta."""
    from hh_npe.npe.prior import from_log1m, log1m_box, to_log1m

    th = SwitchedProposal().sample(4096, seed=0)
    th4 = np.column_stack([th, np.full(len(th), 1.05)])
    flow = to_log1m(th4, PHASE3_RGAMMA)
    seen = []
    prop = {"name": "edge_mixture_switched", "n_train": 4096}

    def lw(t):
        seen.append(t)
        return proposal_log_weight(prop, t)

    class Fixed:
        class _E:
            def sample(self, shape, condition):
                return torch.from_numpy(flow[:shape[0]][:, None, :]).float()
        posterior_estimator = _E()

    box = log1m_box(PHASE3_RGAMMA)
    sample_weighted([Fixed()], torch.zeros(1, 1), 4096, box.low, box.high,
                    invert=lambda t: from_log1m(t, PHASE3_RGAMMA), log_weight=lw)
    assert (seen[0][:, 1] > 0.8).all() and (seen[0][:, 1] < 1.0).all()


def test_one_per_draw_takes_one_random_row_per_draw():
    from scripts.optionA import one_per_draw

    pid = torch.from_numpy(np.repeat(np.arange(50), 16))
    rows = one_per_draw(pid, seed=0)
    assert len(rows) == 50 and len(np.unique(pid.numpy()[rows])) == 50
    assert len(np.unique(rows % 16)) > 5          # not always the first household
