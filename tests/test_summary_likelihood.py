"""The learned summary likelihood (``hh_npe.evaluation.summary_likelihood``,
RESULTS 45): on pairs whose ``p(s | theta)`` is known, the network must learn
it, widths included, since the population layer rests on the widths."""

import numpy as np
import pytest
import torch
from scipy import stats

from hh_npe.evaluation import summary_likelihood as sl


def pairs(n, seed):
    """s = 0.4 u + noise whose sd grows with |u_0|, correlated across the two
    coordinates: a shrunk, heteroskedastic summary like a posterior mean."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1.8, (n, 2))
    sd = 0.3 + 0.2 * np.abs(u[:, :1])
    e = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n) * sd
    return 0.4 * u + e, u, sd


def true_log_prob(s, u, sd):
    cov = np.array([[1, 0.5], [0.5, 1]])
    return np.array([stats.multivariate_normal.logpdf(s[i], 0.4 * u[i], cov * sd[i, 0] ** 2)
                     for i in range(len(s))])


@pytest.fixture(scope="module")
def model():
    s, u, _sd = pairs(20_000, 0)
    m, info = sl.train_mdn(s, u, np.arange(len(s)) // 4, k=2, epochs=300, seed=0)
    return m, info


def test_learns_the_density_widths_included(model):
    m, _info = model
    s, u, sd = pairs(2000, 1)
    with torch.no_grad():
        got = m.log_prob(torch.from_numpy(s), torch.from_numpy(u)).numpy()
    expect = true_log_prob(s, u, sd)
    assert np.mean(got) == pytest.approx(np.mean(expect), abs=0.03)
    assert np.mean(np.abs(got - expect)) < 0.1


def test_samples_follow_the_density(model):
    m, _info = model
    _s, u, sd = pairs(20_000, 2)
    g = torch.Generator().manual_seed(3)
    draw = m.sample(torch.from_numpy(u), generator=g).numpy()
    z = (draw - 0.4 * u) / sd
    np.testing.assert_allclose(z.std(0), [1.0, 1.0], rtol=0.05)
    assert np.corrcoef(z.T)[0, 1] == pytest.approx(0.5, abs=0.05)


def test_save_and_load_round_trip(model, tmp_path):
    m, info = model
    sl.save(m, tmp_path / "m.pt", info)
    m2, info2 = sl.load(tmp_path / "m.pt")
    s, u, _sd = pairs(10, 4)
    with torch.no_grad():
        a = m.log_prob(torch.from_numpy(s), torch.from_numpy(u))
        b = m2.log_prob(torch.from_numpy(s), torch.from_numpy(u))
    torch.testing.assert_close(a, b)
    assert info2 == info
