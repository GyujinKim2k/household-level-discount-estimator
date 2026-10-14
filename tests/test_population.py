"""The population layer (``hh_npe.evaluation.population``, RESULTS 45).

The load-bearing case is conjugate: households whose likelihood is normal in
``u`` (each parameter's logit on its box), whose posterior draws come from
the posterior under a prior flat in ``u``, with the likelihood and that
density both known at any point, as the network's are. There the population
fit is the normal-normal deconvolution, and each pooled posterior is the
analytic normal-normal posterior.
"""

import numpy as np
import pytest
import torch
from scipy import stats

from hh_npe.evaluation import population as pop

LO, HI = np.array([0.3, 0.85, 0.5]), np.array([1.0, 1.0, 5.0])
OMEGA_SD = np.array([0.6, 0.4, 0.5])          # each household's likelihood sd in u
MU0 = np.array([0.5, 1.5, -0.3])
SD0 = np.array([0.5, 0.3, 0.8])
CORR0 = np.array([[1.0, 0.4, 0.0], [0.4, 1.0, -0.3], [0.0, -0.3, 1.0]])
SIGMA0 = CORR0 * SD0[:, None] * SD0[None, :]
N, S, R = 1200, 1000, 2000
#: Fits run where the points are; the GPU only makes them faster.
DEVICE = "cuda" if torch.cuda.is_available() else None


class Conjugate:
    """Households with likelihood N(u(theta); y_i, Omega) and posterior draws
    from N(y_i, Omega) in u, i.e. density l_i |du/dtheta| in theta."""

    def __init__(self, u_true, seed):
        self.u_true = u_true
        rng = np.random.default_rng(seed)
        self.y = u_true + OMEGA_SD * rng.standard_normal(u_true.shape)
        u = self.y[:, None, :] + OMEGA_SD * rng.standard_normal((len(self.y), S, 3))
        self.theta = pop.from_box_logit(u, LO, HI).astype(np.float32)
        th = self.theta.astype(float)
        self.log_l, self.log_q = self._at(th)

    def _at(self, th):
        """(log l, log q) at theta of shape (N, M, 3) or shared (M, 3)."""
        u = pop.box_logit(th, LO, HI)
        if u.ndim == 2:
            u = u[None]
        ll = stats.norm.logpdf(u, self.y[:, None, :], OMEGA_SD).sum(-1)
        return ll, ll + pop.box_logit_log_jacobian(th, LO, HI)

    def evaluate(self, th_ref):
        return self._at(th_ref)

    def recycled(self):
        return pop.HouseholdDraws.from_draws(self.theta, self.log_l - self.log_q, LO, HI)


@pytest.fixture(scope="module")
def conjugate():
    rng = np.random.default_rng(0)
    c = Conjugate(rng.multivariate_normal(MU0, SIGMA0, N), seed=1)
    spec = pop.Spec(d=3)
    p, _ll, hd, _hist = pop.fit_adaptive(c.theta, c.log_l, c.log_q, c.evaluate, LO, HI, spec,
                                         n_ref=R, rounds=1, device=DEVICE)
    return c, spec, p, hd


def test_box_logit_jacobian_is_the_derivative():
    th = np.array([[0.5, 0.97, 2.0], [0.31, 0.999, 4.9]])
    h = 1e-7
    num = sum(np.log((pop.box_logit(th + h * np.eye(3)[k], LO, HI)[:, k]
                      - pop.box_logit(th - h * np.eye(3)[k], LO, HI)[:, k]) / (2 * h))
              for k in range(3))
    np.testing.assert_allclose(pop.box_logit_log_jacobian(th, LO, HI), num, rtol=1e-6)
    np.testing.assert_allclose(pop.from_box_logit(pop.box_logit(th, LO, HI), LO, HI), th)


def test_reference_density_is_its_sampling_density():
    spec = pop.Spec(d=3)
    p = {**spec.zeros(), "mu": torch.from_numpy(MU0[None]), "log_diag": torch.log(
        torch.from_numpy(SD0[None]))}
    ref = pop.Reference(spec, p, (0.5, 2.0))
    u = ref.sample_u(200_000, seed=3)
    # Mixture of N(MU0, (0.5 SD0)^2) and N(MU0, (2 SD0)^2), equal weights.
    np.testing.assert_allclose(u.var(0), 0.5 * (0.25 + 4.0) * SD0 ** 2, rtol=0.02)
    x = np.array([[0.4, 1.4, -0.2]])
    expect = np.log(0.5 * sum(stats.multivariate_normal.pdf(x[0], MU0, np.diag((s * SD0) ** 2))
                              for s in (0.5, 2.0)))
    assert ref.log_density_u(x)[0] == pytest.approx(expect, rel=1e-10)


def test_fit_recovers_the_population(conjugate):
    """Every reported quantity (mean, log sd, correlation in u) within 3
    Hessian standard errors of the households' realised values."""
    c, spec, p, hd = conjugate
    u = c.u_true
    sd = u.std(0, ddof=1)
    i, j = np.tril_indices(3, -1)
    realised = np.concatenate([u.mean(0), np.log(sd), np.corrcoef(u.T)[i, j]])
    fitted = pop.derived(spec, p).numpy()
    se = np.sqrt(np.diag(pop.hessian_cov(spec, p, hd)))
    assert (np.abs(fitted - realised) < 3 * se).all(), (fitted, realised, se)
    # And the spreads are the right size in absolute terms.
    np.testing.assert_allclose(np.exp(fitted[3:6]), SD0, rtol=0.15)


def test_hessian_se_matches_the_analytic_se_of_the_mean(conjugate):
    _c, spec, p, hd = conjugate
    cov = pop.hessian_cov(spec, p, hd)
    se = np.sqrt(np.diag(cov)[:3])
    # Each y's marginal sd is sqrt(SD0^2 + OMEGA^2); the mean's se is that over sqrt(N).
    np.testing.assert_allclose(se, np.sqrt((SD0 ** 2 + OMEGA_SD ** 2) / N), rtol=0.15)


def test_pooled_posteriors_are_the_normal_normal_posteriors(conjugate):
    c, spec, p, hd = conjugate
    m, cv = (t.numpy() for t in pop.moments(spec, p))
    om = np.diag(OMEGA_SD ** 2)
    post_cov = np.linalg.inv(np.linalg.inv(cv) + np.linalg.inv(om))
    idx = np.arange(20)
    sub = hd.subset(idx).to("cpu")
    with torch.no_grad():
        lw = (sub.b + pop.log_density(spec, p, sub.u)).numpy()
    uu = sub.u.numpy()
    w = np.exp(lw - lw.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    u_mean = (w[..., None] * uu).sum(1)
    expect = (post_cov @ (np.linalg.solve(cv, m)[:, None] + np.linalg.solve(om, c.y[idx].T))).T
    np.testing.assert_allclose(u_mean, expect, atol=0.06)
    u_sd = np.sqrt((w[..., None] * (uu - u_mean[:, None]) ** 2).sum(1))
    np.testing.assert_allclose(u_sd.mean(0), np.sqrt(np.diag(post_cov)), rtol=0.08)
    # Pooling tightens every household in u (in theta the logit's curvature
    # can widen a coordinate slightly where the pooled mean moves).
    assert (u_sd < OMEGA_SD).all()
    # pooled() summarizes the same weights in theta.
    r = pop.pooled(sub, spec, p)
    th = np.nan_to_num(sub.theta.astype(float))
    np.testing.assert_allclose(r["mean"], (w[..., None] * th).sum(1), rtol=1e-9)


def test_reference_points_remove_the_recycling_floor():
    """Households sharing one theta: recycling their draws alone leaves the
    fitted spread far from zero; with reference points it is near the
    statistical floor of N households."""
    c = Conjugate(np.tile(MU0, (N, 1)), seed=2)
    spec = pop.Spec(d=3)
    hd0 = c.recycled()
    p0, _ = pop.fit(hd0.to(DEVICE) if DEVICE else hd0, spec)
    p1, _ll, _hd, _h = pop.fit_adaptive(c.theta, c.log_l, c.log_q, c.evaluate, LO, HI, spec,
                                        n_ref=R, rounds=2, device=DEVICE)
    sd0 = np.sqrt(np.diag(pop.moments(spec, p0)[1].numpy())) / OMEGA_SD
    sd1 = np.sqrt(np.diag(pop.moments(spec, p1)[1].numpy())) / OMEGA_SD
    # The statistical floor of 1,200 households' spread estimate is ~0.2.
    assert sd0.mean() > 0.3, sd0
    assert sd1.mean() < 0.25 and (sd1 < sd0).all(), (sd0, sd1)


def test_ess_of_equal_weights_is_the_count():
    th = np.tile(np.array([0.5, 0.95, 2.0]), (1, 50, 1))
    hd = pop.HouseholdDraws.from_draws(th, np.zeros((1, 50)), LO, HI)
    assert pop.pooled(hd)["ess"][0] == pytest.approx(50)


def test_a_two_component_population_is_found():
    rng = np.random.default_rng(3)
    lo_mode, hi_mode = np.array([-1.0, 1.5, -0.3]), np.array([1.5, 1.5, -0.3])
    pick = rng.random(N) < 0.4
    c = Conjugate(np.where(pick[:, None], lo_mode, hi_mode)
                  + 0.2 * rng.standard_normal((N, 3)), seed=4)
    _p, _ll, hd, _h = pop.fit_adaptive(c.theta, c.log_l, c.log_q, c.evaluate, LO, HI,
                                       pop.Spec(d=3), n_ref=R, rounds=1, device=DEVICE)
    p2, ll2 = pop.fit(hd, pop.Spec(d=3, k=2))
    _p1, ll1 = pop.fit(hd, pop.Spec(d=3))
    assert ll2 > ll1 + 20
    mus = p2["mu"][:, 0].numpy()
    np.testing.assert_allclose(np.sort(mus), [-1.0, 1.5], atol=0.25)
    pi = torch.softmax(p2["logit_pi"], 0).numpy()
    assert pi[np.argmin(mus)] == pytest.approx(0.4, abs=0.07)


def test_truncated_family_is_a_density_on_the_box_and_samples_from_it():
    spec = pop.Spec(d=3, diagonal=True, theta_box=(tuple(LO), tuple(HI)))
    p = {**spec.zeros(), "mu": torch.tensor([[0.45, 0.99, 1.0]], dtype=torch.float64),
         "log_diag": torch.log(torch.tensor([[0.2, 0.02, 2.0]], dtype=torch.float64))}
    u = pop.sample_u(spec, p, 100_000, seed=1)
    th = pop.from_box_logit(u, LO, HI)
    assert ((th >= LO) & (th <= HI)).all()
    for k, (m, s) in enumerate([(0.45, 0.2), (0.99, 0.02), (1.0, 2.0)]):
        expect = stats.truncnorm((LO[k] - m) / s, (HI[k] - m) / s, m, s)
        assert th[:, k].mean() == pytest.approx(expect.mean(), rel=0.01)
    # The density in u integrates to 1: E over a broad reference of g / r.
    rng = np.random.default_rng(2)
    v = 4.0 * rng.standard_normal((400_000, 3))
    lr = stats.norm.logpdf(v, 0, 4.0).sum(-1)
    with torch.no_grad():
        lg = pop.log_density(spec, p, torch.from_numpy(v)).numpy()
    assert np.exp(lg - lr).mean() == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_truncation_mass_is_stable_in_both_tails(device):
    """CUDA's ndtr is 0 below about -8.3; the truncated family's mass must not
    go through it (a population centred just outside the box hits this)."""
    from hh_npe.evaluation.population import _log_mass

    a = torch.tensor([-14.9, -1.0, 3.0, -40.0, 8.0], dtype=torch.float64, device=device)
    b = torch.tensor([-8.33, 2.0, 9.0, -30.0, 12.0], dtype=torch.float64, device=device)
    expect = np.log(stats.norm.cdf(b.cpu().numpy()) - stats.norm.cdf(a.cpu().numpy()))
    tail = b.cpu().numpy() < 0
    an, bn = a.cpu().numpy(), b.cpu().numpy()
    expect[tail] = np.log(stats.norm.sf(-bn[tail]) - stats.norm.sf(-an[tail]))
    hi_tail = a.cpu().numpy() > 0
    expect[hi_tail] = np.log(stats.norm.sf(a.cpu().numpy()[hi_tail])
                             - stats.norm.sf(b.cpu().numpy()[hi_tail]))
    np.testing.assert_allclose(_log_mass(a, b).cpu().numpy(), expect, rtol=1e-10)


def test_fit_falls_back_to_the_best_finite_point(conjugate, monkeypatch):
    """A likelihood that overflows past some point (as a truncated normal's
    mean running off the box can, RESULTS 46) must not end the fit on NaN."""
    c, spec, _p, hd = conjugate
    real = pop.log_marginal

    def overflowing(spec_, p, hd_):
        out = real(spec_, p, hd_)
        return torch.where(p["mu"][0, 0] > 0.2, torch.full_like(out, float("nan")), out)

    init = pop.initial(spec, hd)
    init["mu"][0, 0] = -1.0                  # the optimum is near 0.5, past the overflow
    monkeypatch.setattr(pop, "log_marginal", overflowing)
    p, ll = pop.fit(hd, spec, init=init)
    assert np.isfinite(ll)
    assert all(torch.isfinite(v).all() for v in p.values())
    assert float(p["mu"][0, 0]) <= 0.2
