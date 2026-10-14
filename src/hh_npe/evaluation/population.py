"""A population (hierarchical) layer over per-household posteriors (RESULTS 45).

A network trained on draws from a proposal ``p~`` gives
``q_i(theta) ∝ p(x_i | theta) p~(theta)``, so household ``i``'s likelihood is
``l_i ∝ q_i / p~`` (``exp(proposal_log_weight)`` is ``p / p~`` for the uniform
box prior ``p``). For a population density ``g`` on the box, the household's
marginal likelihood ``m_i(g) = ∫ l_i g`` is estimated by importance sampling,
and maximising ``sum_i log m_i(g)`` fits the population. Each household's
posterior under the fitted ``g`` (empirical Bayes) is the same weighted points
reweighted by ``g``.

**Points.** Recycling the household's own posterior draws alone
(``HouseholdDraws.from_draws``) is the standard estimator, but it has a floor:
a ``g`` much narrower than the posteriors leaves few draws carrying weight, the
log of the Monte Carlo average is then biased low, and the fit stays wide. In
4 dimensions with 409 households, households sharing one theta came out at
~0.4 of their posterior width at 2,000-8,000 draws (RESULTS 45). So the
estimator also takes **reference points** drawn near the population
(``from_mis``), at which the network's density is evaluated, and combines
them with the draws by multiple importance sampling (the balance heuristic):
with ``n_i`` draws from the household's posterior ``q~_i`` and ``R``
reference points from ``r``,

    m_i(g) ≈ 1/(n_i + R) sum_t l_i(t) g(t) / psi_i(t),
    psi_i = (n_i q~_i + R r) / (n_i + R).

Every factor but ``g`` is fixed, so fitting stays cheap. Each household gets
its **own** reference points: shared ones would give every household the same
Monte Carlo error, which does not average out over households, and the fit
then follows that one sample's clumps (it halved a spread in a test).

**Coordinates.** ``g`` is a normal (or a mixture of normals) in ``u``, each
parameter's logit on its box, ``u = log((theta - lo) / (hi - theta))``.
``u`` is unbounded, so ``g`` needs no truncation constant; in theta space
``g_theta(theta) = g_u(u(theta)) * prod_k du_k / dtheta_k``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from hh_npe.evaluation.weighted import summarize


def box_logit(theta, lo, hi) -> np.ndarray:
    """Each parameter's logit on its box."""
    th = np.asarray(theta, dtype=float)
    return np.log(th - np.asarray(lo)) - np.log(np.asarray(hi) - th)


def from_box_logit(u, lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo), np.asarray(hi)
    return lo + (hi - lo) / (1.0 + np.exp(-np.asarray(u, dtype=float)))


def box_logit_log_jacobian(theta, lo, hi) -> np.ndarray:
    """``sum_k log |du_k / dtheta_k|`` per row."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    th = np.asarray(theta, dtype=float)
    return (np.log(hi - lo) - np.log(th - lo) - np.log(hi - th)).sum(-1)


def _inside(theta, lo, hi, eps):
    """Points moved ``eps`` of the box inside an edge (a float32 value stored
    at an edge), where the logit is finite; NaN padding to the centre."""
    span = hi - lo
    th = np.asarray(theta, dtype=float)
    ok = np.isfinite(th).all(-1)
    return np.clip(np.where(ok[..., None], th, (lo + hi) / 2), lo + eps * span,
                   hi - eps * span), ok


def posterior_mean_u(theta, log_w, lo, hi, eps: float = 1e-7) -> np.ndarray:
    """Each household's weighted posterior mean in ``u`` (``theta`` (N, S, d)
    and ``log_w`` (N, S), NaN / -inf padded): the summary whose likelihood
    ``summary_likelihood`` learns."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    thc, ok = _inside(theta, lo, hi, eps)
    lw = np.where(ok & np.isfinite(log_w), log_w, -np.inf)
    w = np.exp(lw - lw.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    return (w[..., None] * box_logit(thc, lo, hi)).sum(1)


@dataclass
class HouseholdDraws:
    """Each household's weighted points: ``theta`` (N, S, d; NaN-padded, in
    its own dtype) with ``log_w`` (N, S), weights to the household's posterior
    under the uniform box prior (``-inf`` padding), and ``u`` and
    ``b = log_w + log|du/dtheta|``, weights to its likelihood in ``u``, so
    ``log m_i(g) = logsumexp_t(b_it + log g_u(u_it)) + const``."""

    theta: np.ndarray
    log_w: np.ndarray
    u: torch.Tensor           # (N, S, d) float64, padding 0
    b: torch.Tensor           # (N, S) float64, padding -inf
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def from_draws(cls, theta, log_w, lo, hi, eps: float = 1e-7) -> HouseholdDraws:
        """Posterior draws alone (recycling), with ``log_w`` the weights to the
        uniform prior (``weighted.sample_draws``)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        thc, ok = _inside(theta, lo, hi, eps)
        lw = np.asarray(log_w, dtype=float)
        ok &= np.isfinite(lw)
        if not ok.any(1).all():
            raise ValueError(f"{int((~ok.any(1)).sum())} households have no draws")
        u = np.where(ok[..., None], box_logit(thc, lo, hi), 0.0)
        lw = np.where(ok, lw, -np.inf)
        b = np.where(ok, lw + box_logit_log_jacobian(thc, lo, hi), -np.inf)
        return cls(np.asarray(theta), lw, torch.from_numpy(u), torch.from_numpy(b), lo, hi)

    @classmethod
    def from_mis(cls, theta, log_l, log_q, log_r, theta_ref, log_l_ref, log_q_ref,
                 log_r_ref, lo, hi, eps: float = 1e-7) -> HouseholdDraws:
        """Each household's posterior draws, ``theta`` (N, S, d), and its own
        reference points, ``theta_ref`` (N, R, d), combined by the balance
        heuristic (module docstring) into one set of N x (S + R) points. All
        densities are in theta, up to a constant per household that is the
        same for both:

        - ``log_l``, ``log_l_ref``: the likelihood;
        - ``log_q``, ``log_q_ref``: the density the household's draws came
          from (the network's posterior on the box);
        - ``log_r``, ``log_r_ref``: the reference density.
        """
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        th = np.concatenate([np.asarray(theta, dtype=float), np.asarray(theta_ref, float)], 1)
        S = np.asarray(theta).shape[1]
        thc, ok = _inside(th, lo, hi, eps)
        ll = np.concatenate([log_l, log_l_ref], 1).astype(float)
        lq = np.concatenate([log_q, log_q_ref], 1).astype(float)
        lr = np.concatenate([log_r, log_r_ref], 1).astype(float)
        ok &= np.isfinite(ll) & np.isfinite(lq) & np.isfinite(lr)
        if not ok[:, :S].any(1).all():
            raise ValueError(f"{int((~ok[:, :S].any(1)).sum())} households have no draws")
        if not ok[:, S:].all():
            raise ValueError(f"{int((~ok[:, S:]).sum())} reference points are not usable")
        n = ok[:, :S].sum(1).astype(float)[:, None]
        R = th.shape[1] - S
        with np.errstate(invalid="ignore"):
            log_psi = np.logaddexp(np.log(n) + lq, np.log(R) + lr) - np.log(n + R)
            lw = np.where(ok, ll - log_psi, -np.inf)
        u = np.where(ok[..., None], box_logit(thc, lo, hi), 0.0)
        b = np.where(ok, lw + box_logit_log_jacobian(thc, lo, hi), -np.inf)
        return cls(np.where(ok[..., None], th, np.nan).astype(np.float32), lw,
                   torch.from_numpy(u), torch.from_numpy(b), lo, hi)

    @property
    def device(self) -> torch.device:
        return self.u.device

    def to(self, device) -> HouseholdDraws:
        """``u`` and ``b`` on ``device``; fits then run there. Parameters
        always come back on the CPU."""
        return HouseholdDraws(self.theta, self.log_w, self.u.to(device), self.b.to(device),
                              self.lo, self.hi)

    def subset(self, idx) -> HouseholdDraws:
        idx = np.asarray(idx)
        it = torch.as_tensor(idx, device=self.device)
        return HouseholdDraws(self.theta[idx], self.log_w[idx], self.u[it], self.b[it],
                              self.lo, self.hi)

    def __len__(self) -> int:
        return len(self.u)


# ------------------------------------------------------------------ the model
#
# Parameters live in a dict of tensors so that the likelihood, its Hessian and
# the reported quantities are plain functions of them:
#   mu (k, d), log_diag (k, d), off (k, d(d-1)/2), logit_pi (k,).

@dataclass(frozen=True)
class Spec:
    """A population family: ``k`` normals in ``u``, full or diagonal; or,
    with ``theta_box`` set to ``(lo, hi)``, independent normals in theta
    truncated to that box (``mu`` and ``log_diag`` then in theta's units; one
    component, diagonal). The truncated family cannot pile mass at a box edge,
    which a wide normal in ``u`` can, so the two together say whether the
    data ask for that."""

    d: int
    k: int = 1
    diagonal: bool = False
    theta_box: tuple | None = None

    def __post_init__(self) -> None:
        if self.theta_box is not None and (self.k != 1 or not self.diagonal):
            raise ValueError("the truncated family is one diagonal component")

    def free(self) -> tuple[str, ...]:
        return (("mu", "log_diag") + (() if self.diagonal else ("off",))
                + (("logit_pi",) if self.k > 1 else ()))

    def zeros(self) -> dict[str, torch.Tensor]:
        d, k = self.d, self.k
        z = lambda *s: torch.zeros(*s, dtype=torch.float64)  # noqa: E731
        return {"mu": z(k, d), "log_diag": z(k, d), "off": z(k, d * (d - 1) // 2),
                "logit_pi": z(k)}


def chol(spec: Spec, p: dict) -> torch.Tensor:
    L = torch.diag_embed(torch.exp(p["log_diag"]))
    if not spec.diagonal and spec.d > 1:
        i, j = torch.tril_indices(spec.d, spec.d, -1)
        i, j = i.to(L.device), j.to(L.device)
        L = L + torch.zeros_like(L).index_put((torch.arange(spec.k, device=L.device)[:, None],
                                               i[None],
                                               j[None]), p["off"])
    return L


def _log_mass(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``log(Phi(b) - Phi(a))`` for ``a < b``, stable in both tails (CUDA's
    ``ndtr`` returns 0 below about -8.3, which would put ``log 0`` here)."""
    flip = a > 0
    lo_, hi_ = torch.where(flip, -b, a), torch.where(flip, -a, b)
    lb, la = torch.special.log_ndtr(hi_), torch.special.log_ndtr(lo_)
    return lb + torch.log1p(-torch.exp(la - lb))


def _truncated(spec: Spec, p: dict):
    lo, hi = (torch.as_tensor(np.asarray(v, float), device=p["mu"].device)
              for v in spec.theta_box)
    # A coordinate the data leave flat drives sd to infinity (the truncated
    # normal's uniform limit) and the fit to overflow; e^6 is already flat on
    # any box here.
    sd = torch.exp(torch.clamp(p["log_diag"][0], max=6.0))
    a, b = (lo - p["mu"][0]) / sd, (hi - p["mu"][0]) / sd
    return lo, hi, sd, a, b


def log_density(spec: Spec, p: dict, u: torch.Tensor) -> torch.Tensor:
    """``log g_u(u)`` for ``u`` of shape ``(..., d)``."""
    if spec.theta_box is not None:
        lo, hi, sd, a, b = _truncated(spec, p)
        th = lo + (hi - lo) * torch.sigmoid(u)
        z = (th - p["mu"][0]) / sd
        lp = -0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) - torch.log(sd) - _log_mass(a, b)
        # g_u = g_theta * dtheta/du, dtheta/du = (hi - lo) s (1 - s).
        return (lp + torch.log(hi - lo) + torch.nn.functional.logsigmoid(u)
                + torch.nn.functional.logsigmoid(-u)).sum(-1)
    L = chol(spec, p)
    eye = torch.eye(spec.d, dtype=L.dtype, device=L.device).expand_as(L)
    Linv = torch.linalg.solve_triangular(L, eye, upper=False)       # (k, d, d)
    sol = torch.einsum("kij,...kj->...ki", Linv, u[..., None, :] - p["mu"])
    lp = (-0.5 * (sol ** 2).sum(-1) - p["log_diag"].sum(-1)
          - 0.5 * spec.d * math.log(2 * math.pi))
    return torch.logsumexp(lp + torch.log_softmax(p["logit_pi"], 0), -1)


def moments(spec: Spec, p: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The population's overall mean and covariance in ``u``."""
    L = chol(spec, p)
    pi = torch.softmax(p["logit_pi"], 0)
    m = (pi[:, None] * p["mu"]).sum(0)
    dm = p["mu"] - m
    c = (pi[:, None, None] * (L @ L.transpose(-1, -2) + dm[:, :, None] * dm[:, None, :])).sum(0)
    return m, c


def derived(spec: Spec, p: dict) -> torch.Tensor:
    """What is reported, as one vector: the population mean in ``u`` (d), the
    log sd in ``u`` (d), the correlations (d(d-1)/2, lower triangle by row)."""
    m, c = moments(spec, p)
    sd = torch.sqrt(torch.diagonal(c))
    corr = c / (sd[:, None] * sd[None, :])
    i, j = torch.tril_indices(spec.d, spec.d, -1)
    return torch.cat([m, torch.log(sd), corr[i, j]])


def derived_names(names) -> list[str]:
    names = list(names)
    i, j = np.tril_indices(len(names), -1)
    return ([f"mean_u {n}" for n in names] + [f"log_sd_u {n}" for n in names]
            + [f"corr {names[a]},{names[b]}" for a, b in zip(i, j, strict=True)])


def sample_u(spec: Spec, p: dict, n: int, seed: int = 0) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    if spec.theta_box is not None:
        from scipy import stats

        with torch.no_grad():
            lo, hi, sd, a, b = (t.cpu().numpy() for t in _truncated(spec, p))
        # scipy's truncnorm stays exact far in a tail (a fit can put the mean
        # thousands of sds outside the box), where inverting ndtr rounds to 1.
        th = stats.truncnorm.rvs(a, b, loc=p["mu"][0].cpu().numpy(), scale=sd,
                                 size=(n, spec.d), random_state=np.random.default_rng(seed))
        span = hi - lo
        return box_logit(np.clip(th, lo + 1e-12 * span, hi - 1e-12 * span), lo, hi)
    with torch.no_grad():
        comp = torch.multinomial(torch.softmax(p["logit_pi"], 0), n, True, generator=g)
        z = torch.randn(n, spec.d, 1, generator=g, dtype=torch.float64)
        return (p["mu"][comp] + (chol(spec, p)[comp] @ z)[..., 0]).numpy()


def _on(p: dict, device) -> dict:
    return {k: v.to(device) for k, v in p.items()}


def log_marginal(spec: Spec, p: dict, hd: HouseholdDraws) -> torch.Tensor:
    """``log m_i(g)`` per household, up to its constant."""
    p = _on(p, hd.device)
    return torch.logsumexp(hd.b + log_density(spec, p, hd.u), dim=1)


def initial(spec: Spec, hd: HouseholdDraws) -> dict[str, torch.Tensor]:
    """The households' own (unpooled) posterior means in ``u``: their mean
    and covariance (one normal), or split along the first principal axis."""
    # The uniform-prior weights: the u-flat ones (b) are improper where a
    # likelihood stays flat up to a box edge.
    w = torch.softmax(torch.from_numpy(hd.log_w), 1)
    p = spec.zeros()
    if spec.theta_box is not None:
        th = from_box_logit(hd.u.cpu().numpy(), hd.lo, hd.hi)
        m = (w.numpy()[..., None] * th).sum(1)
        p["mu"][0] = torch.from_numpy(m.mean(0))
        p["log_diag"][0] = torch.from_numpy(np.log(m.std(0) + 1e-3 * (hd.hi - hd.lo)))
        return p
    m = (w[..., None] * hd.u.cpu()).sum(1).numpy()
    c = np.cov(m.T) + 1e-3 * np.eye(spec.d)
    if spec.k == 1:
        p["mu"][0] = torch.from_numpy(m.mean(0))
    else:
        _vals, vecs = np.linalg.eigh(c)
        proj = (m - m.mean(0)) @ vecs[:, -1]
        for j, q in enumerate(np.quantile(proj, (np.arange(spec.k) + 0.5) / spec.k)):
            p["mu"][j] = torch.from_numpy(m.mean(0) + q * vecs[:, -1])
        c = c / spec.k
    Lc = np.linalg.cholesky(np.diag(np.diag(c)) if spec.diagonal else c)
    p["log_diag"][:] = torch.from_numpy(np.log(np.diag(Lc)))
    if not spec.diagonal and spec.d > 1:
        p["off"][:] = torch.from_numpy(Lc[np.tril_indices(spec.d, -1)])
    return p


def fit(hd: HouseholdDraws, spec: Spec, init: dict | None = None, max_iter: int = 300,
        tol: float = 1e-7) -> tuple[dict, float]:
    """Maximum likelihood population; returns its parameters and
    ``sum_i log m_i``. The optimiser sees the mean over households, so its
    tolerances do not depend on their number. Should the optimiser end on
    non-finite parameters or likelihood, the best finite point it evaluated
    is returned instead."""
    p = {k: v.detach().clone().to(hd.device) for k, v in (init or initial(spec, hd)).items()}
    free = [p[k].requires_grad_() for k in spec.free()]
    opt = torch.optim.LBFGS(free, max_iter=max_iter, tolerance_grad=tol,
                            tolerance_change=tol * 1e-3, history_size=50,
                            line_search_fn="strong_wolfe")
    best = {"loss": math.inf, "p": None}

    def closure():
        opt.zero_grad()
        loss = -log_marginal(spec, p, hd).mean()
        loss.backward()
        if torch.isfinite(loss) and float(loss) < best["loss"]:
            best["loss"], best["p"] = float(loss), {k: v.detach().clone() for k, v in p.items()}
        return loss

    # A truncated normal's mean can run far outside the box (an exponential
    # pile at an edge, RESULTS 46), and a line search step there can overflow;
    # torch's strong-Wolfe search can then fail outright on the NaN.
    try:
        opt.step(closure)
        failed = False
    except (IndexError, RuntimeError):
        failed = True
    p = {k: v.detach() for k, v in p.items()}
    with torch.no_grad():
        ll = float(log_marginal(spec, p, hd).sum())
    if failed or not (math.isfinite(ll) and all(torch.isfinite(v).all() for v in p.values())):
        if best["p"] is None:
            raise FloatingPointError("population fit: no finite point evaluated")
        p = best["p"]
        with torch.no_grad():
            ll = float(log_marginal(spec, p, hd).sum())
    return {k: v.cpu() for k, v in p.items()}, ll


def _flat(spec: Spec, p: dict) -> torch.Tensor:
    return torch.cat([p[k].reshape(-1) for k in spec.free()])


def _unflat(spec: Spec, p: dict, flat: torch.Tensor) -> dict:
    out, i = dict(p), 0
    for k in spec.free():
        n = p[k].numel()
        out[k] = flat[i:i + n].reshape(p[k].shape)
        i += n
    return out


def hessian_cov(spec: Spec, p: dict, hd: HouseholdDraws) -> np.ndarray:
    """Covariance of :func:`derived` from the observed information (delta
    method). Unreliable at a boundary (a spread near zero); there the
    bootstrap and the null replicates are the guide."""
    p = _on(p, hd.device)
    flat0 = _flat(spec, p)
    nll = lambda f: -log_marginal(spec, _unflat(spec, p, f), hd).sum()  # noqa: E731
    der = lambda f: derived(spec, _unflat(spec, p, f))  # noqa: E731
    H = torch.autograd.functional.hessian(nll, flat0)
    J = torch.autograd.functional.jacobian(der, flat0)
    return (J @ torch.linalg.pinv(H) @ J.T).cpu().numpy()


def bootstrap(hd: HouseholdDraws, spec: Spec, base: dict, n_boot: int = 200,
              seed: int = 0) -> np.ndarray:
    """:func:`derived` refitted on households resampled with replacement,
    warm-started from ``base``; ``(n_boot, n_derived)``."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        q, _ll = fit(hd.subset(rng.integers(0, len(hd), len(hd))), spec, init=base)
        out.append(derived(spec, q).numpy())
    return np.array(out)


def pooled(hd: HouseholdDraws, spec: Spec | None = None, p: dict | None = None,
           truth: np.ndarray | None = None, probs=(0.05, 0.95)) -> dict[str, np.ndarray]:
    """Each household's posterior summaries in theta and the ESS of its
    weights: under the uniform prior (``p`` None, the network's own) or under
    the population ``g`` (pooled). With ``truth``, the weighted rank ``u``."""
    if p is None:
        lw = hd.log_w
    else:
        p = _on(p, hd.device)
        with torch.no_grad():
            lw = (hd.b + log_density(spec, p, hd.u)).cpu().numpy()
    keys = ("mean", "sd", "lo", "hi", "ess") + (("u",) if truth is not None else ())
    res = {k: [] for k in keys}
    for i in range(len(hd)):
        ok = np.isfinite(lw[i])
        s = summarize(hd.theta[i, ok].astype(float), lw[i, ok],
                      None if truth is None else truth[i], probs)
        for k in keys:
            res[k].append(s[k])
    return {k: np.array(v) for k, v in res.items()}


# ------------------------------------------------------- reference points

@dataclass(frozen=True)
class Reference:
    """The reference points' distribution: an equal mixture over ``scales``
    of the population ``(spec, p)`` with its covariance scaled by
    ``scale ** 2``, plus, with ``broad_sd``, a normal at the box's centre
    (``u = 0``) with that sd in every coordinate. Narrow scales put points
    where a narrow population puts its mass; the widest and the broad one keep
    the box covered whatever the current fit."""

    spec: Spec
    p: dict
    scales: tuple = (0.1, 0.3, 1.0, 2.0)
    broad_sd: float | None = None

    def _components(self):
        comps = [(self.spec, self._scaled(s)) for s in self.scales]
        if self.broad_sd is not None:
            d = self.spec.d
            flat = Spec(d=d, diagonal=True)
            comps.append((flat, {**flat.zeros(), "log_diag": torch.full(
                (1, d), math.log(self.broad_sd), dtype=torch.float64)}))
        return comps

    def _scaled(self, s: float) -> dict:
        return {**self.p, "log_diag": self.p["log_diag"] + math.log(s), "off": self.p["off"] * s}

    #: Points are kept within this logit of the box's edges, where theta is
    #: still distinguishable from the edge in float64 (|u| > ~36 rounds to it).
    U_MAX = 16.0

    def sample_u(self, n: int, seed: int = 0) -> np.ndarray:
        comps = self._components()
        per = -(-n // len(comps))
        u = np.concatenate([sample_u(sp, pp, per, seed + j) for j, (sp, pp) in enumerate(comps)])
        if not np.isfinite(u).all():
            raise ValueError(f"reference distribution gives non-finite points: {self.p}")
        # Clipping moves a little mass onto the bounds; the density used for
        # the weights is the unclipped one, which only misstates those few
        # points' weights, far out in the tails.
        u = np.clip(u, -self.U_MAX, self.U_MAX)
        return u[np.random.default_rng(seed).permutation(len(u))[:n]]

    def log_density_u(self, u) -> np.ndarray:
        ut = torch.as_tensor(np.asarray(u, dtype=float))
        comps = self._components()
        with torch.no_grad():
            lp = torch.stack([log_density(sp, pp, ut) for sp, pp in comps])
        return (torch.logsumexp(lp, 0) - math.log(len(comps))).numpy()

    def log_density_theta(self, theta, lo, hi) -> np.ndarray:
        """Its density in theta; NaN padding gives NaN."""
        th = np.asarray(theta, dtype=float)
        with np.errstate(invalid="ignore"):
            u = box_logit(th, lo, hi)
            return self.log_density_u(np.nan_to_num(u)) + np.where(
                np.isfinite(th).all(-1), box_logit_log_jacobian(th, lo, hi), np.nan)


def fit_adaptive(theta, log_l, log_q, evaluate, lo, hi, spec: Spec, n_ref: int = 4096,
                 scales: tuple = Reference.scales, rounds: int = 2, seed: int = 0,
                 device=None, broad_sd: float | None = None
                 ) -> tuple[dict, float, HouseholdDraws, list]:
    """The population fit with reference points (module docstring).

    ``theta`` (N, S, d), ``log_l`` and ``log_q`` (N, S): each household's
    posterior draws, its log likelihood and the log density they were drawn
    from, at them. ``evaluate(theta_ref)`` returns ``(log_l_ref, log_q_ref)``,
    each (N, R), at each household's own points ``theta_ref`` (N, R, d). A
    first fit recycles the draws alone; each round then draws ``n_ref``
    reference points per household around the current fit and refits on both.
    Returns the parameters, ``sum_i log m_i``, the final points (fixed from
    then on, for bootstraps and other families) and the fit after each
    round, with the share of that round's reference points clipped at
    ``Reference.U_MAX``. Clipped points carry the unclipped density, so a
    population that drifts to the clip (PSID's beta, RESULTS 46) gets a
    meaningless likelihood: a round with a sizeable share is not a fit."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    N, d = np.asarray(theta).shape[0], len(lo)
    hd = HouseholdDraws.from_draws(theta, np.asarray(log_l) - np.asarray(log_q), lo, hi)
    hd = hd.to(device) if device is not None else hd
    p, ll = fit(hd, spec)
    history = [{"round": 0, "derived": derived(spec, p).numpy().tolist(), "ll": ll,
                "ref_clipped": 0.0}]
    for r in range(1, rounds + 1):
        ref = Reference(spec, p, scales, broad_sd)
        u_ref = ref.sample_u(N * n_ref, seed + 1000 * r)
        clipped = float((np.abs(u_ref) >= Reference.U_MAX).any(-1).mean())
        th_ref = from_box_logit(u_ref, lo, hi).reshape(N, n_ref, d)
        l_ref, q_ref = evaluate(th_ref)
        hd = HouseholdDraws.from_mis(theta, log_l, log_q, ref.log_density_theta(theta, lo, hi),
                                     th_ref, l_ref, q_ref, ref.log_density_theta(th_ref, lo, hi),
                                     lo, hi)
        hd = hd.to(device) if device is not None else hd
        p, ll = fit(hd, spec, init=p)
        history.append({"round": r, "derived": derived(spec, p).numpy().tolist(), "ll": ll,
                        "ref_clipped": clipped})
    return p, ll, hd, history
