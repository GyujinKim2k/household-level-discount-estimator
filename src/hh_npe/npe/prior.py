"""Sobol-based prior sampler over the structural parameters, plus a BoxUniform.

The Sobol sequence is used to draw parameter samples that we'll feed through
the simulator to generate ``(theta, x)`` training pairs. The companion
``BoxUniform`` provides the log-density that sbi needs for NPE training. Both
correspond to the uniform distribution on the same box, so density evaluations
remain valid even though sampling is quasi-random rather than IID.

Power-of-2 sample sizes (``n_samples = 2**m``) preserve Sobol's low-discrepancy
guarantees most cleanly; non-power-of-2 sizes work but lose some of the
sequence's balance properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import qmc

if TYPE_CHECKING:
    from sbi.utils import BoxUniform


@dataclass(frozen=True)
class PriorBox:
    """Box bounds on the estimated structural parameters.

    Two-parameter by default — ``(delta, crra)``, the Phase 1-2 setup with
    present bias locked at ``beta = 1``. Setting ``beta_low``/``beta_high``
    adds ``beta`` as a third estimated parameter and puts it **first**, matching
    Laibson et al.'s ``prefs`` ordering ``[beta delta rho]``.

    Pinned values match SIMULATOR_SPEC.md.
    """

    delta_low: float = 0.85
    delta_high: float = 1.00
    crra_low: float = 0.5
    crra_high: float = 5.0
    beta_low: float | None = None
    beta_high: float | None = None

    def __post_init__(self) -> None:
        if (self.beta_low is None) != (self.beta_high is None):
            raise ValueError(
                "beta_low and beta_high must both be set or both be None; got "
                f"{self.beta_low} and {self.beta_high}"
            )
        for lo, hi, name in zip(self.low, self.high, self.names):
            if not lo < hi:
                raise ValueError(f"{name}: need low < high, got {lo} >= {hi}")

    @property
    def estimates_beta(self) -> bool:
        return self.beta_low is not None

    @property
    def low(self) -> np.ndarray:
        base = [self.delta_low, self.crra_low]
        return np.array(([self.beta_low] if self.estimates_beta else []) + base)

    @property
    def high(self) -> np.ndarray:
        base = [self.delta_high, self.crra_high]
        return np.array(([self.beta_high] if self.estimates_beta else []) + base)

    @property
    def names(self) -> tuple[str, ...]:
        return (("beta",) if self.estimates_beta else ()) + ("delta", "crra")

    @property
    def n_params(self) -> int:
        return len(self.names)


#: Phase 3 prior: present bias unlocked (configs/npe/phase3.yaml).
PHASE3 = PriorBox(beta_low=0.30, beta_high=1.00)


#: ``delta = 1`` is unreachable under the log transform below, so the
#: transformed box needs a cap. ``1e-6`` sits well beyond the largest Sobol draw
#: (``1 - 1.15e-6``), so no training point is clipped.
LOG1M_EPS = 1e-6


def delta_index(box: PriorBox = PHASE3) -> int:
    """Column of ``delta``. Not a constant: it is 1 with beta, 0 without."""
    return box.names.index("delta")


def to_log1m(theta, box: PriorBox = PHASE3):
    """Reparameterise ``delta`` as ``log(1 - delta)``, in place of the column.

    ``delta = 1`` is a hard wall that a normalising flow must press a spike
    against, and on PSID roughly a quarter of households want their posterior
    there (RESULTS.md 12.6, 13.1). Under this map the wall moves to minus
    infinity and truncation becomes structurally impossible, since
    ``1 - exp(d') < 1`` for every finite ``d'``.

    Accepts a torch tensor or a numpy array and returns the same kind. The map
    is monotone *decreasing*, which matters for SBC: ranks flip to
    ``n - rank``, leaving uniformity and central-interval coverage unchanged.
    """
    j = delta_index(box)
    if hasattr(theta, "clone"):          # torch
        import torch

        out = theta.clone()
        out[:, j] = torch.log(torch.clamp(1.0 - theta[:, j], min=LOG1M_EPS))
        return out
    out = np.array(theta, copy=True)
    out[:, j] = np.log(np.clip(1.0 - out[:, j], LOG1M_EPS, None))
    return out


def from_log1m(t, box: PriorBox = PHASE3):
    """Invert :func:`to_log1m`, so every reported number is in delta space."""
    j = delta_index(box)
    if hasattr(t, "clone"):              # torch
        import torch

        out = t.clone()
        out[..., j] = 1.0 - torch.exp(t[..., j])
        return out
    out = np.array(t, copy=True)
    out[..., j] = 1.0 - np.exp(out[..., j])
    return out


def log1m_box(box: PriorBox = PHASE3) -> PriorBox:
    """``box`` with the delta axis replaced by its ``log(1 - delta)`` image.

    The bounds swap ends because the map is decreasing: the tight end of delta
    (``delta_high``) becomes the *low* end in log space.
    """
    return PriorBox(beta_low=box.beta_low, beta_high=box.beta_high,
                    delta_low=float(np.log(LOG1M_EPS)),
                    delta_high=float(np.log(1.0 - box.delta_low)),
                    crra_low=box.crra_low, crra_high=box.crra_high)


def sample_sobol(
    n_samples: int,
    box: PriorBox = PriorBox(),
    seed: int = 0,
) -> np.ndarray:
    """Draw ``n_samples`` quasi-random points from the uniform prior on ``box``.

    Uses a scrambled Sobol sequence (``scipy.stats.qmc.Sobol``); returns an
    array of shape ``(n_samples, box.n_params)`` with columns ordered as
    ``box.names``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    engine = qmc.Sobol(d=box.n_params, scramble=True, seed=seed)
    u = engine.random(n_samples)
    return qmc.scale(u, box.low, box.high)


@dataclass(frozen=True)
class EdgeMixture:
    """Training proposal: half uniform on the prior box, half concentrated where
    PSID households sit (RESULTS.md 30).

    Under the uniform prior only 6.7% of draws have delta > 0.99 while 54% of
    PSID households' posteriors sit there, and SBC coverage collapses at the
    upper edges (beta >= 0.95: ~0.52; delta >= 0.98: ~0.71). The concentrated
    component puts training draws where they are needed:

    - beta uniform on ``[beta_lo, box.beta_high]``
    - ``1 - delta`` log-uniform on ``[one_minus_delta_lo, one_minus_delta_hi]``
    - rho uniform on ``[crra_lo, box.crra_high]``

    **The inference prior stays uniform.** A network trained on draws from this
    proposal learns ``q(theta|x) ∝ p(x|theta) p~(theta)``; posterior draws are
    reweighted by ``p(theta) / p~(theta)`` (:meth:`log_weight`). Because half the
    proposal IS the prior, ``p~ >= 0.5 p`` everywhere and every weight lies in
    ``(0, 2]`` -- the correction can never blow up.
    """

    box: PriorBox = None  # type: ignore[assignment]
    frac: float = 0.5
    beta_lo: float = 0.75
    one_minus_delta_lo: float = 1e-4
    one_minus_delta_hi: float = 0.05
    crra_lo: float = 3.5

    def __post_init__(self) -> None:
        if self.box is None:
            object.__setattr__(self, "box", PHASE3)
        if not self.box.estimates_beta:
            raise ValueError("EdgeMixture is defined for the (beta, delta, rho) box")
        if not 0.0 < self.frac < 1.0:
            raise ValueError(f"frac must be in (0, 1); got {self.frac}")

    def _uniform_density(self) -> float:
        return float(1.0 / np.prod(self.box.high - self.box.low))

    def _concentrated_density(self, theta: np.ndarray) -> np.ndarray:
        """Density of the concentrated component at each row (0 outside it)."""
        b, d, r = theta[:, 0], theta[:, 1], theta[:, 2]
        omd = 1.0 - d
        lo, hi = self.one_minus_delta_lo, self.one_minus_delta_hi
        inside = ((b >= self.beta_lo) & (b <= self.box.beta_high)
                  & (omd >= lo) & (omd <= hi)
                  & (r >= self.crra_lo) & (r <= self.box.crra_high))
        f_b = 1.0 / (self.box.beta_high - self.beta_lo)
        f_r = 1.0 / (self.box.crra_high - self.crra_lo)
        # delta = 1 - exp(u), u uniform on [log lo, log hi]: f(delta) =
        # 1 / ((1 - delta) * log(hi / lo)).
        with np.errstate(divide="ignore"):
            f_d = 1.0 / (np.maximum(omd, 1e-300) * np.log(hi / lo))
        return np.where(inside, f_b * f_d * f_r, 0.0)

    def log_weight(self, theta) -> np.ndarray:
        """``log p(theta) - log p~(theta)``: the importance weight restoring the
        uniform prior. Bounded above by ``log(1 / (1 - frac))``."""
        th = np.asarray(theta, dtype=float)
        pu = self._uniform_density()
        mix = (1.0 - self.frac) * pu + self.frac * self._concentrated_density(th)
        return np.log(pu) - np.log(mix)

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """``n`` draws, uniform and concentrated components interleaved.

        Each component is its own scrambled Sobol sequence, interleaved so that
        every even-length prefix is exactly half and half and each half keeps
        Sobol balance. Deterministic in ``seed`` -- the draws are also stored in
        every shard, so training never has to re-derive them.
        """
        i = np.arange(n)
        # Row i is concentrated when floor((i+1)*frac) steps up: exactly
        # floor(n*frac) such rows, spread evenly (every other row at 0.5).
        take_c = np.floor((i + 1) * self.frac) > np.floor(i * self.frac)
        n_c = int(take_c.sum())
        n_u = n - n_c
        uni = sample_sobol(n_u, self.box, seed=seed)
        u = qmc.Sobol(d=3, scramble=True, seed=seed + 1).random(n_c)
        con = np.column_stack([
            self.beta_lo + u[:, 0] * (self.box.beta_high - self.beta_lo),
            1.0 - np.exp(np.log(self.one_minus_delta_lo)
                         + u[:, 1] * np.log(self.one_minus_delta_hi
                                            / self.one_minus_delta_lo)),
            self.crra_lo + u[:, 2] * (self.box.crra_high - self.crra_lo),
        ])
        out = np.empty((n, 3))
        out[take_c], out[~take_c] = con, uni
        return out


def make_sbi_prior(box: PriorBox = PriorBox(),
                   device: str = "cpu") -> "BoxUniform":
    """Return an sbi-compatible ``BoxUniform`` prior on the same ``box``.

    Used by sbi during NPE training for log-density evaluation. Sampling from
    this object is pseudo-random (not Sobol); use :func:`sample_sobol` to
    generate training points and pass them via sbi's pre-existing-samples
    interface.

    ``device`` must match the device passed to ``SNPE_C``; sbi asserts on the
    mismatch rather than moving the tensors itself.
    """
    import torch
    from sbi.utils import BoxUniform

    return BoxUniform(
        low=torch.tensor(box.low, dtype=torch.float32, device=device),
        high=torch.tensor(box.high, dtype=torch.float32, device=device),
    )
