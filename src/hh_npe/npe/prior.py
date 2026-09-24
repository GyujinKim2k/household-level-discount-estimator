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
