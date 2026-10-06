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

    Setting ``rgamma_low``/``rgamma_high`` adds the illiquid return ``R_gamma``
    as a further estimated parameter, placed **last** so every existing column
    index (beta 0, delta 1, rho 2) is unchanged (RESULTS.md 32).

    Pinned values match SIMULATOR_SPEC.md.
    """

    delta_low: float = 0.85
    delta_high: float = 1.00
    crra_low: float = 0.5
    crra_high: float = 5.0
    beta_low: float | None = None
    beta_high: float | None = None
    rgamma_low: float | None = None
    rgamma_high: float | None = None

    def __post_init__(self) -> None:
        if (self.beta_low is None) != (self.beta_high is None):
            raise ValueError(
                "beta_low and beta_high must both be set or both be None; got "
                f"{self.beta_low} and {self.beta_high}"
            )
        if (self.rgamma_low is None) != (self.rgamma_high is None):
            raise ValueError("rgamma_low and rgamma_high must both be set or "
                             "both be None")
        for lo, hi, name in zip(self.low, self.high, self.names):
            if not lo < hi:
                raise ValueError(f"{name}: need low < high, got {lo} >= {hi}")

    @property
    def estimates_beta(self) -> bool:
        return self.beta_low is not None

    @property
    def estimates_rgamma(self) -> bool:
        return self.rgamma_low is not None

    @property
    def low(self) -> np.ndarray:
        base = [self.delta_low, self.crra_low]
        return np.array(([self.beta_low] if self.estimates_beta else []) + base
                        + ([self.rgamma_low] if self.estimates_rgamma else []))

    @property
    def high(self) -> np.ndarray:
        base = [self.delta_high, self.crra_high]
        return np.array(([self.beta_high] if self.estimates_beta else []) + base
                        + ([self.rgamma_high] if self.estimates_rgamma else []))

    @property
    def names(self) -> tuple[str, ...]:
        return ((("beta",) if self.estimates_beta else ()) + ("delta", "crra")
                + (("R_gamma",) if self.estimates_rgamma else ()))

    @property
    def n_params(self) -> int:
        return len(self.names)


#: Phase 3 prior: present bias unlocked (configs/npe/phase3.yaml).
PHASE3 = PriorBox(beta_low=0.30, beta_high=1.00)

#: RESULTS.md 32: the illiquid return estimated as a fourth parameter. The lower
#: bound sits above R_free = 1.0203 -- below it the illiquid asset is dominated
#: and never held (§18) -- and the range brackets Laibson et al.'s 1.05.
PHASE3_RGAMMA = PriorBox(beta_low=0.30, beta_high=1.00,
                         rgamma_low=1.025, rgamma_high=1.075)


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
                    crra_low=box.crra_low, crra_high=box.crra_high,
                    rgamma_low=box.rgamma_low, rgamma_high=box.rgamma_high)


#: Flow targets for beta's column (RESULTS.md 34, 37), the last-resort fix for
#: its short upper tail near beta = 1 (§36.1). ``log1m`` is ``log(high - beta)``,
#: moving the wall at the top to minus infinity as ``to_log1m`` does for delta;
#: ``logit`` maps ``[low, high]`` onto the whole line, removing both walls.
BETA_TRANSFORMS = ("linear", "log1m", "logit")


def _beta_unit(b, box: PriorBox, xp):
    """beta rescaled to [0, 1] and clipped ``LOG1M_EPS`` inside both ends."""
    clip = xp.clamp if xp.__name__ == "torch" else np.clip
    return clip((b - box.beta_low) / (box.beta_high - box.beta_low),
                LOG1M_EPS, 1.0 - LOG1M_EPS)


def beta_to_flow(theta, box: PriorBox, kind: str):
    """Replace beta's column (index 0) by its flow target ``kind``; a copy, numpy
    or torch in, the same kind out."""
    if kind not in BETA_TRANSFORMS:
        raise ValueError(f"beta transform {kind!r} not in {BETA_TRANSFORMS}")
    if kind == "linear":
        return theta
    if hasattr(theta, "clone"):
        import torch as xp

        out = theta.clone()
        clip = xp.clamp
    else:
        xp = np
        out = np.array(theta, copy=True, dtype=float)
        clip = np.clip
    b = out[:, 0]
    if kind == "log1m":
        out[:, 0] = xp.log(clip(box.beta_high - b, LOG1M_EPS, None))
    else:
        s = _beta_unit(b, box, xp)
        out[:, 0] = xp.log(s) - xp.log(1.0 - s)
    return out


def beta_from_flow(t, box: PriorBox, kind: str):
    """Invert :func:`beta_to_flow` (any leading shape; beta is the last axis's
    first entry)."""
    if kind == "linear":
        return t
    if hasattr(t, "clone"):
        import torch

        out, exp, sig = t.clone(), torch.exp, torch.sigmoid
    else:
        from scipy.special import expit

        out, exp, sig = np.array(t, copy=True, dtype=float), np.exp, expit
    if kind == "log1m":
        out[..., 0] = box.beta_high - exp(out[..., 0])
    else:
        out[..., 0] = box.beta_low + (box.beta_high - box.beta_low) * sig(out[..., 0])
    return out


def beta_flow_bounds(box: PriorBox, kind: str) -> tuple[float, float]:
    """Image of ``[beta_low, beta_high]`` under ``kind``, with the
    ``LOG1M_EPS`` cap wherever the map runs to infinity. ``log1m`` is
    decreasing, so its ends swap."""
    if kind == "linear":
        return float(box.beta_low), float(box.beta_high)
    if kind == "log1m":
        return float(np.log(LOG1M_EPS)), float(np.log(box.beta_high - box.beta_low))
    e = float(np.log(LOG1M_EPS) - np.log1p(-LOG1M_EPS))
    return e, -e


def beta_log_jacobian(beta, box: PriorBox, kind: str) -> np.ndarray:
    """``log |d t / d beta|`` at ``beta`` (numpy): added to a flow-space log
    density it gives the density with beta on its own scale, so held-out
    ``log q`` compares across transforms."""
    b = np.asarray(beta, dtype=float)
    if kind == "linear":
        return np.zeros_like(b)
    if kind == "log1m":
        return -np.log(np.clip(box.beta_high - b, LOG1M_EPS, None))
    s = _beta_unit(b, box, np)
    return -np.log(box.beta_high - box.beta_low) - np.log(s) - np.log1p(-s)


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
    - ``1 - delta`` log-uniform on ``[one_minus_delta_lo, one_minus_delta_hi]``,
      or with ``delta_log=False`` uniform on ``[0, one_minus_delta_hi]``
      (``one_minus_delta_lo`` unused; RESULTS 43.3)
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
    delta_log: bool = True

    def __post_init__(self) -> None:
        if self.box is None:
            object.__setattr__(self, "box", PHASE3)
        if not self.box.estimates_beta:
            raise ValueError("EdgeMixture is defined for the (beta, delta, rho) box")
        if self.box.estimates_rgamma:
            raise ValueError("EdgeMixture samples (beta, delta, rho) only; R_gamma "
                             "is drawn per block by the generator (RESULTS 32). "
                             "Pass the 3-parameter box.")
        if not 0.0 < self.frac < 1.0:
            raise ValueError(f"frac must be in (0, 1); got {self.frac}")

    def _uniform_density(self) -> float:
        return float(1.0 / np.prod(self.box.high - self.box.low))

    def _concentrated_density(self, theta: np.ndarray) -> np.ndarray:
        """Density of the concentrated component at each row (0 outside it)."""
        b, d, r = theta[:, 0], theta[:, 1], theta[:, 2]
        omd = 1.0 - d
        lo = self.one_minus_delta_lo if self.delta_log else 0.0
        hi = self.one_minus_delta_hi
        inside = ((b >= self.beta_lo) & (b <= self.box.beta_high)
                  & (omd >= lo) & (omd <= hi)
                  & (r >= self.crra_lo) & (r <= self.box.crra_high))
        f_b = 1.0 / (self.box.beta_high - self.beta_lo)
        f_r = 1.0 / (self.box.crra_high - self.crra_lo)
        if self.delta_log:
            # delta = 1 - exp(u), u uniform on [log lo, log hi]: f(delta) =
            # 1 / ((1 - delta) * log(hi / lo)).
            with np.errstate(divide="ignore"):
                f_d = 1.0 / (np.maximum(omd, 1e-300) * np.log(hi / lo))
        else:
            f_d = 1.0 / hi
        return np.where(inside, f_b * f_d * f_r, 0.0)

    def density(self, theta) -> np.ndarray:
        """The proposal's density at each row of (beta, delta, rho)."""
        th = np.asarray(theta, dtype=float)[:, :3]
        return ((1.0 - self.frac) * self._uniform_density()
                + self.frac * self._concentrated_density(th))

    def log_weight(self, theta) -> np.ndarray:
        """``log p(theta) - log p~(theta)``: the importance weight restoring the
        uniform prior. Bounded above by ``log(1 / (1 - frac))``."""
        # Only (beta, delta, rho) differ between proposal and prior. A 4th
        # column (R_gamma, RESULTS 32) is uniform under both, so its ratio is 1.
        return np.log(self._uniform_density()) - np.log(self.density(theta))

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
        omd = (np.exp(np.log(self.one_minus_delta_lo)
                      + u[:, 1] * np.log(self.one_minus_delta_hi / self.one_minus_delta_lo))
               if self.delta_log else u[:, 1] * self.one_minus_delta_hi)
        con = np.column_stack([
            self.beta_lo + u[:, 0] * (self.box.beta_high - self.beta_lo),
            1.0 - omd,
            self.crra_lo + u[:, 2] * (self.box.crra_high - self.crra_lo),
        ])
        out = np.empty((n, 3))
        out[take_c], out[~take_c] = con, uni
        return out


#: RESULTS.md 32.2: the concentrated region widened so it covers where PSID
#: households sit once R_gamma is free (85-87% of household posterior means,
#: against 39-67% for the original region).
EDGE_WIDENED = dict(beta_lo=0.60, one_minus_delta_hi=0.08, crra_lo=3.0)

#: RESULTS.md 43.3: re-aimed from the option A2 pilot (fixed solver, PSID seed
#: pool), where PSID couples moved to low rho: beta >= 0.40 and delta >= 0.95
#: cover 87% of couples' posterior means (EDGE_WIDENED: 3%). rho is
#: unrestricted, and delta is uniform in the region, not log-uniform -- the
#: log spacing would put fewer draws than the uniform prior near delta = 0.95
#: and 18% of the region's draws above 0.9997, where no couple sits.
EDGE_A2 = dict(beta_lo=0.40, one_minus_delta_hi=0.05, crra_lo=0.5, delta_log=False)


@dataclass(frozen=True)
class UniformProposal:
    """The uniform prior as a proposal: the Sobol draws ``generate_dataset.py
    --proposal uniform`` makes, so it can be the first part of a
    :class:`SwitchedProposal` (the option A2 pilot, RESULTS 43)."""

    box: PriorBox = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.box is None:
            object.__setattr__(self, "box", PHASE3)

    def _uniform_density(self) -> float:
        return float(1.0 / np.prod(self.box.high - self.box.low))

    def density(self, theta) -> np.ndarray:
        return np.full(len(np.asarray(theta)), self._uniform_density())

    def log_weight(self, theta) -> np.ndarray:
        return np.zeros(len(np.asarray(theta)))

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        # Sobol is prefix-stable: sample(n) is the first n of any longer run.
        return sample_sobol(n, self.box, seed=seed)


@dataclass(frozen=True)
class SwitchedProposal:
    """Two proposals in sequence: ``first`` for the first ``n_first`` draws (a
    pilot, already generated), ``second`` after.

    The pilot's draws are kept, not regenerated. Draws ``[0, n_first)``
    reproduce ``first.sample`` exactly; draws from ``n_first`` on are
    ``second.sample`` with their own seed. The default is RESULTS.md 32.2: the
    original EdgeMixture, then ``EdgeMixture(**EDGE_WIDENED)``. :data:`SWITCHED_A2`
    is RESULTS.md 43.3: the uniform pilot, then ``EdgeMixture(**EDGE_A2)``.

    **Weights depend on the training-set size.** A network trained on the first
    ``n`` draws learns the posterior under their empirical theta density,
    ``(n1/n) q1 + (1 - n1/n) q2`` with ``n1 = min(n, n_first)``, so
    :meth:`log_weight` takes ``n``. Both parts are at least half uniform, so
    every weight stays in (0, 2].
    """

    n_first: int = 4096
    second_seed_offset: int = 10
    first: EdgeMixture | UniformProposal = None  # type: ignore[assignment]
    second: EdgeMixture = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.first is None:
            object.__setattr__(self, "first", EdgeMixture())
        if self.second is None:
            object.__setattr__(self, "second", EdgeMixture(**EDGE_WIDENED))

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        n1 = min(n, self.n_first)
        # EdgeMixture.sample is prefix-stable, so sample(n1) is exactly the
        # first n1 draws the pilot generated with sample(n_first).
        parts = [self.first.sample(self.n_first, seed=seed)[:n1]]
        if n > n1:
            parts.append(self.second.sample(n - n1, seed=seed + self.second_seed_offset))
        return np.concatenate(parts)

    def log_weight(self, theta, n: int) -> np.ndarray:
        """``log p(theta) - log q_n(theta)`` for a network trained on the first
        ``n`` draws. Only (beta, delta, rho) matter; a 4th column is ignored."""
        th = np.asarray(theta, dtype=float)[:, :3]
        f1 = min(n, self.n_first) / n
        pu = self.first._uniform_density()
        return np.log(pu) - np.log(f1 * self.first.density(th)
                                   + (1 - f1) * self.second.density(th))


#: RESULTS.md 43.3: the option A2 run -- its 4,096 uniform pilot draws, then the
#: mixture re-aimed at where PSID couples sat in the pilot.
SWITCHED_A2 = SwitchedProposal(first=UniformProposal(), second=EdgeMixture(**EDGE_A2))

#: The proposals ``generate_dataset.py --proposal`` can record, and their
#: samplers (``n``, ``seed``) -> theta of (beta, delta, rho).
PROPOSAL_SAMPLERS = {
    "uniform": UniformProposal().sample,
    "edge_mixture": EdgeMixture().sample,
    "edge_mixture_switched": SwitchedProposal().sample,
    "a2_switched": SWITCHED_A2.sample,
}
PROPOSALS = tuple(PROPOSAL_SAMPLERS)


def proposal_log_weight(proposal: dict, theta) -> np.ndarray:
    """``log p(theta) - log p~(theta)`` for a network trained on draws from the
    named generation proposal: the importance weight back to the uniform prior.

    ``proposal`` is ``{"name": ..., "n_train": ...}``, as saved with the
    checkpoint (``train.save_posterior``) -- the name as the generation run
    recorded it in ``solver_config.json``, ``n_train`` the number of draws
    trained on, which only the switched proposal needs. ``theta`` is in theta
    space (delta, not ``log(1 - delta)``); only (beta, delta, rho) enter.
    """
    th = np.asarray(theta, dtype=float)
    name = proposal["name"]
    if name == "uniform":
        return np.zeros(len(th))
    if name == "edge_mixture":
        return EdgeMixture().log_weight(th)
    if name == "edge_mixture_switched":
        return SwitchedProposal().log_weight(th, int(proposal["n_train"]))
    if name == "a2_switched":
        return SWITCHED_A2.log_weight(th, int(proposal["n_train"]))
    raise ValueError(f"unknown proposal {name!r}; known: {PROPOSALS}")


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
