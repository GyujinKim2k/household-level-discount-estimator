"""Importance-weighted posterior summaries (RESULTS.md 30.2, 35).

A network trained on draws from a proposal ``p~`` learns
``q(theta | x) ∝ p(x | theta) p~(theta)``. The posterior under the uniform
prior ``p`` is ``q`` reweighted by ``w = p / p~`` (``prior.proposal_log_weight``),
so every summary here -- mean, sd, quantiles and the SBC rank -- is a weighted
one.

Order, per draw, is load-bearing: reject against the box the FLOW lives in,
then invert any target transform, then weight. Rejecting after inverting would
let ``log(1 - delta)`` draws that underflow to ``delta = 1.0`` slip through
(``psid_posterior.sample_all``); weighting before inverting would evaluate the
proposal density at ``log(1 - delta)`` instead of ``delta``.

The SBC rank becomes the weighted CDF at the truth,
``u = sum_s w_s 1[theta_s < theta_true] / sum_s w_s``, which is Uniform(0, 1)
for a calibrated posterior; the central 90% interval covers the truth iff
``0.05 <= u <= 0.95``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from scipy import stats

from hh_npe.evaluation.sbc import posterior_device


def weighted_quantile(v: np.ndarray, w: np.ndarray, q) -> np.ndarray:
    """Quantiles ``q`` of the 1-D sample ``v`` with weights ``w``.

    Midpoint plotting positions, so equal weights give numpy's ``hazen``
    method exactly.
    """
    o = np.argsort(v)
    v, w = np.asarray(v, float)[o], np.asarray(w, float)[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, c, v)


def summarize(draws: np.ndarray, log_w: np.ndarray | None = None,
              truth: np.ndarray | None = None,
              probs: tuple[float, float] = (0.05, 0.95)) -> dict:
    """Weighted mean, sd, central interval, effective sample size and (given
    ``truth``) the weighted rank ``u`` of one posterior's draws ``(n, d)``."""
    n, d = draws.shape
    lw = np.zeros(n) if log_w is None else np.asarray(log_w, float)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    mean = w @ draws
    out = {
        "mean": mean,
        "sd": np.sqrt(w @ (draws - mean) ** 2),
        "lo": np.array([weighted_quantile(draws[:, j], w, probs[0]) for j in range(d)]),
        "hi": np.array([weighted_quantile(draws[:, j], w, probs[1]) for j in range(d)]),
        "ess": float(1.0 / (w ** 2).sum()),
    }
    if truth is not None:
        out["u"] = (w[:, None] * (draws < np.asarray(truth)[None, :])).sum(0)
    return out


def sample_weighted(
    members: Sequence,
    x: torch.Tensor,
    n_draws: int,
    low,
    high,
    invert: Callable | None = None,
    log_weight: Callable | None = None,
    truth=None,
    batch: int = 256,
    min_kept: int = 20,
    max_per_call: int = 262_144,
) -> dict[str, np.ndarray]:
    """Weighted posterior summaries for every row of ``x``.

    ``members`` are sbi posteriors, mixed with equal weight (an ensemble);
    ``n_draws`` raw draws per row are split evenly across them. ``low``/``high``
    bound the space the flow lives in. ``invert`` maps kept draws to theta
    space; ``log_weight`` maps theta-space draws to log importance weights
    (``None``: unweighted, the network's own ``q``). ``truth`` (theta space,
    one row per ``x``) adds the weighted SBC rank ``u``.

    Like ``psid_posterior.sample_all`` it draws a fixed budget rather than
    rejection-sampling to a count, so a row with no mass in the box returns NaN
    and its in-box fraction instead of hanging. Returns arrays ``mean``, ``sd``,
    ``lo``, ``hi`` (5th/95th percentiles), ``in_box_frac``, ``n_kept``, ``ess``
    and, with ``truth``, ``u``.

    ``batch`` rows are sampled together, but never more than ``max_per_call``
    draws per member in one call: at 40,000 draws per row, 256 rows put 2 M
    draws through each flow and ran > 25 min on PSID where 32 rows took 86 s.
    """
    truth = None if truth is None else np.asarray(truth, float)
    keys = ("mean", "sd", "lo", "hi") + (("u",) if truth is not None else ())
    res: dict[str, list] = {k: [] for k in keys + ("in_box_frac", "n_kept", "ess")}
    for b0, frac, rows in _kept_rows(members, x, n_draws, low, high, batch, max_per_call):
        res["in_box_frac"].append(frac)
        for j, keep in enumerate(rows):
            res["n_kept"].append(len(keep))
            if len(keep) < min_kept:
                for k in keys:
                    res[k].append(np.full(keep.shape[-1], np.nan))
                res["ess"].append(np.nan)
                continue
            th = invert(keep) if invert is not None else keep
            lw = log_weight(th) if log_weight is not None else None
            r = summarize(th, lw, None if truth is None else truth[b0 + j])
            for k in keys + ("ess",):
                res[k].append(r[k])
    out = {k: np.array(v) for k, v in res.items() if k != "in_box_frac"}
    out["in_box_frac"] = np.concatenate(res["in_box_frac"])
    return out


def _sample_batch(members, xb, per: int, retries: int = 3) -> torch.Tensor:
    """``per`` draws from every member for the rows ``xb``, ``(S, B, d)``.

    nflows' rational-quadratic spline inverse occasionally asserts a negative
    discriminant in float32 (RESULTS 46, 50). The batch is then drawn again
    from where the random stream stands, up to ``retries`` times. Without an
    assertion nothing changes, so earlier results are reproduced bit for bit.
    """
    for attempt in range(retries + 1):
        try:
            return torch.cat([m.posterior_estimator.sample((per,), condition=xb)
                              for m in members], dim=0)
        except AssertionError:
            if attempt == retries:
                raise
            print(f"spline inverse asserted on a batch of {len(xb)} rows; drawing it again "
                  f"({attempt + 1}/{retries})", flush=True)
    raise AssertionError("unreachable")


def _kept_rows(members, x, n_draws, low, high, batch, max_per_call):
    """The sampling loop of :func:`sample_weighted` and :func:`sample_draws`:
    per batch of rows, ``(first row, in-box fraction per row, [each row's
    in-box flow-space draws, float64])``."""
    dev = posterior_device(members[0])
    lo_b = torch.as_tensor(np.asarray(low), dtype=torch.float32, device=dev)
    hi_b = torch.as_tensor(np.asarray(high), dtype=torch.float32, device=dev)
    per = max(1, int(np.ceil(n_draws / len(members))))
    batch = max(1, min(batch, max_per_call // per))
    with torch.no_grad():
        for b0 in range(0, len(x), batch):
            xb = x[b0:b0 + batch].to(dev)
            s = _sample_batch(members, xb, per)                  # (S, B, d)
            inb = ((s >= lo_b) & (s <= hi_b)).all(-1)
            frac = inb.float().mean(0).cpu().numpy()
            s, inb = s.cpu().numpy(), inb.cpu().numpy()
            yield b0, frac, [s[inb[:, j], j, :].astype(float) for j in range(s.shape[1])]


def sample_draws(
    members: Sequence,
    x: torch.Tensor,
    n_draws: int,
    low,
    high,
    invert: Callable | None = None,
    log_weight: Callable | None = None,
    batch: int = 256,
    max_per_call: int = 262_144,
) -> dict[str, np.ndarray]:
    """Every row's weighted draws themselves, for what needs more than
    :func:`sample_weighted`'s summaries (the population layer, RESULTS 45).

    Same sampling as :func:`sample_weighted` (reject in the flow's box, invert,
    then weight), so ``summarize(theta[i, :n_kept[i]], log_w[i, :n_kept[i]])``
    is its row ``i``. Rows are padded to the per-row budget: ``theta``
    ``(N, S, d)`` float32, NaN past ``n_kept``; ``log_w`` ``(N, S)`` float64,
    ``-inf`` past ``n_kept`` (0 for kept draws when ``log_weight`` is None);
    ``n_kept`` and ``in_box_frac`` per row.
    """
    per = max(1, int(np.ceil(n_draws / len(members))))
    s_max = per * len(members)
    d = len(np.asarray(low))
    theta = np.full((len(x), s_max, d), np.nan, dtype=np.float32)
    log_w = np.full((len(x), s_max), -np.inf)
    n_kept = np.zeros(len(x), dtype=np.int64)
    fracs = []
    for b0, frac, rows in _kept_rows(members, x, n_draws, low, high, batch, max_per_call):
        fracs.append(frac)
        for j, keep in enumerate(rows):
            n = len(keep)
            n_kept[b0 + j] = n
            if not n:
                continue
            th = invert(keep) if invert is not None else keep
            theta[b0 + j, :n] = th
            log_w[b0 + j, :n] = log_weight(th) if log_weight is not None else 0.0
    return {"theta": theta, "log_w": log_w, "n_kept": n_kept,
            "in_box_frac": np.concatenate(fracs)}


def sbc_from_u(u: np.ndarray, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Per-parameter central-interval coverage and KS-uniformity p-value from
    weighted ranks ``u`` (rows with NaN, i.e. no admissible draws, dropped)."""
    a = (1.0 - level) / 2.0
    ok = np.isfinite(u).all(1)
    u = u[ok]
    cov = ((u >= a) & (u <= 1.0 - a)).mean(0)
    ks = np.array([stats.kstest(u[:, j], "uniform").pvalue for j in range(u.shape[1])])
    return cov, ks


def resample(draws: np.ndarray, log_w: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """``n`` equally weighted draws from a weighted sample (systematic
    resampling), for plotting functions that take plain samples."""
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    pos = (np.random.default_rng(seed).random() + np.arange(n)) / n
    idx = np.minimum(np.searchsorted(np.cumsum(w), pos), len(w) - 1)
    return draws[idx]
