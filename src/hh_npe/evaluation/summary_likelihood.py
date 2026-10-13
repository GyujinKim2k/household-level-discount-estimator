"""The likelihood of a household's posterior summary, learned from simulation
(RESULTS 45).

The population layer needs each household's likelihood as a function of
theta. The network's implied one, ``q / p~``, is accurate enough for a single
posterior but not for deconvolution: for weakly identified beta, simulated
households drawn uniformly came out with a population a third as wide in
logit as the truth, largely because the ensemble's averaged density is wider
than any member's (RESULTS 45). Deconvolution needs the width of each
likelihood to be right, not just the posterior's coverage.

So the household is summarised by ``s``, its posterior mean in ``u`` (each
parameter's logit on its box), and ``p(s | theta)`` is learned by maximum
likelihood from simulated pairs ``(theta, s)``: its widths are measured from
how ``s`` actually scatters, by construction. Any summary gives a valid
likelihood (one that ignores the rest of the data), so population inference
on it is consistent; using only the mean costs efficiency, never validity.

The model is a mixture of ``k`` full-covariance normals in ``s`` whose
weights, means and Cholesky factors are an MLP of ``u(theta)``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch


class SummaryMDN(torch.nn.Module):
    """``p(s | theta)``: a ``k``-component normal mixture in ``s`` (dim
    ``d_s``), parameters from an MLP of ``u(theta)`` (dim ``d``)."""

    def __init__(self, d: int, d_s: int, k: int = 4, hidden: int = 128, layers: int = 3):
        super().__init__()
        self.d, self.d_s, self.k = d, d_s, k
        self.n_tril = d_s * (d_s - 1) // 2
        dims = [d] + [hidden] * layers
        mods = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            mods += [torch.nn.Linear(a, b), torch.nn.SiLU()]
        self.body = torch.nn.Sequential(*mods)
        self.head = torch.nn.Linear(hidden, k * (1 + 2 * d_s + self.n_tril))
        self.register_buffer("u_mean", torch.zeros(d))
        self.register_buffer("u_sd", torch.ones(d))
        self.register_buffer("s_mean", torch.zeros(d_s))
        self.register_buffer("s_sd", torch.ones(d_s))

    def _params(self, u: torch.Tensor):
        h = self.head(self.body((u - self.u_mean) / self.u_sd))
        k, ds = self.k, self.d_s
        logit = h[..., :k]
        mu = h[..., k:k + k * ds].reshape(*h.shape[:-1], k, ds)
        log_diag = h[..., k + k * ds:k + 2 * k * ds].reshape(*h.shape[:-1], k, ds)
        off = h[..., k + 2 * k * ds:].reshape(*h.shape[:-1], k, self.n_tril)
        return logit, mu, log_diag, off

    def log_prob(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """``log p(s | theta)`` with ``u = u(theta)``; ``s`` and ``u`` broadcast
        over leading dimensions."""
        logit, mu, log_diag, off = self._params(u)
        ds = self.d_s
        z = ((s - self.s_mean) / self.s_sd)[..., None, :] - mu          # (..., k, ds)
        # Solve L y = z by forward substitution: L lower-triangular with
        # diagonal exp(log_diag) and, below it, ``off`` in row-major order
        # (entry (r, c) at r (r - 1) / 2 + c, as torch.tril_indices).
        y = []
        for r in range(ds):
            acc = z[..., r]
            for c in range(r):
                acc = acc - off[..., r * (r - 1) // 2 + c] * y[c]
            y.append(acc * torch.exp(-log_diag[..., r]))
        y = torch.stack(y, -1)
        lp = (-0.5 * (y ** 2).sum(-1) - log_diag.sum(-1) - 0.5 * ds * math.log(2 * math.pi)
              - torch.log(self.s_sd).sum())
        return torch.logsumexp(lp + torch.log_softmax(logit, -1), -1)

    @torch.no_grad()
    def sample(self, u: torch.Tensor, generator=None) -> torch.Tensor:
        """One ``s`` per row of ``u``."""
        logit, mu, log_diag, off = self._params(u)
        comp = torch.multinomial(torch.softmax(logit, -1), 1, generator=generator)[..., 0]
        sel = lambda t: t[torch.arange(len(u)), comp]  # noqa: E731
        m, ld, of = sel(mu), sel(log_diag), sel(off)
        L = torch.diag_embed(torch.exp(ld))
        i, j = torch.tril_indices(self.d_s, self.d_s, -1)
        L[:, i, j] = of
        e = torch.randn(len(u), self.d_s, 1, generator=generator, dtype=u.dtype)
        return (m + (L @ e)[..., 0]) * self.s_sd + self.s_mean


def train_mdn(s: np.ndarray, u: np.ndarray, groups: np.ndarray, k: int = 4, epochs: int = 400,
              batch: int = 1024, lr: float = 1e-3, val_frac: float = 0.1, patience: int = 30,
              seed: int = 0, device: str = "cpu") -> tuple[SummaryMDN, dict]:
    """Maximum likelihood on pairs ``(u(theta), s)``; validation by ``groups``
    (draw ids: households of one theta stay on one side), early stopping."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    val_g = rng.choice(ug, int(len(ug) * val_frac), replace=False)
    val = np.isin(groups, val_g)
    model = SummaryMDN(u.shape[1], s.shape[1], k).double()
    model.u_mean[:] = torch.from_numpy(u[~val].mean(0))
    model.u_sd[:] = torch.from_numpy(u[~val].std(0))
    model.s_mean[:] = torch.from_numpy(s[~val].mean(0))
    model.s_sd[:] = torch.from_numpy(s[~val].std(0))
    model.to(device)
    tr = [torch.from_numpy(a[~val]).to(device) for a in (s, u)]
    va = [torch.from_numpy(a[val]).to(device) for a in (s, u)]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    best, best_state, wait, hist = np.inf, None, 0, []
    n = len(tr[0])
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for b0 in range(0, n, batch):
            idx = perm[b0:b0 + batch]
            loss = -model.log_prob(tr[0][idx], tr[1][idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(-model.log_prob(*va).mean())
        sched.step(v)
        hist.append(v)
        if v < best - 1e-4:
            best, wait = v, 0
            best_state = {kk: t.detach().clone() for kk, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.cpu().eval()
    return model, {"val_nll": best, "epochs": len(hist), "n_train": int(n),
                   "n_val": int(val.sum()), "val_groups": val_g.tolist()}


def save(model: SummaryMDN, path: Path, info: dict) -> None:
    torch.save({"state": model.state_dict(), "d": model.d, "d_s": model.d_s, "k": model.k,
                "info": info}, path)


def load(path: Path) -> tuple[SummaryMDN, dict]:
    ck = torch.load(path, weights_only=False)
    model = SummaryMDN(ck["d"], ck["d_s"], ck["k"]).double()
    model.load_state_dict(ck["state"])
    return model.eval(), ck["info"]
