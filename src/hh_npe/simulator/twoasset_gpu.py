"""CUDA backend for the two-asset backward induction, batched over theta.

The CPU solver in :mod:`hh_npe.simulator.twoasset` stays the reference
implementation -- it is the one validated against Laibson et al.'s published
simulation. This module reproduces its arithmetic on GPU and is verified
against it (``tests/test_twoasset_gpu.py``); the CPU path is never replaced.

**Why batching over theta is the whole point.** The consumption tensor
``C[i, a, j, b] = A[i, j] + B[a, b] + dividend[a]`` depends only on the grids
and the age, *not* on ``(beta, delta, rho)``. Only three things vary per draw:
the CRRA exponent inside the utility, the scalar ``beta * delta``, and the
continuation value. So one tensor build serves an entire batch of parameter
draws, and ``log C`` -- the expensive transcendental -- is computed once and
reused across the batch via ``c**(1-rho) == exp((1-rho) * log c)``.

Everything stays float64. The argmax must resolve utility gaps of ~1e-5
against ``|EV| ~ 1e2``; this is the failure that inflated simulated borrowing
by ~50% when the CPU solver ran in float32. The A100 runs float64 at a 1:2
ratio, so there is no reason to trade it away.
"""

from __future__ import annotations

import numpy as np
import torch

from hh_npe.simulator import grids
from hh_npe.simulator import laibson_calibration as cal
from hh_npe.simulator.twoasset import NEG, ModelSpec, Solution

DEFAULT_DEVICE = "cuda"

#: Larger than any choice index, so untied entries never win the ``amin`` in
#: :func:`_first_argmax`. int32 keeps the temporary at half a float64 slab.
_IDX_SENTINEL = 2**31 - 1


def _nearest_index_np(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Ties round up, matching MATLAB's ``griddedInterpolant(..., 'nearest')``.

    Kept in lockstep with ``twoasset._nearest_index``; see its docstring for why
    the tie direction is load-bearing.
    """
    idx = np.clip(np.searchsorted(grid, values), 1, len(grid) - 1)
    left, right = grid[idx - 1], grid[idx]
    return np.where(values - left < right - values, idx - 1, idx)


def _first_argmax(payoff: "torch.Tensor", ar: "torch.Tensor") -> "torch.Tensor":
    """Index of the maximum along the last axis, **ties resolved to the lowest
    index** -- the rule ``np.argmax`` follows, and therefore the CPU solver's.

    ``torch.argmax`` leaves the choice among tied entries to the reduction
    order, which varies with tensor shape, batch size and GPU architecture. That
    is not a theoretical concern here: the liquid and illiquid grids are both
    built from round dollar amounts, so 98.8% of states have at least one pair
    of ``(X', Z')`` choices leaving bitwise-identical cash on hand. Whenever
    their continuation values tie as well, an unpinned reduction silently picks
    a different -- equally optimal -- portfolio, and the two wealth paths then
    diverge for the rest of the lifecycle. Measured across an A100 and a V100,
    that moved 6 of 16 draws by up to $48,000 of illiquid assets.
    """
    amax = payoff.amax(dim=-1, keepdim=True)
    # The int32 temporary is what keeps this to half a slab; only the reduced
    # result is widened, and ``gather`` requires int64 indices.
    return torch.where(payoff == amax, ar, _IDX_SENTINEL).amin(dim=-1).long()


def _plan_batching(nX: int, nZ: int, theta_batch: int, chunk: int,
                   device: str) -> tuple[int, int]:
    """Pick ``(theta_batch, chunk)`` that fit comfortably in device memory.

    Peak live tensors per chunk are ``C`` and ``log C``, shared across the
    batch, plus ``u`` and ``payoff`` for every theta in it -- and, inside
    :func:`_first_argmax`, a bool mask (1/8 slab) and an int32 index temporary
    (1/2 slab) per theta. That is ``2.625 * B + 2`` slabs of
    ``chunk * nZ * nX * nZ`` float64 elements. Counting only ``B + 2``
    overestimates what fits by more than a factor of two, which an 80 GB card
    absorbs silently and a 16 GB one turns into an OOM mid-run.
    """
    free, _total = torch.cuda.mem_get_info(torch.device(device))
    budget = free * 0.70  # leave headroom for workspace and fragmentation
    per_slab = chunk * nZ * nX * nZ * 8
    max_b = max(1, int((int(budget / per_slab) - 2) / 2.625))
    if theta_batch > max_b:
        theta_batch = max_b
    return theta_batch, chunk


#: Below this, ``(x**(1-rho) - 1) / (1-rho)`` is replaced by its limit,
#: ``log x``. The CPU path (``twoasset._crra``) has always had this branch; the
#: GPU path did not, so ``solve_batch`` at rho == 1 -- log utility, the
#: canonical case -- divided by zero and died with a device-side assert. No
#: Sobol draw lands that close (the nearest is |rho-1| = 5e-5, where the
#: relative error is ~1e-5 in float64), so no generated dataset is affected.
RHO_LOG_EPS = 1e-6


def _safe_omr(omr: torch.Tensor) -> torch.Tensor:
    """``1 - rho`` with zeros replaced by one, for use as a denominator only.

    The numerator keeps the ORIGINAL ``omr``, so for ``|1 - rho| >= eps`` the
    arithmetic is bit-for-bit what it was before this guard existed. That
    matters: rewriting ``x ** omr`` as ``exp(omr * log x)`` is algebraically
    identical but differs in the last ulp at non-integer exponents, and with
    ~99% of states holding exactly-tied choices, last-ulp differences flip
    argmax ties and change the solution. An earlier version of this fix did
    exactly that and stopped reproducing the dataset at rho = 4.5, while
    rho = 2, 3, 5 -- where ``1 - rho`` is an integer and ``pow`` is exact --
    were unaffected.
    """
    return torch.where(omr.abs() < RHO_LOG_EPS, torch.ones_like(omr), omr)


# Both call sites below keep the ORIGINAL operation order, `a * num / den`
# rather than `a * (num / den)`. Floating-point multiplication and division do
# not associate, so the regrouped form differs in the last ulp -- which, with
# ~99% of states holding exactly-tied choices, flips argmax ties and changes
# the solution. That is how the first two attempts at this guard each stopped
# reproducing the dataset at rho = 4.5.


@torch.no_grad()
def solve_batch(
    thetas: np.ndarray,
    spec: ModelSpec = ModelSpec(),
    device: str = DEFAULT_DEVICE,
    theta_batch: int = 32,
    chunk: int = 48,
    progress: bool = False,
) -> list[Solution]:
    """Backward-induct the two-asset model for many ``(beta, delta, rho)`` draws.

    Returns one :class:`~hh_npe.simulator.twoasset.Solution` per row of
    ``thetas``, with policy arrays on the CPU so the existing forward
    simulation and wave aggregation work unchanged.
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    if thetas.ndim != 2 or thetas.shape[1] != 3:
        raise ValueError(f"thetas must be (B, 3) of (beta, delta, rho); got {thetas.shape}")
    if np.any((thetas[:, 1] <= 0.0) | (thetas[:, 1] >= 1.0)):
        raise ValueError("delta must be in (0, 1) for every draw")
    if np.any(thetas[:, 0] <= 0.0) or np.any(thetas[:, 2] <= 0.0):
        raise ValueError("beta and rho must be positive for every draw")

    dev = torch.device(device)
    f64 = torch.float64

    age = grids.ages(spec.age_start, spec.age_end)
    T = len(age)
    X_np, feas_np = grids.liquid_grid(age, spec.xjump, spec.xmax, spec.x_cells_per_step)
    Z_np = grids.illiquid_grid(spec.zjump, spec.zmax, spec.z_cells_per_step)
    states_np, P_np = grids.tauchen(n_states=spec.n_income_states, c=spec.calib)
    nX, nZ, nS = len(X_np), len(Z_np), spec.n_income_states

    hhs = grids.effective_hh_size(age, spec.calib)
    ymean = grids.mean_log_income(age, spec.calib)
    ylevel = grids.mean_income(age, spec.calib)
    zliqpen = grids.liquidation_penalty(age)
    xmin = grids.credit_limit(age, spec.xjump, spec.calib)
    death = cal.DEATH_PROB
    mean_hhs, mean_hhy = hhs.mean(), ylevel.mean()

    X = torch.as_tensor(X_np, dtype=f64, device=dev)
    Z = torch.as_tensor(Z_np, dtype=f64, device=dev)
    P = torch.as_tensor(P_np, dtype=f64, device=dev)
    feasible = torch.as_tensor(feas_np, device=dev)

    cost_next = torch.clamp(X / spec.R, min=0.0) + torch.clamp(X / spec.R_CC, max=0.0)
    A_full = X[:, None] - cost_next[None, :]
    dZ = Z[None, :] - Z[:, None]

    # Transitory-shock support and the resulting grid shifts are theta-independent,
    # so precompute them once per age instead of inside the batch loop.
    shift_plan = []
    for t in range(T):
        per_state = []
        for s2 in range(nS):
            probs, levels = grids.discretize_transitory(
                float(ymean[t] + states_np[s2]),
                xjump=spec.xjump, xmax=spec.xmax, xmin=float(xmin[t]),
                c=spec.calib, disrupt_p=spec.disrupt_p,
                disrupt_mult=spec.disrupt_mult,
            )
            idx = np.stack([_nearest_index_np(X_np, X_np + y) for y in levels])
            per_state.append((
                torch.as_tensor(probs, dtype=f64, device=dev),
                torch.as_tensor(idx, dtype=torch.long, device=dev),
            ))
        shift_plan.append(per_state)

    # Choice index for the tie-break in _first_argmax; broadcasts over (B, ., ., K).
    ar_choice = torch.arange(nX * nZ, dtype=torch.int32, device=dev)

    theta_batch, chunk = _plan_batching(nX, nZ, theta_batch, chunk, device)
    n_total = thetas.shape[0]
    out: list[Solution] = []

    for b0 in range(0, n_total, theta_batch):
        b1 = min(b0 + theta_batch, n_total)
        th = torch.as_tensor(thetas[b0:b1], dtype=f64, device=dev)
        beta, delta, rho = th[:, 0], th[:, 1], th[:, 2]
        B = b1 - b0
        naive = (spec.betahat - beta).abs() > 1e-12
        bd = (beta * delta).view(B, 1, 1, 1)
        bhat_d = (spec.betahat * delta).view(B, 1, 1, 1)
        # ``u`` is built 5-D (B, chunk, nZ, nX, nZ) before being flattened to
        # 4-D for the argmax, so the per-draw exponent needs a 5-D view too.
        omr5 = (1.0 - rho).view(B, 1, 1, 1, 1)

        next_x = torch.empty((T, B, nX, nZ, nS), dtype=torch.int32, device="cpu")
        next_z = torch.empty((T, B, nX, nZ, nS), dtype=torch.int32, device="cpu")
        cons_out = torch.empty((T, B, nX, nZ, nS), dtype=f64, device="cpu")
        solv_out = torch.empty((T, B, nX, nZ, nS), dtype=torch.bool, device="cpu")

        EV = None
        for t in range(T - 1, -1, -1):
            # --- bequest, shared across the batch except for delta and rho ---
            estate = X[:, None] + Z[None, :] * (1.0 - float(zliqpen[t]))
            annuity = max(spec.R - 1.0, 0.0) * torch.clamp(estate, min=0.0)
            r = rho.view(B, 1, 1)
            omr3, log3 = 1.0 - r, _safe_omr(1.0 - r)
            small3 = omr3.abs() < RHO_LOG_EPS
            base_num = torch.where(
                small3,
                torch.log(torch.as_tensor(mean_hhy / mean_hhs, dtype=f64,
                                          device=dev)) * log3,
                (mean_hhy / mean_hhs) ** omr3 - 1.0)
            base = mean_hhs * base_num / log3
            ratio = (mean_hhy + annuity[None]) / mean_hhs
            beq_num = torch.where(small3, torch.log(ratio) * log3,
                                  ratio ** omr3 - 1.0)
            beq = mean_hhs * beq_num / log3
            beq = (spec.alpha / (1.0 - delta.view(B, 1, 1))) * (beq - base)

            if t == T - 1:
                EV = beq[:, :, :, None].expand(B, nX, nZ, nS).clone()
                EV[:, X < 0.0, :, :] = NEG

            allowed = feasible[t + 1] if t < T - 1 else feasible[t]
            EV_t = EV.clone()
            EV_t[:, ~allowed, :, :] = NEG

            Bmat = -dZ + float(zliqpen[t]) * torch.clamp(dZ, max=0.0)
            dividend = (spec.R_gamma - 1.0) * Z
            h = float(hhs[t])

            # Centre EV exactly as the CPU path does, so the two agree bit-for-bit
            # on which choice wins rather than merely closely.
            flat = EV_t.reshape(B, nX * nZ, nS)
            live = flat > NEG / 2
            ev_ref = torch.where(
                live.any(dim=1, keepdim=True),
                torch.where(live, flat, torch.full_like(flat, -np.inf)).amax(dim=1, keepdim=True),
                torch.zeros_like(flat[:, :1, :]),
            )
            ev_c = flat - ev_ref  # (B, nX*nZ, nS)
            V = torch.empty((B, nX, nZ, nS), dtype=f64, device=dev)

            for lo in range(0, nX, chunk):
                hi = min(lo + chunk, nX)
                C = (A_full[lo:hi, None, :, None] + Bmat[None, :, None, :])
                total = C + dividend[None, :, None, None]
                # One log serves the whole theta batch: c**(1-rho) = exp((1-rho) log c)
                logc = torch.log(torch.clamp(total, min=1e-9) / h)
                del total
                bad = C < 0.0

                for s in range(nS):
                    safe5 = _safe_omr(omr5)
                    u_num = torch.where(omr5.abs() < RHO_LOG_EPS,
                                        logc[None] * safe5,
                                        torch.exp(omr5 * logc[None]) - 1.0)
                    u = h * u_num / safe5
                    u.masked_fill_(bad[None], NEG)
                    u_flat = u.reshape(B, hi - lo, nZ, nX * nZ)

                    evs = ev_c[:, :, s].view(B, 1, 1, nX * nZ)
                    payoff = u_flat + bd * evs
                    best = _first_argmax(payoff, ar_choice)
                    ok = payoff.gather(-1, best[..., None]).squeeze(-1) > NEG / 2
                    jx, jz = best // nZ, best % nZ

                    next_x[t, :, lo:hi, :, s] = jx.to(torch.int32).cpu()
                    next_z[t, :, lo:hi, :, s] = jz.to(torch.int32).cpu()
                    solv_out[t, :, lo:hi, :, s] = ok.cpu()
                    cons_out[t, :, lo:hi, :, s] = (
                        C.reshape(1, hi - lo, nZ, nX * nZ)
                        .expand(B, -1, -1, -1)
                        .gather(-1, best[..., None]).squeeze(-1).cpu()
                    )

                    ev_true = EV_t[:, :, :, s].reshape(B, 1, 1, nX * nZ)
                    if naive.any():
                        # payoff_hat = payoff + (betahat - beta) * delta * EV
                        payoff.add_((bhat_d - bd) * evs)
                        best_h = _first_argmax(payoff, ar_choice)
                        max_h = payoff.gather(-1, best_h[..., None]).squeeze(-1)
                        ev_at = ev_true.expand(B, hi - lo, nZ, -1).gather(
                            -1, best_h[..., None]).squeeze(-1)
                        # u_hat = max_hat - betahat*delta*EV, so
                        # V = u_hat + delta*EV = max_hat + (1-betahat)*delta*EV,
                        # with the centring offset added back.
                        v_naive = (
                            max_h + bhat_d.squeeze(-1) * ev_ref[:, :, s].view(B, 1, 1)
                            + (1.0 - spec.betahat) * delta.view(B, 1, 1) * ev_at
                        )
                    # Exponential branch: V = u + delta*EV at the beta*delta argmax.
                    ev_at0 = ev_true.expand(B, hi - lo, nZ, -1).gather(
                        -1, best[..., None]).squeeze(-1)
                    u_gath = u_flat.gather(-1, best[..., None]).squeeze(-1)
                    v_exp = u_gath + delta.view(B, 1, 1) * ev_at0

                    v = (torch.where(naive.view(B, 1, 1), v_naive, v_exp)
                         if naive.any() else v_exp)
                    if lo == 0 and s == 0:
                        V = torch.empty((B, nX, nZ, nS), dtype=f64, device=dev)
                    V[:, lo:hi, :, s] = v
                    del u, u_flat, payoff
                del C, logc, bad

            if t > 0:
                V[:, ~feasible[t], :, :] = NEG
                # These contractions are NOT batch-invariant, and cannot cheaply
                # be made so: cuBLAS picks its summation order from the problem
                # size, so changing theta_batch perturbs EV in the last bits and
                # flips the choice wherever two portfolios leave exactly equal
                # cash (see _first_argmax). Folding B into the row dimension of
                # a plain matmul does not fix it -- the row count varies with B
                # too. An explicit Python accumulation is invariant but runs
                # 18,447 iterations per solve, ~6x slower overall.
                #
                # So the guarantee this solver offers is reproducibility at a
                # *fixed* (device, theta_batch, chunk), not across them. A
                # dataset must therefore be generated under one configuration;
                # generate_dataset records it per shard and refuses to resume
                # under a different one.
                V_shift = torch.empty((B, nS, nX, nZ), dtype=f64, device=dev)
                for s2 in range(nS):
                    probs, idx = shift_plan[t][s2]
                    gathered = V[:, idx.reshape(-1), :, s2].reshape(
                        B, idx.shape[0], nX, nZ)
                    V_shift[:, s2] = torch.einsum("y,byxz->bxz", probs, gathered)
                    del gathered
                EV = torch.einsum("jk,bkxz->bxzj", P, V_shift)
                EV = (1.0 - float(death[t - 1])) * EV \
                    + float(death[t - 1]) * beq[:, :, :, None]
                del V_shift
            if progress and t % 20 == 0:
                print(f"    age index {t}", flush=True)

        for i in range(B):
            out.append(Solution(
                next_x=next_x[:, i].numpy(), next_z=next_z[:, i].numpy(),
                cons=cons_out[:, i].numpy(), solvable=solv_out[:, i].numpy(),
                X=X_np, Z=Z_np, feasible=feas_np, states=states_np, P=P_np,
                age=age, spec=spec, prefs=tuple(thetas[b0 + i]),
            ))
        del next_x, next_z, cons_out, solv_out, EV, V
        torch.cuda.empty_cache()

    return out
