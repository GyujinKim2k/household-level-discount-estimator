"""The simulation audit's own checks (RESULTS 54), on cases with known answers."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from hh_npe.simulator import grids, twoasset
from scripts import audit_simulation as audit


def test_margin_is_zero_at_a_tie_and_relative_otherwise():
    from hh_npe.simulator.twoasset_gpu import _margin

    pay = torch.tensor([[[[1.0, 3.0, 3.0, 0.0]]],
                        [[[1.0, 4.0, 2.0, 0.0]]]], dtype=torch.float64)
    best = pay.argmax(dim=-1)
    u = torch.ones_like(pay)
    ev = torch.zeros_like(pay)
    m = _margin(pay, best, u, ev)
    assert m.shape == (2, 1, 1)
    assert m[0, 0, 0] == 0.0                       # 3 vs 3: a tie
    assert m[1, 0, 0] == pytest.approx(2.0)        # (4 - 2) / |u| = 2


def test_budget_check_accepts_a_simulated_panel():
    """Every year of a panel the port simulates satisfies the identity the
    audit applies to the stored shards; a perturbed consumption does not."""
    spec = twoasset.COARSE
    sol = twoasset.solve(0.7, 0.97, 2.0, spec)
    p = twoasset.simulate(sol, n_households=200, seed=3)
    pen = grids.liquidation_penalty(sol.age)
    rg = np.full((200, 1), spec.R_gamma)
    ok, gap = audit.budget_check(p["cash_on_hand"], p["consumption"], p["illiquid_assets"],
                                 p["income"], rg, sol.X, pen, spec.R, spec.R_CC)
    assert ok.all() and gap.max() < 1e-3
    bad = p["consumption"].copy()
    bad[:, 10] += 700.0
    ok2, _ = audit.budget_check(p["cash_on_hand"], bad, p["illiquid_assets"], p["income"],
                                rg, sol.X, pen, spec.R, spec.R_CC)
    assert not ok2[:, 10].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_margins_flag_changes_no_choice():
    from hh_npe.simulator.twoasset_gpu import solve_batch

    spec = dataclasses.replace(twoasset.COARSE, R_gamma=1.05)
    th = np.array([[0.6, 0.97, 2.0], [0.9, 0.99, 4.6]])
    a = solve_batch(th, spec, theta_batch=2, chunk=16)
    mg: dict = {}
    b = solve_batch(th, spec, theta_batch=2, chunk=16, margins=mg)
    for sa, sb in zip(a, b, strict=True):
        assert np.array_equal(sa.next_x, sb.next_x) and np.array_equal(sa.next_z, sb.next_z)
        assert np.array_equal(sa.cons, sb.cons)
    assert mg["beh"].shape[1] == 2
    # Never NaN and never negative wherever a choice exists. +inf is right: a
    # state with a single affordable choice has a runner-up at -1e30.
    for j, sol in enumerate(b):
        m = mg["beh"][:, j][sol.solvable]
        assert not np.isnan(m).any() and (m >= 0).all()
