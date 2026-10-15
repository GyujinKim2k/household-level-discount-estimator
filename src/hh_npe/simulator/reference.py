"""Independent reference implementation of the two-asset lifecycle model (RESULTS 54).

**For audits only; nothing in production imports it.** The production solver
is :mod:`hh_npe.simulator.twoasset_gpu`, a port of Laibson, Lee, Maxted,
Repetto and Tobacman's MATLAB with :mod:`hh_npe.simulator.twoasset` as its CPU
reference. RESULTS 40 showed that faithfulness to their code is not enough:
their utility formula loses float64 resolution above rho ~ 4.2 and the port
inherited the failure. So this module is written again from the model's
equations, sharing no code with the port, and every audit check compares three
things: their MATLAB run under Octave, this module, and the port.

Written for clarity, not speed: one loop per state, the choice set vectorised,
float64 throughout, and every grid rebuilt here rather than taken from
:mod:`hh_npe.simulator.grids`. First-stage parameters are read straight from
the replication package's ``.mat`` and CSV files, not from
:mod:`hh_npe.simulator.laibson_calibration`, so those frozen constants get
checked too.

The model, for age index ``t = 0..T-1`` (ages 20..90):

- **State:** cash on hand ``x`` after this year's income, on the age's own
  liquid grid ``X_t = [-xmin_t, ..., -xjump, Xpos]``; illiquid balance ``z``;
  persistent income state ``s``.
- **Choice:** next year's liquid position ``x'`` on ``X_{t+1}`` (``X_t`` at the
  last age) and illiquid balance ``z'``, with
  ``c = x - cost(x') - (z' - z) + pen_t * min(z' - z, 0) >= 0`` and
  ``cost(x') = x'/R`` for ``x' >= 0``, ``x'/R_CC`` for ``x' < 0``.
- **Utility:** ``h_t * u((c + (R_gamma - 1) z) / h_t)``, CRRA ``u``. The
  dividend is consumed in the year it is earned.
- **Naive quasi-hyperbolic:** the current self maximises
  ``u + beta * delta * EV_t(x', z', s)``. The value it hands back is the one
  its believed future self (``betahat``) would get:
  ``V_t = u(c^) + delta * EV_t(x^', z^', s)`` at the ``betahat`` argmax.
- **Expectation:** ``EV_{t-1}(x', z', s) = (1 - d_{t-1}) * sum_{s2} P(s, s2)
  * sum_nu p_nu(s2) * V_t(snap_t(x' + y_nu), z', s2) + d_{t-1} * B_t(x', z')``,
  where ``snap_t`` is the nearest point of ``X_t`` (midpoints round up).
- **Terminal:** ``EV_{T-1} = B_{T-1}``, with ``x' < 0`` forbidden (no Ponzi).
- **Bequest:** ``B_t = alpha / (1 - delta) * [U(Ybar + (R - 1) *
  max(x' + (1 - pen_t) z', 0)) - U(Ybar)]``, ``U`` at mean household size.

Options the port does not have, so each audit question can be asked directly:
``centred`` (utility centred at mean income per adult, RESULTS 41, or $1 as
in MATLAB) and ``tie`` (``"z_first"``, MATLAB's ``max`` then ``max``; or
``"x_first"``, the port's flat argmax).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import norm

PACKAGE = Path(__file__).resolve().parents[3] / "replication-package-LLMRT"


# ------------------------------------------------------------- first stage

def first_stage(educ: str = "comphs", package: Path = PACKAGE) -> dict:
    """Their ``FirstStageParams.m``, read from the package's own files."""
    d = package / "LifecycleSimulation" / "input" / educ
    dem = loadmat(d / "est_firststage_demographics.mat")["est_demographics"].ravel()
    inc = loadmat(d / "est_firststage_income.mat")
    est_income, est_ar1 = inc["est_income"].ravel(), inc["est_ar1"].ravel()
    cred = loadmat(d / "est_firststage_creditlim.mat")["est_creditlim"].ravel()
    iw = loadmat(d / "est_firststage_initwealth.mat")["est_initwealth"].ravel()
    dp = package / "ParameterAndMoments" / "DeathProbs"
    # csvread(f, 2, 1): skip the title and header rows and the year column.
    m = np.loadtxt(dp / "DeathProbsE_M_Hist_TR2023.csv", delimiter=",", skiprows=2)[:, 1:]
    f = np.loadtxt(dp / "DeathProbsE_F_Hist_TR2023.csv", delimiter=",", skiprows=2)[:, 1:]
    death = ((m + f) / 2)[100:105, 20:91].mean(axis=0)     # years 2000-04, ages 20-90
    return dict(
        a0kids=dem[0], a1kids=dem[1], a2kids=dem[2],
        a0depadul=dem[3], a1depadul=dem[4], a2depadul=dem[5],
        kidscoeff=est_income[0], spousecoeff=est_income[1], depadulcoeff=est_income[2],
        agecoeff=est_income[3], age2coeff=est_income[4], age3coeff=est_income[5],
        cons=est_income[6], auto=est_ar1[0], vareps=est_ar1[1], varnu=est_ar1[2],
        c0credit=cred[0], c1credit=cred[1], c2credit=cred[2],
        med_total_wealth=iw[0], med_liq_wealth=iw[1], death=death,
    )


# ------------------------------------------------------------- configuration

@dataclasses.dataclass(frozen=True)
class Config:
    """Everything the solution depends on besides (beta, delta, rho)."""

    xjump: float = 1000.0
    xmax: float = 4e5
    x_cells: int = 50               # cells per step size on the positive X grid
    xjump_max: float = 6000.0
    zjump: float = 2000.0
    zmax: float = 3.5e6
    z_cells: int = 10
    zjump_max: float = 96000.0
    R: float = 1.0203
    R_gamma: float = 1.05
    R_CC: float = 1.1059
    alpha: float = 0.5
    betahat: float = 1.0
    card: bool = True               # False: no credit line (A2's no-card type)
    centred: bool = False           # MATLAB: False
    tie: str = "z_first"            # MATLAB: "z_first"; the port: "x_first"
    #: Round income levels in the backward induction too. MATLAB (and the
    #: port) use exp(log(k * xjump)) there, which misses the lattice by an ulp
    #: about half the time, while the forward pass rounds (``simY__ =
    #: round(simY__)``); at exact grid midpoints the two then snap differently.
    round_income: bool = False      # MATLAB: False
    #: Income levels to use for the lattice ``k * xjump`` (k = 1, 2, ...)
    #: instead of numpy's ``exp(log(.))``: e.g. Octave's own, which land on the
    #: other side of the lattice for a few k (RESULTS 54).
    income_levels: tuple | None = None
    n_states: int = 3
    span: float = 1.5
    purge: float = 1e5
    age_start: int = 20
    age_end: int = 90

    def __post_init__(self):
        if self.tie not in ("z_first", "x_first"):
            raise ValueError(f"tie must be z_first or x_first, not {self.tie!r}")


#: Their grids, as named in the port (``twoasset.GRIDS``), for like-for-like runs.
COARSE = dict(xjump=4000.0, x_cells=16, zjump=16000.0, z_cells=5)
MID = dict(xjump=2000.0, x_cells=25, zjump=8000.0, z_cells=7)
FULL: dict = {}


class Model:
    """Age profiles, grids and shock discretisations for one ``Config``."""

    def __init__(self, cfg: Config | None = None, fs: dict | None = None):
        cfg = Config() if cfg is None else cfg
        self.cfg = cfg
        fs = first_stage() if fs is None else fs
        self.fs = fs
        age = np.arange(cfg.age_start, cfg.age_end + 1, dtype=float)
        self.age, self.T = age, len(age)
        spouse = np.full_like(age, 2.0)
        kids = fs["a0kids"] * np.exp(fs["a1kids"] * age - fs["a2kids"] * age**2)
        depadul = fs["a0depadul"] * np.exp(fs["a1depadul"] * age - fs["a2depadul"] * age**2)
        self.hhs = 1.0 * spouse + 1.0 * depadul + 0.4 * kids
        self.ymean = (fs["cons"] + fs["agecoeff"] * age + fs["age2coeff"] * age**2 / 100
                      + fs["age3coeff"] * age**3 / 10000 + fs["spousecoeff"] * spouse
                      + fs["kidscoeff"] * kids + fs["depadulcoeff"] * depadul)
        self.sig_eps = np.sqrt(fs["vareps"])
        self.sig_nu = np.sqrt(fs["varnu"])
        self.Ymean = np.exp(self.ymean + 0.5 * (self.sig_eps**2 / (1 - fs["auto"]**2)
                                                + self.sig_nu**2))
        self.pen = 0.5 / (1 + np.exp((age - 50) / 10))
        self.death = fs["death"]
        credit = (fs["c0credit"] + fs["c1credit"] * age + fs["c2credit"] * age**2
                  if cfg.card else np.zeros_like(age))
        self.xmin = cfg.xjump * np.round(credit * self.Ymean / cfg.xjump)
        xpos = self._doubling_grid(cfg.xjump, cfg.xjump_max, cfg.xmax, cfg.x_cells)
        self.X = [np.concatenate([np.arange(-xm, -cfg.xjump / 2, cfg.xjump), xpos])
                  for xm in self.xmin]
        self.Z = self._doubling_grid(cfg.zjump, cfg.zjump_max, cfg.zmax, cfg.z_cells)
        self.states, self.P = self._tauchen(fs["auto"], self.sig_eps)
        self.mean_hhs, self.mean_y = self.hhs.mean(), self.Ymean.mean()

    @staticmethod
    def _doubling_grid(jump, jump_max, top, cells):
        """0, then ``cells`` steps of ``jump``, the step doubling each block,
        until the step would reach ``jump_max``; then ``jump_max`` steps to
        ``top``. Points above ``top`` are dropped."""
        pts, step = [0.0], jump
        while pts[-1] < top:
            if step < jump_max:
                pts += [pts[-1] + step * k for k in range(1, cells + 1)]
                step *= 2
            else:
                # MATLAB's a:s:b, every point up to and including b. Dollar
                # amounts are whole numbers, so the sums are exact.
                while pts[-1] + jump_max <= top:
                    pts.append(pts[-1] + jump_max)
                break
        g = np.array(pts)
        return g[g <= top]

    def _tauchen(self, psi, sig):
        """Tauchen (1986): ``n`` states spanning +-``span`` unconditional sds,
        ``P[j, k] = P(next = k | now = j)``; the end states take the tails."""
        n = self.cfg.n_states
        sd = sig / np.sqrt(1 - psi**2)
        s = np.linspace(-self.cfg.span * sd, self.cfg.span * sd, n)
        h = (s[1] - s[0]) / 2
        P = np.empty((n, n))
        for j in range(n):
            for k in range(n):
                hi = norm.cdf((s[k] - psi * s[j] + h) / sig)
                lo = norm.cdf((s[k] - psi * s[j] - h) / sig)
                P[j, k] = hi if k == 0 else (1 - lo if k == n - 1 else hi - lo)
        return s, P

    def stationary(self) -> np.ndarray:
        """By iterating the chain, not by an eigenvector."""
        p = np.full(len(self.states), 1 / len(self.states))
        for _ in range(100_000):
            q = p @ self.P
            if np.abs(q - p).max() < 1e-16:
                break
            p = q
        return q

    def transitory(self, t: int, s: int) -> tuple[np.ndarray, np.ndarray]:
        """Income at age ``t`` in persistent state ``s``: the lognormal on the
        lattice ``xjump, 2 xjump, ..., xmax + xmin_t``, density weighted by a
        midpoint Riemann width in logs, points under total/purge dropped."""
        c = self.cfg
        lat = np.arange(c.xjump, c.xmax + self.xmin[t] + c.xjump / 2, c.xjump)
        ly = np.log(lat)
        pdf = norm.pdf(ly, self.ymean[t] + self.states[s], self.sig_nu)
        keep = pdf > pdf.sum() / c.purge
        ly, pdf = ly[keep], pdf[keep]
        gap = np.diff(ly)
        width = (np.r_[gap[0], gap] + np.r_[gap, gap[-1]]) / 2
        w = pdf * width
        y = np.exp(ly)
        if c.income_levels is not None:
            k = np.rint(lat[keep] / c.xjump).astype(int)
            y = np.asarray(c.income_levels, dtype=float)[k - 1]
        return w / w.sum(), (np.round(y) if c.round_income else y)

    def utility(self, c: np.ndarray, h: float, rho: float) -> np.ndarray:
        """``h * u(c / h)``; CRRA centred at $1 (MATLAB) or at mean income per
        adult (``centred``). ``c <= 0`` gives u(0): -inf for rho >= 1."""
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            lc = np.log(np.maximum(c, 0.0) / h)
            if self.cfg.centred:
                lk = np.log(self.mean_y / self.mean_hhs)
                if rho == 1:
                    return h * (lc - lk)
                return h * (np.exp((1 - rho) * lc) - np.exp((1 - rho) * lk)) / (1 - rho)
            if rho == 1:
                return h * lc
            return h * (np.exp((1 - rho) * lc) - 1) / (1 - rho)

    def bequest(self, t: int, Xg: np.ndarray, delta: float, rho: float) -> np.ndarray:
        estate = Xg[:, None] + self.Z[None, :] * (1 - self.pen[t])
        annuity = max(self.cfg.R - 1, 0) * np.maximum(estate, 0)
        u = lambda y: self.utility(np.asarray(y), self.mean_hhs * 1.0, rho)  # noqa: E731
        # U(y) = hbar * u(y / hbar): utility() already scales by h.
        return self.cfg.alpha / (1 - delta) * (u(self.mean_y + annuity) - u(self.mean_y))

    @staticmethod
    def snap(grid: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Index of the nearest grid point; exact midpoints go up; outside the
        grid, the end point."""
        i = np.clip(np.searchsorted(grid, v, side="left"), 1, len(grid) - 1)
        lo, hi = grid[i - 1], grid[i]
        return np.where(v - lo < hi - v, i - 1, i)


@dataclasses.dataclass
class Solution:
    """Policies per age, on that age's own grid: ``ix[t][i, iz, s]`` indexes
    ``X[t+1]`` (``X[t]`` at the last age), ``iz`` indexes ``Z``; ``c`` is
    consumption excluding the dividend, as MATLAB's ``C____``."""

    ix: list
    iz: list
    c: list
    V: list
    EV: list
    W: list
    model: Model
    prefs: tuple


def _argmax(pay: np.ndarray, tie: str) -> tuple[int, int]:
    """Exact-maximum choice with an explicit tie rule over ``(x', z')``.
    MATLAB's ``max`` skips NaN and ``np.argmax`` returns it, so NaN is refused."""
    if np.isnan(pay).any():
        raise FloatingPointError("NaN payoff")
    if tie == "x_first":
        k = int(np.argmax(pay))              # row-major: lowest x', then lowest z'
        return divmod(k, pay.shape[1])
    k = int(np.argmax(pay.T))                # column-major: lowest z', then lowest x'
    jz, jx = divmod(k, pay.shape[0])
    return jx, jz


def solve(beta: float, delta: float, rho: float, model: Model) -> Solution:
    """Backward induction over ages, one state at a time."""
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    m, cfg = model, model.cfg
    T, Z, nZ, nS = m.T, m.Z, len(m.Z), cfg.n_states
    ix_, iz_, c_, V_, EV_, W_ = ([None] * T for _ in range(6))
    naive = cfg.betahat != beta
    EV = None
    for t in range(T - 1, -1, -1):
        Xt = m.X[t]
        beq = m.bequest(t, Xt, delta, rho)
        if t == T - 1:
            Xn = Xt
            EV = np.repeat(beq[:, :, None], nS, axis=2)
            EV[Xn < 0] = -np.inf
        else:
            Xn = m.X[t + 1]
        EV_[t] = EV
        cost = np.where(Xn >= 0, Xn / cfg.R, Xn / cfg.R_CC)
        nX = len(Xt)
        ix = np.empty((nX, nZ, nS), dtype=np.int64)
        iz = np.empty((nX, nZ, nS), dtype=np.int64)
        cc = np.empty((nX, nZ, nS))
        V = np.empty((nX, nZ, nS))
        W = np.empty((nX, nZ, nS))
        for a in range(nZ):
            dz = Z - Z[a]
            zpart = -dz + m.pen[t] * np.minimum(dz, 0.0)
            for i in range(nX):
                C = (Xt[i] - cost)[:, None] + zpart[None, :]
                u = m.utility(C + (cfg.R_gamma - 1) * Z[a], m.hhs[t], rho) - 1e100 * (C < 0)
                for s in range(nS):
                    with np.errstate(invalid="ignore"):
                        pay = u + beta * delta * EV[:, :, s]
                    jx, jz = _argmax(pay, cfg.tie)
                    ix[i, a, s], iz[i, a, s], cc[i, a, s] = jx, jz, C[jx, jz]
                    W[i, a, s] = pay[jx, jz]
                    if naive:
                        with np.errstate(invalid="ignore"):
                            pay_h = u + cfg.betahat * delta * EV[:, :, s]
                        hx, hz = _argmax(pay_h, cfg.tie)
                    else:
                        hx, hz = jx, jz
                    V[i, a, s] = u[hx, hz] + delta * EV[hx, hz, s]
        ix_[t], iz_[t], c_[t], V_[t], W_[t] = ix, iz, cc, V, W
        if t > 0:
            nxt = np.zeros((nX, nZ, nS))
            for s2 in range(nS):
                p, y = m.transitory(t, s2)
                Vs = np.zeros((nX, nZ))
                for pk, yk in zip(p, y, strict=True):
                    Vs += pk * V[m.snap(Xt, Xt + yk), :, s2]
                for s1 in range(nS):
                    nxt[:, :, s1] += m.P[s1, s2] * Vs
            EV = (1 - m.death[t - 1]) * nxt + m.death[t - 1] * beq[:, :, None]
    return Solution(ix_, iz_, c_, V_, EV_, W_, model, (beta, delta, rho))


def uniforms(n: int, T: int, seed: int = 0) -> dict[str, np.ndarray]:
    """The two uniform arrays the forward pass consumes: persistent-state
    draws ``u_state`` and transitory draws ``u_income``, each ``(n, T)``."""
    g = np.random.default_rng(seed)
    return {"u_state": g.random((n, T)), "u_income": g.random((n, T))}


def initial_indices(model: Model, n: int, wealth: np.ndarray | None = None):
    """Age-20 grid indices. ``wealth=None``: their single SCF seed (liquid and
    illiquid medians times age-20 mean income, liquid at least 0). Otherwise
    ``(n, 2)`` (liquid, illiquid) dollar amounts."""
    X0, Z = model.X[0], model.Z
    if wealth is None:
        y0 = model.Ymean[0]
        fs = model.fs
        ix = model.snap(X0, np.array([fs["med_liq_wealth"] * y0]))[0]
        ix = max(ix, int(np.flatnonzero(X0 == 0)[0]))
        iz = model.snap(Z, np.array([(fs["med_total_wealth"] - fs["med_liq_wealth"]) * y0]))[0]
        return np.full(n, ix), np.full(n, iz)
    w = np.asarray(wealth, dtype=float)
    return (model.snap(X0, np.clip(w[:, 0], X0[0], X0[-1])),
            model.snap(Z, np.clip(w[:, 1], 0, Z[-1])))


def simulate(sol: Solution, u: dict, init=None, init_state: np.ndarray | None = None):
    """Forward pass from given uniforms, so any solver's policies can be run on
    the same shocks. Draws map to the first state / income point whose CDF
    reaches the uniform. Returns annual ``(n, T)`` arrays: ``income``,
    ``cash`` (after income), ``liquid`` (cash - income), ``illiquid``,
    ``consumption`` (with the dividend), ``state``, ``ix`` and ``iz``."""
    m = sol.model
    n, T = u["u_state"].shape
    S = np.empty((n, T), dtype=np.int64)
    if init_state is None:
        S[:, 0] = np.searchsorted(np.cumsum(m.stationary()), u["u_state"][:, 0])
    else:
        S[:, 0] = init_state
    cdf = np.cumsum(m.P, axis=1)
    for t in range(1, T):
        S[:, t] = (u["u_state"][:, t, None] > cdf[S[:, t - 1]]).sum(axis=1)
    np.clip(S, 0, len(m.states) - 1, out=S)
    Y = np.empty((n, T))
    for t in range(T):
        for s in range(len(m.states)):
            r = S[:, t] == s
            p, y = m.transitory(t, s)
            k = np.searchsorted(np.cumsum(p), u["u_income"][r, t])
            Y[r, t] = y[np.clip(k, 0, len(y) - 1)]
    Y = np.round(Y)
    ix, iz = initial_indices(m, n) if init is None else init
    ix, iz = np.asarray(ix).copy(), np.asarray(iz).copy()
    out = {k: np.empty((n, T)) for k in ("cash", "illiquid", "consumption")}
    IX, IZ = np.empty((n, T), dtype=np.int64), np.empty((n, T), dtype=np.int64)
    for t in range(T):
        # Last year's choice indexes this year's grid, so snap on X[t].
        ix = m.snap(m.X[t], m.X[t][ix] + Y[:, t])
        IX[:, t], IZ[:, t] = ix, iz
        out["cash"][:, t] = m.X[t][ix]
        out["illiquid"][:, t] = m.Z[iz]
        out["consumption"][:, t] = sol.c[t][ix, iz, S[:, t]]
        if t < T - 1:
            ix, iz = sol.ix[t][ix, iz, S[:, t]], sol.iz[t][ix, iz, S[:, t]]
    out["consumption"] += (m.cfg.R_gamma - 1) * out["illiquid"]
    out["liquid"] = out["cash"] - Y
    return out | {"income": Y, "state": S, "ix": IX, "iz": IZ}
