# Household-Level NPE for Lifecycle Preference Parameters

Per-household Neural Posterior Estimation (NPE) over (β, δ, ρ) for the
Laibson, Lee, Maxted, Repetto, Tobacman (NBER WP 13314 / RFS 2026) lifecycle
consumption-saving model.

**The headline contribution is a posterior distribution per household**, not a
population point estimate. Laibson et al. estimate one preference vector for
the whole population by Method of Simulated Moments on 16 aggregate moments;
we train an amortized neural posterior that maps a *single household's*
observed trajectory to a full posterior over its preferences.

---

## Status at a glance

| Phase | What | State |
|---|---|---|
| 1 | MVP simulator (HARK, single asset, β=1) | **Done** — 1 gate untested, see below |
| 2 | NPE proof of concept, learn (δ, ρ) | **Done** — SBC passed |
| 3 | Two-asset: credit cards, illiquid asset, present bias | **In progress** — port built, validation running |
| 4 | PSID 2005–2013 empirics | Not started |

Current branch: `phase3-illiquid-cc`. 107 tests passing.

---

## The two simulators

The repository contains **two** lifecycle simulators. This is deliberate.

### `hark` — Phases 1–2

Single liquid asset, CRRA utility, exponential discounting (β locked at 1),
AR(1) income via Rouwenhorst, SSA 2020 mortality, no borrowing
(`BoroCnstArt = 0`), ages 20–90 annual. Built on HARK 0.17
`MarkovConsumerType`. Estimates **(δ, ρ)**.

Retained only so the Phase 2 results stay reproducible.

### `twoasset` — Phase 3 (current)

Port of Laibson et al.'s `LifecycleSim*.m` from `replication-package-LLMRT/`,
baseline configuration. Estimates **(β, δ, ρ)**.

- **Liquid assets `X`** may go negative to an age-varying credit limit,
  borrowed at `R_CC = 1.1059`; positive balances earn `R = 1.0203`.
- **Illiquid assets `Z ≥ 0`** earn `R_gamma = 1.0500`, pay a proportional
  age-declining liquidation penalty on withdrawal, and return a consumable
  flow `(R_gamma − 1)·Z`.
- **Naive quasi-hyperbolic discounting**: the current self acts on `β·δ` but
  believes every future self will discount exponentially (`betahat = 1`).
  Backward induction therefore takes *two* argmaxes per state.
- Brute-force discrete DP over the joint `(X', Z')` choice — no first-order
  conditions, no policy interpolation, matching their method.

**Why HARK could not carry Phase 3:** HARK 0.17.2 has no two-asset model with
a proportional liquidation penalty (`ConsRiskyContribModel` is a different
friction), `KinkedRconsumerType` cannot be combined with `MarkovConsumerType`,
and there is no β-δ support anywhere. Building Phase 3 in HARK would have meant
writing this solver anyway.

---

## Pipeline

```
                     SIMULATOR_SPEC.md  (pre-registration, frozen before Phase 4)
                              │
   prior box ────────────────►│
   (β,δ,ρ) Sobol              ▼
                        solve + simulate            ← simulator/twoasset.py
                              │                       simulator/dispatch.py
                              ▼
                    biennial wave aggregation       ← data/waves.py
                    (N, 5 waves, 4 features)
                              │
                              ▼
                    Transformer embedder            ← npe/embedder.py
                              │
                              ▼
                    SNPE-C neural spline flow       ← npe/train.py
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                  SBC                posterior      ← evaluation/sbc.py
            (rank uniformity,       per household
             coverage)
```

### Running it end to end

```bash
uv sync --extra dev
uv run pytest                                    # 107 tests

# Self-checks (no arguments, each asserts and prints a summary)
uv run python -m hh_npe.simulator.laibson_calibration   # re-extract + verify 126 frozen values
uv run python -m hh_npe.simulator.grids                 # grid construction invariants

# Validate the port against Laibson et al.'s published moments
uv run python scripts/validate_twoasset.py --pop 10000 --full

# Generate training data  (--simulator hark reproduces Phase 2)
uv run python scripts/generate_dataset.py --n_samples 8192 --n_jobs -1

# Train and check calibration
uv run python scripts/train_npe.py
uv run python scripts/run_sbc.py
```

Hydra composes `configs/config.yaml` from `simulator/mvp`, `npe/phase3`,
`eval/sbc`. Override on the command line:

```bash
uv run python scripts/train_npe.py npe/…=phase2 npe.training.max_num_epochs=50 seed=1
```

---

## Observation model

NPE inputs are **7 biennial waves** — biennial to match PSID's post-1997
observation schedule. Flows (`income`, `consumption`) are summed over each
2-year window; stocks are read at the end of it. Annual waves (`wave_years=1`)
are implemented and tested but are **not** the pre-registered choice.

**This was 10 waves at ages 30–49 in the Phase 3 pre-registration.** It is 7
because that is what PSID can actually supply with a clean credit-card measure
— card debt is separable only from 2011, giving 2011–2023 — and the
SIMULATOR_SPEC §6.1 question of "whether PSID can supply a balanced 10-wave
panel of usable size" resolved to *no*. See `PSID_DATA.md`.

Window start ages are **24–45**, which is not the same as the wave-0 ages they
produce: `aggregate_waves` reports each wave at the *last* year of its window,
so a window opened at 24 first reports at age 25. Passing 25–46 — as every
pre-Phase-4 run did — yields simulated wave-0 ages of 26–47 against PSID's
25–46. See `RESULTS.md` §10.8.

Shards store the annual panel, so the window is a post-hoc aggregation choice:
changing it re-derives from disk in seconds and never re-solves.

Feature order is positional and load-bearing:

| Set | Features | Used by |
|---|---|---|
| `FEATURES_MVP` | income, consumption, liquid_assets | Phases 1–2 |
| `FEATURES_TWOASSET` | income, consumption, liquid_assets, **illiquid_assets** | Phase 3 |

Phase 3 observes the illiquid balance because eight of Laibson et al.'s sixteen
target moments are wealth *conditional on debt status*, and the identifying
signature of present bias is expensive borrowing held simultaneously with
illiquid wealth.

---

## Results so far

### Phase 2 (complete)

Dataset: 8,192 Sobol draws of (δ, ρ), 8,092 surviving the rebirth filter
(98.8%). Training: 113 epochs, best validation log-prob −2.175.

**SBC passed** (Talts et al. 2018), 90% credible interval coverage against a
0.9 target:

| Parameter | KS *p* | χ² *p* | Coverage |
|---|---|---|---|
| δ | 0.446 | 0.220 | 0.888 |
| ρ | 0.488 | 0.075 | 0.876 |

**Posterior contraction** `c = 1 − Var[posterior]/Var[prior]`, full 8,092
households, 1,000 draws each:

| Parameter | mean *c* | constrained | unconstrained |
|---|---|---|---|
| δ | 0.586 | — | — |
| ρ | 0.397 | **0.732** | **0.242** |

ρ is well identified only for the ~31% of households constrained at every wave.
For the unconstrained majority the posterior is barely narrower than the prior.

**Capacity sweep** — is ρ information-limited or capacity-limited? Three
architectures, same data, same seed, paired standard errors:

| Config | Params | ρ *c̄* | Δ vs baseline |
|---|---|---|---|
| baseline | 186 k | 0.4004 | — |
| wider | 1,459 k | 0.4310 | +0.0305 (SE 0.0022) |
| flow-only-bigger | 929 k | 0.4175 | +0.0171 (SE 0.0015) |

7.8× the parameters buys a 2.6% narrower ρ posterior (sd 1.006 → 0.980 on a
4.5-wide prior). Statistically real, practically negligible, and the
constrained/unconstrained split barely moves. **Conclusion: ρ is
information-limited, not capacity-limited** — the lever is the observation, not
the network. That is the direct motivation for Phase 3.

Caveat: one training seed per config; the paired SE captures household sampling
noise only, not training-seed variation.

### Phase 3 (in progress)

Built and tested:
- 126 calibration values frozen from their `comphs` benchmark, verified
  bit-exact against the replication package
- Grids reproduce theirs exactly: **nX = 190, nZ = 84**, credit limits
  \$5k–\$31k, mean income \$27.7k–\$59.3k
- Their dynamic budget identity (`assert(abs(check) < 1e-5)`) re-derived
  independently and checked over the **entire** grid in float64

**Validation against their table-3 moments PASSES.** At their exact 190×84 grid
and their published benchmark estimates, our simulated moments match theirs to
**mean |log(ours/theirs)| = 0.0102** — all 16 moments within 4%:

| moment | ours ÷ theirs, by age band |
|---|---|
| `%Visa` | 1.00, 1.01, 1.00, 1.00 |
| `meanVisa` | 1.00, 1.03, 1.01, 1.01 |
| `wealth\|debt` | 1.01, 0.99, 0.96, 1.01 |
| `wealth\|no debt` | 1.01, 1.01, 0.99, 0.98 |

The MSM objective is q = 85.3 against their 77.2; that residual is Monte Carlo
noise from independent RNG streams at `pop = 10000`, amplified because q sums
16 squared t-statistics. The objective code itself is verified two ways: fed
their stored `optMoments` it reproduces `optq = 77.15` exactly, and our moment
function applied to their own simulated panel reproduces all 16 values exactly.

Getting there took three bugs, all of which pushed the same direction
(toward borrowing), which is why the aggregate misfit looked like one large
effect:

| # | Bug | Effect on fidelity |
|---|---|---|
| 1 | `P.T` used in the expectation step — not row-stochastic | — |
| 2 | float32 could not resolve the argmax (§4) | 0.248 → 0.102 |
| 3 | Grid-snapping ties rounded **down**, leaking liquid wealth | 0.102 → **0.0102** |

Each has a regression test. Bug 3 is the subtle one: income lands on the
`xjump` lattice but the liquid grid coarsens above \$50k, so 26.9% of
cash-on-hand queries fall exactly midway between grid points. MATLAB's
`griddedInterpolant(...,'nearest')` rounds those up; we rounded down, silently
discarding up to \$6,000 per snap for wealthier households, compounding with
age, in both the forward pass and the DP.

They were located by bisecting against their stored intermediates rather than
guessing from aggregate misfit — their `.mat` contains `income_setup`,
`grid_setup`, the full policy functions and the simulated panel:

| Component | Test | Result |
|---|---|---|
| Calibration + grids | element-wise vs their setups | bit-identical, 11 arrays |
| Age-step | fed **their** `EV` | 99.6–99.7% exact policy match |
| Expectation step | fed **their** `V` | max rel 1.4e-07 |
| Moment code | applied to **their** panel | exact |
| Income process | marginals, mean, sd | ratios 0.9961 |

**Grid-coarsening error**, measured against the full grid (most of the earlier
figure was bug 3, whose severity scaled with refinement):

| Grid | solve | max \|diff\|/se | median relative | ties |
|---|---|---|---|---|
| coarse 81×46 | 72 s | 3.15 | 5.4% | 0.0% |
| mid 107×56 | 234 s | **0.94** | 2.2% | 12.4% |
| full 190×84 | 1678 s | — | — | 26.9% |

The mid grid sits under one standard error of discretization error and is the
working choice for dataset generation: ~5.5 days for 8,192 draws on 4 cores,
~17 h on 32. The full grid is ~159 days on 4 cores and remains the reference
configuration for validation runs.

Measured solve times (4 CPU cores, no GPU):

| Grid | Size | One solve | 8,192 draws / 4 cores |
|---|---|---|---|
| coarse | 81 × 46 | 36 s | ~20 h |
| mid | 107 × 56 | 87 s | ~50 h |
| full (theirs) | 190 × 84 | 1052 s | ~25 days |

Grid coarsening is an approximation whose cost is measured, not assumed — see
`scripts/validate_twoasset.py`. Hardware decision (more cores vs a GPU) is
pending that measurement.

---

## Known gaps

- **The Phase 1 gate has no test.** SIMULATOR_SPEC §7 requires reproducing
  HARK/Carroll buffer-stock lifecycle profiles (income/consumption/wealth means
  by age); nothing in `tests/` checks this.
- **Phase 3 validation is unfinished.** The port is not yet demonstrated to
  reproduce their published moments.
- **No MATLAB or Octave in this environment**, so their `.m` files cannot be
  run as a live oracle. Validation is against their published moment estimates
  plus their own in-code assertions, both ported.

---

## Layout

```
configs/                Hydra configs
  config.yaml           root; composes simulator + npe + eval
  npe/phase2.yaml       Phase 2 (hark, δ ρ)
  npe/phase3.yaml       Phase 3 (twoasset, β δ ρ)
src/hh_npe/
  simulator/
    laibson_calibration.py  frozen first-stage constants + verifier
    grids.py                age profiles, asset grids, Tauchen, shock discretization
    twoasset.py             Phase 3 backward induction + forward simulation
    dispatch.py             theta → wave tensor, shared by dataset gen and SBC
    lifecycle.py            Phase 1–2 HARK agent
    forward.py              Phase 1–2 HARK simulation
    income.py, mortality.py Phase 1–2 Rouwenhorst + SSA survival
  data/
    waves.py            annual → biennial wave aggregation, rebirth filter
    dataset.py          (theta, x) save/load
  npe/
    prior.py            PriorBox (2- or 3-parameter), Sobol sampler
    embedder.py         Transformer summary network
    train.py            SNPE-C training, posterior save/load
  evaluation/sbc.py     rank statistics, coverage, plots
  utils/seeding.py
scripts/
  generate_dataset.py   Sobol → simulate → waves → .pt
  train_npe.py          Hydra entry point
  run_sbc.py            SBC on a trained posterior
  validate_twoasset.py  port vs Laibson et al. table 3
tests/                  pytest (107)
SIMULATOR_SPEC.md       pre-registration spec; frozen before Phase 4
replication-package-LLMRT/   Laibson et al. MATLAB + inputs (gitignored, 653 MB)
```

---

## Reproducibility

- Seeds set centrally via `src/hh_npe/utils/seeding.py`.
- Hydra dumps resolved config to `outputs/<run>/.hydra/config.yaml`.
  `output_dir` is `${hydra:runtime.output_dir}` — **not** `${now:...}`, which
  re-resolves on every access and once caused saved artifacts to drift into a
  different directory from Hydra's own run dir.
- Calibration constants are frozen literals with a verifier, so the 653 MB
  replication package is not a runtime dependency.
- SBC and dataset generation share one simulator definition
  (`simulator/dispatch.py`); SBC is only meaningful if the two are identical.
- W&B runs default to `mode="offline"`; run `wandb login` to enable sync.

---

## References

- Laibson, Lee, Maxted, Repetto, Tobacman — NBER WP 13314, RFS 2026. *The model
  being replicated and extended.*
- Cranmer, Brehmer, Louppe — PNAS 2020 (simulation-based inference review).
- Greenberg, Nonnenmacher, Macke — 2019 (SNPE-C / APT).
- Talts, Betancourt, Simpson, Vehtari, Gelman — 2018 (simulation-based calibration).
- Tauchen — 1986 (AR(1) discretization; used in Phase 3 to match theirs).
- Ward, Cannon, Beaumont, Fasiolo, Schmon — NeurIPS 2022 (robust NPE under
  misspecification; Phase 4).
- Wei & Jiang — Marketing Science 2025 (population-level NNE; the contrast).

## Pre-registration

See [SIMULATOR_SPEC.md](SIMULATOR_SPEC.md). It must be **frozen via a tagged
git commit before any PSID work begins**. Deviations from the spec require a
new tagged version and a documented reason; each current deviation from
Laibson et al.'s code is recorded there with its justification.
