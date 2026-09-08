# Phase 4 — PSID empirical results

**Status:** in progress. Estimates below are conditional on a simulator whose
training set has a known limitation (§7.1), so they are reported as the current
state, not as final.
**Last updated:** 2026-09-07.

Per-household posteriors over (β, δ, ρ) for 889 PSID households observed in
seven biennial waves, 2011–2023, against Laibson, Lee, Maxted, Repetto &
Tobacman's single population MSM estimate.

---

## 1. Headline

```
                          beta       delta        crra
median of means         0.8465      0.9898      4.5002
mean of means           0.8158      0.9802      4.2455
sd across households    0.1029      0.0231      0.8077
median posterior sd     0.1288      0.0130      0.1109

Laibson et al. MSM      0.5305      0.9891      1.9355
  their std error       0.1140      0.0051      0.4350

share of households whose 90% CI covers their estimate:
  beta 0.397    delta 0.735    crra 0.199
```

**δ replicates almost exactly** — 0.9898 against their 0.9891, with 73.5% of
households covering their point. **β and ρ are both substantially higher.**

### Is the heterogeneity real?

```
            between-hh sd   within-hh sd   ratio
beta               0.1029         0.1294    0.79
delta              0.0231         0.0198    1.17
crra               0.8077         0.1154    7.00
```

A ratio below 1 means households differ by *less* than any one of them is
uncertain — the apparent spread is estimation noise, and a single population
estimate would serve as well.

- **ρ heterogeneity is real** (7.0×).
- **δ is marginal** (1.2×).
- **β heterogeneity is not demonstrated** (0.79). This is a caution about the
  project's own premise and belongs in any writeup.

Within households the parameters are close to orthogonal: median
`corr(β,ρ) = +0.017`, `corr(β,δ) = −0.166`, `corr(δ,ρ) = −0.076`. There is no
strong β–ρ ridge, so the estimates are not trading off against each other.

Figures in `figures/`.

---

## 2. Sample

```
rows in the extract                 15044
interviewed in all 7 waves           8472
+ head/reference throughout          4470
+ aged 25-46 at 2011                 2149
+ age progresses 11-13 years         2119
+ non-missing on all features        2021
--- then Laibson et al.'s own filter ---
+ comphs education (12-15 yrs)       1117
+ not self-employed                  1012
+ no business or farm income          889
```

Headship, not attrition, is where the sample goes: requiring the *same person*
as head throughout halves it. Their education restriction removes another 47%.

**Matched to `3_scfanalysis_withCondMoments_bs.do:301–323`**, which is not
optional: their entire first stage — income profile, credit limits, initial
wealth — is estimated on `comphs` households only.

---

## 3. Feature construction

`(N, 7, 5)` = income, consumption, liquid, illiquid, age, in **2010 USD**
(their `cpibaseyear`). Flows deflated at the income reference year, stocks at
the interview year.

| feature | construction |
|---|---|
| income | after-tax **non-asset** income. `nasry = total family income − asset income`, then TAXSIM federal + state + employee FICA, **pro-rated by the non-asset share** exactly as `2_builddata.do:96` |
| consumption | `TOTAL EXPENDITURE − mortgage − property tax − home insurance − vehicle principal` **+ imputed rent for owners** (§5) |
| liquid | checking/saving + CD/bonds − credit-card debt |
| illiquid | stocks + IRA + vehicles + other + other real estate + farm/business + home equity, **net of all debts**, floored at 0 |
| age | head's age |

**Independent validation:** TAXSIM after-tax non-asset income for tax year 2010
has median 40,593 against a simulated 40,000 at age 30, and 51,000 vs 50,838 at
ages 35–44. Nothing was tuned to make that hold.

---

## 4. The consumption investigation

PSID consumption initially sat ~34% below simulated, with a gradient: c/y ran
1.68 in the bottom income decile to 0.40 in the top, against a flat ~0.99 in
simulation, **in every age band**. Five candidates were tested.

| candidate | verdict | slope gap closed |
|---|---|---|
| Differential measurement error | Engel: PSID's food elasticity is 0.575, inside the literature's 0.50–0.60, bounding differential under-reporting at 1.06–1.20× against the 2.2× observed | ~0 |
| Sample mismatch | applied their filter, 2119 → 889 | 3% |
| Household composition | typical-household adjustment (§6) | 2% |
| Calibration vintage | **refuted** — re-estimated their income process on 2011–2023 their way (`xtreg … , fe`): growth over ages 25–55 is 1.53× in 1982–91 and 2.03× now (steeper, not flatter), and `nhead` (0.319 → 0.356) and `kids` (0.013 → 0.021) replicate | ~0 |
| **Owner-occupied housing services** | **confirmed and fixed** (§5) | **42%** |

```
ages 35-44, all 7 waves    slope   med cons     c/y
matched, net               0.368      29660   0.684
+ rental value             0.554      35621   0.822
simulated                  0.807      51276   1.046
```

**A residual gap remains** (0.554 vs 0.807). It is no longer attributable to any
data-handling defect we have been able to find, and is most plausibly the
model's structural limits (§7.1).

---

## 5. Rental equivalence — the largest single correction

PSID reports owners' housing as mortgage + property tax + insurance. We strip
those as saving and transfers — correctly, and identically to PSID's own
`TOTAL CONSUMPTION WITH RENTAL VALUE`, which is
`TOTEXP − (mortgage + property tax + home insurance) + rental value`
(correlation 0.9936 with that formula, median absolute difference 0). That
leaves **owners with no housing consumption at all**, while renters keep their
rent. Owners are 42% of the sample in 2011 rising to 56% by 2023.

PSID began collecting `VALUE OF HOME IF RENTED` in **2019**. For 2011–2017 it is
imputed by **anchoring each household on its own later observed value** and
carrying it back in real terms at 2.49%/yr, self-calibrated from the 2019→2023
waves.

**A cross-sectional model `rent = f(income, wealth, size, age)` was deliberately
rejected.** It would make imputed consumption a function of income and illiquid
wealth — two other features — so the consumption channel would carry no
independent information and the posterior would read a mechanical identity as
behaviour. It would likely *improve* the apparent fit while invalidating it.

Validation and coverage:

```
out-of-sample (2023 -> 2019): median |error| 0.187, median error +0.000
imputed rent = 26.7% of owner consumption -> 5.0% consumption error
early owners with unusable anchor: 5.7% (fall back to the wave median,
  a constant, which adds noise but no spurious correlation)
error propagation, 20 draws: median consumption 35053 +- 56  (0.16%)
```

Effect on model fit:

```
in-box posterior mass: median 0.858 -> 0.986,  p10 0.523 -> 0.645
households the model cannot represent: 0 of 889
```

**This is the number that changed most.** The boundary pinning that made the
earlier estimates artifacts — 92% of δ intervals reaching 1.0 — largely
dissolved, because it was substantially caused by a missing consumption
component rather than by the model.

---

## 6. Matched methodology

Implemented from their code:

- **Typical-household adjustment** (`3_scfanalysis:52–56`). Each moment is
  regressed on `nhead, ndepad, und18, unemprate` plus cohort and age dummies,
  then re-centred on the **model's own** demographic profile
  (`b0·exp(b1·AGE − b2·AGE²)` from `laibson_calibration`). Necessary because the
  model has composition as a deterministic function of age and `grids.py:38`
  sets `spouse = 2.0` for every household — unmarried PSID households are
  otherwise not comparable to anything it generates. Worth 2%.
- **Pro-rated taxes** (`2_builddata.do:96`), removing 3.5% of federal liability
  attributable to asset income.
- **State unemployment**, with 2016+ imputed as the state's 2015 position scaled
  to the national level (the shipped BLS file stops in 2015; BLS blocks scripted
  download).

### Known deviations that cannot be closed

1. **`hasVisa` is not in PSID.** Their sample is conditional on *holding* a
   credit card; PSID only asks about card *debt*. Using debt as a possession
   proxy would select on the outcome that identifies β and bias it downward —
   toward their number, for the wrong reason. "Ever borrowed in 7 waves" gives
   66.6% against a true US holding rate of ~70% and is the best available bound
   (585 households), but it is a bound, not a fix.
2. **α = 2.02 is assumed, not estimated.** `generateAlphaBootstrap.m` is
   literally `2 + 0.5*randn(2000,1)`; the bootstrap only propagates uncertainty
   about the assumed value. Their parameters were fitted against doubled card
   debt, so not applying it is a units mismatch on the margin β rides on;
   applying it assumes PSID's under-reporting matches SCF's.
3. **Cross-section vs panel.** They match four age-band moments from repeated
   cross-sections; we condition on seven-wave trajectories. This is the
   contribution, not a defect, but "same information set" is never literally
   true.

---

## 7. Open issues

### 7.1 The simulator has almost no heterogeneity

The forward pass has **two** stochastic elements: a 3-state persistent income
Markov chain and a transitory income shock. Everything else is deterministic and
**identical across households**:

```python
xind = np.full(N, ix0, dtype=np.int64)   # every household: same initial liquid
zind = np.full(N, iz,  dtype=np.int64)   # every household: same initial illiquid
```

No unemployment spells, health or medical shocks, divorce or marriage, births
beyond the age profile, inheritances, or return heterogeneity. All
cross-sectional dispersion at any age comes from accumulated income shocks
starting from one common initial condition — the SCF median.

Measured income volatility is roughly right in the middle and thin in the tails:
the model implies var 0.2043 for the 2-year change in log income against an
actual 0.2632 (1.29×), but the robust IQR-based sd is 0.275 against the model's
0.452, so the typical household moves *less* than assumed and the excess is
tail events the model has no mechanism for.

**This is the leading explanation for ρ ≈ 4.5.** High ρ is the only channel this
model has for generating precautionary saving, so missing precautionary motives
are routed through it. The ρ between/within ratio of 7.0 says households
genuinely differ — but possibly in exposure to shocks the model cannot see
rather than in risk aversion.

### 7.2 Other

- Consumption slope still 0.554 against 0.807 simulated.
- 9.3% of households have posterior-mean ρ within 5% of the 5.0 ceiling. Down
  sharply from before the rental fix; widening the prior is **no longer the
  obvious next step**, and would move along a ridge the data do not resolve.
- β heterogeneity remains within estimation noise (§1).
- `SIMULATOR_SPEC` §7's Phase 1 gate — reproducing HARK/Carroll buffer-stock
  profiles — still has no test.

---

## 8. Reproducing

```bash
uv run python scripts/taxsim_run.py                       # after-tax income
uv run python scripts/build_psid_tensor.py --match_laibson \
    --out data/processed/psid_x_matched_net.pt            # x tensor, net wealth
uv run python scripts/impute_rental_value.py --n_draws 20 # + imputed rent
uv run python scripts/psid_posterior.py \
    --x data/processed/psid_x_rental.pt --out outputs/psid_rental
uv run python scripts/plot_psid_population.py \
    --posterior_npz outputs/psid_rental/posterior_uncorrected.npz \
    --x data/processed/psid_x_rental.pt --out outputs/psid_rental
```

PSID microdata is not in the repository — see `PSID_DATA.md` for the variable
codes needed to re-pull it. Six Data Center extracts are used: `J364786` (IND
linkage), `J364817` (FAM superset), `J364913` (education), `J364914` (rental
value). `J364812`, `J364814` and `J364816` are strict subsets of `J364817`.

---

## 9. Regeneration gates

Before spending ~9.5 GPU-days regenerating the training set with household
heterogeneity, four candidate sources were priced at a **single fixed θ** =
(0.8465, 0.9898, 4.5002), the posterior median. Holding θ fixed is the point:
an earlier comparison used *pooled* simulated dispersion, which mixes in the
whole uniform prior over θ and cannot answer whether `p(x|θ)` is too narrow.

Each source is one GPU solve (~15 s via `twoasset_gpu.solve_batch`, against
~28 min on one CPU core), so all of this cost minutes rather than days.

```
source                                widening   acts at
education group (4 calib. blocks)       1.79x     every age, GROWS 1.04 -> 3.18
income process (10-param bootstrap)     1.23x     every age
returns (R_gamma +-2pp)                 1.08x     every age
randomised initial wealth               1.00x     forgotten by age 35
```

### 9.1 Initial wealth is a dead end — the plan's §2 fails its own gate

`scripts/gate_within_theta.py`. Widening from randomising the age-20 seed,
by age band:

```
band     illiquid   liquid
25-30      1.38      1.04
31-34      1.05      1.02
35-44      1.00      1.01
45-55      1.00      1.00
```

The model is a buffer-stock model: it converges to its ergodic distribution and
**forgets its initial condition** well before the ages the PSID sample covers.
The gate's stated criterion — *"if `B/A ≈ 1`, regeneration cannot help and must
not run"* — is met at exactly the ages that matter.

**A sign bug was found and fixed before this verdict was accepted.**
`grids.credit_limit` returns the borrowing limit as a *positive magnitude*; the
first version of the clamp read it as a signed floor, so every drawn household
was forced to at least +5,000 liquid and the liquid dimension was never actually
randomised. Re-running after the fix (`logs/gate2_fixed.log`) changes the numbers
in the third decimal and none of the conclusions — PSID's liquid ratios are ~0
for most households (p25 = p50 = 0.000), so a correct draw adds almost nothing
on that axis either. The verdict stands, but it stands on the re-run.

This does not generalise to the other sources, and that distinction is the
substantive finding: **transient heterogeneity decays, persistent heterogeneity
compounds.** Randomised initial wealth is still worth keeping as a free rider on
a regeneration justified by something else (it costs no extra solve and gives
1.38x for the youngest households), but it cannot justify one.

Two wealth problems surfaced alongside, neither of them about dispersion.
**A first pass described the illiquid one as a 16x level gap in medians; that
was wrong** -- 41.8% of PSID household-waves at ages 25-30 sit at exactly zero
illiquid, so the PSID median rests on a mass point created partly by our own
non-negativity floor, and comparing medians across it measures the floor. By
quantile the two distributions **cross**:

```
ages 25-30 illiquid        p10       p25       p50       p75       p90
PSID                         0         0      1939     13106     44600
model                    10000     24000     32000     40000     44000

ages 35-44 illiquid
PSID                         0         0     12882     70352    165611
model                     4000     14000     40000     56000     84000
```

- **Illiquid: the model is far too compressed, not uniformly too high.** At
  ages 25-30 its p90 nearly matches PSID's (44,000 vs 44,600) while its p10 is
  10,000 against PSID's 0 — the model generates **no poor households at all**.
  By 35-44 the error reverses at the top: PSID's p90 is 165,611 against the
  model's 84,000. Both tails are missing, which is the same
  cannot-generate-dispersion story as section 9.2, seen in levels.
- **Liquid: the model puts most households in credit-card debt and PSID does
  not.** At ages 35-44 the simulated p75 is still -1,000, so over 75% of
  simulated households are borrowing, against 21.3% of PSID household-waves.
  Simulated median liquid is negative at every age (-3,000 to -13,000) where
  PSID's is 0. See §9.5 — this one is not about heterogeneity at all.

The age-20 seed itself is correctly ported: their `4_initialwealth.do:63` takes
`r(p50)` of the typical-household-adjusted ratio for `AGE <= 24`. But it is
computed on **credit-card holders only** (`drop if hasVisa != 1`), and our PSID
sample is not card-restricted — deviation 1 in section 6 — so the seed describes
a richer population than the one we apply it to.

### 9.2 Education group is the source that works

`scripts/gate_edu_mixture.py`. All three groups' first-stage estimates are in the
replication package and differ far more than any within-group dispersion:

```
group      auto   vareps   varnu  agecoef    cons  C0_cred  med_liq
comphs   0.8400   0.0571  0.0451   0.1350   7.563  0.16721  +0.0549
somehs   0.8104   0.0588  0.0686   0.0794   8.209  0.00057  -0.0371
compco   0.7624   0.0448  0.0303   0.2467   5.817  0.42195  +0.1923
```

Mixing the three uniformly widens `p(x|θ)` by **1.79x**, and the widening
**grows with age** (1.04 at 25-30 to 3.18 at 45-55) rather than decaying,
because education moves the income profile, AR(1), credit limit and initial
wealth together at every age.

It also moves the simulated spread toward PSID's on the features that were too
narrow — illiquid at 35-44 goes from 0.54x PSID to 1.19x, income from 0.81x to
1.09x.

**Two caveats, stated rather than buried:**

1. The mixture **over-widens liquid assets by 10-22x**. PSID's liquid IQR is
   tiny (776-4,661) because most households sit at ~0; the model already
   generated 4.8x too much liquid dispersion before any mixing. Education makes
   a pre-existing liquid-side mismatch worse, not better.
2. The comparison as run is **apples-to-oranges**: a 3-group mixture against a
   comphs-only PSID sample. The honest version needs the 1,635-household tensor
   without the comphs filter. Uniform weighting also deliberately
   over-represents somehs and compco relative to population shares.

### 9.3 Income process and returns

`scripts/gate_nuisance_mixture.py`. The single 10x10 `VCV_firststage_income.mat`
covers the whole first stage — 7 profile coefficients then 3 AR(1) parameters —
so all ten are drawn jointly rather than the AR(1) alone.

Income process gives **1.23x**, concentrated where it is needed: illiquid 2.00x
at 25-30, 1.43x at 35-44, 1.50x at 45-55. Returns give **1.08x** and are
marginal.

**The conceptual objection stands and is not resolved by the number.** This
bootstrap is *sampling uncertainty in a group-level estimate*, not household
heterogeneity — the relative sd runs from 3.4% on the constant to **62% on
`kidscoeff`**, which is a statement about how precisely the coefficient was
estimated, not about how much households differ. Returns have no calibrated
dispersion at all; +-2pp is a chosen sensitivity. Both must be reported as
sensitivity ranges, never as estimated quantities.

### 9.4 Gate 1 — observation noise: FAIL

`scripts/gate1_compare.py`. σ = 0.30 in logs on the four dollar features (age is
exact in PSID and never perturbed), applied identically to the training,
held-out and SBC windows, 5-seed retrain, then re-run on PSID.

**The two numbers the gate was written to check both moved the "right" way:**

```
                        baseline     noisy    change
rho at ceiling              9.3%      0.1%     -9.2pp
beta between/within         0.80      0.87     +0.06
```

**That was a PASS on the criterion as originally written, and the criterion was
wrong.** It had no guard against the estimates simply becoming uninformative,
which is what happened. Held-out recovery on *simulated* data, where the truth
is known:

```
            corr base  corr noisy   mae base  mae noisy  mae change
beta            0.799       0.611     0.0906     0.1326      +46.4%
delta           0.817       0.747     0.0174     0.0216      +24.1%
crra            0.847       0.683     0.4220     0.7177      +70.1%
```

The network can no longer recover parameters it demonstrably recovered before.
The tell is that **ρ between/within collapsed 6.62 → 0.79** — the project's one
solid heterogeneity finding evaporates, because ρ's within-household sd widened
8.3× (0.107 → 0.887). A posterior that wide stops concentrating near any
boundary, which is the whole reason the ceiling pileup fell.

The two readings — *"noise revealed the sharp estimates were overconfident"* and
*"noise destroyed the signal"* — are **indistinguishable from the PSID numbers
alone**, and separable only on simulated data. There the answer is unambiguous.

The gate now carries a third condition, that held-out recovery must survive.
Recording that the first version would have passed this run is the point: two
sensible-looking criteria were not enough.

### 9.5 Decision

```
Gate 1  observation noise      FAIL   destroys signal, does not explain misfit
Gate 2  initial wealth         FAIL   forgotten by age 35
Gate 2b education group        PASS   1.79x, grows with age
Gate 2c income process         weak   1.23x, and the wrong object conceptually
Gate 2c returns                weak   1.08x, no calibrated dispersion at all
```

Regeneration proceeds on the **narrowed scope**: education group as the
justifying source, with the sources that cost no extra solve carried along.
The four-source design in the original plan is not what survived.

### 9.5 The credit-card margin is wrong at every θ, including theirs

`scripts/borrowing_margin.py`. Share of households in net credit-card debt:

```
ages      PSID gross  PSID net   ours (post. median)   Laibson et al. MSM
25-30          31.5%     19.5%                 47.0%                61.2%
31-34          34.3%     22.0%                 71.1%                59.2%
35-44          34.3%     21.3%                 76.1%                56.1%
45-55          37.9%     21.4%                 77.6%                49.6%
```

**Two denominators, because the model admits only one account.** Its `X` is a
single net position: a household cannot hold cash and card debt at once, they
*are* the same account. PSID households can and do — **13.9% of household-waves
hold card debt alongside non-negative net liquid**, and the model cannot
represent them at all. So PSID's gross incidence (35.2%) and its net-negative
share (21.3%) bracket what `X < 0` is trying to be, and the overshoot is quoted
against the bracket: **1.5–2.2× against gross, 2.4–3.6× against net.**

The two-asset model exists to explain the **credit-card debt puzzle** —
households holding illiquid wealth while revolving expensive card debt — and
that is also the margin β is identified off: present bias is what makes a
household borrow at 10.59% while holding an asset returning 5%. So this is the
one moment the model should get right.

It over-generates borrowers by **1.5–3.6× depending on the denominator, at
Laibson et al.'s own MSM estimate as well as ours**. That rules out the parameters we recovered as the cause. It
also rules out heterogeneity as the fix: this is the location of the whole
distribution, not its width, and §9.1–9.4 are all about width.

PSID's borrowing share is essentially **flat in age** (19.5 / 22.0 / 21.3 /
21.4). The model's is steeply age-varying and *in opposite directions at the two
θ* — rising 47%→78% at ours, falling 61%→50% at theirs. Whatever the model is
doing on this margin, it is not what the data do.

**Caveat.** Their α = 2.02 doubling of SCF card debt (§6, deviation 2) scales
the *amount* owed, not the *incidence* of owing, so it moves this comparison far
less than it moves debt levels — but households who deny card debt outright are
still missed on the PSID side. Deviation 1 compounds it: their sample is
conditional on *holding* a card and ours is not, so our denominator includes
households that cannot borrow at all. Both push the same way, and neither is
remotely large enough to close even the 1.5× low end.

**This displaces heterogeneity as the leading explanation for β.** §7.1 proposed
that missing precautionary motives are routed through ρ; that still stands for
ρ. But β is identified off borrowing, and borrowing is misfit by 3× before any
parameter is estimated. Any β this project reports is conditional on that.

### 9.6 Within-group initial wealth: carried as code, not used in generation

`scripts/gate_initwealth_marginal.py`. Gate 2 killed initial wealth as a
*justification* for regenerating. It costs no extra solve, though — it is a
forward-pass argument — so the remaining question was whether to carry it along
anyway. That is a different question, because **education already moves initial
wealth between groups** as one of its four blocks:

```
MED_LIQ_WEALTH     comphs +0.0549   somehs -0.0371   compco +0.1923
MED_TOTAL_WEALTH   comphs  1.4696   somehs  1.0999   compco  3.5894
```

so what was undecided is only the *within*-group spread on top of that shift.
Measured by holding θ and the three education solves fixed and varying only the
forward-pass seeding, with each group's PSID draws recentred on its own SCF
median:

```
band     marginal widening   share of PSID household-waves
25-30          1.13x                     8.1%
31-34          1.03x                    12.9%
35-44          1.02x                    46.0%
45-55          1.00x                    30.6%
weighted by where our data is:  1.02x
```

**No, and the reason is not just that it is small.** The one place it acts is
illiquid at ages 25-30 (1.50x), and that is precisely where the model is
*already* over-dispersed: education-only illiquid IQR there is 32,000 against
PSID's 13,106, and adding initial wealth takes it to 48,000. It moves the fit
the wrong way at the only ages where it does anything.

That is the Gate 1 failure mode restated — buying width the data does not ask
for, at the cost of information per household. `simulate(initial_wealth=...)`
stays in the code and stays tested; it is simply not exercised during
generation.

### 9.7 The principle Gate 1 established, applied to the remaining sources

Gate 1 added noise to `x` *after* simulation: pure information destruction with
no structural content, and recovery collapsed. Education is different in kind —
households genuinely differ in their income profile and credit access, so the
widening is real variation rather than added noise.

But the decisive difference is that **education is observed in PSID**. That lets
the posterior be *conditioned* on it, `q(θ | x, edu)`, which recovers the
information the extra dispersion costs — exactly as `age` is already handled.
An unobserved nuisance can only be marginalised, and marginalising is pure
width: the Gate 1 mechanism with a nicer justification.

```
source                     widening   observed in PSID   verdict
education group              1.79x           yes         IN, conditioned
income process               1.23x           no          OUT
returns (R_gamma)            1.08x           no          OUT
within-group initial wealth  1.02x           n/a         OUT (section 9.6)
M = 8 households per solve    n/a            n/a         IN (see below)
```

`M = 8` is the one unambiguous free win: `dispatch.py:108` simulates one
household per 12.4 s solve, and the forward pass is negligible against the
solve. These are genuine independent draws from `p(x | θ, nuisance)`, unlike the
`k` window augmentation which reuses one trajectory. It adds no dispersion the
data has to absorb — it reduces Monte Carlo error in the training target.

The income-process bootstrap is dropped on both grounds at once: it is
unobserved, *and* it was never the right object (§9.3 — a 62% relative sd on
`kidscoeff` is estimation precision, not household variation).

---

## 10. Regeneration plumbing

Implemented; the long run has not been started.

### 10.1 `Calibration` bundles

`grids.py` read `cal.*` module globals **at call time**, so a `ModelSpec` could
not express "solve this draw as somehs" — it would return a comphs solution and
nothing would complain. Two thirds of a mixed dataset would have been silently
mislabelled.

`laibson_calibration.Calibration` is a frozen bundle of the blocks education
moves, with `COMPHS` / `SOMEHS` / `COMPCO` instances and `EDUC_GROUPS` fixing
the index order (a reordering would relabel every stored draw).
`ModelSpec.calib` carries the selected bundle, and every `grids.py` function
takes it as `c`, defaulting to `COMPHS`.

**Five blocks, not the four the plan named.** Demographics is education-varying
too (`a1_kids` runs 0.262 / 0.358 / 0.576 across somehs / comphs / compco) and
is included, because excluding it would give somehs households comphs family
structure — incoherent, since family size enters both the income profile and the
consumption equivalence scale. That couples to `scripts/typical_household.py`,
whose `model_kids` / `model_depadul` re-centre PSID moments on the model's own
demographic profile; both now take the bundle, so a per-group posterior must be
re-centred on **its own** group's profile.

Verification:

- `python -m hh_npe.simulator.laibson_calibration` now checks **all three**
  bundles bit-exactly against the `.mat` files (21 values each), plus that
  `COMPHS` agrees with the module globals. A wrong value in `SOMEHS` or `COMPCO`
  would disturb no existing result — it would only corrupt the two thirds of the
  new dataset nothing else checks.
- `tests/test_calibration_bundles.py` (34 tests): defaults are bit-identical to
  explicit comphs at every level from `grids` up to a full `solve`, the bundle
  demonstrably reaches the solver, and an explicit `psi` still overrides it.

### 10.2 M households per solve

`simulate_batch_twoasset_gpu(..., n_households=M)`. The forward pass is
negligible against the ~12.4 s solve, so M > 1 is nearly free, and unlike the
`k` window augmentation — which reuses one trajectory — these are genuine
independent draws from `p(x | θ)`.

**The load-bearing detail.** With M > 1 a shard holds M rows per θ.
`build_windowed` now repeats θ across them while keeping `panel_id` as the
**draw** index, because `_use_grouped_split` keys the train/validation split on
that id: two households of one θ on opposite sides of the split leak exactly as
two windows of one panel would, and neither the loss nor the coverage would show
it. M is recovered from the row count, so pre-M shards — all 256 existing ones,
which have no such field — keep working unchanged.

Education draws are grouped **before** solving, because `solve_batch` shares one
calibration across a GPU batch; mixing groups within a batch would silently
solve them all as the first one. Households are seeded off the draw index rather
than a running counter, so grouping does not change what any draw produces.

### 10.3 What the shard records

`educ` (per draw) and `n_households` are written into every shard and into
`solver_config.json`. The education index cannot be recovered afterwards, and
without it the dataset supports none of marginalising, conditioning, or
reweighting to population shares — which is the entire reason for drawing it.

Two bugs the end-to-end smoke test caught that no unit test would have:

1. `assemble()` assumed one row per draw and crashed on the boolean mask
   (`size of axis is 32 but ... 128`). It now repeats θ per household using the
   shard's own `n_households`.
2. `--help` crashed with `TypeError: %o format` — a pre-existing latent bug,
   since argparse `%`-formats help text and an unescaped `~99% of` parses as a
   conversion. Unrelated to this work, surfaced by it.

Smoke test (32 draws, coarse grid, `--n_households 4 --educ mixed`):

```
educ counts over 32 draws   [10 10 12]     (uniform target ~10.7)
panel rows                  64 = 16 draws x 4 households
windowed                    128 rows, 32 unique panel_id, exactly 4 per draw
theta constant within draw  yes
households within a draw    differ
median income by group      comphs 52000   somehs 30000   compco 72000
```

### 10.4 Choosing `k`

`k` is **windows per simulated panel**, not waves per window. The chain is
`θ ~ prior`, one solve, a full lifecycle panel over ages 20-90, then `k` random
start ages, each cutting an `n_waves`-long observation window. Total rows are
`draws × M × k`.

**`k` is not a generation decision.** Shards store the annual panel and
`build_windowed` cuts windows at training time, so `k` can be swept afterwards
at zero solve cost. Only `M` is fixed at generation, because it changes what the
shard contains.

Direct evidence from the earlier sweep, all at `M = 1`:

```
        beta corr   rho corr   beta mae   rho mae
k=5        0.752      0.817     0.1040    0.4927
k=8        0.761      0.832     0.1007    0.4694
k=10       0.760      0.828     0.1014    0.4662
```

It plateaus at 8. But **`M` and `k` substitute for start-age coverage and not
for independence**: each of the `M` households draws its own start ages, so
windows per θ is `M × k` either way. At `M=8, k=1` those 8 windows come from
**8 independent households**; at `M=1, k=8` they come from **one trajectory** --
overlapping years, one shared income-shock realisation.

So `M=8, k=1` dominates the current `M=1, k=8`: identical row count
(57,344 × 8 = 458,752), identical start-age coverage, identical training time,
but every row is an independent draw from `p(x | θ)` rather than a correlated
slice of one path. That independence is exactly what the network needs in order
to learn the dispersion the education mixture adds.

**Decision: generate at `M = 8`, train at `k = 1`.** Sweep `k = 2` afterwards as
a cheap follow-up -- re-windowing only, no re-solving -- while expecting little
from it, since 8 windows per θ was already the plateau and these 8 carry
strictly more information than the old 8.

### 10.5 Not yet done
- SBC's cached `sbc_sims.pt` becomes invalid — simulated under the old model.
### 10.7 The generation run

`scripts/run_phase4_generation.sh`. Two stages, chained so the second cannot
start if the first fails:

```
1. SBC simulations   ~3.5 h    1000 draws, educ=mixed
2. Generation        ~9.4 d    65536 draws, M=8, educ=mixed
```

**SBC runs first, and not only because it is short.** It exercises the new
`educ` grouping in `solve_batch` at production settings — full grid,
`theta_batch 16`, `chunk 16` — so a defect in that path costs 3.5 h instead of
9.4 days.

SBC must simulate under the *same* process the training set uses. A marginalised
posterior validated against comphs-only simulations would have its calibration
score measure the mismatch rather than the posterior, and would look like
miscalibration no amount of training could fix. `educ` is therefore part of the
`--sbc_cache` key: the existing cache carries no `educ` field, reads back as
`comphs`, and is correctly rejected under `mixed` rather than silently reused.

Settings, and why:

| | | |
|---|---|---|
| grid | `full` | Laibson et al.'s exact 190×84; the 12.45 s/draw the budget is built on |
| draws | 65,536 | unchanged — `M` sharpens the conditional at each θ, not the coverage of θ, so the draw count still sets the effective sample |
| `M` | 8 | nearly free against a 12.4 s solve |
| `k` | 1 at training | §10.4; free to re-sweep later |
| shard dir | `phase4_educ_dataset_shards` | `_check_config` refuses to mix configs, correctly — the new run must not write into `phase3_dataset_shards` |
| GPU | V100 | the same card the existing dataset used, so no cross-architecture mixing |

Disk: 2.18 GB projected at M=8, measured from the smoke shard (16.6 KB/draw at
M=4). 9.9 GB free after clearing 9.3 GB of `uv` cache.

The run is resumable — `generate_dataset.py` skips existing shards — and the
education draw is taken from the run seed, so a shard written after an
interruption describes the same draw it would have originally.

**Run log.** Started 2026-09-08 01:52 UTC.

```
stage 1  SBC simulations   3.46 h   (predicted 3.5)
         educ counts [335 342 323] over 1000 draws, uniform target 333
         same thetas as the comphs cache, different panels -- so SBC now
         differs from the previous run in the generative process only
stage 2  generation        started 05:20 UTC, 12.5 s/sample, ETA 9.40 d
         shard size 16.0 MB x 128 = 2.05 GB (projected 2.18)
```

SBC uses **M = 1**, not M = 8. It validates `q(θ | x)` for a *single*
observation, and each training row is one household's `x`, so one household per
θ is the marginal each row is drawn from. M > 1 would give M observations per θ
and muddle the rank computation.

The precheck's 13.07 s/draw overstated, as expected: 32 draws is three partial
batches against a `theta_batch` of 16, average fill 10.7. At `block = 512` the
fill is 15.5 and the measured rate is 12.5 s/sample, which is the figure the
9.4-day budget was built on.

First shard verified rather than assumed: 512 draws, `M = 8`, `x` of
(4096, 7, 4) = 512 × 8, `educ` of length 512 (draws, not rows), 100% alive, and
the income ordering somehs 33,000 < comphs 49,000 < compco 78,000 reproducing on
simulated data what §10.6 finds independently in PSID.

**One accepted imperfection.** With education grouped before solving, each
group's last batch within a block is partial, and `solve_batch` is reproducible
only at a fixed `theta_batch`. That is a tie-breaking difference, not an error:
~99% of states hold two exactly-tied choices, so either is optimal and the
resulting `(θ, x)` pair is still a valid draw from the joint. It matters only
for bit-exact reproduction, and SBC and training share one config, which is the
property that actually has to hold.

### 10.6 The all-groups PSID tensor

`build_psid_tensor.py --educ_groups` keeps every education group and records the
group index per household, applying the rest of their sample filter (not
self-employed, no business or farm income), which is group-independent.

```
head-in-all-7 cohort           2119
+ usable education             2108
+ not self-employed            1875
+ no business/farm income      1627

comphs  889     somehs  211     compco  527
```

**1,627, not the 1,635 the plan estimated.** The mapping is by year equivalent of
their `2_buildmoments.do:69-74` SCF `EDCL` categories — somehs `EDCL 1`, comphs
`EDCL 2-3`, compco `EDCL 4` — and `EDUC_GROUPS` is asserted equal to
`laibson_calibration.EDUC_GROUPS`, since the stored index is read back through
the latter and a drift would relabel every household.

**Cross-check: the comphs subset reproduces the existing 889 exactly** — same
`psid_row` set, and identical on income, liquid, illiquid and age. Consumption
differs slightly because the rental growth rate is self-calibrated on the anchor
sample, which is now 1,627 households (2.61%/yr) rather than 889 (2.49%/yr).
The effect is immaterial: median relative change 0.14%, p90 0.37%, and the
sample median consumption moves +0.05% — inside the ±0.16% imputation noise §5
already reports.

**The mapping validates against the calibration.** The income ordering holds on
both sides, which it need not have:

```
group      N     PSID income   model income   ratio    PSID cons   borrow%
comphs   889          43353          55601     1.28        35743     21.3%
somehs   211          28947          37028     1.28        28253     11.5%
compco   527          79503          83723     1.05        54462     18.7%
```

**And a new observation worth following up.** The model over-predicts income by
~28% for both non-college groups but only 5% for college graduates — consistent
with the college wage premium widening since their 1982-91 estimation window.
§4 tested "calibration vintage" and refuted it, but that test was comphs-only;
the per-group view is more nuanced. *Caveat:* the model figure is a mean and
PSID's a median, so the levels are not strictly comparable — only the pattern
across groups is, and only to the extent skewness is similar across them.

`somehs` also borrows at half the rate of the other two (11.5% against 21.3%
and 18.7%), which is what its near-zero credit limit (`C0_CREDIT` 0.00057
against comphs's 0.167) would predict.
