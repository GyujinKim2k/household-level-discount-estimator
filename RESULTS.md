# Phase 4 — PSID empirical results

**Status:** Phase 4 complete. The §7.1 limitation that motivated the
regeneration has been addressed; §10.13 reports what that bought and §11 the
corrections an audit turned up afterwards.
**Last updated:** 2026-09-19.

Per-household posteriors over (β, δ, ρ) for 889 PSID households observed in
seven biennial waves, 2011–2023, against Laibson, Lee, Maxted, Repetto &
Tobacman's single population MSM estimate.

---

## 1. Headline

**Current** (Phase 4, `M=8`, corrected window, 889 comphs households). The
numbers this section carried before were computed on the misaligned window of
§10.8 and are superseded; see §10.9 and §11 for what changed and why.

```
                          beta       delta        crra
median of means         0.8100      0.9922      4.4625
mean of means           0.7854      0.9811      4.0986
sd across households    0.0948      0.0241      0.9560
median posterior sd     0.1530      0.0108      0.1766
between/within ratio      0.62        2.23        5.41

Laibson et al. MSM      0.5305      0.9891      1.9355
  their std error       0.1140      0.0051      0.4350

share of households whose 90% CI covers their estimate:
  beta 0.557    delta 0.664    crra 0.238
```

**δ replicates** — 0.9922 against their 0.9891. **β and ρ are both
substantially higher**, and β's gap is not explained by education composition:
it is 0.81 / 0.79 / 0.79 across comphs / somehs / compco (§10.13).

Their estimate is not *excluded*: 55.7% of households' 90% intervals cover their
β and 23.8% cover their ρ, and in `figures/05_phase4_per_group.png` their point
sits outside every group's 68% contour but inside the 95% ones.

### Is the heterogeneity real?

```
            between-hh sd   within-hh sd   ratio
beta               0.0948         0.1530    0.62
delta              0.0241         0.0108    2.23
crra               0.9560         0.1766    5.41
```

A ratio below 1 means households differ by *less* than any one of them is
uncertain — the apparent spread is estimation noise, and a single population
estimate would serve as well.

- **ρ heterogeneity is real** (5.4×).
- **δ heterogeneity is real** (2.2×).
- **β heterogeneity is not demonstrated** (0.62), and Phase 4 made the evidence
  against it *stronger*, not weaker — §10.13. It replicates in all three
  education groups independently (0.62 / 0.83 / 0.74), and survives correcting
  for β's known over-coverage, which would move it only to 0.68 (§11.2).

**The sharper statement (§11.3).** Decomposing
`Var(means) = Var(true θ) + Var(estimation error)` gives a **negative** estimate
of `Var(true β)` in every run and every education group, including the Phase 3
baseline: the posterior means are *less* spread than the posteriors are wide.
So the finding is not merely "the ratio is below 1" but **no evidence of
between-household variation in β at all** — and it predates the regeneration
rather than being caused by it. Implied between-household sds are ρ 0.27–0.65,
δ 0.014–0.019, β none detectable. **This is a finding about the project's own
premise and belongs in any writeup.**

`within-hh sd` is the **median posterior sd across all households**, not one
representative household's. The representative-household version is unstable —
it swung 2.3× across runs and would have reported ρ heterogeneity tripling when
it in fact declined (§10.13).

Within households the parameters are close to orthogonal — re-measured on the
Phase 4 posteriors (60 households, 1,500 draws each): median
`corr(β,ρ) = −0.012`, `corr(β,δ) = −0.164`, `corr(δ,ρ) = −0.010`. Essentially
unchanged from the baseline's `+0.017 / −0.166 / −0.076`. There is no strong
β–ρ ridge, so the estimates are not trading off against each other.

The *median* hides real spread, though: `corr(δ,ρ)` runs from −0.35 at p10 to
+0.61 at p90. Orthogonality is a statement about the typical household, not
about every one of them.

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

## 7. Open issues *(as of Phase 3 — superseded; see §10–§11)*

> **This section records the state that motivated the regeneration.** §7.1 was
> the case for spending nine GPU-days; §10.13 reports what that bought and §11
> corrects two claims made here and in §10. Read §1 for current numbers.
> Specifically: the ρ figure below is a mean-proximity statistic that §11.1
> retires in favour of credible-interval truncation, and §7.1's "ρ ratio of 7.0"
> is a single-household estimate that §10.13 shows to be unstable.

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

- Consumption slope still 0.554 against 0.807 simulated. **Still open.**
- 9.3% of households have posterior-mean ρ within 5% of the 5.0 ceiling. Down
  sharply from before the rental fix; widening the prior is **no longer the
  obvious next step**, and would move along a ridge the data do not resolve.
  **Now 2.5% on this metric, and 0.1% by the truncation measure §11.1 argues
  for. Effectively closed.**
- β heterogeneity remains within estimation noise (§1). **Still true, and Phase
  4 strengthened it: the ratio fell 0.80 → 0.62 and replicates in all three
  education groups.**
- `SIMULATOR_SPEC` §7's Phase 1 gate — reproducing HARK/Carroll buffer-stock
  profiles — still has no test. **Still open.**

**Added since, and now the largest open items:**

- **The credit-card margin over-predicts by ~1.3× at ages 31–40** (§12.2,
  retracting §9.5's 1.5–3.6×). The model reproduces its own SCF `%Visa` target
  within 5% and is exact at ages 41–50; what remains is that SCF and PSID
  disagree on the *age gradient* of card debt and the model inherits SCF's.
  No fix exists inside the model — PSID records balances, not limits.
- **δ's upper bound truncates ~24% of comphs and 52% of compco posteriors**
  (§11.1). Not the "pinning" §10.13 claimed, and improving, but real.
- **compco's wealth range exceeds what the model can represent** — 17.1% of its
  households have under half their posterior mass inside the prior box.

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

> **Largely retracted — see §12.2.** This section compares a *cardholder-
> conditional* model against an *unconditional* PSID sample, which guarantees
> an apparent over-prediction. Corrected, the ratio is 1.14× sample-weighted
> rather than 1.5–3.6×, and the model reproduces its own `%Visa` calibration
> target within 5%. The text below is kept as the record of what was believed.

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

### 10.5 The generation run

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

### 10.8 An off-by-one in the window start age, found mid-run

Validating the training input path against real shards at the shard-16 milestone
(8,192 draws — a power-of-2 Sobol prefix, so a valid partial dataset) turned up
a **pre-existing** misalignment that affects every previous result too.

`aggregate_waves` reports each wave at the *last* year of its window
(`series[:, t1 - 1]`), so a window opened at `start_age = S` reports its first
wave at age **S + 1**. Passing `--start_low 25 --start_high 46` therefore
produces simulated wave-0 ages of **26–47**, while PSID's are **25–46**:

```
start_low=25 start_high=46   wave0 age 26-47   wave6 38-59
start_low=24 start_high=45   wave0 age 25-46   wave6 37-58
PSID all-groups              wave0 age 25-46   wave6 36-59
```

**Impact: ~5% of PSID households sit outside the training support in `age`** —
48 of 889 comphs and 78 of 1,627 all-groups are aged 25 at wave 0, an age the
network never saw. It has to extrapolate in a feature it otherwise anchors on.
How much that moved those households' posteriors is **not measured**; the point
here is the gap, not a quantified bias.

**The fix is free and belongs at training time:** `--start_low 24
--start_high 45`. Window start ages are a `compare_windows` argument, not a
generation one — the shards store annual panels — so this costs nothing and
needs no change to the run in progress.

The residual difference at the far end (37–58 against 36–59) is **not**
fixable by shifting. PSID interviews are 11–13 years apart across seven waves
(§2's own filter), while the simulator's are exactly 12. Shifting the start
cannot manufacture a variable span.

The rest of the validation passed on real shards: 13,824 draws → 110,592 rows at
`M=8, k=1`, exactly 8 rows per draw, `panel_id` contiguous and equal to the draw
index, θ matching its Sobol draw on 2,000 spot checks, education recoverable
through `panel_id` and near-uniform, and the group income ordering holding
(comphs 49,000 / somehs 32,000 / compco 79,000).

### 10.9 The start-age correction: hypothesis refuted, but the fix matters anyway

The §10.8 fix was re-run as a paired experiment — `outputs/startage_fix`,
identical to `outputs/flow_fix` in every argument except
`--start_low 24 --start_high 45`, same shards, same `k`, same five seeds.

**The hypothesis was that the ~5% of households aged 25 at wave 0, which were
extrapolating outside the training support, would move and the rest would not.
That is not what happened.**

```
change, corrected minus baseline   d beta    d delta    d crra    d in-box
age 25 (affected, N=48)           -0.0253    +0.0033   -0.0203    -0.0017
age >25 (control, N=841)          -0.0232    +0.0009   -0.0396    +0.0035
all households (N=889)            -0.0232    +0.0009   -0.0322    +0.0015
```

β moved by the same −0.023 in both groups, and ρ moved **more** in the control
(−0.040) than in the supposedly affected group (−0.020). By the criterion stated
before the run — *"a shift in both is retraining noise; a shift confined to the
age-25 households is the alignment"* — the support gap had **no detectable
differential effect**. The 48 households that were extrapolating were not
meaningfully harmed by it.

**But the correction changed the population estimates anyway, and one change is
large:**

```
                      baseline   corrected
beta    median          0.8465      0.8234
crra    median          4.5002      4.4680
rho at ceiling            9.3%        4.2%      <- less than half
delta at ceiling         43.8%       45.2%
in-box mass p10           0.638       0.739
```

**The ρ ceiling pileup more than halved.** That is one of the two numbers the
whole regeneration exists to move (§9.4), and part of it came free from a window
alignment. In-box mass at p10 improved 0.638 → 0.739, so the model represents the
hard tail of the sample better. δ pinning is unchanged, slightly worse.

**This is not a worse model on a better window — it is the same model.** Held-out
recovery is flat:

```
            corr base  corr fixed   mae base  mae fixed
beta            0.799       0.793     0.0906     0.0917
delta           0.817       0.816     0.0174     0.0175
crra            0.847       0.846     0.4220     0.4238
coverage_90    .920/.910/.909       .918/.904/.913
```

The per-seed `log q` is lower for the corrected runs (5.153–5.202 against
5.234–5.274) and the ensemble's is 5.436 against 5.471. **That is not evidence
of a worse fit**: each is scored on its own window, and the corrected window
covers younger households, where less wealth has accumulated and there is less
signal about (β, δ, ρ). It is a harder task, not a worse model, and the two
numbers are not on a common scale.

**Consequence for §1.** The headline numbers there were computed on the
misaligned window. The corrected values are β 0.8234, δ 0.9907, ρ 4.4680, with
the ρ pileup at 4.2%. β moves *toward* Laibson et al.'s 0.5305 but remains far
from it. Seed-level uncertainty on these shifts has **not** been quantified —
the ρ pileup change is the robust signal, being a large relative move; the
β shift of −0.023 against a between-household sd of 0.10 is 0.23 sd and should
be treated as suggestive.

### 10.10 What the regeneration can and cannot deliver — a correction to §9.7

§9.7 argued education in on the grounds that it is **observed**, so the posterior
can be conditioned on it and recover the information the extra dispersion costs.
That reasoning is right, and it has a consequence I did not draw out at the time:

**A comphs household under the conditioned model gets the comphs calibration —
which is exactly the old baseline.** Conditioning on education does not widen
`p(x | θ)` for that household at all. So for the 889 comphs households, the
education mixture cannot by itself move the ρ pileup or the β ratio.

The widening that §9.2 measured (1.79×) is a property of the **marginalised**
model, which deliberately discards education even though we observe it. That is
the Gate 1 pattern in a new costume: width bought by throwing information away.
It is worth computing as one variant, but it is not the variant to prefer, and I
should not have implied the 1.79× would transfer to the conditioned case.

**So what does move the two questions? `M = 8`, not education.** With `M = 1` the
network sees one `x` per θ and must infer how much `x` varies at fixed θ from the
population. With `M = 8` it sees eight, which teaches the conditional spread
directly. That should *widen* within-household posteriors where they were
over-confident, which is precisely the mechanism behind the inflated ρ
between/within ratio of 7.0 — if the model cannot generate within-θ dispersion,
the network has to attribute observed dispersion to θ.

Revised expectations, recorded before the results exist so they cannot be fitted
to them afterwards:

| deliverable | source | confidence |
|---|---|---|
| ρ between/within ratio falls toward truth | `M = 8` | the clearest mechanism |
| β between/within ratio rises above 1.0 | `M = 8`, only if between-household variation is real | genuinely uncertain |
| ρ ceiling pileup falls further | already 9.3% → 4.2% from §10.9 | most of it may be spent |
| per-group posteriors for somehs (211) and compco (527) | education bundles | new, and the clearest gain |
| pooled conditioned posterior over all 1,627 | education bundles | sharper than marginalised |

**The honest summary is that the strongest case for the nine days is the
per-group science — 738 households that could not be analysed at all before —
and `M = 8`'s effect on within-household calibration. The education mixture's
1.79× widening is real but applies to the variant we have reason not to prefer.**

### 10.11 A confound in the per-group comparison, found at launch

The per-group models train on **one third of the θ draws**. Uniform education
sampling gives ~21,845 draws per group, and after the train/held-out split the
comphs model sees **19,049 draws** against the baseline's **57,344**:

```
                        draws    rows      windows per draw
baseline (comphs-only)  57344  412880      k=8 x M=1
phase4 comphs           19049  152392      k=1 x M=8
phase4 conditioned      57344  458752      k=1 x M=8
```

Rows are comparable; **independent θ draws are not**. The effective sample for
learning `p(θ | x)` is the draw count, so the per-group comphs model is
θ-starved by 3× relative to the baseline it would naturally be compared against.
Any degradation it shows could be that alone, with nothing to do with `M = 8` or
education.

**So the comphs-vs-baseline comparison should lead with the *conditioned* model,
not the per-group one.** The conditioned model keeps all 57,344 draws and still
gives a comphs household the comphs calibration, which is the like-for-like
contrast. The per-group models remain the right vehicle for somehs and compco —
there is no baseline for those at all — but for comphs they answer a different
question than they appear to.

This was foreseeable from the plan's own arithmetic ("uniform gives ~21,845
draws per group") and I did not draw the consequence until the filter printed
its row count. Recorded here so the comparison is not read the wrong way later.

### 10.12 Conditioned vs marginalised: right on simulated data, mixed on PSID

Same 1,627 households, same rows, differing only in whether the observed
education group is supplied to the network.

**On held-out simulated data, conditioning wins on everything**, as §10.10
predicted:

```
                  beta corr  beta mae   crra corr  crra mae    log q
cond (uses educ)      0.763    0.0983       0.838    0.4615    5.090
marg (withholds)      0.735    0.1051       0.822    0.4953    4.759
```

**On PSID it is mixed, and that is the interesting part:**

```
                        beta     delta      crra
median (cond)         0.8220    0.9942    4.4437
median (marg)         0.8000    0.9846    4.5305

median posterior sd, cond    0.1561    0.0102    0.1236
median posterior sd, marg    0.1556    0.0185    0.1030
marg / cond width             0.997     1.820     0.833

at ceiling, cond               0.0%     54.0%      8.2%
at ceiling, marg               0.1%     32.8%     16.8%
in-box median, cond           0.962  (p10 0.643)
in-box median, marg           0.981  (p10 0.632)
```

Conditioning makes δ **1.82× sharper**, leaves β unchanged, and makes ρ
**0.83× — that is, wider**. It also pins far more households at the δ ceiling
(54.0% against 32.8%) while pinning fewer at the ρ ceiling (8.2% against 16.8%),
and the marginalised model represents the data slightly better (in-box 0.981
against 0.962).

**Why the two disagree.** "Conditioning is strictly sharper" holds for the
*true* posterior and for a correctly specified model. On simulated data the
education-specific calibration is correct by construction, so supplying it adds
real information and every metric improves. On PSID the model is misspecified in
ways we have already documented — the credit-card margin is wrong by 1.5-3.6×
at every θ (§9.5), and compco households sit outside the wealth range the model
can represent (§10.9's per-group in-box p10 of 0.434). Conditioning forces the
network to commit to a particular group's calibration; marginalising hedges
across all three. **Under misspecification, hedging can be the more robust
choice, and that is what the ρ and in-box numbers show.**

So §10.10's argument was right about the mechanism and too confident about the
conclusion. The correct statement is: conditioning is preferable when the model
is right, and we have independent evidence this model is wrong in specific,
identified ways. Neither variant should be reported as *the* answer — the
disagreement between them is itself a measure of how much the education
calibration is doing, and how much of it the data will not support.

### 10.13 Phase 4 headline results

Figure: `figures/05_phase4_per_group.png`. **Laibson et al.'s estimate sits
outside all three groups' 68% contours but inside their 95% contours** — in the
β–ρ panel especially, the dashed contours extend down toward it. Their point is
not excluded by these posteriors; it sits in the tail.

```
group      N     beta     delta     crra    d@ceil   r@ceil   in-box p10
comphs   889   0.8100    0.9922   4.4625     49.3%     2.5%        0.687
somehs   211   0.7881    0.9760   4.7155     30.3%    31.8%        0.714
compco   527   0.7888    0.9973   4.2254     76.7%     0.0%        0.434

Laibson et al. MSM (comphs only)
               0.5305    0.9891   1.9355
```

**The two questions the regeneration existed to answer:**

```
                              rho@ceiling    beta between/within
baseline (misaligned window)         9.3%                   0.80
+ start-age fix                      4.2%                   0.64
+ Phase 4 (M=8)                      2.5%                   0.62
```

- **ρ ceiling pileup: answered, yes.** 9.3% → 2.5%, and most of the first step
  came free from the §10.8 window fix rather than from the nine days.
- **β between/within: answered, no — and it moved the wrong way.** `M = 8`
  taught the network how much `x` varies at fixed θ, and the honest consequence
  is *wider* within-household posteriors (0.129 → 0.153) against a flat
  between-household spread (~0.095). **β heterogeneity is not demonstrated**,
  and the regeneration made the evidence against it stronger. That is a finding
  about the project's premise, not a shortfall to be explained away.

The result holds in all three education groups independently (ratios 0.62 /
0.83 / 0.74), so it is not an artifact of the comphs sample.

**A statistic that nearly misled us.** Computing within-household sd from a
single representative household gave ρ ratios of 7.38 → 17.67 → 13.96, which
reads as "ρ heterogeneity tripled". The denominator had swung 2.3× because it
rested on one household. Using the median posterior sd across all households
gives 7.28 → 7.07 → **5.41**, a modest decline. The fragility was flagged before
the run and the robust statistic is what §1 and this section report.

**Known gaps, recorded rather than hidden:**

- The five `cond` seeds saved their posteriors but died before writing their own
  scores (the SBC-conditioning bug of §10.12's run). The **ensemble** is fully
  scored; only the ensemble-vs-members comparison is missing for that variant,
  and recovering it would cost a full retrain for a diagnostic.
- `somehs`'s 31.8% ρ pinning and `compco`'s in-box p10 of 0.434 mean those two
  groups' estimates carry structural caveats the comphs one does not.
- δ pinning at the 1.0 ceiling **worsened** across the whole programme (43.8% →
  49.3% for comphs, 76.7% for compco). Nothing here addressed it, and it is the
  clearest remaining defect.

---

## 11. Audit corrections

### 11.1 The δ "ceiling pinning" claim was wrong

§10.13 reported δ pinning at 43.8% → 49.3% and called it "the clearest remaining
defect", worsening across the programme. **That is an artifact of the metric, and
the direction is backwards.**

The metric was "posterior *mean* within 5% of the prior span of the upper bound".
For δ, whose span is 0.15, that band starts at 0.9925 — and Laibson et al.'s own
estimate is 0.9891, only 0.003 below it. The band flags ordinary patience as
"pinned". It also conflates *concentrated near the top* with *truncated by the
prior*, which are different things.

δ is not against the wall at all:

```
run          p50      p90      p99      max   >0.999  >0.9999
comphs    0.9922   0.9982   0.9993   0.9997     3.1%     0.0%
somehs    0.9760   0.9971   0.9983   0.9986     0.0%     0.0%
compco    0.9973   0.9993   0.9996   0.9998    16.1%     0.0%
```

**No household reaches 0.9999, let alone 1.0.**

The correct diagnostic is whether the prior *truncates* the posterior — whether
a household's 95th percentile sits at the bound:

```
run           beta p95@hi   delta p95@hi   crra p95@hi
baseline            0.0%          29.9%          0.6%
startage            0.0%          25.8%          0.1%
ph4 comphs          0.0%          24.3%          0.1%
ph4 somehs          0.0%          15.6%          0.0%
ph4 compco          0.0%          52.0%          0.0%
```

By that measure:

- **δ truncation *improved*, 29.9% → 24.3%** — the opposite of what §10.13 said.
- **ρ truncation was already negligible (0.6%) and is now 0.1%.** The headline
  "ρ pileup 9.3% → 2.5%" is a mean-proximity number, not a truncation number;
  the improvement is real but the magnitude was overstated by the metric.
- **β is never truncated, at either bound, in any run.** Its bounds are not
  binding, so the β result of §10.13 does not depend on the prior.

**§10.13's "δ pinning worsened" sentence is retracted.** The real residual is
that ~24% of comphs and 52% of compco households have posteriors the δ ≤ 1
bound cuts into. That is worth reporting as a limitation — δ > 1 is not
economically meaningful, so this is the model saying "at least as patient as the
data can express" — but it is a modest and *improving* issue, not the clearest
remaining defect.

**Lesson for the metric, not just the number.** "Within 5% of the span" behaves
completely differently across parameters with different spans and different
posterior widths. Truncation of the credible interval is the parameter-free
question and should be the reported statistic.

### 11.2 β over-coverage does not rescue the heterogeneity result

β's 90% coverage is 0.913–0.950 against a 0.900 target, i.e. the posteriors are
slightly *too wide*. That inflates the within-household sd and biases the
between/within ratio **downward — against the heterogeneity hypothesis** — so it
had to be checked rather than assumed harmless.

Removing it generously (treating all the over-coverage as spurious width, a
1.098× inflation at coverage 0.929) moves the comphs ratio from **0.62 to 0.68**.
Still far below 1.0. The conclusion does not depend on it.

### 11.3 The variance decomposition, which is sharper than any ratio

`Var(posterior means) = Var(true θ) + Var(estimation error)`. With calibrated
posteriors the second term is the mean posterior variance, so

```
Var(true θ)  =  Var(means)  −  mean posterior variance
```

A **negative** estimate means the posterior means are less spread out than the
posteriors are wide: there is no between-household variation left after
accounting for estimation error.

```
                 Var(means)   mean post var    Var(true)   implied sd
ph4 comphs
  beta             0.008992        0.024484    -0.015492    none detectable
  delta            0.000581        0.000384    +0.000198        0.0141
  crra             0.913917        0.497606    +0.416311        0.6452
ph4 somehs
  beta             0.017026        0.024816    -0.007791    none detectable
  delta            0.001003        0.000630    +0.000374        0.0193
  crra             0.594133        0.520034    +0.074099        0.2722
ph4 compco
  beta             0.016512        0.029710    -0.013198    none detectable
  delta            0.000140        0.000165    -0.000025    none detectable
  crra             0.985424        0.771142    +0.214282        0.4629
baseline
  beta             0.010586        0.019107    -0.008520    none detectable
  crra             0.652451        0.510289    +0.142162        0.3770
```

**β's variance estimate is negative in every run and every education group,
including the Phase 3 baseline.** So this was never a Phase 4 artifact — β
heterogeneity was not detectable before the regeneration either, and the
regeneration only made that clearer. The honest statement is stronger than
"the ratio is below 1": **there is no evidence of between-household variation in
β at all.**

#### With bootstrap uncertainty, which the point estimates above hide

2,000 resamples over households. `P` is the probability that `Var(true θ) > 0`,
i.e. that between-household variation survives estimation error.

```
run             N   beta P   delta P   crra P    crra implied sd [95% CI]
ph4 comphs    889    0.000     1.000    1.000    0.643 [0.522, 0.745]
ph4 somehs    211    0.000     1.000    0.752    0.264 [0.000, 0.540]
ph4 compco    527    0.000     0.060    0.997    0.458 [0.265, 0.583]
ph4 cond     1627    0.000     1.000    0.597    0.096 [0.000, 0.298]
baseline      889    0.000     1.000    0.996    0.375 [0.210, 0.500]
```

- **β: `P = 0.000` in every run without exception.** The strongest statement in
  this document. Not a thin-sample artifact either — somehs's β ratio bootstraps
  to 0.83 with a 95% interval of [0.76, 0.90], entirely below 1.
- **δ heterogeneity is real** (`P = 1.000`) everywhere **except compco**
  (`P = 0.060`).
- **ρ heterogeneity is *not* uniformly established, which the point estimates
  above obscured.** It is solid for comphs (`P = 1.000`) and compco (0.997), but
  only 0.752 for somehs and **0.597 for the pooled conditioned model**, whose
  implied sd interval includes zero.

That last one deserves care rather than a headline. The conditioned model has
*wider* ρ posteriors than the marginalised one (§10.12), so more of the observed
spread is attributable to estimation error and less to real variation — a
mechanical effect, not necessarily an economic one. But there is also a
substantive reading available: conditioning on education absorbs between-group
differences in the income process into the calibration, leaving less residual
variation to attribute to risk aversion. **These two explanations are not
separated by anything we have run**, and the ρ heterogeneity claim should be
stated as "clear within comphs, not established pooled" rather than as a single
number.

The earlier sentence here — "ρ heterogeneity is real in all three groups,
implied sd 0.27–0.65" — was a point estimate reported without uncertainty and is
**superseded by this table**.

This should lead any writeup of the heterogeneity question, in place of the
between/within ratio — the ratio compresses a two-term decomposition into one
number and hides that one term exceeds the other.

### 11.4 Robustness to the consumption correction — and δ brackets Laibson

Every number reported anywhere in this document uses the **uncorrected** arm.
The corrected arm — which applies the Engel upper bound on differential
under-reporting, `φ' = 0.130`, so richer households are assumed to under-report
more (Aguiar & Bils) — has been computed throughout and never reported. It is a
free robustness check and it had not been run.

```
ph4 comphs      uncorrected   corrected     shift   as frac of between-sd
beta                 0.8100      0.8150   +0.0051                    0.05
delta                0.9922      0.9835   -0.0087                   -0.36
crra                 4.4625      4.4950   +0.0325                    0.03
```

**β and ρ are essentially unmoved** (≤0.20 of a between-household sd in the
worst group), and `Var(true β)` stays **negative in every group under both
arms**. The §11.3 conclusion does not depend on the correction.

**δ is the interesting one.** The correction moves it *down* past Laibson et
al.'s 0.9891, from 0.9922 to 0.9835. The two arms therefore **bracket their
estimate**: with no under-reporting correction we are slightly more patient than
them, with the maximum defensible correction slightly less. δ replicates their
value to within the width of the measurement-error assumption — a stronger
statement than the point comparison §1 makes, and one that cost nothing to
establish.

### 11.5 The illiquid floor biases β upward for indebted households

The illiquid feature is floored at zero because the model requires `Z ≥ 0`. The
code already flags this as real rather than an artifact, but its consequence for
the estimates had not been measured.

```
household-waves with net illiquid < 0 before flooring     17.9%
  median shortfall                                      -14,042
  p10 shortfall                                         -53,866
households floored in at least one wave                   45.4%
households floored in every wave                           1.9%
```

Comparing households floored in at least half their waves (N=122) against the
rest:

```
param      floored      rest      diff   in between-sd
beta        0.8419    0.8048   +0.0371            0.39
delta       0.9862    0.9928   -0.0066           -0.28
crra        4.5694    4.4492   +0.1203            0.13
```

**Flooring makes indebted households look wealthier than they are, and they are
assigned a higher β — less present bias — by 0.39 of a between-household sd.**
The direction is exactly what the mechanism predicts: a household whose net
illiquid position is −$50,000 but recorded as 0 does not look like it is
behaving impatiently.

This is a **candidate contributor to the β gap** against Laibson et al., and a
new one — it sits alongside §9.5's credit-card margin rather than replacing it.
Its size bounds the contribution: +0.037 on the 13.7% of households heavily
affected, against a total gap of 0.28, so it explains a modest fraction at most.

There is **no fix inside this model** — it cannot represent negative illiquid
wealth — so this belongs in the limitations rather than the to-do list.

### 11.6 The port still reproduces Laibson et al. after the Calibration refactor

§10.1 rewrote every `cal.*` read site in `grids.py` and `twoasset.py` to take an
education bundle. Unit tests confirmed the comphs bundle is bit-identical to the
old globals, but the end-to-end check — does the port still reproduce their
published table-3 moments — had not been re-run.

`scripts/validate_twoasset.py`, at their own estimates
(β=0.5305, δ=0.9891, ρ=1.9355):

```
grid 107x56 ("mid")        MSM objective q = 75.3   (theirs 77.2)
port fidelity               mean |log(ours/theirs)| = 0.0327
grid 81x46 ("coarse")      MSM objective q = 83.6
port fidelity               mean |log(ours/theirs)| = 0.0685
```

Moment-by-moment, `ours/theirs` runs 0.96–1.11 at the mid grid. **3.3% mean
deviation, against the "within 4%" recorded when the port was first validated
(commit `149206b`).** The refactor is clean.

The MSM objective itself moves between runs — 75.3 here against 85.3 recorded
earlier — because the forward simulation is stochastic and `q` is a sum of 16
squared standardised deviations, so it amplifies Monte Carlo noise. **Port
fidelity, not `q`, is the statistic to track**: it is a direct per-moment
comparison and is stable.

---

## 12. What the audit changed

Run after Phase 4 was declared complete. Two reported conclusions were wrong,
one was materially understated, and one recurring bug class was closed.

| | finding |
|---|---|
| **Retracted** | δ "ceiling pinning worsened, the clearest remaining defect" (§10.13). Metric artifact; by credible-interval truncation it *improved*, 29.9% → 24.3% (§11.1) |
| **Overstated** | ρ pileup "9.3% → 2.5%" is mean-proximity; by truncation it was 0.6% → 0.1%, already negligible before Phase 4 (§11.1) |
| **Understated** | β heterogeneity: not "ratio 0.62" but `Var(true β) < 0` with bootstrap `P = 0.000` in every run and group, *including the Phase 3 baseline* (§11.3) |
| **Overstated** | "ρ heterogeneity is real in all three groups" — `P = 0.752` for somehs and 0.597 for the pooled conditioned model (§11.3) |
| **New** | The illiquid floor biases β *upward* by 0.39 sd for indebted households — a previously unidentified contributor to the β gap (§11.5) |
| **New** | The two consumption-correction arms **bracket** Laibson's δ, a stronger replication claim than the point comparison (§11.4) |
| **Fixed** | `ensemble_eval` now refuses to score an ensemble on data its members were not trained on — the bug that produced an invalid Phase 4 comparison (§12.1) |
| **Verified** | Port fidelity 3.3% after the Calibration refactor (§11.6); headline numbers robust to dropping households the model cannot represent; orthogonality claim re-measured |

### 12.1 The recurring bug, and the guard

Three separate times a transformation was added at one site and missed at
another that consumed it: SBC not conditioned, SBC not filtered by group, and
`ensemble_eval` inheriting neither. The first aborted loudly on a feature-count
mismatch. **The second and third did not** — they produced plausible numbers
from the wrong data, and one of them was reported before being caught.

`compare_windows` now records `shards`, `sbc_cache`, `start_low`, `educ_group`
and `condition_educ` in each run's `_config`, and `ensemble_eval` refuses to
proceed when its own arguments disagree with what the members recorded. Verified
against the original failure: it catches both the `start_low` and `educ_group`
mismatches, and passes the correct invocation.

The general lesson is that `ensemble_eval` *rebuilds* its evaluation data rather
than reusing the members', which makes every training argument a silent
correctness dependency. Defaults that are right for one phase are wrong for the
next, and nothing in the type system notices.

### 12.2 §9.5's credit-card misfit is largely a denominator error — retracted

§9.5 reported the model over-generating card borrowers by **1.5–3.6× at every θ
including Laibson et al.'s own**, called it a location error no heterogeneity
could fix, and said it "caps what any β estimate here can mean". Setting out to
fix it showed there is much less to fix.

**The model matches its own calibration target.** Their `%Visa` moment is the
share holding card debt **conditional on holding a card** — `4_initialwealth.do`
drops `hasVisa != 1`. The port reproduces it within 5%:

```
ages        SCF target   our sim   ours/theirs
21-30           0.6395    0.6249          1.03
31-40           0.6292    0.6089          1.04
41-50           0.5884    0.5468          1.05
51-60           0.5027    0.4802          1.01
```

§9.5 compared that cardholder-conditional model against an **unconditional**
PSID sample. Non-cardholders sit in the denominator and can never enter the
numerator, so the comparison was guaranteed to show the model over-borrowing.
This is §6's deviation 1 — "`hasVisa` is not in PSID" — which was documented and
then not applied to the one comparison it governs.

PSID does not observe possession, so the conditional share is the unconditional
share divided by the holding rate. That rate is **at least 65.8%** (the share
ever reporting debt across seven waves) and ~70–76% for US families:

```
ages      uncond   /0.66   /0.70   /0.76     SCF   model   model/PSID@0.70
25-30      31.5%   47.7%   45.0%   41.4%   63.9%   62.5%              1.39
31-40      33.5%   50.8%   47.9%   44.1%   62.9%   60.9%              1.27
41-50      37.2%   56.4%   53.2%   49.0%   58.8%   54.7%              1.03
51-59      36.3%   55.0%   51.9%   47.8%   50.3%   48.0%              0.93
```

**Sample-weighted across our households' actual age distribution: 57.1% model
against 50.2% PSID-implied — 1.14×, not 1.5–3.6×.** At ages 41–50 the model is
exact (1.03) and by 51–59 it slightly *under*-predicts (0.93).

**What survives is smaller and more specific: an age-gradient disagreement
between SCF and PSID.** SCF has card-debt incidence *falling* with age
(63.9% → 50.3%); PSID has it *rising* (45.0% → 51.9%). The model is calibrated
to SCF and inherits SCF's gradient. Because our sample is concentrated at ages
31–50, it sits where the two sources disagree most, and the model over-predicts
by ~1.3× at the younger end.

**There is no fix inside the model.** The gradient comes from the credit-limit
function and income profile, both estimated on SCF. Re-estimating them against
PSID would be calibrating to the data we are testing against — and PSID records
balances, not limits, so it cannot identify the limit function anyway.

**Consequence for §9.5's strongest claim.** "β rides on a margin misfit by 3×,
capping what any β estimate here can mean" is **retracted**. The margin is
matched at ages 41+ and over-predicted by ~1.3× at 31–40. That is worth stating
as a limitation on β, but it is not the disqualifying defect §9.5 described.

This does not disturb §11.3: β heterogeneity is undetectable on internal
evidence — the variance decomposition — which does not depend on this
comparison at all.

### 12.3 "compco's wealth range exceeds the model" — wrong direction, wrong cause

§11 listed compco's in-box p10 of 0.434 as the model being unable to represent
its **wealthiest** households. Setting out to fix it showed the opposite.

`in_box_frac` is the fraction of posterior draws inside the **θ** box — it is a
statement about parameters, not about the wealth grid, and I read it as the
latter. The 90 compco households with in-box mass below 0.5 are the **poorest**
in the group:

```
                  low in-box (90)     rest (437)   ratio
median income               36,474         81,168    0.45
median consumption          27,925         56,277    0.50
median illiquid              9,458         90,174    0.10
```

In-box mass correlates **+0.57 with consumption** and +0.38 with income. Poor
college-educated households, not rich ones.

**The cause is the per-group model, not the model's wealth range.** The compco
calibration has the steepest income profile of the three (`agecoeff` 0.247,
mean income ~83,700 at ages 35–44). A per-group model trained only on compco
draws has **never seen a household earning 36,000**, because the compco
calibration does not generate one. Those households fall outside its training
support entirely.

The same 527 households under all three models:

```
                     per-group     cond     marg
all compco (527)         0.825    0.883    0.979
low in-box (90)          0.416    0.676    0.940
households < 0.5          17.1%     2.1%     3.2%
```

**Sharing one network across education groups fixes it** — the conditioned model
can borrow support from the comphs and somehs calibrations for a household its
own group's calibration cannot produce. Marginalising fixes it further still,
which is §10.12's hedging-under-misspecification result showing up concretely
rather than in the abstract.

**What this changes, and what it does not.** Switching compco to the conditioned
model barely moves the estimates — β +0.020, δ −0.0001, ρ +0.044 — so the
reported per-group numbers were not far wrong. But two things do change:

- **compco's ρ heterogeneity does not survive the switch.** Per-group implies a
  between-household sd of 0.463; conditioned implies none detectable. That
  finding was model-dependent and should not have been reported as a property
  of the group. Consistent with §11.3's `P = 0.597` for pooled-conditioned ρ.
- **δ truncation is unaffected** — 52.0% per-group against 51.8% conditioned.
  That is a genuinely separate problem and remains open.

**Recommendation: the conditioned model should be the default for all three
groups**, not the per-group models. The per-group models are θ-starved by 3×
(§10.11) *and* lack support for atypical members of their own group. Their only
advantage — that they cannot leak calibration across groups — is not worth
either cost.

**That prediction was wrong.** `somehs` should, on the group-calibration story,
have *rich* misfits — its calibration generates the least wealth. It does not:
its low-in-box households are also the poorest (income 10,801 against 33,219),
and comphs shows the same sign (`corr(in-box, income) = +0.38`,
`corr(in-box, consumption) = +0.57`). The pattern is universal, so the
explanation cannot be group-specific. §12.4 has the real one.

### 12.4 The real constraint: the income process cannot generate poor households

```
           model p1   model p5      PSID p1   PSID p5
comphs       15,000     19,000            0     6,819
somehs       10,000     13,000        1,516     1,516
compco       22,000     32,000        2,386    20,234
```

The simulated income floor is a 3-state Tauchen AR(1) times a lognormal
transitory shock — bounded support with thin tails. PSID has households at or
near **zero** income in every group. **Nothing the model can produce at any θ
reaches them.**

Households with at least one wave below the model's 1st-percentile income:

```
           floor     waves below    households with any wave below
comphs    15,000           14.5%                             33.3%
somehs    10,000           17.0%                             40.8%
compco    22,000            5.7%                             16.1%
```

and their in-box mass is much lower — median 0.855 vs 0.988 (comphs), 0.871 vs
0.992 (somehs), **0.498 vs 0.895 (compco)**, with correlations −0.34 to −0.50.
compco's apparent severity is not that its households are unusual but that its
floor is the highest (22,000), so more of its members fall under it.

**This is §7.1's diagnosis localised.** §7.1 observed the model lacks tail
events — "the excess is tail events the model has no mechanism for". This is
specifically the *left* tail: unemployment spells, disability, zero-income
years. A 3-state AR(1) cannot produce them, and no preference parameter
substitutes.

**Excluding the affected households changes nothing material:**

```
comphs  889 -> 720 (81% kept):  beta +0.0009  delta +0.0005  crra +0.0024
somehs  211 -> 125 (59% kept):  beta -0.0005  delta -0.0188  crra -0.0305
compco  527 -> 442 (84% kept):  beta -0.0100  delta +0.0002  crra +0.0120
```

`Var(true β)` remains **none detectable in every group on the clean subsample**,
so §11.3's central result does not rest on households the model cannot
represent. (`somehs`'s ρ heterogeneity does not survive the restriction —
0.272 → none — consistent with its borderline `P = 0.752`.)

**There is no fix within the current simulator**, and adding one means an
unemployment/disability process, which §"Not in scope" excluded for needing
external data. The actionable steps are the two already available: prefer the
conditioned model, which mitigates the symptom (compco in-box below 0.5 falls
17.1% → 2.1%), and report the below-floor share alongside any per-group
estimate so readers know what fraction of the sample the model is extrapolating
over.

### 12.5 Why income differs, and what could calibrate it

§12.4 established that the model cannot generate poor households. This is the
diagnosis of *why*, and an assessment of the fixes.

**It is not a level problem.** Medians match within 2–14%; the failure is
entirely in the shape of the residual distribution. Removing each source's own
age profile from log income, ages 25–59:

```
group    src       sd   IQR-sd   skew    kurt      p1      p5     p95     p99
comphs   model  0.510    0.536   0.00   -0.67   -1.03   -0.84    0.84    1.03
         PSID   0.901    0.774  -1.83    6.88   -3.53   -1.79    0.77    1.10
somehs   model  0.505    0.527  -0.00   -0.51   -1.08   -0.84    0.83    1.07
         PSID   0.961    0.798  -1.42    3.45   -3.68   -2.01    0.96    1.28
compco   model  0.381    0.390  -0.00   -0.57   -0.79   -0.63    0.63    0.79
         PSID   0.720    0.573  -1.70    7.80   -2.49   -1.21    0.85    1.31
```

Three things, and the third is the one that matters:

1. **The model is symmetric; PSID is strongly left-skewed** (0.00 vs −1.4 to
   −1.8). Tauchen discretises a Gaussian AR(1) and the transitory shock is
   lognormal, so nothing in the process *can* generate skew.
2. **The model is platykurtic; PSID is leptokurtic** (−0.5 to −0.7 vs +3.5 to
   +7.8). Negative excess kurtosis is the signature of a 3-state
   discretisation — mass on three points is flat-topped, not bell-shaped.
3. **The upside roughly matches; only the downside is missing.** Model p99
   +0.79 to +1.07 against PSID's +1.10 to +1.31 — comparable. Model p1 −0.79 to
   −1.08 against PSID's **−2.49 to −3.68** — three times deeper. Plus 1.1–3.3%
   of PSID waves at *exactly zero* income.

**The characters are opposite, not merely different in degree.** Defining a
disruption as a wave below half that household's *own* median:

```
group     P(disrupt)   median drop   P(recover next wave)
comphs          6.8%          0.27                  76.9%
somehs         10.0%          0.19                  77.6%
compco          5.0%          0.36                  71.8%

model's lowest Tauchen state   0.52–0.61x mean, P(stay) 0.73–0.83
```

PSID has **occasional, deep, transitory** collapses — to a fifth or a quarter of
usual income, with ~77% recovering by the next wave. The model has **permanent-
ish, moderate** variation — a worst state at 0.52× mean that households stay in
80% of the time. No amount of re-tuning a symmetric AR(1) turns one into the
other.

#### Calibration options, in order of what they would actually buy

| option | effect | cost |
|---|---|---|
| More Tauchen states | fixes kurtosis (flat-top → bell) but adds **no** skew and does not widen support | cheap, regeneration |
| Widen `AR1_GRID_SPAN` | deepens the left tail but **symmetrically**, overshooting an upside that already matches | cheap, regeneration |
| Re-estimate AR(1) on PSID | our own data, but a Gaussian AR(1) still cannot produce skew | moderate |
| **Add a disruption state** | **the mechanism the data show**; targets above are directly estimable | regeneration + revalidation |
| Restrict the sample | already validated: §12.4 shows β shifts ≤0.010 | **free** |

**The disruption state is the right fix and it is fully specified by the table
above**: probability ~5–10% per wave, income multiplier 0.19–0.36, persistence
~23%. That is a standard unemployment state in the lifecycle literature, and
PSID identifies all three parameters directly. Implementation touches
`discretize_transitory` and the solver's expectation step — the policy must
*anticipate* the risk or precautionary saving is wrong, which is the whole point.

**Two objections that have to be stated.**

1. **It deviates from the paper being replicated.** Laibson et al. deliberately
   remove unemployment — `unemprate` is a control stripped by the
   typical-household adjustment (§6). Adding it estimates a different model.
2. **It would break port fidelity.** The port currently reproduces their
   table-3 moments to 3.3% (§11.6), which is the evidence the implementation is
   correct. Changing the income process forfeits that check, and the model would
   need revalidating against a target it no longer shares with them.

**Recommendation: do not add it to replicate Laibson et al.** The honest framing
is that their model, faithfully ported, cannot represent a third of PSID
households because its income process has no left tail — and that this is a
finding about the model rather than a defect in the port. The cheap mitigation
(restricting to households the model can generate) is already validated and
moves nothing. A disruption state is the right next study, not a patch to this
one.
