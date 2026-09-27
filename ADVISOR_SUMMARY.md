# Household-level preference estimation from PSID — summary for review

*Prepared 2026-09-26. Detailed record: `RESULTS.md` (section numbers below refer to it).*

## What the project does

Laibson, Lee, Maxted, Repetto & Tobacman estimate **one** set of preference
parameters for the US population — present bias β, long-run discount factor δ,
and risk aversion ρ — by fitting their two-asset lifecycle model to SCF moments
(simulated method of moments).

This project estimates a **posterior distribution over (β, δ, ρ) for each
household** in a 7-wave PSID panel (2011–2023), using the same structural model.
The estimator is **neural posterior estimation** (NPE): simulate the model at
many parameter values, train a neural network to map a household's observed
trajectory (income, consumption, liquid wealth, illiquid wealth, age in each
wave) to a posterior over its parameters, then apply it to real households.

The data differ from Laibson et al.'s — PSID panel with consumption versus SCF
cross-section without it — so different estimates are expected and are not by
themselves evidence of error.

**Current model.** Transformer summary network (width 128, 3 layers, 64-number
summary) feeding a neural-spline normalising flow; δ estimated as
`log(1 − δ)` so the posterior cannot put mass at or above 1; five independently
trained networks averaged (ensemble). Trained on 65,536 simulated parameter draws
× 8 households each; results below are for high-school-complete households
(`comphs`, 889 in PSID) unless stated.

---

## 1. Training: is the model trained well?

Evaluated on simulated households the network never saw, where the true
parameters are known.

### Calibration — are the uncertainty statements honest?

Simulation-based calibration (SBC, Talts et al. 2018): for 335 fresh
simulations, check how often the true parameter falls inside the stated 90%
credible interval. A well-calibrated posterior covers 90% of the time.

```
                      90% coverage    rank-uniformity test (p)
beta                     0.896              0.21
delta                    0.842              0.007   <- under-covers
rho                      0.899              0.17
```

β and ρ are essentially exactly calibrated. **δ under-covers**: its intervals are
somewhat too narrow, and the rank test rejects uniformity. This is the cost of
the `log(1 − δ)` parameterisation (§20.3): it removes an artefact at δ = 1 on
real data but makes δ harder for the network to represent across the whole prior
range. The binomial standard error of these coverage figures is ±0.016.

### Choices behind the configuration (§19)

Each change was tested against a matched baseline, five seeds each:

```
                           held-out log q   mean |coverage - 0.90|
Phase 4 baseline                5.095              0.027
+ wider summary network         5.130              0.015
+ log(1 - delta) target           —                0.021   (log q not comparable)
```

- Enlarging the **flow** (the density part) made every metric worse — it
  over-fits. Enlarging the **summary network** improved every metric: the
  bottleneck was how much of each household's trajectory reached the flow.
- A learning curve (train on 1/4, 1/2, all of the data) shows each doubling of
  simulations buys half the gain of the previous one; another doubling (~10 GPU
  days) would add about as much as the summary-network change, which was free.
- Training converges in 64–85 epochs out of 200 (early stopping on a held-out
  validation set grouped by simulated household).

---

## 2. Forecasting simulated households: can it recover parameters?

Held-out simulated `comphs` households, true parameters known. Average over five
networks:

```
               correlation(true, estimate)   mean abs error   no-model error*
beta                    0.79                      0.094            0.175
delta                   0.79                      0.020            0.038
rho                     0.86                      0.411            1.125
```

\* Error from always guessing the middle of the prior range. The model removes
roughly half the error for β and δ and two thirds for ρ.

### One household in detail

![posterior for one simulated household](figures/10_simulated_household_posterior.png)

`figures/10_simulated_household_posterior.png` — one held-out simulated
household. Chosen by rule, not by eye: among 162 households whose true values
are away from the prior edges, the one with the **median** estimation error.
Shaded region = 68% credible region, dashed = 95%; red star = true parameters;
black cross = posterior mean (the point estimate).

```
          true    estimate    90% interval          covers truth
beta     0.551     0.548      [0.405, 0.700]         yes
delta    0.978     0.983      [0.964, 0.994]         yes
rho      3.496     2.835      [1.903, 3.723]         yes
```

What the figure shows: β and δ are pinned down well; **β and δ trade off**
against each other (the tilted region — more patience in one can substitute for
the other); ρ is the least precise, and its point estimate is off by 0.66 but
the truth is still inside the interval.

### Is the wide posterior the data's limit or the network's?

A correct posterior is only as narrow as the data allow. To check, the model is
given more data generated by the **same** θ: every simulated draw has 8
independent households sharing one θ, pooled 1, 2, 4, 8 at a time (60 draws):

```
 M households     posterior sd (relative to M=1)    median |error|
 sharing theta     beta   delta   rho                beta    delta    rho
      1            1.00   1.00   1.00               0.059   0.0171   0.241
      2            0.64   0.61   0.57               0.036   0.0069   0.113
      4            0.41   0.41   0.42               0.035   0.0059   0.069
      8            0.27   0.27   0.28               0.027   0.0053   0.052
```

With more data the posteriors tighten around the truth — ρ's error falls from
0.24 to 0.05 with 8 households. **The width in figure 10 is what one household's
seven noisy waves can support, not a network failure.** (Pooled coverage drifts
to 0.72–0.80 at M = 8, an expected side effect of multiplying approximate
posteriors.)

---

## 3. Application to PSID

### Headline (889 comphs households)

```
                          beta      delta      rho
median household         0.840     0.994     4.454
spread across households 0.099     0.023     0.889

Laibson et al. (population) 0.531  0.989     1.936
```

![PSID comphs against the literature](figures/08_adopted_literature_comparison.png)

`figures/08` — each household's posterior mean summarised as 68%/95% regions,
against published ranges (grey) and Laibson et al.'s estimate (star, tan box =
their 95% interval).

### One real household in detail

![posterior for one PSID household](figures/15_psid_household_posterior.png)

`figures/15_psid_household_posterior.png` — the counterpart of figure 10 on real
data. There is no true θ, so the panel shows the posterior, its mean (black
cross), Laibson et al.'s population estimate (red star) and the published ranges
(grey). Chosen by rule: the **typical** comphs household, whose posterior mean is
closest to the median across all 889.

The household, ages 35–47: income $49–89k, consumption $34–67k, card debt of
about $7k paid down to +$30k liquid savings by the last wave, illiquid wealth
$59–131k.

```
          estimate    90% interval        population median
beta       0.829     [0.523, 0.983]            0.840
delta      0.993     [0.976, 0.999]            0.994
rho        4.462     [4.378, 4.550]            4.454
```

How to read it:
- **β is barely pinned down** for a single real household: the interval spans
  most of the plausible range. This is the per-household face of the "no
  detectable β heterogeneity" result below.
- **δ is precise** and sits in the published range.
- **ρ's interval is very narrow**, but that is not a sign of trouble with real
  data. On simulated households the typical ρ interval is 1.48 wide, but for
  those whose true ρ exceeds 4 it is 0.29 — the model genuinely pins ρ down in
  that region, where behaviour changes sharply with ρ. Across PSID the median
  width is 0.22. The open question is not the width but whether ρ ≈ 4.5 is
  right, which is the misspecification issue below.

```
90% interval width (median)      beta     delta     rho
simulated, all households        0.357    0.064    1.48
simulated, true rho > 4            —        —      0.29
PSID                             0.457    0.018    0.22
```

### Comparison with the literature

Laibson et al.'s number is a representative-agent fit to population moments, so
the primary benchmark is the published range of estimates:

| | our median | published range | households inside |
|---|---|---|---|
| β | 0.840 | 0.66–0.94 (convex-time-budget meta-analyses; Imai et al. pooled 0.82) | 90.7% |
| δ | 0.994 | 0.95–1.00 (Carroll et al., heterogeneous discount factors) | 89.3% |
| ρ | 4.45 | 1–7 (Elminejad et al.: ≈1 in consumption studies, 2–7 in finance) | 99.8% |

- **β = 0.84 agrees with the experimental literature.** Laibson et al.'s 0.53 is
  below it.
- **δ agrees** with both Laibson et al. and Carroll et al.
- **ρ = 4.45 is the weak result.** It is only inside the range because finance
  estimates reach 7; consumption-based estimates are near 1. Separate checks
  (§15, §18) show the model cannot match PSID's wealth levels and its
  consumption comovement at the same time, and ρ absorbs that misfit — wealth
  alone implies ρ ≈ 4.6, consumption alone ρ ≈ 1.

### Do households actually differ?

The central question for a household-level method. Spread in household
estimates is not evidence by itself — noisy estimates of one common value also
spread. The test subtracts estimation noise:
`true variance = variance of household estimates − average posterior variance`,
bootstrapped over households. A negative value means **no detectable
differences**.

```
                 comphs (889)        somehs (211)        compco (527)
beta             none  (P=1.00)      none  (P=1.00)      none  (P=1.00)
delta            yes, sd 0.014       yes, sd 0.019       none  (P=0.99)
rho              not identified — see below
```

- **β: no detectable heterogeneity**, in every group and every model variant
  tried. Households' present bias cannot be told apart with this data.
- **δ: small, real differences** in comphs and somehs (1.4–1.9 percentage points
  in the annual discount factor), close to Carroll et al.'s ~2 points. **Not in
  compco**, whose households sit tightly at δ ≈ 0.998.
- **ρ: not identified.** Across three reasonable model variants its estimated
  heterogeneity is strongly positive, absent, and strongly positive again, with
  non-overlapping intervals. It tracks modelling choices, not households.

### Education groups

| | N | median β | median δ | median ρ | all three inside published ranges |
|---|---|---|---|---|---|
| comphs | 889 | 0.840 | 0.994 | 4.45 | 82.7% |
| somehs | 211 | 0.808 | 0.975 | 4.70 | 67.3% |
| compco | 527 | 0.797 | 0.998 | 4.17 | 78.2% |

Figures `11_somehs_full…`, `13_compco_full…` (current model) and
`12_somehs_arch…`, `14_compco_arch…` (without the `log(1 − δ)` change), each
against that group's earlier baseline. Only comphs is directly comparable to
Laibson et al., whose sample is high-school complete.

### Out-of-sample forecasts: do household parameters predict?

Each household's parameters are estimated from **waves 1–5 only**; the model is
then started at its observed wave-5 state and run forward to forecast waves 6–7
(200 simulations each, identical income shocks across forecast rules). Scored
with CRPS, which rewards both accuracy and honest spread; lower is better.

```
household θ minus population θ     diff      95% CI           households better
consumption   wave 6              -0.038  [-0.048, -0.028]         62%
              wave 7              -0.024  [-0.032, -0.016]         59%
liquid wealth wave 6              -0.096  [-0.238, +0.050]         48%
              wave 7              -0.004  [-0.119, +0.113]         47%
illiquid      wave 6              +0.064  [-0.008, +0.135]         47%
              wave 7              +0.124  [+0.045, +0.201]         48%
all targets, averaged             +0.004  [-0.039, +0.049]         48%
```

- **Overall, household-specific parameters forecast no better than one
  population parameter set.** The out-of-sample counterpart of "no detectable β
  heterogeneity".
- **Consumption is the exception:** household parameters forecast it 7–11%
  better, clearly outside noise. That is where household-level information shows.
- **"Nothing changes" beats every model forecast**, by a wide margin on liquid
  wealth (CRPS 1.38 against 2.25–2.79). Started from a real household's state,
  the model makes large, lumpy wealth moves within two years that real
  households do not — the model's known wealth-dynamics misfit, measured
  directly.
- **Laibson et al.'s parameters forecast wealth better than ours** (their lower
  ρ makes less aggressive wealth moves), though worse for consumption.

---

## Honest assessment

**Defensible now**
- The method works and is well calibrated on simulated data (β, ρ exactly; δ
  somewhat over-confident).
- Population-level β and δ are consistent with the published literature.
- A clear negative finding: **no detectable household heterogeneity in present
  bias.** This bears directly on the premise of estimating β per household.
- Small but real heterogeneity in δ for two of three education groups.

**Not yet defensible**
- That the model predicts an *individual* household's β or ρ. For δ, about 36%
  of the spread across comphs household estimates is real signal; for β, none.
  Out of sample, household parameters improve consumption forecasts only.
- That ρ ≈ 4.5 is a genuine risk-aversion estimate rather than the model
  absorbing its own misfit.

**Main limitation.** The structural model does not reproduce PSID's wealth and
consumption-comovement patterns at any parameter value (misses by ~2×). The
posteriors are "best fit within a misspecified model".

---

## In progress and planned

1. **Quasi-hyperbolic vs exponential discounting (planned, §21).** Train a second
   model with β fixed at 1 and compare out-of-sample forecasts, an amortised
   model-comparison classifier, and fit to PSID moments. Needs ~2.5 GPU-days of
   new simulation. Preliminary: 95% of households' β intervals do not exclude 1.

## Questions for discussion

1. Given no detectable β heterogeneity, should the paper's contribution be
   framed around the **population** estimate and the δ distribution, with the β
   null as a result in its own right?
2. How to treat ρ: report with the misspecification caveat, or constrain it
   (e.g., fix ρ from the consumption literature) and re-estimate β and δ?
3. Household parameters help forecast consumption but not wealth, and the model
   loses to "nothing changes" on wealth. Is fixing the wealth dynamics (lumpy
   illiquid adjustment, over-accumulation) the priority before any further
   estimation work?
4. Worth pursuing model fixes (income left tail, wealth moments) before the
   exponential-discounting comparison, or after?
