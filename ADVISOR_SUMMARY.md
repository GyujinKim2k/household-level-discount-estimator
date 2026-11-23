# Household-level preference estimation from PSID — summary for review

*Updated 2026-11-23 (§73's results). This replaces the summary of 2026-09-28, whose headline (β 0.85, ρ 4.5, "no detectable β heterogeneity") has since been superseded or withdrawn; see "How we got here". Section numbers (§) refer to `RESULTS.md`, the full record. The plan for the end of the project is `CONCLUSION_DIRECTION.md`.*

---

## The short story

1. **The method.** A neural posterior estimator gives each household its own posterior for present bias β, the long-run discount factor δ, risk aversion ρ and the illiquid asset's return R_γ, from its panel data.
   - It is trained on Laibson, Lee, Maxted, Repetto & Tobacman's two-asset lifecycle model, rebuilt in Python and audited line by line against their own code.
   - It is calibrated on simulated households: its 90% intervals contain the truth about 90% of the time.
2. **PSID, with seven waves.** Applied to 409 PSID couples (889 households in all), each observed in seven biennial waves from 2011 to 2023, one household's data narrow the parameters but do not pin them down.
   - One couple's 90% interval for β is about as wide as the spread of all couples' estimates.
   - Seven waves is PSID's ceiling: card debt is recorded on its own only from 2011. That is the main limitation.
3. **The population, more precise than the benchmark.** Couples' representative θ (one θ for the population, the same kind of object as Laibson et al.'s estimate):

   | | β | δ | ρ |
   |---|---|---|---|
   | ours | 0.428 (0.027) | 0.9926 (0.0009) | 1.75 (0.093) |
   | Laibson et al. | 0.531 (0.114) | 0.9891 (0.0051) | 1.94 (0.435) |

   - Ours lies within their reported errors.
   - Our standard errors are 1.9–2.2 times smaller than their sampling-only ones, from 409 couples against their 5,710 SCF households.
4. **More waves.** On simulated households, precision keeps growing with the number of waves (2.2–2.4 times the information at 20 biennial waves, §71). The model is calibrated on annual panels too: seven annual waves tell one household about as much as PSID's seven biennial ones, and in the same years annual data carry about twice the information on β. Longer panels would estimate much more precisely.

**The main caveat** is the noise model. How much measurement error PSID's data carry moves the population β by 0.24–0.30, about ten of our standard errors. Every population claim is therefore made under our reference noise model, with the alternatives reported beside it.

---

## 1. What the project does

- **Benchmark.** Laibson et al. estimate one (β, δ, ρ) for the US population by fitting their two-asset lifecycle model to SCF moments, by the method of simulated moments. Their model has liquid wealth that can go negative through credit cards, and an illiquid asset paying a return R_γ, which they calibrate at 1.05.
- **Here.** We estimate a posterior for each household from its own trajectory, with the same structural model:
  - **simulate:** solve and simulate the model at about 40,000 parameter draws, 16 households each;
  - **train:** a transformer summary network feeds a neural-spline normalising flow, which maps a household's seven waves (income, consumption, liquid and illiquid wealth, age) to a posterior over (β, δ, ρ, R_γ);
  - **ensemble:** five independently trained networks are averaged;
  - **apply:** the networks are run on PSID households.
- **Data.**
  - PSID 2011–2023, high-school-complete households, as in Laibson et al.'s comparison group: 889 households, 409 of them couples (§2).
  - Consumption is PSID's, with rental equivalence for owners. That was the largest single data correction: owners' housing was otherwise zero (§5).
- **Observation noise.** Simulated data are noised to look like survey data before training. The amount comes from PSID couples, and it matters a great deal (§3.5 below).
- **R_γ is a nuisance parameter.** It is estimated so that the model can fit; it is not a result (decided by the user, 2026-11-14).

---

## 2. How we got here

| when | stage | what it taught us |
|---|---|---|
| Sep 2026 | First PSID headline (§14, §20, §29): β 0.85, ρ 4.5, "no β heterogeneity" | ρ 4.5 was the model absorbing a misfit: wealth levels implied ρ ≈ 4.6, consumption ρ ≈ 1 (§15). Out of sample, household parameters forecast consumption better than one population θ, and nothing else (§22) |
| Sep–Oct | Bugs found and fixed | Every solve used one credit limit (§25); the solver lost float64 resolution at high ρ (§40), the region of the old ρ 4.5 |
| Oct | Model rebuilt step by step | Card-access types, which the network marginalises over (§26, §36); R_γ estimated as a fourth parameter (§32); logit β (§39); starting wealth from a pool of PSID couples (§43); observation noise calibrated from PSID, with which couples' ρ reads about 2 instead of sitting at its floor (§43.8–43.9). Out of sample, a couple's own θ forecasts its consumption better than one population θ, and every model rule now beats "nothing changes" on wealth (§43.12) |
| Oct (dropped) | Two branches tried and stopped by the user | A hierarchical population layer (§45–47: no meaningful result), and an illiquid valuation shock (§44: built, then put on hold) |
| Oct 15–24 | Simulation audit against Laibson et al.'s own code (§54) | Three solvers compared: their MATLAB run unmodified in Octave, an independent reference written from the equations, and our production port. They agree. Inherited and own flaws were measured and fixed, and all data were regenerated ("v2", §55–56) |
| Oct 30–31 | PSID's noise, examined | PSID's illiquid wealth jitters where the model's moves smoothly (§57). The jitter leaves no trace in the rest of the budget, so it is reporting error, not behaviour (§59). Networks trained with too little noise fail (§58) |
| Oct 31 | Household resolution, and the like-for-like comparison | The old heterogeneity test was invalid; couples do differ, but one couple's β interval is about as wide as couples' spread (§60). Laibson et al.'s estimate is set against a representative θ, not against median household estimates (§61) |
| Nov 1–5 | Noise arms, and the β floor | A noise-free network breaks on noisy data, and PSID-level noise drives the population β to the old floor of 0.3 (§62). The prior box is extended to β 0.10 (§63); the floor stays at 0.10 (user) |
| Nov 1 | Conclusion direction set by the user | The four-point story above (`CONCLUSION_DIRECTION.md`) |
| Nov 6–8 | Waves experiment on simulated households | 7 to 20 biennial waves (§64), annual waves (§65), and noise-free arms (§66) |
| Nov 9–14 | Training improvements, and a new reference | Noisy copies (§67), a new noise draw every epoch (§68) and reduce-on-plateau (§69) raise held-out log q. The best network, v2_fresh_rop, becomes the reference (user, §70) |
| Nov 15–17 | Waves experiment with the current model (§71) | The current training beats the old at every length; precision grows with waves at both spacings; per household, the number of waves matters more than their spacing |
| Nov 18–20 | A lower learning-rate floor on four models (§72) | A floor of 1e-7 instead of 1e-5 adds nothing; annual against biennial in the same years is unchanged |
| Nov 21–22 | A longer patience at that floor (§73) | A stop after 60 passes instead of 20 adds nothing: 16 of 20 networks keep the same weights; annual against biennial unchanged |

---

## 3. Current results (reference network: v2_fresh_rop, §1, §69–70)

### 3.1 Calibration on simulated households

- **Coverage.** Held-out 90% coverage is 0.898–0.905 and SBC coverage 0.882–0.901, per parameter.
- **R_γ's rank test fails** (KS p 0.004), as it does for every noisy network on this window. Its coverage passes, and the user accepted the failure, since R_γ is a nuisance parameter.
- **Edges.** Coverage by true value need not be 0.90: near the edges of the prior box the prior pulls estimates inward.

### 3.2 What seven waves tell about one household

| | β | δ | ρ |
|---|---|---|---|
| share of prior variance removed (simulated, held-out) | 0.53 | 0.49 | 0.55 |
| corr(posterior mean, truth), simulated | 0.73 | 0.70 | 0.75 |
| PSID couples' median posterior sd | 0.178 | 0.0105 | 0.94 |

- **β and ρ are confounded:** both raise saving. A β estimate correlates 0.73 with the true β and 0.39 with the true ρ.
- **One household's β interval is 0.5–0.6 wide anywhere in the box.** β below 0.3 is resolved as well as any band 0.2 wide (§64.12).
- **Figures:** a simulated household with its truth (`figures/147_simulated_household_lr_plateau_v2_fresh_rop.png`), and the PSID couple nearest couples' median (`figures/148_psid_household_lr_plateau_v2_fresh_rop.png`), both with 68/95% contours. Clean versions with β, δ and ρ only: `figures/181_advisor_simulated_household.png` and `figures/180_advisor_psid_couple.png`. The same simulated household under 20 annual waves: `figures/183_advisor_simulated_household_annual20.png`, and both designs overlaid: `figures/184_advisor_simulated_household_biennial_vs_annual.png`.

### 3.3 PSID couples (409)

| | β | δ | ρ | R_γ |
|---|---|---|---|---|
| median of posterior means | 0.473 | 0.9889 | 2.69 | 1.057 |
| sd across couples | 0.181 | 0.023 | 0.68 | 0.013 |
| share whose 90% interval covers Laibson et al.'s value | 0.86 | 0.71 | 0.91 | 0.69 |

- **Household by household, Laibson et al.'s estimate is not excluded.**
- **The medians cannot be set against their point.** Posterior means are pulled toward the prior's centre: households simulated at exactly their θ come out at β 0.553 and ρ 2.86. Hence the representative θ in §3.5.

### 3.4 Do couples differ?

- **Yes.** Couples' estimates spread 1.4–2.0 times as much as those of simulated households that share one θ. This is clearest in β and R_γ, which exceed both null benchmarks (§60, §70.4).
  - For δ and ρ, couples do not exceed the stricter null.
  - For all 889 households, δ and ρ just exceed it under the reference network but not under the previous one: unsettled.
- **But one couple cannot be ranked against another** on β or ρ. Its 90% interval is 0.97 times the spread of couples' estimates in β and 1.38 times in ρ, and only 23% (β) and 10% (ρ) of couple pairs can be ordered with 90% confidence. See `figures/149_household_resolution_v2_fresh_rop.png`.
- *Withdrawn:* the earlier "no detectable β heterogeneity". It rested on a test that subtracts what it should add (§60.4).

### 3.5 The population against Laibson et al.

The representative θ is the single θ whose simulated households the network reads as it reads PSID couples (§61). Its standard errors come from a bootstrap over couples, with the Jacobian at the estimate.

| | β | δ | ρ | R_γ |
|---|---|---|---|---|
| Laibson et al. (reported se; sampling-only se) | 0.531 (0.114; 0.050) | 0.9891 (0.0051; 0.0020) | 1.94 (0.435; 0.204) | 1.05, calibrated |
| **PSID couples, reference noise model** | **0.428 (0.027)** | **0.9926 (0.0009)** | **1.75 (0.093)** | **1.046 (0.0015)** |
| PSID couples, the noise model of "psid noise methodology.md" (written from the literature) | 0.191 (0.036) | 0.9984 | 2.34 | 1.050 |
| PSID couples, PSID-level noise (typical / gross) | 0.132 / 0.157 | 0.9991 / 0.9986 | 1.95 / 1.11 | 1.038 / 1.036 |

- **Under the reference noise model:**
  - each estimate is within their reported error (|z| ≤ 0.87);
  - against their sampling-only error, β is 1.8 combined se below theirs, δ 1.6 above and ρ 0.8 below;
  - our standard errors are 1.9–2.2 times smaller than their sampling-only ones, and 4.2–5.5 times smaller than their reported ones;
  - the fit misses PSID couples' statistics by at most 0.15 standard errors.
- **The noise model is the main uncertainty.** It moves β by 0.24–0.30. Five ensembles trained for the same noise and data give β 0.384–0.453 (sd 0.027, about one se), so the network's own variability is small beside the noise model's effect.
- **The noise model is needed.** With no noise, the network reads PSID's consumption noise as ρ near its floor (median 0.76), and no noise-free population fits couples (§66).
- **The old headline is reversed.** β 0.85 sat inside the experimental meta-analyses' band [0.66, 0.94]. Now 20% of couples' estimates do.
- **Figure:** `figures/146_laibson_comparison_lr_plateau.png`, their point and covariance against our representative θ under every noise model, 68/95%. A clean version for the reference network alone, β, δ and ρ: `figures/179_advisor_laibson_comparison.png`.

### 3.6 More waves and annual waves (simulated households, the current model and training, §71)

Eight arms of five networks: biennial 7, 10, 14 and 20 waves, annual 7, 10, 14 and 20, every year read inside ages 24–64. All are calibrated for β, δ and ρ (90% coverage 0.890–0.918).

- **More biennial waves** (7 to 20): one household's data carry 2.2–2.4 times the information, and the curve has not levelled off.
  - The share of prior variance removed for β goes 0.53 → 0.73, and β's correlation with the truth 0.73 → 0.85.
  - Household pairs ordered on β with 90% confidence go from 29% to 47%, and 95–97% of those orderings are right.
  - β below 0.3 gains less (its interval narrows 17% by 20 waves, against 36% at the top): those households spend most years at the borrowing limit.
- **Annual waves** (what real households often have):
  - at the same number of waves, spacing matters little for one household: seven annual waves (8 years) tell about as much as PSID's seven biennial waves (14 years);
  - in about the same years, every year instead of every other carries about twice the information on β (1.95–1.99; 1.94–1.98 with a lower learning-rate floor, §72; 1.92–1.99 with a longer patience too, §73). That is an upper bound, because the simulated noise is independent every wave (1.53 without noise, §66);
  - for the population, years matter more than frequency for β and δ: at the same count, biennial's longer span gives the smaller se.
- **The training recipe matters at every length:** the current one beats the previous one by +0.18 to +0.40 held-out log q (more for longer windows) and adds 11–17% information on β; the population se does not move. Lowering its learning-rate floor from 1e-5 to 1e-7 adds nothing (§72), nor does a longer patience at that floor (§73).
- **Measurement matters as much as length:** without noise, seven waves tell as much as 20 noisy ones (§66).
- **The population:** households sharing one θ get standard errors at 20 biennial waves of 0.26–0.48 times seven waves'. For PSID couples, β's se would fall from 0.027 to between 0.007 and 0.022, depending on how much of it is couples differing, which more waves do not narrow.
- **Figures:** `figures/150_waves_default_precision.png` (precision against waves and years read), `figures/159_waves_default_one_household.png` (one household, biennial against annual, 68/95%), `figures/160_waves_default_population.png` (the population, 68/95%). Seven biennial against twenty annual waves in one figure: `figures/182_advisor_biennial7_vs_annual20.png`.

---

## 4. Last run: a longer patience at the low floor (§73, done 2026-11-22)

- **The user's request** (2026-11-21): "try longer patience with the low floor".
- **Why:** §72's lower floor (1e-7) added nothing, but its stop came close: some networks went up to 17 passes without a gain before their best, against a stop after 20.
- **Design:** §72's four models (7 biennial waves against 14 annual, 10 biennial against 20 annual), retrained with a stop after 60 passes instead of 20 and nothing else changed. They were compared with §72's networks on identical held-out windows, and with §71's default beside that.
- **Results:**
  - all four calibrated for β, δ and ρ (0.891–0.913);
  - the longer stop buys nothing. In 16 of the 20 networks a stop after 20 would have kept the very same weights, and the other four gained at most 0.0003 in validation loss. It costs 36% more training;
  - by the 95% rule set before the run, one arm gains (+0.024) and one loses (−0.030). Neither can come from the patience, since those networks kept what a stop after 20 would have kept. Both are the spread between training runs. The four-test intervals and the member check find nothing;
  - information, pairs ordered and the population se are within 3.2%;
  - in about the same years annual waves still carry about twice the information on β (1.92–1.99 under the three settings), an upper bound.
- **Consequence:** the training recipe stays as it is (floor 1e-5, patience 20), and no claim changes.
- **New, on method:** the paired household interval holds the networks fixed, so it leaves out the spread between training runs. Between ensembles, differences under about 0.03 need the member-level check as well. The recipe's gain over the previous one (+0.18 to +0.40) is far beyond that.
- **Before it, §72** (done 2026-11-20): a floor of 1e-7 instead of 1e-5 added nothing in any arm (−0.024 to −0.001, every interval spanning 0).

---

## 5. Honest assessment

**Defensible now:**
- The method is calibrated on simulated households and audited against the benchmark's own code.
- The population estimate is more precise than Laibson et al.'s from 409 couples, and consistent with it within their reported error, under the reference noise model.
- Couples differ, in β at least, by more than estimation noise.
- Longer and annual panels would sharpen household estimates substantially (simulated).

**Not defensible, or conditional:**
- **Ranking individual households** on β or ρ at seven waves.
- **The location of the population β** without naming the noise model: 0.43 under ours, 0.13–0.19 under PSID-level or literature-based noise.
- **"One θ for everyone".** Both estimates assume it, though couples differ.
- **Like-for-like errors.** Our standard errors hold the first stage (income process, demographics, credit limits) fixed, so the fair comparison is their sampling-only error.
- **Specification.**
  - The model's wealth dynamics are smoother than PSID's (§24, §57).
  - Laibson et al.'s own model fails its overidentification test (p = 7×10⁻¹⁵).
  - The latest out-of-sample forecast test (§43.12, October) predates the audit and the current model and has not been rerun. There, a couple's own parameters forecast its consumption better than one population θ (about 60% of couples better), but not its wealth.

---

## 6. What remains

1. **The conclusion's point 4:** PSID couples shown where seven biennial waves, and annual panels, sit on the precision curve (§71 supplies the curve).
2. **The write-up:** a final headline built on the four-point story.
3. **Open proposals awaiting a decision:**
   - a simulated population whose θ differ as couples' do, to measure how much of the population se is couples differing (a few GPU-hours);
   - seeding the population bootstrap per network (it changes earlier sections' se slightly).
4. **Parked as future work:**
   - durables and housing in the illiquid asset;
   - a couples-specific income process;
   - a single-adult household type;
   - a finer solution grid.

## Questions for discussion

1. **The noise model.** The population β moves by about ten standard errors between our noise model and PSID-level ones. Is presenting the reference model's estimate with the alternatives beside it the right framing, or should the paper lead with the range?
2. **Annual data.** If the method is to be applied to annual panels, which datasets carry consumption, liquid wealth with card debt, and illiquid wealth every year? The simulated annual gain is an upper bound until tested on real data.
3. **The household-level contribution.** With one couple's β interval about as wide as couples' spread, should the household-level claim rest on heterogeneity existing and on the precision path with more data, rather than on individual estimates?
4. **Misspecification.** Is the wealth-dynamics misfit a limitation to state, or something to fix before the write-up?
