# Conclusion direction

*Decided by the user on 2026-11-01, while RESULTS 62 (the noise-free and noise-unknown arms) was running. This file is the guideline for every plan from here to the end of the project.*

> "make the new more input waves model and show whether it is able to estimate the parameter or not. & show the result from PSID and include the limitation that we only have 7 waves. & add the population estimation result and show we got the better result even it has short waves. To sum up, it is the right way to develop and with more waves of data, it might predict much more precisely."

## The story the project ends on

1. **The method.** A neural posterior estimator gives each household its own posterior for β, δ, ρ and R_γ from its panel data. It is trained on Laibson et al.'s lifecycle model, rebuilt and audited against their code, and is calibrated on simulated households.
2. **PSID, with seven waves.** It is applied to PSID couples, who are observed in seven biennial waves (2011–2023). That is the most PSID offers with credit-card debt measured on its own (`PSID_DATA.md` §3). With seven waves, one household's data narrow the parameters but do not pin them down. That is the main limitation, and it is stated as such.
3. **The population, already better than the benchmark.** Even with seven waves, the population estimate from 409 couples is more precise than Laibson et al.'s method-of-simulated-moments estimate from 5,710 SCF households.
4. **More waves.** Simulated households show how precision grows with the number of waves. The approach is the right way to develop, and longer panels would let it estimate much more precisely.

## Evidence in hand

| claim | evidence | where |
|---|---|---|
| calibrated on simulated households | on the wider box (β 0.10–1), held-out 90% coverage 0.903–0.913; SBC 0.889–0.908 (old box 0.899–0.919 and 0.895–0.904) | RESULTS 63.3 (56.2) |
| what seven waves give one household | on the wider box, share of prior variance removed: β 0.52, δ 0.46, ρ 0.53, R_γ 0.46 (not comparable with the old box's 0.40, 0.54, 0.61, 0.53: β's prior is wider and the held-out set includes the strip). PSID couples' median posterior sd: β 0.183, δ 0.0123, ρ 0.93, R_γ 0.0102 | RESULTS 63.3, 63.5 (56.2–56.3) |
| one couple cannot be ranked against another on β or ρ | on the wider box, one couple's 90% interval is 1.07 (β) and 1.44 (ρ) times the spread of couples' estimates; 21% and 8% of pairs can be ordered | RESULTS 1 (60's method) |
| couples differ | their estimates spread 1.4–2.0 times as much as households that share θ; clearest in β (and R_γ), which exceed both nulls | RESULTS 1 (60's method) |
| population, under v2's noise model, box extended to β 0.10 | representative θ: β 0.437 (se 0.027), δ 0.9928 (0.0009), ρ 1.58 (0.085), R_γ 1.042 (0.0014); misfit ≤ 0.02 PSID se, no bootstrap draw beyond the box. Standard errors (Jacobian at the estimate) 1.9–2.4 times smaller than their sampling-only ones, 4.3–5.9 times smaller than their reported ones. Within their reported errors (\|z\| ≤ 0.80); 1.6–1.7 combined se from their sampling-only ones. The old box's 0.531 is superseded: its floor held couples' estimates up | RESULTS 63.10 (63.6, 61.4) |
| more waves were already sharper in an earlier model | three-parameter model (before card types and R_γ): 10 waves gave held-out log q 5.446 against 7 waves' 5.007, both calibrated | `PSID_DATA.md` §3 |
| households simulated at their θ | under v2, individual 90% intervals contain their θ for 99% (β), 91% (δ), 96% (ρ) and 86% (R_γ). The average posterior mean lands further away the noisier the data; without noise it is within 0.013 of their β (ρ stays 0.5 high: its long upper tail) | RESULTS 62.6 |
| what noise PSID couples have | a network that estimates each household's noise reads couples, on average, as noisy as the gross arm or noisier, far above v2's training noise; one couple at a time, a third or fewer exclude v2's range; income reads noisier than the box allows (likely the couples' income process) | RESULTS 62.3 |
| the population β under PSID-level noise | with the Jacobian recomputed at each point, typical and gross converge: β 0.132 (se 0.019) and 0.157 (0.011), δ 0.9991 / 0.9986, ρ 1.95 / 1.11, R_γ 1.038 / 1.036; z about −3.3 against their reported β error. Noise-unknown stops at β 0.143 (θ misfit ≤ 1.9 se), its income noise unmatched at the box top. Typical's 90% interval reaches the floor of 0.10 (7% of the bootstrap below it). The noise model moves β by about 0.3, 10–11 of v2's standard errors | RESULTS 63.10 (63.7, 62.4) |
| one household's β below 0.3 | not resolved with seven waves: simulated households at β 0.1–0.2 average a posterior mean of 0.35; no simulated or PSID household has its 95th percentile below 0.3 | RESULTS 63.4–63.5 |
| a denser training set | doubling the data (20k to 40k draws) leaves the share of prior variance removed unchanged (±0.011) and adds +0.19 log q; another doubling would cost about 6 GPU-days for about +0.1 log q | RESULTS 63.8 |

## Still to do, in order

1. **RESULTS 62 (done, 2026-11-01).**
   - The noise-free arm fails on noisy data and reads PSID couples' ρ at the floor.
   - The noise-unknown arm is calibrated on all eight targets; it reads PSID couples as noisy as gross or more, and under that noise the population β goes to the floor (no fit inside the box).
   - **Decided by the user (2026-11-02):** "let's generate data including beta below 0.3 to try the models. I think v2 looks best until now." v2 stays the reference noise model, and the β floor is extended by a strip (below).
2. **RESULTS 63 (done, 2026-11-04): the β strip below 0.3.**
   - New draws on β ∈ [0.10, 0.30), trained with v2's main run and R_γ strip; the box is β ∈ [0.10, 1] (`scripts/run_beta_strip.sh`). Every arm is calibrated on it.
   - Under v2 the population fit converges inside the box at β 0.437 (se 0.027 with the Jacobian at the estimate, RESULTS 63.10), replacing 0.531.
   - One household's β below 0.3 is not resolved with seven waves.
   - No denser dataset is needed (the learning curve).
   - **The Jacobian recomputed at each point (RESULTS 63.10, 2026-11-04, user's request):**
     - under PSID-level noise the population β is 0.13–0.16: typical and gross converge, and noise-unknown stops there with income noise unmatched;
     - v2's 0.437 holds.
   - Typical's 90% interval reaches the floor of 0.10 (7% of its bootstrap below it). **Decided by the user (2026-11-05):** "keep the floor at 0.10".
   - **§1 switched to v2 on the wider box (user, 2026-11-05):** the household tables, calibration, heterogeneity and resolution (figure 103) are now `optionA2v2_b010`'s.
3. **The waves experiment (simulated households).**
   - **Models.** Train with 7 (existing), 10, 14 and about 20 waves, under v2's noise model on the box with the β strip, with everything else as v2. If the longer windows underfit (training loss still falling while held-out loss stalls), add one wider-embedder arm. That costs GPU-hours, not a regeneration. The training set is dense enough as it is (RESULTS 63.8).
   - **No regeneration needed.** The stored panels cover ages 20–90 for every household, so longer windows can be cut from the existing shards. `optionA.py train --n_waves` exists. The window start range must leave room: start + 2 × (waves − 1) ≤ 90.
   - **Measures, per number of waves:**
     - coverage;
     - share of prior variance removed and posterior sd;
     - how many household pairs can be ordered;
     - one household's interval against the spread across households;
     - the population standard error at a fixed number of households;
     - whether one household's β below 0.3 becomes resolvable (RESULTS 63.4: with seven waves it is not; behaviour at the credit limit may keep it so).
   - **Report whatever comes out.** If β's precision stops improving with more waves, that is a property of the model, not of the panel length, and the conclusion must say so.
4. **PSID, framed by the experiment.** Show couples' posteriors as they are, and mark where seven waves sit on the precision-against-waves curve.
5. **The write-up.** A final headline (RESULTS §1 and a summary) built on points 1–4 of the story.

## Caveats each claim must carry

- **"Better than Laibson et al. at the population level":**
  - it means more precise, and the location depends on the noise model (β 0.437 under v2's, 0.13–0.16 under PSID-level noise, RESULTS 63.10), so the claim is made under the reference noise model, v2's (the user's choice, 2026-11-02), on the box extended to β 0.10, with the PSID-level-noise arms reported beside it;
  - on that box v2's β is 0.437, not their 0.53; the agreement the old box showed was partly its floor's work (RESULTS 63.6);
  - our standard errors hold the first stage (income process, demographics, credit limits) fixed, so the like-for-like comparison is their sampling-only error;
  - both estimates are one θ for everyone, though couples differ;
  - both models share their income process, whose transitory variance absorbs PSID's income measurement error;
  - their model fails its own overidentification test (p = 7×10⁻¹⁵).
- **"More waves would predict much more precisely":**
  - a hypothesis until the waves experiment measures it;
  - even then it is measured on simulated households under our model;
  - real longer panels also bring attrition, changes of household head and more measurement error.
- **"Only seven waves" is PSID's limit, not our choice** (`PSID_DATA.md` §3, verified against PSID's codebooks):
  - PSID is biennial from 1999, so 13 waves exist in calendar terms;
  - the model needs credit-card debt on its own, since liquid wealth goes negative only through card debt;
  - before 2011, PSID reports one lumped total of "credit card charges, student loans, medical or legal bills, or loans from relatives", from which card debt cannot be recovered;
  - from 2011 card debt is a separate item, so 2011–2023 (seven biennial waves) is the ceiling with a clean debt measure;
  - ten waves over 2005–2023 were rejected: the first three would mismeasure card debt, in the borrow-and-repay years that identify β;
  - more PSID waves arrive one at a time, every two years (2025, 2027, …).

## How this guides the next plans

- **Every next step should serve the story above.** After RESULTS 63, the waves experiment comes first.
- **The parked model changes become limitations and future work** unless they bear on the conclusion:
  - durables and housing in the illiquid asset;
  - the couples' income process;
  - a lower β floor (taken up: RESULTS 63).
- **The β floor is the exception.** If the reference noise model puts the population β at the floor (as gross and typical do), extending the floor becomes necessary for the population claim. RESULTS 62 triggered it, and RESULTS 63 extends it to 0.10.
- **The standing rules hold:**
  - no cherry-picking;
  - every estimation result ships a 68/95% contour figure;
  - RESULTS cites committed scripts only;
  - PSID microdata are never committed;
  - long GPU jobs need the user's decision.
