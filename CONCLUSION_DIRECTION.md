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
| calibrated on simulated households | held-out 90% coverage 0.899–0.919; SBC 0.895–0.904 | RESULTS 56.2 |
| what seven waves give one household | share of prior variance removed: β 0.40, δ 0.54, ρ 0.61, R_γ 0.53. PSID couples' median posterior sd: β 0.164, δ 0.0121, ρ 0.90, R_γ 0.0095 | RESULTS 56.2–56.3 |
| one couple cannot be ranked against another on β or ρ | one couple's 90% interval is 1.3–1.4 times the spread of couples' estimates; 10–13% of pairs can be ordered | RESULTS 60 |
| couples differ | their estimates spread 1.5–2 times as much as households that share θ; clearest in β | RESULTS 60 |
| population, under v2's noise model | representative θ: β 0.531 (se 0.022), δ 0.9901 (0.0009), ρ 1.44 (0.098), R_γ 1.043 (0.0016). Standard errors 2.1–2.3 times smaller than their sampling-only ones, 4.4–5.9 times smaller than their reported ones | RESULTS 61.4 |
| more waves were already sharper in an earlier model | three-parameter model (before card types and R_γ): 10 waves gave held-out log q 5.446 against 7 waves' 5.007, both calibrated | `PSID_DATA.md` §3 |
| households simulated at their θ | individual 90% intervals contain their θ for 99% (β), 91% (δ), 96% (ρ) and 86% (R_γ). The average posterior mean does not land on it (shrinkage, ρ's long upper tail) | to be recorded in RESULTS 62 |

## Still to do, in order

1. **RESULTS 62 (running).**
   - The noise-free arm.
   - The noise-unknown arm, in which the network estimates each household's measurement-error level alongside θ.
   - It decides which noise model the final population estimate uses. That choice is open: the population β is 0.53 under v2, 0.335 under gross, and below the 0.3 floor under typical (RESULTS 61).
2. **The waves experiment (simulated households).**
   - **Models.** Train with 7 (existing), 10, 14 and about 20 waves, under the noise model chosen after RESULTS 62, with everything else as v2.
   - **No regeneration needed.** The stored panels cover ages 20–90 for every household, so longer windows can be cut from the existing shards. `optionA.py train --n_waves` exists. The window start range must leave room: start + 2 × (waves − 1) ≤ 90.
   - **Measures, per number of waves:**
     - coverage;
     - share of prior variance removed and posterior sd;
     - how many household pairs can be ordered;
     - one household's interval against the spread across households;
     - the population standard error at a fixed number of households.
   - **Report whatever comes out.** If β's precision stops improving with more waves, that is a property of the model, not of the panel length, and the conclusion must say so.
3. **PSID, framed by the experiment.** Show couples' posteriors as they are, and mark where seven waves sit on the precision-against-waves curve.
4. **The write-up.** A final headline (RESULTS §1 and a summary) built on points 1–4 of the story.

## Caveats each claim must carry

- **"Better than Laibson et al. at the population level":**
  - it means more precise, and the location depends on the noise model (above), so the claim is made under the reference noise model chosen after RESULTS 62;
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

- **Every next step should serve the story above.** After RESULTS 62, the waves experiment comes first.
- **The parked model changes become limitations and future work** unless they bear on the conclusion:
  - durables and housing in the illiquid asset;
  - the couples' income process;
  - a lower β floor.
- **The β floor is the exception.** If the reference noise model puts the population β at the floor (as gross and typical do), extending the floor becomes necessary for the population claim.
- **The standing rules hold:**
  - no cherry-picking;
  - every estimation result ships a 68/95% contour figure;
  - RESULTS cites committed scripts only;
  - PSID microdata are never committed;
  - long GPU jobs need the user's decision.
