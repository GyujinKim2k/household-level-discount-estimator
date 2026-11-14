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
| calibrated on simulated households | the reference, v2_fresh_rop (since 2026-11-14): held-out 90% coverage 0.898–0.905; SBC 0.882–0.901. R_γ's SBC rank test fails (KS p 0.004), as for every noisy network on its window; the open point of RESULTS 66.3 (v2_b010: 0.903–0.913 and 0.889–0.908, KS p 0.14; old box 0.899–0.919 and 0.895–0.904) | RESULTS 69.3, 70.3 (63.3, 56.2) |
| what seven waves give one household | share of prior variance removed: β 0.53, δ 0.49, ρ 0.55, R_γ 0.48 (v2_b010 0.52, 0.46, 0.53, 0.46; not comparable with the old box's 0.40, 0.54, 0.61, 0.53: β's prior is wider and the held-out set includes the strip). PSID couples' median posterior sd: β 0.178, δ 0.0105, ρ 0.94, R_γ 0.0095 (v2_b010 0.183, 0.0123, 0.93, 0.0102) | RESULTS 69.6, 70.2 (63.3, 63.5, 56.2–56.3) |
| one couple cannot be ranked against another on β or ρ | one couple's 90% interval is about as wide as the spread of couples' estimates in β (0.97 times) and wider in ρ (1.38); 23% and 10% of pairs can be ordered (v2_b010 1.07 and 1.44; 21% and 8%) | RESULTS 1, 70.4 (60's method) |
| couples differ | their estimates spread 1.4–2.0 times as much as households that share θ; clearest in β (and R_γ), which exceed both nulls. For all 889 (not couples), δ and ρ just exceed the second null under v2_fresh_rop but not under v2_b010: unsettled | RESULTS 1, 70.4 (60's method) |
| population, under v2's noise model, box extended to β 0.10 | representative θ under the reference, v2_fresh_rop: β 0.428 (se 0.027), δ 0.9926 (0.0009), ρ 1.75 (0.093), R_γ 1.046 (0.0015); misfit ≤ 0.15 PSID se, no bootstrap draw beyond the box. Standard errors (Jacobian at the estimate) 1.9–2.2 times smaller than their sampling-only ones, 4.2–5.5 times smaller than their reported ones. Within their reported errors (\|z\| ≤ 0.87); against their sampling-only ones β 1.8, δ 1.6 and ρ 0.8 combined se away. v2_b010, the reference until 2026-11-14: β 0.437 (0.027), δ 0.9928, ρ 1.58, R_γ 1.042. The old box's 0.531 is superseded: its floor held couples' estimates up | RESULTS 70.5, 69.8 (63.10, 63.6, 61.4) |
| more waves were already sharper in an earlier model | three-parameter model (before card types and R_γ): 10 waves gave held-out log q 5.446 against 7 waves' 5.007, both calibrated | `PSID_DATA.md` §3 |
| households simulated at their θ | under v2, individual 90% intervals contain their θ for 99% (β), 91% (δ), 96% (ρ) and 86% (R_γ). The average posterior mean lands further away the noisier the data; without noise it is within 0.013 of their β (ρ stays 0.5 high: its long upper tail) | RESULTS 62.6 |
| what noise PSID couples have | a network that estimates each household's noise reads couples, on average, as noisy as the gross arm or noisier, far above v2's training noise; one couple at a time, a third or fewer exclude v2's range; income reads noisier than the box allows (likely the couples' income process) | RESULTS 62.3 |
| the population β under PSID-level noise | with the Jacobian recomputed at each point, typical and gross converge: β 0.132 (se 0.019) and 0.157 (0.011), δ 0.9991 / 0.9986, ρ 1.95 / 1.11, R_γ 1.038 / 1.036; z about −3.3 against their reported β error. Noise-unknown stops at β 0.143 (θ misfit ≤ 1.9 se), its income noise unmatched at the box top. Typical's 90% interval reaches the floor of 0.10 (7% of the bootstrap below it). The noise model moves β by about 0.3, 10–11 of v2's standard errors. The noise model of "psid noise methodology.md", written from the literature, gives 0.15–0.18 at seven waves, with them | RESULTS 63.10 (63.7, 62.4), 67.8 |
| one household's β below 0.3 | resolved like the rest of the box at seven waves: in every band 0.2 wide one household's 90% interval is 0.5–0.6 wide, and the network ranks bands 0.2 apart the right way round 0.70–0.73 of the time; no simulated or PSID household has its 95th percentile below 0.3, but at the top only 0.9% have their 5th above 0.8 (63.4's reading, "not resolved", is corrected) | RESULTS 64.12 (63.4–63.5) |
| a denser training set | doubling the data (20k to 40k draws) leaves the share of prior variance removed unchanged (±0.011) and adds +0.19 log q; another doubling would cost about 6 GPU-days for about +0.1 log q | RESULTS 63.8 |
| noisy copies of the training windows | 8 fixed noisy copies of every clean 7-wave window, under v2's noise: held-out log q +0.115 [+0.085, +0.142], every member better, information on β, δ and ρ +5–6%, by delaying the overfit to one noise draw; about what another doubling of the draws would buy, with no new simulation. The population se at θ₀ does not fall for β and δ, and PSID couples do not move. Under the methodology file's heavier noise the copies add nothing (−0.005). A new noise draw every epoch does not beat the copies under v2's noise (+0.014 [−0.012, +0.039]), with one eighth of their GPU memory, and gains +0.027 [+0.006, +0.050] under the file's; nothing downstream moves. What is left of the training gap is the clean households memorised. Halving the learning rate on a plateau adds +0.03–0.05 for three of four arms (v2_fresh_rop 3.828, the best 7-wave v2-noise network); patience 60 instead of 20 adds nothing; the schedule's v2 networks lose more on the file's noise | RESULTS 67.4–67.6, 68.4–68.6, 69.4–69.7 |
| more waves sharpen one household (simulated) | 7, 10, 14 and 20 waves inside ages 24–64, all calibrated (SBC 0.888–0.912): share of prior variance removed β 0.50 → 0.70 (all four 0.47–0.52 → 0.65–0.70), 2.1–2.3 times the information for 2.9 times the waves, not levelled off; β's corr with truth 0.71 → 0.84; pairs ordered at p ≥ 0.9 26% → 44% (97–98% right) | RESULTS 64.4–64.5 |
| more waves help β below 0.3 less | from 7 to 20 waves its 90% interval narrows 16% (top of the box 32%) and its ranking rises 0.04 (top 0.12). Cardholders mostly at the borrowing limit barely sharpen (sd 0.179 → 0.160) while those rarely there do (0.177 → 0.109), and below 0.3 that is 63–67% of them. Behaviour keeps moving with β below 0.3, and simple summaries add nothing to the network. 64.6's "flat curve" is corrected | RESULTS 64.12 (64.6) |
| the population with more waves | households sharing θ₀: se at 20 waves 0.28–0.47 of 7 waves'. PSID couples: β se 0.027 → 0.007 if their extra spread shrinks like noise, → 0.022 if it is couples differing (63% of β's variance at 7 waves is that remainder); more couples, not more waves, narrow the latter. Under both, more precise than Laibson et al.'s sampling-only se (0.53 of it at 7 waves, 0.15–0.44 at 20) | RESULTS 64.8 |
| the 7-wave arm on the matched range reproduces PSID | couples' medians within 0.003 (β), ρ +0.15 (a sixth of a posterior sd); couple-by-couple corr 0.93–0.96 | RESULTS 64.7 |

## Still to do, in order

1. **RESULTS 62 (done, 2026-11-01).**
   - The noise-free arm fails on noisy data and reads PSID couples' ρ at the floor.
   - The noise-unknown arm is calibrated on all eight targets; it reads PSID couples as noisy as gross or more, and under that noise the population β goes to the floor (no fit inside the box).
   - **Decided by the user (2026-11-02):** "let's generate data including beta below 0.3 to try the models. I think v2 looks best until now." v2 stays the reference noise model, and the β floor is extended by a strip (below).
2. **RESULTS 63 (done, 2026-11-04): the β strip below 0.3.**
   - New draws on β ∈ [0.10, 0.30), trained with v2's main run and R_γ strip; the box is β ∈ [0.10, 1] (`scripts/run_beta_strip.sh`). Every arm is calibrated on it.
   - Under v2 the population fit converges inside the box at β 0.437 (se 0.027 with the Jacobian at the estimate, RESULTS 63.10), replacing 0.531.
   - One household's β below 0.3 is not resolved with seven waves. *(Corrected in RESULTS 64.12: no worse than any other band 0.2 wide; one household's β interval is 0.5–0.6 wide everywhere.)*
   - No denser dataset is needed (the learning curve).
   - **The Jacobian recomputed at each point (RESULTS 63.10, 2026-11-04, user's request):**
     - under PSID-level noise the population β is 0.13–0.16: typical and gross converge, and noise-unknown stops there with income noise unmatched;
     - v2's 0.437 holds.
   - Typical's 90% interval reaches the floor of 0.10 (7% of its bootstrap below it). **Decided by the user (2026-11-05):** "keep the floor at 0.10".
   - **§1 switched to v2 on the wider box (user, 2026-11-05):** the household tables, calibration, heterogeneity and resolution (figure 103) were `optionA2v2_b010`'s until 2026-11-14 (then v2_fresh_rop, RESULTS 70).
3. **The waves experiment (simulated households). Done 2026-11-06 (RESULTS 64), with every window inside ages 24–64 (the user's choice): 7, 10, 14 and 20 waves.**
   - **Results:** every arm is calibrated. One household's precision grows steadily without levelling off by 20 waves (2.1–2.3 times the information, 44% of pairs ordered instead of 26%). β below 0.3 gains less than the rest of the box: resolved as well as anywhere at seven waves, slower with more waves, because those households spend most years at the borrowing limit (RESULTS 64.12, correcting "a flat curve"). PSID couples' population se falls by 19–72% at 20 waves, depending on how much of it is couples differing (two cases, 64.8). No arm is short of capacity (64.9), so no wider-embedder arm was needed. Nothing looked odd, so the run-forward fallback was not run.
   - **Open, proposed (needs the user):** a simulated population whose θ differ as couples' do, to measure the two population cases directly (a few GPU-hours).
   - **Annual waves (RESULTS 65, done 2026-11-07, the user's request):** 14 and 20 one-year waves (a14, a20), every year read inside ages 24–64. Against the biennial arms with the same number of waves (w14, w20: more years against more frequent observations) and with about the same years (w07, w10: what the years between biennial waves add). No PSID counterpart (card debt on its own only from 2011, biennial); the simulated annual gain is an upper bound, because the noise model draws fresh error every wave.
     - **Results:** both arms calibrated.
       - In about the same years, every year instead of every other carries 1.9 times the information on one household's β (1.7–1.9 for δ and ρ, 1.4 for R_γ).
       - At the same number of waves, annual beats biennial for β and ρ (1.07–1.17 times), is even for δ and loses for R_γ (0.81).
       - β below 0.3 gains with the number of waves, not their spacing.
       - For the population at θ₀, at the same number of waves the longer biennial span gives the smaller se for β and δ. The annual windows start later under the age rule, and β's se runs through δ, which needs years.
       - PSID couples' β se under case (b) stays at 0.023–0.024.
   - **Noise-free arms (RESULTS 66, done 2026-11-08, the user's request):** w07, a14 and w14 retrained with no observation noise. That measures the cost of noise at each length and spacing, and how much of the annual gain was averaging fresh noise. The noise-free w07 also completes the 7-wave noise ladder on PSID couples (none, v2, typical, gross). It is a benchmark, not a candidate model for PSID.
     - **Results:** all three calibrated on noise-free data.
       - Without noise one household's information on β, δ and ρ grows 1.8–3.1 times (R_γ 1.3). Seven noise-free waves tell as much as 20 noisy ones.
       - The annual gain is partly noise: in about the same years annual still carries 1.53 times the information on β without noise (1.93 with it), and at the same number of waves biennial wins every parameter (annual 0.74–0.82).
       - β below 0.3 stays the weakest band without noise (interval −8 to −14%, the top −46 to −59%): behaviour, not measurement.
       - The noise-free networks break on v2's noise and read it as low ρ, low R_γ and high δ. On PSID couples none_w07 puts ρ near the floor (median 0.76, 69% below 1, against 2.5–2.7 under every noisy arm), and no noise-free population in the box fits them.
       - **Checked at the user's request (66.9):** simulated couples given PSID-level consumption noise come out the same under none_w07 (ρ 0.72–0.92), and consumption noise alone does it. The low ρ measures PSID's noise, not couples' preferences: the noise model is needed.
       - New: w07's 24–51 window moves the representative β by −0.05 and ρ by +0.25 against v2_b010's 24–45 (about two se); §1 keeps v2_b010. *(Revised in RESULTS 67.8: most of that gap can be the members.)*
   - **Noisy copies and the methodology file's noise (RESULTS 67, done 2026-11-09, the user's request):** v2_k8 (8 fixed noisy copies of every clean 7-wave window, v2's noise), file_k1 (the noise model of "psid noise methodology.md" as written) and file_k8 (both), against w07. Per-epoch training CSVs now go to `training_metrics/<run>/s<seed>.csv`.
     - **Results:** all three calibrated (0.886–0.906).
       - Under v2's noise fixed copies pass the rule set before the run: held-out log q +0.115 [+0.085, +0.142], information +5–6%. Under the file's noise they add nothing (−0.005 [−0.036, +0.023]). The population se and PSID couples barely move.
       - The file's noise costs one household 21–33% of its information and puts couples' representative β at 0.15–0.18, with typical and gross; its heavier consumption noise does not bring back low ρ.
       - New caveat: two ensembles of the same design differ by about two bootstrap se in the representative θ (w07 β 0.384, v2_k8 0.453).
       - **New noise every epoch (RESULTS 68, done 2026-11-10, the user's request; both noises):** v2_fresh and file_fresh, a new noise draw on the training windows every epoch, against one draw and 8 fixed copies on identical held-out windows.
         - **Results:** both calibrated (0.886–0.913). Under v2's noise it does not beat fixed copies (+0.014 [−0.012, +0.039]; +0.129 against one draw). Under the file's noise it gains +0.027 [+0.006, +0.050], where copies gained nothing. One household, the population se and PSID couples barely move.
         - A third v2-noise ensemble puts couples' representative β at 0.444, beside v2_k8's 0.453 and v2_b010's 0.437; w07's 0.384 is the outlier. §1's ensemble caveat now cites three.
         - Found: `laibson_comparison.py` draws all arms' bootstraps from one stream, so adding an arm moves the bootstrap se of the arms after it (estimates unchanged). Left as it is; seeding by arm would change every earlier section's se (the user's call).
         - **Open (needs the user):** fresh noise as the training default, and whether to retrain §1's reference with it.
       - **Reduce-on-plateau and a harder stop (RESULTS 69, done 2026-11-12, the user's request):** v2_k8_rop, v2_fresh_rop, file_k8_rop and file_fresh_rop. Each is its RESULTS 67/68 counterpart with the learning rate halved (floor 1e-5) at 5 passes without a gain, a stop after 60 passes without one, and a cap of 600 (fixed copies as in 67, all 8 per epoch).
         - **Tests, set before the run:** each arm against its counterpart, and fresh against fixed under the schedule, by paired held-out log q.
         - **Results:** all four calibrated (0.882–0.911). The schedule helps three arms of four by the rule (+0.039 v2's copies, +0.051 v2's fresh draws, +0.031 the file's copies; all three stand after the six-test correction); the file's fresh draws gain nothing (−0.010). Under the schedule fresh beats fixed on v2's noise by the 95% rule only (+0.026 [+0.001, +0.051]). One household gains up to 6% information; the population se does not move.
         - **The harder stop bought nothing:** in every member a stop after 20 passes would have kept the same weights (post hoc, from the CSVs); patience 60 cost a third of the training time. No gain came below a rate of 3.1e-5.
         - **The price is robustness to the noise model:** the schedule's v2 networks score 0.09–0.32 log q lower on the file's noise. PSID couples' β reads about 0.01 lower under all four; couples' representative β is 0.428 under both v2-noise arms. Five v2-noise ensembles now give 0.384–0.453 (sd 0.027, about one bootstrap se); §1 keeps v2_b010 and its caveat cites five.
         - The server shut down mid-run; two members retrained from scratch (RESULTS 69.2).
         - **Decided by the user (2026-11-14):** "make v2_fresh_rop as a reference. I think it's the best." §1 now reads v2_fresh_rop as trained here, with nothing retrained (RESULTS 70): PSID couples move by about a quarter of a posterior sd, couples' representative β is 0.428 (0.027). It brings two caveats: R_γ's SBC rank test fails (as for every noisy network on w07's window), and it is the least robust v2-noise network to the file's noise.
         - **Open (needs the user):** the schedule (with patience 20) and fresh draws as the training default for new runs.
   - **Models.** Train with 7 (retrained on the matched range), 10, 14 and 20 waves, under v2's noise model on the box with the β strip, with everything else as v2. If the longer windows underfit (training loss still falling while held-out loss stalls), add one wider-embedder arm. That costs GPU-hours, not a regeneration. The training set is dense enough as it is (RESULTS 63.8).
   - **No regeneration needed.** The stored panels cover ages 20–90 for every household, so longer windows are cut from the existing shards (`optionA.py train --n_waves --start_high`). Every window lies inside ages 24–64 (start ≤ 65 − 2 × waves; the user's choice, 2026-11-05); PSID's start ages run forward to 83 is the fallback.
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
  - it means more precise, and the location depends on the noise model (β 0.428 under v2's with the reference v2_fresh_rop, 0.437 with v2_b010; 0.13–0.19 under PSID-level noise and the methodology file's, RESULTS 63.10, 69.8, 70.5), so the claim is made under the reference noise model, v2's (the user's choice, 2026-11-02), on the box extended to β 0.10, with the PSID-level-noise arms reported beside it;
  - on that box v2's β is 0.428 (v2_b010 0.437), not their 0.53; the agreement the old box showed was partly its floor's work (RESULTS 63.6);
  - our standard errors hold the first stage (income process, demographics, credit limits) fixed, so the like-for-like comparison is their sampling-only error;
  - both estimates are one θ for everyone, though couples differ;
  - the bootstrap se resamples households, not networks: five ensembles trained for the same noise and window give representative β 0.384, 0.453, 0.444, 0.428 and 0.428, an sd of about one se (RESULTS 67.8, 68.8, 69.8), which is small beside the noise model's 0.24–0.30; the reference, v2_fresh_rop, is the last of these;
  - the reference fits v2's noise most closely and is the least robust v2-noise network to the methodology file's noise (held-out log q 1.19 there against 1.45–1.51; RESULTS 69.7), so its PSID reading leans more on v2's noise model being right;
  - both models share their income process, whose transitory variance absorbs PSID's income measurement error;
  - their model fails its own overidentification test (p = 7×10⁻¹⁵).
- **"More waves would predict much more precisely":**
  - measured on simulated households under our model (RESULTS 64): true for one household's posterior and for telling households apart; less so for β below 0.3, where households spend most years at the borrowing limit (RESULTS 64.12); for the population of 409 couples, only partly, since couples' differences do not shrink with waves;
  - observing every year instead of every other (RESULTS 65) adds more to one household's β per wave than a wave further on (0.93–0.95 against 0.66–0.67 of an average wave), but this is an upper bound (the simulated noise is fresh every wave) and has no PSID counterpart; for the population's β and δ, years matter more than frequency;
  - without observation noise (RESULTS 66) a wave further on is worth more than a year in between for every parameter (β 0.87 against 0.53 of an average wave): the in-between years' edge came from separating fresh noise;
  - the precision gains are under our noise model; measured exactly, seven waves would tell as much as 20 noisy ones (RESULTS 66), so better measurement is worth as much as longer panels;
  - the per-household gain is steady but less than proportional (each wave beyond the seventh adds 0.60–0.72 of an average first-seven wave);
  - real longer panels also bring attrition, changes of household head and more measurement error.
- **"Only seven waves" is PSID's limit, not our choice** (`PSID_DATA.md` §3, verified against PSID's codebooks):
  - PSID is biennial from 1999, so 13 waves exist in calendar terms;
  - the model needs credit-card debt on its own, since liquid wealth goes negative only through card debt;
  - before 2011, PSID reports one lumped total of "credit card charges, student loans, medical or legal bills, or loans from relatives", from which card debt cannot be recovered;
  - from 2011 card debt is a separate item, so 2011–2023 (seven biennial waves) is the ceiling with a clean debt measure;
  - ten waves over 2005–2023 were rejected: the first three would mismeasure card debt, in the borrow-and-repay years that identify β;
  - more PSID waves arrive one at a time, every two years (2025, 2027, …).

## How this guides the next plans

- **Every next step should serve the story above.** After RESULTS 64, items 4 (PSID framed by the experiment) and 5 (the write-up) remain.
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
