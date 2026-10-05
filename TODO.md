# TODO

Deferred work, with the evidence that motivates each item. RESULTS.md holds the
results; this file holds what is still to be tried.

## Single-adult household type

**Why.** The model is always a 2-adult household (`grids.household_composition`
sets `spouse = 2` at every age, as Laibson et al. do), but only 42–47% of our
889 comphs PSID heads are married in a given wave (A3, legal marriage). At ages
35–44 the two groups barely overlap:

```
ages 35-44             n waves   income p50   liquid p50   in debt   illiquid p50   zero illiquid
married                  1,334      $66,314           $0       44%        $62,871             15%
not married              1,530      $26,641           $0       26%         $3,465             40%
model (fixed solver)                                                   $56k-$204k
```

The model's illiquid "over-accumulation" (RESULTS 41.5) is mostly a
comparison against single-adult households the model does not describe. For
now the PSID headline uses couples (married in at least 4 of 7 waves, 409
households; 310 married in all 7) and generation stays a 2-adult model.

**Design, when we try it.**
- A per-draw composition type, `spouse = 1` or `2`, recorded per draw like the
  card type. Each draw is its own solve already, so it costs no extra compute
  per draw, but each type gets half the draws, or the run grows to about 7
  GPU-days.
- Marital status is observed in PSID, so condition the network on it, as
  education was handled (RESULTS 9.7). Households whose status changes inside
  the window need a rule (modal status, or exclude).
- **Income will not match.** With `spouse = 1` the calibrated profile scales
  income by `exp(-0.319) = 0.73`; PSID's unmarried households earn 0.40 of the
  married ones' median. Their regression's spouse coefficient is an average
  effect, not a singles' income process. Either accept it or re-estimate the
  profile for singles, which departs from the replication.
- No separate IPUMS kids profile for singles exists in the replication
  package; the averaged one would be reused.
- Seed pools for initial wealth by type (married / not married young heads).
- The SCF first stage (credit limit, initial wealth) is adjusted to two heads;
  a singles version would need re-running it with `nhead = 1`.

## Cohabiting couples

A3 marital status counts only legal marriage. PSID's "wife" includes
long-term cohabiting partners, but our extract does not carry their presence:
spouses and partners are in it only partially. Needs a new PSID pull of the
wife/partner presence variables (or MARITAL STATUS-GEN) for 2011–2023, then the
couples rule can include cohabitation.

## denser grid regeneration
generate the dataset for the final model training.
make the final biggest regeneration and build the most precise model.

## Other deferred items

- **δ as a logit target** (RESULTS 39.1): fold into the post-regeneration
  retraining as a second arm.
- **Liquidation-penalty sensitivity** (RESULTS 41.3): PSID at penalty scale 0.5
  and 1.5 after regeneration, not as a fifth parameter.
- **α = 2.02 sensitivity** (RESULTS 6, deviation 2): PSID card debt × 2.02 at
  inference. Low priority: RESULTS 42.3 found PSID's raw balances already
  about 2× SCF's raw at the median, so doubling would overshoot.
- **Extra card-possession signal** (RESULTS 42.6): 2019 GSD11 and 2021
  GCOVID10 ("used credit card") imply a card for a few never-borrowers.
  Needs a PSID pull.
- **Option B** (RESULTS 26): somehs and compco, repairing RESULTS 25's
  credit-limit bug for those groups.
- **Quasi-hyperbolic vs exponential** (RESULTS 21): the β = 1 dataset.
