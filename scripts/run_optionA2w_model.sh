#!/usr/bin/env bash
# Option A2w (RESULTS 43.5): the A2 model with R_gamma's prior widened to
# [1.021, 1.075] and R_gamma's flow target logit, trained on the main run plus
# the R_gamma strip below it (run_optionA2_strip.sh).
#
# Waits for the strip run to exit cleanly, then refuses to go on unless both
# runs pass `optionA.py verify` together (all draws, theta = each run's
# proposal sequence, R_gamma tiling [1.021, 1.075], same simulator settings,
# init states, finite panels). Five members with logit beta and logit R_gamma,
# two at a time; PSID posteriors for couples and all 889; the held-out check
# (weighted to the prior, with the floors); figures 39-40 (one held-out
# simulated household, the typical PSID couple) and 41 (couples and all 889
# against A2).
#
# Launch:  nohup setsid ./scripts/run_optionA2w_model.sh <run_optionA2_strip.sh PID> \
#              >> logs/optionA2w_model.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
MAIN=data/processed/couples_dataset_shards
STRIP=data/processed/couples_strip_dataset_shards
OUT=outputs/optionA2w
mkdir -p logs "$OUT"

WRAPPER=${1:-}
if [ -n "$WRAPPER" ]; then
    while kill -0 "$WRAPPER" 2>/dev/null; do sleep 300; done
fi
# The last exit line only: the log is appended to across stops and resumes.
grep "strip run exited" logs/couples_strip.log | tail -1 | grep -q "rc=0$" \
    || { echo "$(date -u '+%F %T') strip run did not finish cleanly; not training"; exit 1; }
echo "$(date -u '+%F %T') strip run finished; verifying both runs"
$PY scripts/optionA.py verify --shards $MAIN $STRIP || { echo "verify FAILED; not training"; exit 1; }

echo "$(date -u '+%F %T') training 5 members, 2 at a time"
for s in 0 1 2 3 4; do echo "$s"; done |
    xargs -P 2 -L 1 bash -c '"$0" scripts/optionA.py train --seed "$1" \
        --shards "'"$MAIN"'" "'"$STRIP"'" --beta_transform logit --rgamma_transform logit \
        --out "'"$OUT"'" > "logs/optionA2w_train_s$1.log" 2>&1;
        rc=$?; echo "$(date -u "+%F %T") seed $1 rc=$rc"' "$PY"
DIRS=""
for s in 0 1 2 3 4; do
    [ -f "$OUT/s$s/posterior_7w.pt" ] || { echo "seed $s has no checkpoint"; exit 1; }
    DIRS="$DIRS $OUT/s$s"
done

$PY scripts/optionA.py psid --run_dirs $DIRS --x data/processed/psid_x_comphs_couples.pt \
    --out outputs/psid_optionA2w_couples > logs/optionA2w_psid_couples.log 2>&1
rc=$?; echo "$(date -u '+%F %T') PSID couples rc=$rc"
$PY scripts/optionA.py psid --run_dirs $DIRS --x data/processed/psid_x_comphs_optionA.pt \
    --out outputs/psid_optionA2w_all > logs/optionA2w_psid_all.log 2>&1
rc=$?; echo "$(date -u '+%F %T') PSID all 889 rc=$rc"

$PY scripts/check_optionA2_heldout.py --run_dirs $DIRS --shards $MAIN $STRIP \
    --psid "couples, A2w=outputs/psid_optionA2w_couples" "couples, A2=outputs/psid_optionA2_couples" \
           "all 889, A2w=outputs/psid_optionA2w_all" "all 889, A2=outputs/psid_optionA2_all" \
    --out $OUT/heldout_check.json > logs/optionA2w_heldout.log 2>&1
rc=$?; echo "$(date -u '+%F %T') held-out check rc=$rc"

$PY scripts/plot_optionA2_households.py --run_dirs $DIRS --shards $MAIN $STRIP \
    --psid_out outputs/psid_optionA2w_couples --x data/processed/psid_x_comphs_couples.pt \
    --label "option A2w (R_gamma 1.021-1.075, logit target; main run + strip; 5 members)" \
    --sim_name 39_simulated_household_optionA2w.png --psid_name 40_psid_household_optionA2w.png \
    > logs/optionA2w_household_figures.log 2>&1
rc=$?; echo "$(date -u '+%F %T') household figures rc=$rc"
$PY scripts/plot_optionA2_psid.py \
    --series "A2w, couples (5 members)=outputs/psid_optionA2w_couples" \
             "A2w, all 889 (5 members)=outputs/psid_optionA2w_all" \
             "A2, couples (R_gamma floor 1.025)=outputs/psid_optionA2_couples" \
    --out figures/41_optionA2w_psid.png \
    --title "Per-household posterior means, PSID comphs: option A2w (R_gamma 1.021-1.075, logit target) against A2" \
    > logs/optionA2w_psid_figure.log 2>&1
rc=$?; echo "$(date -u '+%F %T') population figure rc=$rc"
