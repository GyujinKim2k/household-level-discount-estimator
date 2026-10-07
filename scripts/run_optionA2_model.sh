#!/usr/bin/env bash
# Option A2 model on the full couples regeneration (RESULTS 43), and the two
# single-household posterior figures.
#
# Waits for run_optionA2_full.sh to exit cleanly, then refuses to go on unless
# every shard passes `optionA.py verify` (all 32,768 draws, theta = the
# a2_switched sequence, R_gamma and card per block, init states, finite
# panels). Five logit-beta members, two at a time (the option A recipe,
# RESULTS 35/38, card type marginalised); PSID posteriors for couples (primary)
# and all 889; then figures 36 (one held-out simulated household against its
# truth) and 37 (the typical PSID couple).
#
# No SBC here: option A's SBC cache was simulated under the old process (SCF
# seed, no pool), so the new one is part of the evaluation step that follows.
#
# Launch:  nohup setsid ./scripts/run_optionA2_model.sh > logs/optionA2_model.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
SH=data/processed/couples_dataset_shards
OUT=outputs/optionA2
mkdir -p logs "$OUT"

WRAPPER=${1:-}
if [ -n "$WRAPPER" ]; then
    while kill -0 "$WRAPPER" 2>/dev/null; do sleep 300; done
fi
grep -q "full run exited rc=0" logs/couples_full.log \
    || { echo "$(date -u '+%F %T') generation did not finish cleanly; not training"; exit 1; }
echo "$(date -u '+%F %T') generation finished; verifying shards"
$PY scripts/optionA.py verify --shards $SH || { echo "verify FAILED; not training"; exit 1; }

echo "$(date -u '+%F %T') training 5 members, 2 at a time"
for s in 0 1 2 3 4; do echo "$s"; done |
    xargs -P 2 -L 1 bash -c '"$0" scripts/optionA.py train --seed "$1" --shards "'"$SH"'" \
        --beta_transform logit --out "'"$OUT"'" > "logs/optionA2_train_s$1.log" 2>&1;
        echo "$(date -u "+%F %T") seed $1 rc=$?"' "$PY"
DIRS=""
for s in 0 1 2 3 4; do
    [ -f "$OUT/s$s/posterior_7w.pt" ] || { echo "seed $s has no checkpoint"; exit 1; }
    DIRS="$DIRS $OUT/s$s"
done

$PY scripts/optionA.py psid --run_dirs $DIRS --x data/processed/psid_x_comphs_couples.pt \
    --out outputs/psid_optionA2_couples > logs/optionA2_psid_couples.log 2>&1
echo "$(date -u '+%F %T') PSID couples rc=$?"
$PY scripts/optionA.py psid --run_dirs $DIRS --x data/processed/psid_x_comphs_optionA.pt \
    --out outputs/psid_optionA2_all > logs/optionA2_psid_all.log 2>&1
echo "$(date -u '+%F %T') PSID all 889 rc=$?"

$PY scripts/plot_optionA2_households.py --run_dirs $DIRS --shards $SH \
    --psid_out outputs/psid_optionA2_couples --x data/processed/psid_x_comphs_couples.pt \
    --label "option A2 (couples regeneration, PSID seed pool; 5 members)" \
    > logs/optionA2_household_figures.log 2>&1
echo "$(date -u '+%F %T') household figures rc=$?"
