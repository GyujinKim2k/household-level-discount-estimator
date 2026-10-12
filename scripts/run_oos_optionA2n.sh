#!/usr/bin/env bash
# RESULTS 43.12: the out-of-sample test (§22) and wealth dynamics (§24) on
# option A2n. Five 5-wave members on A2n's setup (main run + R_gamma strip,
# logit beta and R_gamma, the same observation noise), two at a time; then
# scripts/oos_optionA.py on PSID couples: parameters from waves 1-5, each
# couple run forward from its wave-5 state, scored on waves 6-7, as simulated
# and with the training noise on the forecasts. The model's own households
# start from the couples' seed pool, as A2's training households did.
#
# Launch:  nohup setsid ./scripts/run_oos_optionA2n.sh >> logs/oos_optionA2n.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
MAIN=data/processed/couples_dataset_shards
STRIP=data/processed/couples_strip_dataset_shards
CAL=outputs/optionA2n/noise_calibration.json
OUT=outputs/oos_optionA2n
mkdir -p logs "$OUT"

NOISE=$($PY -c "import json; d = json.load(open('$CAL'))['sigma_range']; \
print(' '.join(f'{k}={lo}:{hi}' for k, (lo, hi) in d.items()))")
echo "$(date -u '+%F %T') training 5 five-wave members, 2 at a time; observation noise: $NOISE"
for s in 0 1 2 3 4; do echo "$s"; done |
    xargs -P 2 -L 1 bash -c '"$0" scripts/optionA.py train --seed "$1" --n_waves 5 \
        --shards "'"$MAIN"'" "'"$STRIP"'" --beta_transform logit --rgamma_transform logit \
        --obs_noise '"$NOISE"' --out "'"$OUT"'/w5" > "logs/oos_optionA2n_train_s$1.log" 2>&1;
        rc=$?; echo "$(date -u "+%F %T") seed $1 rc=$rc"' "$PY"

DIRS=""
for s in 0 1 2 3 4; do
    [ -f "$OUT/w5/s$s/posterior_5w.pt" ] || { echo "seed $s has no checkpoint"; exit 1; }
    DIRS="$DIRS $OUT/w5/s$s"
done
$PY scripts/oos_optionA.py --run_dirs $DIRS --x data/processed/psid_x_comphs_couples.pt \
    --init_pool data/processed/seed_pool_couples_income.npz --out "$OUT" \
    > logs/oos_optionA2n_forecast.log 2>&1
rc=$?; echo "$(date -u '+%F %T') forecast rc=$rc"
