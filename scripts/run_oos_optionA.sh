#!/usr/bin/env bash
# RESULTS 39: the out-of-sample test (§22) and wealth dynamics (§24) on the
# adopted option A model (logit beta, RESULTS 38). Five 5-wave members, two at
# a time (~4 h), then scripts/oos_optionA.py (~5 h of solves: every household
# theta, both card types for households never seen borrowing).
#
# Launch:  nohup setsid ./scripts/run_oos_optionA.sh > logs/oos_optionA.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
OUT=outputs/oos_optionA
mkdir -p logs "$OUT"

echo "$(date -u '+%F %T') training 5 five-wave members, 2 at a time"
for s in 0 1 2 3 4; do echo "$s"; done |
    xargs -P 2 -L 1 bash -c '"$0" scripts/optionA.py train --seed "$1" --n_waves 5 \
        --beta_transform logit --out "'"$OUT"'/w5" \
        > "logs/oos_optionA_train_s$1.log" 2>&1;
        echo "$(date -u "+%F %T") seed $1 rc=$?"' "$PY"

DIRS=""
for s in 0 1 2 3 4; do
    [ -f "$OUT/w5/s$s/posterior_5w.pt" ] || { echo "seed $s has no checkpoint"; exit 1; }
    DIRS="$DIRS $OUT/w5/s$s"
done
$PY scripts/oos_optionA.py --run_dirs $DIRS --out "$OUT" > logs/oos_optionA_forecast.log 2>&1
echo "$(date -u '+%F %T') forecast rc=$?"
