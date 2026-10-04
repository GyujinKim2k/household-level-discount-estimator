#!/usr/bin/env bash
# RESULTS 37: beta-transform screen. Option A retrained with beta's flow target
# log(1 - beta) and logit, five seeds each, two jobs at a time on the GPU (each
# starts as soon as a slot frees); then evaluation and PSID per arm, and the
# comparison against the linear arm (RESULTS 36) on the criteria fixed in
# scripts/compare_beta_transforms.py.
#
# Launch:  nohup setsid ./scripts/run_beta_transform_screen.sh > logs/beta_screen.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
mkdir -p logs

echo "$(date -u '+%F %T') training 10 jobs, 2 at a time"
for bt in log1m logit; do for s in 0 1 2 3 4; do echo "$bt $s"; done; done |
    xargs -P 2 -L 1 bash -c '"$0" scripts/optionA.py train --seed "$2" \
        --beta_transform "$1" --out "outputs/optionA_beta_$1" \
        > "logs/beta_$1_train_s$2.log" 2>&1;
        echo "$(date -u "+%F %T") $1 seed $2 rc=$?"' "$PY"

for bt in log1m logit; do
    OUT=outputs/optionA_beta_$bt
    DIRS="$OUT/s0 $OUT/s1 $OUT/s2 $OUT/s3 $OUT/s4"
    missing=0
    for s in 0 1 2 3 4; do
        [ -f "$OUT/s$s/posterior_7w.pt" ] || { echo "$bt seed $s has no checkpoint"; missing=1; }
    done
    [ "$missing" = 0 ] || continue
    $PY scripts/optionA.py evaluate --run_dirs $DIRS --out "$OUT/evaluation" \
        > "logs/beta_${bt}_evaluate.log" 2>&1
    echo "$(date -u '+%F %T') $bt evaluate rc=$?"
    $PY scripts/optionA.py psid --run_dirs $DIRS --out "outputs/psid_optionA_beta_$bt" \
        > "logs/beta_${bt}_psid.log" 2>&1
    echo "$(date -u '+%F %T') $bt PSID rc=$?"
done

$PY scripts/compare_beta_transforms.py > logs/beta_screen_compare.log 2>&1
echo "$(date -u '+%F %T') compare rc=$?"
