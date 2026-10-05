#!/usr/bin/env bash
# Option A2 pilot model (RESULTS 43): what the 4,096 uniform pilot draws say
# about where PSID couples sit, to re-aim the concentrated proposal before the
# remaining 28,672 draws.
#
# Three members (logit beta, log(1 - delta), R_gamma free; 6 training shards,
# 2 held out), PSID posteriors for couples (primary) and all 889, then the
# re-aiming report. Run after run_optionA2_pilot.sh has finished.
#
# Launch:  nohup setsid ./scripts/run_optionA2_pilot_model.sh > logs/optionA2_pilot_model.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
SH=data/processed/optionA2_dataset_shards
OUT=outputs/optionA2_pilot

$PY scripts/optionA.py verify --shards $SH --partial || exit 1
for s in 0 1 2; do
    echo "$(date -u '+%F %T') train member $s"
    $PY scripts/optionA.py train --seed $s --shards $SH --beta_transform logit \
        --out $OUT > logs/optionA2_pilot_train_s$s.log 2>&1 || { echo "train $s failed"; exit 1; }
done
RUNS="$OUT/s0 $OUT/s1 $OUT/s2"
echo "$(date -u '+%F %T') PSID couples"
$PY scripts/optionA.py psid --run_dirs $RUNS --x data/processed/psid_x_comphs_couples.pt \
    --out outputs/psid_optionA2_pilot_couples > logs/optionA2_pilot_psid_couples.log 2>&1 || exit 1
echo "$(date -u '+%F %T') PSID all 889"
$PY scripts/optionA.py psid --run_dirs $RUNS --x data/processed/psid_x_comphs_optionA.pt \
    --out outputs/psid_optionA2_pilot_all > logs/optionA2_pilot_psid_all.log 2>&1 || exit 1
echo "$(date -u '+%F %T') report"
$PY scripts/optionA2_pilot_report.py > logs/optionA2_pilot_report.log 2>&1
echo "$(date -u '+%F %T') done rc=$?"
