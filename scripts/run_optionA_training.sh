#!/usr/bin/env bash
# RESULTS 35: option A's four-parameter model -- train, evaluate, apply to PSID.
#
# Waits for the generation run to exit, then refuses to go on unless every
# shard passes `optionA.py verify` (all 32,768 draws, theta = the switched
# proposal's sequence, R_gamma and card per block, finite panels). Five seeds
# of the adopted configuration (RESULTS 29) with R_gamma as a fourth target,
# two at a time; then the importance-weighted SBC / held-out evaluation, the
# PSID run on psid_x_comphs_optionA.pt, the heterogeneity test, and the
# literature figure against the RESULTS 29 headline.
#
# Launch:  nohup setsid ./scripts/run_optionA_training.sh > logs/optionA_training.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
OUT=outputs/optionA
mkdir -p logs "$OUT"

while pgrep -f "scripts/generate_dataset.py.*optionA_card_dataset" > /dev/null; do
    sleep 300
done
echo "$(date -u '+%F %T') generation finished; verifying shards"
$PY scripts/optionA.py verify || { echo "verify FAILED; not training"; exit 1; }

for wave in "0 1" "2 3" "4"; do
    for s in $wave; do
        $PY scripts/optionA.py train --seed "$s" --out "$OUT" \
            > "logs/optionA_train_s${s}.log" 2>&1 &
    done
    wait
    echo "$(date -u '+%F %T') seeds [$wave] done"
done
for s in 0 1 2 3 4; do
    [ -f "$OUT/s$s/posterior_7w.pt" ] || { echo "seed $s has no checkpoint"; exit 1; }
done

DIRS="$OUT/s0 $OUT/s1 $OUT/s2 $OUT/s3 $OUT/s4"
$PY scripts/optionA.py evaluate --run_dirs $DIRS --out "$OUT/evaluation" \
    > logs/optionA_evaluate.log 2>&1
echo "$(date -u '+%F %T') evaluate rc=$?"
$PY scripts/optionA.py psid --run_dirs $DIRS --out outputs/psid_optionA \
    > logs/optionA_psid.log 2>&1
echo "$(date -u '+%F %T') PSID rc=$?"
$PY scripts/heterogeneity_test.py --runs outputs/psid_adopted_anchor_level outputs/psid_optionA \
    --labels headline_29 optionA --out "$OUT/heterogeneity.json" \
    > logs/optionA_heterogeneity.log 2>&1
echo "$(date -u '+%F %T') heterogeneity rc=$?"
$PY scripts/plot_adopted_posterior.py --current outputs/psid_optionA \
    --label "option A (β, δ, ρ of 4 estimated)" \
    --baseline outputs/psid_adopted_anchor_level --baseline_label "RESULTS 29 headline" \
    --out figures/31_optionA_literature_comparison.png \
    > logs/optionA_figure.log 2>&1
echo "$(date -u '+%F %T') figure rc=$?"
