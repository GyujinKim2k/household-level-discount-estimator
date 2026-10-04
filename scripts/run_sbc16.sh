#!/usr/bin/env bash
# RESULTS 38: a fresh SBC set for option A, through the training code path.
# 1,024 uniform-prior draws (scrambled Sobol on the box), 16 households each,
# R_gamma and card type per block of 16, every other setting as generation
# (run_optionA_generation.sh stage 2). Seed 30,000,000: household shock streams
# (seed + draw + 1) overlap neither training (1-32,768) nor the first SBC set
# (20,260,822-20,261,821). ~3.5 h on the V100, then the evaluation (minutes).
#
# Launch:  nohup setsid ./scripts/run_sbc16.sh > logs/sbc16.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
mkdir -p logs

echo "$(date -u '+%F %T') generation: 1024 uniform draws, 16 households each"
$PY scripts/generate_dataset.py \
    --simulator twoasset --grid full --device cuda \
    --n_samples 1024 --block 512 --theta_batch 16 --chunk 16 \
    --n_households 16 --card_types --proposal uniform \
    --rgamma_range 1.025 1.075 \
    --n_waves 7 --wave_years 2 --start_age 30 --seed 30000000 \
    --out data/processed/optionA_sbc16.pt > logs/sbc16_generation.log 2>&1
rc=$?
echo "$(date -u '+%F %T') generation rc=$rc"
[ $rc -eq 0 ] || exit 1

$PY scripts/sbc_households.py > logs/sbc16_evaluate.log 2>&1
echo "$(date -u '+%F %T') evaluate rc=$?"
