#!/usr/bin/env bash
# Resume option A's generation (stage 2 only) after an interruption.
#
# Stage 1 (SBC simulations, outputs/optionA/sbc_sims.pt) finished on
# 2026-09-29 and is not re-run. generate_dataset.py skips shards already on
# disk and writes each new one to a temp name before renaming it, so an
# interrupted shard leaves nothing behind and is simply redone. The run's
# config check refuses to resume under any different setting (including a
# different GPU model).
#
#   nohup setsid ./scripts/resume_optionA_generation.sh > logs/optionA_resume.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
echo "$(date -u '+%F %T UTC') resuming generation; $(ls data/processed/optionA_card_dataset_shards/shard_*.npz | wc -l) shards on disk"
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/generate_dataset.py \
    --simulator twoasset --grid full --device cuda \
    --n_samples 32768 --block 512 --theta_batch 16 --chunk 16 \
    --n_households 16 --card_types --proposal edge_mixture_switched \
    --rgamma_range 1.025 1.075 \
    --n_waves 7 --wave_years 2 --start_age 30 --seed 0 \
    --out data/processed/optionA_card_dataset.pt >> logs/optionA_generation.log 2>&1
echo "$(date -u '+%F %T UTC') generation exited rc=$?"
