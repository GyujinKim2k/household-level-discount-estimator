#!/usr/bin/env bash
# Option A2 pilot (RESULTS 43): the first 4,096 draws of the couples
# regeneration.
#
# Same generative process as option A (comphs, card 50/50, R_gamma on
# [1.025, 1.075] per block of 16, M = 16 households per draw, full grid,
# stored panels), with two changes:
#   - the fixed solver: utility centred at mean income (RESULTS 40-41);
#   - each household's age-20 wealth drawn from the PSID married young-head
#     pool (RESULTS 42.2), liquid >= 0, instead of Laibson et al.'s single
#     SCF seed.
# Proposal: uniform on the prior box for these 4,096 draws. The rest of the
# run (28,672 draws) waits for the re-aimed concentrated region; switching
# then is a documented solver_config change, verified to reproduce these
# shards, as in RESULTS 32.2.
#
# ~12.4 s per draw on the V100: about 14 h. Resumable (existing shards are
# skipped; theta, R_gamma, card and seeds all come from the run seed).
#
# Launch:  nohup setsid ./scripts/run_optionA2_pilot.sh > logs/optionA2_pilot.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

POOL=data/processed/seed_pool_couples.npz
[ -f "$POOL" ] || { echo "missing $POOL: run scripts/build_couples_inputs.py"; exit 1; }

echo "$(date -u '+%F %T UTC') pilot: first 4096 of 32768 draws, uniform, PSID seed pool"
.venv/bin/python scripts/generate_dataset.py \
    --simulator twoasset --grid full --device cuda \
    --n_samples 32768 --max_draws 4096 --block 512 --theta_batch 16 --chunk 16 \
    --n_households 16 --card_types --proposal uniform \
    --rgamma_range 1.025 1.075 --init_pool "$POOL" \
    --n_waves 7 --wave_years 2 --start_age 30 --seed 0 \
    --out data/processed/optionA2_dataset.pt >> logs/optionA2_generation.log 2>&1
echo "$(date -u '+%F %T UTC') pilot exited rc=$?"
