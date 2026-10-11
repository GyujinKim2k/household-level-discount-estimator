#!/usr/bin/env bash
# Option A2, the R_gamma strip below the box (RESULTS 43.5): 3,072 draws with
# R_gamma on [1.021, 1.025), so the prior can widen to [1.021, 1.075] without
# regenerating the main run. 88% of PSID couples' posteriors reached the old
# floor at 1.025 (43.4); 1.021 stays above R_free = 1.0203, below which the
# illiquid asset is dominated (18).
#
# Same generative process as the main run (run_optionA2_full.sh): comphs, card
# 50/50 and R_gamma per block of 16, M = 16 households per draw, full grid,
# stored panels, fixed solver, PSID seed pool. (beta, delta, rho) come from the
# A2 mixture alone (A2_STRIP). Its own seed, so its income shocks and starting
# states (seed + draw + 1) never repeat the main run's (1-32,768).
#
# 6 shards of 512: optionA holds out the last one, so 2,560 train -- the main
# run's training density per unit of R_gamma (31,744 / 0.050 vs 2,560 / 0.004).
# ~12.8 s per draw on the V100: about 11 h. Resumable.
#
# Launch:  nohup setsid ./scripts/run_optionA2_strip.sh >> logs/couples_strip.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

POOL=data/processed/seed_pool_couples_income.npz
[ -f "$POOL" ] || { echo "missing $POOL: run scripts/build_couples_inputs.py"; exit 1; }

echo "$(date -u '+%F %T UTC') strip run: 3072 draws, a2_strip, R_gamma 1.021-1.025, seed 100000"
.venv/bin/python scripts/generate_dataset.py \
    --simulator twoasset --grid full --device cuda \
    --n_samples 3072 --block 512 --theta_batch 16 --chunk 16 \
    --n_households 16 --card_types --proposal a2_strip \
    --rgamma_range 1.021 1.025 --init_pool "$POOL" \
    --n_waves 7 --wave_years 2 --start_age 30 --seed 100000 \
    --out data/processed/couples_strip_dataset.pt >> logs/couples_strip_generation.log 2>&1
rc=$?
echo "$(date -u '+%F %T UTC') strip run exited rc=$rc"
