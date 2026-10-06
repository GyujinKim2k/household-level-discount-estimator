#!/usr/bin/env bash
# Option A2, the rest of the run (RESULTS 43.3): draws 4,096-32,767 of the
# couples regeneration, on top of the 4,096-draw uniform pilot.
#
# Same generative process as the pilot (run_optionA2_pilot.sh): comphs, card
# 50/50 and R_gamma on [1.025, 1.075] per block of 16, M = 16 households per
# draw, full grid, stored panels, fixed solver, PSID seed pool (wealth and
# income). Only the proposal changes, to SWITCHED_A2: the pilot's uniform
# draws are kept exactly, and the remaining 28,672 are half uniform, half on
# beta >= 0.40, delta >= 0.95 (all rho, all R_gamma) -- where 87% of PSID
# couples' pilot posterior means sit. The shard directory's solver_config.json
# records the switch in a `_note`.
#
# ~12.8 s per draw on the V100: about 4.2 days. Resumable (existing shards are
# skipped; theta, R_gamma, card and seeds all come from the run seed).
#
# Launch:  nohup setsid ./scripts/run_optionA2_full.sh > logs/couples_full.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

POOL=data/processed/seed_pool_couples_income.npz
[ -f "$POOL" ] || { echo "missing $POOL: run scripts/build_couples_inputs.py"; exit 1; }

echo "$(date -u '+%F %T UTC') full run: 32768 draws, a2_switched, PSID seed pool (wealth + income)"
.venv/bin/python scripts/generate_dataset.py \
    --simulator twoasset --grid full --device cuda \
    --n_samples 32768 --block 512 --theta_batch 16 --chunk 16 \
    --n_households 16 --card_types --proposal a2_switched \
    --rgamma_range 1.025 1.075 --init_pool "$POOL" \
    --n_waves 7 --wave_years 2 --start_age 30 --seed 0 \
    --out data/processed/couples_dataset.pt >> logs/couples_generation.log 2>&1
echo "$(date -u '+%F %T UTC') full run exited rc=$?"
