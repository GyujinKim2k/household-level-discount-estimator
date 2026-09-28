#!/usr/bin/env bash
# Option A (RESULTS.md 26): comphs regeneration with card-access types.
#
# PREPARED, NOT LAUNCHED. Awaiting a decision after reviewing RESULTS 27.
#
# Two stages, chained as in Phase 4:
#   1. SBC simulations (~3.5 h): 1000 draws, same generative process (comphs,
#      card types 50/50). Runs first so a defect costs hours, not days.
#   2. Generation (~4.7 days): 32,768 draws from the edge-mixture proposal
#      (RESULTS 30; half uniform, half near the upper edges), M = 16
#      households each, card type drawn 50/50 per draw and stored as `card`,
#      theta stored per draw. Measured 12.4 s per
#      draw on the V100 (32-draw smoke test, 2026-09-27).
#
# Uses the RESULTS 25-fixed solvers, so each draw's credit line reaches the
# solve. Resumable: existing shards are skipped, and the card draw comes from
# the run seed, so a resumed shard describes the same draw.
#
# Launch:  nohup ./scripts/run_optionA_generation.sh > logs/optionA.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator

OUT=data/processed/optionA_card_dataset.pt
SBC=outputs/optionA/sbc_sims.pt
mkdir -p logs outputs/optionA

echo "$(date -u '+%F %T UTC') stage 1/2: SBC simulations (comphs, card types)"
PYTHONPATH=. .venv/bin/python - <<'PY' > logs/optionA_sbc.log 2>&1
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
from scripts.compare_windows import simulate_sbc_once
cfg = {"grid": "full", "theta_batch": 16, "chunk": 16, "device": "cuda"}
simulate_sbc_once(1000, 20260822, cfg, cache=Path("outputs/optionA/sbc_sims.pt"),
                  educ="comphs", card_types=True, rgamma_range=(1.025, 1.075))
print("SBC simulations complete")
PY
rc=$?
if [ $rc -ne 0 ] || [ ! -f "$SBC" ]; then
    echo "$(date -u '+%F %T UTC') SBC FAILED (rc=$rc); not starting generation"
    tail -20 logs/optionA_sbc.log
    exit 1
fi
echo "$(date -u '+%F %T UTC') stage 1 done -> $SBC"

echo "$(date -u '+%F %T UTC') stage 2/2: generation (32768 draws, M=16, card types, edge mixture)"
PYTHONPATH=. .venv/bin/python scripts/generate_dataset.py \
    --simulator twoasset --grid full --device cuda \
    --n_samples 32768 --block 512 --theta_batch 16 --chunk 16 \
    --n_households 16 --card_types --proposal edge_mixture \
    --rgamma_range 1.025 1.075 \
    --n_waves 7 --wave_years 2 --start_age 30 --seed 0 \
    --out "$OUT" >> logs/optionA_generation.log 2>&1
echo "$(date -u '+%F %T UTC') stage 2 exited rc=$?"
