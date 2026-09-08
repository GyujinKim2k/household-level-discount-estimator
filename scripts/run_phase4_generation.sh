#!/usr/bin/env bash
# Phase 4 regeneration: education-group heterogeneity, M=8 households per solve.
#
# Two stages, chained deliberately:
#
#   1. SBC simulations (~3.5 h).  Runs FIRST, and not only because it is short.
#      It exercises the new `educ` grouping in solve_batch at production
#      settings -- full grid, theta_batch 16, chunk 16 -- so a defect in that
#      path costs 3.5 h rather than 9.4 days. If it fails, generation never
#      starts.
#   2. Generation (~9.4 days).  65,536 Sobol draws, one solve each, 8 households
#      per solve, education group drawn uniformly per draw.
#
# The stages share the GPU serially. Running them concurrently would risk an OOM
# partway through a nine-day job and would change the batching regime, which the
# solver is explicitly not invariant to.
#
# Resumable: generate_dataset.py skips shards that already exist, so an
# interrupted run continues where it stopped. The education draw is taken from
# the run seed, so a resumed shard describes the same draw as it would have.
set -uo pipefail
cd /home/household-level-discount-estimator

OUT=data/processed/phase4_educ_dataset.pt
SBC=outputs/phase4_educ/sbc_sims.pt
GRID=full
THETA_BATCH=16
CHUNK=16

mkdir -p logs outputs/phase4_educ

echo "$(date -u '+%F %T UTC') stage 1/2: SBC simulations (educ=mixed)"
PYTHONPATH=. .venv/bin/python - <<'PY' > logs/phase4_sbc.log 2>&1
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
from scripts.compare_windows import simulate_sbc_once

cfg = {"grid": "full", "theta_batch": 16, "chunk": 16, "device": "cuda"}
# Same seed as the comphs cache: the SBC draws should differ only in the
# generative process, not in which thetas were drawn.
simulate_sbc_once(1000, 20260822, cfg,
                  cache=Path("outputs/phase4_educ/sbc_sims.pt"),
                  educ="mixed")
print("SBC simulations complete")
PY
rc=$?
if [ $rc -ne 0 ] || [ ! -f "$SBC" ]; then
    echo "$(date -u '+%F %T UTC') SBC FAILED (rc=$rc); not starting generation"
    tail -20 logs/phase4_sbc.log
    exit 1
fi
echo "$(date -u '+%F %T UTC') stage 1 done -> $SBC"

echo "$(date -u '+%F %T UTC') stage 2/2: generation (65536 draws, M=8, educ=mixed)"
PYTHONPATH=. .venv/bin/python scripts/generate_dataset.py \
    --simulator twoasset --grid "$GRID" --device cuda \
    --n_samples 65536 --block 512 \
    --theta_batch "$THETA_BATCH" --chunk "$CHUNK" \
    --n_households 8 --educ mixed \
    --n_waves 7 --wave_years 2 --start_age 30 --seed 0 \
    --out "$OUT" >> logs/phase4_generation.log 2>&1
echo "$(date -u '+%F %T UTC') stage 2 exited rc=$?"
