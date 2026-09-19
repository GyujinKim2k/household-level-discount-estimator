#!/usr/bin/env bash
# Phase 4 training: five posterior variants on the regenerated dataset.
#
# Run AFTER generation completes. It checks for all 128 shards and refuses to
# start otherwise -- training on a partial run is legitimate only at a
# power-of-2 prefix, and doing it by accident would silently use a different
# dataset than the one reported.
#
# Window start ages are 24-45, not 25-46. aggregate_waves reports each wave at
# the last year of its window, so a window opened at S reports its first wave at
# S+1; 24-45 is what aligns simulated wave-0 ages to PSID's 25-46 (RESULTS.md
# 10.8). The misaligned window is what every pre-Phase-4 result used.
#
# k=1, not 8. At M=8 each draw already contributes 8 windows, one per
# independent household, so k=1 gives the same row count and the same start-age
# coverage as the old M=1/k=8 -- but from 8 independent trajectories rather than
# 8 overlapping slices of one (RESULTS.md 10.4).
#
# Variants, in priority order. The per-group models come first because they are
# the clearest gain from the regeneration: somehs and compco are 738 households
# that could not be analysed at all before. The marginalised model is computed
# but is NOT the preferred variant -- it discards education, which we observe
# (RESULTS.md 10.10).
set -uo pipefail
cd /home/household-level-discount-estimator

SHARDS=data/processed/phase4_educ_dataset_shards
SBC=outputs/phase4_educ/sbc_sims.pt
SEEDS="0 1 2 3 4"
COMMON="--windows 7 --start_low 24 --start_high 45 --k 1 --educ mixed
        --batch_size 1024 --learning_rate 1e-3 --n_sbc 1000
        --shards $SHARDS --sbc_cache $SBC"

n=$(ls "$SHARDS"/shard_*.npz 2>/dev/null | wc -l)
if [ "$n" -ne 128 ]; then
    echo "$(date -u '+%F %T') only $n/128 shards present; refusing to start"
    exit 1
fi
mkdir -p logs outputs/phase4

# Three at a time: five concurrent fitted on an idle card, but a partial failure
# here costs a rerun rather than a nine-day job, so there is no reason to crowd.
run_variant () {
    local tag=$1; shift
    echo "$(date -u '+%F %T') === $tag ==="
    for wave in "0 1 2" "3 4"; do
        pids=()
        for s in $wave; do
            .venv/bin/python scripts/compare_windows.py $COMMON "$@" \
                --train_seed "$s" --out "outputs/phase4/${tag}_s${s}" \
                > "logs/phase4_${tag}_s${s}.log" 2>&1 &
            pids+=($!); sleep 45
        done
        for p in "${pids[@]}"; do wait "$p" || echo "  seed failed in [$wave]"; done
    done
    .venv/bin/python scripts/ensemble_eval.py --waves 7 \
        --run_dirs outputs/phase4/${tag}_s0 outputs/phase4/${tag}_s1 \
                   outputs/phase4/${tag}_s2 outputs/phase4/${tag}_s3 \
                   outputs/phase4/${tag}_s4 \
        --tag "phase4_${tag}" --out outputs/ensemble \
        > "logs/ensemble_phase4_${tag}.log" 2>&1
    echo "$(date -u '+%F %T') $tag done"
}

run_variant comphs     --educ_group comphs
run_variant somehs     --educ_group somehs
run_variant compco     --educ_group compco
run_variant cond       --condition_educ
run_variant marg
echo "$(date -u '+%F %T') all variants complete"
