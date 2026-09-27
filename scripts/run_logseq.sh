#!/usr/bin/env bash
# RESULTS 27: per-household ("row-wise") normalisation AFTER a signed-log
# transform, on the current dataset. Matched to the embed_big arm (wide
# embedder, linear delta, comphs, same SBC draws) in every other argument, so
# the comparison isolates the input representation.
#
# The earlier per-household test (outputs/per_seq, Phase 3 data, levels) lost
# badly -- log q 2.51 vs 5.47 -- while staying calibrated. The motivation here:
# in logs, within-household variation becomes proportional change rather than
# dollar change, and normalising it removes the household's level, which would
# make the model immune to the liquid-level misfit of RESULTS 24.
set -uo pipefail
cd /home/household-level-discount-estimator
COMMON="--windows 7 --start_low 24 --start_high 45 --k 1 --educ mixed
        --batch_size 1024 --learning_rate 1e-3 --n_sbc 1000 --educ_group comphs
        --d_model 128 --n_layers 3 --embed_dim 64
        --log_features --per_sequence
        --shards data/processed/phase4_educ_dataset_shards
        --sbc_cache outputs/phase4_educ/sbc_sims.pt"
mkdir -p logs outputs/logseq
for wave in "0 1" "2 3" "4"; do
    for s in $wave; do
        PYTHONPATH=. .venv/bin/python scripts/compare_windows.py $COMMON \
            --train_seed "$s" --out "outputs/logseq/s${s}" \
            > "logs/logseq_s${s}.log" 2>&1 &
    done
    wait
    echo "$(date -u '+%F %T') seeds [$wave] done"
done
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/ensemble_eval.py --waves 7 \
    --run_dirs outputs/logseq/s0 outputs/logseq/s1 outputs/logseq/s2 outputs/logseq/s3 outputs/logseq/s4 \
    --shards data/processed/phase4_educ_dataset_shards --sbc_cache outputs/phase4_educ/sbc_sims.pt \
    --start_low 24 --start_high 45 --educ_group comphs --log_features \
    --tag logseq --out outputs/ensemble > logs/ensemble_logseq.log 2>&1
echo "$(date -u '+%F %T') ensemble rc=$?"
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/psid_posterior.py \
    --run_dirs outputs/logseq/s0 outputs/logseq/s1 outputs/logseq/s2 outputs/logseq/s3 outputs/logseq/s4 \
    --waves 7 --x data/processed/psid_x_educ_rental.pt --educ_group comphs --log_features \
    --out outputs/psid_logseq > logs/psid_logseq.log 2>&1
echo "$(date -u '+%F %T') PSID rc=$?"
