#!/usr/bin/env bash
# RESULTS 27: two further input representations, each matched to embed_big
# (wide embedder, linear delta, comphs, same SBC draws), run after the
# log + per-household arm (run_logseq.sh) so the GPU is never shared three ways.
#   anchor       : --anchor_log
#   logseq_ratio : --log_features --per_sequence --household_ratios
set -uo pipefail
cd /home/household-level-discount-estimator
until grep -q "PSID rc=" logs/logseq_chain.log 2>/dev/null; do sleep 60; done

arm () {   # tag, flags (training), flags (evaluation and PSID)
    local tag=$1 tflags=$2 eflags=$3
    local common="--windows 7 --start_low 24 --start_high 45 --k 1 --educ mixed
        --batch_size 1024 --learning_rate 1e-3 --n_sbc 1000 --educ_group comphs
        --d_model 128 --n_layers 3 --embed_dim 64
        --shards data/processed/phase4_educ_dataset_shards
        --sbc_cache outputs/phase4_educ/sbc_sims.pt"
    mkdir -p "outputs/$tag"
    for wave in "0 1" "2 3" "4"; do
        for s in $wave; do
            PYTHONPATH=. .venv/bin/python scripts/compare_windows.py $common $tflags \
                --train_seed "$s" --out "outputs/$tag/s${s}" > "logs/${tag}_s${s}.log" 2>&1 &
        done
        wait
    done
    echo "$(date -u '+%F %T') $tag trained"
    local dirs="outputs/$tag/s0 outputs/$tag/s1 outputs/$tag/s2 outputs/$tag/s3 outputs/$tag/s4"
    PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/ensemble_eval.py \
        --waves 7 --run_dirs $dirs --shards data/processed/phase4_educ_dataset_shards \
        --sbc_cache outputs/phase4_educ/sbc_sims.pt --start_low 24 --start_high 45 \
        --educ_group comphs $eflags --tag "$tag" --out outputs/ensemble > "logs/ensemble_${tag}.log" 2>&1
    echo "$(date -u '+%F %T') $tag ensemble rc=$?"
    PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/psid_posterior.py \
        --run_dirs $dirs --waves 7 --x data/processed/psid_x_educ_rental.pt --educ_group comphs \
        $eflags --out "outputs/psid_$tag" > "logs/psid_${tag}.log" 2>&1
    echo "$(date -u '+%F %T') $tag PSID rc=$?"
}
arm anchor "--anchor_log" "--anchor_log"
arm logseq_ratio "--log_features --per_sequence --household_ratios" "--log_features --household_ratios"
echo "$(date -u '+%F %T') all representation arms complete"
