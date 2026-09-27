#!/usr/bin/env bash
# RESULTS 29: anchor + level (RESULTS 28) retrained with the log(1 - delta)
# target, matched to the current headline model (outputs/adopted/log1m_s*,
# levels + log(1 - delta), wide embedder, comphs) in every other argument.
# Static standardisation on age and log(mean income) only.
set -uo pipefail
cd /home/household-level-discount-estimator
ARGS="--transform log1m --d_model 128 --n_layers 3 --embed_dim 64 --save_posterior
      --anchor_log --mean_income_channel --static_norm_channels age log_mean_income
      --shards data/processed/phase4_educ_dataset_shards --educ_group comphs"
mkdir -p outputs/adopted_anchor_level logs
for wave in "0 1" "2 3" "4"; do
    for s in $wave; do
        PYTHONPATH=. .venv/bin/python scripts/test_delta_transform.py $ARGS \
            --train_seed "$s" --out "outputs/adopted_anchor_level/log1m_s${s}" \
            > "logs/adopted_anchor_level_s${s}.log" 2>&1 &
    done
    wait
    echo "$(date -u '+%F %T') seeds [$wave] done"
done
DIRS="outputs/adopted_anchor_level/log1m_s0 outputs/adopted_anchor_level/log1m_s1 outputs/adopted_anchor_level/log1m_s2 outputs/adopted_anchor_level/log1m_s3 outputs/adopted_anchor_level/log1m_s4"
FL="--delta_transform --anchor_log --mean_income_channel"
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/ensemble_eval.py --waves 7 \
    --run_dirs $DIRS --shards data/processed/phase4_educ_dataset_shards \
    --sbc_cache outputs/phase4_educ/sbc_sims.pt --start_low 24 --start_high 45 \
    --educ_group comphs $FL --tag adopted_anchor_level --out outputs/ensemble \
    > logs/ensemble_adopted_anchor_level.log 2>&1
echo "$(date -u '+%F %T') ensemble rc=$?"
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/psid_posterior.py \
    --run_dirs $DIRS --waves 7 --x data/processed/psid_x_educ_rental.pt --educ_group comphs \
    $FL --out outputs/psid_adopted_anchor_level > logs/psid_adopted_anchor_level.log 2>&1
echo "$(date -u '+%F %T') PSID rc=$?"
