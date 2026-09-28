#!/usr/bin/env bash
# RESULTS 31: where does rho ~ 4.5 come from? Tests 3 and 2 (test 1, DC
# pensions, needs no training).
#   test 3: the adopted configuration without illiquid wealth in the inputs
#   test 2: rho as a known input; beta and delta estimated at fixed rho
set -uo pipefail
cd /home/household-level-discount-estimator
SH="--shards data/processed/phase4_educ_dataset_shards"

# --- test 3 ---------------------------------------------------------------
ARGS="--transform log1m --d_model 128 --n_layers 3 --embed_dim 64 --save_posterior
      --anchor_log --mean_income_channel --static_norm_channels age log_mean_income
      --features noilliq_age $SH --educ_group comphs"
mkdir -p outputs/noilliq logs
for wave in "0 1" "2 3" "4"; do
    for s in $wave; do
        PYTHONPATH=. .venv/bin/python scripts/test_delta_transform.py $ARGS \
            --train_seed "$s" --out "outputs/noilliq/log1m_s${s}" > "logs/noilliq_s${s}.log" 2>&1 &
    done
    wait
done
echo "$(date -u '+%F %T') test 3 trained"
DIRS="outputs/noilliq/log1m_s0 outputs/noilliq/log1m_s1 outputs/noilliq/log1m_s2 outputs/noilliq/log1m_s3 outputs/noilliq/log1m_s4"
FL="--delta_transform --anchor_log --mean_income_channel --features noilliq_age"
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/ensemble_eval.py --waves 7 \
    --run_dirs $DIRS $SH --sbc_cache outputs/phase4_educ/sbc_sims.pt --start_low 24 --start_high 45 \
    --educ_group comphs $FL --tag noilliq --out outputs/ensemble > logs/ensemble_noilliq.log 2>&1
echo "$(date -u '+%F %T') test 3 ensemble rc=$?"
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/psid_posterior.py \
    --run_dirs $DIRS --waves 7 --x data/processed/psid_x_educ_rental.pt --educ_group comphs \
    $FL --out outputs/psid_noilliq > logs/psid_noilliq.log 2>&1
echo "$(date -u '+%F %T') test 3 PSID rc=$?"

# --- test 2 ---------------------------------------------------------------
for wave in "0 1" "2 3" "4"; do
    for s in $wave; do
        PYTHONPATH=. .venv/bin/python scripts/rho_conditioned.py train --seed "$s" \
            > "logs/rho_conditioned_s${s}.log" 2>&1 &
    done
    wait
done
echo "$(date -u '+%F %T') test 2 trained"
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/rho_conditioned.py psid \
    > logs/rho_conditioned_psid.log 2>&1
echo "$(date -u '+%F %T') test 2 PSID rc=$?"
echo "$(date -u '+%F %T') rho tests complete"
