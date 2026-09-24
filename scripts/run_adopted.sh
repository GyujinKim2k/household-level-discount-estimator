#!/usr/bin/env bash
# The configuration RESULTS 19.6 recommends, trained and applied to PSID.
#
# Two changes against the Phase 4 headline, both retraining-only:
#
#   * the wider embedder (d_model 128, 3 layers, 64 out) -- section 19.3/19.6,
#     the only arm that improved log q, all six recovery measures and
#     calibration together;
#   * log(1 - delta) as the estimation target -- section 13.1, which removes the
#     delta truncation affecting ~24% of comphs households, and which section
#     19.1 independently showed is aimed at the one parameter still
#     under-covering after ensembling.
#
# Neither has been run with the other before, so this is also the first test
# that they compose. Five seeds, ensembled, then PSID.
set -uo pipefail
cd /home/household-level-discount-estimator

ARCH="--d_model 128 --n_layers 3 --embed_dim 64"
mkdir -p logs outputs/adopted

for wave in "0 1" "2 3" "4"; do
    for s in $wave; do
        PYTHONPATH=. .venv/bin/python scripts/test_delta_transform.py \
            --transform log1m --train_seed "$s" $ARCH --save_posterior \
            --shards data/processed/phase4_educ_dataset_shards \
            --educ_group comphs --out "outputs/adopted/log1m_s${s}" \
            > "logs/adopted_log1m_s${s}.log" 2>&1 &
    done
    wait
    echo "$(date -u '+%F %T') seeds [$wave] done"
done

echo "$(date -u '+%F %T') training complete; running PSID"
PYTHONPATH=. .venv/bin/python scripts/psid_posterior.py \
    --run_dirs outputs/adopted/log1m_s0 outputs/adopted/log1m_s1 \
               outputs/adopted/log1m_s2 outputs/adopted/log1m_s3 \
               outputs/adopted/log1m_s4 \
    --waves 7 --x data/processed/psid_x_educ_rental.pt --educ_group comphs \
    --delta_transform --out outputs/psid_adopted \
    > logs/psid_adopted.log 2>&1
echo "$(date -u '+%F %T') PSID rc=$?"
