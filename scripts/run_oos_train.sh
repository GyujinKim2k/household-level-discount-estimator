#!/usr/bin/env bash
# 5-wave comphs model for the out-of-sample test: it sees waves 1-5 only, so
# PSID waves 6-7 stay genuinely unseen. Wide embedder (RESULTS 19.3), linear
# delta: compare_windows is the script that trains at arbitrary window length.
# SBC skipped -- the prediction test is the evaluation here.
set -uo pipefail
cd /home/household-level-discount-estimator
for s in 0 1 2; do
    PYTHONPATH=. .venv/bin/python scripts/compare_windows.py \
        --windows 5 --start_low 24 --start_high 45 --k 1 --educ mixed \
        --batch_size 1024 --learning_rate 1e-3 --educ_group comphs --skip_sbc \
        --d_model 128 --n_layers 3 --embed_dim 64 --train_seed "$s" \
        --shards data/processed/phase4_educ_dataset_shards \
        --out "outputs/oos/w5_s${s}" > "logs/oos_w5_s${s}.log" 2>&1 &
done
wait
echo "$(date -u '+%F %T') oos training done"
