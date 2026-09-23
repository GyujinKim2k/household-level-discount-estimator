#!/usr/bin/env bash
# The two winning arms together, then five seeds of whichever wins.
#
# RESULTS 19.3 and 19.4: widening the embedder and appending the derived moment
# channels each improve recovery on their own, and they act on different parts of
# the pipeline -- one gives the flow a wider summary, the other hands it features
# it was otherwise inferring. Nothing says they compose, so it is one run to
# find out before committing five.
set -uo pipefail
cd /home/household-level-discount-estimator

until grep -q "embed_big ensemble rc=" logs/embed_seeds.log 2>/dev/null; do sleep 60; done
echo "$(date -u '+%F %T') embed_big chain done, starting combo"

COMMON="--windows 7 --start_low 24 --start_high 45 --k 1 --educ mixed
        --batch_size 1024 --learning_rate 1e-3 --n_sbc 1000 --educ_group comphs
        --shards data/processed/phase4_educ_dataset_shards
        --sbc_cache outputs/phase4_educ/sbc_sims.pt"
COMBO="--derived_features --d_model 128 --n_layers 3 --embed_dim 64"

PYTHONPATH=. .venv/bin/python scripts/compare_windows.py $COMMON $COMBO \
    --train_seed 0 --out outputs/arch/combo > logs/arch_combo.log 2>&1
echo "$(date -u '+%F %T') combo rc=$?"
