#!/usr/bin/env bash
# embed_big at five seeds, then ensembled.
#
# It is the one arm of the capacity sweep that improved recovery (RESULTS 19.2):
# better corr and mae on all three parameters, but member coverage 0.03-0.05
# short of the baseline's. Sharper members that are over-confident about their
# sharpness. Ensembling exists to correct exactly that, and 19.1 showed it
# contributes roughly twice as much to members that have more of it to correct
# -- so the single-seed table cannot settle whether this is a win.
#
# Seed 0 already exists as outputs/arch/embed_big; only 1-4 are run here.
set -uo pipefail
cd /home/household-level-discount-estimator

until grep -q "sweep2 complete" logs/arch_sweep2.log 2>/dev/null; do sleep 60; done
echo "$(date -u '+%F %T') sweep2 done, starting embed_big seeds"

COMMON="--windows 7 --start_low 24 --start_high 45 --k 1 --educ mixed
        --batch_size 1024 --learning_rate 1e-3 --n_sbc 1000 --educ_group comphs
        --d_model 128 --n_layers 3 --embed_dim 64
        --shards data/processed/phase4_educ_dataset_shards
        --sbc_cache outputs/phase4_educ/sbc_sims.pt"

for wave in "1 2" "3 4"; do
    for s in $wave; do
        PYTHONPATH=. .venv/bin/python scripts/compare_windows.py $COMMON \
            --train_seed "$s" --out "outputs/arch/embed_big_s${s}" \
            > "logs/arch_embed_big_s${s}.log" 2>&1 &
    done
    wait
    echo "$(date -u '+%F %T') seeds [$wave] rc=$?"
done

mkdir -p outputs/arch/embed_big_s0
cp -a outputs/arch/embed_big/. outputs/arch/embed_big_s0/

PYTHONPATH=. .venv/bin/python scripts/ensemble_eval.py --waves 7 \
    --run_dirs outputs/arch/embed_big_s0 outputs/arch/embed_big_s1 \
               outputs/arch/embed_big_s2 outputs/arch/embed_big_s3 \
               outputs/arch/embed_big_s4 \
    --shards data/processed/phase4_educ_dataset_shards \
    --sbc_cache outputs/phase4_educ/sbc_sims.pt --start_low 24 \
    --educ_group comphs --tag arch_embed_big --out outputs/ensemble \
    > logs/ensemble_embed_big.log 2>&1
echo "$(date -u '+%F %T') embed_big ensemble rc=$?"
