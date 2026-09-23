#!/usr/bin/env bash
# Training-side sweep: flow and embedder capacity, on the comphs group.
#
# `hidden_features=50, num_transforms=5` are sbi's own defaults. They have been
# carried unchanged since Phase 1 and never swept, and they size the flow for a
# dataset two orders of magnitude smaller than the one we now have (152k rows
# from 19,049 panels for one education group). The embedder dims (d_model 64,
# 2 layers, output 32) are in the same position.
#
# One seed per arm. The 5-seed spread on this group is sd 0.041 in log q, 0.003
# -0.006 in corr and 0.010-0.018 in coverage, so anything smaller than roughly
# 0.1 log q or 0.03 coverage is noise and must not be read as an effect.
#
# The control is outputs/phase4/comphs_s0 -- identical arguments, defaults for
# every knob below. It is not re-run here.
set -uo pipefail
cd /home/household-level-discount-estimator

COMMON="--windows 7 --start_low 24 --start_high 45 --k 1 --educ mixed
        --batch_size 1024 --learning_rate 1e-3 --n_sbc 1000 --train_seed 0
        --educ_group comphs
        --shards data/processed/phase4_educ_dataset_shards
        --sbc_cache outputs/phase4_educ/sbc_sims.pt"
mkdir -p logs outputs/arch

run () {
    local tag=$1; shift
    echo "$(date -u '+%F %T') start $tag"
    PYTHONPATH=. .venv/bin/python scripts/compare_windows.py $COMMON "$@" \
        --out "outputs/arch/$tag" > "logs/arch_${tag}.log" 2>&1
    echo "$(date -u '+%F %T') $tag rc=$?"
}

# Two at a time: the re-scoring job shares the card.
run flow_wide  --hidden_features 256 --num_transforms 5 &
run flow_deep  --hidden_features 50  --num_transforms 10 &
wait
run flow_big   --hidden_features 128 --num_transforms 8 &
run embed_big  --d_model 128 --n_layers 3 --embed_dim 64 &
wait
echo "$(date -u '+%F %T') sweep complete"
