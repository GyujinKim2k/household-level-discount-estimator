#!/usr/bin/env bash
# Second training-side sweep: derived features, and a learning curve.
#
# **Derived features.** card_debt = 1{liquid<0}, liquid/income, illiquid/income
# appended as channels. Exact functions of what the network already sees, so
# this cannot add information -- it tests whether the flow is spending capacity
# rediscovering Laibson et al.'s own moment definitions.
#
# **Learning curve.** Is the model short of data or short of capacity? Training
# on 1/4 and 1/2 of the draws answers it directly, and it is the evidence the
# pending regenerate-or-not decision actually turns on: if log q is still
# climbing steeply at the full prefix, more draws buy something; if it has
# flattened, nine GPU-days buy nothing and the architecture is the binding
# constraint.
#
# The learning-curve arms must be scored with ensemble_eval --train_n 57344,
# NOT by their own results.json: compare_windows holds out everything above
# --train_n, so each arm scores itself on a different held-out set and the log q
# values are not comparable across arms.
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

run derived   --derived_features &
run half      --train_n 28672 &
wait
run quarter   --train_n 14336 &
run derivlog  --derived_features --log_features &
wait
echo "$(date -u '+%F %T') sweep2 complete"
