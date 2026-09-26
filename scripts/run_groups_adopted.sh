#!/usr/bin/env bash
# The section 20 configurations for the other two education groups.
#
# Both current models (wide embedder alone, and wide embedder + log(1-delta))
# were trained on comphs only. This trains each for somehs and compco at five
# seeds and runs PSID on each group's own households, so the per-group
# literature figures can be drawn with the same models as figures 08 and 09.
set -uo pipefail
cd /home/household-level-discount-estimator

ARCH="--d_model 128 --n_layers 3 --embed_dim 64"
SH="--shards data/processed/phase4_educ_dataset_shards"
COMMON="--windows 7 --start_low 24 --start_high 45 --k 1 --educ mixed
        --batch_size 1024 --learning_rate 1e-3 --n_sbc 1000 $SH
        --sbc_cache outputs/phase4_educ/sbc_sims.pt $ARCH"
mkdir -p logs outputs/groups

jobs_for () {   # group seed
    local g=$1 s=$2
    PYTHONPATH=. .venv/bin/python scripts/compare_windows.py $COMMON \
        --educ_group "$g" --train_seed "$s" --skip_sbc \
        --out "outputs/groups/arch_${g}_s${s}" \
        > "logs/groups_arch_${g}_s${s}.log" 2>&1 &
    PYTHONPATH=. .venv/bin/python scripts/test_delta_transform.py \
        --transform log1m --train_seed "$s" $ARCH --save_posterior $SH \
        --educ_group "$g" --out "outputs/groups/full_${g}_s${s}" \
        > "logs/groups_full_${g}_s${s}.log" 2>&1 &
}

for g in somehs compco; do
    for s in 0 1 2 3 4; do
        jobs_for "$g" "$s"
        wait
        echo "$(date -u '+%F %T') $g seed $s trained"
    done
    for cfg in arch full; do
        extra=""; [ "$cfg" = full ] && extra="--delta_transform"
        PYTHONPATH=. .venv/bin/python scripts/psid_posterior.py \
            --run_dirs outputs/groups/${cfg}_${g}_s{0,1,2,3,4} \
            --waves 7 --x data/processed/psid_x_educ_rental.pt \
            --educ_group "$g" $extra --out "outputs/psid_groups/${cfg}_${g}" \
            > "logs/psid_groups_${cfg}_${g}.log" 2>&1
        echo "$(date -u '+%F %T') PSID ${cfg} ${g} rc=$?"
    done
done
echo "$(date -u '+%F %T') all groups complete"
