#!/usr/bin/env bash
# The derived-feature arm at five seeds, then ensembled.
#
# It is the decisive run of section 19. Single-seed it has the best held-out log
# q of anything tried (+0.129 over the matched baseline seed), and unlike the
# wider embedder it needs no architecture change -- three appended columns, each
# an exact function of features already present.
#
# The combo arm settles which to carry forward: derived + wide embedder is WORSE
# than either alone on log q (4.818 against 4.936 and 4.886) and converges at 73
# epochs against derived's 156. They are two routes to the same gain, so taking
# both only over-parameterises. Derived is the cheaper route.
set -uo pipefail
cd /home/household-level-discount-estimator

until grep -q "learning-curve scoring complete" logs/learncurve_chain.log 2>/dev/null; do sleep 60; done
echo "$(date -u '+%F %T') starting derived seeds"

COMMON="--windows 7 --start_low 24 --start_high 45 --k 1 --educ mixed
        --batch_size 1024 --learning_rate 1e-3 --n_sbc 1000 --educ_group comphs
        --derived_features
        --shards data/processed/phase4_educ_dataset_shards
        --sbc_cache outputs/phase4_educ/sbc_sims.pt"

mkdir -p outputs/arch/derived_s0
cp -a outputs/arch/derived/. outputs/arch/derived_s0/
for wave in "1 2" "3 4"; do
    for s in $wave; do
        PYTHONPATH=. .venv/bin/python scripts/compare_windows.py $COMMON \
            --train_seed "$s" --out "outputs/arch/derived_s${s}" \
            > "logs/arch_derived_s${s}.log" 2>&1 &
    done
    wait
    echo "$(date -u '+%F %T') seeds [$wave] rc=$?"
done

PYTHONPATH=. .venv/bin/python scripts/ensemble_eval.py --waves 7 \
    --run_dirs outputs/arch/derived_s0 outputs/arch/derived_s1 \
               outputs/arch/derived_s2 outputs/arch/derived_s3 \
               outputs/arch/derived_s4 \
    --shards data/processed/phase4_educ_dataset_shards \
    --sbc_cache outputs/phase4_educ/sbc_sims.pt --start_low 24 --start_high 45 \
    --educ_group comphs --derived_features \
    --tag arch_derived_ens --out outputs/ensemble \
    > logs/ensemble_derived.log 2>&1
echo "$(date -u '+%F %T') derived ensemble rc=$?"
