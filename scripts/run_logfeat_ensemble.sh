#!/usr/bin/env bash
# Settle RESULTS.md 13.2: does the single-seed calibration gain from signed-log
# features survive ensembling, and does it help on the out-of-support PSID
# households that motivated it?
#
# Both questions need a full 5-seed ensemble. The single-seed gain (coverage
# 0.751 -> 0.842) may not survive, because ensembling already adds ~0.18 and the
# two together could overshoot 0.900 into over-coverage.
#
# Matched to outputs/phase4/comphs_* in every argument except --log_features.
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=.

COMMON="--windows 7 --start_low 24 --start_high 45 --k 1 --educ mixed
        --educ_group comphs --log_features --batch_size 1024
        --learning_rate 1e-3 --n_sbc 1000
        --shards data/processed/phase4_educ_dataset_shards
        --sbc_cache outputs/phase4_educ/sbc_sims.pt"

# seed 0 already exists from the single-seed test; reuse it.
for wave in "1 2 3" "4"; do
    pids=()
    for s in $wave; do
        [ -f "outputs/test_logfeat/w7_s${s}/posterior_7w.pt" ] && continue
        .venv/bin/python scripts/compare_windows.py $COMMON --train_seed "$s" \
            --out "outputs/test_logfeat/w7_s${s}" \
            > "logs/logfeat_s${s}.log" 2>&1 &
        pids+=($!); sleep 45
    done
    for p in "${pids[@]:-}"; do [ -n "${p:-}" ] && wait "$p"; done
done
echo "$(date -u '+%F %T') training done"

.venv/bin/python scripts/ensemble_eval.py --waves 7 \
    --shards data/processed/phase4_educ_dataset_shards \
    --sbc_cache outputs/phase4_educ/sbc_sims.pt --start_low 24 \
    --educ_group comphs --log_features \
    --run_dirs outputs/test_logfeat/w7_s0 outputs/test_logfeat/w7_s1 \
               outputs/test_logfeat/w7_s2 outputs/test_logfeat/w7_s3 \
               outputs/test_logfeat/w7_s4 \
    --tag logfeat_w7 --out outputs/ensemble > logs/ensemble_logfeat.log 2>&1
echo "$(date -u '+%F %T') ensembling rc=$?"

.venv/bin/python scripts/psid_posterior.py --waves 7 --n_post 6000 \
    --log_features --x data/processed/psid_x_rental.pt \
    --run_dirs outputs/test_logfeat/w7_s0 outputs/test_logfeat/w7_s1 \
               outputs/test_logfeat/w7_s2 outputs/test_logfeat/w7_s3 \
               outputs/test_logfeat/w7_s4 \
    --out outputs/psid_logfeat > logs/psid_logfeat.log 2>&1
echo "$(date -u '+%F %T') psid rc=$?"
echo "$(date -u '+%F %T') logfeat ensemble test complete"
