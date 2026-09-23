#!/usr/bin/env bash
# Re-score the Phase 4 ensembles with the window their members actually trained
# on.
#
# ensemble_eval hardcoded start_high = START_HIGH[7] = 46, which belongs to the
# pre-Phase 4 window convention. Every Phase 4 member trained at --start_high
# 45 (RESULTS.md 10.8), so the ensemble and its members were scored on held-out
# and SBC windows cut one start age wider than training. start_high was already
# recorded in each member's _config; it simply was not checked. It is now.
#
# Together with the member-rescoring change this makes each arm's own
# results.json the single comparable record: start_high 45, education-filtered
# SBC, identical held-out draws.
set -uo pipefail
cd /home/household-level-discount-estimator

until grep -q "sweep2 complete" logs/arch_sweep2.log 2>/dev/null; do sleep 60; done

SH="--shards data/processed/phase4_educ_dataset_shards
    --sbc_cache outputs/phase4_educ/sbc_sims.pt --start_low 24 --start_high 45"

rescore () {
    local tag=$1 dirs=$2; shift 2
    echo "$(date -u '+%F %T') rescoring $tag"
    PYTHONPATH=. .venv/bin/python scripts/ensemble_eval.py --waves 7 $SH "$@" \
        --run_dirs $dirs --tag "$tag" --out outputs/ensemble \
        > "logs/rescore_${tag}.log" 2>&1
    echo "$(date -u '+%F %T') $tag rc=$?"
}

rescore phase4_comphs "outputs/phase4/comphs_s0 outputs/phase4/comphs_s1 outputs/phase4/comphs_s2 outputs/phase4/comphs_s3 outputs/phase4/comphs_s4" --educ_group comphs
rescore logfeat_w7 "outputs/test_logfeat/w7_s0 outputs/test_logfeat/w7_s1 outputs/test_logfeat/w7_s2 outputs/test_logfeat/w7_s3 outputs/test_logfeat/w7_s4" --educ_group comphs --log_features
echo "$(date -u '+%F %T') rescore complete"
