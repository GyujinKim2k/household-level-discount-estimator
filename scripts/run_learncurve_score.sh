#!/usr/bin/env bash
# Score the learning-curve arms on ONE held-out set.
#
# compare_windows holds out every draw above --train_n, so the quarter and half
# arms each scored themselves on a different held-out set and their own log q
# and correlations are not comparable across arms. Re-scoring all three through
# ensemble_eval --train_n 57344 puts them on identical draws. train_n is
# deliberately NOT a provenance field: differing from the member here is the
# point.
#
# SBC coverage needs no repair -- the SBC set is fixed and independent of
# train_n -- but it is recomputed anyway so one file holds the whole row.
set -uo pipefail
cd /home/household-level-discount-estimator

until grep -q "rescore complete" logs/rescore_chain.log 2>/dev/null; do sleep 60; done

SH="--shards data/processed/phase4_educ_dataset_shards
    --sbc_cache outputs/phase4_educ/sbc_sims.pt --start_low 24 --start_high 45
    --train_n 57344 --educ_group comphs"

for tag in quarter half derived; do
    echo "$(date -u '+%F %T') scoring $tag"
    PYTHONPATH=. .venv/bin/python scripts/ensemble_eval.py --waves 7 $SH \
        $( [ "$tag" = derived ] && echo --derived_features ) \
        --seeds 0 --run_dirs "outputs/arch/$tag" \
        --tag "arch_$tag" --out outputs/ensemble \
        > "logs/score_${tag}.log" 2>&1
    echo "$(date -u '+%F %T') $tag rc=$?"
done
echo "$(date -u '+%F %T') learning-curve scoring complete"
