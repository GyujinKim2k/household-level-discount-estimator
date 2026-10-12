#!/usr/bin/env bash
# RESULTS 43.11: SBC for A2n, the liquidation-penalty misspecification check
# and the alpha check, on the five A2n members.
#
# 1. SBC set: 1,024 uniform-prior draws on the A2w box (R_gamma 1.021-1.075),
#    16 households each, every A2 training setting (full grid, fixed solver,
#    card type and R_gamma per block of 16, PSID couples' seed pool). Seed
#    40,000,000: household streams 40,000,001-40,001,024 overlap no training
#    run and no earlier SBC set. ~3.5 h on the V100.
# 2. Penalty sets: the SBC set's first 512 draws again, same seed, at
#    liquidation-penalty scale 0.5 and 1.5. ~1.8 h each.
# 3. sbc_optionA2.py: SBC (windows carry A2n's noise) and the paired penalty
#    shifts; figure 45.
# 4. Alpha: PSID couples with card debt x 2.02 (RESULTS 42.3); figure 46
#    against A2n couples.
#
# Launch:  nohup setsid ./scripts/run_optionA2n_sbc.sh >> logs/optionA2n_sbc.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
POOL=data/processed/seed_pool_couples_income.npz
DIRS="outputs/optionA2n/s0 outputs/optionA2n/s1 outputs/optionA2n/s2 outputs/optionA2n/s3 outputs/optionA2n/s4"
mkdir -p logs outputs/optionA2n/sbc

gen() {   # gen OUT_NAME LOG [extra args]
    local out=$1 log=$2; shift 2
    $PY scripts/generate_dataset.py --simulator twoasset --grid full --device cuda \
        --n_samples 1024 --block 512 --theta_batch 16 --chunk 16 \
        --n_households 16 --card_types --proposal uniform \
        --rgamma_range 1.021 1.075 --init_pool "$POOL" \
        --n_waves 7 --wave_years 2 --start_age 30 --seed 40000000 \
        "$@" --out "data/processed/$out.pt" > "logs/$log.log" 2>&1
}

echo "$(date -u '+%F %T') SBC set: 1024 uniform draws, 16 households each"
gen optionA2_sbc16 optionA2n_sbc_generation
rc=$?; echo "$(date -u '+%F %T') SBC generation rc=$rc"
[ $rc -eq 0 ] || exit 1
for s in 050 150; do
    gen "optionA2_sbc16_liqpen$s" "optionA2n_sbc_liqpen$s" --liqpen_scale "${s:0:1}.${s:1}" --max_draws 512
    rc=$?; echo "$(date -u '+%F %T') penalty set x${s:0:1}.${s:1} rc=$rc"
done

$PY scripts/sbc_optionA2.py --run_dirs $DIRS > logs/optionA2n_sbc_evaluate.log 2>&1
rc=$?; echo "$(date -u '+%F %T') SBC evaluation rc=$rc"

$PY scripts/optionA.py psid --run_dirs $DIRS --x data/processed/psid_x_comphs_couples.pt \
    --debt_scale 2.02 --out outputs/psid_optionA2n_couples_alpha202 > logs/optionA2n_alpha.log 2>&1
rc=$?; echo "$(date -u '+%F %T') alpha (card debt x 2.02) rc=$rc"
$PY scripts/plot_optionA2_psid.py \
    --series "A2n, couples=outputs/psid_optionA2n_couples" \
             "A2n, couples, card debt x 2.02=outputs/psid_optionA2n_couples_alpha202" \
    --out figures/46_optionA2n_alpha.png \
    --title "Per-household posterior means, PSID couples: option A2n with PSID card debt as reported and doubled (alpha = 2.02)" \
    > logs/optionA2n_alpha_figure.log 2>&1
rc=$?; echo "$(date -u '+%F %T') alpha figure rc=$rc"
