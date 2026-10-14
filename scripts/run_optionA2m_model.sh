#!/usr/bin/env bash
# Option A2m (RESULTS 50): A2n with PSID's imputed values treated as missing.
#
# PSID's accuracy codes mark the couple-waves whose consumption, liquid or
# illiquid amount PSID imputed. Where the imputed share of the amount is at
# least TAU, the value is missing: PSID's own is masked, and each simulated
# training household takes the whole mask pattern of a PSID couple drawn at
# random (psid_accuracy_codes.py --write_mask TAU). The network gets one
# missing-value channel per feature. The observation noise is recalibrated on
# PSID changes between unmasked waves only (noise_calibration.py --mask_tau).
# Retraining only; nothing is simulated.
#
# Five members two at a time; PSID posteriors for couples (masks exist for
# the 409 couples only); the held-out check; SBC on A2n's SBC set (figure 57);
# figures 58-59 (one held-out simulated household, the typical PSID couple)
# and 60 (couples against A2n).
#
# Launch:  nohup setsid ./scripts/run_optionA2m_model.sh 0.25 >> logs/optionA2m_model.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
TAU=${1:-0.25}
MAIN=data/processed/couples_dataset_shards
STRIP=data/processed/couples_strip_dataset_shards
OUT=outputs/optionA2m
[ "$TAU" = "0.25" ] || OUT=outputs/optionA2m_tau$TAU
CAL=outputs/optionA2m/noise_calibration_tau$TAU.json
MASK=data/processed/psid_mask_couples_tau$TAU.pt
PSID_OUT=outputs/psid_$(basename $OUT)_couples
mkdir -p logs "$OUT"

for f in "$CAL" "$MASK"; do
    [ -f "$f" ] || { echo "$(date -u '+%F %T') no $f (RESULTS 50 lists the commands)"; exit 1; }
done
NOISE=$($PY -c "import json; d = json.load(open('$CAL'))['sigma_range']; \
print(' '.join(f'{k}={lo}:{hi}' for k, (lo, hi) in d.items()))")
echo "$(date -u '+%F %T') tau $TAU; masks $MASK; observation noise: $NOISE"

echo "$(date -u '+%F %T') training 5 members, 2 at a time"
for s in 0 1 2 3 4; do echo "$s"; done |
    xargs -P 2 -L 1 bash -c '"$0" scripts/optionA.py train --seed "$1" \
        --shards "'"$MAIN"'" "'"$STRIP"'" --beta_transform logit --rgamma_transform logit \
        --obs_noise '"$NOISE"' --mask_patterns "'"$MASK"'" --out "'"$OUT"'" \
        > "logs/'"$(basename $OUT)"'_train_s$1.log" 2>&1;
        rc=$?; echo "$(date -u "+%F %T") seed $1 rc=$rc"' "$PY"
DIRS=""
for s in 0 1 2 3 4; do
    [ -f "$OUT/s$s/posterior_7w.pt" ] || { echo "seed $s has no checkpoint"; exit 1; }
    DIRS="$DIRS $OUT/s$s"
done

$PY scripts/optionA.py psid --run_dirs $DIRS --x data/processed/psid_x_comphs_couples.pt \
    --out $PSID_OUT > logs/$(basename $OUT)_psid_couples.log 2>&1
rc=$?; echo "$(date -u '+%F %T') PSID couples rc=$rc"

$PY scripts/check_optionA2_heldout.py --run_dirs $DIRS --shards $MAIN $STRIP \
    --psid "couples, A2m=$PSID_OUT" "couples, A2n=outputs/psid_optionA2n_couples" \
    --out $OUT/heldout_check.json > logs/$(basename $OUT)_heldout.log 2>&1
rc=$?; echo "$(date -u '+%F %T') held-out check rc=$rc"

$PY scripts/sbc_optionA2.py --run_dirs $DIRS --out $OUT/sbc \
    --figure figures/57_sbc_$(basename $OUT).png > logs/$(basename $OUT)_sbc.log 2>&1
rc=$?; echo "$(date -u '+%F %T') SBC rc=$rc"

$PY scripts/plot_optionA2_households.py --run_dirs $DIRS --shards $MAIN $STRIP \
    --psid_out $PSID_OUT --x data/processed/psid_x_comphs_couples.pt \
    --label "option A2m (A2n + PSID's imputed values missing, tau $TAU; 5 members)" \
    --sim_name 58_simulated_household_$(basename $OUT).png \
    --psid_name 59_psid_household_$(basename $OUT).png \
    > logs/$(basename $OUT)_household_figures.log 2>&1
rc=$?; echo "$(date -u '+%F %T') household figures rc=$rc"
$PY scripts/plot_optionA2_psid.py \
    --series "A2m, couples (5 members)=$PSID_OUT" \
             "A2n, couples (5 members)=outputs/psid_optionA2n_couples" \
    --out figures/60_$(basename $OUT)_psid.png \
    --title "Per-household posterior means, PSID couples: A2m (imputed values missing, tau $TAU) against A2n" \
    > logs/$(basename $OUT)_psid_figure.log 2>&1
rc=$?; echo "$(date -u '+%F %T') population figure rc=$rc"
