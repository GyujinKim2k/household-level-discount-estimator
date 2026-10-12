#!/usr/bin/env bash
# RESULTS 43.10: how far couples' rho under A2n moves with the observation-noise
# ranges. Retraining only, two members (seeds 0, 1) per arm, on the A2n setup
# (main run + R_gamma strip, logit beta and R_gamma). Arms, from the calibrated
# ranges in outputs/optionA2n/noise_calibration.json (RESULTS 43.9):
#
#   cons010  consumption 0-0.10, the other features as calibrated
#   cons020  consumption 0-0.20, the other features as calibrated
#   all067   every feature's whole range x 2/3
#   all133   every feature's whole range x 4/3
#
# The all-feature arms move every noisy feature together, so no feature is
# singled out. Scaling illiquid's lower end by 2/3 and 4/3 (0.117, 0.233) also
# spans the 90% interval of its minimum (0.115-0.220). Income stays at 0.
#
# Per arm: PSID couples, the held-out check (held-out windows carry the arm's
# noise) and the known-rho recovery in couples' A2n region. The A2n reference
# is its own seeds 0 and 1, through the same steps, so every arm is compared
# at two members.
#
# Launch:  nohup setsid ./scripts/run_optionA2n_noise_sensitivity.sh >> logs/optionA2n_sens.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
MAIN=data/processed/couples_dataset_shards
STRIP=data/processed/couples_strip_dataset_shards
CAL=outputs/optionA2n/noise_calibration.json
OUT=outputs/optionA2n_sens
COUPLES=data/processed/psid_x_comphs_couples.pt
mkdir -p logs "$OUT"

# noise SCALE [FEATURE=HIGH ...]: the calibrated ranges, each end times SCALE,
# then any FEATURE's upper end replaced by HIGH.
noise() {
    $PY - "$@" <<EOF
import json, sys
scale, over = float(sys.argv[1]), dict(a.split("=") for a in sys.argv[2:])
d = json.load(open("$CAL"))["sigma_range"]
print(" ".join(f"{k}={round(lo * scale, 4)}:{round(float(over.get(k, hi * scale)), 4)}"
               for k, (lo, hi) in d.items()))
EOF
}

# evaluate NAME DIRS...: PSID couples, held-out check, recovery.
evaluate() {
    local name=$1; shift
    $PY scripts/optionA.py psid --run_dirs "$@" --x $COUPLES \
        --out outputs/psid_optionA2n_sens_${name}_couples > logs/optionA2n_sens_${name}_psid.log 2>&1
    echo "$(date -u '+%F %T') $name PSID couples rc=$?"
    $PY scripts/check_optionA2_heldout.py --run_dirs "$@" --shards $MAIN $STRIP \
        --psid "couples, $name=outputs/psid_optionA2n_sens_${name}_couples" \
        --out $OUT/$name/heldout_check.json > logs/optionA2n_sens_${name}_heldout.log 2>&1
    echo "$(date -u '+%F %T') $name held-out check rc=$?"
    $PY scripts/rho_floor_diagnostic.py --out $OUT/$name/rho_floor_diagnostic.json recovery \
        --run_dirs "$@" --region a2n > logs/optionA2n_sens_${name}_recovery.log 2>&1
    echo "$(date -u '+%F %T') $name recovery rc=$?"
}

mkdir -p $OUT/a2n_2m
evaluate a2n_2m outputs/optionA2n/s0 outputs/optionA2n/s1

declare -A ARMS=(
    [cons010]="$(noise 1 consumption=0.10)"
    [cons020]="$(noise 1 consumption=0.20)"
    [all067]="$(noise 0.6666667)"
    [all133]="$(noise 1.3333333)"
)
for arm in cons010 cons020 all067 all133; do
    echo "$(date -u '+%F %T') $arm: observation noise ${ARMS[$arm]}"
    for s in 0 1; do echo "$s"; done |
        xargs -P 2 -L 1 bash -c '"$0" scripts/optionA.py train --seed "$1" \
            --shards "'"$MAIN"'" "'"$STRIP"'" --beta_transform logit --rgamma_transform logit \
            --obs_noise '"${ARMS[$arm]}"' --out "'"$OUT/$arm"'" > "logs/optionA2n_sens_'"$arm"'_s$1.log" 2>&1;
            rc=$?; echo "$(date -u "+%F %T") '"$arm"' seed $1 rc=$rc"' "$PY"
    [ -f "$OUT/$arm/s0/posterior_7w.pt" ] && [ -f "$OUT/$arm/s1/posterior_7w.pt" ] ||
        { echo "$arm: missing checkpoint"; continue; }
    evaluate $arm $OUT/$arm/s0 $OUT/$arm/s1
done
echo "$(date -u '+%F %T') done"
