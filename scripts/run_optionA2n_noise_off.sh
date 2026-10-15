#!/usr/bin/env bash
# RESULTS 52-53: how much of A2n's posterior width one feature's observation
# noise costs. A2n with FEATURE's noise off and everything else unchanged
# (the other ranges as calibrated in outputs/optionA2n/noise_calibration.json,
# RESULTS 43.9). add_obs_noise draws every feature's z and sigma whatever its
# range, so the other features' noise is draw for draw A2n's. Two members,
# seeds 0 and 1, compared with A2n's members 0 and 1 (RESULTS 43.10's
# reference, outputs/optionA2n_sens/a2n_2m).
#
#   FEATURE          arm      RESULTS  figures
#   illiquid_assets  illoff   52       61 (households), 62 (PSID couples)
#   consumption      consoff  53       63 (households), 64 (PSID couples)
#
# Steps: train; PSID couples, the held-out check (held-out windows carry the
# arm's noise) and the known-rho recovery, as RESULTS 43.10's evaluate();
# then the same 1,024 held-out households through A2w, the arm and A2n at two
# members each, and the arm on A2n's windows (compare_optionA2_widths.py
# --noise_off); the PSID figure; and the noise-arm table
# (optionA2n_noise_sensitivity.py).
#
# Launch:  nohup setsid ./scripts/run_optionA2n_noise_off.sh consumption >> logs/optionA2n_consoff.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
FEATURE=${1:?usage: run_optionA2n_noise_off.sh illiquid_assets|consumption}
case $FEATURE in
    illiquid_assets) ARM=illoff;  LABEL="illiquid noise off"; FIG=62 ;;
    consumption)     ARM=consoff; LABEL="consumption noise off"; FIG=64 ;;
    *) echo "no arm for $FEATURE"; exit 1 ;;
esac
MAIN=data/processed/couples_dataset_shards
STRIP=data/processed/couples_strip_dataset_shards
CAL=outputs/optionA2n/noise_calibration.json
OUT=outputs/optionA2n_sens/$ARM
PSID_OUT=outputs/psid_optionA2n_sens_${ARM}_couples
COUPLES=data/processed/psid_x_comphs_couples.pt
NOISE=$($PY -c "import json; d = json.load(open('$CAL'))['sigma_range']; d['$FEATURE'] = [0.0, 0.0]; \
print(' '.join(f'{k}={lo}:{hi}' for k, (lo, hi) in d.items()))")
mkdir -p logs "$OUT"

echo "$(date -u '+%F %T') $ARM: observation noise $NOISE"
for s in 0 1; do echo "$s"; done |
    xargs -P 2 -L 1 bash -c '"$0" scripts/optionA.py train --seed "$1" \
        --shards "'"$MAIN"'" "'"$STRIP"'" --beta_transform logit --rgamma_transform logit \
        --obs_noise '"$NOISE"' --out "'"$OUT"'" > "logs/optionA2n_'"$ARM"'_s$1.log" 2>&1;
        rc=$?; echo "$(date -u "+%F %T") '"$ARM"' seed $1 rc=$rc"' "$PY"
[ -f "$OUT/s0/posterior_7w.pt" ] && [ -f "$OUT/s1/posterior_7w.pt" ] ||
    { echo "$ARM: missing checkpoint"; exit 1; }

$PY scripts/optionA.py psid --run_dirs $OUT/s0 $OUT/s1 --x $COUPLES \
    --out $PSID_OUT > logs/optionA2n_${ARM}_psid.log 2>&1
echo "$(date -u '+%F %T') $ARM PSID couples rc=$?"
$PY scripts/check_optionA2_heldout.py --run_dirs $OUT/s0 $OUT/s1 --shards $MAIN $STRIP \
    --psid "couples, $ARM=$PSID_OUT" \
    --out $OUT/heldout_check.json > logs/optionA2n_${ARM}_heldout.log 2>&1
echo "$(date -u '+%F %T') $ARM held-out check rc=$?"
$PY scripts/rho_floor_diagnostic.py --out $OUT/rho_floor_diagnostic.json recovery \
    --run_dirs $OUT/s0 $OUT/s1 --region a2n > logs/optionA2n_${ARM}_recovery.log 2>&1
echo "$(date -u '+%F %T') $ARM recovery rc=$?"
$PY scripts/compare_optionA2_widths.py --noise_off "$FEATURE=$OUT" \
    > logs/compare_optionA2_widths_${ARM}.log 2>&1
echo "$(date -u '+%F %T') same-household widths rc=$?"
$PY scripts/plot_optionA2_psid.py \
    --series "$LABEL, couples (2 members)=$PSID_OUT" \
             "A2n, couples (members 0, 1)=outputs/psid_optionA2n_sens_a2n_2m_couples" \
    --out figures/${FIG}_optionA2n_${ARM}_psid.png \
    --title "Per-household posterior means, PSID couples: A2n with the $LABEL against A2n (members 0 and 1 each)" \
    > logs/optionA2n_${ARM}_psid_figure.log 2>&1
echo "$(date -u '+%F %T') PSID figure rc=$?"
$PY scripts/optionA2n_noise_sensitivity.py > logs/optionA2n_${ARM}_summary.log 2>&1
echo "$(date -u '+%F %T') summary rc=$?"
echo "$(date -u '+%F %T') done"
