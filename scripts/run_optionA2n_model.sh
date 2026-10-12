#!/usr/bin/env bash
# Option A2n (RESULTS 43.7): A2w (main run + R_gamma strip, logit beta and
# R_gamma) trained on simulated windows carrying observation noise. Each
# household's sigma per dollar feature is uniform between the minimum PSID
# couples require and the typical requirement over couples' region
# (noise_calibration.py's sigma_range), so the network marginalises over it.
# Retraining only; nothing is simulated. PSID gets no added noise.
#
# Five members two at a time; PSID posteriors for couples and all 889; the
# held-out check (held-out windows carry the same noise); the rho-floor
# ablation on the new ensemble; figures 42-43 (one held-out simulated
# household, the typical PSID couple) and 44 (couples and all 889 against A2w).
#
# Launch:  nohup setsid ./scripts/run_optionA2n_model.sh >> logs/optionA2n_model.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
MAIN=data/processed/couples_dataset_shards
STRIP=data/processed/couples_strip_dataset_shards
OUT=outputs/optionA2n
CAL=$OUT/noise_calibration.json
mkdir -p logs "$OUT"

[ -f "$CAL" ] || { echo "$(date -u '+%F %T') no $CAL; run noise_calibration.py first"; exit 1; }
NOISE=$($PY -c "import json; d = json.load(open('$CAL'))['sigma_range']; \
print(' '.join(f'{k}={lo}:{hi}' for k, (lo, hi) in d.items()))")
echo "$(date -u '+%F %T') observation noise: $NOISE"

echo "$(date -u '+%F %T') training 5 members, 2 at a time"
for s in 0 1 2 3 4; do echo "$s"; done |
    xargs -P 2 -L 1 bash -c '"$0" scripts/optionA.py train --seed "$1" \
        --shards "'"$MAIN"'" "'"$STRIP"'" --beta_transform logit --rgamma_transform logit \
        --obs_noise '"$NOISE"' --out "'"$OUT"'" > "logs/optionA2n_train_s$1.log" 2>&1;
        rc=$?; echo "$(date -u "+%F %T") seed $1 rc=$rc"' "$PY"
DIRS=""
for s in 0 1 2 3 4; do
    [ -f "$OUT/s$s/posterior_7w.pt" ] || { echo "seed $s has no checkpoint"; exit 1; }
    DIRS="$DIRS $OUT/s$s"
done

$PY scripts/optionA.py psid --run_dirs $DIRS --x data/processed/psid_x_comphs_couples.pt \
    --out outputs/psid_optionA2n_couples > logs/optionA2n_psid_couples.log 2>&1
rc=$?; echo "$(date -u '+%F %T') PSID couples rc=$rc"
$PY scripts/optionA.py psid --run_dirs $DIRS --x data/processed/psid_x_comphs_optionA.pt \
    --out outputs/psid_optionA2n_all > logs/optionA2n_psid_all.log 2>&1
rc=$?; echo "$(date -u '+%F %T') PSID all 889 rc=$rc"

$PY scripts/check_optionA2_heldout.py --run_dirs $DIRS --shards $MAIN $STRIP \
    --psid "couples, A2n=outputs/psid_optionA2n_couples" "couples, A2w=outputs/psid_optionA2w_couples" \
           "all 889, A2n=outputs/psid_optionA2n_all" "all 889, A2w=outputs/psid_optionA2w_all" \
    --out $OUT/heldout_check.json > logs/optionA2n_heldout.log 2>&1
rc=$?; echo "$(date -u '+%F %T') held-out check rc=$rc"

$PY scripts/rho_floor_diagnostic.py --out $OUT/rho_floor_diagnostic.json ablate \
    --run_dirs $DIRS --device cuda > logs/optionA2n_rho_ablate.log 2>&1
rc=$?; echo "$(date -u '+%F %T') rho-floor ablation rc=$rc"

$PY scripts/plot_optionA2_households.py --run_dirs $DIRS --shards $MAIN $STRIP \
    --psid_out outputs/psid_optionA2n_couples --x data/processed/psid_x_comphs_couples.pt \
    --label "option A2n (A2w + observation noise, sigma per household; 5 members)" \
    --sim_name 42_simulated_household_optionA2n.png --psid_name 43_psid_household_optionA2n.png \
    > logs/optionA2n_household_figures.log 2>&1
rc=$?; echo "$(date -u '+%F %T') household figures rc=$rc"
$PY scripts/plot_optionA2_psid.py \
    --series "A2n, couples (5 members)=outputs/psid_optionA2n_couples" \
             "A2n, all 889 (5 members)=outputs/psid_optionA2n_all" \
             "A2w, couples (no noise)=outputs/psid_optionA2w_couples" \
    --out figures/44_optionA2n_psid.png \
    --title "Per-household posterior means, PSID comphs: option A2n (simulated windows with calibrated observation noise) against A2w" \
    > logs/optionA2n_psid_figure.log 2>&1
rc=$?; echo "$(date -u '+%F %T') population figure rc=$rc"
