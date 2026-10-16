#!/usr/bin/env bash
# RESULTS 55: regenerate option A2's data with the simulation audit's fixes
# (RESULTS 54), then train and evaluate A2n's model on it, unattended.
#
# The fixes, all recorded in each shard directory's solver_config.json:
#   - income rounded to whole dollars when planning, as it is realised (H7);
#   - MATLAB's infinities: u(0) = -inf for rho >= 1, impossible choices
#     u - 1e100, -inf for dying in debt (the port used finite sentinels);
#   - MATLAB's tie order among exactly tied choices (lowest Z', then X');
#   - R_gamma per block shuffled across blocks (H8, --rgamma_permute);
#   - flows read a year before stocks, PSID's timing (H1, --flow_lag 1).
# Everything else is v1's: the same theta sequences, card draws, seeds, seed
# pool, full grid, V100 with theta_batch 16 and chunk 16, M = 16, stored
# panels, so v2 pairs with v1 draw by draw (R_gamma re-assigned by block).
#
# Order: the SBC and penalty sets first (~7 h), checked by the audit's
# integrity and invariants before the main run (~4.85 d) and the strip
# (~11 h); everything checked again; then the noise recalibration, five
# members, PSID posteriors, held-out and SBC checks, figures 66-69 and the
# comparison with A2n. Each stage appends to outputs/regen_v2/STATUS; a failed
# stage writes FAILED and stops the chain. Rerunning the same command resumes:
# finished shards and checkpoints are skipped.
#
# Launch:  nohup setsid ./scripts/run_regen_v2.sh >> logs/regen_v2.log 2>&1 &
# Smoke:   SMOKE=<dir> ./scripts/run_regen_v2.sh   (every stage on a few shards
#          of 32 draws, 1 member, 1 epoch; all outputs under <dir>)
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
POOL=data/processed/seed_pool_couples_income.npz
SMOKE=${SMOKE:-}

if [ -n "$SMOKE" ]; then
    DATA=$SMOKE/data; OUT=$SMOKE/outputs/optionA2v2; PSIDO=$SMOKE/outputs
    FIG=$SMOKE/figures; ST=$SMOKE/outputs/regen_v2; LOGD=$SMOKE/logs
    BLOCK=32; MAIN_MAX="--max_draws 96"; STRIP_MAX="--max_draws 64"
    SBC_MAX="--max_draws 32"; PEN_MAX="--max_draws 32"
    SEEDS="0"; TRAIN_EXTRA="--max_epochs 1"; DRAWS="--n_draws 500"
    CHECK_EXTRA="--data_root $DATA --partial"; TAG=_smoke; MIN_KB=0; EVERY="--every 1"
    FIGS_EXTRA="--pool 20 --pool_draws 500 --n_raw 2000 --n_plot 1000"
else
    DATA=data/processed; OUT=outputs/optionA2v2; PSIDO=outputs
    FIG=figures; ST=outputs/regen_v2; LOGD=logs
    BLOCK=512; MAIN_MAX=""; STRIP_MAX=""; SBC_MAX=""; PEN_MAX="--max_draws 512"
    SEEDS="0 1 2 3 4"; TRAIN_EXTRA=""; DRAWS=""
    CHECK_EXTRA=""; TAG=""; MIN_KB=$((3 * 1024 * 1024)); EVERY=""
    FIGS_EXTRA=""
fi
MAIN=$DATA/couples_v2_dataset_shards
STRIP=$DATA/couples_v2_strip_dataset_shards
SBC=$DATA/optionA2v2_sbc16_shards
mkdir -p "$DATA" "$OUT" "$FIG" "$ST" "$LOGD"

status() { echo "$(date -u '+%F %T') $*" | tee -a "$ST/STATUS"; }
fail() { status "FAILED $1"; exit 1; }
run() {   # run STAGE LOG command...
    local name=$1 log=$2; shift 2
    status "start $name"
    "$@" > "$LOGD/$log.log" 2>&1
    local rc=$?   # captured before anything else can reset it
    [ $rc -eq 0 ] || fail "$name (rc=$rc, see $LOGD/$log.log)"
    status "done $name"
}
gen() {   # gen OUT_NAME [args...]: v2 generation with the common settings
    local out=$1; shift
    $PY scripts/generate_dataset.py --simulator twoasset --grid full --device cuda \
        --block "$BLOCK" --theta_batch 16 --chunk 16 --n_households 16 --card_types \
        --init_pool "$POOL" --n_waves 7 --wave_years 2 --start_age 30 \
        --rgamma_permute --flow_lag 1 "$@" --out "$DATA/$out.pt"
}
check() {   # check NAME GROUP: the audit's integrity and invariants, strict
    run "integrity $1" "regen_v2_integrity_$1" $PY scripts/audit_simulation.py integrity \
        --datasets "$2" --tag "${TAG}_$1" --strict $CHECK_EXTRA
    run "invariants $1" "regen_v2_invariants_$1" $PY scripts/audit_simulation.py invariants \
        --datasets "$2" --tag "${TAG}_$1" --strict $CHECK_EXTRA
}

status "chain pid $$ (smoke: ${SMOKE:-no})"
[ -f "$POOL" ] || fail "missing $POOL"
avail=$(df --output=avail -k "$DATA" | tail -1)
[ "$avail" -ge "$MIN_KB" ] || fail "disk: ${avail} kB free, need ${MIN_KB}"

# --- 1. SBC and penalty sets, then the early check ---------------------------
SBC_ARGS=(--n_samples 1024 --proposal uniform --rgamma_range 1.021 1.075 --seed 40000000)
run "generate SBC set" regen_v2_gen_sbc gen optionA2v2_sbc16 "${SBC_ARGS[@]}" $SBC_MAX
run "generate penalty x0.5" regen_v2_gen_liqpen050 gen optionA2v2_sbc16_liqpen050 \
    "${SBC_ARGS[@]}" --liqpen_scale 0.5 $PEN_MAX
run "generate penalty x1.5" regen_v2_gen_liqpen150 gen optionA2v2_sbc16_liqpen150 \
    "${SBC_ARGS[@]}" --liqpen_scale 1.5 $PEN_MAX
check sbc v2_sbc

# --- 2. main run and strip, then everything checked ------------------------
run "generate main run" regen_v2_gen_main gen couples_v2_dataset \
    --n_samples 32768 --proposal a2_switched --rgamma_range 1.025 1.075 --seed 0 $MAIN_MAX
run "generate strip" regen_v2_gen_strip gen couples_v2_strip_dataset \
    --n_samples 3072 --proposal a2_strip --rgamma_range 1.021 1.025 --seed 100000 $STRIP_MAX
check all v2

# --- 3. noise recalibrated on v2, then five members as A2n ------------------
CAL=$OUT/noise_calibration.json
if [ ! -f "$CAL" ]; then
    run "noise calibration" regen_v2_noise $PY scripts/noise_calibration.py \
        --shards "$MAIN" "$STRIP" --out "$CAL" $EVERY
fi
NOISE=$($PY -c "import json; d = json.load(open('$CAL'))['sigma_range']; \
print(' '.join(f'{k}={lo}:{hi}' for k, (lo, hi) in d.items()))") || fail "reading $CAL"
status "observation noise: $NOISE"
status "start training"
for s in $SEEDS; do [ -f "$OUT/s$s/posterior_7w.pt" ] || echo "$s"; done |
    xargs -r -P 2 -I{} bash -c '"$0" scripts/optionA.py train --seed {} \
        --shards "$1" "$2" --beta_transform logit --rgamma_transform logit \
        --obs_noise $3 --out "$4" $5 > "$6/regen_v2_train_s{}.log" 2>&1' \
        "$PY" "$MAIN" "$STRIP" "$NOISE" "$OUT" "$TRAIN_EXTRA" "$LOGD"
rc=$?
DIRS=""
for s in $SEEDS; do
    [ -f "$OUT/s$s/posterior_7w.pt" ] || fail "training: seed $s has no checkpoint (xargs rc=$rc)"
    DIRS="$DIRS $OUT/s$s"
done
status "done training"

# --- 4. PSID posteriors and the checks ---------------------------------------
run "PSID couples" regen_v2_psid_couples $PY scripts/optionA.py psid --run_dirs $DIRS \
    --x data/processed/psid_x_comphs_couples.pt --out "$PSIDO/psid_optionA2v2_couples" $DRAWS
run "PSID all 889" regen_v2_psid_all $PY scripts/optionA.py psid --run_dirs $DIRS \
    --x data/processed/psid_x_comphs_optionA.pt --out "$PSIDO/psid_optionA2v2_all" $DRAWS
run "held-out check" regen_v2_heldout $PY scripts/check_optionA2_heldout.py --run_dirs $DIRS \
    --shards "$MAIN" "$STRIP" \
    --psid "couples, v2=$PSIDO/psid_optionA2v2_couples" "couples, A2n=outputs/psid_optionA2n_couples" \
           "all 889, v2=$PSIDO/psid_optionA2v2_all" "all 889, A2n=outputs/psid_optionA2n_all" \
    --out "$OUT/heldout_check.json" $DRAWS
run "SBC and penalty check" regen_v2_sbc $PY scripts/sbc_optionA2.py --run_dirs $DIRS \
    --shards "$SBC" --train "$MAIN" \
    --penalty "0.5=$DATA/optionA2v2_sbc16_liqpen050_shards" "1.5=$DATA/optionA2v2_sbc16_liqpen150_shards" \
    --out "$OUT/sbc" --figure "$FIG/69_sbc_optionA2v2.png" $DRAWS

# --- 5. figures and the comparison with A2n ----------------------------------
run "household figures" regen_v2_household_figures $PY scripts/plot_optionA2_households.py \
    --run_dirs $DIRS --shards "$MAIN" "$STRIP" --psid_out "$PSIDO/psid_optionA2v2_couples" \
    --x data/processed/psid_x_comphs_couples.pt --fig_dir "$FIG" $FIGS_EXTRA \
    --label "option A2v2 (A2n's model on the regenerated data, RESULTS 55)" \
    --sim_name 66_simulated_household_optionA2v2.png --psid_name 67_psid_household_optionA2v2.png
run "population figure" regen_v2_psid_figure $PY scripts/plot_optionA2_psid.py \
    --series "v2, couples=$PSIDO/psid_optionA2v2_couples" "v2, all 889=$PSIDO/psid_optionA2v2_all" \
             "A2n, couples=outputs/psid_optionA2n_couples" \
    --out "$FIG/68_optionA2v2_psid.png" \
    --title "Per-household posterior means, PSID comphs: option A2v2 (regenerated data, RESULTS 55) against A2n"
run "comparison with A2n" regen_v2_compare $PY scripts/compare_regen_v2.py \
    --pair couples outputs/psid_optionA2n_couples "$PSIDO/psid_optionA2v2_couples" \
    --pair all outputs/psid_optionA2n_all "$PSIDO/psid_optionA2v2_all" \
    --out "$OUT/comparison.json"
status "DONE"
