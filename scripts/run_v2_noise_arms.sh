#!/usr/bin/env bash
# RESULTS 58: v2's model retrained with the PSID-only noise ranges of
# RESULTS 57.4, on v2's data, two arms:
#   typical  - the IQR-based range, a typical record's error;
#   gross    - the winsorised range, which also counts gross errors.
# Everything else is v2's (run_regen_v2.sh): main run + strip, logit beta and
# R_gamma, five members (seeds 0-4), two at a time. Each arm then gets v2's
# evaluation: PSID posteriors (couples and all 889), the held-out check and
# SBC with the penalty sets (each on windows carrying the arm's own noise),
# the household contour figures, and the three-way comparison with v2.
# Each stage appends to outputs/v2_noise_arms/STATUS; a failed stage writes
# FAILED and stops. Rerunning the same command resumes (finished members and
# outputs are skipped).
#
# Launch:  nohup setsid ./scripts/run_v2_noise_arms.sh >> logs/v2_noise_arms.log 2>&1 &
# Smoke:   SMOKE=<dir> ./scripts/run_v2_noise_arms.sh   (1 member, 1 epoch,
#          500 draws; every stage, all outputs under <dir>)
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
DATA=data/processed
MAIN=$DATA/couples_v2_dataset_shards
STRIP=$DATA/couples_v2_strip_dataset_shards
SBC=$DATA/optionA2v2_sbc16_shards
RANGES=outputs/psid_iid_error/psid_iid_error.json
V2=outputs/optionA2v2
SMOKE=${SMOKE:-}

if [ -n "$SMOKE" ]; then
    OUTB=$SMOKE/outputs; FIG=$SMOKE/figures; ST=$SMOKE/outputs/v2_noise_arms; LOGD=$SMOKE/logs
    SEEDS="0"; TRAIN_EXTRA="--max_epochs 1"; DRAWS="--n_draws 500"
    FIGS_EXTRA="--pool 20 --pool_draws 500 --n_raw 2000 --n_plot 1000"
else
    OUTB=outputs; FIG=figures; ST=outputs/v2_noise_arms; LOGD=logs
    SEEDS="0 1 2 3 4"; TRAIN_EXTRA=""; DRAWS=""; FIGS_EXTRA=""
fi
mkdir -p "$OUTB" "$FIG" "$ST" "$LOGD"
# arm name, key in sigma_range_network, figure numbers (simulated, PSID, SBC)
ARMS=("typical typical 72 73 74" "gross with_gross_errors 75 76 77")

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
noise_of() {   # noise_of KEY: --obs_noise arguments from RESULTS 57.4's range
    $PY -c "import json; d = json.load(open('$RANGES'))['sigma_range_network']['$1']; \
print(' '.join(f'{k}={lo}:{hi}' for k, (lo, hi) in d.items()))"
}

status "chain pid $$ (smoke: ${SMOKE:-no})"
[ -f "$RANGES" ] || fail "missing $RANGES (scripts/psid_iid_error.py)"
for d in "$MAIN" "$STRIP" "$SBC"; do [ -d "$d" ] || fail "missing $d"; done

# --- 1. ten members, two at a time ------------------------------------------------
JOBS=""
for spec in "${ARMS[@]}"; do
    read -r arm key _ _ _ <<< "$spec"
    NOISE=$(noise_of "$key") || fail "reading $RANGES ($key)"
    status "arm $arm: observation noise $NOISE"
    for s in $SEEDS; do
        out=$OUTB/optionA2v2_noise_$arm
        [ -f "$out/s$s/posterior_7w.pt" ] || JOBS="$JOBS$arm|$s|$NOISE|$out\n"
    done
done
status "start training"
printf "$JOBS" | xargs -r -P 2 -I{} bash -c 'IFS="|" read -r arm s noise out <<< "$1"; \
    "$0" scripts/optionA.py train --seed "$s" --shards "$2" "$3" \
    --beta_transform logit --rgamma_transform logit --obs_noise $noise --out "$out" $4 \
    > "$5/v2_noise_${arm}_train_s$s.log" 2>&1' "$PY" {} "$MAIN" "$STRIP" "$TRAIN_EXTRA" "$LOGD"
rc=$?
for spec in "${ARMS[@]}"; do
    read -r arm _ _ _ _ <<< "$spec"
    for s in $SEEDS; do
        [ -f "$OUTB/optionA2v2_noise_$arm/s$s/posterior_7w.pt" ] ||
            fail "training: $arm seed $s has no checkpoint (xargs rc=$rc)"
    done
done
status "done training"

# --- 2. each arm: PSID, held-out, SBC, figures -----------------------------------
for spec in "${ARMS[@]}"; do
    read -r arm _ fsim fpsid fsbc <<< "$spec"
    OUT=$OUTB/optionA2v2_noise_$arm
    PC=$OUTB/psid_optionA2v2_noise_${arm}_couples
    PA=$OUTB/psid_optionA2v2_noise_${arm}_all
    DIRS=""
    for s in $SEEDS; do DIRS="$DIRS $OUT/s$s"; done
    run "$arm: PSID couples" "v2_noise_${arm}_psid_couples" $PY scripts/optionA.py psid \
        --run_dirs $DIRS --x data/processed/psid_x_comphs_couples.pt --out "$PC" $DRAWS
    run "$arm: PSID all 889" "v2_noise_${arm}_psid_all" $PY scripts/optionA.py psid \
        --run_dirs $DIRS --x data/processed/psid_x_comphs_optionA.pt --out "$PA" $DRAWS
    run "$arm: held-out check" "v2_noise_${arm}_heldout" $PY scripts/check_optionA2_heldout.py \
        --run_dirs $DIRS --shards "$MAIN" "$STRIP" \
        --psid "couples, $arm=$PC" "couples, v2=outputs/psid_optionA2v2_couples" \
               "all 889, $arm=$PA" "all 889, v2=outputs/psid_optionA2v2_all" \
        --out "$OUT/heldout_check.json" $DRAWS
    run "$arm: SBC and penalty check" "v2_noise_${arm}_sbc" $PY scripts/sbc_optionA2.py \
        --run_dirs $DIRS --shards "$SBC" --train "$MAIN" \
        --penalty "0.5=$DATA/optionA2v2_sbc16_liqpen050_shards" \
                  "1.5=$DATA/optionA2v2_sbc16_liqpen150_shards" \
        --out "$OUT/sbc" --figure "$FIG/${fsbc}_sbc_optionA2v2_noise_$arm.png" $DRAWS
    run "$arm: household figures" "v2_noise_${arm}_household_figures" \
        $PY scripts/plot_optionA2_households.py --run_dirs $DIRS --shards "$MAIN" "$STRIP" \
        --psid_out "$PC" --x data/processed/psid_x_comphs_couples.pt --fig_dir "$FIG" $FIGS_EXTRA \
        --label "option A2v2, PSID-only noise ($arm range, RESULTS 57.4)" \
        --sim_name "${fsim}_simulated_household_optionA2v2_noise_$arm.png" \
        --psid_name "${fpsid}_psid_household_optionA2v2_noise_$arm.png"
done

# --- 3. the three-way comparison -----------------------------------------------------
run "population figure" v2_noise_psid_figure $PY scripts/plot_optionA2_psid.py \
    --series "v2, couples=outputs/psid_optionA2v2_couples" \
             "typical, couples=$OUTB/psid_optionA2v2_noise_typical_couples" \
             "gross, couples=$OUTB/psid_optionA2v2_noise_gross_couples" \
    --out "$FIG/78_optionA2v2_noise_arms_psid.png" \
    --title "PSID couples' posterior means: v2 against the PSID-only noise ranges (RESULTS 57.4)"
# the seed baseline: PSID couples from two disjoint member pairs of each arm
SEEDP=()
if [ -z "$SMOKE" ]; then
    for spec in "v2 $V2" "typical $OUTB/optionA2v2_noise_typical" "gross $OUTB/optionA2v2_noise_gross"; do
        read -r arm dir <<< "$spec"
        for pair in "0 1" "2 3"; do
            read -r a b <<< "$pair"
            run "$arm: PSID couples, members $a+$b" "v2_noise_seedpair_${arm}_s$a$b" \
                $PY scripts/optionA.py psid --run_dirs "$dir/s$a" "$dir/s$b" \
                --x data/processed/psid_x_comphs_couples.pt --out "$ST/seed_pairs/${arm}_s$a$b"
        done
        SEEDP+=(--seed_pair "$arm" "$ST/seed_pairs/${arm}_s01" "$ST/seed_pairs/${arm}_s23")
    done
fi
run "comparison" v2_noise_compare $PY scripts/compare_noise_arms.py \
    --arm v2 "$V2" outputs/psid_optionA2v2_couples outputs/psid_optionA2v2_all \
    --arm typical "$OUTB/optionA2v2_noise_typical" "$OUTB/psid_optionA2v2_noise_typical_couples" \
        "$OUTB/psid_optionA2v2_noise_typical_all" \
    --arm gross "$OUTB/optionA2v2_noise_gross" "$OUTB/psid_optionA2v2_noise_gross_couples" \
        "$OUTB/psid_optionA2v2_noise_gross_all" \
    "${SEEDP[@]}" --out "$ST/comparison.json"
# every network on every arm's noise (the same held-out windows in each cell)
[ -n "$SMOKE" ] || run "cross-noise held-out" v2_noise_cross_heldout \
    $PY scripts/cross_noise_heldout.py --arm v2 "$V2" \
    --arm typical "$OUTB/optionA2v2_noise_typical" --arm gross "$OUTB/optionA2v2_noise_gross" \
    --shards "$MAIN" "$STRIP" --out "$ST/cross_noise_heldout.json"
status "DONE"
