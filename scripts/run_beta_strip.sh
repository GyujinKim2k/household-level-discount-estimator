#!/usr/bin/env bash
# RESULTS 63: a beta strip below the A2 box, beta on [0.10, 0.30), so PSID
# couples' posteriors and the population fits (RESULTS 61-62) can go below 0.3.
# A strip, as RESULTS 43.5's R_gamma strip: the v2 main run and R_gamma strip
# are reused unchanged, and training weights the three runs back to the
# uniform prior on [0.10, 1] x ... x [1.021, 1.075] (prior._parts_log_weight).
#   1. the SBC strip (288 uniform draws on the strip), checked;
#   2. the beta strip (6,144 draws, v2's mixture on the strip), checked, and
#      the three training runs' weights against the uniform prior;
#   3. training, two members at a time: v2 (its noise), typical and gross
#      (RESULTS 58's ranges), noise-unknown (RESULTS 62), five members each;
#      and v2 members 0-1 on half of every run's training shards (the
#      learning curve);
#   4. each arm: PSID couples and all 889, held-out check, SBC on the set plus
#      the strip, household figures (68/95% contours); unknown: SBC and PSID;
#   5. the learning curve: half- and full-data members on one held-out set;
#   6. the population fits (RESULTS 61's indirect inference), then the
#      noise-unknown arm's eight-unknown fit.
# Every setting but the beta range is v2's (run_regen_v2.sh). Each stage
# appends to outputs/beta_strip/STATUS; a failure writes FAILED and stops.
# Rerunning the same command resumes: finished shards, members and outputs
# are skipped.
#
# Launch:  nohup setsid ./scripts/run_beta_strip.sh >> logs/beta_strip.log 2>&1 &
# Smoke:   SMOKE=<dir> ./scripts/run_beta_strip.sh   (64 strip and 32 SBC-strip
#          draws, 1 member per arm, 1 epoch on one shard per run, few draws;
#          every stage, all new outputs under <dir>)
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
POOL=data/processed/seed_pool_couples_income.npz
RANGES=outputs/psid_iid_error/psid_iid_error.json
CAL=outputs/optionA2v2/noise_calibration.json
SMOKE=${SMOKE:-}
DATA=data/processed
MAIN=$DATA/couples_v2_dataset_shards
STRIP=$DATA/couples_v2_strip_dataset_shards
SBC=$DATA/optionA2v2_sbc16_shards
LC_SHARED=outputs/laibson_comparison

if [ -n "$SMOKE" ]; then
    NEW=$SMOKE/data; OUTB=$SMOKE/outputs; FIG=$SMOKE/figures; ST=$SMOKE/outputs/beta_strip
    LOGD=$SMOKE/logs; BLOCK=32; SBCS_BLOCK=32
    BSTRIP_MAX="--max_draws 64"; SBCS_MAX="--max_draws 32"
    SEEDS="0"; HALF_SEEDS="0"; TRAIN_EXTRA="--max_epochs 1 --train_shards 1"
    DRAWS="--n_draws 500"; CHECK_EXTRA="--data_root $NEW --partial"; TAG=_smoke; MIN_KB=0
    FIGS_EXTRA="--pool 20 --pool_draws 500 --n_raw 2000 --n_plot 1000"
    LC_CACHE=$SMOKE/laibson_comparison; KNOWN=""
else
    NEW=$DATA; OUTB=outputs; FIG=figures; ST=outputs/beta_strip; LOGD=logs
    BLOCK=512; SBCS_BLOCK=96; BSTRIP_MAX=""; SBCS_MAX=""
    SEEDS="0 1 2 3 4"; HALF_SEEDS="0 1"; TRAIN_EXTRA=""; DRAWS=""; CHECK_EXTRA=""; TAG=""
    MIN_KB=$((3 * 512 * 1024)); FIGS_EXTRA=""
    LC_CACHE=$LC_SHARED; KNOWN="v2_b010 gross_b010"
fi
BSTRIP=$NEW/couples_v2_betastrip_dataset_shards
SBCS=$NEW/optionA2v2_sbc16_betastrip_shards
V2=$OUTB/optionA2v2_b010
HALF=$OUTB/optionA2v2_b010_half
UNK=$OUTB/optionA2v2_b010_noise_unknown
mkdir -p "$NEW" "$OUTB" "$FIG" "$ST" "$LOGD" "$LC_CACHE"

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
        --theta_batch 16 --chunk 16 --n_households 16 --card_types \
        --init_pool "$POOL" --n_waves 7 --wave_years 2 --start_age 30 \
        --rgamma_permute --flow_lag 1 "$@" --out "$NEW/$out.pt"
}
check() {   # check NAME DATASET: the audit's integrity and invariants, strict
    run "integrity $1" "beta_strip_integrity_$1" $PY scripts/audit_simulation.py integrity \
        --datasets "$2" --tag "_betastrip${TAG}_$1" --strict $CHECK_EXTRA
    run "invariants $1" "beta_strip_invariants_$1" $PY scripts/audit_simulation.py invariants \
        --datasets "$2" --tag "_betastrip${TAG}_$1" --strict $CHECK_EXTRA
}
noise_of() {   # noise_of KEY: --obs_noise arguments from RESULTS 57.4's range
    $PY -c "import json; d = json.load(open('$RANGES'))['sigma_range_network']['$1']; \
print(' '.join(f'{k}={lo}:{hi}' for k, (lo, hi) in d.items()))"
}
dirs_of() {   # dirs_of RUN SEEDS...: the member directories
    local r=$1; shift
    for s in "$@"; do printf ' %s/s%s' "$r" "$s"; done
}

status "chain pid $$ (smoke: ${SMOKE:-no})"
for f in "$POOL" "$RANGES" "$CAL"; do [ -f "$f" ] || fail "missing $f"; done
for d in "$MAIN" "$STRIP" "$SBC"; do [ -d "$d" ] || fail "missing $d"; done
avail=$(df --output=avail -k "$DATA" | tail -1)
[ "$avail" -ge "$MIN_KB" ] || fail "disk: ${avail} kB free, need ${MIN_KB}"

# --- 1. the SBC strip, then the early check --------------------------------------
run "generate SBC strip" beta_strip_gen_sbc gen optionA2v2_sbc16_betastrip \
    --n_samples 288 --block "$SBCS_BLOCK" --proposal uniform --beta_range 0.10 0.30 \
    --rgamma_range 1.021 1.075 --seed 41000000 $SBCS_MAX
check sbc sbc16_betastrip_v2

# --- 2. the beta strip, checked, and the three runs' weights ----------------------
run "generate beta strip" beta_strip_gen_strip gen couples_v2_betastrip_dataset \
    --n_samples 6144 --block "$BLOCK" --proposal a2_beta_strip --beta_range 0.10 0.30 \
    --rgamma_range 1.021 1.075 --seed 200000 $BSTRIP_MAX
check strip betastrip_v2
run "verify the three runs" beta_strip_verify $PY scripts/optionA.py verify \
    --shards "$MAIN" "$STRIP" "$BSTRIP" $([ -n "$SMOKE" ] && echo --partial)
[ -n "$SMOKE" ] || run "proposal weights" beta_strip_weights $PY -c "
from scripts.audit_simulation import proposal_check
r = proposal_check(['main_v2', 'strip_v2', 'betastrip_v2'])
assert r['max_abs_z'] < 4, r"

# --- 3. training: 20 members and the 2 half-data members, two at a time -----------
V2_NOISE=$($PY -c "import json; d = json.load(open('$CAL'))['sigma_range']; \
print(' '.join(f'{k}={lo}:{hi}' for k, (lo, hi) in d.items()))") || fail "reading $CAL"
TYP_NOISE=$(noise_of typical) || fail "reading $RANGES (typical)"
GRO_NOISE=$(noise_of with_gross_errors) || fail "reading $RANGES (gross)"
status "noise: v2 $V2_NOISE; typical $TYP_NOISE; gross $GRO_NOISE"
JOBS=""
for s in $SEEDS; do
    [ -f "$V2/s$s/posterior_7w.pt" ] || JOBS="${JOBS}known|$s|$V2|$V2_NOISE|\n"
    for arm in typical gross; do
        out=$OUTB/optionA2v2_b010_noise_$arm
        n=$([ $arm = typical ] && echo "$TYP_NOISE" || echo "$GRO_NOISE")
        [ -f "$out/s$s/posterior_7w.pt" ] || JOBS="${JOBS}known|$s|$out|$n|\n"
    done
    [ -f "$UNK/s$s/posterior_7w.pt" ] || JOBS="${JOBS}unknown|$s|$UNK||\n"
done
for s in $HALF_SEEDS; do
    [ -f "$HALF/s$s/posterior_7w.pt" ] || JOBS="${JOBS}known|$s|$HALF|$V2_NOISE|--train_frac 0.5\n"
done
status "start training ($(printf "$JOBS" | grep -c .) members)"
printf "$JOBS" | xargs -r -P 2 -I{} bash -c '
    IFS="|" read -r kind s out noise extra <<< "$1"
    log="$7/beta_strip_train_$(basename "$out")_s$s.log"
    if [ "$kind" = unknown ]; then
        "$0" scripts/noise_unknown.py train --seed "$s" --shards "$2" "$3" "$4" \
            --out "$out" $5 > "$log" 2>&1
    else
        "$0" scripts/optionA.py train --seed "$s" --shards "$2" "$3" "$4" \
            --beta_transform logit --rgamma_transform logit --obs_noise $noise \
            --out "$out" $5 $extra > "$log" 2>&1
    fi' "$PY" {} "$MAIN" "$STRIP" "$BSTRIP" "$TRAIN_EXTRA" "" "$LOGD"
rc=$?
for s in $SEEDS; do
    for out in "$V2" "$OUTB/optionA2v2_b010_noise_typical" "$OUTB/optionA2v2_b010_noise_gross" "$UNK"; do
        [ -f "$out/s$s/posterior_7w.pt" ] || fail "training: $out seed $s (xargs rc=$rc)"
    done
done
for s in $HALF_SEEDS; do
    [ -f "$HALF/s$s/posterior_7w.pt" ] || fail "training: $HALF seed $s (xargs rc=$rc)"
done
status "done training"

# --- 4. each noise-known arm: PSID, held-out, SBC, figures -------------------------
# arm, run directory, figure numbers (simulated, PSID, SBC), label
ARMS=("v2 $V2 88 89 90 v2's noise"
      "typical $OUTB/optionA2v2_b010_noise_typical 91 92 93 typical PSID-only noise"
      "gross $OUTB/optionA2v2_b010_noise_gross 94 95 96 gross PSID-only noise")
for spec in "${ARMS[@]}"; do
    read -r arm OUT fsim fpsid fsbc label <<< "$spec"
    name=$([ $arm = v2 ] && echo optionA2v2_b010 || echo "optionA2v2_b010_noise_$arm")
    PC=$OUTB/psid_${name}_couples
    PA=$OUTB/psid_${name}_all
    OLD=$([ $arm = v2 ] && echo outputs/psid_optionA2v2 || echo "outputs/psid_optionA2v2_noise_$arm")
    DIRS=$(dirs_of "$OUT" $SEEDS)
    [ -f "$PC/summary.json" ] || run "$arm: PSID couples" "beta_strip_${arm}_psid_couples" \
        $PY scripts/optionA.py psid --run_dirs $DIRS --x $DATA/psid_x_comphs_couples.pt \
        --out "$PC" $DRAWS
    [ -f "$PA/summary.json" ] || run "$arm: PSID all 889" "beta_strip_${arm}_psid_all" \
        $PY scripts/optionA.py psid --run_dirs $DIRS --x $DATA/psid_x_comphs_optionA.pt \
        --out "$PA" $DRAWS
    [ -f "$OUT/heldout_check.json" ] || run "$arm: held-out check" "beta_strip_${arm}_heldout" \
        $PY scripts/check_optionA2_heldout.py --run_dirs $DIRS --shards "$MAIN" "$STRIP" "$BSTRIP" \
        --psid "couples, ${arm}_b010=$PC" "couples, $arm=${OLD}_couples" \
               "all 889, ${arm}_b010=$PA" "all 889, $arm=${OLD}_all" \
        --out "$OUT/heldout_check.json" $DRAWS
    [ -f "$OUT/sbc/sbc16.json" ] || run "$arm: SBC and penalty check" "beta_strip_${arm}_sbc" \
        $PY scripts/sbc_optionA2.py --run_dirs $DIRS --shards "$SBC" "$SBCS" --train "$MAIN" \
        --penalty "0.5=$DATA/optionA2v2_sbc16_liqpen050_shards" \
                  "1.5=$DATA/optionA2v2_sbc16_liqpen150_shards" \
        --out "$OUT/sbc" --figure "$FIG/${fsbc}_sbc_$name.png" $DRAWS
    [ -f "$FIG/${fsim}_simulated_household_$name.png" ] || run "$arm: household figures" \
        "beta_strip_${arm}_household_figures" \
        $PY scripts/plot_optionA2_households.py --run_dirs $DIRS --shards "$MAIN" "$STRIP" "$BSTRIP" \
        --psid_out "$PC" --x $DATA/psid_x_comphs_couples.pt --fig_dir "$FIG" $FIGS_EXTRA \
        --label "option A2v2 with the beta strip (beta 0.1-1), $label (RESULTS 63)" \
        --sim_name "${fsim}_simulated_household_$name.png" \
        --psid_name "${fpsid}_psid_household_$name.png"
done

# --- 4b. the noise-unknown arm: SBC and PSID ---------------------------------------
UNK_PSID=psid_optionA2v2_b010_noise_unknown
[ -f "$UNK/sbc/sbc16.json" ] || run "unknown: SBC" beta_strip_unknown_sbc \
    $PY scripts/noise_unknown.py sbc --run "$UNK" --sbc "$SBC" "$SBCS" --out "$UNK/sbc" \
    --figure "$FIG/97_sbc_optionA2v2_b010_noise_unknown.png" $DRAWS
[ -f "$OUTB/${UNK_PSID}_couples/summary.json" ] || run "unknown: PSID" beta_strip_unknown_psid \
    $PY scripts/noise_unknown.py psid --run "$UNK" --root "$OUTB" --psid_name "$UNK_PSID" \
    --figure "$FIG/98_psid_optionA2v2_b010_noise_unknown.png" $DRAWS

# --- 5. the learning curve: half- and full-data members, one held-out set ----------
for spec in "half $HALF" "full $V2"; do
    read -r part dir <<< "$spec"
    f=$ST/learning_curve_$part.json
    [ -f "$f" ] || run "learning curve: $part data" "beta_strip_learning_$part" \
        $PY scripts/check_optionA2_heldout.py --run_dirs $(dirs_of "$dir" $HALF_SEEDS) \
        --shards "$MAIN" "$STRIP" "$BSTRIP" --psid "couples, v2_b010=$OUTB/psid_optionA2v2_b010_couples" \
        --out "$f" $DRAWS
done

# --- 6. the population fits ----------------------------------------------------------
if [ -n "$SMOKE" ]; then   # the shared points' simulations, so the smoke test skips them
    for f in "$LC_SHARED"/sim_laibson.npz "$LC_SHARED"/sim_plus?.npz "$LC_SHARED"/sim_minus?.npz \
             "$LC_SHARED"/sim_v2_median.npz; do
        [ -f "$LC_CACHE/$(basename "$f")" ] || cp "$f" "$LC_CACHE/"
    done
fi
LCJ=$ST/laibson_comparison_b010.json
[ -f "$LCJ" ] || run "population fits" beta_strip_population $PY scripts/laibson_comparison.py \
    --arms --cache "$LC_CACHE" --out "$LCJ" --figure "$FIG/99_laibson_comparison_b010.png" \
    --arm v2_b010 "$V2" "$OUTB/psid_optionA2v2_b010_couples" \
    --arm typical_b010 "$OUTB/optionA2v2_b010_noise_typical" \
          "$OUTB/psid_optionA2v2_b010_noise_typical_couples" \
    --arm gross_b010 "$OUTB/optionA2v2_b010_noise_gross" \
          "$OUTB/psid_optionA2v2_b010_noise_gross_couples" \
    --title "Laibson et al.'s MSM point against PSID couples' representative θ, the box with \
the β strip down to 0.1 (RESULTS 63)"
[ -f "$OUTB/noise_unknown_b010/population.json" ] || run "unknown: population" \
    beta_strip_unknown_population $PY scripts/noise_unknown.py population --run "$UNK" \
    --tag unknown_b010 --root "$OUTB" --psid_name "$UNK_PSID" --known $KNOWN \
    --out "$OUTB/noise_unknown_b010" --sim_cache "$LC_CACHE" --with_arms "$LCJ" \
    --figure "$FIG/100_population_noise_arms_b010.png" \
    --title "Laibson et al.'s MSM point against PSID couples' representative θ under each \
noise arm, the box with the β strip down to 0.1 (RESULTS 63; unknown: θ and noise fitted together)"
status "DONE"
