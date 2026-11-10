#!/usr/bin/env bash
# RESULTS 67: noisy copies of clean households, and the methodology file's noise
# (the user's choices, 2026-11-09). Three 7-wave arms on w07's window (biennial,
# start 24-51; everything else as RESULTS 64's w07), five members each:
#   v2_k8    v2's noise, K = 8 noisy copies of every clean training window
#   file_k1  the methodology file's noise (scripts/psid_noise.py), one draw
#   file_k8  the file's noise, K = 8 copies
# against w07 (RESULTS 64: v2's noise, one draw).
#   0. the file's noise calibrated on PSID couples (psid_noise.py calibrate, check);
#      the baselines' training CSVs backfilled (training_metrics.py backfill);
#   1. training, two members at a time; every member writes its per-epoch CSV to
#      training_metrics/noise_copies/<arm>/w07/s<seed>.csv as it trains;
#   2. each arm: PSID couples and all 889, held-out check, SBC (figures 126-128);
#   3. the population at 409 households sharing theta0 (RESULTS 64's simulations);
#   4. cross-noise: w07 and the three arms on each other's held-out windows, with
#      per-household log q for the paired comparisons;
#   5. PSID couples' representative theta: the three arms beside w07, v2_b010,
#      typical and gross (cached), figure 130;
#   6. file_k8's simulated and PSID household figures (131, 132);
#   7. noise_copies.py measures and figures 125 and 129.
# Each stage appends to outputs/noise_copies/STATUS; a failure writes FAILED and
# stops. Rerunning the same command resumes: finished members and outputs are
# skipped.
#
# Launch:  nohup setsid ./scripts/run_noise_copies.sh >> logs/noise_copies.log 2>&1 < /dev/null &
# Smoke:   SMOKE=<dir> ./scripts/run_noise_copies.sh   (1 member per arm, max_epochs 1 on
#          one shard per run, few draws; every stage. w07 is trained afresh the same
#          way, since cross-noise needs members trained on the same shards, but
#          the population fit takes RESULTS 64's w07, whose Newton points the
#          fit's cache holds; the population simulations and the fit's cache are
#          linked in read-only; the fit takes no refreshed Newton steps; CSVs go
#          under the smoke directory)
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=.:src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
SMOKE=${SMOKE:-}
DATA=data/processed
MAIN=$DATA/couples_v2_dataset_shards
STRIP=$DATA/couples_v2_strip_dataset_shards
BSTRIP=$DATA/couples_v2_betastrip_dataset_shards
SBC=$DATA/optionA2v2_sbc16_shards
SBCS=$DATA/optionA2v2_sbc16_betastrip_shards
REAL_V2=outputs/waves
REAL_LAI=outputs/laibson_comparison
ARMS="v2_k8 file_k1 file_k8"
K=8
WINDOW="--n_waves 7 --wave_years 2 --start_low 24 --start_high 51"

if [ -n "$SMOKE" ]; then
    ROOT=$SMOKE/outputs/noise_copies; BASE=$SMOKE/outputs/waves_smoke; LAI=$SMOKE/laibson_cache
    FIG=$SMOKE/figures; LOGD=$SMOKE/logs; MDIR=$SMOKE/training_metrics
    SEEDS="0"; TRAIN_EXTRA="--max_epochs 1 --train_shards 1 --metrics_dir $MDIR"
    DRAWS="--n_draws 500"; REFRESH=0; NBOOT=20
    FIGS_EXTRA="--pool 20 --pool_draws 500 --n_raw 2000 --n_plot 1000"
    TRAINED="w07 $ARMS"
else
    ROOT=outputs/noise_copies; BASE=$REAL_V2; LAI=$REAL_LAI; FIG=figures; LOGD=logs
    MDIR=training_metrics; SEEDS="0 1 2 3 4"; TRAIN_EXTRA=""; DRAWS=""; REFRESH=4; NBOOT=1000
    FIGS_EXTRA=""; TRAINED="$ARMS"
fi
NOISE=$ROOT/psid_file_noise.json
ST=$ROOT/STATUS
mkdir -p "$ROOT" "$FIG" "$LOGD" "$MDIR"

status() { echo "$(date -u '+%F %T') $*" | tee -a "$ST"; }
fail() { status "FAILED $1"; exit 1; }
run() {   # run STAGE LOG command...
    local name=$1 log=$2; shift 2
    status "start $name"
    "$@" > "$LOGD/$log.log" 2>&1
    local rc=$?   # captured before anything else can reset it
    [ $rc -eq 0 ] || fail "$name (rc=$rc, see $LOGD/$log.log)"
    status "done $name"
}
dirs_of() {   # dirs_of RUN SEEDS...: the member directories
    local r=$1; shift
    for s in "$@"; do printf ' %s/s%s' "$r" "$s"; done
}
aroot() {   # aroot ARM: where its members (w07/), population and PSID posteriors live
    if [ "$1" = w07 ]; then echo "$BASE"; else echo "$ROOT/$1"; fi
}

status "chain pid $$ (smoke: ${SMOKE:-no})"
for d in "$MAIN" "$STRIP" "$BSTRIP" "$SBC" "$SBCS"; do [ -d "$d" ] || fail "missing $d"; done
for f in heldout_check.json sbc/sbc16.json sbc/sbc16.npz s0/results.json; do
    [ -f "$REAL_V2/w07/$f" ] || fail "missing $REAL_V2/w07/$f (RESULTS 64)"
done
for t in centre plus0 minus0 plus1 minus1 plus2 minus2 plus3 minus3; do
    [ -f "$REAL_V2/population/post_w07_$t.npz" ] || fail "missing $REAL_V2/population/post_w07_$t.npz"
    [ -f "$REAL_V2/population/sim_$t.npz" ] || fail "missing $REAL_V2/population/sim_$t.npz"
done
for f in "$REAL_V2/population.json" "$REAL_V2/psid_w07_couples/posterior_uncorrected.npz" \
         outputs/waves_none/w07/s0/training_summary.json outputs/psid_iid_error/psid_iid_error.json; do
    [ -f "$f" ] || fail "missing $f"
done
avail=$(df --output=avail -k "$DATA" | tail -1)
[ "$avail" -ge $((2 * 1024 * 1024)) ] || fail "disk: ${avail} kB free, need 2 GB"
if [ -n "$SMOKE" ]; then   # read-only links to the simulations and the fit's cache
    mkdir -p "$BASE/population" "$LAI"
    for f in "$REAL_V2"/population/sim_*.npz; do
        ln -sf "$(realpath "$f")" "$BASE/population/$(basename "$f")"
    done
    for f in "$REAL_LAI"/*.npz; do ln -sf "$(realpath "$f")" "$LAI/$(basename "$f")"; done
    SIMS=$BASE/population
else
    SIMS=$REAL_V2/population
fi
V2_NOISE=$($PY -c "import json; d = json.load(open('$REAL_V2/w07/s0/results.json'))['_config']['obs_noise']; \
print(' '.join(f'{k}={lo}:{hi}' for k, (lo, hi) in d.items()))") || fail "reading w07's noise"
status "v2 noise (w07's): $V2_NOISE; copies K=$K"

# --- 0. the file's noise, and the baselines' CSVs -----------------------------------
[ -f "$NOISE" ] || run "calibrate the file's noise" noise_copies_calibrate \
    $PY scripts/psid_noise.py calibrate --out "$NOISE"
[ -f "$ROOT/noise_check.json" ] || run "check the file's noise" noise_copies_check \
    $PY scripts/psid_noise.py check --noise_model "$NOISE" --out "$ROOT/noise_check.json"
[ -f "$MDIR/waves_none/w07/s0.csv" ] && [ -f "$MDIR/waves/w07/s0.csv" ] || \
    run "backfill baseline CSVs" noise_copies_backfill \
    $PY scripts/training_metrics.py backfill "$REAL_V2/w07" outputs/waves_none/w07 --metrics_dir "$MDIR"

# --- 1. training: 5 members per arm, two at a time, the K = 8 arms first -------------
noise_args() {   # noise_args ARM: the train flags of its noise and copies
    case $1 in
        w07) echo "--obs_noise $V2_NOISE" ;;
        v2_k8) echo "--obs_noise $V2_NOISE --noise_copies $K" ;;
        file_k1) echo "--noise_model $NOISE" ;;
        file_k8) echo "--noise_model $NOISE --noise_copies $K" ;;
    esac
}
JOBS=""
for s in $SEEDS; do   # interleaved, so both K = 8 arms show early
    for a in $TRAINED; do
        [ "$a" = file_k1 ] && continue
        [ -f "$(aroot "$a")/w07/s$s/posterior_7w.pt" ] || JOBS="${JOBS}$a|$s|$(aroot "$a")/w07|$(noise_args "$a")\n"
    done
done
for s in $SEEDS; do
    [ -f "$ROOT/file_k1/w07/s$s/posterior_7w.pt" ] || JOBS="${JOBS}file_k1|$s|$ROOT/file_k1/w07|$(noise_args file_k1)\n"
done
status "start training ($(printf "$JOBS" | grep -c .) members)"
printf "$JOBS" | xargs -r -P 2 -I{} bash -c '
    IFS="|" read -r a s out noise <<< "$1"
    "$0" scripts/optionA.py train --seed "$s" --shards "$2" "$3" "$4" \
        --beta_transform logit --rgamma_transform logit $noise $5 --out "$out" $6 \
        > "$7/noise_copies_train_${a}_s$s.log" 2>&1' \
    "$PY" {} "$MAIN" "$STRIP" "$BSTRIP" "$WINDOW" "$TRAIN_EXTRA" "$LOGD"
rc=$?
for a in $TRAINED; do
    for s in $SEEDS; do
        [ -f "$(aroot "$a")/w07/s$s/posterior_7w.pt" ] || fail "training: $a seed $s (xargs rc=$rc)"
    done
done
status "done training"

# --- 2. each arm: PSID, held-out check and SBC -------------------------------------
fig=126
for a in $TRAINED; do
    R=$(aroot "$a"); OUT=$R/w07
    DIRS=$(dirs_of "$OUT" $SEEDS)
    for g in "couples psid_x_comphs_couples" "all psid_x_comphs_optionA"; do
        read -r grp x <<< "$g"
        [ -f "$R/psid_w07_$grp/summary.json" ] || run "$a: PSID $grp" "noise_copies_${a}_psid_$grp" \
            $PY scripts/optionA.py psid --run_dirs $DIRS --x "$DATA/$x.pt" --out "$R/psid_w07_$grp" $DRAWS
    done
    [ -f "$OUT/heldout_check.json" ] || run "$a: held-out check" "noise_copies_${a}_heldout" \
        $PY scripts/check_optionA2_heldout.py --run_dirs $DIRS --shards "$MAIN" "$STRIP" "$BSTRIP" \
        --psid "couples, $a=$R/psid_w07_couples" --out "$OUT/heldout_check.json" $DRAWS
    if [ "$a" = w07 ]; then f="$FIG/smoke_sbc_w07.png"; else f="$FIG/${fig}_sbc_noise_copies_$a.png"; fig=$((fig + 1)); fi
    [ -f "$OUT/sbc/sbc16.json" ] || run "$a: SBC" "noise_copies_${a}_sbc" \
        $PY scripts/sbc_optionA2.py --run_dirs $DIRS --shards "$SBC" "$SBCS" --train "$MAIN" \
        --penalty --out "$OUT/sbc" --figure "$f" --name "$a" $DRAWS
done

# --- 3. the population at 409 households -------------------------------------------
for a in $TRAINED; do
    R=$(aroot "$a")
    [ -f "$R/population.json" ] || run "$a: population" "noise_copies_${a}_population" \
        $PY scripts/waves_experiment.py population --root "$R" --sim_cache "$SIMS" \
        --arms w07 --out "$R/population.json"
done

# --- 4. cross-noise, with per-household log q --------------------------------------
[ -f "$ROOT/cross_noise.json" ] || run "cross-noise" noise_copies_cross \
    $PY scripts/cross_noise_heldout.py --arm w07 "$BASE/w07" --arm v2_k8 "$ROOT/v2_k8/w07" \
    --arm file_k1 "$ROOT/file_k1/w07" --arm file_k8 "$ROOT/file_k8/w07" \
    --shards "$MAIN" "$STRIP" "$BSTRIP" --out "$ROOT/cross_noise.json" \
    --per_row "$ROOT/cross_noise_rows.npz" $DRAWS

# --- 5. PSID couples' representative theta -----------------------------------------
[ -f "$ROOT/laibson_comparison.json" ] || run "population fit" noise_copies_laibson \
    $PY scripts/laibson_comparison.py --arms --refresh "$REFRESH" --cache "$LAI" \
    --out "$ROOT/laibson_comparison.json" \
    --figure "$FIG/130_laibson_comparison_noise_copies.png" \
    --arm v2_b010 outputs/optionA2v2_b010 outputs/psid_optionA2v2_b010_couples \
    --arm typical_b010 outputs/optionA2v2_b010_noise_typical outputs/psid_optionA2v2_b010_noise_typical_couples \
    --arm gross_b010 outputs/optionA2v2_b010_noise_gross outputs/psid_optionA2v2_b010_noise_gross_couples \
    --arm w07 "$REAL_V2/w07" "$REAL_V2/psid_w07_couples" \
    --arm v2_k8 "$ROOT/v2_k8/w07" "$ROOT/v2_k8/psid_w07_couples" \
    --arm file_k1 "$ROOT/file_k1/w07" "$ROOT/file_k1/psid_w07_couples" \
    --arm file_k8 "$ROOT/file_k8/w07" "$ROOT/file_k8/psid_w07_couples" \
    --title "Laibson et al.'s MSM point against PSID couples' representative θ at seven waves: 8 noisy copies (v2_k8) and the methodology file's noise (file_k1, file_k8) beside w07, v2_b010, typical and gross (RESULTS 67)"

# --- 6. file_k8's household figures ------------------------------------------------
[ -f "$FIG/131_simulated_household_noise_copies_file_k8.png" ] || run "household figures" \
    noise_copies_household_figures \
    $PY scripts/plot_optionA2_households.py --run_dirs $(dirs_of "$ROOT/file_k8/w07" $SEEDS) \
    --shards "$MAIN" "$STRIP" "$BSTRIP" --psid_out "$ROOT/file_k8/psid_w07_couples" \
    --x $DATA/psid_x_comphs_couples.pt --fig_dir "$FIG" $FIGS_EXTRA \
    --label "file_k8: the methodology file's noise, 8 copies (w07's window; RESULTS 67)" \
    --sim_name "131_simulated_household_noise_copies_file_k8.png" \
    --psid_name "132_psid_household_noise_copies_file_k8.png"

# --- 7. measures and figures --------------------------------------------------------
run "measures" noise_copies_measures $PY scripts/noise_copies.py measures \
    --root "$ROOT" --base "$BASE" --metrics_dir "$MDIR" --sim_cache "$SIMS" --n_boot "$NBOOT"
run "figures" noise_copies_figures $PY scripts/noise_copies.py figures \
    --root "$ROOT" --base "$BASE" --metrics_dir "$MDIR" --fig_dir "$FIG"
status "DONE"
