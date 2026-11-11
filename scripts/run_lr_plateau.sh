#!/usr/bin/env bash
# RESULTS 69: reduce-on-plateau and a harder early stop (the user's request of
# 2026-11-11: fresh and fixed copies on both noises; fixed copies as RESULTS 67,
# all 8 in one epoch; patience 60 passes, cap 600). Four 7-wave arms on w07's
# window (biennial, start 24-51; everything else as RESULTS 64's w07), five
# members each:
#   v2_k8_rop       v2's noise, 8 fixed copies (as v2_k8)
#   v2_fresh_rop    v2's noise, a new draw every epoch (as v2_fresh)
#   file_k8_rop     the methodology file's noise, 8 fixed copies (as file_k8)
#   file_fresh_rop  the file's noise, a new draw every epoch (as file_fresh)
# Each trains as its counterpart except: the learning rate halves (floor 1e-5) at
# the 5th pass in a row without a gain in validation loss, training stops after
# 60 passes without one, the cap is 600 passes and the best weights are restored
# there (optionA train --lr_plateau 5 --patience 60 --cap 600; in 8-pass epochs a
# cut at the first epoch without a gain, a stop after 8, a cap of 75).
#   1. training, two members at a time; every member writes its per-epoch CSV to
#      training_metrics/lr_plateau/<arm>/w07/s<seed>.csv as it trains;
#   2. each arm: PSID couples and all 889, held-out check, SBC (figures 141-144);
#   3. the population at 409 households sharing theta0 (RESULTS 64's simulations);
#   4. cross-noise: the ten networks (the four, w07, v2_k8, v2_fresh, file_k1,
#      file_k8, file_fresh) on each other's held-out windows, with per-household
#      log q for the paired comparisons;
#   5. PSID couples' representative theta: RESULTS 68's arms in its order (cached),
#      then the four, figure 146;
#   6. v2_fresh_rop's simulated and PSID household figures (147, 148);
#   7. lr_plateau.py measures and figures 140 and 145.
# Each stage appends to outputs/lr_plateau/STATUS; a failure writes FAILED and
# stops. Rerunning the same command resumes: finished members and outputs are
# skipped.
#
# Launch:  nohup setsid ./scripts/run_lr_plateau.sh >> logs/lr_plateau.log 2>&1 < /dev/null &
# Smoke:   SMOKE=<dir> ./scripts/run_lr_plateau.sh   (1 member per arm, max_epochs 3 on
#          one shard per run; few draws; every stage. w07 and RESULTS 67's and 68's
#          arms are trained afresh the same way, since cross-noise needs members
#          trained on the same shards, but the population fit takes the real arms
#          before RESULTS 69, whose Newton points the fit's cache holds; the
#          simulations and the fit's cache are linked in read-only; the fit takes no
#          refreshed Newton steps; CSVs go under the smoke directory)
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
REAL_COPIES=outputs/noise_copies
REAL_FRESH=outputs/fresh_noise
REAL_LAI=outputs/laibson_comparison
NOISE=$REAL_COPIES/psid_file_noise.json
ARMS="v2_k8_rop v2_fresh_rop file_k8_rop file_fresh_rop"
WINDOW="--n_waves 7 --wave_years 2 --start_low 24 --start_high 51"
SCHEDULE="--lr_plateau 5 --patience 60 --cap 600"

if [ -n "$SMOKE" ]; then
    ROOT=$SMOKE/outputs/lr_plateau; BASE=$SMOKE/outputs/waves_smoke
    COPIES=$SMOKE/outputs/noise_copies_smoke; FRESH=$SMOKE/outputs/fresh_noise_smoke
    LAI=$SMOKE/laibson_cache; FIG=$SMOKE/figures; LOGD=$SMOKE/logs
    MDIR=$SMOKE/training_metrics
    SEEDS="0"; TRAIN_EXTRA="--max_epochs 3 --train_shards 1 --metrics_dir $MDIR"
    DRAWS="--n_draws 500"; REFRESH=0; NBOOT=20
    FIGS_EXTRA="--pool 20 --pool_draws 500 --n_raw 2000 --n_plot 1000"
    TRAINED="w07 v2_k8 v2_fresh file_k1 file_k8 file_fresh $ARMS"
else
    ROOT=outputs/lr_plateau; BASE=$REAL_V2; COPIES=$REAL_COPIES; FRESH=$REAL_FRESH
    LAI=$REAL_LAI; FIG=figures; LOGD=logs; MDIR=training_metrics; SEEDS="0 1 2 3 4"
    TRAIN_EXTRA=""; DRAWS=""; REFRESH=4; NBOOT=1000; FIGS_EXTRA=""; TRAINED="$ARMS"
fi
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
    case $1 in
        w07) echo "$BASE" ;;
        v2_fresh|file_fresh) echo "$FRESH/$1" ;;
        v2_k8|file_k1|file_k8) echo "$COPIES/$1" ;;
        *) echo "$ROOT/$1" ;;
    esac
}

status "chain pid $$ (smoke: ${SMOKE:-no})"
for d in "$MAIN" "$STRIP" "$BSTRIP" "$SBC" "$SBCS"; do [ -d "$d" ] || fail "missing $d"; done
[ -f "$NOISE" ] || fail "missing $NOISE (RESULTS 67)"
for a in w07 v2_k8 file_k1 file_k8 v2_fresh file_fresh; do
    case $a in
        w07) r=$REAL_V2 ;;
        v2_fresh|file_fresh) r=$REAL_FRESH/$a ;;
        *) r=$REAL_COPIES/$a ;;
    esac
    for f in w07/heldout_check.json w07/sbc/sbc16.npz w07/s0/results.json population.json \
             psid_w07_couples/posterior_uncorrected.npz; do
        [ -f "$r/$f" ] || fail "missing $r/$f (RESULTS 64/67/68)"
    done
done
[ -f "$REAL_FRESH/laibson_comparison.json" ] || fail "missing $REAL_FRESH/laibson_comparison.json"
for t in centre plus0 minus0 plus1 minus1 plus2 minus2 plus3 minus3; do
    [ -f "$REAL_V2/population/sim_$t.npz" ] || fail "missing $REAL_V2/population/sim_$t.npz"
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
status "v2 noise (w07's): $V2_NOISE; the file's noise: $NOISE; schedule: $SCHEDULE"

# --- 1. training: 5 members per arm, two at a time, interleaved ---------------------
noise_args() {   # noise_args ARM: the train flags of its noise and schedule
    case $1 in
        w07) echo "--obs_noise $V2_NOISE" ;;
        v2_k8) echo "--obs_noise $V2_NOISE --noise_copies 8" ;;
        v2_fresh) echo "--obs_noise $V2_NOISE --fresh_noise" ;;
        file_k1) echo "--noise_model $NOISE" ;;
        file_k8) echo "--noise_model $NOISE --noise_copies 8" ;;
        file_fresh) echo "--noise_model $NOISE --fresh_noise" ;;
        v2_k8_rop) echo "--obs_noise $V2_NOISE --noise_copies 8 $SCHEDULE" ;;
        v2_fresh_rop) echo "--obs_noise $V2_NOISE --fresh_noise $SCHEDULE" ;;
        file_k8_rop) echo "--noise_model $NOISE --noise_copies 8 $SCHEDULE" ;;
        file_fresh_rop) echo "--noise_model $NOISE --fresh_noise $SCHEDULE" ;;
    esac
}
JOBS=""
for s in $SEEDS; do
    for a in $TRAINED; do
        [ -f "$(aroot "$a")/w07/s$s/posterior_7w.pt" ] || JOBS="${JOBS}$a|$s|$(aroot "$a")/w07|$(noise_args "$a")\n"
    done
done
status "start training ($(printf "$JOBS" | grep -c .) members)"
printf "$JOBS" | xargs -r -P 2 -I{} bash -c '
    IFS="|" read -r a s out noise <<< "$1"
    "$0" scripts/optionA.py train --seed "$s" --shards "$2" "$3" "$4" \
        --beta_transform logit --rgamma_transform logit $noise $5 --out "$out" $6 \
        > "$7/lr_plateau_train_${a}_s$s.log" 2>&1' \
    "$PY" {} "$MAIN" "$STRIP" "$BSTRIP" "$WINDOW" "$TRAIN_EXTRA" "$LOGD"
rc=$?
for a in $TRAINED; do
    for s in $SEEDS; do
        [ -f "$(aroot "$a")/w07/s$s/posterior_7w.pt" ] || fail "training: $a seed $s (xargs rc=$rc)"
    done
done
status "done training"

# --- 2. each arm: PSID, held-out check and SBC -------------------------------------
for a in $TRAINED; do
    R=$(aroot "$a"); OUT=$R/w07
    DIRS=$(dirs_of "$OUT" $SEEDS)
    for g in "couples psid_x_comphs_couples" "all psid_x_comphs_optionA"; do
        read -r grp x <<< "$g"
        [ -f "$R/psid_w07_$grp/summary.json" ] || run "$a: PSID $grp" "lr_plateau_${a}_psid_$grp" \
            $PY scripts/optionA.py psid --run_dirs $DIRS --x "$DATA/$x.pt" --out "$R/psid_w07_$grp" $DRAWS
    done
    [ -f "$OUT/heldout_check.json" ] || run "$a: held-out check" "lr_plateau_${a}_heldout" \
        $PY scripts/check_optionA2_heldout.py --run_dirs $DIRS --shards "$MAIN" "$STRIP" "$BSTRIP" \
        --psid "couples, $a=$R/psid_w07_couples" --out "$OUT/heldout_check.json" $DRAWS
    case $a in
        v2_k8_rop) f="$FIG/141_sbc_lr_plateau_v2_k8_rop.png" ;;
        v2_fresh_rop) f="$FIG/142_sbc_lr_plateau_v2_fresh_rop.png" ;;
        file_k8_rop) f="$FIG/143_sbc_lr_plateau_file_k8_rop.png" ;;
        file_fresh_rop) f="$FIG/144_sbc_lr_plateau_file_fresh_rop.png" ;;
        *) f="$FIG/smoke_sbc_$a.png" ;;
    esac
    [ -f "$OUT/sbc/sbc16.json" ] || run "$a: SBC" "lr_plateau_${a}_sbc" \
        $PY scripts/sbc_optionA2.py --run_dirs $DIRS --shards "$SBC" "$SBCS" --train "$MAIN" \
        --penalty --out "$OUT/sbc" --figure "$f" --name "$a" $DRAWS
done

# --- 3. the population at 409 households -------------------------------------------
for a in $TRAINED; do
    R=$(aroot "$a")
    [ -f "$R/population.json" ] || run "$a: population" "lr_plateau_${a}_population" \
        $PY scripts/waves_experiment.py population --root "$R" --sim_cache "$SIMS" \
        --arms w07 --out "$R/population.json"
done

# --- 4. cross-noise, with per-household log q --------------------------------------
ARM_FLAGS=""
for a in w07 v2_k8 v2_fresh v2_k8_rop v2_fresh_rop file_k1 file_k8 file_fresh file_k8_rop file_fresh_rop; do
    ARM_FLAGS="$ARM_FLAGS --arm $a $(aroot "$a")/w07"
done
[ -f "$ROOT/cross_noise.json" ] || run "cross-noise" lr_plateau_cross \
    $PY scripts/cross_noise_heldout.py $ARM_FLAGS \
    --shards "$MAIN" "$STRIP" "$BSTRIP" --out "$ROOT/cross_noise.json" \
    --per_row "$ROOT/cross_noise_rows.npz" $DRAWS

# --- 5. PSID couples' representative theta -----------------------------------------
# RESULTS 68's arms first, in its order: the bootstrap draws from one stream in
# list order (68.8), so they keep their se
[ -f "$ROOT/laibson_comparison.json" ] || run "population fit" lr_plateau_laibson \
    $PY scripts/laibson_comparison.py --arms --refresh "$REFRESH" --cache "$LAI" \
    --out "$ROOT/laibson_comparison.json" \
    --figure "$FIG/146_laibson_comparison_lr_plateau.png" \
    --arm v2_b010 outputs/optionA2v2_b010 outputs/psid_optionA2v2_b010_couples \
    --arm typical_b010 outputs/optionA2v2_b010_noise_typical outputs/psid_optionA2v2_b010_noise_typical_couples \
    --arm gross_b010 outputs/optionA2v2_b010_noise_gross outputs/psid_optionA2v2_b010_noise_gross_couples \
    --arm w07 "$REAL_V2/w07" "$REAL_V2/psid_w07_couples" \
    --arm v2_k8 "$REAL_COPIES/v2_k8/w07" "$REAL_COPIES/v2_k8/psid_w07_couples" \
    --arm v2_fresh "$REAL_FRESH/v2_fresh/w07" "$REAL_FRESH/v2_fresh/psid_w07_couples" \
    --arm file_k1 "$REAL_COPIES/file_k1/w07" "$REAL_COPIES/file_k1/psid_w07_couples" \
    --arm file_k8 "$REAL_COPIES/file_k8/w07" "$REAL_COPIES/file_k8/psid_w07_couples" \
    --arm file_fresh "$REAL_FRESH/file_fresh/w07" "$REAL_FRESH/file_fresh/psid_w07_couples" \
    --arm v2_k8_rop "$ROOT/v2_k8_rop/w07" "$ROOT/v2_k8_rop/psid_w07_couples" \
    --arm v2_fresh_rop "$ROOT/v2_fresh_rop/w07" "$ROOT/v2_fresh_rop/psid_w07_couples" \
    --arm file_k8_rop "$ROOT/file_k8_rop/w07" "$ROOT/file_k8_rop/psid_w07_couples" \
    --arm file_fresh_rop "$ROOT/file_fresh_rop/w07" "$ROOT/file_fresh_rop/psid_w07_couples" \
    --title "Laibson et al.'s MSM point against PSID couples' representative θ at seven waves: the plateau schedule with patience 60 (…_rop) beside RESULTS 67/68's arms, v2_b010, typical and gross (RESULTS 69)"

# --- 6. v2_fresh_rop's household figures -------------------------------------------
[ -f "$FIG/147_simulated_household_lr_plateau_v2_fresh_rop.png" ] || run "household figures" \
    lr_plateau_household_figures \
    $PY scripts/plot_optionA2_households.py --run_dirs $(dirs_of "$ROOT/v2_fresh_rop/w07" $SEEDS) \
    --shards "$MAIN" "$STRIP" "$BSTRIP" --psid_out "$ROOT/v2_fresh_rop/psid_w07_couples" \
    --x $DATA/psid_x_comphs_couples.pt --fig_dir "$FIG" $FIGS_EXTRA \
    --label "v2_fresh_rop: v2's noise, a new draw every epoch, the plateau schedule (w07's window; RESULTS 69)" \
    --sim_name "147_simulated_household_lr_plateau_v2_fresh_rop.png" \
    --psid_name "148_psid_household_lr_plateau_v2_fresh_rop.png"

# --- 7. measures and figures --------------------------------------------------------
run "measures" lr_plateau_measures $PY scripts/lr_plateau.py measures \
    --root "$ROOT" --base "$BASE" --copies_root "$COPIES" --fresh_root "$FRESH" \
    --metrics_dir "$MDIR" --sim_cache "$SIMS" --n_boot "$NBOOT" \
    --laibson_before "$REAL_FRESH/laibson_comparison.json"
run "figures" lr_plateau_figures $PY scripts/lr_plateau.py figures \
    --root "$ROOT" --base "$BASE" --copies_root "$COPIES" --fresh_root "$FRESH" \
    --metrics_dir "$MDIR" --fig_dir "$FIG"
status "DONE"
