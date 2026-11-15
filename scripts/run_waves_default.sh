#!/usr/bin/env bash
# RESULTS 71: the waves experiment under the default recipe, biennial and
# annual waves (the user, 2026-11-14: "if we apply this to the real household,
# they usually have the annual data. So, let's test the same model with annual
# & more wave versions"; they chose the full annual ladder). RESULTS 1's model
# (v2's noise on the box with the beta strip, logit beta and R_gamma, the main
# run, R_gamma strip and beta strip), trained with the default recipe for new
# runs (optionA train --recipe plateau_fresh: a new noise draw every epoch, the
# rate halved at 5 passes without a gain, floor 1e-5, a stop after 20, a cap
# of 600), five members per arm. Every year a window reads lies inside ages
# 24-64 (waves_experiment.start_low/start_high):
#   w07             7 biennial waves: v2_fresh_rop as trained (RESULTS 69),
#                   linked in read-only, not retrained
#   w10 w14 w20     10, 14, 20 biennial waves (starts 24-45, 24-37, 24-25)
#   a07 a10 a14 a20 7, 10, 14, 20 annual waves (starts 25-58, 25-55, 25-51, 25-45)
#   1. training, two members at a time, arm by arm; every member writes its
#      per-epoch CSV to training_metrics/waves_default/<arm>/s<seed>.csv;
#   2. each trained arm: held-out check (no PSID: PSID has seven biennial
#      waves), SBC (figures 152-158);
#   3. the recipe against RESULTS 64/65's on identical windows: cross-noise
#      with per-household log q, one length at a time (w07's pair is RESULTS
#      69's);
#   4. the population at 409 households sharing theta0 (RESULTS 64's
#      simulations; w07's scorings linked from RESULTS 69);
#   5. waves_default.py measures and figures 150, 151, 159 and 160.
# Each stage appends to outputs/waves_default/STATUS; a failure writes FAILED
# and stops. Rerunning the same command resumes: finished members and outputs
# are skipped. RESULTS 64/65's arms (outputs/waves) and v2_fresh_rop
# (outputs/lr_plateau/v2_fresh_rop) are read, never written.
#
# Launch:  nohup setsid ./scripts/run_waves_default.sh >> logs/waves_default.log 2>&1 < /dev/null &
# Smoke:   SMOKE=<dir> ./scripts/run_waves_default.sh   (1 member per arm, max_epochs 3
#          on one shard per run, few draws; every stage. RESULTS 64/65's arms
#          w10 w14 w20 a14 a20 are trained afresh the same way with the old
#          recipe, since cross-noise needs members trained on the same shards;
#          w07 (both recipes), the simulations and w07's scorings are linked in
#          read-only; CSVs go under the smoke directory)
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
REAL_OLD=outputs/waves
REAL_W07=outputs/lr_plateau/v2_fresh_rop
W07_ROWS=outputs/lr_plateau/cross_noise_rows.npz
PSID_REF=outputs/lr_plateau/laibson_comparison.json
TRAINED="a07 w10 a10 w14 a14 w20 a20"
ALL="w07 a07 w10 a10 w14 a14 w20 a20"
PAIRED="w10 w14 w20 a14 a20"
OLD="w07 w10 w14 w20 a14 a20"
POINTS="centre plus0 minus0 plus1 minus1 plus2 minus2 plus3 minus3"

if [ -n "$SMOKE" ]; then
    ROOT=$SMOKE/outputs/waves_default; OLD_ROOT=$SMOKE/outputs/waves_old
    SIMS=$SMOKE/sims; FIG=$SMOKE/figures; LOGD=$SMOKE/logs; MDIR=$SMOKE/training_metrics
    SEEDS="0"; TRAIN_EXTRA="--max_epochs 3 --train_shards 1 --metrics_dir $MDIR"
    DRAWS="--n_draws 500"; NBOOT=20; OLD_TRAINED=$PAIRED
else
    ROOT=outputs/waves_default; OLD_ROOT=$REAL_OLD; SIMS=$REAL_OLD/population
    FIG=figures; LOGD=logs; MDIR=training_metrics
    SEEDS="0 1 2 3 4"; TRAIN_EXTRA=""; DRAWS=""; NBOOT=1000; OLD_TRAINED=""
fi
ST=$ROOT/STATUS
mkdir -p "$ROOT/population" "$ROOT/recipe" "$FIG" "$LOGD" "$MDIR"

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
n_of() { echo "$((10#${1:1}))"; }                      # w07 -> 7
wy_of() { case $1 in w*) echo 2 ;; a*) echo 1 ;; esac; }

status "chain pid $$ (smoke: ${SMOKE:-no})"
for d in "$MAIN" "$STRIP" "$BSTRIP" "$SBC" "$SBCS"; do [ -d "$d" ] || fail "missing $d"; done
for f in w07/heldout_check.json w07/sbc/sbc16.json w07/sbc/sbc16.npz w07/s0/results.json \
         population.json; do
    [ -f "$REAL_W07/$f" ] || fail "missing $REAL_W07/$f (RESULTS 69)"
done
for t in $POINTS; do
    [ -f "$REAL_W07/population/post_w07_$t.npz" ] || fail "missing $REAL_W07/population/post_w07_$t.npz"
    [ -f "$REAL_OLD/population/sim_$t.npz" ] || fail "missing $REAL_OLD/population/sim_$t.npz"
done
for f in "$W07_ROWS" "$PSID_REF"; do [ -f "$f" ] || fail "missing $f (RESULTS 69)"; done
for a in $OLD; do   # RESULTS 64/65's arms, compared against
    for f in heldout_check.json sbc/sbc16.json sbc/sbc16.npz; do
        [ -f "$REAL_OLD/$a/$f" ] || fail "missing $REAL_OLD/$a/$f (RESULTS 64/65)"
    done
    for t in $POINTS; do
        [ -f "$REAL_OLD/population/post_${a}_$t.npz" ] || fail "missing $REAL_OLD/population/post_${a}_$t.npz"
    done
done
avail=$(df --output=avail -k "$DATA" | tail -1)
[ "$avail" -ge $((1024 * 1024)) ] || fail "disk: ${avail} kB free, need 1 GB"
# w07 under the default is v2_fresh_rop: its members and scorings, linked read-only
ln -sfn "$(realpath "$REAL_W07/w07")" "$ROOT/w07"
for t in $POINTS; do
    ln -sf "$(realpath "$REAL_W07/population/post_w07_$t.npz")" "$ROOT/population/post_w07_$t.npz"
done
if [ -n "$SMOKE" ]; then   # the simulations, and RESULTS 64's w07 with its scorings
    mkdir -p "$SIMS" "$OLD_ROOT/population"
    for t in $POINTS; do
        ln -sf "$(realpath "$REAL_OLD/population/sim_$t.npz")" "$SIMS/sim_$t.npz"
        ln -sf "$(realpath "$REAL_OLD/population/post_w07_$t.npz")" \
            "$OLD_ROOT/population/post_w07_$t.npz"
    done
    ln -sfn "$(realpath "$REAL_OLD/w07")" "$OLD_ROOT/w07"
fi
V2_NOISE=$($PY -c "import json; d = json.load(open('$REAL_W07/w07/s0/results.json'))['_config']['obs_noise']; \
print(' '.join(f'{k}={lo}:{hi}' for k, (lo, hi) in d.items()))") || fail "reading w07's noise"
status "v2 noise (v2_fresh_rop's): $V2_NOISE; recipe plateau_fresh"

# --- 1. training: 5 members per arm, two at a time, arm by arm ----------------------
JOBS=""
add_jobs() {   # add_jobs ROOT RECIPE ARMS...: the members not yet trained
    local root=$1 recipe=$2; shift 2
    for a in "$@"; do
        local n wy sl sh
        n=$(n_of "$a"); wy=$(wy_of "$a")
        sl=$($PY scripts/waves_experiment.py start_low --wave_years "$wy") || fail "start_low $a"
        sh=$($PY scripts/waves_experiment.py start_high "$n" --wave_years "$wy") || fail "start_high $a"
        for s in $SEEDS; do
            [ -f "$root/$a/s$s/posterior_${n}w.pt" ] || \
                JOBS="${JOBS}$a|$s|$root/$a|--n_waves $n --wave_years $wy --start_low $sl --start_high $sh $recipe\n"
        done
    done
}
add_jobs "$ROOT" "--recipe plateau_fresh" $TRAINED
[ -n "$OLD_TRAINED" ] && add_jobs "$OLD_ROOT" "" $OLD_TRAINED
status "start training ($(printf "$JOBS" | grep -c .) members)"
printf "$JOBS" | xargs -r -P 2 -I{} bash -c '
    IFS="|" read -r a s out win <<< "$1"
    tag=$(basename "$(dirname "$out")")_$a      # waves_default_a07, waves_old_w10 (smoke)
    "$0" scripts/optionA.py train --seed "$s" --shards "$2" "$3" "$4" \
        --beta_transform logit --rgamma_transform logit --obs_noise $5 $win --out "$out" $6 \
        > "$7/train_${tag}_s$s.log" 2>&1' \
    "$PY" {} "$MAIN" "$STRIP" "$BSTRIP" "$V2_NOISE" "$TRAIN_EXTRA" "$LOGD"
rc=$?
for a in $TRAINED; do
    for s in $SEEDS; do
        [ -f "$ROOT/$a/s$s/posterior_$(n_of "$a")w.pt" ] || fail "training: $a seed $s (xargs rc=$rc)"
    done
done
for a in $OLD_TRAINED; do
    for s in $SEEDS; do
        [ -f "$OLD_ROOT/$a/s$s/posterior_$(n_of "$a")w.pt" ] || fail "training: old $a seed $s (xargs rc=$rc)"
    done
done
status "done training"

# --- 2. each trained arm: held-out check and SBC ------------------------------------
fig=152
for a in $TRAINED; do
    OUT=$ROOT/$a
    DIRS=$(dirs_of "$OUT" $SEEDS)
    [ -f "$OUT/heldout_check.json" ] || run "$a: held-out check" "waves_default_${a}_heldout" \
        $PY scripts/check_optionA2_heldout.py --run_dirs $DIRS --shards "$MAIN" "$STRIP" "$BSTRIP" \
        --psid --out "$OUT/heldout_check.json" $DRAWS
    [ -f "$OUT/sbc/sbc16.json" ] || run "$a: SBC" "waves_default_${a}_sbc" \
        $PY scripts/sbc_optionA2.py --run_dirs $DIRS --shards "$SBC" "$SBCS" --train "$MAIN" \
        --penalty --out "$OUT/sbc" --figure "$FIG/${fig}_sbc_waves_default_$a.png" \
        --name "$a under the default recipe" $DRAWS
    fig=$((fig + 1))
done
for a in $OLD_TRAINED; do   # smoke only: the old recipe's stand-ins
    OUT=$OLD_ROOT/$a
    DIRS=$(dirs_of "$OUT" $SEEDS)
    [ -f "$OUT/heldout_check.json" ] || run "old $a: held-out check" "waves_default_old_${a}_heldout" \
        $PY scripts/check_optionA2_heldout.py --run_dirs $DIRS --shards "$MAIN" "$STRIP" "$BSTRIP" \
        --psid --out "$OUT/heldout_check.json" $DRAWS
    [ -f "$OUT/sbc/sbc16.json" ] || run "old $a: SBC" "waves_default_old_${a}_sbc" \
        $PY scripts/sbc_optionA2.py --run_dirs $DIRS --shards "$SBC" "$SBCS" --train "$MAIN" \
        --penalty --out "$OUT/sbc" --figure "$FIG/smoke_sbc_old_$a.png" --name "old $a" $DRAWS
done

# --- 3. the recipe against RESULTS 64/65's, on identical windows ----------------------
for a in $PAIRED; do
    [ -f "$ROOT/recipe/cross_noise_$a.json" ] || run "$a: the two recipes" "waves_default_${a}_recipe" \
        $PY scripts/cross_noise_heldout.py --arm default "$ROOT/$a" --arm old "$OLD_ROOT/$a" \
        --shards "$MAIN" "$STRIP" "$BSTRIP" --out "$ROOT/recipe/cross_noise_$a.json" \
        --per_row "$ROOT/recipe/cross_noise_$a.npz" $DRAWS
done

# --- 4. the population at 409 households sharing theta0 -------------------------------
[ -f "$ROOT/population.json" ] || run "population" waves_default_population \
    $PY scripts/waves_experiment.py population --root "$ROOT" --sim_cache "$SIMS" \
    --arms $ALL --out "$ROOT/population.json"
if [ -n "$SMOKE" ]; then
    [ -f "$OLD_ROOT/population.json" ] || run "old population" waves_default_old_population \
        $PY scripts/waves_experiment.py population --root "$OLD_ROOT" --sim_cache "$SIMS" \
        --arms $OLD --out "$OLD_ROOT/population.json"
fi

# --- 5. measures and figures --------------------------------------------------------
run "measures" waves_default_measures $PY scripts/waves_default.py measures \
    --root "$ROOT" --old_root "$OLD_ROOT" --metrics_dir "$MDIR" --sim_cache "$SIMS" \
    --n_boot "$NBOOT"
run "figures" waves_default_figures $PY scripts/waves_default.py figures \
    --root "$ROOT" --old_root "$OLD_ROOT" --fig_dir "$FIG"
status "DONE"
