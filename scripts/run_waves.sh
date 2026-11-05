#!/usr/bin/env bash
# RESULTS 64: the waves experiment. v2 on the box with the beta strip (RESULTS
# 63, RESULTS 1 since 2026-11-05) trained on windows of 7, 10, 14 and 20
# biennial waves, every window inside the same ages: it starts at 24 at the
# earliest and its last wave starts by 63 (the user's choice, 2026-11-05), so
# longer windows start earlier (waves_experiment.start_high). No new data: the
# stored panels cover ages 20-90.
#   1. training, two members at a time: 5 members per arm, v2's noise, the
#      three runs of RESULTS 63 (main, R_gamma strip, beta strip);
#   2. each arm: held-out check, SBC on the set plus the strip; the 7-wave arm
#      also scores PSID couples and all 889 (its consistency check against
#      v2_b010, which starts at 24-45);
#   3. the population at 409 households (waves_experiment.py population);
#   4. the measures and figures 104-106.
# Each stage appends to outputs/waves/STATUS; a failure writes FAILED and
# stops. Rerunning the same command resumes: finished members and outputs are
# skipped.
#
# Launch:  nohup setsid ./scripts/run_waves.sh >> logs/waves.log 2>&1 &
# Smoke:   SMOKE=<dir> ./scripts/run_waves.sh   (1 member per arm, 1 epoch on
#          one shard per run, few draws; every stage, all outputs under <dir>
#          except the population's 9 simulations, which are shared)
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
CAL=outputs/optionA2v2/noise_calibration.json
SMOKE=${SMOKE:-}
DATA=data/processed
MAIN=$DATA/couples_v2_dataset_shards
STRIP=$DATA/couples_v2_strip_dataset_shards
BSTRIP=$DATA/couples_v2_betastrip_dataset_shards
SBC=$DATA/optionA2v2_sbc16_shards
SBCS=$DATA/optionA2v2_sbc16_betastrip_shards
WAVES="7 10 14 20"

if [ -n "$SMOKE" ]; then
    ROOT=$SMOKE/outputs/waves; FIG=$SMOKE/figures; LOGD=$SMOKE/logs
    SEEDS="0"; TRAIN_EXTRA="--max_epochs 1 --train_shards 1"; DRAWS="--n_draws 500"
else
    ROOT=outputs/waves; FIG=figures; LOGD=logs
    SEEDS="0 1 2 3 4"; TRAIN_EXTRA=""; DRAWS=""
fi
ST=$ROOT
mkdir -p "$ROOT" "$FIG" "$LOGD"

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
dirs_of() {   # dirs_of RUN SEEDS...: the member directories
    local r=$1; shift
    for s in "$@"; do printf ' %s/s%s' "$r" "$s"; done
}
arm() { printf '%s/w%02d' "$ROOT" "$1"; }

status "chain pid $$ (smoke: ${SMOKE:-no})"
[ -f "$CAL" ] || fail "missing $CAL"
for d in "$MAIN" "$STRIP" "$BSTRIP" "$SBC" "$SBCS"; do [ -d "$d" ] || fail "missing $d"; done
avail=$(df --output=avail -k "$DATA" | tail -1)
[ "$avail" -ge $((1024 * 1024)) ] || fail "disk: ${avail} kB free, need 1 GB"

# --- 1. training: 5 members per arm, two at a time --------------------------------
V2_NOISE=$($PY -c "import json; d = json.load(open('$CAL'))['sigma_range']; \
print(' '.join(f'{k}={lo}:{hi}' for k, (lo, hi) in d.items()))") || fail "reading $CAL"
status "noise: v2 $V2_NOISE"
JOBS=""
for n in $WAVES; do
    sh=$($PY scripts/waves_experiment.py start_high "$n") || fail "start_high $n"
    for s in $SEEDS; do
        [ -f "$(arm "$n")/s$s/posterior_${n}w.pt" ] || JOBS="${JOBS}$n|$sh|$s|$(arm "$n")\n"
    done
done
status "start training ($(printf "$JOBS" | grep -c .) members)"
printf "$JOBS" | xargs -r -P 2 -I{} bash -c '
    IFS="|" read -r n sh s out <<< "$1"
    "$0" scripts/optionA.py train --seed "$s" --shards "$2" "$3" "$4" \
        --beta_transform logit --rgamma_transform logit --obs_noise $5 \
        --n_waves "$n" --start_high "$sh" --out "$out" $6 \
        > "$7/waves_train_w${n}_s$s.log" 2>&1' \
    "$PY" {} "$MAIN" "$STRIP" "$BSTRIP" "$V2_NOISE" "$TRAIN_EXTRA" "$LOGD"
rc=$?
for n in $WAVES; do
    for s in $SEEDS; do
        [ -f "$(arm "$n")/s$s/posterior_${n}w.pt" ] || fail "training: $n waves seed $s (xargs rc=$rc)"
    done
done
status "done training"

# --- 2. each arm: held-out check and SBC; the 7-wave arm also PSID ---------------
fig=107
for n in $WAVES; do
    OUT=$(arm "$n")
    DIRS=$(dirs_of "$OUT" $SEEDS)
    PSID_ARGS=()    # NAME=DIR entries contain a comma and a space
    if [ "$n" = 7 ]; then
        for g in "couples psid_x_comphs_couples" "all psid_x_comphs_optionA"; do
            read -r grp x <<< "$g"
            [ -f "$ROOT/psid_w07_$grp/summary.json" ] || run "7 waves: PSID $grp" \
                "waves_w07_psid_$grp" $PY scripts/optionA.py psid --run_dirs $DIRS \
                --x "$DATA/$x.pt" --out "$ROOT/psid_w07_$grp" $DRAWS
        done
        PSID_ARGS=("couples, w07=$ROOT/psid_w07_couples")
    fi
    [ -f "$OUT/heldout_check.json" ] || run "$n waves: held-out check" "waves_w${n}_heldout" \
        $PY scripts/check_optionA2_heldout.py --run_dirs $DIRS --shards "$MAIN" "$STRIP" "$BSTRIP" \
        --psid "${PSID_ARGS[@]}" --out "$OUT/heldout_check.json" $DRAWS
    [ -f "$OUT/sbc/sbc16.json" ] || run "$n waves: SBC" "waves_w${n}_sbc" \
        $PY scripts/sbc_optionA2.py --run_dirs $DIRS --shards "$SBC" "$SBCS" --train "$MAIN" \
        --penalty --out "$OUT/sbc" --figure "$FIG/${fig}_sbc_waves_w$(printf %02d "$n").png" $DRAWS
    fig=$((fig + 1))
done

# --- 3. the population at 409 households ------------------------------------------
[ -f "$ROOT/population.json" ] || run "population" waves_population \
    $PY scripts/waves_experiment.py population --root "$ROOT" --with_psid_arm

# --- 4. measures and figures ------------------------------------------------------
run "measures" waves_measures $PY scripts/waves_experiment.py measures --root "$ROOT"
run "figures" waves_figures $PY scripts/waves_experiment.py figures --root "$ROOT" --fig_dir "$FIG"
status "DONE"
