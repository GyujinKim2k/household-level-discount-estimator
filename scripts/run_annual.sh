#!/usr/bin/env bash
# RESULTS 65: annual waves. v2 on the box with the beta strip, as every
# RESULTS 64 arm, trained on 14 and 20 one-year waves (a14, a20; the user's
# choice, 2026-11-07). Every year a window reads lies inside ages 24-64: an
# annual wave at age s reads its stocks at s and its flows at s - 1, so the
# windows start at 25-51 and 25-45 (waves_experiment.start_low/start_high).
# No new data: the stored panels are annual, ages 20-90.
#   1. training, two members at a time: 5 members per arm, v2's noise, the
#      three runs of RESULTS 63 (main, R_gamma strip, beta strip);
#   2. each arm: held-out check (no PSID: annual PSID with card debt on its
#      own does not exist), SBC on the set plus the strip (figures 113-114);
#   3. the population at 409 households: the six arms w07 a14 w14 w10 a20 w20
#      and v2_b010 (the biennial scorings from RESULTS 64's cache), written to
#      population_annual.json;
#   4. annual_waves.py measures and figures 112 and 115.
# Each stage appends to outputs/waves/STATUS_annual; a failure writes FAILED
# and stops. Rerunning the same command resumes: finished members and outputs
# are skipped.
#
# Launch:  nohup setsid ./scripts/run_annual.sh >> logs/annual.log 2>&1 &
# Smoke:   SMOKE=<dir> ./scripts/run_annual.sh   (1 member per arm, 1 epoch on
#          one shard per run, few draws; every stage. RESULTS 64's biennial
#          arms and their population scorings are linked in read-only; the
#          population's 9 simulations are shared)
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
REAL=outputs/waves
ANNUAL="14 20"
BIENNIAL="w07 w10 w14 w20"

if [ -n "$SMOKE" ]; then
    ROOT=$SMOKE/outputs/waves; FIG=$SMOKE/figures; LOGD=$SMOKE/logs
    SEEDS="0"; TRAIN_EXTRA="--max_epochs 1 --train_shards 1"; DRAWS="--n_draws 500"
else
    ROOT=$REAL; FIG=figures; LOGD=logs
    SEEDS="0 1 2 3 4"; TRAIN_EXTRA=""; DRAWS=""
fi
ST=$ROOT/STATUS_annual
mkdir -p "$ROOT/population" "$FIG" "$LOGD"

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
arm() { printf '%s/a%02d' "$ROOT" "$1"; }

status "chain pid $$ (smoke: ${SMOKE:-no})"
[ -f "$CAL" ] || fail "missing $CAL"
for d in "$MAIN" "$STRIP" "$BSTRIP" "$SBC" "$SBCS"; do [ -d "$d" ] || fail "missing $d"; done
for b in $BIENNIAL; do   # RESULTS 64's arms, compared against
    for f in heldout_check.json sbc/sbc16.json sbc/sbc16.npz; do
        [ -f "$REAL/$b/$f" ] || fail "missing $REAL/$b/$f (RESULTS 64)"
    done
    for t in centre plus0 minus0 plus1 minus1 plus2 minus2 plus3 minus3; do
        [ -f "$REAL/population/post_${b}_$t.npz" ] || fail "missing $REAL/population/post_${b}_$t.npz"
    done
done
avail=$(df --output=avail -k "$DATA" | tail -1)
[ "$avail" -ge $((1024 * 1024)) ] || fail "disk: ${avail} kB free, need 1 GB"
if [ -n "$SMOKE" ]; then   # read-only links to RESULTS 64's arms and population scorings
    for b in $BIENNIAL; do ln -sfn "$(realpath "$REAL/$b")" "$ROOT/$b"; done
    for f in "$REAL"/population/post_w*.npz "$REAL"/population/post_v2_b010_*.npz; do
        ln -sf "$(realpath "$f")" "$ROOT/population/$(basename "$f")"
    done
fi

# --- 1. training: 5 members per arm, two at a time --------------------------------
V2_NOISE=$($PY -c "import json; d = json.load(open('$CAL'))['sigma_range']; \
print(' '.join(f'{k}={lo}:{hi}' for k, (lo, hi) in d.items()))") || fail "reading $CAL"
status "noise: v2 $V2_NOISE"
SL=$($PY scripts/waves_experiment.py start_low --wave_years 1) || fail "start_low"
JOBS=""
for n in $ANNUAL; do
    sh=$($PY scripts/waves_experiment.py start_high "$n" --wave_years 1) || fail "start_high $n"
    status "a$n: starts $SL-$sh"
    for s in $SEEDS; do
        [ -f "$(arm "$n")/s$s/posterior_${n}w.pt" ] || JOBS="${JOBS}$n|$sh|$s|$(arm "$n")\n"
    done
done
status "start training ($(printf "$JOBS" | grep -c .) members)"
printf "$JOBS" | xargs -r -P 2 -I{} bash -c '
    IFS="|" read -r n sh s out <<< "$1"
    "$0" scripts/optionA.py train --seed "$s" --shards "$2" "$3" "$4" \
        --beta_transform logit --rgamma_transform logit --obs_noise $5 \
        --n_waves "$n" --wave_years 1 --start_low "$8" --start_high "$sh" --out "$out" $6 \
        > "$7/annual_train_a${n}_s$s.log" 2>&1' \
    "$PY" {} "$MAIN" "$STRIP" "$BSTRIP" "$V2_NOISE" "$TRAIN_EXTRA" "$LOGD" "$SL"
rc=$?
for n in $ANNUAL; do
    for s in $SEEDS; do
        [ -f "$(arm "$n")/s$s/posterior_${n}w.pt" ] || fail "training: a$n seed $s (xargs rc=$rc)"
    done
done
status "done training"

# --- 2. each arm: held-out check and SBC --------------------------------------------
fig=113
for n in $ANNUAL; do
    OUT=$(arm "$n")
    DIRS=$(dirs_of "$OUT" $SEEDS)
    [ -f "$OUT/heldout_check.json" ] || run "a$n: held-out check" "annual_a${n}_heldout" \
        $PY scripts/check_optionA2_heldout.py --run_dirs $DIRS --shards "$MAIN" "$STRIP" "$BSTRIP" \
        --psid --out "$OUT/heldout_check.json" $DRAWS
    [ -f "$OUT/sbc/sbc16.json" ] || run "a$n: SBC" "annual_a${n}_sbc" \
        $PY scripts/sbc_optionA2.py --run_dirs $DIRS --shards "$SBC" "$SBCS" --train "$MAIN" \
        --penalty --out "$OUT/sbc" --figure "$FIG/${fig}_sbc_annual_a${n}.png" $DRAWS
    fig=$((fig + 1))
done

# --- 3. the population at 409 households ------------------------------------------
[ -f "$ROOT/population_annual.json" ] || run "population" annual_population \
    $PY scripts/waves_experiment.py population --root "$ROOT" --sim_cache "$REAL/population" \
    --arms $BIENNIAL a14 a20 --with_psid_arm --out "$ROOT/population_annual.json"

# --- 4. measures and figures ------------------------------------------------------
run "measures" annual_measures $PY scripts/annual_waves.py measures --root "$ROOT"
run "figures" annual_figures $PY scripts/annual_waves.py figures --root "$ROOT" --fig_dir "$FIG"
status "DONE"
