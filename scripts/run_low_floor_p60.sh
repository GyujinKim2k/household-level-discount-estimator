#!/usr/bin/env bash
# RESULTS 73: a longer patience with the low floor, four models. The user,
# 2026-11-21: "try longer patience with the low floor".
# RESULTS 72 lowered the floor to 1e-7 and found no gain; a14 and a20 went up to
# 17 passes without a gain before their best, against a stop after 20 (72.4).
# Here RESULTS 72's command is kept in everything but the stop: 60 passes
# without a gain instead of 20 (RESULTS 69's patience), with the floor at 1e-7,
# a new noise draw every epoch, the rate halved at 5 passes without a gain and a
# cap of 600; RESULTS 1's model (v2's noise on the box with the beta strip, logit
# beta and R_gamma, the main run, R_gamma strip and beta strip), five members
# per arm. The arms are RESULTS 72's, the two same-years pairs:
#   w07 (14 years read) against a14 (15),  w10 (20) against a20 (21).
#   1. training, two members at a time, arm by arm in pair order; CSVs to
#      training_metrics/low_floor_p60/<arm>/s<seed>.csv;
#   2. each arm: held-out check and SBC (figures 173-176);
#   3. patience 60 against RESULTS 72's networks (outputs/low_floor, a stop
#      after 20) and RESULTS 71's default (outputs/waves_default; w07 is
#      v2_fresh_rop) on identical windows: cross-noise with per-household log q,
#      the three settings in one scoring, one arm at a time;
#   4. the population at 409 households sharing theta0 (RESULTS 64's
#      simulations; RESULTS 72's and 71's scorings serve for theirs);
#   5. low_floor_p60.py measures and figures 170, 171, 172, 177, 178.
# Each stage appends to outputs/low_floor_p60/STATUS; a failure writes FAILED and
# stops. Rerunning the same command resumes: finished members and outputs are
# skipped. RESULTS 64's, 71's and 72's outputs are read, never written.
#
# Launch:  nohup setsid ./scripts/run_low_floor_p60.sh >> logs/low_floor_p60.log 2>&1 < /dev/null &
# Smoke:   SMOKE=<dir> ./scripts/run_low_floor_p60.sh   (1 member per arm and setting,
#          max_epochs 3 on one shard per run, few draws; every stage. The patience-20 and
#          default arms are trained afresh the same way, since cross-noise needs members
#          trained on the same shards; the simulations are linked in read-only; CSVs go
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
REAL_P20=outputs/low_floor
REAL_FIVE=outputs/waves_default
REAL_SIMS=outputs/waves/population
REAL_W07=outputs/lr_plateau/v2_fresh_rop
W07_METRICS=training_metrics/lr_plateau/v2_fresh_rop/w07
FLOOR=1e-7
PATIENCE=60
ARMS="w07 a14 w10 a20"
POINTS="centre plus0 minus0 plus1 minus1 plus2 minus2 plus3 minus3"
P60="--fresh_noise --lr_plateau 5 --lr_factor 0.5 --lr_floor $FLOOR --patience $PATIENCE --cap 600"
P20="--fresh_noise --lr_plateau 5 --lr_factor 0.5 --lr_floor $FLOOR --patience 20 --cap 600"

if [ -n "$SMOKE" ]; then
    ROOT=$SMOKE/outputs/low_floor_p60; PTW=$SMOKE/outputs/p20; FIVE=$SMOKE/outputs/floor5
    SIMS=$SMOKE/sims; FIG=$SMOKE/figures; LOGD=$SMOKE/logs; MDIR=$SMOKE/training_metrics
    FIVE_METRICS=$MDIR/floor5/w07
    SEEDS="0"; TRAIN_EXTRA="--max_epochs 3 --train_shards 1 --metrics_dir $MDIR"
    DRAWS="--n_draws 500"; NBOOT=20; STANDINS=$ARMS
else
    ROOT=outputs/low_floor_p60; PTW=$REAL_P20; FIVE=$REAL_FIVE; SIMS=$REAL_SIMS
    FIG=figures; LOGD=logs; MDIR=training_metrics; FIVE_METRICS=$W07_METRICS
    SEEDS="0 1 2 3 4"; TRAIN_EXTRA=""; DRAWS=""; NBOOT=1000; STANDINS=""
fi
ST=$ROOT/STATUS
mkdir -p "$ROOT/population" "$ROOT/compare" "$FIG" "$LOGD" "$MDIR"

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
for t in $POINTS; do
    [ -f "$REAL_SIMS/sim_$t.npz" ] || fail "missing $REAL_SIMS/sim_$t.npz (RESULTS 64)"
done
if [ -z "$SMOKE" ]; then   # RESULTS 72's and 71's arms, compared against
    for pair in "$REAL_P20:RESULTS 72" "$REAL_FIVE:RESULTS 71"; do
        r=${pair%%:*}; tag=${pair#*:}
        [ -f "$r/population.json" ] || fail "missing $r/population.json ($tag)"
        for a in $ARMS; do
            for f in heldout_check.json sbc/sbc16.json sbc/sbc16.npz s0/results.json; do
                [ -f "$r/$a/$f" ] || fail "missing $r/$a/$f ($tag)"
            done
            for t in $POINTS; do
                [ -f "$r/population/post_${a}_$t.npz" ] || \
                    fail "missing $r/population/post_${a}_$t.npz ($tag)"
            done
        done
    done
fi
avail=$(df --output=avail -k "$DATA" | tail -1)
[ "$avail" -ge $((512 * 1024)) ] || fail "disk: ${avail} kB free, need 0.5 GB"
if [ -n "$SMOKE" ]; then   # the simulations, read-only
    mkdir -p "$SIMS"
    for t in $POINTS; do ln -sf "$(realpath "$REAL_SIMS/sim_$t.npz")" "$SIMS/sim_$t.npz"; done
fi
V2_NOISE=$($PY -c "import json; d = json.load(open('$REAL_W07/w07/s0/results.json'))['_config']['obs_noise']; \
print(' '.join(f'{k}={lo}:{hi}' for k, (lo, hi) in d.items()))") || fail "reading w07's noise"
status "v2 noise (v2_fresh_rop's): $V2_NOISE; patience $PATIENCE: $P60"

# --- 1. training: 5 members per arm, two at a time, arm by arm ----------------------
JOBS=""
add_jobs() {   # add_jobs ROOT FLAGS ARMS...: the members not yet trained
    local root=$1 flags=$2; shift 2
    for a in "$@"; do
        local n wy sl sh
        n=$(n_of "$a"); wy=$(wy_of "$a")
        sl=$($PY scripts/waves_experiment.py start_low --wave_years "$wy") || fail "start_low $a"
        sh=$($PY scripts/waves_experiment.py start_high "$n" --wave_years "$wy") || fail "start_high $a"
        for s in $SEEDS; do
            [ -f "$root/$a/s$s/posterior_${n}w.pt" ] || \
                JOBS="${JOBS}$a|$s|$root/$a|--n_waves $n --wave_years $wy --start_low $sl --start_high $sh $flags\n"
        done
    done
}
add_jobs "$ROOT" "$P60" $ARMS
if [ -n "$STANDINS" ]; then
    add_jobs "$PTW" "$P20" $STANDINS
    add_jobs "$FIVE" "--recipe plateau_fresh" $STANDINS
fi
status "start training ($(printf "$JOBS" | grep -c .) members)"
printf "$JOBS" | xargs -r -P 2 -I{} bash -c '
    IFS="|" read -r a s out win <<< "$1"
    tag=$(basename "$(dirname "$out")")_$a      # low_floor_p60_w07, p20_w07 (smoke)
    "$0" scripts/optionA.py train --seed "$s" --shards "$2" "$3" "$4" \
        --beta_transform logit --rgamma_transform logit --obs_noise $5 $win --out "$out" $6 \
        > "$7/train_${tag}_s$s.log" 2>&1' \
    "$PY" {} "$MAIN" "$STRIP" "$BSTRIP" "$V2_NOISE" "$TRAIN_EXTRA" "$LOGD"
rc=$?
for a in $ARMS; do
    for s in $SEEDS; do
        [ -f "$ROOT/$a/s$s/posterior_$(n_of "$a")w.pt" ] || fail "training: $a seed $s (xargs rc=$rc)"
    done
done
for r in ${STANDINS:+"$PTW" "$FIVE"}; do
    for a in $STANDINS; do
        for s in $SEEDS; do
            [ -f "$r/$a/s$s/posterior_$(n_of "$a")w.pt" ] || fail "training: $r $a seed $s (xargs rc=$rc)"
        done
    done
done
status "done training"

# --- 2. each arm: held-out check and SBC ---------------------------------------------
heldout_sbc() {   # heldout_sbc ROOT ARM LABEL FIGURE NAME
    local out=$1/$2 dirs
    dirs=$(dirs_of "$out" $SEEDS)
    [ -f "$out/heldout_check.json" ] || run "$3: held-out check" "low_floor_p60_${3}_heldout" \
        $PY scripts/check_optionA2_heldout.py --run_dirs $dirs --shards "$MAIN" "$STRIP" "$BSTRIP" \
        --psid --out "$out/heldout_check.json" $DRAWS
    [ -f "$out/sbc/sbc16.json" ] || run "$3: SBC" "low_floor_p60_${3}_sbc" \
        $PY scripts/sbc_optionA2.py --run_dirs $dirs --shards "$SBC" "$SBCS" --train "$MAIN" \
        --penalty --out "$out/sbc" --figure "$4" --name "$5" $DRAWS
}
fig=173
for a in $ARMS; do
    heldout_sbc "$ROOT" "$a" "$a" "$FIG/${fig}_sbc_low_floor_p60_$a.png" "$a, patience 60, floor 1e-7"
    fig=$((fig + 1))
done
for a in $STANDINS; do   # smoke only: the stand-ins
    heldout_sbc "$PTW" "$a" "p20_$a" "$FIG/smoke_sbc_p20_$a.png" "$a, patience 20"
    heldout_sbc "$FIVE" "$a" "floor5_$a" "$FIG/smoke_sbc_floor5_$a.png" "$a, the default"
done

# --- 3. patience 60 against RESULTS 72's and 71's, on identical windows ---------------
for a in $ARMS; do
    [ -f "$ROOT/compare/cross_noise_$a.json" ] || run "$a: the three settings" "low_floor_p60_${a}_compare" \
        $PY scripts/cross_noise_heldout.py --arm p60 "$ROOT/$a" --arm p20 "$PTW/$a" --arm floor5 "$FIVE/$a" \
        --shards "$MAIN" "$STRIP" "$BSTRIP" --out "$ROOT/compare/cross_noise_$a.json" \
        --per_row "$ROOT/compare/cross_noise_$a.npz" $DRAWS
done

# --- 4. the population at 409 households sharing theta0 -------------------------------
[ -f "$ROOT/population.json" ] || run "population" low_floor_p60_population \
    $PY scripts/waves_experiment.py population --root "$ROOT" --sim_cache "$SIMS" \
    --arms $ARMS --out "$ROOT/population.json"
for r in ${STANDINS:+"$PTW" "$FIVE"}; do   # smoke only
    [ -f "$r/population.json" ] || run "$(basename "$r") population" "low_floor_p60_$(basename "$r")_population" \
        $PY scripts/waves_experiment.py population --root "$r" --sim_cache "$SIMS" \
        --arms $ARMS --out "$r/population.json"
done

# --- 5. measures and figures --------------------------------------------------------
run "measures" low_floor_p60_measures $PY scripts/low_floor_p60.py measures \
    --root "$ROOT" --p20_root "$PTW" --floor5_root "$FIVE" --metrics_dir "$MDIR" \
    --w07_metrics "$FIVE_METRICS" --sim_cache "$SIMS" --n_boot "$NBOOT"
run "figures" low_floor_p60_figures $PY scripts/low_floor_p60.py figures \
    --root "$ROOT" --p20_root "$PTW" --floor5_root "$FIVE" --metrics_dir "$MDIR" \
    --w07_metrics "$FIVE_METRICS" --fig_dir "$FIG"
status "DONE"
