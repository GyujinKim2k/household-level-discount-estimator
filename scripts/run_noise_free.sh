#!/usr/bin/env bash
# RESULTS 66: the noise-free arms. w07, a14 and w14 of RESULTS 64-65 retrained
# with no observation noise (the user's choice, 2026-11-07): five members each,
# everything else as v2_b010; every year a window reads inside ages 24-64.
#   1. training, two members at a time (no --obs_noise: no noise);
#   2. each arm: held-out check and SBC (figures 117-119); w07 also scores PSID
#      couples and all 889;
#   3. the population at 409 households sharing theta0, on RESULTS 64's
#      simulations and window starts (paired with the noisy arms);
#   4. cross-noise: each arm's noisy and noise-free networks on each noise's
#      held-out windows;
#   5. PSID couples' representative theta at seven waves: none_w07 and w07
#      beside v2_b010, typical and gross (cached; they reproduce 63.10),
#      figure 121;
#   6. none_w07's simulated and PSID household figures (122, 123);
#   7. noise_free_waves.py measures and figures 116 and 120.
# Each stage appends to outputs/waves_none/STATUS; a failure writes FAILED and
# stops. Rerunning the same command resumes: finished members and outputs are
# skipped.
#
# Launch:  nohup setsid ./scripts/run_noise_free.sh >> logs/noise_free.log 2>&1 < /dev/null &
# Smoke:   SMOKE=<dir> ./scripts/run_noise_free.sh   (1 member per arm, 1 epoch on
#          one shard per run, few draws; every stage. The noisy arms, their
#          population scorings, PSID posteriors and the population fit's cache
#          are linked in read-only; cross-noise compares smoke-trained noisy
#          members, since a member's config records its epochs and shards; the
#          population fit takes no refreshed Newton steps)
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
REAL_V2=outputs/waves
REAL_LAI=outputs/laibson_comparison
ARMS="w07 a14 w14"

if [ -n "$SMOKE" ]; then
    ROOT=$SMOKE/outputs/waves_none; V2=$SMOKE/outputs/waves; LAI=$SMOKE/laibson_cache
    XV2=$SMOKE/outputs/waves_v2_smoke; FIG=$SMOKE/figures; LOGD=$SMOKE/logs
    SEEDS="0"; TRAIN_EXTRA="--max_epochs 1 --train_shards 1"; DRAWS="--n_draws 500"
    REFRESH=0; FIGS_EXTRA="--pool 20 --pool_draws 500 --n_raw 2000 --n_plot 1000"; NBOOT=20
else
    ROOT=outputs/waves_none; V2=$REAL_V2; LAI=$REAL_LAI; XV2=$REAL_V2; FIG=figures; LOGD=logs
    SEEDS="0 1 2 3 4"; TRAIN_EXTRA=""; DRAWS=""; REFRESH=4; FIGS_EXTRA=""; NBOOT=1000
fi
ST=$ROOT/STATUS
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
window_args() {   # window_args LABEL: n, wave years, start range
    local n=$((10#${1:1})) wy=$([ "${1:0:1}" = a ] && echo 1 || echo 2)
    local sl sh
    sl=$($PY scripts/waves_experiment.py start_low --wave_years "$wy") || return 1
    sh=$($PY scripts/waves_experiment.py start_high "$n" --wave_years "$wy") || return 1
    echo "$n $wy $sl $sh"
}

status "chain pid $$ (smoke: ${SMOKE:-no})"
[ -f "$CAL" ] || fail "missing $CAL"
for d in "$MAIN" "$STRIP" "$BSTRIP" "$SBC" "$SBCS"; do [ -d "$d" ] || fail "missing $d"; done
for a in $ARMS; do   # RESULTS 64-65's noisy arms, compared against
    for f in heldout_check.json sbc/sbc16.json sbc/sbc16.npz; do
        [ -f "$REAL_V2/$a/$f" ] || fail "missing $REAL_V2/$a/$f (RESULTS 64-65)"
    done
    for t in centre plus0 minus0 plus1 minus1 plus2 minus2 plus3 minus3; do
        [ -f "$REAL_V2/population/post_${a}_$t.npz" ] || fail "missing $REAL_V2/population/post_${a}_$t.npz"
        [ -f "$REAL_V2/population/sim_$t.npz" ] || fail "missing $REAL_V2/population/sim_$t.npz"
    done
done
for f in "$REAL_V2/population_annual.json" "$REAL_V2/psid_w07_couples/posterior_uncorrected.npz"; do
    [ -f "$f" ] || fail "missing $f"
done
avail=$(df --output=avail -k "$DATA" | tail -1)
[ "$avail" -ge $((1024 * 1024)) ] || fail "disk: ${avail} kB free, need 1 GB"
if [ -n "$SMOKE" ]; then   # read-only links to the noisy arms, their caches and the fit's cache
    mkdir -p "$V2/population" "$LAI"
    for a in $ARMS psid_w07_couples population_annual.json; do
        ln -sfn "$(realpath "$REAL_V2/$a")" "$V2/$a"
    done
    for f in "$REAL_V2"/population/*.npz; do ln -sf "$(realpath "$f")" "$V2/population/$(basename "$f")"; done
    for f in "$REAL_LAI"/*.npz; do ln -sf "$(realpath "$f")" "$LAI/$(basename "$f")"; done
fi

# --- 1. training: 5 members per arm, two at a time, no observation noise -----------
JOBS=""
for a in $ARMS; do
    read -r n wy sl sh <<< "$(window_args "$a")" || fail "window of $a"
    [ -n "$sh" ] || fail "window of $a"
    status "$a: $n waves of $wy years, starts $sl-$sh, no noise"
    for s in $SEEDS; do
        [ -f "$ROOT/$a/s$s/posterior_${n}w.pt" ] || JOBS="${JOBS}$a|$n|$wy|$sl|$sh|$s|$ROOT/$a|\n"
    done
    if [ -n "$SMOKE" ]; then   # noisy smoke members for cross-noise
        V2_NOISE=$($PY -c "import json; d = json.load(open('$CAL'))['sigma_range']; \
print(' '.join(f'{k}={lo}:{hi}' for k, (lo, hi) in d.items()))") || fail "reading $CAL"
        [ -f "$XV2/$a/s0/posterior_${n}w.pt" ] || JOBS="${JOBS}$a|$n|$wy|$sl|$sh|0|$XV2/$a|--obs_noise $V2_NOISE\n"
    fi
done
status "start training ($(printf "$JOBS" | grep -c .) members)"
printf "$JOBS" | xargs -r -P 2 -I{} bash -c '
    IFS="|" read -r a n wy sl sh s out noise <<< "$1"
    tag=$([ -n "$noise" ] && echo v2smoke || echo none)
    "$0" scripts/optionA.py train --seed "$s" --shards "$2" "$3" "$4" \
        --beta_transform logit --rgamma_transform logit $noise \
        --n_waves "$n" --wave_years "$wy" --start_low "$sl" --start_high "$sh" --out "$out" $5 \
        > "$6/noise_free_train_${tag}_${a}_s$s.log" 2>&1' \
    "$PY" {} "$MAIN" "$STRIP" "$BSTRIP" "$TRAIN_EXTRA" "$LOGD"
rc=$?
for a in $ARMS; do
    n=$((10#${a:1}))
    for s in $SEEDS; do
        [ -f "$ROOT/$a/s$s/posterior_${n}w.pt" ] || fail "training: $a seed $s (xargs rc=$rc)"
    done
    [ -z "$SMOKE" ] || [ -f "$XV2/$a/s0/posterior_${n}w.pt" ] || fail "training: smoke noisy $a"
done
status "done training"

# --- 2. each arm: held-out check and SBC; w07 also PSID ------------------------------
fig=117
for a in $ARMS; do
    OUT=$ROOT/$a
    DIRS=$(dirs_of "$OUT" $SEEDS)
    PSID_ARGS=()    # NAME=DIR entries contain a comma and a space
    if [ "$a" = w07 ]; then
        for g in "couples psid_x_comphs_couples" "all psid_x_comphs_optionA"; do
            read -r grp x <<< "$g"
            [ -f "$ROOT/psid_w07_$grp/summary.json" ] || run "none w07: PSID $grp" \
                "noise_free_w07_psid_$grp" $PY scripts/optionA.py psid --run_dirs $DIRS \
                --x "$DATA/$x.pt" --out "$ROOT/psid_w07_$grp" $DRAWS
        done
        PSID_ARGS=("couples, none_w07=$ROOT/psid_w07_couples")
    fi
    [ -f "$OUT/heldout_check.json" ] || run "none $a: held-out check" "noise_free_${a}_heldout" \
        $PY scripts/check_optionA2_heldout.py --run_dirs $DIRS --shards "$MAIN" "$STRIP" "$BSTRIP" \
        --psid "${PSID_ARGS[@]}" --out "$OUT/heldout_check.json" $DRAWS
    [ -f "$OUT/sbc/sbc16.json" ] || run "none $a: SBC" "noise_free_${a}_sbc" \
        $PY scripts/sbc_optionA2.py --run_dirs $DIRS --shards "$SBC" "$SBCS" --train "$MAIN" \
        --penalty --out "$OUT/sbc" --figure "$FIG/${fig}_sbc_noise_free_$a.png" $DRAWS
    fig=$((fig + 1))
done

# --- 3. the population at 409 households -------------------------------------------
[ -f "$ROOT/population.json" ] || run "population" noise_free_population \
    $PY scripts/waves_experiment.py population --root "$ROOT" --sim_cache "$REAL_V2/population" \
    --arms $ARMS --out "$ROOT/population.json"

# --- 4. cross-noise: noisy and noise-free networks on each noise's windows ----------
for a in $ARMS; do
    [ -f "$ROOT/cross_noise_$a.json" ] || run "cross-noise $a" "noise_free_cross_$a" \
        $PY scripts/cross_noise_heldout.py --arm v2 "$XV2/$a" --arm none "$ROOT/$a" \
        --shards "$MAIN" "$STRIP" "$BSTRIP" --out "$ROOT/cross_noise_$a.json" $DRAWS
done

# --- 5. PSID couples' representative theta at seven waves --------------------------
[ -f "$ROOT/laibson_comparison_noise_free.json" ] || run "population fit" noise_free_laibson \
    $PY scripts/laibson_comparison.py --arms --refresh "$REFRESH" --cache "$LAI" \
    --out "$ROOT/laibson_comparison_noise_free.json" \
    --figure "$FIG/121_laibson_comparison_noise_free.png" \
    --arm v2_b010 outputs/optionA2v2_b010 outputs/psid_optionA2v2_b010_couples \
    --arm typical_b010 outputs/optionA2v2_b010_noise_typical outputs/psid_optionA2v2_b010_noise_typical_couples \
    --arm gross_b010 outputs/optionA2v2_b010_noise_gross outputs/psid_optionA2v2_b010_noise_gross_couples \
    --arm w07 "$V2/w07" "$V2/psid_w07_couples" \
    --arm none_w07 "$ROOT/w07" "$ROOT/psid_w07_couples" \
    --title "Laibson et al.'s MSM point against PSID couples' representative θ at seven waves, with no observation noise (none_w07) and with v2's, typical and gross noise (RESULTS 66)"

# --- 6. none_w07's household figures ------------------------------------------------
[ -f "$FIG/122_simulated_household_noise_free_w07.png" ] || run "household figures" \
    noise_free_household_figures \
    $PY scripts/plot_optionA2_households.py --run_dirs $(dirs_of "$ROOT/w07" $SEEDS) \
    --shards "$MAIN" "$STRIP" "$BSTRIP" --psid_out "$ROOT/psid_w07_couples" \
    --x $DATA/psid_x_comphs_couples.pt --fig_dir "$FIG" $FIGS_EXTRA \
    --label "w07 with no observation noise (beta 0.1-1, start 24-51; RESULTS 66)" \
    --sim_name "122_simulated_household_noise_free_w07.png" \
    --psid_name "123_psid_household_noise_free_w07.png"

# --- 7. measures and figures --------------------------------------------------------
run "measures" noise_free_measures $PY scripts/noise_free_waves.py measures \
    --root_v2 "$V2" --root_none "$ROOT" --laibson "$ROOT/laibson_comparison_noise_free.json" \
    --n_boot "$NBOOT"
run "figures" noise_free_figures $PY scripts/noise_free_waves.py figures \
    --root_v2 "$V2" --root_none "$ROOT" --fig_dir "$FIG"
status "DONE"
