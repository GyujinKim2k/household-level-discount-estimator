#!/usr/bin/env bash
# RESULTS 62: v2's model with no observation noise ("none"), and with the
# noise level unknown and estimated ("unknown", scripts/noise_unknown.py).
# Everything else is v2's: data, windows, logit beta and R_gamma, five members.
#   1. training, both arms (finished members are skipped);
#   2. none: v2's evaluation (PSID couples and all 889, held-out, SBC with the
#      penalty sets, household figures), as run_v2_noise_arms.sh;
#   3. unknown: SBC on v2's set and PSID, eight targets;
#   4. every 4-parameter network on every arm's noise, none included;
#   5. RESULTS 61's indirect inference with none added, then the unknown arm's
#      eight-unknown version, which also scores how far each network's theta
#      estimate moves when the true noise moves.
# Each stage appends to outputs/v2_noise_none_unknown/STATUS; a failure writes
# FAILED and stops. Rerunning resumes.
#
# Launch:  nohup setsid ./scripts/run_v2_noise_none_unknown.sh >> logs/v2_noise_none_unknown.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
DATA=data/processed
MAIN=$DATA/couples_v2_dataset_shards
STRIP=$DATA/couples_v2_strip_dataset_shards
SBC=$DATA/optionA2v2_sbc16_shards
NONE=outputs/optionA2v2_noise_none
UNK=outputs/optionA2v2_noise_unknown
ST=outputs/v2_noise_none_unknown
mkdir -p "$ST" logs

status() { echo "$(date -u '+%F %T') $*" | tee -a "$ST/STATUS"; }
fail() { status "FAILED $1"; exit 1; }
run() {   # run STAGE LOG command...
    local name=$1 log=$2; shift 2
    status "start $name"
    "$@" > "logs/$log.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] || fail "$name (rc=$rc, see logs/$log.log)"
    status "done $name"
}
status "chain pid $$"

# --- 1. training: ten members, two of each arm at a time -------------------------
printf "0\n1\n2\n3\n4\n" | xargs -P 2 -I{} bash -c '[ -f "$1/s$0/posterior_7w.pt" ] || \
    .venv/bin/python scripts/optionA.py train --seed "$0" --shards "$2" "$3" \
    --beta_transform logit --rgamma_transform logit --out "$1" > "logs/v2_noise_none_train_s$0.log" 2>&1' \
    {} "$NONE" "$MAIN" "$STRIP" &
P1=$!
printf "0\n1\n2\n3\n4\n" | xargs -P 2 -I{} bash -c '[ -f "$1/s$0/posterior_7w.pt" ] || \
    .venv/bin/python scripts/noise_unknown.py train --seed "$0" --out "$1" \
    > "logs/v2_noise_unknown_train_s$0.log" 2>&1' {} "$UNK" &
P2=$!
wait $P1; wait $P2
for d in "$NONE" "$UNK"; do
    for s in 0 1 2 3 4; do [ -f "$d/s$s/posterior_7w.pt" ] || fail "training: $d seed $s"; done
done
status "done training"

# --- 2. none: v2's evaluation ----------------------------------------------------------
DIRS="$NONE/s0 $NONE/s1 $NONE/s2 $NONE/s3 $NONE/s4"
PC=outputs/psid_optionA2v2_noise_none_couples
PA=outputs/psid_optionA2v2_noise_none_all
run "none: PSID couples" v2_noise_none_psid_couples $PY scripts/optionA.py psid \
    --run_dirs $DIRS --x $DATA/psid_x_comphs_couples.pt --out $PC
run "none: PSID all 889" v2_noise_none_psid_all $PY scripts/optionA.py psid \
    --run_dirs $DIRS --x $DATA/psid_x_comphs_optionA.pt --out $PA
run "none: held-out check" v2_noise_none_heldout $PY scripts/check_optionA2_heldout.py \
    --run_dirs $DIRS --shards "$MAIN" "$STRIP" \
    --psid "couples, none=$PC" "couples, v2=outputs/psid_optionA2v2_couples" \
           "all 889, none=$PA" "all 889, v2=outputs/psid_optionA2v2_all" \
    --out "$NONE/heldout_check.json"
run "none: SBC and penalty check" v2_noise_none_sbc $PY scripts/sbc_optionA2.py \
    --run_dirs $DIRS --shards "$SBC" --train "$MAIN" \
    --penalty "0.5=$DATA/optionA2v2_sbc16_liqpen050_shards" \
              "1.5=$DATA/optionA2v2_sbc16_liqpen150_shards" \
    --out "$NONE/sbc" --figure figures/84_sbc_optionA2v2_noise_none.png
run "none: household figures" v2_noise_none_household_figures \
    $PY scripts/plot_optionA2_households.py --run_dirs $DIRS --shards "$MAIN" "$STRIP" \
    --psid_out $PC --x $DATA/psid_x_comphs_couples.pt --fig_dir figures \
    --label "option A2v2, no observation noise (RESULTS 62)" \
    --sim_name 82_simulated_household_optionA2v2_noise_none.png \
    --psid_name 83_psid_household_optionA2v2_noise_none.png

# --- 3. unknown: SBC and PSID (skipped when an earlier run wrote them) -----------------
if [ -f "$UNK/sbc/sbc16.json" ]; then status "skip unknown: SBC (done)"; else
    run "unknown: SBC" v2_noise_unknown_sbc $PY scripts/noise_unknown.py sbc; fi
if [ -f outputs/psid_optionA2v2_noise_unknown_couples/summary.json ]; then
    status "skip unknown: PSID (done)"; else
    run "unknown: PSID" v2_noise_unknown_psid $PY scripts/noise_unknown.py psid; fi

# --- 4. every 4-parameter network on every arm's noise --------------------------------
run "cross-noise held-out" v2_noise_none_cross_heldout $PY scripts/cross_noise_heldout.py \
    --arm v2 outputs/optionA2v2 --arm typical outputs/optionA2v2_noise_typical \
    --arm gross outputs/optionA2v2_noise_gross --arm none "$NONE" \
    --shards "$MAIN" "$STRIP" --out "$ST/cross_noise_heldout.json"

# --- 5. the population ------------------------------------------------------------------
run "Laibson comparison with none" v2_noise_none_laibson $PY scripts/laibson_comparison.py \
    --arms v2 typical gross none --out outputs/laibson_comparison/laibson_comparison_with_none.json \
    --figure "$ST/laibson_comparison_with_none.png"
run "unknown: population" v2_noise_unknown_population $PY scripts/noise_unknown.py population
status "DONE"
