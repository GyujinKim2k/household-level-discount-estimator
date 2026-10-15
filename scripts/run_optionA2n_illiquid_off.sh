#!/usr/bin/env bash
# RESULTS 52: how much of A2n's posterior width is the illiquid observation
# noise (the test RESULTS 48.2 proposed). A2n with the illiquid noise off and
# everything else unchanged: consumption 0-0.15 and liquid 0-0.08 as
# calibrated, illiquid 0, income 0. add_obs_noise draws every feature's z and
# sigma whatever its range, so the consumption and liquid noise is draw for
# draw A2n's. Two members, seeds 0 and 1, compared with A2n's members 0 and 1
# (RESULTS 43.10's reference, outputs/optionA2n_sens/a2n_2m).
#
# Steps: train; PSID couples, the held-out check (held-out windows carry the
# arm's noise) and the known-rho recovery, as RESULTS 43.10's evaluate();
# then the same 1,024 held-out households through A2w, this arm and A2n at
# two members each (compare_optionA2_widths.py --illiquid_off; figure 61),
# the arm also on A2n's windows; figure 62 (PSID couples against A2n); and
# the noise-arm table (optionA2n_noise_sensitivity.py).
#
# Launch:  nohup setsid ./scripts/run_optionA2n_illiquid_off.sh >> logs/optionA2n_illoff.log 2>&1 &
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
MAIN=data/processed/couples_dataset_shards
STRIP=data/processed/couples_strip_dataset_shards
OUT=outputs/optionA2n_sens/illoff
COUPLES=data/processed/psid_x_comphs_couples.pt
NOISE="income=0.0:0.0 consumption=0.0:0.15 liquid_assets=0.0:0.08 illiquid_assets=0.0:0.0"
mkdir -p logs "$OUT"

echo "$(date -u '+%F %T') illoff: observation noise $NOISE"
for s in 0 1; do echo "$s"; done |
    xargs -P 2 -L 1 bash -c '"$0" scripts/optionA.py train --seed "$1" \
        --shards "'"$MAIN"'" "'"$STRIP"'" --beta_transform logit --rgamma_transform logit \
        --obs_noise '"$NOISE"' --out "'"$OUT"'" > "logs/optionA2n_illoff_s$1.log" 2>&1;
        rc=$?; echo "$(date -u "+%F %T") illoff seed $1 rc=$rc"' "$PY"
[ -f "$OUT/s0/posterior_7w.pt" ] && [ -f "$OUT/s1/posterior_7w.pt" ] ||
    { echo "illoff: missing checkpoint"; exit 1; }

$PY scripts/optionA.py psid --run_dirs $OUT/s0 $OUT/s1 --x $COUPLES \
    --out outputs/psid_optionA2n_sens_illoff_couples > logs/optionA2n_illoff_psid.log 2>&1
echo "$(date -u '+%F %T') illoff PSID couples rc=$?"
$PY scripts/check_optionA2_heldout.py --run_dirs $OUT/s0 $OUT/s1 --shards $MAIN $STRIP \
    --psid "couples, illoff=outputs/psid_optionA2n_sens_illoff_couples" \
    --out $OUT/heldout_check.json > logs/optionA2n_illoff_heldout.log 2>&1
echo "$(date -u '+%F %T') illoff held-out check rc=$?"
$PY scripts/rho_floor_diagnostic.py --out $OUT/rho_floor_diagnostic.json recovery \
    --run_dirs $OUT/s0 $OUT/s1 --region a2n > logs/optionA2n_illoff_recovery.log 2>&1
echo "$(date -u '+%F %T') illoff recovery rc=$?"
$PY scripts/compare_optionA2_widths.py --illiquid_off $OUT \
    > logs/compare_optionA2_widths_illoff.log 2>&1
echo "$(date -u '+%F %T') same-household widths rc=$?"
$PY scripts/plot_optionA2_psid.py \
    --series "illiquid noise off, couples (2 members)=outputs/psid_optionA2n_sens_illoff_couples" \
             "A2n, couples (members 0, 1)=outputs/psid_optionA2n_sens_a2n_2m_couples" \
    --out figures/62_optionA2n_illoff_psid.png \
    --title "Per-household posterior means, PSID couples: A2n with the illiquid noise off against A2n (members 0 and 1 each)" \
    > logs/optionA2n_illoff_psid_figure.log 2>&1
echo "$(date -u '+%F %T') PSID figure rc=$?"
$PY scripts/optionA2n_noise_sensitivity.py > logs/optionA2n_illoff_summary.log 2>&1
echo "$(date -u '+%F %T') summary rc=$?"
echo "$(date -u '+%F %T') done"
