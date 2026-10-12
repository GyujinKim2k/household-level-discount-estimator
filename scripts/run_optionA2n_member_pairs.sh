#!/usr/bin/env bash
# RESULTS 43.10: the yardstick for the noise-range arms. Each arm has two
# members, so its PSID medians move with the training seeds as well as with the
# noise. This runs PSID couples and the known-rho recovery (couples' A2n region)
# on every pair of the five A2n members (10 pairs), as the arms were run; the
# spread across pairs is what two members give with the noise unchanged.
# Inference only, ~1.5 GPU-min per pair.
#
# Launch:  ./scripts/run_optionA2n_member_pairs.sh >> logs/optionA2n_sens_pairs.log 2>&1
set -uo pipefail
cd /home/household-level-discount-estimator
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
for p in "0 1" "0 2" "0 3" "0 4" "1 2" "1 3" "1 4" "2 3" "2 4" "3 4"; do
    set -- $p
    d=outputs/optionA2n_sens/pairs/s$1s$2
    mkdir -p $d
    $PY scripts/optionA.py psid --run_dirs outputs/optionA2n/s$1 outputs/optionA2n/s$2 \
        --x data/processed/psid_x_comphs_couples.pt --out $d/psid > $d/psid.log 2>&1
    echo "$(date -u '+%F %T') s$1s$2 psid rc=$?"
    $PY scripts/rho_floor_diagnostic.py --out $d/rho_floor_diagnostic.json recovery \
        --run_dirs outputs/optionA2n/s$1 outputs/optionA2n/s$2 --region a2n > $d/recovery.log 2>&1
    echo "$(date -u '+%F %T') s$1s$2 recovery rc=$?"
done
