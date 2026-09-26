#!/usr/bin/env bash
# Figures 10-13: the section 20 configurations for somehs and compco, each
# against its own group's Phase 4 baseline. Run after run_groups_adopted.sh.
set -euo pipefail
cd /home/household-level-discount-estimator
n=11
for g in somehs compco; do
    for cfg in full arch; do
        if [ "$cfg" = full ]; then lab="wide embedder, log(1-δ)"; else lab="wide embedder only (linear δ)"; fi
        PYTHONPATH=. .venv/bin/python scripts/plot_adopted_posterior.py \
            --current "outputs/psid_groups/${cfg}_${g}" --label "$lab" \
            --baseline "outputs/psid_phase4_${g}" --title_group "$g" \
            --out "figures/${n}_${g}_${cfg}_literature_comparison.png"
        n=$((n + 1))
    done
done
