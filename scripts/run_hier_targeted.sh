#!/usr/bin/env bash
# RESULTS 46: the summary likelihood retrained with couples simulated where
# PSID's population sits, then everything RESULTS 45 ran on it.
# Run after `hierarchical_summary.py targeted` has written targeted.npz.
set -euo pipefail
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

[ -f outputs/hier_summary/targeted/mdn.pt ] || $PY scripts/hierarchical_summary.py train --variant targeted
$PY scripts/hierarchical_validation.py diagnose --variant base
$PY scripts/hierarchical_validation.py diagnose --variant targeted
for arm in uncorrected corrected; do
  $PY scripts/hierarchical_population.py fit --variant targeted --arm "$arm"
  $PY scripts/hierarchical_population.py ppc --variant targeted --arm "$arm"
done
$PY scripts/hierarchical_population.py plot --variant targeted \
  --fig figures/54_psid_population_targeted.png
$PY scripts/hierarchical_validation.py validate --variant targeted
$PY scripts/hierarchical_validation.py report --variant targeted \
  --fig figures/55_population_validation_targeted.png
