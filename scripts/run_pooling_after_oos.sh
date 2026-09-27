#!/usr/bin/env bash
# Run the pooling check only once the out-of-sample forecast has released the
# GPU. The forecast process alone holds ~15 GB of the 16 GB card (PyTorch keeps
# freed memory reserved), so running both risks an OOM in the forecast.
set -uo pipefail
cd /home/household-level-discount-estimator
while pgrep -f "scripts/oos_prediction.py --out outputs/oos" >/dev/null; do sleep 60; done
echo "$(date -u '+%F %T') forecast finished; starting pooling check"
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python scripts/pooling_check.py --out outputs/pooling_check.json \
    > logs/pooling_check.log 2>&1
echo "$(date -u '+%F %T') pooling check rc=$?"
