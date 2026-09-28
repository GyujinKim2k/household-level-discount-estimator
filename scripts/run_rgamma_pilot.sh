#!/usr/bin/env bash
# RESULTS 32 pilot: the first 4,096 draws of the option A run with R_gamma as a
# fourth parameter, then a four-parameter model to test whether R_gamma is
# identifiable.
#
# Written into the FINAL run's shard directory with the final run's flags: the
# samplers are prefix-stable (tests/test_rgamma_param.py), so continuing with
# --n_samples 32768 (or more) later reuses these shards unchanged. The config
# check refuses a continuation with different flags.
set -uo pipefail
cd /home/household-level-discount-estimator
mkdir -p logs
echo "$(date -u '+%F %T UTC') pilot generation: 4096 draws"
PYTHONPATH=. .venv/bin/python scripts/generate_dataset.py \
    --simulator twoasset --grid full --device cuda \
    --n_samples 4096 --block 512 --theta_batch 16 --chunk 16 \
    --n_households 16 --card_types --proposal edge_mixture \
    --rgamma_range 1.025 1.075 \
    --n_waves 7 --wave_years 2 --start_age 30 --seed 0 \
    --out data/processed/optionA_card_dataset.pt >> logs/rgamma_pilot_generation.log 2>&1
rc=$?
echo "$(date -u '+%F %T UTC') generation rc=$rc"
[ $rc -ne 0 ] && exit 1
for wave in "0 1" "2"; do
    for s in $wave; do
        PYTHONPATH=. .venv/bin/python scripts/rgamma_pilot.py --seed "$s" \
            > "logs/rgamma_pilot_s${s}.log" 2>&1 &
    done
    wait
done
PYTHONPATH=. .venv/bin/python scripts/rgamma_pilot.py --report > logs/rgamma_pilot_report.log 2>&1
echo "$(date -u '+%F %T UTC') pilot complete"
cat logs/rgamma_pilot_report.log
