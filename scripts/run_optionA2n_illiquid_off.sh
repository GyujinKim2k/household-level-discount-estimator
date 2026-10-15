#!/usr/bin/env bash
# RESULTS 52: A2n with the illiquid observation noise off (the test RESULTS
# 48.2 proposed). The steps live in run_optionA2n_noise_off.sh, which writes
# the same outputs (outputs/optionA2n_sens/illoff, figures 61 and 62).
#
# Launch:  nohup setsid ./scripts/run_optionA2n_illiquid_off.sh >> logs/optionA2n_illoff.log 2>&1 &
exec "$(dirname "$0")/run_optionA2n_noise_off.sh" illiquid_assets
