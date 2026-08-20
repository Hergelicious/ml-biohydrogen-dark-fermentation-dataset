#!/usr/bin/env bash
# Reproduce every number and figure in the paper, in order.
# Script 02 and 05 dominate the runtime (repeated resampling); allow ~15 minutes.
set -euo pipefail
cd "$(dirname "$0")"

python src/01_audit_dataset.py      | tee results/01_audit.log
python src/02_validation_ladder.py  | tee results/02_validation.log
python src/03_statistics.py         | tee results/03_statistics.log
python src/04_energy_balance.py     | tee results/04_energy.log
python src/05_robustness.py         | tee results/05_robustness.log

echo
echo "Done. Tables and logs are in results/."
