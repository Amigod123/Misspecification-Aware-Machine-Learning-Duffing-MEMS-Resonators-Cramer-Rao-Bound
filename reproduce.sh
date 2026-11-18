#!/usr/bin/env bash
# One-command reproduction.  Usage: ./reproduce.sh          (full run, about 1-1.5 h on 2 CPU cores)
#                                    ./reproduce.sh --smoke  (tiny end-to-end test, about 2 min)
set -euo pipefail
cd "$(dirname "$0")"
CFG=configs/default.json
[[ "${1:-}" == "--smoke" ]] && CFG=configs/smoke.json
echo "== config: $CFG"
python checks.py --config "$CFG"
python gen_data.py --config "$CFG"
python run_exp.py --config "$CFG" --force
python aggregate.py --config "$CFG"
python figures.py --config "$CFG"
echo "== done"
