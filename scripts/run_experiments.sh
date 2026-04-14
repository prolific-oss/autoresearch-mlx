#!/bin/bash
# Run N autoresearch experiments autonomously.
#
# Usage:
#   bash scripts/run_experiments.sh 10
#
# Creates a branch autoresearch/<date>, establishes a baseline, then runs
# N experiments following the protocol in program.md.
# Each experiment is capped at 15 minutes via timeout in the agent instructions.
# The entire session is capped at N * 15 minutes as a hard wall-clock limit.

set -e

N=${1:?Usage: bash scripts/run_experiments.sh <N>}
BRANCH="autoresearch/$(date +%b%d | tr '[:upper:]' '[:lower:]')"
PER_EXPERIMENT_TIMEOUT=900   # 15 minutes in seconds (matches program.md)
TOTAL_TIMEOUT=$((N * PER_EXPERIMENT_TIMEOUT))

echo "Branch:      $BRANCH"
echo "Experiments: $N"
echo "Per-run cap: ${PER_EXPERIMENT_TIMEOUT}s (15 min)"
echo "Total cap:   ${TOTAL_TIMEOUT}s ($(( TOTAL_TIMEOUT / 60 )) min)"
echo ""

git checkout -b "$BRANCH"

timeout "$TOTAL_TIMEOUT" claude --dangerously-skip-permissions -p "
$(cat program.md)

---
IMPORTANT ADDITIONS:
- Run exactly $N experiments total (including the baseline), then stop.
- Do not run more than $N. Do not ask for confirmation.
- Each individual training run must use: timeout ${PER_EXPERIMENT_TIMEOUT} uv run train.py > run.log 2>&1
  If it exits non-zero (timed out), treat it as a crash, log it, and move on.
- After $N experiments are logged in results.tsv, output a brief summary and exit.
"
