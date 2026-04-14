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
BASE="autoresearch/$(date +%b%d | tr '[:upper:]' '[:lower:]')"
BRANCH="$BASE"
i=2
while git show-ref --verify --quiet "refs/heads/$BRANCH"; do
    BRANCH="${BASE}-${i}"
    i=$((i + 1))
done
PER_EXPERIMENT_TIMEOUT=900   # 15 minutes in seconds (matches program.md)
TOTAL_TIMEOUT=$((N * PER_EXPERIMENT_TIMEOUT))

echo "Branch:      $BRANCH"
echo "Experiments: $N"
echo "Per-run cap: ${PER_EXPERIMENT_TIMEOUT}s (15 min)"
echo "Total cap:   ${TOTAL_TIMEOUT}s ($(( TOTAL_TIMEOUT / 60 )) min)"
echo ""

git checkout -b "$BRANCH"

python3 -c "
import subprocess, sys
try:
    r = subprocess.run(sys.argv[1:], timeout=$TOTAL_TIMEOUT)
    sys.exit(r.returncode)
except subprocess.TimeoutExpired:
    print('Session timed out after ${TOTAL_TIMEOUT}s')
    sys.exit(124)
" claude --dangerously-skip-permissions -p "
$(cat program.md)

---
IMPORTANT ADDITIONS:
- Run exactly $N experiments total (including the baseline), then stop.
- Do not run more than $N. Do not ask for confirmation.
- Each individual training run must use:
    python3 -c \"import subprocess,sys; r=subprocess.run(['uv','run','train.py'],timeout=${PER_EXPERIMENT_TIMEOUT},stdout=open('run.log','w'),stderr=subprocess.STDOUT); sys.exit(r.returncode)\"
  If it exits non-zero (timed out), treat it as a crash, log it, and move on.
- After $N experiments are logged in results.tsv, output a brief summary and exit.
"
