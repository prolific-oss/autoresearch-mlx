#!/bin/bash
# Run N autoresearch experiments autonomously.
#
# Usage:
#   bash scripts/run_experiments.sh 10
#
# Creates a branch autoresearch/<date> and runs N experiments, one claude call
# per experiment. The shell loop controls progress reporting.

N=${1:?Usage: bash scripts/run_experiments.sh <N>}
BASE="autoresearch/$(date +%b%d | tr '[:upper:]' '[:lower:]')"
BRANCH="$BASE"
i=2
while git show-ref --verify --quiet "refs/heads/$BRANCH"; do
    BRANCH="${BASE}-${i}"
    i=$((i + 1))
done

echo "Branch:      $BRANCH"
echo "Experiments: $N"
echo ""

git checkout -b "$BRANCH"

# Start fresh: clear results and checkpoints from any previous runs
echo -e "commit\tval_bpb\tmemory_gb\tstatus\tdescription" > results.tsv
rm -rf checkpoints && mkdir checkpoints

CONTEXT="$(cat program.md)"

for experiment in $(seq 1 "$N"); do
    echo ""
    echo "========================================"
    echo "Experiment $experiment / $N"
    echo "========================================"

    if [ "$experiment" -eq 1 ]; then
        TASK="Run the baseline experiment (train.py as-is, no changes). Establish the baseline val_bpb for this hardware."
    else
        TASK="Run one experiment: propose and implement one change to train.py to try to improve val_bpb. Read results.tsv first to see what has been tried so far."
    fi

    claude --dangerously-skip-permissions -p "
$CONTEXT

---
CURRENT TASK (do only this, then stop):
$TASK

Use this command to run training:
    uv run train.py > run.log 2>&1

After training, read the result with:
    grep '^val_bpb:\|^peak_vram_mb:' run.log

Then update results.tsv and commit as described in the protocol. Do not start another experiment.
" || echo "Warning: experiment $experiment exited with an error, continuing."

    echo "Done with experiment $experiment / $N."
done

echo ""
echo "All $N experiments complete."
