#!/usr/bin/env bash
# Train all 8 checkpoints. Skips any tag whose final checkpoint already
# exists; resumes from snapshot if one is present (handled inside train.py).
#
# Usage:
#   bash scripts/_resume_chain.sh
#
# To stop cleanly:
#   pkill -f "uv run train.py"   (SIGTERM the in-flight train.py)
#   or kill the parent shell process. The next run will resume from the
#   most recent snapshot (saved every AR_SNAPSHOT_INTERVAL training seconds,
#   default 300).

set -e

LOG="train_chain2.log"
CKPT_DIR="checkpoints"

# spec format: tag:time_budget:total_batch:device_batch:warmdown
SPECS=(
  "84a8401:300:65536:16:0.5"
  "bb04035:300:65536:32:0.5"
  "4926214:300:65536:32:0.3"
  "0b820f1:300:32768:16:0.3"
  "098bc40:300:8192:4:0.3"
  "84a8401-long:1800:65536:16:0.5"
  "0b820f1-long:1800:32768:16:0.3"
  "098bc40-long:1800:8192:4:0.3"
)

echo "=== START $(date) ===" | tee -a "$LOG"

for spec in "${SPECS[@]}"; do
  IFS=: read -r tag tb tot dev wd <<< "$spec"
  final="${CKPT_DIR}/${tag}.safetensors"
  snapshot="${CKPT_DIR}/${tag}.snapshot.safetensors"

  if [ -f "$final" ]; then
    echo "--- $tag (already done, skipping) at $(date) ---" | tee -a "$LOG"
    continue
  fi

  if [ -f "$snapshot" ]; then
    echo "--- $tag (resuming from snapshot) at $(date) ---" | tee -a "$LOG"
  else
    echo "--- $tag (TB=$tb TOT=$tot DEV=$dev WD=$wd) at $(date) ---" | tee -a "$LOG"
  fi

  AR_TIME_BUDGET=$tb AR_TOTAL_BATCH=$tot AR_DEVICE_BATCH=$dev AR_WARMDOWN=$wd AR_TAG=$tag \
    uv run train.py >> "$LOG" 2>&1 || {
      echo "ERROR: $tag failed (exit $?)" | tee -a "$LOG"
      exit 1
    }
done

echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
