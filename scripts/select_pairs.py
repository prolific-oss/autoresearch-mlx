"""
Select cross-checkpoint pairs for the Prolific human eval study.

For each prompt, pairs checkpoints that are ~3-4 steps apart in the val_bpb
ranking. Always includes the worst and best checkpoint in at least one pair.

Reads:  dataset.jsonl
Writes: pairs.jsonl  — one record per (prompt, checkpoint_a, checkpoint_b)

Usage:
    python scripts/select_pairs.py
"""

import json
from itertools import combinations

DATASET_FILE = "dataset.jsonl"
OUT          = "pairs.jsonl"
STEP_GAP     = 1   # pair checkpoints this many steps apart in val_bpb ranking

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

records = []
with open(DATASET_FILE) as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

# Sorted unique checkpoints by val_bpb ascending (worst → best)
seen = {}
for r in records:
    cid = r["checkpoint_id"]
    if cid not in seen:
        seen[cid] = r["val_bpb"]
checkpoints = sorted(seen.items(), key=lambda x: x[1])  # [(commit, val_bpb), ...]
print(f"Checkpoints ({len(checkpoints)}, worst → best):")
for i, (cid, bpb) in enumerate(checkpoints):
    print(f"  [{i}] {cid}  val_bpb={bpb}")

# ---------------------------------------------------------------------------
# Build pairs: every checkpoint with the one STEP_GAP steps ahead
# ---------------------------------------------------------------------------

pairs_index = []
for i in range(len(checkpoints) - STEP_GAP):
    j = i + STEP_GAP
    pairs_index.append((i, j))

print(f"\nSelected {len(pairs_index)} checkpoint pairs (gap={STEP_GAP}):")
for i, j in pairs_index:
    print(f"  [{i}]{checkpoints[i][0]} (bpb={checkpoints[i][1]}) vs [{j}]{checkpoints[j][0]} (bpb={checkpoints[j][1]})")

# ---------------------------------------------------------------------------
# Build prompt lookup: {(checkpoint_id, prompt_id): record}
# ---------------------------------------------------------------------------

lookup = {(r["checkpoint_id"], r["prompt_id"]): r for r in records}
prompt_ids = sorted({r["prompt_id"] for r in records})

# ---------------------------------------------------------------------------
# Write pairs
# ---------------------------------------------------------------------------

pair_records = []
for i, j in pairs_index:
    cid_a, bpb_a = checkpoints[i]
    cid_b, bpb_b = checkpoints[j]
    for pid in prompt_ids:
        rec_a = lookup.get((cid_a, pid))
        rec_b = lookup.get((cid_b, pid))
        if rec_a is None or rec_b is None:
            print(f"WARNING: missing record for ({cid_a} or {cid_b}) × {pid}, skipping.")
            continue
        pair_records.append({
            "prompt_id":        pid,
            "prompt_text":      rec_a["prompt_text"],
            "discipline":       rec_a.get("discipline"),
            "checkpoint_a":     cid_a,
            "val_bpb_a":        bpb_a,
            "generated_text_a": rec_a["generated_text"],
            "checkpoint_b":     cid_b,
            "val_bpb_b":        bpb_b,
            "generated_text_b": rec_b["generated_text"],
            "generation_params": rec_a["generation_params"],
        })

with open(OUT, "w") as f:
    for p in pair_records:
        f.write(json.dumps(p) + "\n")

print(f"\nWrote {len(pair_records)} pairs to {OUT}")
print(f"  ({len(pairs_index)} checkpoint pairs × {len(prompt_ids)} prompts)")
