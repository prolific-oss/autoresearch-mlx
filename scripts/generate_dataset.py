"""
Batch generation for the autoresearch human eval pipeline.
Runs every checkpoint × every prompt and writes one JSONL record per pair.

Requires prompts.jsonl (run scripts/curate_prompts.py first).
Resumes automatically if dataset.jsonl already exists.

Usage:
    uv run scripts/generate_dataset.py
"""

import json
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generate import load_model, sample_token
from prepare import Tokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROMPTS_FILE    = "prompts.jsonl"
CHECKPOINTS_DIR = "checkpoints"
OUT             = "dataset.jsonl"
TEMPERATURE     = 0.8
TOP_K           = 0       # disabled in favor of nucleus sampling
TOP_P           = 0.9     # nucleus sampling
FREQ_PENALTY    = 0.5     # additive frequency penalty (per-token-occurrence) to avoid loops
SEED            = 0
MAX_TOKENS      = 200
STRIP_TRAILING_SPACE_PERIOD = True  # eLife/PLOS prompts end in " ." which traps long-trained models in dot-loops

# ---------------------------------------------------------------------------
# Load prompts
# ---------------------------------------------------------------------------

with open(PROMPTS_FILE) as f:
    prompts = [json.loads(line) for line in f if line.strip()]
print(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}")

# ---------------------------------------------------------------------------
# Load checkpoint list, sorted by val_bpb ascending
# ---------------------------------------------------------------------------

cfg_files = sorted(f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith("_config.json"))
checkpoints = []
for cfg_file in cfg_files:
    cfg_path = os.path.join(CHECKPOINTS_DIR, cfg_file)
    with open(cfg_path) as f:
        cfg = json.load(f)
    commit = cfg_file.replace("_config.json", "")
    weights_path = os.path.join(CHECKPOINTS_DIR, f"{commit}.safetensors")
    if not os.path.isfile(weights_path):
        print(f"WARNING: weights missing for {commit}, skipping.")
        continue
    checkpoints.append({"commit": commit, "val_bpb": cfg.get("val_bpb"), "weights_path": weights_path, "config_path": cfg_path})

checkpoints.sort(key=lambda r: r["val_bpb"] if r["val_bpb"] is not None else float("inf"))
print(f"Found {len(checkpoints)} checkpoints")
print(f"Total pairs: {len(checkpoints) * len(prompts)}\n")

# ---------------------------------------------------------------------------
# Resume: skip already-written pairs
# ---------------------------------------------------------------------------

done = set()
if os.path.isfile(OUT):
    with open(OUT) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                done.add((rec["checkpoint_id"], rec["prompt_id"]))
    print(f"Resuming — {len(done)} pairs already written.\n")

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

tokenizer = Tokenizer.from_directory()
total = len(checkpoints) * len(prompts)
n = 0

with open(OUT, "a") as out_f:
    for ckpt in checkpoints:
        print(f"Loading {ckpt['commit']} (val_bpb={ckpt['val_bpb']})...")
        model, config = load_model(ckpt["weights_path"], ckpt["config_path"])

        for prompt in prompts:
            n += 1
            key = (ckpt["commit"], prompt["prompt_id"])
            if key in done:
                print(f"  [{n}/{total}] skip {key}")
                continue

            print(f"  [{n}/{total}] {ckpt['commit']} × {prompt['prompt_id']}", end=" ... ", flush=True)

            mx.random.seed(SEED)
            text = prompt["prompt_text"]
            if STRIP_TRAILING_SPACE_PERIOD:
                text = text.rstrip(" .\t\n")
            bos = tokenizer.get_bos_token_id()
            prompt_ids = tokenizer.encode(text, prepend=bos)
            ids = mx.array([prompt_ids], dtype=mx.int32)
            output_tokens = []
            context = list(prompt_ids)
            for _ in range(MAX_TOKENS):
                if ids.shape[1] >= config["sequence_len"]:
                    break
                logits = model(ids)
                mx.eval(logits)
                next_id = sample_token(
                    logits[0, -1],
                    TEMPERATURE,
                    TOP_K,
                    top_p=TOP_P,
                    freq_penalty=FREQ_PENALTY,
                    prev_ids=context,
                )
                ids = mx.concatenate([ids, mx.array([[next_id]], dtype=mx.int32)], axis=1)
                output_tokens.append(next_id)
                context.append(next_id)
            generated_text = tokenizer.decode(output_tokens)

            record = {
                "checkpoint_id":   ckpt["commit"],
                "val_bpb":         ckpt["val_bpb"],
                "prompt_id":       prompt["prompt_id"],
                "prompt_text":     prompt["prompt_text"],
                "discipline":      prompt.get("discipline"),
                "generated_text":  generated_text,
                "generation_params": {
                    "temperature":  TEMPERATURE,
                    "top_k":        TOP_K,
                    "top_p":        TOP_P,
                    "freq_penalty": FREQ_PENALTY,
                    "seed":         SEED,
                    "max_tokens":   MAX_TOKENS,
                },
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            print("done")

print(f"\nFinished. Output: {OUT}")
