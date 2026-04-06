# Field Notes: autoresearch-mlx × Prolific Human Evaluation

## What This Repo Is

An Apple Silicon (MLX) port of Karpathy's autoresearch project. An LLM agent autonomously discovers better neural network configurations through iterative experimentation on a fixed 5-minute training budget.

The agent edits `train.py`, commits the change, trains for 5 minutes on the ClimbMix dataset, evaluates **bits-per-byte (val_bpb)** on a held-out validation set, keeps the change if it improved, and reverts it if it didn't.

## Goal

Validate whether lower val_bpb actually correlates with better **perceived text quality** using human evaluations on Prolific. The idea: run A/B preference tests where participants choose between completions from models at different val_bpb levels.

---

## Experiment Progress So Far

| Commit | val_bpb | Memory | Status | Description |
|--------|---------|--------|--------|-------------|
| 383abb4 | 2.667000 | 26.9 GB | keep | baseline (AdamW, default config) |
| c67ad2a | 2.695369 | 26.9 GB | discard | lower weight decay to 0.1 |
| 909dd59 | 2.588904 | 26.9 GB | keep | halve total batch size to 2^16 |
| 4161af3 | 2.533728 | 26.9 GB | keep | increase matrix LR to 0.04 |
| 5efc7aa | 1.807902 | 20.7 GB | keep | reduce depth from 8 to 4 layers |

Key finding so far: with a fixed 5-minute wall-clock budget, **smaller models win** — they fit more optimizer steps in the same time, beating larger models despite having fewer parameters.

---

## What Was Added for Human Evaluation

### 1. Model Checkpointing (`train.py`)

After each training run, the model is now saved automatically to `checkpoints/`:

- `checkpoints/<commit_hash>.safetensors` — model weights
- `checkpoints/<commit_hash>_config.json` — architecture config + val_bpb + commit hash

The commit hash ties each checkpoint to its row in `results.tsv`.

### 2. Generation Script (`generate.py`)

Loads a saved checkpoint and generates text completions from a prompt.

```bash
# Use the best checkpoint automatically (lowest val_bpb)
uv run generate.py --prompt "The history of machine learning"

# Use a specific checkpoint by commit hash
uv run generate.py --checkpoint 5efc7aa --prompt "Once upon a time"

# See all available checkpoints and their val_bpb scores
uv run generate.py --list-checkpoints

# Tune sampling parameters
uv run generate.py --prompt "Hello" --temperature 1.0 --top-k 100 --max-tokens 300
```

Sampling options:
- `--temperature` — 0.0 = greedy, 0.8 = default, 1.0 = unscaled
- `--top-k` — restricts sampling to top-k tokens (default: 50)
- `--seed` — fix for reproducibility (important: use same seed across models for fair comparison)

---

## Pipeline Test Instructions

### Prerequisites

Data and tokenizer are already downloaded to `~/.cache/autoresearch/`. No extra setup needed.

### Step 1 — Train and save a checkpoint

```bash
uv run train.py > run.log 2>&1
```

Expected duration: ~6–7 minutes total (1–2 min startup/compile + 5 min training + eval).

When done, the output will include:
```
val_bpb:      x.xxxxxx
Checkpoint saved: checkpoints/<commit_hash>.safetensors
Config saved:     checkpoints/<commit_hash>_config.json
```

### Step 2 — Verify the checkpoint was saved

```bash
uv run generate.py --list-checkpoints
```

Expected output:
```
COMMIT       VAL_BPB      WEIGHTS  PATH
------------------------------------------------------------
d6a6680      x.xxxxxx     yes      checkpoints/d6a6680.safetensors
```

### Step 3 — Generate text

```bash
uv run generate.py --prompt "Once upon a time"
```

To compare two models side by side (for Prolific study prep), run the same prompt with the same seed against different checkpoints:

```bash
uv run generate.py --checkpoint <commit_A> --prompt "The world changed when" --seed 0
uv run generate.py --checkpoint <commit_B> --prompt "The world changed when" --seed 0
```

---

## Next Steps for the Prolific Study

- [ ] Re-run `train.py` on key commits (baseline `383abb4` and best `5efc7aa`) to save their checkpoints
- [ ] Batch-generate comparison samples: same prompts, fixed seeds, two models
- [ ] Design A/B preference task: show participants two unlabeled completions, ask which is more coherent/natural
- [ ] Set up Prolific study (Qualtrics or custom interface)
- [ ] Analyze whether participant preference correlates with val_bpb improvement
