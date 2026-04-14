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

### 3. Human Eval Pipeline (`scripts/`)

Three scripts that turn autoresearch checkpoints into a Prolific-ready dataset. Run them in order after completing autoresearch experiments.

#### `scripts/curate_prompts.py` → `prompts.jsonl`

Pulls opening sentences from the [`tomasg25/scientific_lay_summarisation`](https://huggingface.co/datasets/tomasg25/scientific_lay_summarisation) dataset (eLife + PLOS subsets, CC-BY-4.0). Samples 4 prompts per discipline across 5 disciplines = 20 prompts total. Discipline is inferred from the dataset's `keywords` field.

Disciplines: biomedical, neuroscience, psychology/social, environmental, physics/tech.

```bash
uv run scripts/curate_prompts.py
```

Output format per line: `{prompt_id, prompt_text, discipline, source, title}`

#### `scripts/generate_dataset.py` → `dataset.jsonl`

Runs every checkpoint × every prompt and writes one JSONL record per pair. Loads model/sampling logic from `generate.py`. Resumes automatically if interrupted — already-written pairs are skipped.

```bash
uv run scripts/generate_dataset.py
```

Generation params (edit constants at top of file): `temperature=0.8, top_k=50, seed=0, max_tokens=200`

Output format per line: `{checkpoint_id, val_bpb, prompt_id, prompt_text, discipline, generated_text, generation_params}`

#### `scripts/select_pairs.py` → `pairs.jsonl`

Pairs checkpoints that are `STEP_GAP` steps apart in val_bpb ranking (default: 3). Tune `STEP_GAP` after seeing the actual spread from your runs.

```bash
python scripts/select_pairs.py
```

Output format per line: `{prompt_id, prompt_text, discipline, checkpoint_a, val_bpb_a, generated_text_a, checkpoint_b, val_bpb_b, generated_text_b, generation_params}`

#### Full pipeline

```bash
# 1. Curate prompts (one-time)
uv run scripts/curate_prompts.py        # → prompts.jsonl

# 2. After autoresearch runs:
uv run scripts/generate_dataset.py     # → dataset.jsonl

# 3. Select pairs for Prolific:
python scripts/select_pairs.py         # → pairs.jsonl
```

---

## Known Issues & Fixes

### `token_bytes.npy` missing on first run

**Error:**
```
FileNotFoundError: Missing token_bytes lookup at ~/.cache/autoresearch/tokenizer/token_bytes.npy. Run prepare.py first.
```

**Why it happens:** The tokenizer was originally trained by the PyTorch version of this repo, which saved `token_bytes.pt`. The MLX port expects `token_bytes.npy`. Running `prepare.py` would retrain the tokenizer from scratch (slow), but the file can be regenerated directly from the existing `tokenizer.pkl` in seconds.

**Fix:**
```bash
uv run python - <<'EOF'
import pickle, numpy as np, os

TOKENIZER_DIR = os.path.expanduser("~/.cache/autoresearch/tokenizer")
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]

with open(os.path.join(TOKENIZER_DIR, "tokenizer.pkl"), "rb") as f:
    enc = pickle.load(f)

special_set = set(SPECIAL_TOKENS)
token_bytes_list = []
for token_id in range(enc.n_vocab):
    token_str = enc.decode([token_id])
    token_bytes_list.append(0 if token_str in special_set else len(token_str.encode("utf-8")))

token_bytes = np.array(token_bytes_list, dtype=np.int32)
out = os.path.join(TOKENIZER_DIR, "token_bytes.npy")
np.save(out, token_bytes)
print(f"Saved {len(token_bytes_list)} token byte lengths to {out}")
EOF
```

Then re-run `uv run train.py` as normal.

---

## Full Pipeline: Commands to Get It Running

### Step 1 — (If needed) Fix the PyTorch tokenizer issue

If you see `FileNotFoundError: Missing token_bytes lookup`, run this once:

```bash
uv run python - <<'EOF'
import pickle, numpy as np, os

TOKENIZER_DIR = os.path.expanduser("~/.cache/autoresearch/tokenizer")
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]

with open(os.path.join(TOKENIZER_DIR, "tokenizer.pkl"), "rb") as f:
    enc = pickle.load(f)

special_set = set(SPECIAL_TOKENS)
token_bytes_list = []
for token_id in range(enc.n_vocab):
    token_str = enc.decode([token_id])
    token_bytes_list.append(0 if token_str in special_set else len(token_str.encode("utf-8")))

token_bytes = np.array(token_bytes_list, dtype=np.int32)
out = os.path.join(TOKENIZER_DIR, "token_bytes.npy")
np.save(out, token_bytes)
print(f"Saved {len(token_bytes_list)} token byte lengths to {out}")
EOF
```

### Step 2 — Train and save a checkpoint

```bash
# Terminal 1: run training (~6-7 min)
uv run train.py > run.log 2>&1

# Terminal 2: watch progress live
tail -f run.log
```

When done, `checkpoints/<commit_hash>.safetensors` and `checkpoints/<commit_hash>_config.json` are saved automatically.

### Step 3 — Verify the checkpoint

```bash
uv run generate.py --list-checkpoints
```

### Step 4 — Generate and save samples

```bash
# Uses best available checkpoint by default
uv run generate.py --prompt "Once upon a time"

# Use a specific checkpoint by commit hash
uv run generate.py --checkpoint a732c8c --prompt "Once upon a time"
uv run generate.py --checkpoint a732c8c --prompt "The history of machine learning began"
uv run generate.py --checkpoint a732c8c --prompt "Scientists recently discovered"
```

Each run saves two files to `samples/`:
- `commit-<hash>_bpb<score>_seed0_<prompt-slug>.txt` — the text shown to Prolific participants
- `commit-<hash>_bpb<score>_seed0_<prompt-slug>.json` — full metadata

Use `--no-save` to print to terminal only without saving.

---

## Next Steps for the Prolific Study

### Phase 1 — Curate prompts (done)

`scripts/curate_prompts.py` handles this. Run it once to produce `prompts.jsonl`.

### Phase 2 — Run autoresearch experiments from scratch

Follow `program.md` on a fresh branch. Aim for **8-10 kept commits** spanning the val_bpb improvement curve. Each kept commit automatically saves a checkpoint to `checkpoints/`.

Do not reuse checkpoints or val_bpb numbers from other machines — all comparisons must be on the same hardware.

### Phase 3 — Generate the dataset and select pairs

```bash
uv run scripts/generate_dataset.py   # → dataset.jsonl
python scripts/select_pairs.py       # → pairs.jsonl
```

Tune `STEP_GAP` in `select_pairs.py` after seeing the actual val_bpb spread.

### Phase 4 — Design the Prolific study

**Recommended task:** forced-choice A/B preference
- Show one prompt + two unlabeled completions (Model A vs Model B, randomized order)
- Ask: *"Which continuation reads more naturally?"*
- One question per page, ~10 questions per participant
- Target **50–100 participants** for statistical power

**What to measure:** % of participants preferring the better model (lower val_bpb). A result significantly above 50% validates that val_bpb improvements are perceptible to humans.

**Suggested sample size:** 10 prompts × 2 models × ~5 ratings each = ~100 judgments minimum. With 50 participants each doing 10 comparisons, you get 500 judgments total — enough for a clear signal.

### Phase 4 — Analyze results

After collecting responses, check:
1. Does the better model (val_bpb 1.808) win significantly more than 50% of pairwise comparisons?
2. Is the preference consistent across different prompt types?
3. Is the effect size large enough to be practically meaningful?
