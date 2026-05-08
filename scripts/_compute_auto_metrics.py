"""
Compute automated quality metrics on dataset.jsonl for triangulation
against the eventual Prolific human-preference data.

Cheap metrics (no extra deps):
- val_bpb               (already in each record — autoresearch's metric)
- word_count, char_count
- loop_4gram_frac       — fraction of repeated 4-grams
- unique_word_frac      — set(words)/len(words)
- punctuation_density   — fraction of tokens that are pure punctuation
- digit_density         — fraction of tokens containing digits
- avg_sentence_length   — words per sentence (split on .!?)
- avg_word_length       — chars per word

Optional reference-LM perplexity:
- ref_perplexity        — perplexity of generated text under SmolLM-135M-fp16
                          (~10x bigger than our model — meaningfully stronger
                          fluency reference, but small enough to stay light).
                          Requires `uv add mlx-lm` and ~260MB model download.

Usage:
    uv run scripts/_compute_auto_metrics.py
    uv run scripts/_compute_auto_metrics.py --no-perplexity      # skip ref-LM scoring
    uv run scripts/_compute_auto_metrics.py --ref-model mlx-community/SmolLM2-360M
"""

import argparse
import json
import math
import re
from collections import defaultdict
from statistics import mean


DATASET = "dataset.jsonl"
OUT_JSON = "auto_metrics.json"
OUT_TSV = "auto_metrics.tsv"
DEFAULT_REF_MODEL = "mlx-community/SmolLM-135M-fp16"

PUNCT_RE = re.compile(r"^[^\w\s]+$")
DIGIT_RE = re.compile(r".*\d.*")
SENT_SPLIT_RE = re.compile(r"[.!?]+\s+")


# ---------------------------------------------------------------------------
# Cheap text-level metrics
# ---------------------------------------------------------------------------

def loop_4gram_frac(text):
    words = text.split()
    if len(words) < 4:
        return 0.0
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return 1 - len(set(grams)) / len(grams)


def unique_word_frac(text):
    ws = text.split()
    return len(set(ws)) / len(ws) if ws else 0.0


def punctuation_density(text):
    ws = text.split()
    return sum(1 for w in ws if PUNCT_RE.match(w)) / len(ws) if ws else 0.0


def digit_density(text):
    ws = text.split()
    return sum(1 for w in ws if DIGIT_RE.match(w)) / len(ws) if ws else 0.0


def avg_sentence_length(text):
    sents = [s for s in SENT_SPLIT_RE.split(text) if s.strip()]
    if not sents:
        return 0.0
    return mean(len(s.split()) for s in sents)


def avg_word_length(text):
    ws = text.split()
    return mean(len(w) for w in ws) if ws else 0.0


CHEAP_METRICS = {
    "word_count": lambda t: len(t.split()),
    "char_count": len,
    "loop_4gram_frac": loop_4gram_frac,
    "unique_word_frac": unique_word_frac,
    "punctuation_density": punctuation_density,
    "digit_density": digit_density,
    "avg_sentence_length": avg_sentence_length,
    "avg_word_length": avg_word_length,
}


# ---------------------------------------------------------------------------
# Reference-LM perplexity (optional)
# ---------------------------------------------------------------------------

def make_perplexity_scorer(model_name):
    """Returns a callable(text) -> perplexity, or None if mlx_lm isn't installed."""
    try:
        from mlx_lm import load
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError:
        return None

    print(f"Loading reference LM: {model_name} (first run downloads weights)...")
    model, tokenizer = load(model_name)

    def score(text):
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            return float("nan")
        ids = mx.array([ids], dtype=mx.int32)
        inputs = ids[:, :-1]
        targets = ids[:, 1:]
        logits = model(inputs)
        # logits: (1, T, V); targets: (1, T)
        loss = nn.losses.cross_entropy(logits[0], targets[0], reduction="mean")
        mx.eval(loss)
        return math.exp(float(loss.item()))

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--no-perplexity", action="store_true",
                        help="Skip reference-LM perplexity (only compute cheap metrics).")
    parser.add_argument("--ref-model", default=DEFAULT_REF_MODEL,
                        help=f"mlx-lm model id (default: {DEFAULT_REF_MODEL})")
    args = parser.parse_args()

    with open(args.dataset) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(recs)} records from {args.dataset}")

    # Cheap metrics per record
    for r in recs:
        text = r["generated_text"]
        for name, fn in CHEAP_METRICS.items():
            r[f"_metric_{name}"] = fn(text)

    # Optional perplexity
    perplexity_done = False
    if not args.no_perplexity:
        scorer = make_perplexity_scorer(args.ref_model)
        if scorer is None:
            print("[ref_perplexity] mlx-lm not installed; skipping. "
                  "Install with `uv add mlx-lm` to enable.")
        else:
            print(f"Scoring perplexity for {len(recs)} records...")
            for i, r in enumerate(recs):
                r["_metric_ref_perplexity"] = scorer(r["generated_text"])
                if (i + 1) % 25 == 0:
                    print(f"  {i+1}/{len(recs)}")
            perplexity_done = True

    metric_keys = list(CHEAP_METRICS.keys())
    if perplexity_done:
        metric_keys.append("ref_perplexity")

    # Aggregate per checkpoint (mean over prompts)
    by_ckpt = defaultdict(list)
    for r in recs:
        by_ckpt[(r["checkpoint_id"], r["val_bpb"])].append(r)

    summary = []
    for (cid, bpb), rows in sorted(by_ckpt.items(), key=lambda x: x[0][1]):
        agg = {"checkpoint_id": cid, "val_bpb": bpb, "n_prompts": len(rows)}
        for k in metric_keys:
            vals = [r[f"_metric_{k}"] for r in rows
                    if not (isinstance(r[f"_metric_{k}"], float) and math.isnan(r[f"_metric_{k}"]))]
            agg[f"{k}_mean"] = mean(vals) if vals else float("nan")
        summary.append(agg)

    # Print table
    print(f"\n{'checkpoint':<20}{'val_bpb':>9}{'words':>8}{'loop':>7}{'div':>7}{'digit':>7}{'punct':>7}{'sent_l':>8}{'word_l':>8}", end="")
    if perplexity_done:
        print(f"{'ref_ppl':>10}", end="")
    print()
    for s in summary:
        line = (
            f"{s['checkpoint_id']:<20}"
            f"{s['val_bpb']:>9.4f}"
            f"{s['word_count_mean']:>8.1f}"
            f"{s['loop_4gram_frac_mean']:>7.3f}"
            f"{s['unique_word_frac_mean']:>7.3f}"
            f"{s['digit_density_mean']:>7.3f}"
            f"{s['punctuation_density_mean']:>7.3f}"
            f"{s['avg_sentence_length_mean']:>8.1f}"
            f"{s['avg_word_length_mean']:>8.2f}"
        )
        if perplexity_done:
            line += f"{s['ref_perplexity_mean']:>10.1f}"
        print(line)

    # Save outputs
    with open(OUT_JSON, "w") as f:
        json.dump({"per_record": recs, "per_checkpoint": summary}, f, indent=2)

    # TSV summary
    cols = ["checkpoint_id", "val_bpb", "n_prompts"] + [f"{k}_mean" for k in metric_keys]
    with open(OUT_TSV, "w") as f:
        f.write("\t".join(cols) + "\n")
        for s in summary:
            row = []
            for c in cols:
                v = s[c]
                if isinstance(v, float):
                    row.append(f"{v:.6f}")
                else:
                    row.append(str(v))
            f.write("\t".join(row) + "\n")

    print(f"\nWrote {OUT_JSON} and {OUT_TSV}")


if __name__ == "__main__":
    main()
