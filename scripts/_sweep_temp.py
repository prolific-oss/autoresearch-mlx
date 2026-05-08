"""
Quick sampling sweep across temperatures on a single checkpoint.
Used to pick the best temp for the final dataset regen after long training.

Usage:
    uv run scripts/_sweep_temp.py --checkpoint 84a8401-long
"""

import argparse
import json
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generate import load_model, sample_token
from prepare import Tokenizer

# A small fixed slice of prompts (one per discipline) so we see breadth.
SAMPLE_PROMPT_IDS = [
    "biomedical_00",
    "neuroscience_04",
    "psychology_social_08",
    "environmental_12",
    "physics_tech_16",
]
TEMPS = [0.3, 0.5, 0.8]
TOP_K = 50
SEED = 0
MAX_TOKENS = 150
STRIP_TRAILING_SPACE_PERIOD = True  # eLife/PLOS prompts end in " ." which causes dot-collapse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Tag (filename stem in checkpoints/)")
    parser.add_argument("--prompts", default="prompts.jsonl")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    weights = os.path.join(args.checkpoint_dir, f"{args.checkpoint}.safetensors")
    cfg_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint}_config.json")
    if not os.path.isfile(weights):
        raise SystemExit(f"Missing weights: {weights}")
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"Missing config: {cfg_path}")

    with open(args.prompts) as f:
        all_prompts = {json.loads(l)["prompt_id"]: json.loads(l) for l in f if l.strip()}
    prompts = [all_prompts[p] for p in SAMPLE_PROMPT_IDS if p in all_prompts]

    print(f"Loading {args.checkpoint}...")
    model, config = load_model(weights, cfg_path)
    tokenizer = Tokenizer.from_directory()

    for temp in TEMPS:
        print(f"\n{'='*72}\nTEMPERATURE = {temp}\n{'='*72}")
        for p in prompts:
            mx.random.seed(SEED)
            text = p["prompt_text"]
            if STRIP_TRAILING_SPACE_PERIOD:
                text = text.rstrip(" .\t\n")
            bos = tokenizer.get_bos_token_id()
            ids = mx.array([tokenizer.encode(text, prepend=bos)], dtype=mx.int32)
            output_tokens = []
            for _ in range(MAX_TOKENS):
                if ids.shape[1] >= config["sequence_len"]:
                    break
                logits = model(ids)
                mx.eval(logits)
                next_id = sample_token(logits[0, -1], temp, TOP_K)
                ids = mx.concatenate([ids, mx.array([[next_id]], dtype=mx.int32)], axis=1)
                output_tokens.append(next_id)
            text = tokenizer.decode(output_tokens)
            print(f"\n[{p['prompt_id']}] PROMPT: {p['prompt_text'][:90]}...")
            print(f"GEN: {text}")


if __name__ == "__main__":
    main()
