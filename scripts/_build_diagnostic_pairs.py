"""
Build a 'diagnostic' pairs file for self-annotation.

Differs from select_pairs.py in two ways:
- Larger STEP_GAP (default 3) so adjacent val_bpb pairs aren't too close
- Globally shuffles the pairs so you see different checkpoint comparisons
  as you click through, rather than 20 prompts of the same comparison

Usage:
    uv run scripts/_build_diagnostic_pairs.py
    uv run scripts/_build_diagnostic_pairs.py --step-gap 4 --prompts-per-group 3
"""

import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--out", default="pairs_diagnostic.jsonl")
    ap.add_argument("--step-gap", type=int, default=3)
    ap.add_argument("--prompts-per-group", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    with open(args.dataset) as f:
        recs = [json.loads(l) for l in f if l.strip()]

    # Unique checkpoints sorted by val_bpb ascending (lowest = "best by metric")
    ckpts = sorted({(r["checkpoint_id"], r["val_bpb"]) for r in recs}, key=lambda x: x[1])
    if len(ckpts) <= args.step_gap:
        raise SystemExit(f"Only {len(ckpts)} checkpoints; step_gap={args.step_gap} is too large.")

    pair_groups = []
    for i in range(len(ckpts) - args.step_gap):
        pair_groups.append((ckpts[i], ckpts[i + args.step_gap]))

    print(f"Selected {len(pair_groups)} pair groups (step_gap={args.step_gap}):")
    for (cida, bpa), (cidb, bpb) in pair_groups:
        print(f"  {cida} (bpb={bpa:.4f}) vs {cidb} (bpb={bpb:.4f})  Δ={bpb - bpa:+.4f}")

    all_prompts = sorted({r["prompt_id"] for r in recs})
    lookup = {(r["checkpoint_id"], r["prompt_id"]): r for r in recs}

    out_records = []
    for (cida, bpa), (cidb, bpb) in pair_groups:
        sampled = rng.sample(all_prompts, min(args.prompts_per_group, len(all_prompts)))
        for pid in sampled:
            ra = lookup.get((cida, pid))
            rb = lookup.get((cidb, pid))
            if ra is None or rb is None:
                print(f"  WARN missing record for ({cida} or {cidb}) × {pid}")
                continue
            out_records.append({
                "prompt_id": pid,
                "prompt_text": ra["prompt_text"],
                "discipline": ra.get("discipline"),
                "checkpoint_a": cida,
                "val_bpb_a": bpa,
                "generated_text_a": ra["generated_text"],
                "checkpoint_b": cidb,
                "val_bpb_b": bpb,
                "generated_text_b": rb["generated_text"],
                "generation_params": ra["generation_params"],
            })

    rng.shuffle(out_records)

    with open(args.out, "w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(out_records)} pairs to {args.out} (shuffled)")


if __name__ == "__main__":
    main()
