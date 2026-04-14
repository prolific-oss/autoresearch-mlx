"""
Curate 20 prompts from the tomasg25/scientific_lay_summarisation dataset.
Saves to prompts.jsonl in the repo root.

Usage:
    uv run scripts/curate_prompts.py
"""

import json
import random
import re
from datasets import load_dataset

SEED = 42
PER_DISCIPLINE = 4
OUT = "prompts.jsonl"

DISCIPLINES = {
    "biomedical":        ["cancer", "protein", "gene", "cell", "therapy", "disease", "drug", "immune", "infection", "virus", "vaccine", "genome"],
    "neuroscience":      ["neuron", "brain", "neural", "cortex", "synapse", "cognitive", "cerebral", "dopamine"],
    "psychology_social": ["psychology", "social", "behavior", "behaviour", "decision", "emotion", "mental health", "anxiety", "depression"],
    "environmental":     ["ecology", "environment", "climate", "species", "biodiversity", "ecosystem", "conservation", "ocean", "marine"],
    "physics_tech":      ["physics", "quantum", "algorithm", "computing", "machine learning", "material", "engineering", "robot", "chemistry"],
}

rng = random.Random(SEED)
buckets = {d: [] for d in DISCIPLINES}

for source in ("elife", "plos"):
    print(f"Loading {source}...")
    ds = load_dataset("tomasg25/scientific_lay_summarisation", source, trust_remote_code=True)
    for row in ds["train"]:
        keywords = (row.get("keywords") or "").lower()
        discipline = next((d for d, terms in DISCIPLINES.items() if any(t in keywords for t in terms)), None)
        if discipline is None:
            continue
        sentence = re.split(r'(?<=[.!?])\s+', row["summary"].strip())[0]
        if len(sentence) >= 20:
            buckets[discipline].append({
                "discipline":  discipline,
                "source":      source,
                "title":       row.get("title", ""),
                "prompt_text": sentence,
            })

prompts = []
for i, (discipline, candidates) in enumerate(buckets.items()):
    print(f"  {discipline}: {len(candidates)} candidates")
    if len(candidates) < PER_DISCIPLINE:
        print(f"  SKIP {discipline}: only {len(candidates)} candidates, need {PER_DISCIPLINE}")
        continue
    for row in rng.sample(candidates, PER_DISCIPLINE):
        prompts.append({
            "prompt_id":   f"{discipline}_{i:02d}",
            "prompt_text": row["prompt_text"],
            "discipline":  row["discipline"],
            "source":      row["source"],
            "title":       row["title"],
        })

with open(OUT, "w") as f:
    for p in prompts:
        f.write(json.dumps(p) + "\n")

print(f"\nSaved {len(prompts)} prompts to {OUT}")
for p in prompts:
    print(f"  [{p['discipline']}] {p['prompt_text'][:80]}...")
