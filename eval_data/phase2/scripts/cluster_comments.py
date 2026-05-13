"""
Cluster the free-text annotator comments thematically using Claude.

Writes report/comment_themes_llm.json which build_report.py will prefer
over keyword-bucketing when present. Re-run after new annotations land.

Usage:
    ANTHROPIC_API_KEY=... python cluster_comments.py
    ANTHROPIC_API_KEY=... python cluster_comments.py --in annotations_real.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent.parent
DEFAULT_IN = HERE / "report" / "annotations_real.json"
DEFAULT_OUT = HERE / "report" / "comment_themes_llm.json"

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are analyzing comments from human annotators in a pairwise preference study.

Annotators compared pairs of AI-generated responses to the same prompt and picked
which they preferred (A is better / About equal / B is better). They could
optionally leave a comment explaining why.

We'll feed you a batch of comments. Your job: cluster them into 5-8 distinct
thematic groups based on the QUALITY DIMENSION the annotator was reacting to.
Examples of useful themes: "concise and direct", "felt more natural / human",
"too verbose or rambling", "more accurate or factual", "better structured /
formatted", "showed empathy or warmth", "off-topic or missed the point",
"repetitive or stuck in a loop". Use the language annotators actually used.

For each theme, return:
  - a short label (3-6 words)
  - a one-sentence description of what annotators were reacting to
  - 2-3 representative quote excerpts from the actual comments (verbatim)
  - the total count of comments in that theme

Return strict JSON in this shape:
{
  "themes": [
    {
      "label": "Concise and direct",
      "description": "Annotators preferred responses that got to the point.",
      "count": 42,
      "quotes": ["A was more direct", "B was too rambling"]
    },
    ...
  ]
}

Order themes by count, descending. Do not invent quotes — use actual comment
text verbatim. If a comment doesn't fit any of your 5-8 themes, allocate it
to a final "Other / mixed feedback" theme. Aim for thematic coverage, not
keyword matching — quotes about "B was clunky" and "A felt smoother" belong
together even though they use different words.
"""


def collect_comments(annotations):
    out = []
    for a in annotations:
        c = (a.get("responses") or {}).get("comment", "").strip()
        if not c:
            continue
        out.append({
            "comment": c,
            "axis": a.get("axis"),
            "choice": a.get("choice_canonical"),
        })
    return out


def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def call_claude(client, comments, batch_idx, n_batches):
    """Send one batch of comments and get themes. Returns parsed JSON dict."""
    lines = []
    for i, c in enumerate(comments):
        # Light context: was this a disagreement (we don't tell the LLM which model)
        lines.append(f"[{i+1}] (axis={c['axis']}, picked={c['choice']}) {c['comment']}")
    body = "\n".join(lines)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Batch {batch_idx + 1} of {n_batches}. "
                f"Cluster these {len(comments)} comments into 5-8 thematic groups. "
                f"Return STRICT JSON only — no markdown, no preamble.\n\n{body}"
            ),
        }],
    )
    text = msg.content[0].text.strip()
    # Strip code fences if Claude added them
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    return json.loads(text)


def merge_batches(batch_results, all_comments):
    """Initial merge across batches by exact label match. Often produces 30-50
    themes due to slight label variations across batches; consolidate_themes()
    runs a final LLM pass to dedupe."""
    by_label = defaultdict(lambda: {"count": 0, "quotes": [], "descriptions": []})
    for r in batch_results:
        for t in r.get("themes", []):
            key = t["label"].strip().lower()
            by_label[key]["count"] += int(t.get("count", 0))
            by_label[key]["quotes"].extend(t.get("quotes", [])[:3])
            if t.get("description"):
                by_label[key]["descriptions"].append(t["description"])

    themes = []
    for key, data in sorted(by_label.items(), key=lambda kv: -kv[1]["count"]):
        themes.append({
            "label": key.title(),
            "description": data["descriptions"][0] if data["descriptions"] else "",
            "count": data["count"],
            "quotes": data["quotes"][:4],
        })
    return {"themes": themes}


CONSOLIDATE_PROMPT = """You are consolidating per-batch theme clusters from a pairwise preference study.
A previous LLM pass produced N themes across batches; many are near-duplicates that should be merged.

Merge into 6-9 final coherent themes. Combine duplicates ("Better Structure" and
"Clarity And Structure" → one theme). Preserve total counts (sum across merged
themes). Pick the best 2-4 quotes from the inputs for each final theme. Order
by total count, descending.

Return strict JSON only:
{
  "themes": [
    {"label": "...", "description": "...", "count": <int>, "quotes": ["...", "..."]},
    ...
  ]
}
"""


def consolidate_themes(client, raw_themes):
    """Second LLM pass: merge the 30-50 raw themes into 6-9 final ones."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=CONSOLIDATE_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Here are the {len(raw_themes)} per-batch themes to merge. "
                f"Consolidate into 6-9 final themes, summing counts and keeping the best quotes. "
                f"Return JSON only.\n\n"
                + json.dumps(raw_themes, ensure_ascii=False)
            ),
        }],
    )
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-size", type=int, default=120,
                        help="Comments per Claude call (lower = more API calls, less context per call)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment.")
        sys.exit(1)

    try:
        from anthropic import Anthropic
    except ImportError:
        print("Install with: pip install anthropic")
        sys.exit(1)

    annotations = json.load(open(args.in_path))
    comments = collect_comments(annotations)
    print(f"Loaded {len(annotations)} annotations, {len(comments)} non-empty comments")

    if not comments:
        print("No comments to cluster.")
        return

    client = Anthropic(api_key=api_key)
    batches = list(chunk(comments, args.batch_size))
    print(f"Processing {len(batches)} batches of up to {args.batch_size} comments each")

    batch_results = []
    for i, batch in enumerate(batches):
        print(f"  Batch {i+1}/{len(batches)} ({len(batch)} comments)...", flush=True)
        try:
            result = call_claude(client, batch, i, len(batches))
            batch_results.append(result)
            print(f"    -> {len(result.get('themes', []))} themes")
        except json.JSONDecodeError as e:
            print(f"    -> JSON parse error: {e}; skipping")
        except Exception as e:
            print(f"    -> ERROR: {e}; skipping")

    if not batch_results:
        print("No successful batches.")
        return

    merged = merge_batches(batch_results, comments)
    print(f"  Raw merged themes: {len(merged['themes'])}")
    if len(merged["themes"]) > 10:
        print(f"  Consolidating into 6-9 final themes…", flush=True)
        try:
            consolidated = consolidate_themes(client, merged["themes"])
            merged["themes"] = consolidated["themes"]
            print(f"  Consolidated to {len(merged['themes'])} themes")
        except Exception as e:
            print(f"  Consolidation failed ({e}); keeping raw merged themes")
    merged["n_comments"] = len(comments)
    merged["n_batches"] = len(batch_results)
    merged["model"] = MODEL

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2))
    print(f"\nWrote {len(merged['themes'])} themes to {out_path}")
    for t in merged["themes"][:8]:
        print(f"  · {t['label']}  ({t['count']} comments)")


if __name__ == "__main__":
    main()
