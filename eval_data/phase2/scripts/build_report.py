"""
Build the interactive HTML report for the Phase 2 study.

Single self-contained HTML file with 10 sections:
  1. Header
  2. TL;DR (3 paragraphs)
  3. The workflow we tested (Agent · Researcher · Prolific)
  4. The recipes (table with provenance tags)
  5. Headline result (chart + best-by-val vs best-by-humans)
  6. The interesting wrinkle (three-jumps along the val axis)
  7. What's behind it (GPT-4-as-judge mechanism + category breakdown)
  8. What humans saw that the metric didn't (themes + disagreement examples)
  9. What this means (three places HITL adds value)
  10. Appendix (per-pair table, BT ranking, IAA, category, methodology)

Reads from eval_data/phase2/ artifacts plus annotations_real.json (the
Prolific download via download_results.py).

Usage:
    python build_report.py                            # uses report/annotations_real.json
    python build_report.py --annotations other.json   # override the input
    python build_report.py --out custom_report.html
"""
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape as h
from pathlib import Path

HERE = Path(__file__).parent.parent
GRID_PATH = HERE / "prolific_pair_grid.json"
GENERATIONS_PATH = HERE / "prolific_generations.json"
PROMPTS_PATH = HERE / "prolific_prompts.json"
DEFAULT_ANNOTATIONS = HERE / "report" / "annotations_real.json"
DEFAULT_OUT = HERE / "report" / "index.html"

VAL_PREF_ACC = {
    "base": 0.500,  # untrained: policy = reference by construction → chance
    "prod-baseline": 0.464,
    "prod-adambeta2": 0.492,
    "prod-bestbet": 0.628,
    "prod-bestbet-v2": 0.648,
}
# Models actually trained (excluded from "untrained reference" comparisons)
TRAINED_MODELS = {"prod-baseline", "prod-adambeta2", "prod-bestbet", "prod-bestbet-v2"}
# Provenance: where each recipe came from in the workflow.
#   untrained   — SmolLM2-360M-Instruct, no DPO applied
#   agent       — produced by the autoresearch agent during its many-hour run
#   collab      — designed by the researcher inspecting the agent's results,
#                 still trained by the same DPO loop (agent + researcher)
PROVENANCE = {
    "base":            ("untrained",          "untrained"),
    "prod-baseline":   ("agent",              "agent"),
    "prod-adambeta2":  ("agent",              "agent"),
    "prod-bestbet":    ("agent + researcher", "collab"),
    "prod-bestbet-v2": ("agent + researcher", "collab"),
}
# Recipe names (without provenance tag — tag is rendered as a CSS pill alongside)
MODEL_DISPLAY = {
    "base":            "Original model",
    "prod-baseline":   "Recipe A · default DPO",
    "prod-adambeta2":  "Recipe B · agent's pick",
    "prod-bestbet":    "Recipe C · LoRA + filtered data",
    "prod-bestbet-v2": "Recipe D · deeper LoRA + filtered data",
}
# Short variant for chart axis labels
MODEL_SHORT = {
    "base":            "Original",
    "prod-baseline":   "Recipe A",
    "prod-adambeta2":  "Recipe B",
    "prod-bestbet":    "Recipe C",
    "prod-bestbet-v2": "Recipe D",
}
MODEL_DESC = {
    "base":            "SmolLM2-360M-Instruct as published — no extra training.",
    "prod-baseline":   "DPO at default knobs (β=0.1, lr=5e-7, 1 epoch over a sample of UltraFeedback). One of the agent's earlier checkpoints, kept as a reference.",
    "prod-adambeta2":  "The recipe the autoresearch agent settled on: constant_with_warmup LR + NEFTune + adam_β₂=0.95. The best of the agent's many checkpoints by val_pref_acc.",
    "prod-bestbet":    "Came from a researcher-prompted re-run: the researcher asked the agent to inspect its own trajectory and find what it was missing. The agent proposed LoRA (rank 32) + UltraFeedback filtered to high-margin preference pairs.",
    "prod-bestbet-v2": "Same as Recipe C with LoRA rank 64 (larger adapter — more capacity to adapt to UltraFeedback's chosen style).",
}
MODEL_ORDER = ["base", "prod-baseline", "prod-adambeta2", "prod-bestbet", "prod-bestbet-v2"]


def display_name(model_id):
    """Recipe name without provenance tag (pair the tag separately when needed)."""
    return MODEL_DISPLAY.get(model_id, model_id)


def short_name(model_id):
    """Compact name for charts / dense tables."""
    return MODEL_SHORT.get(model_id, model_id)


def provenance_tag(model_id):
    """Render a small CSS pill showing where this recipe came from."""
    label, css_class = PROVENANCE.get(model_id, ("?", "unknown"))
    return f'<span class="tag tag-{css_class}">{label}</span>'


def named(model_id):
    """Provenance tag + recipe name. Use this in prose and table cells."""
    return f"{provenance_tag(model_id)} {MODEL_DISPLAY.get(model_id, model_id)}"


def named_short(model_id):
    """Provenance tag + SHORT recipe name. For dense tables (per-pair, BT, IAA)."""
    return f"{provenance_tag(model_id)} {MODEL_SHORT.get(model_id, model_id)}"


# ──────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ──────────────────────────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """Wilson 95% CI for k of n successes."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - margin) / denom, (center + margin) / denom)


def fleiss_kappa(items):
    """items: list of dicts {label: count}, equal totals per item."""
    if not items:
        return None
    labels = sorted({l for d in items for l in d})
    n_raters = sum(items[0].values())
    if n_raters < 2:
        return None
    P_i = []
    for d in items:
        n = sum(d.values())
        if n != n_raters:
            return None
        P_i.append((sum(c * c for c in d.values()) - n) / (n * (n - 1)))
    P_bar = sum(P_i) / len(items)
    totals = Counter()
    for d in items:
        for l, c in d.items():
            totals[l] += c
    total_assignments = sum(totals.values())
    p_j = [totals[l] / total_assignments for l in labels]
    Pe = sum(p ** 2 for p in p_j)
    return (P_bar - Pe) / (1 - Pe) if Pe < 1 else 1.0


def bradley_terry(wins_matrix, models):
    """Iterative MLE (Hunter MM). Returns dict model → strength."""
    n = len(models)
    s = [1.0] * n
    for _ in range(500):
        new_s = [0.0] * n
        for i in range(n):
            num = sum(wins_matrix[i][j] for j in range(n) if j != i)
            denom = 0.0
            for j in range(n):
                if j == i:
                    continue
                nij = wins_matrix[i][j] + wins_matrix[j][i]
                if nij > 0:
                    denom += nij / (s[i] + s[j])
            new_s[i] = num / denom if denom > 0 else 0.001
        total = sum(new_s)
        new_s = [x / total * n for x in new_s]
        if max(abs(a - b) for a, b in zip(s, new_s)) < 1e-6:
            break
        s = new_s
    return {models[i]: s[i] for i in range(n)}


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate(annotations, grid):
    """
    Returns per unordered model-pair: model_x wins, model_y wins, ties.
    Canonical ordering: by MODEL_ORDER index (smaller index first).
    """
    grid_by_sid = {g["id"]: g for g in grid}
    # Also try opaque IDs (real data uses these)
    grid_by_opaque = {}
    import hashlib
    for g in grid:
        opaque = f"p2_{hashlib.sha256(g['id'].encode()).hexdigest()[:10]}"
        grid_by_opaque[opaque] = g

    def lookup_scenario(a):
        sid = a.get("scenario_id")
        return grid_by_opaque.get(sid) or grid_by_sid.get(sid) or {}

    # Pair stats keyed by tuple(sorted-by-MODEL_ORDER (mx, my))
    pair_stats = defaultdict(lambda: {"mx_wins": 0, "my_wins": 0, "ties": 0,
                                       "n": 0, "by_category": defaultdict(lambda: {"mx_wins": 0, "my_wins": 0, "ties": 0}),
                                       "ratings_by_scenario": defaultdict(Counter)})

    # Per-scenario rater label counts for IAA
    per_scenario_labels = defaultdict(Counter)

    for a in annotations:
        a_model = a.get("a_model")
        b_model = a.get("b_model")
        winner = a.get("choice_canonical")  # "A", "B", or "tie"
        if not (a_model and b_model and winner):
            continue
        scenario = lookup_scenario(a)
        cat = a.get("prompt_id") and scenario.get("prompt_id")
        # Pull category from scenario by mapping prompt_id → category
        prompt_id = scenario.get("prompt_id") or a.get("prompt_id")

        # Canonical ordering: smaller MODEL_ORDER index first
        i_a, i_b = MODEL_ORDER.index(a_model), MODEL_ORDER.index(b_model)
        if i_a <= i_b:
            mx, my = a_model, b_model
            # winner "A" = a_model = mx
            if winner == "A":
                outcome = "mx_wins"
            elif winner == "B":
                outcome = "my_wins"
            else:
                outcome = "ties"
        else:
            mx, my = b_model, a_model
            # winner "A" = a_model = my (since b is mx)
            if winner == "A":
                outcome = "my_wins"
            elif winner == "B":
                outcome = "mx_wins"
            else:
                outcome = "ties"

        ps = pair_stats[(mx, my)]
        ps[outcome] += 1
        ps["n"] += 1
        # Track per-scenario labels for IAA (with canonical orientation)
        if outcome == "mx_wins":
            label = "X"
        elif outcome == "my_wins":
            label = "Y"
        else:
            label = "T"
        sid = a.get("scenario_id")
        ps["ratings_by_scenario"][sid][label] += 1
        if prompt_id:
            ps["by_category"][prompt_id][outcome] += 1

    return pair_stats


def categorize_by_prompt(prompts_doc):
    """prompt_id → category name."""
    return {p["id"]: p["category"] for p in prompts_doc["prompts"]}


def compute_iaa_per_pair(pair_stats):
    """Fleiss' kappa per pair, using scenarios with 3 raters."""
    out = {}
    for (mx, my), ps in pair_stats.items():
        items = []
        for sid, counts in ps["ratings_by_scenario"].items():
            total = sum(counts.values())
            if total == 3:  # only complete scenarios
                items.append({"X": counts.get("X", 0), "Y": counts.get("Y", 0), "T": counts.get("T", 0)})
        if len(items) >= 5:
            k = fleiss_kappa(items)
            out[(mx, my)] = {"kappa": k, "n_items": len(items)}
    return out


def find_disagreement_pairs(pair_stats, grid, generations_doc, n_per_pair=2):
    """For each pair, find scenarios where humans most strongly preferred the
    LOWER-val_pref_acc model. Returns list of (mx, my, scenario_dict, vote_breakdown)."""
    out = []
    grid_by_sid = {g["id"]: g for g in grid}
    import hashlib
    grid_by_opaque = {f"p2_{hashlib.sha256(g['id'].encode()).hexdigest()[:10]}": g for g in grid}

    for (mx, my), ps in pair_stats.items():
        # val_pref_acc: my is "higher" if my has greater val (base treated as 0)
        v_mx = VAL_PREF_ACC.get(mx) or 0.0
        v_my = VAL_PREF_ACC.get(my) or 0.0
        if v_my <= v_mx:
            continue  # only look at pairs where my > mx in val_pref_acc
        # Disagreement = scenario where mx (lower val) won
        candidates = []
        for sid, counts in ps["ratings_by_scenario"].items():
            mx_wins = counts.get("X", 0)
            my_wins = counts.get("Y", 0)
            ties = counts.get("T", 0)
            total = mx_wins + my_wins + ties
            if total == 0:
                continue
            # Score: how strongly humans went against val_pref_acc on this scenario
            disagreement_score = mx_wins - my_wins
            scenario_meta = grid_by_opaque.get(sid) or grid_by_sid.get(sid)
            if scenario_meta:
                candidates.append({
                    "sid": sid, "score": disagreement_score,
                    "mx_wins": mx_wins, "my_wins": my_wins, "ties": ties,
                    "scenario": scenario_meta,
                })
        candidates.sort(key=lambda c: -c["score"])
        for c in candidates[:n_per_pair]:
            if c["score"] >= 1:  # at least mild disagreement
                out.append((mx, my, c))
    out.sort(key=lambda x: -x[2]["score"])
    return out


def per_category_breakdown(pair_stats, prompt_categories):
    """Returns dict: category → {(mx,my) → {mx_wins, my_wins, ties}}."""
    out = defaultdict(lambda: defaultdict(lambda: {"mx_wins": 0, "my_wins": 0, "ties": 0}))
    for (mx, my), ps in pair_stats.items():
        for prompt_id, cat_stats in ps["by_category"].items():
            cat = prompt_categories.get(prompt_id, "other")
            for k, v in cat_stats.items():
                out[cat][(mx, my)][k] += v
    return out


def characterize_failures(annotations, grid, prompt_categories):
    """Programmatic characterization of disagreement scenarios.

    Returns a dict with:
      - per-category disagreement rate (humans pref lower-val)
      - length ratio in agreement vs disagreement scenarios
      - top comment themes among disagreements
    """
    import hashlib, re
    grid_by_opaque = {f"p2_{hashlib.sha256(g['id'].encode()).hexdigest()[:10]}": g for g in grid}

    by_sid = defaultdict(list)
    for a in annotations:
        by_sid[a["scenario_id"]].append(a)

    agreement = []
    disagreement = []
    for sid, anns in by_sid.items():
        meta = grid_by_opaque.get(sid)
        if not meta:
            continue
        a_model, b_model = meta["a_model"], meta["b_model"]
        a_val = VAL_PREF_ACC.get(a_model) or 0.5
        b_val = VAL_PREF_ACC.get(b_model) or 0.5
        if a_val == b_val:
            continue
        if a_val <= b_val:
            mx_text, my_text, mx_model, my_model = meta["a_text"], meta["b_text"], a_model, b_model
        else:
            mx_text, my_text, mx_model, my_model = meta["b_text"], meta["a_text"], b_model, a_model
        mx_wins = sum(1 for a in anns if
                      (a.get("choice_canonical") == "A" and a.get("a_model") == mx_model) or
                      (a.get("choice_canonical") == "B" and a.get("b_model") == mx_model))
        my_wins = sum(1 for a in anns if
                      (a.get("choice_canonical") == "A" and a.get("a_model") == my_model) or
                      (a.get("choice_canonical") == "B" and a.get("b_model") == my_model))
        if mx_wins == my_wins:
            continue
        record = {
            "category": prompt_categories.get(meta["prompt_id"], "other"),
            "mx_len": len(mx_text), "my_len": len(my_text),
            "comments": [(a.get("responses", {}) or {}).get("comment", "")
                          for a in anns if (a.get("responses", {}) or {}).get("comment")],
        }
        (disagreement if mx_wins > my_wins else agreement).append(record)

    # Per-category disagreement rate
    cat_totals = Counter()
    cat_disagree = Counter()
    for r in disagreement + agreement:
        cat_totals[r["category"]] += 1
    for r in disagreement:
        cat_disagree[r["category"]] += 1
    per_cat_rate = [
        {"category": c, "total": cat_totals[c], "disagree": cat_disagree[c],
         "rate": cat_disagree[c] / cat_totals[c] if cat_totals[c] else 0}
        for c in sorted(cat_totals, key=lambda c: -cat_disagree[c] / cat_totals[c] if cat_totals[c] else 0)
    ]

    # Length stats
    def mean_or_none(xs):
        return (sum(xs) / len(xs)) if xs else None
    len_agree_my = mean_or_none([r["my_len"] for r in agreement])
    len_agree_mx = mean_or_none([r["mx_len"] for r in agreement])
    len_disag_my = mean_or_none([r["my_len"] for r in disagreement])
    len_disag_mx = mean_or_none([r["mx_len"] for r in disagreement])

    # Comment-theme buckets specifically for disagreements (used as fallback
    # when LLM-clustered themes aren't available)
    themes = [
        ("concise / direct",         ["concis", "short", "to the point", "direct", "succinct"]),
        ("natural / human",          ["natural", "human", "warm", "kind", "genuine", "personal"]),
        ("verbose / wordy",          ["verbose", "long", "wordy", "rambl", "padd", "filler", "way too"]),
        ("off-topic",                ["off-topic", "off topic", "miss", "tangent", "doesn't address", "irrelevant"]),
        ("repetitive",               ["repet", "stuck", "loop", "same thing"]),
        ("better explanation",       ["explain", "clear", "specific", "informative"]),
        ("structured / formatted",   ["list", "bullet", "structur", "format", "numbered"]),
        ("robotic / cold",           ["robot", "stiff", "mechan", "formal", "cold", "AI"]),
    ]
    theme_counts = Counter()
    theme_samples = defaultdict(list)
    for r in disagreement:
        for c in r["comments"]:
            lower = c.lower()
            for theme, kws in themes:
                if any(kw in lower for kw in kws):
                    theme_counts[theme] += 1
                    if len(theme_samples[theme]) < 2:
                        theme_samples[theme].append(c)
                    break  # one theme per comment

    return {
        "n_disagreement": len(disagreement),
        "n_agreement": len(agreement),
        "per_category_disagreement": per_cat_rate,
        "length_ratio_agreement": (len_agree_my / len_agree_mx) if len_agree_mx else None,
        "length_ratio_disagreement": (len_disag_my / len_disag_mx) if len_disag_mx else None,
        "disagreement_themes": [
            {"theme": t, "count": n, "samples": theme_samples[t]}
            for t, n in theme_counts.most_common()
        ],
    }


# Keyword buckets for comment-theme analysis. Each entry: (theme_label, [keywords])
COMMENT_THEMES = [
    ("More concise / direct",        ["concis", "direct", "succinct", "short", "to the point", "not verbose"]),
    ("Too verbose / long-winded",    ["verbose", "long-winded", "wordy", "too long", "rambl", "padd", "filler"]),
    ("More natural / readable",      ["natural", "readable", "flow", "easy to read", "conversational"]),
    ("Hallucination / wrong facts",  ["wrong", "incorrect", "hallucin", "made up", "fabricat", "inaccurat", "false"]),
    ("Repetitive / mode collapse",   ["repetit", "repeat", "stuck", "loop", "same thing", "over and over"]),
    ("More on-topic / relevant",     ["on-topic", "on topic", "relevant", "answered the", "addressed the"]),
    ("Off-topic / missed point",     ["off-topic", "off topic", "missed", "didn't answer", "doesn't address", "tangent"]),
    ("More empathetic / warm",       ["empath", "kind", "warm", "compassion", "gentle", "thoughtful"]),
    ("Robotic / cold",               ["robot", "cold", "stiff", "mechan", "bullet", "list"]),
    ("Better structure / format",    ["structure", "organized", "format", "clear", "well-laid"]),
    ("Confident / clear",            ["confident", "definitive", "clear", "decisive"]),
    ("Hedging / uncertain",          ["hedg", "uncertain", "wishy", "wishy-washy", "vague", "unclear"]),
]


def extract_comment_themes(annotations, pair_stats):
    """Bucket annotator comments by theme and tie them to the canonical winner.

    Returns:
        global_themes: list of (theme, count, sample_comments)
        per_pair_themes: dict (mx,my) -> {theme: count}
    """
    def comment_of(a):
        c = (a.get("responses") or {}).get("comment", "")
        if not c:
            c = a.get("comment", "")
        return (c or "").strip()

    global_counts = Counter()
    sample_comments = defaultdict(list)
    per_pair_counts = defaultdict(Counter)

    for a in annotations:
        text = comment_of(a)
        if not text:
            continue
        lower = text.lower()
        for theme, keywords in COMMENT_THEMES:
            if any(kw in lower for kw in keywords):
                global_counts[theme] += 1
                if len(sample_comments[theme]) < 3:
                    sample_comments[theme].append(text)
                a_model, b_model = a.get("a_model"), a.get("b_model")
                if a_model and b_model:
                    i_a, i_b = MODEL_ORDER.index(a_model), MODEL_ORDER.index(b_model)
                    mx, my = (a_model, b_model) if i_a <= i_b else (b_model, a_model)
                    per_pair_counts[(mx, my)][theme] += 1

    global_themes = [
        {"theme": t, "count": c, "samples": sample_comments[t]}
        for t, c in global_counts.most_common()
    ]
    return global_themes, per_pair_counts


# ──────────────────────────────────────────────────────────────────────────────
# HTML rendering
# ──────────────────────────────────────────────────────────────────────────────

CSS = """
:root {
  --fg: #1a1a1a;
  --muted: #5b5b5b;
  --bg: #ffffff;
  --panel: #f7f7f4;
  --accent: #b85c00;
  --accent-soft: #fcefe1;
  --border: #d8d6d2;
  --good: #1f6f3a;
  --bad: #b8412f;
  --tie: #888;
  --hl-a: #2f6fb8;
  --hl-b: #b8602f;
}
* { box-sizing: border-box; }
body {
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  color: var(--fg);
  background: var(--bg);
  margin: 0; padding: 0;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 40px 28px 80px; }
header { margin-bottom: 36px; }
header h1 { font-size: 28px; margin: 0 0 6px; letter-spacing: -0.01em; }
header .subtitle { color: var(--muted); font-size: 15px; }
header .meta { color: var(--muted); font-size: 12px; margin-top: 12px;
               font-variant-numeric: tabular-nums; }
.section { margin: 38px 0; }
.section h2 { font-size: 20px; margin: 0 0 14px; letter-spacing: -0.005em; }
.section h3 { font-size: 16px; margin: 22px 0 8px; color: var(--muted); }
p { margin: 0 0 10px; }

.theme-sample {
  background: white; border-left: 2px solid var(--accent);
  padding: 6px 12px; margin: 6px 0; font-size: 13px;
  color: var(--muted); font-style: italic;
}

table { width: 100%; border-collapse: collapse; font-size: 14px;
        margin: 8px 0 18px; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }
th { background: var(--panel); font-weight: 600; font-size: 12px;
     text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:hover td { background: rgba(184,92,0,0.04); }
.pill {
  display: inline-block; padding: 2px 8px; border-radius: 10px;
  font-size: 11px; font-weight: 600;
}
.pill.good { background: #d8f0e0; color: var(--good); }
.pill.bad  { background: #f4dcd0; color: var(--bad); }
.pill.tie  { background: #e8e8e8; color: var(--tie); }
.kappa {
  display: inline-block; padding: 2px 6px;
  font-family: ui-monospace, monospace; font-size: 12px;
  background: var(--panel); border-radius: 3px;
  white-space: nowrap;
}
.bar {
  display: inline-block; height: 10px; vertical-align: middle;
  background: var(--accent); border-radius: 2px;
}
.bar-bg { background: var(--panel); border-radius: 2px;
          display: inline-block; vertical-align: middle; }
details {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; padding: 12px 18px; margin: 10px 0;
}
details > summary { cursor: pointer; font-weight: 500; color: var(--fg);
                    list-style: none; padding: 4px 0; }
details > summary::-webkit-details-marker { display: none; }
details > summary::before { content: "▸ "; color: var(--muted); margin-right: 4px;
                             transition: transform 0.15s; display: inline-block; }
details[open] > summary::before { transform: rotate(90deg); }

.disagree-pair { background: white; border: 1px solid var(--border);
                  padding: 14px 16px; border-radius: 4px; margin: 12px 0; }
.disagree-pair .prompt-text {
  font-style: italic; color: var(--muted); margin-bottom: 12px;
  border-left: 2px solid var(--border); padding-left: 12px;
}
.disagree-pair .responses { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.response-card { padding: 12px 14px; border-radius: 4px; }
.response-card.winner { background: #f0fbf4; border: 1px solid #bfe5cc; }
.response-card.loser  { background: #fff8f5; border: 1px solid #e8c8b8; }
.response-card .role  { font-size: 11px; font-weight: 700; color: var(--muted);
                         text-transform: uppercase; letter-spacing: 0.06em;
                         margin-bottom: 6px; }
.response-card .text  { font-size: 13.5px; line-height: 1.55; white-space: pre-wrap;
                         word-break: break-word; max-height: 280px; overflow-y: auto; }
.disagree-pair .vote-summary {
  font-size: 13px; color: var(--muted); margin-top: 10px;
  font-variant-numeric: tabular-nums;
}

.method { background: var(--panel); padding: 18px 22px; border-radius: 6px;
          font-size: 14px; line-height: 1.6; }
.method ul { padding-left: 20px; }

footer { color: var(--muted); font-size: 12px; margin-top: 60px;
         padding-top: 18px; border-top: 1px solid var(--border); }

/* Provenance pills — applied next to recipe names everywhere */
.tag {
  display: inline-block; font-size: 10px; font-weight: 600;
  letter-spacing: 0.04em; text-transform: uppercase;
  padding: 2px 7px; border-radius: 3px; vertical-align: 1px;
  margin-right: 6px;
}
.tag-untrained { background: #e8e6e1; color: #5b5b5b; }
.tag-agent     { background: #dbe7f5; color: #2f6fb8; }
.tag-collab    { background: #fcefe1; color: #b85c00; }

/* TL;DR card replacing the verdict + takeaways list */
.tldr {
  background: #f7f7f4; border-left: 3px solid #1a1a1a;
  padding: 20px 26px; margin: 18px 0 28px;
  border-radius: 4px;
}
.tldr p { font-size: 15.5px; line-height: 1.6; margin: 0 0 12px; }
.tldr p:last-child { margin-bottom: 0; }

/* Three-stage workflow grid */
.workflow {
  display: grid; grid-template-columns: 7fr 7fr 6fr; gap: 14px;
  margin: 16px 0 22px;
}
.stage {
  background: var(--panel); border: 1px solid var(--border);
  padding: 14px 16px; border-radius: 4px;
}
.stage .stage-label {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--muted);
  margin-bottom: 8px;
}
.stage h3 { font-size: 15px; margin: 0 0 8px; color: var(--fg); }
.stage p  { font-size: 13.5px; line-height: 1.55; margin: 0; }
@media (max-width: 720px) {
  .workflow { grid-template-columns: 1fr; }
}
"""

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def truncate(text, n=900):
    if not text:
        return ""
    return text if len(text) <= n else text[:n - 1].rsplit(" ", 1)[0] + "…"


def render_html(report_data, annotations_path, mock_mode):
    """Stitch the 10-section HTML report from computed report_data."""
    sections = []

    bt = report_data["bt"]
    bt_dict = report_data["bt_dict"]
    pair_table = report_data["per_pair"]
    spearman = report_data["spearman"]
    n_annotations = report_data["n_annotations"]
    n_annotators = report_data["n_annotators"]

    best_val_id = max(TRAINED_MODELS, key=lambda m: VAL_PREF_ACC[m])
    best_human_id = bt[0][0]

    # Each model's win-rate vs `base` (used in headline chart + 3-jump table)
    vs_base = {}
    for row in pair_table:
        if row["mx"] == "base":
            vs_base[row["my"]] = row["my_pref_excl_ties"]
        elif row["my"] == "base":
            vs_base[row["mx"]] = row["mx_pref_excl_ties"]

    chart_models = [m for m in MODEL_ORDER if m != "base"]
    chart_x = [VAL_PREF_ACC[m] for m in chart_models]
    chart_y = [vs_base.get(m, 0.5) for m in chart_models]
    chart_text = [MODEL_SHORT[m] for m in chart_models]
    # Color each point by provenance — matches the .tag pill colors so the
    # chart visually distinguishes agent-only from agent+researcher recipes.
    PROVENANCE_COLOR = {"agent": "#2f6fb8", "collab": "#b85c00", "untrained": "#5b5b5b"}
    chart_colors = [PROVENANCE_COLOR.get(PROVENANCE[m][1], "#888") for m in chart_models]

    # ── 1. Header ──
    sections.append(f"""
<header>
  <h1>When does an AI metric agree with humans?</h1>
  <div class="subtitle">A Prolific preference study on 5 DPO recipes for SmolLM2-360M-Instruct.</div>
  <div class="meta">
    Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
    · {n_annotations} annotations
    · {n_annotators} annotators
    {('· MOCK DATA' if mock_mode else '')}
  </div>
</header>""")

    # ── 2. TL;DR ──
    sections.append(f"""
<section class="section">
  <h2>TL;DR</h2>
  <div class="tldr">
    <p>We trained <strong>SmolLM2-360M-Instruct</strong> on UltraFeedback preference
    pairs using <a href="https://github.com/karpathy/autoresearch">Karpathy's autoresearch</a>
    — a Claude agent that edits a training script, runs experiments, and iterates
    against a held-out preference-accuracy metric (<code>val_pref_acc</code>).
    The agent ran 50 experiments and produced many checkpoints, rejecting most
    because they didn't beat the starting recipe; we kept 2 of its best, within
    a small val range, since the agent didn't push the metric far. A researcher
    then asked the agent to inspect what it had tried and look for gaps. The
    agent's response — LoRA adapters and high-margin data filtering, categories
    it hadn't explored on its own — produced 2 more recipes.
    Prolific annotators ran all 10 pairwise head-to-heads on 50 general-audience
    prompts ({n_annotations} ratings).</p>

    <p><strong>The agent's autonomous run did not improve the metric.</strong>
    On its own, the autoresearch agent never lifted <code>val_pref_acc</code>
    above the untrained reference's level of 0.500 — Recipe A
    (val={VAL_PREF_ACC['prod-baseline']:.3f}) and Recipe B
    (val={VAL_PREF_ACC['prod-adambeta2']:.3f}, the agent's best across 50
    experiments) both sit <em>below</em> chance. Humans agreed: in head-to-heads
    against the untrained base, annotators preferred the base ~{int((1 - (vs_base.get('prod-adambeta2') or 0.5)) * 100)}%
    of the time over Recipe B. Standard full-FT DPO at the hyperparameters
    the agent explored was, by both metric and humans, a regression.</p>

    <p><strong>The researcher's meta-prompt unlocked the only recipes that
    beat the untrained base.</strong> Once the researcher asked the agent to
    look for gaps, the agent proposed LoRA adapters + high-margin data
    filtering, lifting val_pref_acc from 0.492 to
    {VAL_PREF_ACC['prod-bestbet-v2']:.3f}. Spearman ρ between
    <code>val_pref_acc</code> and human-preference rank is
    <strong>{spearman:+.2f}</strong> — the broad ordering matches. But at the
    top the metric and humans disagree: the highest-scoring recipe
    ({named(best_val_id)}, val={VAL_PREF_ACC[best_val_id]:.3f}) is
    <em>not</em> the one humans prefer most ({named(best_human_id)},
    val={VAL_PREF_ACC.get(best_human_id, 0.5):.3f}).</p>

    <p><strong>HITL moved the result at two stages.</strong> Research-time:
    the researcher's meta-prompt was the difference between regression and
    improvement. Evaluation-time: Prolific annotators disambiguated the top
    two recipes the metric couldn't tell apart. Without either step, we'd
    have shipped a noticeably worse model.</p>
  </div>
</section>""")

    # ── 3. The workflow we tested ──
    sections.append(f"""
<section class="section">
  <h2>The workflow we tested</h2>
  <p>Three stages, three contributors. Each tried to make the model better;
  each had a blind spot the next stage filled in.</p>
  <div class="workflow">
    <div class="stage">
      <div class="stage-label">{provenance_tag('prod-baseline')} Stage 1</div>
      <h3>Search the recipe space</h3>
      <p>The autoresearch agent edited <code>train.py</code>, launched DPO
      runs, and iterated across <strong>50 experiments</strong> (45 produced
      valid output; 5 failed). It rejected most checkpoints because they
      didn't beat the starting recipe. We kept 2 of its best
      ({VAL_PREF_ACC['prod-baseline']:.3f}–{VAL_PREF_ACC['prod-adambeta2']:.3f}).
      Both stayed <em>below</em> the untrained reference's chance level
      (0.500) — the agent's autonomous search did not lift the metric over
      no training at all.</p>
    </div>
    <div class="stage">
      <div class="stage-label">{provenance_tag('prod-bestbet')} Stage 2</div>
      <h3>Ask the agent what's missing</h3>
      <p>A researcher prompted the agent to inspect its own trajectory and
      identify what it hadn't tried. The agent's response added LoRA adapters
      and score-margin-filtered UltraFeedback — categories of change it hadn't
      explored on its own. The same DPO loop with those additions produced
      Recipes C and D, lifting val_pref_acc to
      {VAL_PREF_ACC['prod-bestbet']:.3f}–{VAL_PREF_ACC['prod-bestbet-v2']:.3f}.</p>
    </div>
    <div class="stage">
      <div class="stage-label" style="color:#444;">Stage 3 · Prolific annotators</div>
      <h3>Test with humans</h3>
      <p>Five models × 10 pairs × 50 prompts × 3 annotators =
      {n_annotations} pairwise judgements from {n_annotators} participants.
      Annotators saw responses blind — no labels, no provenance — and picked
      the one they preferred or called it a tie.</p>
    </div>
  </div>
</section>""")

    # ── 4. The recipes ──
    setup_rows = []
    for m in MODEL_ORDER:
        val = VAL_PREF_ACC[m]
        val_str = f"{val:.3f}" if val is not None else "—"
        setup_rows.append(f"""
<tr>
  <td>{provenance_tag(m)}<strong>{h(MODEL_DISPLAY[m])}</strong>
      <div style="font-size:11px;color:var(--muted);margin-top:2px;font-family:ui-monospace,monospace;">{h(m)}</div></td>
  <td class='num'>{val_str}</td>
  <td style="font-size:13px;">{h(MODEL_DESC[m])}</td>
</tr>""")
    sections.append(f"""
<section class="section">
  <h2>The recipes</h2>
  <p>Five models. The pill next to each name shows which stage produced it.</p>
  <table style="table-layout: fixed;">
    <colgroup>
      <col style="width: 38%;">
      <col style="width: 10%;">
      <col style="width: 52%;">
    </colgroup>
    <thead><tr><th>Recipe</th><th class='num'>val_pref_acc</th><th>What it is</th></tr></thead>
    <tbody>{''.join(setup_rows)}</tbody>
  </table>
</section>""")

    # ── 5. Headline result ──
    sections.append(f"""
<section class="section">
  <h2>Headline result</h2>
  <p>If <code>val_pref_acc</code> were a perfect predictor of human preference,
  the four trained recipes would line up monotonically along the dotted "chance"
  line — higher score, more often preferred over the untrained base. Here's
  what {n_annotations} pairwise ratings actually show:</p>
  <div id="headline-chart" style="height:420px;"></div>
  <p style="color: var(--muted); font-size: 13px; margin-top: 14px;">
    Points colored by provenance: {provenance_tag('prod-baseline')}agent
    recipes (A, B) · {provenance_tag('prod-bestbet')}agent + researcher
    recipes (C, D). Dotted lines mark chance: horizontal = 50% human
    preference vs base, vertical = val_pref_acc of the untrained reference.
  </p>
  <p>Three things stand out:</p>
  <ul>
    <li><strong>The agent's runs alone made the model worse than no training.</strong>
        Recipes A and B both score <em>below</em> 0.5 on val_pref_acc — by the
        autoresearch agent's own metric, they're less aligned with
        UltraFeedback's preference labels than the untrained reference (which
        sits at 0.5 by construction). Humans agree: against the untrained
        base, annotators preferred the base ~{int((1 - (vs_base.get('prod-baseline') or 0.5)) * 100)}%
        of the time over Recipe A and ~{int((1 - (vs_base.get('prod-adambeta2') or 0.5)) * 100)}%
        of the time over Recipe B.</li>
    <li><strong>Only the researcher-prompted recipes cross the line.</strong>
        Recipes C and D — the ones the agent proposed after the researcher
        asked it to look for gaps — are the only DPO runs that humans prefer
        over the untrained base, and only by a small margin
        ({int((vs_base.get('prod-bestbet') or 0.5) * 100)}% and
        {int((vs_base.get('prod-bestbet-v2') or 0.5) * 100)}% respectively).</li>
    <li><strong>At the top, the metric and humans disagree.</strong> Spearman
        ρ between val_pref_acc and human preference is
        <strong>{spearman:+.2f}</strong> — the broad ordering matches. But
        within the agent + researcher pair, the metric ranks D higher
        ({named(best_val_id)}, val={VAL_PREF_ACC[best_val_id]:.3f}) while
        humans pick {named(best_human_id)} (BT strength
        {bt_dict.get(best_human_id, 0):.2f}, highest of any recipe).</li>
  </ul>
  <p>So the headline isn't "DPO works." It's: <em>the agent's autonomous
  recipe search regressed the model; the researcher's meta-prompt unlocked
  the only recipes that beat the untrained base; and within those, humans
  picked a different winner than the metric.</em></p>
</section>""")

    # ── 6. The interesting wrinkle ──
    sections.append(f"""
<section class="section">
  <h2>The interesting wrinkle</h2>
  <p>{named('prod-bestbet')} and {named('prod-bestbet-v2')} are the same recipe
  with one knob changed: LoRA rank 32 vs 64 (D has more adapter capacity). By
  the metric, D is the better model — val_pref_acc rises from
  {VAL_PREF_ACC['prod-bestbet']:.3f} to {VAL_PREF_ACC['prod-bestbet-v2']:.3f}.
  By humans, D is the <em>worse</em> model — Bradley-Terry strength drops from
  {bt_dict.get('prod-bestbet', 0):.2f} to {bt_dict.get('prod-bestbet-v2', 0):.2f}.</p>

  <p>Tracking three steps along the val_pref_acc axis shows how the metric's
  signal changes character as it climbs:</p>
  <table style="table-layout: fixed;">
    <colgroup>
      <col style="width: 38%;">
      <col style="width: 14%;">
      <col style="width: 18%;">
      <col style="width: 30%;">
    </colgroup>
    <thead><tr>
      <th>From → To</th>
      <th class='num'>Δ val_pref_acc</th>
      <th class='num'>Human pref vs base</th>
      <th>Interpretation</th>
    </tr></thead>
    <tbody>
      <tr>
        <td>{named('prod-baseline')} → {named('prod-adambeta2')}</td>
        <td class='num'>+{VAL_PREF_ACC['prod-adambeta2'] - VAL_PREF_ACC['prod-baseline']:.3f}</td>
        <td class='num'>{(vs_base.get('prod-baseline') or 0.5)*100:.0f}% → {(vs_base.get('prod-adambeta2') or 0.5)*100:.0f}%</td>
        <td>Small metric gain, no clear effect on humans.</td>
      </tr>
      <tr>
        <td>{named('prod-adambeta2')} → {named('prod-bestbet')}</td>
        <td class='num'>+{VAL_PREF_ACC['prod-bestbet'] - VAL_PREF_ACC['prod-adambeta2']:.3f}</td>
        <td class='num'>{(vs_base.get('prod-adambeta2') or 0.5)*100:.0f}% → {(vs_base.get('prod-bestbet') or 0.5)*100:.0f}%</td>
        <td><strong>Big metric gain, clearly visible to humans.</strong></td>
      </tr>
      <tr>
        <td>{named('prod-bestbet')} → {named('prod-bestbet-v2')}</td>
        <td class='num'>+{VAL_PREF_ACC['prod-bestbet-v2'] - VAL_PREF_ACC['prod-bestbet']:.3f}</td>
        <td class='num'>{(vs_base.get('prod-bestbet') or 0.5)*100:.0f}% → {(vs_base.get('prod-bestbet-v2') or 0.5)*100:.0f}%</td>
        <td><strong>Small metric gain, humans <em>reverse</em>.</strong></td>
      </tr>
    </tbody>
  </table>
  <p>The pattern: the metric is useful for finding the right neighborhood and
  even tracks human preference within it. At the very top it starts pointing
  at the wrong door.</p>
</section>""")

    # ── 7. What's behind it ──
    fa = report_data["failure_analysis"]
    cat_rows = []
    for c in fa["per_category_disagreement"]:
        pct = c["rate"] * 100
        bar_color = "#b8412f" if pct >= 50 else "#b85c00" if pct >= 40 else "#888"
        bar_w = int(pct * 1.6)
        cat_rows.append(f"""
<tr>
  <td>{h(c['category'].replace('_', ' '))}</td>
  <td class='num'>{c['disagree']} / {c['total']}</td>
  <td style='white-space: nowrap;'>
    <span class='bar-bg' style='width:140px;'>
      <span class='bar' style='width:{bar_w}px;background:{bar_color};'></span>
    </span>
    <span style='margin-left:8px;font-variant-numeric:tabular-nums;font-size:13px;color:{bar_color};font-weight:600;'>{pct:.0f}%</span>
  </td>
</tr>""")
    len_agree = fa["length_ratio_agreement"]
    len_disag = fa["length_ratio_disagreement"]
    len_para = ""
    if len_agree and len_disag:
        len_para = (f"<p>Length isn't the whole story either. Higher-val responses are "
                    f"<strong>{len_agree:.2f}×</strong> longer than lower-val ones in "
                    f"scenarios where humans agreed with the metric, and "
                    f"<strong>{len_disag:.2f}×</strong> longer in scenarios where humans "
                    f"<em>disagreed</em>. Humans aren't blanket-rejecting longer responses "
                    f"— they reject extra length on prompts where it doesn't earn its keep.</p>")

    sections.append(f"""
<section class="section">
  <h2>What's behind it</h2>
  <p>UltraFeedback's "chosen" labels — what <code>val_pref_acc</code> is measured
  against — come from <strong>GPT-4 acting as a judge</strong>. So the metric
  measures one thing precisely: <em>how often does our policy agree with GPT-4
  about which response is better?</em></p>
  <p>GPT-4-as-judge has well-documented systematic biases. It prefers longer
  responses, more structured ones (bullets, headers), and ones that <em>look</em>
  thorough and helpful (preambles, comprehensive lists, hedging). Within the
  sweet spot, optimizing toward GPT-4 is useful — humans agree GPT-4-style
  structure is often clearer. Past the sweet spot, the model becomes <em>too</em>
  GPT-4-like: too long, too bulleted, too hedged. That's where humans tap out.</p>
  <p>{named('prod-bestbet-v2')} has more capacity to mirror UltraFeedback's
  chosen style — the larger LoRA adapter is what tips it past the sweet spot.
  Same recipe, more parameters, higher metric score, worse model by humans.</p>

  <h3>Where the metric and humans diverge most</h3>
  <p>How often humans preferred the lower-val_pref_acc model, by prompt
  category. Higher rate = the metric is less reliable in this category (humans
  more often went the other way).</p>
  <table style="table-layout: fixed; max-width: 860px;">
    <colgroup>
      <col style="width: 26%;">
      <col style="width: 28%;">
      <col style="width: 46%;">
    </colgroup>
    <thead><tr>
      <th>Category</th>
      <th class='num'>Lower-val wins / total</th>
      <th>Disagreement rate</th>
    </tr></thead>
    <tbody>{''.join(cat_rows)}</tbody>
  </table>
  <p>The metric works reasonably on <strong>instructional / advice</strong>
  prompts — the closest match to UltraFeedback's training distribution. It
  fails hardest on <strong>emotional tone</strong> and <strong>pedagogical
  clarity</strong>, where humans want warm, concise, naturally-flowing
  language and the metric pushes toward structured-looking thoroughness.</p>

  {len_para}
</section>""")

    # ── 8. What humans saw that the metric didn't ──
    themes_html = ""
    if report_data.get("llm_themes") and report_data["llm_themes"].get("themes"):
        themes = report_data["llm_themes"]["themes"]
        total = report_data["llm_themes"].get("n_comments", sum(t["count"] for t in themes))
        method_note = (f"Comments clustered thematically by an LLM "
                       f"({h(report_data['llm_themes'].get('model', 'Claude'))}). "
                       f"Each theme groups comments by the quality dimension annotators "
                       f"reacted to, not by surface keywords.")
        theme_rows = []
        for t in themes[:10]:
            quotes_html = "".join(
                f'<div class="theme-sample">&ldquo;{h(truncate(q, 220))}&rdquo;</div>'
                for q in t.get("quotes", [])
            )
            desc_html = (f'<div style="color: var(--muted); font-size: 13px; margin: 4px 0 8px;">'
                         f'{h(t.get("description", ""))}</div>') if t.get("description") else ""
            theme_rows.append(f"""
<details>
  <summary>
    <strong>{h(t['label'])}</strong>
    <span style="color: var(--muted); font-size: 13px;">· {t['count']} comments</span>
  </summary>
  <div style="margin-top: 8px;">{desc_html}{quotes_html}</div>
</details>""")
        themes_html = f"""
<h3>What annotators told us, in their own words</h3>
<p>{total} non-empty comments. {method_note}</p>
{''.join(theme_rows)}"""
    elif report_data["comment_themes"]:
        theme_rows = []
        total_themed = sum(t["count"] for t in report_data["comment_themes"])
        for t in report_data["comment_themes"][:8]:
            samples_html = "".join(
                f'<div class="theme-sample">&ldquo;{h(truncate(s, 180))}&rdquo;</div>'
                for s in t["samples"]
            )
            theme_rows.append(f"""
<details>
  <summary>
    <strong>{h(t['theme'])}</strong>
    <span style="color: var(--muted); font-size: 13px;">· {t['count']} comments</span>
  </summary>
  <div style="margin-top: 8px;">{samples_html}</div>
</details>""")
        themes_html = f"""
<h3>What annotators told us, in their own words</h3>
<p>{total_themed} comments matched one or more themes (keyword-bucketed — to
get richer LLM thematic clustering, run <code>cluster_comments.py</code>).</p>
{''.join(theme_rows)}"""

    disagree_html_parts = []
    if report_data["disagreement_pairs"]:
        for mx, my, dp in report_data["disagreement_pairs"][:5]:
            scenario = dp["scenario"]
            mx_text = scenario["a_text"] if scenario["a_model"] == mx else scenario["b_text"]
            my_text = scenario["a_text"] if scenario["a_model"] == my else scenario["b_text"]
            mx_val_str = f" (val={VAL_PREF_ACC[mx]:.3f})" if VAL_PREF_ACC.get(mx) is not None else " (untrained)"
            my_val_str = f" (val={VAL_PREF_ACC[my]:.3f})" if VAL_PREF_ACC.get(my) is not None else " (untrained)"
            disagree_html_parts.append(f"""
<details class="disagree-pair">
  <summary>
    {named_short(mx)} beat {named_short(my)}
    ({dp['mx_wins']}–{dp['ties']}–{dp['my_wins']})
    · prompt {h(scenario.get('prompt_id', ''))}
  </summary>
  <div style="margin-top: 10px;">
    <div class="prompt-text">{h(truncate(scenario['prompt'], 200))}</div>
    <div class="responses">
      <div class="response-card winner">
        <div class="role">Picked by humans · {h(MODEL_DISPLAY.get(mx, mx))}{h(mx_val_str)}</div>
        <div class="text">{h(truncate(mx_text, 1100))}</div>
      </div>
      <div class="response-card loser">
        <div class="role">Higher val_pref_acc · {h(MODEL_DISPLAY.get(my, my))}{h(my_val_str)}</div>
        <div class="text">{h(truncate(my_text, 1100))}</div>
      </div>
    </div>
    <div class="vote-summary">
      Annotator votes — {h(MODEL_DISPLAY.get(mx, mx))}: {dp['mx_wins']} · ties: {dp['ties']} · {h(MODEL_DISPLAY.get(my, my))}: {dp['my_wins']}
    </div>
  </div>
</details>""")

    sections.append(f"""
<section class="section">
  <h2>What humans saw that the metric didn't</h2>
  <p>The metric outputs a single number. Humans wrote comments and showed
  taste at the per-scenario level. Examples below: scenarios where humans
  preferred the <em>lower</em>-val_pref_acc response — click to expand prompt
  + both responses.</p>
  {''.join(disagree_html_parts)}
  {themes_html}
</section>""")

    # ── 9. What this means ──
    sections.append(f"""
<section class="section">
  <h2>What this means</h2>
  <p>The picture isn't "the metric is broken." <code>val_pref_acc</code>
  points in the right direction (Spearman ρ = {spearman:+.2f}) — it's a fine
  signal to <em>search</em> with. The failure mode is specific: near the top
  it stops being able to distinguish recipes that are statistically separable
  by humans, because at that scale it's measuring "GPT-4-likeness" more than
  "humans like this."</p>

  <h3>What the agent searches is shaped by how you prompt it</h3>
  <p>The agent's 50-experiment run stayed within standard full-fine-tune DPO
  and never proposed LoRA or data filtering on its own. The unlock wasn't
  more compute or a different agent — it was a researcher asking the agent
  a different question: "look at what you've tried and find what's missing."
  The agent's response to <em>that</em> prompt was LoRA + data filtering. The
  search space an agent explores is shaped as much by how a researcher frames
  it as by the agent's loop itself; periodic meta-prompts — "what aren't you
  trying?" — are how you widen it.</p>

  <h3>Automated metrics are coarse selectors at the top</h3>
  <p>{named('prod-bestbet')} and {named('prod-bestbet-v2')} are statistically
  separable by humans but inverted by the metric. {n_annotations} Prolific
  pairwise ratings — a few hundred dollars of annotation — flipped the
  production decision. That's the standard preference-eval setup: humans
  don't have to agree individually for the aggregate to be informative, and
  the aggregate is what you ship from.</p>

  <h3>Three places HITL adds value</h3>
  <p>Not just one stage of the loop:</p>
  <ul>
    <li><strong>Research-time HITL</strong> — researcher prompting the agent
        to inspect its own trajectory and identify what it isn't trying.</li>
    <li><strong>Evaluation-time HITL</strong> — Prolific (or any structured
        human eval) telling you which of the metric's top finalists actually
        wins with users.</li>
    <li><strong>Interpretation-time HITL</strong> — researcher reading
        comments to figure out <em>why</em> humans preferred what they did
        ("too verbose," "too structured," "felt natural") — feeds back into
        the next iteration of the agent's design space.</li>
  </ul>

  <p>The combination is what produced the actual best recipe: agent for fast
  search within a defined space, researcher to widen that space, Prolific to
  pick the winner at the top, researcher again to understand what won.</p>
</section>""")

    # ── 10. Appendix ──
    # Per-pair table
    pair_rows = []
    for row in pair_table:
        mx, my = row["mx"], row["my"]
        mx_w, my_w, t = row["mx_wins"], row["my_wins"], row["ties"]
        v_mx, v_my = VAL_PREF_ACC.get(mx) or 0.0, VAL_PREF_ACC.get(my) or 0.0
        higher_model = my if v_my > v_mx else mx
        nt = mx_w + my_w
        if nt == 0:
            higher_rate, ci_str, flag = "—", "—", "<span class='pill tie'>n/a</span>"
        else:
            higher_wins = my_w if higher_model == my else mx_w
            rate = higher_wins / nt
            lo, hi = wilson_ci(higher_wins, nt)
            higher_rate = f"{rate*100:.1f}%"
            ci_str = f"[{lo*100:.1f}, {hi*100:.1f}]"
            if lo > 0.5:
                flag = "<span class='pill good'>tracks</span>"
            elif hi < 0.5:
                flag = "<span class='pill bad'>flips</span>"
            else:
                flag = "<span class='pill tie'>noisy</span>"
        gap_str = f"{v_my-v_mx:+.3f}" if v_my != v_mx else "—"
        pair_rows.append(f"""
<tr>
  <td>{named_short(mx)} <small style="color:var(--muted)">vs</small> {named_short(my)}</td>
  <td class='num'>{gap_str}</td>
  <td class='num'>{mx_w}</td>
  <td class='num'>{t}</td>
  <td class='num'>{my_w}</td>
  <td class='num' style='white-space: nowrap;'>{higher_rate} <small style="color:var(--muted)">{ci_str}</small></td>
  <td>{flag}</td>
</tr>""")

    # BT ranking
    bt_rows = []
    for i, (model, strength) in enumerate(bt):
        v = VAL_PREF_ACC.get(model)
        v_str = f"{v:.3f}" if v is not None else "—"
        bt_rows.append(
            f"<tr><td>{i+1}</td><td style='white-space: nowrap;'>{named_short(model)}</td>"
            f"<td class='num'>{strength:.3f}</td><td class='num'>{v_str}</td></tr>"
        )

    # IAA
    iaa_rows = []
    for (mx, my), iaa_v in sorted(report_data["iaa"].items()):
        k = iaa_v["kappa"]
        if k is None:
            continue
        if k < 0.2:
            tag, cls = "poor", "bad"
        elif k < 0.4:
            tag, cls = "fair", "tie"
        elif k < 0.6:
            tag, cls = "moderate", "good"
        else:
            tag, cls = "substantial", "good"
        iaa_rows.append(f"""
<tr>
  <td style='white-space: nowrap;'>{named_short(mx)} vs {named_short(my)}</td>
  <td class='num'><span class='kappa'>κ = {k:+.3f}</span></td>
  <td><span class='pill {cls}'>{tag}</span></td>
  <td class='num'>{iaa_v['n_items']}</td>
</tr>""")

    # Per-category breakdown
    cat_html = []
    for cat, pair_data in sorted(report_data["per_category"].items()):
        rows = []
        for (mx, my), stats in sorted(pair_data.items()):
            v_mx = VAL_PREF_ACC.get(mx) or 0.0
            v_my = VAL_PREF_ACC.get(my) or 0.0
            higher = my if v_my > v_mx else mx
            nt = stats["mx_wins"] + stats["my_wins"]
            if nt == 0:
                continue
            higher_wins = stats["my_wins"] if higher == my else stats["mx_wins"]
            rate = higher_wins / nt * 100
            rows.append(
                f"<tr><td style='white-space: nowrap;'>{named_short(mx)} vs {named_short(my)}</td>"
                f"<td class='num'>{stats['mx_wins']}/{stats['ties']}/{stats['my_wins']}</td>"
                f"<td class='num'>{rate:.0f}%</td></tr>"
            )
        if rows:
            cat_html.append(f"""
<details>
  <summary><strong>{h(cat.replace('_', ' '))}</strong></summary>
  <table style="margin-top: 8px;">
    <thead><tr><th>Pair</th><th class='num'>X/tie/Y</th><th class='num'>Higher-val win rate</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</details>""")

    sections.append(f"""
<section class="section">
  <h2>Appendix</h2>
  <p>The detail behind the summary — per-pair stats, ranking, agreement,
  category breakdown, methodology.</p>

  <details open>
    <summary><strong>Per-pair preference results</strong></summary>
    <p style="margin-top: 8px;">All 10 unique pairs, 50 prompts × 3 annotators
    = 150 ratings each. "tracks" = higher-val recipe won at &gt; 50% with
    Wilson 95% CI excluding chance. "flips" = lower-val recipe won. "noisy"
    = CI overlaps 50%.</p>
    <table style="table-layout: fixed;">
      <colgroup>
        <col style="width: 30%;">
        <col style="width: 11%;">
        <col style="width: 7%;">
        <col style="width: 6%;">
        <col style="width: 7%;">
        <col style="width: 27%;">
        <col style="width: 12%;">
      </colgroup>
      <thead>
        <tr>
          <th>Pair (X vs Y)</th>
          <th class='num'>Δ val_pref_acc</th>
          <th class='num'>X wins</th>
          <th class='num'>Ties</th>
          <th class='num'>Y wins</th>
          <th class='num'>Higher-val win rate (95% CI)</th>
          <th>Signal</th>
        </tr>
      </thead>
      <tbody>{''.join(pair_rows)}</tbody>
    </table>
  </details>

  <details>
    <summary><strong>Bradley-Terry global ranking</strong></summary>
    <p style="margin-top: 8px;">Combines all pairwise outcomes into a single
    ranking. Higher BT strength = humans preferred this recipe more often
    across the whole study.</p>
    <table style="table-layout: fixed; max-width: 720px;">
      <colgroup>
        <col style="width: 12%;">
        <col style="width: 46%;">
        <col style="width: 21%;">
        <col style="width: 21%;">
      </colgroup>
      <thead><tr><th>Rank</th><th>Recipe</th><th class='num'>BT strength</th><th class='num'>val_pref_acc</th></tr></thead>
      <tbody>{''.join(bt_rows)}</tbody>
    </table>
    <div id="bt-chart" style="height:340px; margin-top: 12px;"></div>
  </details>

  <details>
    <summary><strong>Inter-annotator agreement (Fleiss' κ)</strong></summary>
    <p style="margin-top: 8px;">Annotators within a single pair frequently
    disagreed — taste is heterogeneous. Low κ tells you taste varies at the
    per-scenario level; it doesn't invalidate the aggregate signal, which is
    averaged over hundreds of ratings.</p>
    <table style="table-layout: fixed;">
      <colgroup>
        <col style="width: 52%;">
        <col style="width: 16%;">
        <col style="width: 18%;">
        <col style="width: 14%;">
      </colgroup>
      <thead><tr><th>Pair</th><th class='num'>Fleiss' κ</th><th>Agreement</th><th class='num'>n items</th></tr></thead>
      <tbody>{''.join(iaa_rows)}</tbody>
    </table>
  </details>

  <details>
    <summary><strong>Per prompt category</strong></summary>
    <p style="margin-top: 8px;">How the preference pattern shifts by prompt
    type. Higher-val win rate &lt; 50% = humans systematically picked the
    lower-scoring recipe in that category.</p>
    {''.join(cat_html)}
  </details>

  <details>
    <summary><strong>Why we used val_pref_acc</strong></summary>
    <p style="margin-top: 8px;">Karpathy's autoresearch uses
    <code>val_bpb</code> — held-out cross-entropy in bits per byte — because
    it's pretraining. Our task is DPO fine-tuning, so the natural analog is
    <code>val_pref_acc</code>: on a held-out preference set, what fraction
    does the policy "agree" with — i.e., assign higher implicit reward to the
    chosen response than the rejected one. Both metrics measure
    fit-to-held-out-data. The question for both is the same: does fit to the
    validation set track what people actually want?</p>
    <table>
      <thead><tr><th style="width:20%;"></th><th>Karpathy's autoresearch</th><th>This study</th></tr></thead>
      <tbody>
        <tr><td>Task</td><td>Pretraining (LM)</td><td>DPO fine-tuning</td></tr>
        <tr><td>Held-out data</td><td>Validation web text</td><td>UltraFeedback test_prefs</td></tr>
        <tr><td>Metric</td><td><code>val_bpb</code> ↓</td><td><code>val_pref_acc</code> ↑</td></tr>
        <tr><td>Interpretation</td><td>How well does it predict text?</td><td>How often does it agree with the preference labels?</td></tr>
      </tbody>
    </table>
  </details>

  <details>
    <summary><strong>Methodology</strong></summary>
    <div class="method" style="margin-top: 8px;">
      <ul>
        <li><strong>Base model:</strong> SmolLM2-360M-Instruct (HuggingFaceTB).
          4 DPO-trained variants + the untrained base = 5 models.</li>
        <li><strong>Agent stage:</strong> autoresearch (Claude editing
          <code>train.py</code>) ran 50 experiments — ~9.5 min of agent
          reasoning per call plus DPO training between calls; 45 calls
          produced a valid response, 5 hit empty-response failures. Total
          Claude API cost: $19.25. (Wall-clock spanned May 8–9 with the run
          paused for ~10h between experiments 23 and 24; usable runtime is
          ~8h.) We kept 2 checkpoints: {MODEL_SHORT['prod-baseline']}
          (default DPO, val_pref_acc = {VAL_PREF_ACC['prod-baseline']:.3f})
          and {MODEL_SHORT['prod-adambeta2']} (the agent's best, val_pref_acc
          = {VAL_PREF_ACC['prod-adambeta2']:.3f}). Both stayed below the
          untrained reference's 0.500.</li>
        <li><strong>Researcher + agent stage:</strong> researcher prompted
          the agent to inspect its own trajectory and look for gaps; the
          agent proposed two design-space expansions — LoRA adapters and
          score-margin-filtered UltraFeedback (margin ≥ 2). Same training
          loop. Produced {MODEL_SHORT['prod-bestbet']} (LoRA r=32) and
          {MODEL_SHORT['prod-bestbet-v2']} (LoRA r=64, α=128).</li>
        <li><strong>Prompt set:</strong> 50 general-audience prompts across 8
          categories (creative writing, personal advice, pedagogy, sensory
          descriptive, persuasive, emotional tone, light planning, common
          factual). See <code>prolific_prompts.json</code>.</li>
        <li><strong>Generation:</strong> per-model sampling seeds; EOS-stopping
          (no truncation). <code>temperature=0.7, top_p=0.9,
          repetition_penalty=1.05</code>.</li>
        <li><strong>Pairing:</strong> C(5,2) = 10 unique pairs × 50 prompts =
          500 pair instances × 3 annotators = 1500 annotations.</li>
        <li><strong>UI:</strong> Prolific participants saw the prompt + two
          responses (A and B), made a forced 3-way choice (A better / About
          equal / B better) with optional comment. A/B display order
          randomized per participant via stable hash.</li>
        <li><strong>Statistics:</strong> Wilson 95% CIs on per-pair preference
          rates; Bradley-Terry via Hunter MM for global ranking; Fleiss' κ per
          pair for IAA; Spearman ρ over trained recipes only.</li>
        <li><strong>Source:</strong> annotations loaded from
          <code>{h(str(annotations_path))}</code>.</li>
      </ul>
    </div>
  </details>
</section>

""")

    # Headline chart: val_pref_acc vs vs-base preference (trained models only)
    chart_x_json = json.dumps(chart_x)
    chart_y_json = json.dumps(chart_y)
    chart_text_json = json.dumps(chart_text)
    chart_colors_json = json.dumps(chart_colors)

    # Appendix chart: val_pref_acc vs BT strength (trained models only)
    bt_chart_x = [VAL_PREF_ACC[m] for m in chart_models]
    bt_chart_y = [bt_dict.get(m, 0.0) for m in chart_models]
    bt_chart_x_json = json.dumps(bt_chart_x)
    bt_chart_y_json = json.dumps(bt_chart_y)
    bt_chart_text_json = json.dumps(chart_text)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Autoresearch × Prolific — Phase 2 results</title>
  <script src="{PLOTLY_CDN}"></script>
  <style>{CSS}</style>
</head>
<body>
<div class="wrap">
{''.join(sections)}
</div>
<script>
(function() {{
  var data = [{{
    x: {chart_x_json},
    y: {chart_y_json},
    text: {chart_text_json},
    mode: 'markers+text',
    type: 'scatter',
    textposition: 'top center',
    marker: {{ size: 14, color: {chart_colors_json}, line: {{ color: '#1a1a1a', width: 1 }} }},
    hovertemplate: '%{{text}}<br>val_pref_acc=%{{x:.3f}}<br>vs-base win rate=%{{y:.2f}}<extra></extra>',
  }}];
  var layout = {{
    margin: {{t: 20, l: 60, r: 20, b: 50}},
    xaxis: {{ title: 'val_pref_acc (autoresearch metric)', range: [0.44, 0.68], zeroline: false }},
    yaxis: {{ title: 'Human preference vs base (excl. ties)', range: [0, 1], zeroline: false,
              tickformat: '.0%' }},
    shapes: [
      {{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0.5, y1: 0.5,
         line: {{ color: '#888', width: 1, dash: 'dot' }} }},
      {{ type: 'line', xref: 'x', yref: 'paper', x0: 0.5, x1: 0.5, y0: 0, y1: 1,
         line: {{ color: '#888', width: 1, dash: 'dot' }} }}
    ],
    annotations: [
      {{ xref: 'paper', yref: 'y', x: 0.99, y: 0.5, text: 'humans: chance vs base',
         showarrow: false, font: {{ size: 11, color: '#888' }},
         xanchor: 'right', yanchor: 'bottom' }},
      {{ xref: 'x', yref: 'paper', x: 0.5, y: 0.02, text: 'metric: chance (untrained)',
         showarrow: false, font: {{ size: 11, color: '#888' }},
         xanchor: 'left', yanchor: 'bottom', textangle: -90 }}
    ],
    plot_bgcolor: '#fff', paper_bgcolor: '#fff',
    font: {{ family: '-apple-system, system-ui, sans-serif', size: 13 }},
  }};
  Plotly.newPlot('headline-chart', data, layout, {{displayModeBar: false, responsive: true}});

  var btData = [{{
    x: {bt_chart_x_json},
    y: {bt_chart_y_json},
    text: {bt_chart_text_json},
    mode: 'markers+text',
    type: 'scatter',
    textposition: 'top center',
    marker: {{ size: 14, color: {chart_colors_json}, line: {{ color: '#1a1a1a', width: 1 }} }},
    hovertemplate: '%{{text}}<br>val_pref_acc=%{{x:.3f}}<br>BT strength=%{{y:.3f}}<extra></extra>',
  }}];
  var btLayout = {{
    margin: {{t: 20, l: 60, r: 20, b: 50}},
    xaxis: {{ title: 'val_pref_acc (autoresearch metric)', range: [0.44, 0.68], zeroline: false }},
    yaxis: {{ title: 'Bradley-Terry strength (human pref)', zeroline: false }},
    plot_bgcolor: '#fff', paper_bgcolor: '#fff',
    font: {{ family: '-apple-system, system-ui, sans-serif', size: 13 }},
  }};
  Plotly.newPlot('bt-chart', btData, btLayout, {{displayModeBar: false, responsive: true}});
}})();
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    out_path = Path(args.out)
    mock_mode = "mock" in ann_path.name.lower()

    annotations = json.load(open(ann_path))
    grid = json.load(open(GRID_PATH))
    prompts_doc = json.load(open(PROMPTS_PATH))
    prompt_categories = categorize_by_prompt(prompts_doc)

    # Aggregate
    pair_stats = aggregate(annotations, grid)
    iaa = compute_iaa_per_pair(pair_stats)
    disagreements = find_disagreement_pairs(pair_stats, grid, None)
    per_cat = per_category_breakdown(pair_stats, prompt_categories)
    comment_themes, per_pair_themes = extract_comment_themes(annotations, pair_stats)
    failure_analysis = characterize_failures(annotations, grid, prompt_categories)

    # Prefer LLM-clustered comment themes if cluster_comments.py was run.
    llm_themes_path = HERE / "report" / "comment_themes_llm.json"
    llm_themes = None
    if llm_themes_path.exists():
        try:
            llm_themes = json.load(open(llm_themes_path))
            print(f"  Using LLM-clustered themes from {llm_themes_path.name}")
        except Exception as e:
            print(f"  Warning: failed to load {llm_themes_path}: {e}")

    # Per-pair table
    per_pair_table = []
    for (mx, my), ps in sorted(pair_stats.items()):
        nt = ps["mx_wins"] + ps["my_wins"]
        per_pair_table.append({
            "mx": mx, "my": my, "n": ps["n"],
            "mx_wins": ps["mx_wins"], "my_wins": ps["my_wins"], "ties": ps["ties"],
            "mx_pref_excl_ties": ps["mx_wins"] / nt if nt else None,
            "my_pref_excl_ties": ps["my_wins"] / nt if nt else None,
        })

    # Bradley-Terry
    models = [m for m in MODEL_ORDER if any(
        (mx == m or my == m) for (mx, my) in pair_stats.keys()
    )]
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    wins_matrix = [[0.0] * n for _ in range(n)]
    for (mx, my), ps in pair_stats.items():
        if mx in idx and my in idx:
            i, j = idx[mx], idx[my]
            wins_matrix[i][j] += ps["mx_wins"] + 0.5 * ps["ties"]
            wins_matrix[j][i] += ps["my_wins"] + 0.5 * ps["ties"]
    bt_dict = bradley_terry(wins_matrix, models)
    bt = sorted(bt_dict.items(), key=lambda kv: -kv[1])

    # Spearman ρ between val_pref_acc and BT strength.
    # Both rankings ASCENDING — rank 0 = lowest val / lowest BT. Trained models
    # only (untrained base has no meaningful val_pref_acc; it's chance by
    # construction).
    trained = [(m, s) for m, s in bt_dict.items() if m in TRAINED_MODELS]
    if len(trained) >= 3:
        models_by_val = sorted(trained, key=lambda x: VAL_PREF_ACC[x[0]])
        models_by_bt = sorted(trained, key=lambda x: x[1])
        rank_val_of = {m: r for r, (m, _) in enumerate(models_by_val)}
        rank_bt_of = {m: r for r, (m, _) in enumerate(models_by_bt)}
        d_sq = sum((rank_val_of[m] - rank_bt_of[m]) ** 2 for m, _ in trained)
        n_t = len(trained)
        spearman = 1 - 6 * d_sq / (n_t * (n_t ** 2 - 1)) if n_t > 1 else 0.0
    else:
        spearman = 0.0

    # Significance counts (per-pair: does val_pref_acc track human preference?)
    sig_tracks = sum(1 for r in per_pair_table
                     if r["mx_pref_excl_ties"] is not None and
                     ((VAL_PREF_ACC.get(r["my"]) or 0) > (VAL_PREF_ACC.get(r["mx"]) or 0) and
                      wilson_ci(r["my_wins"], r["mx_wins"] + r["my_wins"])[0] > 0.5))
    sig_flips = sum(1 for r in per_pair_table
                    if r["mx_pref_excl_ties"] is not None and
                    ((VAL_PREF_ACC.get(r["my"]) or 0) > (VAL_PREF_ACC.get(r["mx"]) or 0) and
                     wilson_ci(r["my_wins"], r["mx_wins"] + r["my_wins"])[1] < 0.5))

    report_data = {
        "n_annotations": len(annotations),
        "n_annotators": len({a.get("prolific_pid") for a in annotations}),
        "per_pair": per_pair_table,
        "bt": bt,
        "bt_dict": bt_dict,
        "spearman": spearman,
        "sig_tracks": sig_tracks,
        "sig_flips": sig_flips,
        "disagreement_pairs": disagreements,
        "per_category": per_cat,
        "iaa": iaa,
        "comment_themes": comment_themes,
        "per_pair_themes": dict(per_pair_themes),
        "failure_analysis": failure_analysis,
        "llm_themes": llm_themes,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(report_data, ann_path, mock_mode))
    print(f"Report written to {out_path}")
    print(f"  Annotations: {len(annotations)}")
    print(f"  Spearman ρ (val_pref_acc vs BT ranking): {spearman:.3f}")
    print(f"  Pairs where val_pref_acc tracks preference: {sig_tracks}/{len(per_pair_table)}")
    print(f"  Pairs where humans flip val_pref_acc: {sig_flips}/{len(per_pair_table)}")
    print(f"  Disagreement examples surfaced: {len(disagreements)}")


if __name__ == "__main__":
    main()
