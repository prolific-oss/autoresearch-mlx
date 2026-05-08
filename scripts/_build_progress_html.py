"""
Build a standalone HTML progress doc for Viviana.

Embeds PNGs as base64 data URIs so the output is a single shareable file.

Usage:
    uv run scripts/_build_progress_html.py
"""

import base64
import json
import os
from datetime import datetime

REPORT_DIR = "report"
DATASET = "dataset.jsonl"
AUTO_METRICS = "auto_metrics.json"
OUT = os.path.join(REPORT_DIR, "progress_for_viviana.html")

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
  --warn: #b85c00;
}
body {
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  color: var(--fg);
  background: var(--bg);
  max-width: 880px;
  margin: 40px auto 80px;
  padding: 0 24px;
}
h1 { font-size: 28px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 21px; margin: 36px 0 10px; padding-top: 6px; border-top: 1px solid var(--border); letter-spacing: -0.01em; }
h3 { font-size: 17px; margin: 22px 0 6px; color: var(--fg); }
p, li { margin: 8px 0; }
small.byline { color: var(--muted); }
.meta { color: var(--muted); font-size: 13px; }
.callout {
  background: var(--accent-soft);
  border-left: 3px solid var(--accent);
  padding: 12px 16px;
  margin: 16px 0;
  font-size: 15px;
}
.muted { color: var(--muted); }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }
th { background: var(--panel); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
img { max-width: 100%; height: auto; display: block; margin: 14px auto; border: 1px solid var(--border); border-radius: 4px; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
code { font-size: 13px; background: var(--panel); padding: 1px 5px; border-radius: 3px; }
pre { background: var(--panel); padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 13px; }
.gen { background: var(--panel); padding: 10px 14px; border-radius: 4px; font-size: 14px; line-height: 1.5; margin: 6px 0 14px; border-left: 2px solid var(--border); }
.gen-tag { display: inline-block; font-weight: 600; margin-right: 8px; }
.gen-bpb { color: var(--muted); font-size: 13px; }
ul.timeline { list-style: none; padding-left: 0; }
ul.timeline li { padding: 4px 0 4px 18px; position: relative; }
ul.timeline li::before { content: "→"; color: var(--accent); position: absolute; left: 0; }
"""


def b64_img(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def img_tag(path, alt=""):
    if not os.path.isfile(path):
        return f'<p class="muted"><em>(missing: {os.path.basename(path)})</em></p>'
    return f'<img src="{b64_img(path)}" alt="{alt}">'


def load_metrics():
    if not os.path.isfile(AUTO_METRICS):
        return None
    with open(AUTO_METRICS) as f:
        return json.load(f)


def load_dataset():
    if not os.path.isfile(DATASET):
        return []
    with open(DATASET) as f:
        return [json.loads(l) for l in f if l.strip()]


def metrics_table(per_ckpt):
    rows = ["<table>",
            "<thead><tr><th>checkpoint</th>"
            "<th class='num'>val_bpb</th>"
            "<th class='num'>loop</th>"
            "<th class='num'>unique-word</th>"
            "<th class='num'>digit-density</th>"
            "<th class='num'>avg-sent-len</th>"
            "<th class='num'>ref ppl (SmolLM-135M)</th>"
            "</tr></thead>",
            "<tbody>"]
    for c in per_ckpt:
        rows.append("<tr>")
        rows.append(f"<td><code>{c['checkpoint_id']}</code></td>")
        rows.append(f"<td class='num'>{c['val_bpb']:.4f}</td>")
        rows.append(f"<td class='num'>{c.get('loop_4gram_frac_mean', 0):.3f}</td>")
        rows.append(f"<td class='num'>{c.get('unique_word_frac_mean', 0):.3f}</td>")
        rows.append(f"<td class='num'>{c.get('digit_density_mean', 0):.3f}</td>")
        rows.append(f"<td class='num'>{c.get('avg_sentence_length_mean', 0):.1f}</td>")
        ppl = c.get('ref_perplexity_mean')
        rows.append(f"<td class='num'>{ppl:.1f}</td>" if ppl is not None else "<td class='num'>—</td>")
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def samples_section(dataset):
    if not dataset:
        return ""
    pid = "biomedical_00"
    rows = [r for r in dataset if r["prompt_id"] == pid]
    if not rows:
        return ""
    rows.sort(key=lambda r: r["val_bpb"])
    parts = ["<h3>Sample completions on the same prompt</h3>"]
    parts.append(f"<p class='muted'>Prompt (eLife biomedical): <em>“{rows[0]['prompt_text']}”</em></p>")
    parts.append("<p class='muted'>Sampling: temp=0.8, top-p=0.9, freq-penalty=0.5. Trailing <code> .</code> stripped from prompt.</p>")
    for r in rows:
        snippet = r["generated_text"][:380].replace("\n", " ").strip()
        parts.append(
            f"<div class='gen'><span class='gen-tag'><code>{r['checkpoint_id']}</code></span>"
            f"<span class='gen-bpb'>val_bpb {r['val_bpb']:.4f}</span><br>"
            f"…{snippet}…</div>"
        )
    return "\n".join(parts)


def main():
    metrics = load_metrics()
    dataset = load_dataset()
    n_ckpts = len(metrics["per_checkpoint"]) if metrics else 0
    n_dataset = len(dataset)

    long_in_progress = "84a8401-long, 0b820f1-long, 098bc40-long"

    today = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Autoresearch × Prolific — progress (interim)</title>",
        f"<style>{CSS}</style></head><body>",

        f"<h1>Autoresearch × Prolific — interim progress</h1>",
        f"<p class='meta'>Generated {today} · Nora &amp; Claude tag-team</p>",

        f"<div class='callout'>"
        f"Hey Viviana! Picking up from your <code>autoresearch/apr13-9</code> branch and the "
        f"reply doc from 2026-04-16. The pipeline you wrote works end-to-end; the work below "
        f"is methodology fixes that surfaced once we started looking at outputs more carefully, "
        f"plus a longer-training arm of the experiment. <strong>Nothing is concluded — the "
        f"Prolific human eval is still the actual experiment.</strong></div>",

        f"<h2>What we found and fixed (and why)</h2>",
        "<ul class='timeline'>",
        "<li><strong>Sampling artifact (top-k=50 mode-trapping).</strong> Long-trained checkpoints were "
        "collapsing into letter/number-spam (<code>SYSYSYSY…</code>, <code>1222222…</code>) at the "
        "default top-k=50. This looked like a dramatic val_bpb-vs-quality divergence at first, but "
        "switching to <code>top-p=0.9 + freq_penalty=0.5</code> made the spam disappear. Most of "
        "the apparent divergence was a sampling artifact — Holtzman 2019 territory.</li>",
        "<li><strong>Prompt-shape artifact.</strong> eLife / PLOS prompts all end in <code> .</code> "
        "(space + period). Long-trained models had memorised <em>“after</em> <code> .</code> "
        "<em>predict</em> <code>.</code><em>”</em> and produced pure-period output. Fix: "
        "<code>STRIP_TRAILING_SPACE_PERIOD = True</code> in <code>generate_dataset.py</code>.</li>",
        "<li><strong>Tokenizer / training corpus too narrow.</strong> The original setup used "
        "10 shards out of ClimbMix's 6,543 (~0.15% of available data). 30-min training on that "
        "encouraged memorisation of prefix patterns. Fix: full reset — "
        "<code>prepare.py --num-shards=50</code>, retrain BPE tokenizer, retrain all 8 checkpoints "
        "from scratch.</li>",
        "<li><strong>Resumable training.</strong> Added periodic snapshot saves to "
        "<code>train.py</code> (every 5 min of training) plus a <code>scripts/_resume_chain.sh</code> "
        "that auto-skips done tags / auto-resumes from snapshot. Means we can stop and resume the "
        "long runs cleanly.</li>",
        "</ul>",

        f"<h2>Current state</h2>",
        f"<ul>"
        f"<li>5 short-budget (5-min) checkpoints retrained on the new 50-shard tokenizer — "
        f"all 5 saved in <code>checkpoints/</code></li>"
        f"<li>3 long-budget (30-min) checkpoints in progress: <code>{long_in_progress}</code></li>"
        f"<li><code>dataset.jsonl</code> generated on the 5 short ckpts ({n_dataset} records, "
        f"top-p sampling) — <em>partial</em>, will be regenerated once the long ckpts finish</li>"
        f"<li>Auto-metrics computed on the partial dataset ({n_ckpts} ckpts × 8 metrics + "
        f"reference-LM perplexity via SmolLM-135M)</li>"
        f"</ul>",

        f"<h2>Training dynamics (5 short + 1 in-progress)</h2>",
        img_tag(os.path.join(REPORT_DIR, "training_loss_curves.png"), "loss curves"),
        f"<p>The familiar autoresearch step-efficiency story still holds at the broader data scale: "
        f"<code>098bc40</code> (smallest batch → ~7× more optimiser steps) drives training loss lowest "
        f"in the same wallclock. <code>84a8401-long</code> (yellow, in progress) is currently tracking "
        f"the same trajectory as the short runs — it'll continue past where they stopped.</p>",

        img_tag(os.path.join(REPORT_DIR, "val_bpb_summary.png"), "val_bpb summary"),

        f"<h2>Auto-metrics (interim, 5 short ckpts)</h2>",
    ]

    if metrics:
        parts.append(metrics_table(metrics["per_checkpoint"]))
        parts.append(
            "<p>Reference perplexity computed under <code>mlx-community/SmolLM-135M-fp16</code> — "
            "a model ~10× bigger than ours, used purely as an external fluency proxy. "
            "Lower ppl = SmolLM finds the text more natural. "
            "Loop / digit-density / punctuation-density are basically dead across all 5 ckpts "
            "(top-p + freq-penalty handled the catastrophic failure modes).</p>"
        )

        parts.append("<h3>val_bpb vs reference perplexity</h3>")
        parts.append(img_tag(os.path.join(REPORT_DIR, "val_bpb_vs_ref_perplexity.png"), "val_bpb vs ref ppl"))
        parts.append(
            "<p>Roughly positive trend (lower val_bpb → lower ref ppl) with a clear inversion: "
            "<code>4926214</code> (val_bpb 1.71, ref ppl 166) is judged more natural by SmolLM "
            "than <code>0b820f1</code> (val_bpb 1.61, ref ppl 189) despite higher val_bpb.</p>"
        )

        parts.append("<h3>val_bpb vs vocabulary diversity (unique-word fraction)</h3>")
        parts.append(img_tag(os.path.join(REPORT_DIR, "val_bpb_vs_unique_word_frac.png"), "val_bpb vs diversity"))
        parts.append(
            "<p>The opposite direction: lower val_bpb → less diverse vocabulary. "
            "<code>098bc40</code> (best val_bpb) has the lowest diversity (0.76) — consistent with "
            "<em>“the autoresearch winner is the most over-confident sampler.”</em></p>"
        )

        parts.append(
            "<div class='callout'>So the auto-metrics give us two signals pulling different ways: "
            "val_bpb agrees with the fluency-proxy (ref-ppl) but disagrees with the diversity-proxy. "
            "This is interim — only 5 ckpts, one prompt-set — and crucially "
            "<strong>auto-metrics aren't a substitute for the human eval</strong>. The Prolific "
            "study still has to tell us what humans actually prefer.</div>"
        )
    else:
        parts.append("<p class='muted'>(auto_metrics.json not present)</p>")

    parts.append(samples_section(dataset))

    parts.append(f"<h2>What's next</h2>")
    parts.append(
        "<ol>"
        "<li>Finish the 3 long-budget runs (~80 min remaining wallclock).</li>"
        "<li>Regenerate <code>dataset.jsonl</code> on the full 8-checkpoint set with the "
        "canonical pipeline (top-p + freq-penalty, strip-trailing).</li>"
        "<li>Recompute auto-metrics + scatter plots on the full set.</li>"
        "<li>Run <code>scripts/select_pairs.py</code> with <code>STEP_GAP=1</code> for "
        "<code>pairs.jsonl</code>.</li>"
        "<li>Hand off to Sean for the actual Prolific A/B preference study — that's the "
        "experiment the auto-metrics here are pre-conditions for, not a substitute.</li>"
        "</ol>"
    )

    parts.append(
        "<h2>Honest caveats</h2>"
        "<ul>"
        "<li>Tiny model (11.5M params), tiny vocab (8,192), limited corpus (50 shards of 6,543). "
        "Anything we conclude is specifically about <em>the autoresearch regime</em>, not about "
        "loss-vs-quality at production scale.</li>"
        "<li>The narrative we wrote in your <code>field_notes.md</code> (lower val_bpb = math "
        "gibberish) didn't fully survive the methodology fixes. With proper sampling + broader "
        "data the catastrophic divergence largely disappears at the auto-metric level. Whether "
        "humans see a divergence is what the Prolific study is for.</li>"
        "<li>Auto-metrics are <em>triangulation axes</em>, not preference data.</li>"
        "</ul>"
    )

    parts.append("</body></html>")

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(parts))
    size_kb = os.path.getsize(OUT) / 1024
    print(f"Wrote {OUT} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
