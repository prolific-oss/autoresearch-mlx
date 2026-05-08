"""
Build training-dynamics + auto-metrics report from the chain log + dataset.

Reads:
- train_chain2.log         — per-step loss + final val_bpb per checkpoint
- auto_metrics.json        — optional, produced by _compute_auto_metrics.py
- dataset.jsonl            — optional, for sample-text snippets in the report

Writes:
- report/training_loss_curves.png
- report/val_bpb_summary.png
- report/val_bpb_vs_<metric>.png       (one per auto-metric, if available)
- report/REPORT.md

Usage:
    uv run scripts/_make_report.py
"""

import json
import os
import re
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "train_chain2.log"
AUTO_METRICS = "auto_metrics.json"
DATASET = "dataset.jsonl"
OUT_DIR = "report"

CKPT_HEADER_RE = re.compile(r"^--- (\S+) \(.*?\) at (.+) ---$")
RESUME_HEADER_RE = re.compile(r"^--- (\S+) \(resuming from snapshot\) at (.+) ---$")
DONE_HEADER_RE = re.compile(r"^--- (\S+) \(already done, skipping\) at (.+) ---$")
STEP_RE = re.compile(r"step\s+(\d+)\s+\([\d.]+%\)\s+\|\s+loss:\s+([\d.]+)\s+\|")
VAL_BPB_RE = re.compile(r"^val_bpb:\s+([\d.]+)")


# ---------------------------------------------------------------------------
# Parse log
# ---------------------------------------------------------------------------

def parse_chain_log(path):
    """Returns dict: ckpt_tag -> {'steps': [(step, loss), ...], 'val_bpb': float|None}.

    If a tag appears in multiple sections (e.g. log contains a previous killed
    chain's runs followed by a fresh chain), only the LATEST section's data is
    kept.
    """
    if not os.path.isfile(path):
        return OrderedDict()
    with open(path) as f:
        text = f.read().replace("\r", "\n")

    # Split into sections by checkpoint-header line.
    sections = []  # list of (tag, [lines])
    current_tag = None
    current_lines = []
    for line in text.splitlines():
        m = CKPT_HEADER_RE.match(line) or RESUME_HEADER_RE.match(line) or DONE_HEADER_RE.match(line)
        if m:
            if current_tag is not None:
                sections.append((current_tag, current_lines))
            current_tag = m.group(1)
            current_lines = []
        else:
            if current_tag is not None:
                current_lines.append(line)
    if current_tag is not None:
        sections.append((current_tag, current_lines))

    # First-seen order, latest-section wins.
    order = []
    seen = set()
    latest = {}
    for tag, lines in sections:
        if tag not in seen:
            order.append(tag)
            seen.add(tag)
        latest[tag] = lines

    runs = OrderedDict()
    for tag in order:
        runs[tag] = {"steps": [], "val_bpb": None}
        for line in latest[tag]:
            m_step = STEP_RE.search(line)
            if m_step:
                runs[tag]["steps"].append((int(m_step.group(1)), float(m_step.group(2))))
            m_bpb = VAL_BPB_RE.match(line.strip())
            if m_bpb:
                runs[tag]["val_bpb"] = float(m_bpb.group(1))
    return runs


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_curves(runs, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    color_iter = iter(plt.cm.viridis([i / max(1, len(runs) - 1) for i in range(len(runs))]))
    for tag, data in runs.items():
        if not data["steps"]:
            continue
        steps, losses = zip(*data["steps"])
        bpb_str = f", val_bpb={data['val_bpb']:.4f}" if data["val_bpb"] is not None else ", in-progress"
        ax.plot(steps, losses, label=f"{tag}{bpb_str}", color=next(color_iter), linewidth=1.5, alpha=0.85)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Training loss (debiased EMA)")
    ax.set_title("Training loss curves per checkpoint")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_val_bpb_summary(runs, out_path):
    done = [(t, d["val_bpb"]) for t, d in runs.items() if d["val_bpb"] is not None]
    if not done:
        return
    done.sort(key=lambda x: x[1])
    tags, bpbs = zip(*done)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#1f77b4" if not t.endswith("-long") else "#d62728" for t in tags]
    ax.barh(range(len(tags)), bpbs, color=colors, edgecolor="black", linewidth=0.5)
    for i, (t, b) in enumerate(zip(tags, bpbs)):
        ax.text(b + 0.005, i, f"{b:.4f}", va="center", fontsize=9)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags)
    ax.set_xlabel("val_bpb (lower is 'better' by metric)")
    ax.set_title("Final val_bpb per checkpoint (blue=5min, red=30min)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_val_bpb_vs_metrics(per_ckpt, out_dir):
    """Scatter of val_bpb vs each auto-metric across checkpoints."""
    if not per_ckpt:
        return []
    keys = [k for k in per_ckpt[0].keys() if k.endswith("_mean")]
    paths = []
    for k in keys:
        metric_label = k[:-len("_mean")]
        xs = [c["val_bpb"] for c in per_ckpt]
        ys = [c[k] for c in per_ckpt]
        labels = [c["checkpoint_id"] for c in per_ckpt]
        fig, ax = plt.subplots(figsize=(8, 5))
        is_long = ["-long" in l for l in labels]
        for x, y, lab, long in zip(xs, ys, labels, is_long):
            color = "#d62728" if long else "#1f77b4"
            ax.scatter(x, y, color=color, s=50, edgecolor="black", linewidth=0.5)
            ax.annotate(lab, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("val_bpb")
        ax.set_ylabel(metric_label)
        ax.set_title(f"val_bpb vs {metric_label}\n(red=30min, blue=5min)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"val_bpb_vs_{metric_label}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_markdown(runs, per_ckpt, metric_imgs, dataset_recs, out_path):
    lines = ["# Training & auto-metrics report\n"]
    lines.append("## Per-checkpoint val_bpb\n")
    lines.append("| checkpoint | val_bpb | steps | type |")
    lines.append("|---|---:|---:|---|")
    done = [(t, d) for t, d in runs.items() if d["val_bpb"] is not None]
    done.sort(key=lambda x: x[1]["val_bpb"])
    for tag, data in done:
        kind = "long (30 min)" if tag.endswith("-long") else "short (5 min)"
        n_steps = data["steps"][-1][0] if data["steps"] else "n/a"
        lines.append(f"| {tag} | {data['val_bpb']:.4f} | {n_steps} | {kind} |")
    in_progress = [t for t, d in runs.items() if d["val_bpb"] is None]
    if in_progress:
        lines.append(f"\n*In-progress (no val_bpb yet): {', '.join(in_progress)}*\n")

    lines.append("\n## Training loss curves\n")
    lines.append("![training loss](training_loss_curves.png)\n")

    lines.append("## Final val_bpb summary\n")
    lines.append("![val_bpb](val_bpb_summary.png)\n")

    if per_ckpt:
        lines.append("## Per-checkpoint auto-metrics\n")
        cols = [k for k in per_ckpt[0].keys() if k != "checkpoint_id"]
        header = "| checkpoint | " + " | ".join(c.replace("_mean", "") for c in cols) + " |"
        sep = "|---|" + "|".join(":---:" for _ in cols) + "|"
        lines.append(header)
        lines.append(sep)
        for c in per_ckpt:
            row = [c["checkpoint_id"]]
            for col in cols:
                v = c[col]
                if isinstance(v, float):
                    row.append(f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}")
                else:
                    row.append(str(v))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        lines.append("## val_bpb vs each auto-metric\n")
        for img in metric_imgs:
            name = os.path.basename(img)
            lines.append(f"![{name}]({name})")
        lines.append("")

    if dataset_recs:
        lines.append("\n## Sample completions (one prompt across all checkpoints)\n")
        sample_pid = "biomedical_00"
        sample = next((r for r in dataset_recs if r["prompt_id"] == sample_pid), None)
        if sample:
            lines.append(f"**Prompt:** {sample['prompt_text']}\n")
            ckpt_records = [r for r in dataset_recs if r["prompt_id"] == sample_pid]
            ckpt_records.sort(key=lambda r: r["val_bpb"])
            for r in ckpt_records:
                lines.append(f"**{r['checkpoint_id']}** (val_bpb={r['val_bpb']:.4f})  ")
                snippet = r["generated_text"][:300].replace("\n", " ")
                lines.append(f"> {snippet}...\n")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    runs = parse_chain_log(LOG)
    print(f"Parsed {len(runs)} checkpoint sections from {LOG}")
    for tag, data in runs.items():
        n_steps = len(data["steps"])
        print(f"  {tag}: {n_steps} step entries, val_bpb={data['val_bpb']}")

    plot_training_curves(runs, os.path.join(OUT_DIR, "training_loss_curves.png"))
    plot_val_bpb_summary(runs, os.path.join(OUT_DIR, "val_bpb_summary.png"))

    per_ckpt = []
    if os.path.isfile(AUTO_METRICS):
        with open(AUTO_METRICS) as f:
            data = json.load(f)
        per_ckpt = data.get("per_checkpoint", [])
    metric_imgs = plot_val_bpb_vs_metrics(per_ckpt, OUT_DIR) if per_ckpt else []

    dataset_recs = []
    if os.path.isfile(DATASET):
        with open(DATASET) as f:
            dataset_recs = [json.loads(l) for l in f if l.strip()]

    build_markdown(runs, per_ckpt, metric_imgs, dataset_recs, os.path.join(OUT_DIR, "REPORT.md"))
    print(f"\nReport written to {OUT_DIR}/REPORT.md")


if __name__ == "__main__":
    main()
