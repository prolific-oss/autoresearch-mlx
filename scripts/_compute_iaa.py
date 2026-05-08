"""
Inter-annotator agreement (IAA) on the pairwise preference task.

Takes 2+ annotation JSON files (as exported from annotation_ui_*.html),
matches them by pair_id, and reports:

- Percent agreement (raw)
- Cohen's kappa for each pair of annotators (chance-corrected)
- Confusion matrix per annotator pair
- Per-pair-group agreement breakdown (which val_bpb gaps annotators agree on most)
- Agreement with val_bpb (A-pick rate per annotator: % of picks that match the
  lower-val_bpb-side prediction; >50% means the annotator tends to agree with
  the metric, ignoring ties)
- A binomial sign-test p-value vs. 50/50 for each annotator

No new dependencies — uses stdlib only.

Usage:
    uv run scripts/_compute_iaa.py annotations_nora.json annotations_viviana.json
    uv run scripts/_compute_iaa.py *.json --out iaa_report.json
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict


LABELS = ("A", "tie", "B")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_annotations(path):
    """Returns (annotator_name, dict[pair_id -> annotation])."""
    with open(path) as f:
        data = json.load(f)
    name = os.path.splitext(os.path.basename(path))[0]
    if name.startswith("annotations_"):
        name = name[len("annotations_"):]
    by_id = {a["pair_id"]: a for a in data["annotations"]}
    return name, by_id


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def cohens_kappa(labels1, labels2):
    """Two annotators, categorical labels. Returns (kappa, p_o, p_e)."""
    assert len(labels1) == len(labels2)
    n = len(labels1)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p_o = sum(1 for a, b in zip(labels1, labels2) if a == b) / n
    cats = sorted(set(labels1) | set(labels2))
    p_e = 0.0
    for c in cats:
        p1 = sum(1 for x in labels1 if x == c) / n
        p2 = sum(1 for x in labels2 if x == c) / n
        p_e += p1 * p2
    if abs(1 - p_e) < 1e-12:
        return float("nan"), p_o, p_e
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa, p_o, p_e


def confusion_matrix(labels1, labels2, cats=LABELS):
    cm = {a: {b: 0 for b in cats} for a in cats}
    for a, b in zip(labels1, labels2):
        if a in cm and b in cm[a]:
            cm[a][b] += 1
    return cm


def two_sided_binomial_p(k, n, p=0.5):
    """Exact two-sided binomial test p-value (k successes in n trials vs p)."""
    if n == 0:
        return float("nan")

    def binom_pmf(k, n, p):
        # log-space combination to avoid overflow
        log_c = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        return math.exp(log_c + k * math.log(p) + (n - k) * math.log(1 - p)) if 0 < p < 1 else (1.0 if k == n * p else 0.0)

    # P(X = k_obs) and the symmetric tail-probability sum
    pmf = [binom_pmf(i, n, p) for i in range(n + 1)]
    obs = pmf[k]
    return sum(prob for prob in pmf if prob <= obs + 1e-12)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def kappa_strength(k):
    if k != k:  # NaN
        return "n/a"
    if k < 0: return "worse than chance"
    if k < 0.20: return "slight"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"


def print_pairwise_iaa(name1, ann1, name2, ann2):
    common = sorted(set(ann1) & set(ann2))
    print(f"\n=== {name1}  vs  {name2}  ({len(common)} shared pairs) ===")
    if not common:
        print("  (no shared pairs)")
        return None
    l1 = [ann1[p]["canonical_winner"] for p in common]
    l2 = [ann2[p]["canonical_winner"] for p in common]
    kappa, p_o, p_e = cohens_kappa(l1, l2)
    cm = confusion_matrix(l1, l2)
    print(f"  Percent agreement: {p_o*100:5.1f}%   (chance: {p_e*100:5.1f}%)")
    print(f"  Cohen's kappa:     {kappa:+.3f}   ({kappa_strength(kappa)})")
    print(f"  Confusion matrix (rows = {name1}, cols = {name2}):")
    header = "        " + "".join(f"{c:>6}" for c in LABELS) + "   total"
    print(header)
    for r in LABELS:
        row = f"  {r:>4}: " + "".join(f"{cm[r][c]:>6}" for c in LABELS)
        print(row + f"   {sum(cm[r].values()):>5}")
    return {
        "annotators": [name1, name2],
        "n_shared": len(common),
        "percent_agreement": p_o,
        "expected_agreement": p_e,
        "cohens_kappa": kappa,
        "kappa_strength": kappa_strength(kappa),
        "confusion_matrix": cm,
    }


def print_vs_valbpb(name, ann):
    """How often did the annotator pick the lower-val_bpb side (canonical A)?"""
    items = list(ann.values())
    n = len(items)
    a = sum(1 for x in items if x["canonical_winner"] == "A")
    b = sum(1 for x in items if x["canonical_winner"] == "B")
    t = sum(1 for x in items if x["canonical_winner"] == "tie")
    decisive = a + b
    a_rate = a / decisive if decisive else float("nan")
    p = two_sided_binomial_p(a, decisive) if decisive else float("nan")
    print(f"\n=== {name}  vs  val_bpb (A = lower-val_bpb-side wins) ===")
    print(f"  n={n}: A={a}, B={b}, tie={t}")
    print(f"  Among {decisive} decisive picks: {a_rate*100:5.1f}% chose lower-val_bpb side")
    print(f"  Two-sided binomial p-value vs 50/50: {p:.4f}")
    return {
        "annotator": name,
        "n_total": n,
        "n_A": a, "n_B": b, "n_tie": t,
        "a_rate_among_decisive": a_rate,
        "binomial_p_vs_50_50": p,
    }


def per_pair_group_breakdown(annotators):
    """For each (checkpoint_a, checkpoint_b) group, agreement rates."""
    # annotators is list of (name, dict[pair_id -> ann])
    groups = defaultdict(list)
    for name, ann in annotators:
        for pid, a in ann.items():
            key = (a["checkpoint_a"], a["checkpoint_b"], a["val_bpb_a"], a["val_bpb_b"])
            groups[key].append((name, a["canonical_winner"]))

    print("\n=== Per-pair-group breakdown ===")
    print(f"  {'group (val_bpb gap)':<55} {'n_annotated':>11} {'majority':>10}")
    for key, items in sorted(groups.items(), key=lambda kv: -(kv[0][3] - kv[0][2])):
        cida, cidb, va, vb = key
        gap = vb - va
        labels = [w for _, w in items]
        c = Counter(labels)
        majority = max(c.items(), key=lambda kv: kv[1])
        desc = f"{cida} vs {cidb} (Δ={gap:+.3f})"
        print(f"  {desc:<55} {len(items):>11} {majority[0]+f' ({majority[1]}/{len(items)})':>15}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="Annotation JSON files (annotations_*.json)")
    ap.add_argument("--out", default=None, help="Optional path to save the IAA summary JSON.")
    args = ap.parse_args()

    annotators = [load_annotations(p) for p in args.files]
    if len(annotators) < 1:
        raise SystemExit("Need at least one annotation file.")

    print(f"Loaded {len(annotators)} annotator(s):")
    for name, ann in annotators:
        print(f"  {name}: {len(ann)} pairs")

    summary = {"annotators": [], "pairwise_iaa": [], "vs_valbpb": []}

    for name, ann in annotators:
        summary["vs_valbpb"].append(print_vs_valbpb(name, ann))
        summary["annotators"].append({"name": name, "n_pairs": len(ann)})

    if len(annotators) >= 2:
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                pw = print_pairwise_iaa(
                    annotators[i][0], annotators[i][1],
                    annotators[j][0], annotators[j][1],
                )
                if pw is not None:
                    summary["pairwise_iaa"].append(pw)

    per_pair_group_breakdown(annotators)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {args.out}")


if __name__ == "__main__":
    main()
