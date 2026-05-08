"""
Build a standalone HTML annotation UI from pairs.jsonl.

The output is a single file that:
- Embeds the full pairs dataset inline (no server needed)
- Randomizes A/B presentation per pair (deterministic per session)
- Hides checkpoint identity during annotation (blind A/B)
- Persists annotations to localStorage between reloads
- Lets you export annotations as JSON when done
- Includes a "reveal" view to inspect which checkpoint produced which side

Usage:
    uv run scripts/_build_annotation_ui.py
"""

import argparse
import json
import os

DEFAULT_PAIRS_FILE = "pairs.jsonl"
DEFAULT_OUT = os.path.join("report", "annotation_ui.html")
DEFAULT_STORAGE_KEY = "autoresearch_annotations_v1"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

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
  --pick-a: #2f6fb8;
  --pick-b: #b8602f;
  --pick-tie: #6b6b6b;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  color: var(--fg);
  background: var(--bg);
  margin: 0;
}
header {
  position: sticky; top: 0; z-index: 10;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  padding: 14px 28px;
  display: flex; align-items: center; gap: 24px;
}
header h1 { font-size: 18px; margin: 0; font-weight: 600; letter-spacing: -0.01em; }
header .progress { color: var(--muted); font-size: 14px; font-variant-numeric: tabular-nums; }
header .progress-bar { flex: 1; height: 4px; background: var(--panel); border-radius: 2px; overflow: hidden; }
header .progress-fill { height: 100%; background: var(--accent); transition: width .2s; }
header .actions { display: flex; gap: 8px; }
header .actions button {
  background: var(--bg);
  border: 1px solid var(--border);
  color: var(--fg);
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}
header .actions button:hover { background: var(--panel); }

main { max-width: 1100px; margin: 0 auto; padding: 28px; }
section.prompt {
  background: var(--accent-soft);
  border-left: 3px solid var(--accent);
  padding: 14px 18px;
  margin-bottom: 22px;
  border-radius: 4px;
}
section.prompt .label {
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--accent); margin-bottom: 4px; font-weight: 600;
}
section.prompt .text { font-size: 16px; line-height: 1.5; }
section.prompt .meta { color: var(--muted); font-size: 13px; margin-top: 6px; }

section.completions { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 22px; }
.completion {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 16px 18px;
}
.completion h3 {
  font-size: 24px; margin: 0 0 12px;
  font-weight: 700;
  letter-spacing: 0.01em;
}
.completion .text {
  font-size: 15px; line-height: 1.55;
  white-space: pre-wrap; word-wrap: break-word;
}
.completion.reveal-tag .ckpt-info {
  display: block;
  margin-bottom: 8px;
  padding: 6px 10px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 3px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: var(--muted);
}
.completion .ckpt-info { display: none; }

section.question { margin-bottom: 22px; }
section.question .q {
  font-size: 16px; font-weight: 500; margin-bottom: 12px;
}
.choices { display: flex; gap: 10px; }
.choices button {
  flex: 1;
  padding: 14px 18px;
  border: 1px solid var(--border);
  background: var(--bg);
  border-radius: 4px;
  cursor: pointer;
  font-size: 15px;
  font-weight: 500;
  transition: all .12s;
}
.choices button:hover { background: var(--panel); }
.choices button.picked-A { background: var(--pick-a); color: #fff; border-color: var(--pick-a); }
.choices button.picked-tie { background: var(--pick-tie); color: #fff; border-color: var(--pick-tie); }
.choices button.picked-B { background: var(--pick-b); color: #fff; border-color: var(--pick-b); }
.choices kbd {
  display: inline-block; min-width: 20px; padding: 1px 5px; margin-left: 6px;
  background: rgba(0,0,0,0.06); border-radius: 3px;
  font-size: 11px; font-family: ui-monospace, monospace;
}

section.comment textarea {
  width: 100%;
  min-height: 60px;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 4px;
  font: inherit;
  resize: vertical;
}

nav.pager {
  display: flex; justify-content: space-between; align-items: center;
  margin-top: 28px; padding-top: 18px; border-top: 1px solid var(--border);
}
nav.pager button {
  background: var(--bg); border: 1px solid var(--border);
  padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 14px;
}
nav.pager button:hover:not(:disabled) { background: var(--panel); }
nav.pager button:disabled { color: var(--muted); cursor: not-allowed; }

.help { color: var(--muted); font-size: 13px; }
.help kbd {
  display: inline-block; min-width: 18px; padding: 1px 4px;
  background: rgba(0,0,0,0.06); border-radius: 3px;
  font-size: 11px; font-family: ui-monospace, monospace;
}

#export-modal {
  display: none;
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.4);
  align-items: center; justify-content: center;
  z-index: 100;
}
#export-modal.show { display: flex; }
#export-modal .body {
  background: var(--bg); border-radius: 6px;
  padding: 22px 26px; width: 480px;
  box-shadow: 0 10px 40px rgba(0,0,0,0.2);
}
#export-modal h2 { margin: 0 0 12px; font-size: 18px; }
#export-modal .summary { font-size: 14px; color: var(--muted); margin-bottom: 14px; }
#export-modal .actions { display: flex; gap: 10px; justify-content: flex-end; margin-top: 18px; }
#export-modal button {
  padding: 8px 16px; border: 1px solid var(--border); background: var(--bg);
  border-radius: 4px; cursor: pointer; font-size: 14px;
}
#export-modal button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
"""


# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------

JS_TEMPLATE = """
const PAIRS = __PAIRS_JSON__;
const STORAGE_KEY = '__STORAGE_KEY__';
const SESSION_KEY = '__STORAGE_KEY___session';
const IDX_KEY     = '__STORAGE_KEY___idx';

// Stable session salt — used for deterministic A/B randomization.
let session = JSON.parse(localStorage.getItem(SESSION_KEY) || 'null');
if (!session) {
  session = { salt: Math.floor(Math.random() * 1e9), startedAt: new Date().toISOString() };
  localStorage.setItem(SESSION_KEY, JSON.stringify(session));
}

// Pair ID = stable identifier across reloads.
function pairId(p) { return `${p.prompt_id}__${p.checkpoint_a}__${p.checkpoint_b}`; }

// Deterministic "random" 0/1 per pair from session salt.
function shouldFlip(pid) {
  let h = session.salt;
  for (let i = 0; i < pid.length; i++) {
    h = (h * 31 + pid.charCodeAt(i)) | 0;
  }
  return (Math.abs(h) % 2) === 1;
}

let annotations = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
let revealMode = false;
let currentIdx = parseInt(localStorage.getItem(IDX_KEY) || '0', 10);
if (currentIdx >= PAIRS.length || currentIdx < 0) currentIdx = 0;

const $ = (id) => document.getElementById(id);

function saveAnnotations() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
  localStorage.setItem(IDX_KEY, String(currentIdx));
}

function showPair() {
  const p = PAIRS[currentIdx];
  const pid = pairId(p);
  const flip = shouldFlip(pid);

  const leftCkpt = flip ? p.checkpoint_b : p.checkpoint_a;
  const rightCkpt = flip ? p.checkpoint_a : p.checkpoint_b;
  const leftText = flip ? p.generated_text_b : p.generated_text_a;
  const rightText = flip ? p.generated_text_a : p.generated_text_b;
  const leftBpb = flip ? p.val_bpb_b : p.val_bpb_a;
  const rightBpb = flip ? p.val_bpb_a : p.val_bpb_b;

  $('prompt-text').textContent = p.prompt_text;
  $('prompt-meta').textContent = `Discipline: ${p.discipline || '—'} · prompt_id ${p.prompt_id}`;
  $('completion-a-text').textContent = leftText;
  $('completion-b-text').textContent = rightText;

  $('completion-a-info').textContent = `${leftCkpt} · val_bpb ${leftBpb.toFixed(4)}`;
  $('completion-b-info').textContent = `${rightCkpt} · val_bpb ${rightBpb.toFixed(4)}`;

  // Existing annotation, if any
  const ann = annotations[pid];
  $('comment').value = ann?.comment || '';
  document.querySelectorAll('.choices button').forEach(b => {
    b.classList.remove('picked-A', 'picked-tie', 'picked-B');
  });
  if (ann?.ui_choice) {
    const btn = document.querySelector(`[data-choice="${ann.ui_choice}"]`);
    if (btn) btn.classList.add(`picked-${ann.ui_choice}`);
  }

  // Pager + progress
  $('progress-text').textContent = `Pair ${currentIdx + 1} of ${PAIRS.length}`;
  $('progress-fill').style.width = `${((currentIdx + 1) / PAIRS.length) * 100}%`;
  $('prev').disabled = currentIdx === 0;
  $('next').disabled = currentIdx === PAIRS.length - 1;

  // Annotation count in header
  const annotated = Object.keys(annotations).length;
  $('annotated-count').textContent = `${annotated} annotated`;

  // Reveal toggle
  document.querySelectorAll('.completion').forEach(el => {
    el.classList.toggle('reveal-tag', revealMode);
  });
}

function recordChoice(choice) {
  const p = PAIRS[currentIdx];
  const pid = pairId(p);
  const flip = shouldFlip(pid);
  // Map UI choice (A/B as displayed) back to actual ckpt mapping.
  let actualWinner = choice;
  if (choice === 'A') actualWinner = flip ? 'B' : 'A';   // UI-A is actual ckpt_b if flipped
  else if (choice === 'B') actualWinner = flip ? 'A' : 'B';
  // tie stays tie
  annotations[pid] = {
    pair_id: pid,
    prompt_id: p.prompt_id,
    discipline: p.discipline,
    checkpoint_a: p.checkpoint_a,   // canonical (lower val_bpb side per select_pairs)
    checkpoint_b: p.checkpoint_b,
    val_bpb_a: p.val_bpb_a,
    val_bpb_b: p.val_bpb_b,
    display_left_was: flip ? 'B' : 'A',  // which canonical side appeared on the left in UI
    ui_choice: choice,                    // what the user clicked (A/tie/B as shown)
    canonical_winner: actualWinner,       // mapped back: A means canonical ckpt_a wins (lower val_bpb)
    comment: $('comment').value || '',
    annotated_at: new Date().toISOString(),
  };
  saveAnnotations();
  // Apply the visual highlight immediately so the click feels responsive,
  // independent of whether showPair re-runs.
  document.querySelectorAll('.choices button').forEach(b => {
    b.classList.remove('picked-A', 'picked-tie', 'picked-B');
  });
  const pressed = document.querySelector(`[data-choice="${choice}"]`);
  if (pressed) pressed.classList.add(`picked-${choice}`);
  // Update the "annotated" header counter
  $('annotated-count').textContent = `${Object.keys(annotations).length} annotated`;
}

function next() {
  // Persist comment on advance even if no choice yet
  const p = PAIRS[currentIdx];
  const pid = pairId(p);
  const comment = $('comment').value || '';
  if (annotations[pid]) {
    annotations[pid].comment = comment;
    saveAnnotations();
  }
  if (currentIdx < PAIRS.length - 1) currentIdx++;
  saveAnnotations();
  showPair();
}

function prev() {
  if (currentIdx > 0) currentIdx--;
  saveAnnotations();
  showPair();
}

function exportJson() {
  const out = {
    session,
    exportedAt: new Date().toISOString(),
    n_pairs: PAIRS.length,
    n_annotated: Object.keys(annotations).length,
    annotations: Object.values(annotations),
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `annotations_${new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function showSummary() {
  const total = PAIRS.length;
  const done = Object.keys(annotations).length;
  let aWins = 0, bWins = 0, ties = 0;
  for (const a of Object.values(annotations)) {
    if (a.canonical_winner === 'A') aWins++;
    else if (a.canonical_winner === 'B') bWins++;
    else if (a.canonical_winner === 'tie') ties++;
  }
  $('export-summary').innerHTML =
    `<strong>${done} of ${total}</strong> pairs annotated.<br>` +
    `Among annotated: <strong>${aWins}</strong> A-canonical wins (lower-val_bpb side), ` +
    `<strong>${bWins}</strong> B-canonical wins (higher-val_bpb side), ` +
    `<strong>${ties}</strong> ties.`;
  $('export-modal').classList.add('show');
}

function clearAll() {
  if (!confirm('Clear all annotations and start over? This cannot be undone.')) return;
  annotations = {};
  currentIdx = 0;
  saveAnnotations();
  showPair();
  $('export-modal').classList.remove('show');
}

// Wire up
function init() {
  // Event delegation for choice buttons — robust to inner elements like <kbd>
  document.body.addEventListener('click', (e) => {
    const choiceBtn = e.target.closest('[data-choice]');
    if (choiceBtn) {
      e.preventDefault();
      recordChoice(choiceBtn.dataset.choice);
      return;
    }
    const action = e.target.closest('[data-action]');
    if (action) {
      e.preventDefault();
      const a = action.dataset.action;
      if (a === 'next') next();
      else if (a === 'prev') prev();
      else if (a === 'export') showSummary();
      else if (a === 'reveal') {
        revealMode = !revealMode;
        action.textContent = revealMode ? 'Hide identities' : 'Reveal identities';
        showPair();
      }
      else if (a === 'clear') clearAll();
      else if (a === 'do-export') exportJson();
      else if (a === 'close-modal') $('export-modal').classList.remove('show');
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'a' || e.key === 'A') recordChoice('A');
    else if (e.key === 'b' || e.key === 'B') recordChoice('B');
    else if (e.key === 't' || e.key === 'T') recordChoice('tie');
    else if (e.key === 'ArrowRight') next();
    else if (e.key === 'ArrowLeft') prev();
  });
  showPair();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=DEFAULT_PAIRS_FILE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--storage-key", default=DEFAULT_STORAGE_KEY,
                    help="localStorage key — use a distinct value for each separate annotation deck.")
    ap.add_argument("--title", default="Pairwise preference",
                    help="Header title shown in the UI.")
    args = ap.parse_args()

    if not os.path.isfile(args.pairs):
        raise SystemExit(f"Missing {args.pairs} — run scripts/select_pairs.py or _build_diagnostic_pairs.py first.")
    with open(args.pairs) as f:
        pairs = [json.loads(l) for l in f if l.strip()]
    if not pairs:
        raise SystemExit(f"{args.pairs} is empty.")

    pairs_json = json.dumps(pairs)
    js = JS_TEMPLATE.replace("__PAIRS_JSON__", pairs_json).replace("__STORAGE_KEY__", args.storage_key)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Autoresearch × Prolific — {args.title}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>{args.title}</h1>
  <span class="progress" id="progress-text">Pair 1 of {len(pairs)}</span>
  <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
  <span class="progress" id="annotated-count">0 annotated</span>
  <div class="actions">
    <button data-action="reveal" title="Show which checkpoint produced which side (post-annotation only)">Reveal identities</button>
    <button data-action="export">Export…</button>
    <button data-action="clear" title="Wipe all annotations">Clear</button>
  </div>
</header>

<main>
  <section class="prompt">
    <div class="label">Prompt</div>
    <div class="text" id="prompt-text"></div>
    <div class="meta" id="prompt-meta"></div>
  </section>

  <section class="completions">
    <div class="completion" id="completion-a">
      <h3>A</h3>
      <div class="ckpt-info" id="completion-a-info"></div>
      <div class="text" id="completion-a-text"></div>
    </div>
    <div class="completion" id="completion-b">
      <h3>B</h3>
      <div class="ckpt-info" id="completion-b-info"></div>
      <div class="text" id="completion-b-text"></div>
    </div>
  </section>

  <section class="question">
    <div class="q">Which continuation reads better — more natural and trustworthy as a science explanation?</div>
    <div class="choices">
      <button data-choice="A">A is better<kbd>A</kbd></button>
      <button data-choice="tie">About equal<kbd>T</kbd></button>
      <button data-choice="B">B is better<kbd>B</kbd></button>
    </div>
  </section>

  <section class="comment">
    <textarea id="comment" placeholder="Optional comment / what made you pick…"></textarea>
  </section>

  <nav class="pager">
    <button data-action="prev">← Previous</button>
    <span class="help">Keys: <kbd>A</kbd> / <kbd>T</kbd> / <kbd>B</kbd> to choose · <kbd>←</kbd> / <kbd>→</kbd> to navigate</span>
    <button data-action="next">Next →</button>
  </nav>
</main>

<div id="export-modal" role="dialog" aria-modal="true">
  <div class="body">
    <h2>Export annotations</h2>
    <div class="summary" id="export-summary"></div>
    <p class="help">Downloads a JSON file with one entry per annotated pair: prompt info, both checkpoint identities, your choice (mapped back to canonical A=lower-val_bpb), the display side they appeared on, and any comment.</p>
    <div class="actions">
      <button data-action="close-modal">Close</button>
      <button data-action="do-export" class="primary">Download JSON</button>
    </div>
  </div>
</div>

<script>{js}</script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"Wrote {args.out} ({size_kb:.0f} KB) — {len(pairs)} pairs embedded")


if __name__ == "__main__":
    main()
