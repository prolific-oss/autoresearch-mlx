"""
Build the sanity-check UI from the proper generations (different seeds, EOS-stopping).
Also prepares the full 500-instance pair grid for Prolific deployment.

Visual style mirrors report/annotation_ui_diagnostic.html (sticky header, warm
palette, color-coded buttons, reveal-identities toggle, export modal).

Usage:
    python build_prolific_ui.py
       (assumes prolific_generations.json is present in same dir)
"""
import json
import random
import itertools
from pathlib import Path

EVAL_DIR = Path(__file__).parent

MODELS = ["base", "prod-baseline", "prod-adambeta2", "prod-bestbet", "prod-bestbet-v2"]
VAL_LABELS = {
    "base": "untrained (no DPO)",
    "prod-baseline": "val=0.464",
    "prod-adambeta2": "val=0.492",
    "prod-bestbet": "val=0.628",
    "prod-bestbet-v2": "val=0.648",
}
ALL_PAIRS = list(itertools.combinations(MODELS, 2))


def build_pair_grid(generations, prompts, prompt_ids):
    random.seed(42)
    grid = []
    for pi, prompt in enumerate(prompts):
        for m1, m2 in ALL_PAIRS:
            a, b = (m1, m2) if random.random() < 0.5 else (m2, m1)
            grid.append({
                "id": f"{prompt_ids[pi]}__{m1}__vs__{m2}",
                "prompt_id": prompt_ids[pi],
                "prompt": prompt,
                "axis": f"{m1} vs {m2}",
                "a_model": a,
                "b_model": b,
                "a_val_label": VAL_LABELS[a],
                "b_val_label": VAL_LABELS[b],
                "a_text": generations[a][pi],
                "b_text": generations[b][pi],
            })
    return grid


def sample_sanity_pairs(grid, n_per_axis=5):
    by_axis = {}
    for p in grid:
        by_axis.setdefault(p["axis"], []).append(p)
    sampled = []
    rng = random.Random(7)
    for axis, items in by_axis.items():
        rng.shuffle(items)
        sampled.extend(items[:n_per_axis])
    rng.shuffle(sampled)
    return sampled


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --fg: #1a1a1a;
  --muted: #5b5b5b;
  --bg: #ffffff;
  --panel: #f7f7f4;
  --accent: #b85c00;
  --accent-soft: #fcefe1;
  --border: #d8d6d2;
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
  padding: 16px 28px;
  display: flex; align-items: center; gap: 22px;
}
header h1 { font-size: 17px; margin: 0; font-weight: 600; letter-spacing: -0.01em; }
header .progress { color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; white-space: nowrap; }
header .progress-bar { flex: 1; height: 4px; background: var(--panel); border-radius: 2px; overflow: hidden; min-width: 100px; }
header .progress-fill { height: 100%; background: var(--accent); transition: width .25s ease; }

main { max-width: 1100px; margin: 0 auto; padding: 32px 28px; }

.intro {
  background: var(--accent-soft);
  border-radius: 6px;
  padding: 18px 22px;
  margin-bottom: 28px;
  font-size: 15px;
  line-height: 1.55;
}
.intro h2 { margin: 0 0 8px; font-size: 16px; font-weight: 600; color: var(--accent); }
.intro p { margin: 0; }
.intro.dismissed { display: none; }
.intro .dismiss {
  margin-top: 12px;
  background: var(--bg); border: 1px solid var(--border);
  padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 13px;
}
.intro .dismiss:hover { background: var(--panel); }

section.prompt {
  background: var(--accent-soft);
  border-left: 3px solid var(--accent);
  padding: 16px 20px;
  margin-bottom: 22px;
  border-radius: 4px;
}
section.prompt .label {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--accent); margin-bottom: 6px; font-weight: 600;
}
section.prompt .text { font-size: 16px; line-height: 1.55; }

section.completions { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 24px; }
.completion {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 18px 20px;
}
.completion h3 {
  font-size: 22px; margin: 0 0 14px;
  font-weight: 700;
  letter-spacing: 0.01em;
  color: var(--muted);
}
.completion .text {
  font-size: 15px; line-height: 1.6;
  white-space: pre-wrap; word-wrap: break-word;
}

section.question { margin-bottom: 18px; }
section.question .q {
  font-size: 16px; font-weight: 500; margin-bottom: 14px;
}
.choices { display: flex; gap: 10px; }
.choices button {
  flex: 1;
  padding: 14px 18px;
  border: 1px solid var(--border);
  background: var(--bg);
  border-radius: 6px;
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
  display: inline-block; min-width: 20px; padding: 1px 5px; margin-left: 8px;
  background: rgba(255,255,255,0.18); border-radius: 3px;
  font-size: 11px; font-family: ui-monospace, monospace;
}
.choices button:not(.picked-A):not(.picked-tie):not(.picked-B) kbd {
  background: rgba(0,0,0,0.06);
}

section.comment { margin-bottom: 12px; }
section.comment label {
  display: block; font-size: 13px; color: var(--muted); margin-bottom: 6px;
}
section.comment textarea {
  width: 100%;
  min-height: 56px;
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
  padding: 8px 18px; border-radius: 4px; cursor: pointer; font-size: 14px;
}
nav.pager button:hover:not(:disabled) { background: var(--panel); }
nav.pager button:disabled { color: var(--muted); cursor: not-allowed; }

.help { color: var(--muted); font-size: 12px; }
.help kbd {
  display: inline-block; min-width: 18px; padding: 1px 4px;
  background: rgba(0,0,0,0.06); border-radius: 3px;
  font-size: 11px; font-family: ui-monospace, monospace;
}

.submit-area {
  margin-top: 28px; padding: 20px 22px;
  background: var(--accent-soft); border-radius: 6px;
  display: flex; justify-content: space-between; align-items: center;
}
.submit-area .summary { font-size: 14px; color: var(--muted); }
.submit-area button {
  padding: 10px 22px; border: none;
  background: var(--accent); color: #fff;
  border-radius: 4px; cursor: pointer; font-size: 15px; font-weight: 500;
}
.submit-area button:hover { filter: brightness(1.05); }
.submit-area button:disabled { background: var(--border); color: var(--muted); cursor: not-allowed; filter: none; }
</style>
</head>
<body>
<header>
  <h1>__HEADER_TITLE__</h1>
  <span class="progress" id="progress-text">Pair 1 of N</span>
  <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
  <span class="progress" id="annotated-count">0 of N answered</span>
</header>

<main>
  <div class="intro" id="intro">
    <h2>Compare AI responses</h2>
    <p>You'll see a prompt and two AI-generated responses (A and B). Pick the one you find <strong>clearer, more helpful, and more on-topic</strong>. If they feel about equal, that's a valid answer too.</p>
    <p style="margin-top: 8px;">There are no right or wrong answers — your honest preference is what we're looking for.</p>
    <button class="dismiss" data-action="dismiss-intro">Got it</button>
  </div>

  <section class="prompt">
    <div class="label">Prompt</div>
    <div class="text" id="prompt-text"></div>
  </section>

  <section class="completions">
    <div class="completion" id="completion-a">
      <h3>A</h3>
      <div class="text" id="completion-a-text"></div>
    </div>
    <div class="completion" id="completion-b">
      <h3>B</h3>
      <div class="text" id="completion-b-text"></div>
    </div>
  </section>

  <section class="question">
    <div class="q">Which response do you prefer?</div>
    <div class="choices">
      <button data-choice="A">A is better<kbd>A</kbd></button>
      <button data-choice="tie">About equal<kbd>T</kbd></button>
      <button data-choice="B">B is better<kbd>B</kbd></button>
    </div>
  </section>

  <section class="comment">
    <label for="comment">Optional: anything notable about why you picked? (one short sentence is plenty)</label>
    <textarea id="comment" placeholder=""></textarea>
  </section>

  <nav class="pager">
    <button data-action="prev" id="prev">← Previous</button>
    <span class="help">Keys: <kbd>A</kbd> · <kbd>T</kbd> · <kbd>B</kbd> to choose · <kbd>←</kbd> <kbd>→</kbd> to navigate</span>
    <button data-action="next" id="next">Next →</button>
  </nav>

  <div class="submit-area">
    <div class="summary" id="submit-summary">Answer all pairs to submit.</div>
    <button id="submit-btn" disabled>Submit answers</button>
  </div>
</main>

<script id="pairs-data" type="application/json">__PAIRS_JSON__</script>
<script>
const PAIRS = JSON.parse(document.getElementById('pairs-data').textContent);
const STORAGE_KEY = '__STORAGE_KEY__';
const SESSION_KEY = STORAGE_KEY + '_session';
const IDX_KEY     = STORAGE_KEY + '_idx';
const INTRO_KEY   = STORAGE_KEY + '_intro_dismissed';

let session = JSON.parse(localStorage.getItem(SESSION_KEY) || 'null');
if (!session) {
  session = { salt: Math.floor(Math.random() * 1e9), startedAt: new Date().toISOString() };
  localStorage.setItem(SESSION_KEY, JSON.stringify(session));
}

function pairId(p) { return p.id; }
function shouldFlip(pid) {
  let h = session.salt;
  for (let i = 0; i < pid.length; i++) {
    h = (h * 31 + pid.charCodeAt(i)) | 0;
  }
  return (Math.abs(h) % 2) === 1;
}

let annotations = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
function firstUnlabeled() {
  for (let i = 0; i < PAIRS.length; i++) {
    if (!annotations[pairId(PAIRS[i])]) return i;
  }
  return PAIRS.length - 1;
}
let savedIdx = parseInt(localStorage.getItem(IDX_KEY) || '-1', 10);
let currentIdx = (savedIdx >= 0 && savedIdx < PAIRS.length && !annotations[pairId(PAIRS[savedIdx])]) ? savedIdx : firstUnlabeled();
if (localStorage.getItem(INTRO_KEY) === '1') {
  document.getElementById('intro').classList.add('dismissed');
}

const $ = (id) => document.getElementById(id);

function saveAnnotations() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
  localStorage.setItem(IDX_KEY, String(currentIdx));
}

function showPair() {
  const p = PAIRS[currentIdx];
  const pid = pairId(p);
  const flip = shouldFlip(pid);
  const leftText  = flip ? p.b_text : p.a_text;
  const rightText = flip ? p.a_text : p.b_text;

  $('prompt-text').textContent = p.prompt;
  $('completion-a-text').textContent = leftText;
  $('completion-b-text').textContent = rightText;

  const ann = annotations[pid];
  $('comment').value = ann?.comment || '';
  document.querySelectorAll('.choices button').forEach(b => {
    b.classList.remove('picked-A', 'picked-tie', 'picked-B');
  });
  if (ann?.ui_choice) {
    const btn = document.querySelector(`[data-choice="${ann.ui_choice}"]`);
    if (btn) btn.classList.add(`picked-${ann.ui_choice}`);
  }

  $('progress-text').textContent = `Pair ${currentIdx + 1} of ${PAIRS.length}`;
  $('progress-fill').style.width = `${((currentIdx + 1) / PAIRS.length) * 100}%`;
  $('prev').disabled = currentIdx === 0;
  $('next').disabled = currentIdx === PAIRS.length - 1;

  const done = Object.keys(annotations).length;
  $('annotated-count').textContent = `${done} of ${PAIRS.length} answered`;
  $('submit-btn').disabled = done < PAIRS.length;
  $('submit-summary').textContent = done < PAIRS.length
    ? `Answer all ${PAIRS.length} pairs to submit (${PAIRS.length - done} left).`
    : `All ${PAIRS.length} pairs answered. Ready to submit.`;
}

function recordChoice(choice) {
  const p = PAIRS[currentIdx];
  const pid = pairId(p);
  const flip = shouldFlip(pid);
  let canonical_winner = choice;
  if (choice === 'A') canonical_winner = flip ? 'B' : 'A';
  else if (choice === 'B') canonical_winner = flip ? 'A' : 'B';
  annotations[pid] = {
    pair_id: pid,
    prompt_id: p.prompt_id,
    prompt: p.prompt,
    axis: p.axis,
    a_model: p.a_model,
    b_model: p.b_model,
    display_left_was: flip ? 'B' : 'A',
    ui_choice: choice,
    canonical_winner,
    comment: $('comment').value || '',
    annotated_at: new Date().toISOString(),
  };
  saveAnnotations();
  document.querySelectorAll('.choices button').forEach(b => {
    b.classList.remove('picked-A', 'picked-tie', 'picked-B');
  });
  const pressed = document.querySelector(`[data-choice="${choice}"]`);
  if (pressed) pressed.classList.add(`picked-${choice}`);
  const done = Object.keys(annotations).length;
  $('annotated-count').textContent = `${done} of ${PAIRS.length} answered`;
  $('submit-btn').disabled = done < PAIRS.length;
  $('submit-summary').textContent = done < PAIRS.length
    ? `Answer all ${PAIRS.length} pairs to submit (${PAIRS.length - done} left).`
    : `All ${PAIRS.length} pairs answered. Ready to submit.`;
  // Auto-advance to next unlabeled after a short pause for visual feedback
  if (currentIdx < PAIRS.length - 1) {
    setTimeout(() => { currentIdx++; saveAnnotations(); showPair(); }, 220);
  }
}

function next() {
  const p = PAIRS[currentIdx];
  const pid = pairId(p);
  if (annotations[pid]) {
    annotations[pid].comment = $('comment').value || '';
    saveAnnotations();
  }
  if (currentIdx < PAIRS.length - 1) currentIdx++;
  saveAnnotations();
  showPair();
}
function prev() { if (currentIdx > 0) { currentIdx--; saveAnnotations(); showPair(); } }

function submitAnswers() {
  const out = {
    session,
    submittedAt: new Date().toISOString(),
    n_pairs: PAIRS.length,
    n_annotated: Object.keys(annotations).length,
    annotations: Object.values(annotations),
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `__DOWNLOAD_NAME__`;
  a.click();
  URL.revokeObjectURL(url);
  $('submit-summary').textContent = 'Thank you! Your answers have been saved.';
}

document.querySelectorAll('.choices button').forEach(b => {
  b.addEventListener('click', () => recordChoice(b.dataset.choice));
});
$('next').addEventListener('click', next);
$('prev').addEventListener('click', prev);
$('submit-btn').addEventListener('click', submitAnswers);
document.querySelector('[data-action="dismiss-intro"]').addEventListener('click', () => {
  document.getElementById('intro').classList.add('dismissed');
  localStorage.setItem(INTRO_KEY, '1');
});
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA') return;
  const k = e.key.toLowerCase();
  if (k === 'a') recordChoice('A');
  else if (k === 't') recordChoice('tie');
  else if (k === 'b') recordChoice('B');
  else if (e.key === 'ArrowLeft') prev();
  else if (e.key === 'ArrowRight') next();
});

showPair();
</script>
</body>
</html>
"""


def render_html(pairs, title, header_title, storage_key, download_name):
    pairs_json = json.dumps(pairs).replace("</", "<\\/")
    return (HTML_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__HEADER_TITLE__", header_title)
            .replace("__STORAGE_KEY__", storage_key)
            .replace("__DOWNLOAD_NAME__", download_name)
            .replace("__PAIRS_JSON__", pairs_json))


def main():
    src = json.load(open(EVAL_DIR / "prolific_generations.json"))
    prompts = src["prompts"]
    prompt_ids = src["prompt_ids"]
    generations = src["generations"]

    grid = build_pair_grid(generations, prompts, prompt_ids)
    identical = sum(1 for p in grid if p["a_text"] == p["b_text"])
    print(f"Built pair grid: {len(grid)} instances ({len(ALL_PAIRS)} pairs × {len(prompts)} prompts)")
    print(f"Identical text pairs: {identical}/{len(grid)}")
    for m in MODELS:
        lens = [len(t) for t in generations[m]]
        print(f"  {m:<22}  mean={sum(lens)/len(lens):.0f}  median={sorted(lens)[len(lens)//2]}  max={max(lens)}")

    (EVAL_DIR / "prolific_pair_grid.json").write_text(json.dumps(grid, indent=2))
    print(f"\nFull pair grid → prolific_pair_grid.json ({len(grid)} instances)")

    sanity = sample_sanity_pairs(grid, n_per_axis=5)
    html = render_html(
        sanity,
        title="Phase 2 — Pairwise sanity check",
        header_title="Phase 2 — Pairwise sanity check",
        storage_key="phase2_prolific_sanity_v1",
        download_name="phase2_prolific_sanity_labels.json",
    )
    (EVAL_DIR / "phase2_prolific_sanity_ui.html").write_text(html)
    print(f"Sanity-check UI → phase2_prolific_sanity_ui.html ({len(sanity)} pairs)")


if __name__ == "__main__":
    main()
