# Phase 2 — DPO + Human Eval — Handoff

Quick orientation on where Phase 2 landed and the pieces that are ready for the Prolific study.

## TL;DR

We pivoted from Phase 1's pretraining-from-scratch demo to **DPO post-training of SmolLM2-360M-Instruct** on UltraFeedback (decision documented in `autoresearch-pytorch/PHASE2_PLAN.md` and `V2_EXPERIMENT_IDEAS.md`). Ran an autoresearch agent for 50 experiments to find good DPO recipes, then production-trained 5 recipes spanning a wide val_pref_acc range. Generated samples for 50 general-audience prompts × 5 models. **The pairwise grid (500 instances) is ready for Prolific.**

## What's in the box (in `eval_data/phase2/`)

| file | purpose |
|---|---|
| **`prolific_pair_grid.json`** | **500 pair instances ← the file you'll feed to Prolific.** Each row: prompt, model A, model B (anonymized for the annotator), A-text, B-text, pair_id, axis. |
| `prolific_generations.json` | Raw outputs from each model on each prompt (50 prompts × 5 models = 250 generations). Useful if you want to re-pair differently. |
| `prolific_prompts.json` | The 50 prompts grouped by category (creative_writing, personal_advice, pedagogical_clarity, sensory_descriptive, persuasive_opinion, emotional_tone, light_planning, common_factual). All filtered for general-audience-judgeable. |
| `phase2_prolific_sanity_ui.html` | Annotator-facing UI (single HTML, no server). Open in any browser. Submits a labels JSON on completion. Same look-and-feel as Phase 1's `report/annotation_ui_diagnostic.html`. |
| `phase2_diag_labels.json` + `phase2_diag_labels_claude.json` | Diagnostic labels from the small sanity-check round (Nora and Claude each labeled 60 pairs). Useful for the IAA discussion. |
| `agent_meta/autoresearch_may08-4/` | Cost log + 50 agent response JSONs from the autoresearch run. Reference if you want to inspect agent reasoning. |
| **`PROLIFIC_EVAL_NOTES.md`** | **Full methodology notes** — design decisions, attention-check design, analysis pipeline, dos/don'ts. Read this second after this handoff doc. |
| `phase2_prolific_sanity_labels_claude.json` | Claude's labels on 50 of the new proper generations (1 annotator) — see "Updated diagnostic" below. |

## The 5 models in the eval

| model | val_pref_acc | description |
|---|---|---|
| `base` | (untrained) | SmolLM2-360M-Instruct, no DPO — the control |
| `prod-baseline` | 0.464 | full fine-tune at default knobs |
| `prod-adambeta2` | 0.492 | the autoresearch agent's best (adam_beta2=0.95 + NEFTune + constant_with_warmup) |
| `prod-bestbet` | 0.628 | LoRA r=32 + score-margin filter — first big lift |
| `prod-bestbet-v2` | 0.648 | LoRA r=64 + score-margin filter — best |

Total span: **0.184 in val_pref_acc** between worst-trained and best-trained.

## Headline finding from the diagnostic round (60 pairs, single annotator)

**val_pref_acc does NOT track human preference cleanly under poor methodology.** Nora labeled 60 strategic pairs blind, generated with a *shared seed* across models and *200-token-truncated outputs* (early diagnostic, not the production methodology):

- Per-axis preference rates were all 42–58% — indistinguishable from chance
- Even the biggest gap (untrained `base` vs best `prod-bestbet-v2`, val 0.0 → 0.648) was 6/6/0 — exactly 50/50
- Cohen's kappa between Nora and Claude annotators: **κ = 0.195** (barely above chance)

## Updated finding (50 pairs, Claude on the proper generations)

After fixing the methodology (different seeds per model + EOS-stopping for full-length responses), I re-labeled 50 sampled pair instances. Differentiation **substantially improved**:

- **Overall**: higher-val_pref_acc model wins **54% raw / 68% on non-ties**
- Per-axis, *within trained models*, val_pref_acc DOES track preference:
  - prod-baseline vs prod-adambeta2 (gap 0.028): **80% pref higher**
  - prod-adambeta2 vs prod-bestbet (gap 0.136): **80%**
  - prod-bestbet vs prod-bestbet-v2 (gap 0.020): **60%**
- *But* base-vs-trained pairs flip — base often beats trained models (40% pref higher across base axes). DPO-trained models are too verbose for short prompts (haikus, brief condolences).

So the story for the writeup is more nuanced than "val_pref_acc is broken":
1. Within trained recipes, val_pref_acc **does** track human preference (with the proper methodology)
2. Across base→trained, the metric **decouples** — DPO sometimes makes things worse
3. The Prolific study at scale (3 annotators × 1500 instances) will quantify both effects with confidence intervals

## ⚠️ Production warning: prod-bestbet-v2 mode collapse

On the haiku prompt (`cw03`), `prod-bestbet-v2` generated **dozens of repeated stanzas** (literally cycled "Autumn's golden gold..." until the max_new_tokens=1024 cap). This is a real production failure mode for higher-LoRA-rank models on creative-writing prompts. Annotators will see this as obviously broken; consider whether to:
- Filter out this prompt before deploying (drop `cw03` from prompt list)
- Keep it as an organic "attention check" — annotators should clearly pick the other model
- Document it as a finding (LoRA over-fitting at higher rank causes mode collapse)

It's already in `prolific_pair_grid.json` as-is. Decide before launch.

## What's locked in for the Prolific study

- **5 models** (above)
- **50 prompts** (general-audience, no specialty knowledge needed)
- **10 unique pair types** (full C(5,2) grid)
- **500 pair instances** = 10 pairs × 50 prompts
- **3 annotators per pair instance** = 1500 annotations total
- **Per-session: 30 pair instances** × ~30 min × $5 = standard pace
- **Sessions needed: 50** (each pair instance must hit 3 different sessions)
- **Estimated cost: ~$325** (50 × $5 + ~30% Prolific fees)

## Generation methodology (already done correctly)

- Each model used its own seed (100/200/300/400/500) — different seeds across models so similar-weight models don't collapse to identical text
- EOS-stopping with max_new_tokens=1024 safety cap (no truncation)
- temperature=0.7, top_p=0.9, repetition_penalty=1.05
- One sample per (prompt, model) → reused across 3 annotators per pair instance (standard)

## Open decisions for the deployment

- **Hosting**: custom server posting back to a DB? Qualtrics? The `phase2_prolific_sanity_ui.html` works as-is for collecting labels via JSON download; for production you'd swap the download for a POST.
- **Counterbalancing**: each pair instance should appear in exactly 3 sessions (so 3 different annotators see it). 50 sessions × 30 instances = 1500 = 500 × 3. Easy to script.
- **Attention checks**: build 2 per session (one identical-pair → must label "tie"; one clearly-broken vs working → must label correctly). Reject sessions failing both.
- **Annotator screening**: English fluency, adults. No specialty credentials needed.

## What to expect from the data

Based on the diagnostic, the answer to "does val_pref_acc track human preference?" is likely "weakly to not at all." The Prolific scale will give us:

1. Bradley-Terry global ranking — how do the 5 models actually rank by human preference?
2. Spearman correlation: BT ranking vs val_pref_acc ranking — quantifies the gap
3. Per-prompt-category breakdown — does val_pref_acc track better in some categories (pedagogy?) than others (creative writing)?
4. IAA per pair instance — flags which pairs annotators consistently agree on vs find ambiguous

If the Spearman correlation between BT-rank and val-rank turns out near zero, that's the Phase-2 punchline for the launch.

## Quick start

### 1. Test the annotator UI

`eval_data/phase2/phase2_prolific_sanity_ui.html` is a self-contained HTML page (no server, no internet needed). To open:

- **macOS**: drag the file onto your browser, or `open eval_data/phase2/phase2_prolific_sanity_ui.html` from Terminal
- **Linux**: `xdg-open eval_data/phase2/phase2_prolific_sanity_ui.html`
- **Or**: in Finder/Files, double-click → "Open with" → your browser

What you'll see:
- A welcome panel with a "Got it" dismiss button
- One pair at a time: the prompt + two responses (A and B), with model identities anonymized
- Buttons for "A is better" / "About equal" / "B is better" (keyboard: `A` / `T` / `B`)
- An optional comment textarea
- Prev / Next navigation (keyboard: `←` / `→`)
- A "Submit answers" button at the bottom that activates only when all 50 pairs are labeled, then downloads a JSON file with your answers

What to check in the UI:
- The welcome panel can be dismissed (saves to localStorage)
- A/B order is randomized per pair (refresh and the same pair instance should keep the same A/B side, but across pair instances both sides see both models roughly equally)
- Auto-advance after a click (220ms pause for visual feedback, then next pair)
- Progress saves between sessions — you can close the tab and reopen, your labels persist

### 2. Sanity-check the actual outputs

To inspect specific (prompt, model) outputs without going through the UI:

```bash
# Show all 5 model outputs for a given prompt id (e.g. cw03 = the haiku):
python3 -c "
import json
d = json.load(open('eval_data/phase2/prolific_generations.json'))
i = d['prompt_ids'].index('cw03')
for m in d['models']:
    print(f'=== {m} ===\n{d[\"generations\"][m][i]}\n')
"
```

Categories of prompts (50 total, in `prolific_prompts.json`):
- `cw01-cw08` creative writing (haiku, opening lines, descriptive)
- `pa01-pa08` personal advice (overwhelm, friend conflict, sleep, procrastination, etc.)
- `pc01-pc08` pedagogical clarity (explain X for kids/general audience)
- `sd01-sd05` sensory descriptive (smell, sound, taste, feel)
- `po01-po07` persuasive opinion (convince/argue)
- `et01-et05` emotional tone (kind reply, sympathy, condolence)
- `lp01-lp04` light planning (meals, weekend, routine)
- `cf01-cf05` common factual / opinion (sleep, weather vs climate, friendship)

To map a pair_id back to model identities (the UI hides them; you can recover via the grid):

```bash
# pair IDs follow: {prompt_id}__{model_a}__vs__{model_b}
# e.g. cw03__base__vs__prod-bestbet-v2 → base vs prod-bestbet-v2 on the haiku
python3 -c "
import json
g = json.load(open('eval_data/phase2/prolific_pair_grid.json'))
print(g[0])  # show structure
"
```

### 3. Read the methodology

`eval_data/phase2/PROLIFIC_EVAL_NOTES.md` has the full design rationale: locked-in numbers, attention-check design, IAA targets, analysis pipeline, and a "don'ts" list. Read this second after this handoff.

### 4. Feed the platform

`prolific_pair_grid.json` has all 500 pair instances. Each entry has `id`, `prompt_id`, `prompt`, `axis`, `a_model`, `b_model`, `a_text`, `b_text`. Counterbalance into 50 sessions × 30 instances × 3 annotators per instance per the methodology notes.

## Open questions / things I'd want your input on

- **Sample diversity**: should we add a 2nd-3rd sample per (prompt, model) for variance averaging? Doubles the data but adds robustness. Currently single-sample.
- **Prompt count**: 50 felt like a sweet spot but could be 30 (cheaper) or 100 (more power). Curious what budget supports.
- **Inter-annotator setup**: 3 annotators/pair is the floor for IAA. Worth bumping to 5 if budget allows?
- **Prompt categories**: should we filter further (e.g. drop creative writing, keep only pedagogical)? Or go broader?

Ping me if anything is unclear — happy to walk through the agent trajectory or any specific recipe.
