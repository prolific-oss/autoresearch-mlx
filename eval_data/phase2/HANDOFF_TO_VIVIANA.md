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

The methodology is in `autoresearch-pytorch/PROLIFIC_EVAL_NOTES.md` — read that for the full design rationale.

## The 5 models in the eval

| model | val_pref_acc | description |
|---|---|---|
| `base` | (untrained) | SmolLM2-360M-Instruct, no DPO — the control |
| `prod-baseline` | 0.464 | full fine-tune at default knobs |
| `prod-adambeta2` | 0.492 | the autoresearch agent's best (adam_beta2=0.95 + NEFTune + constant_with_warmup) |
| `prod-bestbet` | 0.628 | LoRA r=32 + score-margin filter — first big lift |
| `prod-bestbet-v2` | 0.648 | LoRA r=64 + score-margin filter — best |

Total span: **0.184 in val_pref_acc** between worst-trained and best-trained.

## Headline finding from the diagnostic (60 pairs, single annotator)

**val_pref_acc does NOT track human preference cleanly.** When Nora labeled 60 strategic pairs blind:

- Per-axis preference rates were all 42–58% — indistinguishable from chance
- Even the biggest gap (untrained `base` vs best `prod-bestbet-v2`, val 0.0 → 0.648) was 6/6/0 — exactly 50/50
- Cohen's kappa between Nora and Claude annotators: **κ = 0.195** (barely above chance)

This is the methodological story Phase 2 is set up to test at scale: does autoresearch's optimization metric (val_pref_acc) actually track what humans prefer? Diagnostic says no; the Prolific study is the rigorous answer.

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

1. Open `phase2_prolific_sanity_ui.html` in a browser to see the annotator experience.
2. `prolific_pair_grid.json` is the data payload — pipe into your Prolific deployment system.
3. `PROLIFIC_EVAL_NOTES.md` (in `autoresearch-pytorch/`) has the full methodology including locked-in numbers, attention-check design, and analysis pipeline.

## Open questions / things I'd want your input on

- **Sample diversity**: should we add a 2nd-3rd sample per (prompt, model) for variance averaging? Doubles the data but adds robustness. Currently single-sample.
- **Prompt count**: 50 felt like a sweet spot but could be 30 (cheaper) or 100 (more power). Curious what budget supports.
- **Inter-annotator setup**: 3 annotators/pair is the floor for IAA. Worth bumping to 5 if budget allows?
- **Prompt categories**: should we filter further (e.g. drop creative writing, keep only pedagogical)? Or go broader?

Ping me if anything is unclear — happy to walk through the agent trajectory or any specific recipe.
