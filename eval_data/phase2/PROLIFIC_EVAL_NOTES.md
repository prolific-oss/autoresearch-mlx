# Prolific Eval — Methodology Notes

Living document of methodology decisions / pitfalls / TODOs to apply when we move from the diagnostic sanity-check to the real human eval. Add as we go.

## Generation

- **Let models complete naturally — do NOT cap `max_new_tokens` for the real eval.** Use `eos_token_id` stopping (and a generous safety cap like 1024) so participants see full responses, not truncated mid-sentence text. The diagnostic-UI generations used `max_new_tokens=200` which cut some responses mid-thought; that's fine for a quick sanity check but unacceptable for production data.
- **Different seeds per model.** Sample each model's response independently, not with a shared seed. Closer to deployment reality. The diagnostic UI used a shared seed because we wanted to attribute output differences purely to model differences; for Prolific we want what an end-user would actually see.
- **Multiple samples per (prompt, model)** — at least 3, ideally 5. Averages out sampling variance. Each "labeling pair" presented to a Prolific annotator should be one fresh sample from each model.
- **Lock down sampling params** — temperature, top_p, repetition_penalty — and use the same values across all models. Document them. The current diagnostic uses `temperature=0.7, top_p=0.9, repetition_penalty=1.05`.

## Prompts

- **Scale to ~50-100 prompts** (current diagnostic has 12). Span domains: creative writing, advice, pedagogical explanations of common concepts, sensory description, debate / persuasion, light planning.
- **Source from a public benchmark** for legitimacy — AlpacaEval prompts (805 across diverse categories) is the standard for small-LM eval. Optionally augment with MT-Bench prompts. **Filter for general-audience-judgeable** before using.
- **Filter prompts that don't differentiate models** based on diagnostic evidence (e.g. very simple greeting-style prompts where all models say roughly the same thing). The actual eval should use prompts where outputs vary meaningfully.

### Prompt selection: must be judgeable by a general audience

Annotators on Prolific are not domain experts. Prompts must let *anyone* tell which response is better — quality has to come from properties that any literate person can evaluate (clarity, helpfulness, on-topic-ness, naturalness of writing, appropriate tone, common-sense correctness).

**Good prompt types** (general-audience-judgeable):
- Creative / descriptive writing (haiku, opening lines, sensory descriptions)
- Common-sense advice (work overwhelm, friend conflict, basic life decisions)
- Pedagogical clarity for *common* concepts (photosynthesis to a kid, what is gravity, why does ice float)
- Persuasive / opinion writing on universally-relatable topics
- Light planning (meal planning, packing lists, daily schedule)
- Emotional tone / kindness in writing (condolence message, encouragement)

**Avoid** (require specialized knowledge to judge):
- Framework comparisons (React vs Vue, Postgres vs MySQL)
- Programming-language-specific code questions (the diagnostic's "what's wrong with this Python code" assumed Python literacy)
- Advanced technical / scientific concepts where annotators can't tell if the explanation is correct
- Niche cultural references, specific domain jargon
- Medical / legal / financial advice (annotators can't reliably evaluate, also Prolific often restricts these)
- Anything where "the better answer" requires specialized credentials

**Borderline (case-by-case)**:
- Basic math intuition (Pythagorean theorem) — fine if framed for general intuition; not fine if it requires algebra to evaluate
- Simple debugging examples — fine if the bug is logical (off-by-one, missing edge case stated in plain English); not fine if it requires language-specific knowledge

The diagnostic-UI prompts that should be CUT or REPLACED for the real eval:
- "Compare React and Vue from a maintainability standpoint" — requires frontend expertise
- "What's wrong with this Python code: `def avg(lst): return sum(lst) / len(lst)`" — requires Python knowledge to spot the empty-list edge case (ZeroDivisionError)

The rest of the diagnostic-UI prompts mostly survive this filter.

## Models in the eval

- **Include the untuned base model (SmolLM2-360M-Instruct) as a reference point.** Without it, we can only show "model A is preferred over model B"; we can't show that *any* of the trained models actually improve over the starting point. The base→best-trained comparison is the headline finding ("DPO did something") that the rest of the eval rests on.
- **Recipe portfolio for the real eval** (TBD, finalize after diagnostic labels):
  - `base` — SmolLM2-360M-Instruct, no DPO (control)
  - `prod-baseline` — full-FT, val_pref_acc=0.464 (weakest trained)
  - `prod-adambeta2` — agent's best, val=0.492
  - `prod-bestbet` — LoRA + score-margin filter, val=0.628
  - `prod-bestbet-v2` — higher LoRA rank, val=0.648 (best)
  - (Likely drop `prod-neftune` since diagnostic showed it's indistinguishable from prod-baseline.)
- **5 models = 10 pairs**, manageable for a full-grid eval.

## Pairing

- **Full pairwise grid is preferred for Bradley-Terry / Elo ranking.** With N models, that's C(N,2) unique pairs × P prompts × S samples per pair = a lot. Budget accordingly.
- **Alternative: dense subset that hits each model in many comparisons** — good if budget is constrained and we don't need full transitivity validation.
- **Always randomize A/B order** (50/50 left/right) per pair to avoid position bias.
- **Anonymize models in the UI** — annotators must not see which is which.

## Annotators

- **Multiple annotators per pair** (≥3) for inter-annotator agreement. Cohen's kappa per axis tells us whether the differentiation is real or noise.
- **Prolific screening** — restrict to fluent English speakers; for code-related prompts add a software-experience screener (Prolific has these tags).
- **Per-pair compensation budget** that accounts for read-time of full responses (not 200-token previews). At ~250-500 tokens per response, each pair takes ~30-60s to read and judge.
- **Practice/calibration round** — Nora + Claude self-annotate first to set the rubric and spot-check the UI before paying participants.

## Analysis

- **Per-axis preference rate** (fraction where model A wins per pair-type), with confidence intervals.
- **Bradley-Terry model fitting** for full-grid comparisons → produces a global ranking with uncertainty.
- **Correlation with val_pref_acc** — does the autoresearch metric track human preference? This is the headline question of Phase 2.
- **Within-axis variance** — how consistent are annotators? IAA per axis flags axes that are genuinely ambiguous to humans.

## Concrete deployment plan (medium tier — confirmed)

**Locked-in numbers:**
- 5 models: `base`, `prod-baseline`, `prod-adambeta2`, `prod-bestbet`, `prod-bestbet-v2` (drop `prod-neftune` per diagnostic)
- 10 unique model pairs (full pairwise grid)
- 50 prompts (general-audience-judgeable, in `eval_data/phase2/prolific_prompts.json`)
- 3 annotators per pair instance
- **Total annotations: 1500** (50 prompts × 10 pairs × 3 annotators)
- **Sessions: 50** (30 pairs each, ~30 min, $5 compensation)
- **Estimated cost: ~$325** (incl. Prolific platform fees)

**Per-session assignment (counterbalanced):**
- Each Prolific session = 30 randomly-sampled pair instances from the full 1500
- Each pair instance must appear in exactly 3 sessions (different annotators)
- Randomize A/B order per pair instance per session
- Anonymize models (no labels visible to annotator)

**Sample generation (already done correctly per the diagnostic learnings):**
- Different seeds per model — see `eval_data/phase2/prolific_generations.json` (file produced by `gen_prolific.py`)
- EOS-stopping + max_new_tokens=1024 safety cap (no truncation)
- temperature=0.7, top_p=0.9, repetition_penalty=1.05 (locked)
- One sample per (prompt, model) — gives 250 unique generations
- Each pair instance reuses the same A-text and B-text across 3 annotators (standard practice — variance comes from annotator differences, not sample noise)

**IAA targets:**
- Cohen's kappa ≥ 0.4 (moderate) per axis = differentiation is detectable
- κ < 0.2 = models are humanly indistinguishable on this axis
- Diagnostic finding (Nora vs Claude): κ=0.195 across 60 pairs — at the edge of detectability with single-annotator samples. 3-annotator averaging should raise effective signal.

**Attention checks** (build into each session):
- 1 identical-text pair (must label "tie" — failure = bot or distracted)
- 1 obvious-quality-gap pair (e.g. one well-formed response vs one of broken/truncated text — must label correctly)
- Reject sessions failing both checks

**Analysis pipeline** (after data collection):
1. Per-axis preference rate (with 95% CI from binomial)
2. Bradley-Terry model fitting → global ranking
3. Spearman correlation: BT ranking vs val_pref_acc ranking → headline number
4. Per-prompt-category breakdown (does the gap appear more in creative writing than in pedagogy?)
5. IAA per pair instance → flag pair instances with κ < 0.2 as ambiguous

## Don'ts

- ❌ Don't truncate generations (the thing that prompted this note).
- ❌ Don't use shared seeds across models in the production eval.
- ❌ Don't show the same model pair twice to the same annotator (potential consistency bias).
- ❌ Don't use prompts that an LLM judge rates 10/10 in seconds for everything — those don't differentiate.
- ❌ Don't reveal model identities or val_pref_acc to annotators (preference contamination).

## Tracking what we already know

From the diagnostic on the ~48-pair sanity check:
- prod-baseline (val=0.464) and prod-neftune (val=0.474) produce identical outputs ~10% of the time on the same seed — likely indistinguishable for human eval. *Implication: probably drop one of them from the production eval portfolio.*
- Diagnostic axes that gave humanly-perceptible differences: TBD after Nora finishes labeling.
