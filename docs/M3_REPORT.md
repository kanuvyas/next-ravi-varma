# Milestone 3 Report: Text Parsing and Semantic Scene Generation

**Team:** A (pipeline, SHAP), B (evaluation), C (KB, parser, name-aware LoRA), D (expander, report assembly). **Assembled by:** D.

> **Rules for this file**
> - **Every number comes from a file in `results/` or from B's tables in `eval/EVAL_REPORT.md`** (ratings sheet included), and each one names its source next to it. "Not tested" is an acceptable entry. No number is typed in from memory or from a chat.
> - **B audits this report line by line on Day 6** (claims audit). Anything not backed by a file is softened or removed.
> - **Timing risk (Day 5):** contributors' sections are due **Day 5 noon**, but B's held-out run and held-out ratings finish only on **Day 5 evening**. The report is therefore assembled in two passes: (1) by Day 5 evening, with dev results and every held-out cell marked `TODO(B, held-out)`; (2) held-out tables added when B posts them, then the text is frozen before B's Day-6 audit. If held-out numbers are late, the report ships with the cells marked "pending" rather than with estimates.
> - Markers: `TODO(A|B|C|D)` = owner must supply; `TODO(results)` = needs a measured number.

---

## 1. Goal and system

**Inputs:** Overview v2 §1-§3 (D). A's `pipeline/PIPELINE.md` for the run instructions and timings. **Feeds from `results/`:** none.

**Goal.** A user types an epic scene in plain words, for example "Yudhishthira weeping over the fallen body of Karna on the battlefield of Kurukshetra". The system produces a painting in Raja Ravi Varma's style that shows that scene. Milestone 2 gave us a LoRA on Stable Diffusion 1.5 (`rrv_v1`, the M2 winner `D_lr2e-4` at step 800) that knows Varma's *style* but not the *stories*. Its training captions had no names and its text encoder was frozen, so "Karna" means nothing to it (`hub/rrvarma_lora_v1/NAMES_TEST.md`). Milestone 3 adds the text side: parse the scene, turn names into visual descriptions, and fit them into a prompt the model can use.

```
user text ──► [C] parse_scene ──► SceneSpec ──► [D] expand ──► ExpandedPrompt ──► [A] generate ──► image
                 (KB lookup,                     (local LLM or template,              (M2 model + LoRA)
                  no LLM)                         token-fitted, names scrubbed)
                                                                     [A] SHAP attribution of prompt parts
                                [B] test set, checklist judge, blind ratings, stats, evaluate the whole chain
                                [C] optional: name-aware LoRA (model-side fix)
```

- **Parser and KB (C):** finds characters (including misspellings and aliases), setting and mood, and attaches name-free visual descriptions grounded in Varma's paintings (`kb/KB.md`).
- **Expander (D):** writes name-free scene phrases with a local instruct LLM (temperature 0, cached), or a deterministic template. The code puts the trigger first and the style tail last, fits the prompt into 70 CLIP tokens and scrubs names (`expander/EXPANDER.md`).
- **Generator (A):** SD1.5 + `rrv_v1`, DPM++ 30 steps, CFG 7.0, 512×512, settings from `config/m3.json` (`pipeline/PIPELINE.md`). One call: `text_to_image(text, seed)`.
- **Constraint:** everything runs on a free Colab T4. The LLM, the diffusion model and the judge run in separate sessions and communicate through files.

`TODO(A)`: one sentence on the measured seconds per image and the load time, from `PIPELINE.md`. `TODO(D)`: an example input → expansion → image triple, chosen after results (seed 1, dev scene).

## 2. Research question and what is new since the halfway version

**Inputs:** Overview v2 §1, §9 (D). **Feeds from `results/`:** none.

**Research question.** The M2 model knows the style but not the stories. **Which part of the system actually makes the picture match the story: the knowledge base, the LLM, the prompt structure, or the model itself? And at what cost to style?**

**What is new.** The halfway feedback asked for "the simplest thing possible". We kept the pipeline simple (no UI, one base model, one LoRA, local models only) and put the effort into answering the question with evidence:
1. a controlled **ablation ladder** c0-c6 (§3) that adds one component at a time;
2. a **Shapley (SHAP) attribution** of the prompt's parts, which measures which phrases buy faithfulness and which buy style (A);
3. a **model-side fix**: a LoRA retrained with names in the captions, gated by a coverage audit and compared with the prompt-side fix (C);
4. an evaluation we can defend: **blind human ratings** plus a **checklist judge validated against those ratings**, with **hypotheses pre-registered** in `experiments/HYPOTHESES.md` before any scored run (B).

`TODO(C)`: one line on the gate outcome (GO/NO-GO) from `train/NAMES_MODEL.md`.

## 3. Method

**Inputs:** B's `eval/EVAL_REPORT.md` §1 (setup); `experiments/HYPOTHESES.md` (all sections); A's `pipeline/TOKEN_BUDGET.md`; C's `kb/KB.md` (KB schema, coverage thresholds). **Feeds from `results/`:** none (method only).

**Test sets.** The dev set (`eval/test_scenes_dev.json`) has 32 scenes, 8 written by each member, with at least one per tier:
1. one well-known character;
2. two characters interacting;
3. multi-figure;
4. rare or absent-in-training character, or a spelling variant;
5. stress: vague, non-epic, 100+ words, misspellings.

The team-wide targets were at least 10 Ramayana and at least 10 Mahabharata scenes. Each scene has 3-6 name-free, style-free yes/no checklist questions, written before any image was generated and reviewed by a second member. B also wrote a held-out set of 12 scenes privately and revealed it only after the S3 freeze. `seen_in_training` marks scenes whose episode appears in M1's `scene_metadata`. `TODO(B)`: final tier, epic and seen/unseen counts.

**Conditions** (all on `rrv_v1` unless marked `@rrv_names`):

| ID | What the model sees | Owner |
|---|---|---|
| `c0_base_raw` | base SD1.5, raw text, no LoRA | A |
| `c1_lora_raw` | LoRA, trigger + raw text + style tail | A |
| `c2_template` | LoRA, deterministic template from the SceneSpec | D |
| `c3_llm_kb` | LoRA, LLM expansion with KB descriptions (the main system) | D |
| `c4_llm_nokb` | LoRA, LLM expansion without KB | D |
| `c5_llm_kb_layout` (optional) | c3 + figure count / position / pose phrases | D |
| `c6_llm_kbllm` | c3 with LLM-written instead of image-grounded KB descriptions | C |
| `@rrv_names` (gated) | the same prompts on the name-aware LoRA | C |

`TODO(D)`: add the rows for the optional D conditions that actually ran (`c3_7b`, `c3_neg`, `c3_v2`) and the final `prompt_version`, `kb_version`, LLM checkpoint and config values (`style_tail`, `negative`, `max_prompt_tokens`) from the frozen `config/m3.json`.

**Metrics.**
- *Primary:* blind human ratings, 1-5, on scene faithfulness, style fidelity and visual quality. At least 2 raters per image; contact sheets carry random codes only. Raters also tag failures from a fixed taxonomy. Dev ratings use seed 1.
- *Supporting:* checklist judge (vision-language model, P(yes) per question); `clip_raw` (CLIP against the raw user text); `style_margin`; `max_train_dino` (memorisation guardrail).
- *Process:* expansion validity, name leaks and latency.

`TODO(B)`: judge checkpoint and the validation threshold.

**Statistics.** Comparisons are paired per scene. We report win/tie/loss counts and a paired bootstrap 95% CI over scenes, with no significance claims; a CI that includes 0 means "no clear difference". Tie bands are 0.25 for ratings and M2's 0.0295 for automated scores.

**Choosing the default pipeline.** The default for M4 is chosen at S3 **on dev only**, by the rule fixed in Overview §10: highest mean faithfulness, subject to the style and memorisation guardrails; on a CI including 0, the simpler condition wins. `TODO(A)`: the decision as recorded in `docs/TEAM_DECISIONS_M3.md`.

**Hypotheses.** Pre-registered per owner in `experiments/HYPOTHESES.md` (sections A-D). Each section's merge timestamp must precede that owner's first scored run; `TODO(B)`: confirm this from git history during the Day-6 audit. `TODO(D)`: list each hypothesis ID with its one-line statement, and link the Deviations lines.

## 4. Results

**Inputs:**
- B: `eval/EVAL_REPORT.md` §2-§4 (tables, paired comparisons, judge validation, grids);
- A: `analysis/shap/SHAP_ANALYSIS.md` (chart, interactions, value per token), `pipeline/TOKEN_BUDGET.md`;
- C: `train/NAMES_MODEL.md` (2×2) and `kb/KB.md` (c6, parser precision/recall);
- D: `expander/EXPANDER.md` (process metrics, optional experiments).

**Feeds from `results/`:** `results/<condition>@<model>_dev.json` and `results/<condition>@<model>_heldout.json` for c0-c6, the optional D conditions and `@rrv_names`; A's SHAP results; `outputs/expansions/*.json` via `check_expansions.py`.

### 4.1 Ablation, dev
`TODO(B)`: condition × (faithfulness, style, quality) means; condition × (checklist, `clip_raw`, `style_margin`, `max_train_dino` p95); paired comparisons c0→c1, c1→c3, c4→c3, c2→c3, c6→c3, c3→c5 with diff, CI and win/tie/loss.

### 4.2 Ablation, held-out
`TODO(B, held-out)`: the same tables on 12 scenes × 4 seeds. Arrives Day 5 evening (see timing risk above).

### 4.3 Expansion ladder and expander hypotheses
`TODO(D)`: H-D1 to H-D6 outcomes with the pre-registered rule (supported / no clear difference / contradicted / not run), plus the process table (% valid, leaks before and after repair, fallback rate, latency) from `EXPANDER.md`.

### 4.4 SHAP attribution
`TODO(A)`: the one chart (players × {faithfulness, style}, CI whiskers), the interaction table, and H-A outcomes, from `SHAP_ANALYSIS.md`.

### 4.5 Model-side fix and KB grounding
`TODO(C)`: the GO/NO-GO result; if GO, the 2×2 (`c1`/`c3` × `rrv_v1`/`rrv_names`) and the answer to "did it learn names or memorise paintings?"; if NO-GO, the descriptor-length result. Also the c6 vs c3 comparison and parser precision/recall.

### 4.6 Judge validation
`TODO(B)`: Spearman correlation between judge and human faithfulness, and between CLIP and human faithfulness; inter-rater agreement; the question-order check; whether the pre-registered threshold was met.

## 5. Analysis

**Inputs:**
- B: `EVAL_REPORT.md` (failure taxonomy counts, seen vs unseen, coverage, cost);
- A: `SHAP_ANALYSIS.md` §3-§4;
- C: `KB.md` (coverage table, known gaps);
- D: `EXPANDER.md` §11 (failure modes).

**Feeds from `results/`:** per-tier and seen/unseen splits of the dev and held-out results; `latency_s` from the expansion files; seconds per image from the manifests.

- **What helped:** `TODO(D)`, written from §4 only. If c1 or c2 is as good as c3, that is said first (B's rule).
- **What did not help:** `TODO(D)`.
- **Failure taxonomy:** `TODO(B)`: counts per condition, with 2-3 example images (codes decoded).
- **Coverage, seen vs unseen:** `TODO(B)`: do scenes with `low`/`none`-coverage characters, or unseen scenes, score lower?
- **Cost:** `TODO(A, D)`: LLM seconds per scene, seconds per image, GPU sessions needed for one scene end to end.

## 6. Limitations

**Inputs:** all four owners' "limits" sections. **Feeds from `results/`:** sample sizes and the judge-validation numbers, cited where used.

These are known now and hold whatever the results show; owners add their own.
- **Small n.** 32 dev scenes and 12 held-out scenes; tier subsets have about 4-8 scenes each and A's SHAP uses 8. Most CIs will be wide. "No clear difference" is not evidence of no effect.
- **Proxy metrics.** The checklist judge, `clip_raw` and `style_margin` are proxies. CLIP shares an encoder family with the generator's text encoder. The judge is trusted only as far as B's validation shows.
- **Raters knew the project.** All raters are team members who wrote the scenes and built the conditions. Blind codes hide the condition, but not the intent of the project.
- **77 tokens.** CLIP reads at most 77 tokens. Expansions are fitted to 70 (excluding start/end), so long scenes lose detail. The raw text in c0/c1 and in `clip_raw` is truncated, which matters most for tier-5 texts with 100+ words.
- **One model.** One base model (SD1.5), one LoRA (`rrv_v1`, trained with one seed), one sampler setting and one small local LLM family. Nothing here is shown to transfer to other models.
- **Style tail.** The model was **not trained with the style tail** we append: the winner's train_v1 captions have no fixed tail. `TODO(A)`: what the token-budget and SHAP results say about it.
- **Name scrub.** It removes every listed name, including rivers and mounts that double as setting words (Ganga, Nandi), so a setting phrase can be lost. `TODO(D)`: measured frequency.
- **Self-made benchmark.** Checklists were written by the team, and the dev set was also used to tune prompts and the KB (allowed by our protocol). Only the held-out set is untouched.
- `TODO(A, B, C)`: owner-specific limits (SHAP position effects; the judge's failure cases; KB description sources and gaps; name-aware LoRA memorisation).

## 7. Handoff to M4

**Inputs:** A's `docs/HANDOFF_M4.md`; B's failure taxonomy; D's `layout` field (if c5 was built). **Feeds from `results/`:** failure tag counts by type.

- **Interface for M4:** `text_to_image(text, seed, mode)` returns `(image, ExpandedPrompt, SceneSpec)`, and `generate(..., control_image=None)` already has a slot for ControlNet. SD1.5 is compatible with ControlNet 1.x (OpenPose, Canny). Everything is 512×512 only.
- **Composition failures:** `TODO(B, D)`: which failures are pose or composition problems (wrong figure count, wrong action), i.e. ControlNet territory, and which are identity or setting problems that the KB and prompt can still fix.
- **`layout`:** `TODO(D)`: if c5 ran, the `layout` JSON (`n_figures`, per-figure role, pose, position) is the bridge to an OpenPose skeleton in M4; whether it helped (H-D3).
- **Default pipeline and known failure modes:** `TODO(A)`: from `HANDOFF_M4.md` and `TEAM_DECISIONS_M3.md`.
- **Cold-start test (Day 6, D):** `TODO(D)`: pass or fail from a fresh runtime using only the README (demo on 3 inputs, CLI once, one image reproduced with the same seed), and the issues filed as `needs:A`.
