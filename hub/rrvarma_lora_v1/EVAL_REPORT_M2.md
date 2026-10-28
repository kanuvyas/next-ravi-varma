# Evaluation Report: Milestone 2 (rrvarma LoRA v1)

Author: Member B. All numbers come from `compare.py` (dev set), `bundles/B/final_test/results.json` (test set) and C's `INFERENCE.md` (scale sweep). Nothing here is from memory.

**Headline.** On 15 held-out test prompts, the chosen LoRA (D_lr2e-4, checkpoint-800, scale 0.8) improved `style_margin` by **+0.037 (95% CI 0.025 to 0.049)** over the strongest plain Stable Diffusion baseline, winning on 14 of 15 prompts, with no loss in text alignment. Images clearly carry Ravi Varma's palette, lighting and drapery, but **faces and hands are distorted in essentially every image** we reviewed.

---

## 1. Setup

**Base and sampling.** Stable Diffusion v1.5, 512x512, DPM++ 30 steps, CFG 7.0, fixed negative prompt, safety checker disabled (it falsely blocks classical paintings). Trigger phrase `rrvarma style` placed first.

**Prompt sets.**
- *Dev* (8 prompts, trigger first): used to choose the checkpoint.
- *Controls* (7 prompts, no trigger, e.g. "a photo of a cat sitting on a sofa", "an oil painting of a European noblewoman in a velvet dress"): used to check the LoRA did not damage normal generation.
- *Test* (15 prompts): BLIP-2 captions of the 15 held-out paintings, hand-corrected, no character names. Used **once**, on the chosen checkpoint only, so the winner was not picked and reported on the same prompts.

**Seeds.** Screening: dev seeds 1-2, controls seed 1. Deep scoring and test: seeds 1-4.

**Reference sets.**
- Holdout: 15 real Varma paintings never used in training (the "Varma" reference).
- Control art: non-Varma, public-domain figure paintings (C's set, see `eval/control_art/control_art_log.csv`; 17 images).
- Training images: used only to detect copying.

**Metrics** (all are proxies; we also looked at the images):

| Metric | What it measures |
|---|---|
| `style_margin` | CLIP similarity to the holdout Varma centroid minus similarity to the control-art centroid. Higher = more Varma-like. |
| `clip_text` | CLIP similarity between image and prompt (prompt-following). |
| `max_train_dino_p95` | 95th percentile of each image's DINOv2 similarity to its closest training image. High = possible copying. Flag at 0.85. |
| `style_bleed`, `drift` | On no-trigger control prompts: style change and overall image change vs the base model (forgetting). |
| `holdout_loss` | Denoising loss on held-out paintings (overfitting signal independent of CLIP). |

**Ceiling.** Real held-out Varma paintings score `style_margin` = **0.123** against the others. It is clearly above 0, so the metric does separate Varma from the control art.

**Limitations of the metrics.** CLIP ViT-L/14 is the same model family as SD1.5's text encoder, so `clip_text` partly scores the generator with its own encoder. With 15 holdout images, all confidence intervals are wide.

**Baselines** (plain SD1.5, no LoRA, same machine, same seeds):
- (a) prompt with trigger words only (the words mean nothing to the base model)
- (b) prompt + "in the style of Raja Ravi Varma"
- (d) prompt + engineered style words ("oil painting, 19th century Indian academic realism, soft skin rendering, chiaroscuro, rich silk drapery, warm palette")

## 2. Selection on the dev set

Four runs, each changing one thing from A (single training seed each):

| Run | Change vs A | Best deep checkpoint | style_margin | gain vs (d) | clip_text | dino_p95 |
|---|---|---|---|---|---|---|
| A_base | baseline (rank 16, LR 1e-4, train_v1) | 1400 | 0.1189 | 0.1019 | 0.2987 | 0.834 |
| B_v2caps | train_v2 captions (constant style words) | 1600 | 0.0985 | 0.0815 | 0.3017 | 0.821 |
| C_r32 | rank 32 | 1200 | 0.1219 | 0.1050 | 0.2898 | 0.835 |
| D_lr2e-4 | learning rate 2e-4 | 1600 | 0.1325 | 0.1155 | 0.2920 | 0.841 |

**Rule (written before comparison).** Thresholds were derived from the baseline only: `tie_band` = 0.0295 (half-width of the 95% bootstrap CI of baseline (d) over dev prompts), `max_style_bleed` = 0.059, `max_holdout_loss_rise` = 0.01, `mem_flag_dino_p95` = 0.85. They were filled after training finished but before any results were compared (see `TEAM_DECISIONS_M2.md`).

**Outcome.** All 11 deep-scored checkpoints were eligible. The top score was D_lr2e-4 step 1600 (gain 0.1155), but 9 checkpoints fell inside the tie band. The tie rule (fewest steps, then lower rank) selected **D_lr2e-4 checkpoint-800** (gain 0.1058, dino_p95 0.846). Its memorization sheet was checked by eye: no near-copies. D800 also follows prompts better than D's later checkpoints (`clip_text` 0.300 vs about 0.29 at steps 1400-1600).

**Not done:** no selection meeting, no human blind test, no combo run.

## 3. Ablation findings (one training seed each, so indicative only)

- **Captions (B vs A):** the train_v2 captions, which repeat the same style words in every caption, gave clearly lower style (best gain 0.082 vs 0.102). This is consistent with the style words "absorbing" style that the trigger would otherwise learn. *Caveat:* the suffix-only diagnostic in `eval_run.py` used the wrong suffix words ("oil painting, dramatic lighting, classical Indian attire" instead of the actual "oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry"), so it does not test this mechanism directly and is not reported.
- **Rank (C vs A):** rank 32 is within the tie band of rank 16 (0.105 vs 0.102) at twice the file size. No meaningful benefit measured.
- **Learning rate (D vs A):** LR 2e-4 reached A's style level earlier (D at step 800: 0.106; A at step 800 screening: 0.094) and scored highest at 1600, but the difference from A's best (0.0136) is within the tie band. Prompt-following declined in D's later checkpoints. *Caveat:* D's run also used bf16 mixed precision while A, B and C used fp16, so this comparison changes two things and is confounded with precision.

## 4. Scale sweep (from C, on D800)

3 prompts x 2 seeds per scale (`sweep_winner/grid.png`):

| scale | style_margin | clip_text |
|---|---|---|
| 0.0 | 0.0230 | 0.2876 |
| 0.4 | 0.0687 | 0.3019 |
| 0.6 | 0.0814 | 0.3114 |
| 0.8 | 0.1081 | 0.2976 |
| 1.0 | 0.1175 | 0.2996 |
| 1.2 | 0.1186 | 0.2967 |

Recommended scale **0.8**: about 90% of the style gain at 1.0, with prompt-following roughly unchanged. Small sample; differences of a few thousandths are noise.

## 5. Final test-set results (used once)

D800 at scale 0.8 vs plain SD on the 15 test prompts, seeds 1-4:

| | style_margin | clip_text |
|---|---|---|
| **D800 @ 0.8** | **0.1071** | **0.3033** |
| Baseline (a) trigger words only | 0.0703 | 0.2948 |
| Baseline (b) "in the style of Raja Ravi Varma" | 0.0632 | 0.2995 |
| Baseline (d) engineered style words | 0.0571 | 0.2937 |

Paired per-prompt gains, 95% bootstrap CI over prompts (2000 resamples):

| Compared with | Prompts | Gain | 95% CI | LoRA wins |
|---|---|---|---|---|
| Baseline (a) | all 15 | +0.0368 | [0.0250, 0.0487] | 14/15 |
| Baseline (a) | without Kadambari | +0.0366 | [0.0241, 0.0487] | 13/14 |
| Baseline (d) | all 15 | +0.0499 | [0.0381, 0.0631] | 15/15 |
| Baseline (d) | without Kadambari | +0.0494 | [0.0370, 0.0632] | 14/14 |

Test `dino_p95` = 0.827 (below the 0.85 flag).

**How to read this honestly:**
- All intervals are above zero, so the style improvement on held-out paintings is not prompt luck.
- The conservative figure is the gain over **(a)**, the strongest baseline here. The gain over (d) is inflated: on all 15 test prompts, (d)'s added style words pushed the prompt past CLIP's 77-token limit and were partly cut off, which is why (d) scored below (a).
- The test captions also end with style words ("oil painting, classical Indian academic realism, ..."), contrary to the plan, which raises every baseline. This is the main reason the test gain (about 0.04-0.05) is smaller than the dev gain (about 0.11).
- Two holdout paintings (`shankar_milli_c11`, `shankar_v2_c02`) have identical captions, so there are 14 distinct test prompts.

## 6. Forgetting and memorization

- **Style bleed** on no-trigger prompts stayed small for every checkpoint (maximum 0.022 vs the 0.059 limit). The LoRA does not push Varma style into unrelated prompts.
- **Drift** (overall change on control prompts) grows with training, from about 0.07-0.11 at step 200 to about 0.15-0.17 at the latest checkpoints. Earlier checkpoints disturb the base model less.
- **Memorization:** every checkpoint's `dino_p95` was below 0.85 (range 0.786-0.847). The chosen checkpoint's memorization sheet showed no near-copies.
- **Holdout loss was uninformative:** every checkpoint scored 0.2005-0.2009, so it neither confirms nor rules out overfitting. (The base-model value, 0.201, is also higher than the roughly 0.1 we expected.)

## 7. Data issue: near-duplicate composition in the holdout

Holdout `varma_kadambari_sitar_b04` and training image `varma_kadambari_sitar_a51` are not the same scan but show the same composition (same figure, same pose). Varma painted repeated versions, and our plan counts this as a leak. The datasets were already frozen, so it was not removed; instead, test results are reported with and without it (section 5). Excluding it changes the gain by less than 0.001.

## 8. Failure modes (by eye, test images at scale 0.8)

- **Faces:** distorted in essentially every image reviewed (asymmetric or misaligned eyes, odd features). We did not find a fully correct face. This is the biggest weakness.
- **Hands:** generally distorted (malformed fingers and shapes).
- **Multi-figure scenes:** figures generally do *not* merge into each other.
- **Overall style:** strong. Warm palette, chiaroscuro lighting, sari drapery and jewelry read clearly as Ravi Varma.

Likely contributors: SD1.5 at 512x512 is weak on small faces and hands in general, and about 100 training images is a small dataset. Milestone 4's pose control (ControlNet) may help hands and body structure; faces may need a face-restoration step or a higher resolution.

## 9. Human evaluation

Not done. No blind test was run, so there is no human check on the metric-based choice.

## 10. What was not tested

SDXL; resolutions other than 512x512; character-name prompts at scale (only C's small names test); hands/anatomy systematically; variance across training seeds (one seed per run); robustness to long LLM-written prompts (relevant for Milestone 3 given the 77-token limit); other artists.

## 11. Other caveats

- Each member's run was evaluated on their own machine against that machine's own baseline. Cross-run comparisons are therefore approximate, which is why gains are measured against each run's own baseline.
- Thresholds were filled after training but before comparison; they come only from baseline numbers.
- `max_holdout_loss_rise` never came into play because holdout loss was flat.

---

*To add before final: [N] control-art images; 2-3 grids comparing baseline (d) and D800 on the same prompts and seeds (from `bundles/B/final_test/imgs/`); link to D's `training_summary.png` once it is in the hub.*
