# rrvarma LoRA v1

A LoRA adapter for Stable Diffusion v1.5 that pushes generations toward the look of Raja Ravi Varma's oil paintings (classical Indian academic realism: soft skin rendering, rich drapery and jewelry).

- **Base model:** stable-diffusion-v1-5/stable-diffusion-v1-5 (CreativeML OpenRAIL-M; see the model's Hugging Face page for its current licence terms)
- **Adapter:** LoRA, rank 16, UNet attention layers only (3,188,736 trainable parameters, 0.371% of the UNet). File size 6.4 MB.
- **Released weights:** checkpoint-800 of the run `D_lr2e-4` (the run continued to step 1600; step 800 was chosen by the team's selection rule, see below).
- **Trigger phrase:** `rrvarma style`, placed first in the prompt.
- **Recommended LoRA scale:** 0.8

## How to use

```python
from generate import load, gen
pipe = load("<path to this folder>", 0.8)
gen(pipe, "rrvarma style, a woman in a silk sari standing in a palace hall", seed=1).save("out.png")
```

Sampling used in our tests: DPM++ (dpmsolver++), 30 steps, CFG 7.0, 512x512. Negative prompt: `lowres, blurry, deformed, extra limbs, bad anatomy, text, watermark`. Keep prompts at 77 CLIP tokens or fewer. See `INFERENCE.md` for details.

## Training data

147 paintings by Raja Ravi Varma (collected set of 149, two dropped after a leak check against the evaluation holdout). Captions are plain descriptions without character names and without a style suffix (`train_v1`). Sources and authorship flags are in the M1 provenance log; see `DATASET_CARD.md`.

## Training

| Setting | Value |
|---|---|
| Method | diffusers `train_text_to_image_lora.py` |
| Rank / learning rate | 16 / 2e-4 |
| Steps run / released | 1600 / 800 |
| Effective batch size | 4 (1 x 4 gradient accumulation) |
| Schedule | constant with 100 warmup steps |
| Seed | 42 |
| Precision | bf16 mixed precision with gradient checkpointing |
| Hardware | Colab T4, about 9 s per optimizer step (approximate, read from the training progress bar) |

The run was resumed from checkpoints several times because of Colab disconnects. Full settings: `training_config.json`. Loss: `training_summary.png`, `loss_curve.png`, `loss_log.csv`.

## Why checkpoint 800

Eleven checkpoints were scored in depth. Step 1600 had the top score, but nine checkpoints fell within the tie band (0.0295, set from baseline numbers before results were compared). The team's tie rule prefers fewer steps, which gave step 800. Details are in `TEAM_DECISIONS_M2.md`. Across the four training runs compared, only the caption change made a clear difference (lower style); rank and learning rate were within noise (B's comparison). D's run also used bf16 while the others used fp16, so the learning-rate comparison is confounded with precision.

## Measured results (checkpoint 800, deep evaluation)

| Metric | Value | Note |
|---|---|---|
| style_margin (dev prompts) | 0.123 | gain over baseline (d) on dev prompts = 0.106; on the 15 test prompts the score is 0.107 |
| style_margin without the trigger phrase | 0.016 | the style needs the trigger |
| clip_text (prompt alignment) | 0.300 | |
| copying score (max_train_dino_p95) | 0.846 | below our 0.85 flag; sheet checked by eye, no near-copies found |
| style_bleed | 0.011 | limit was 0.059 |
| holdout loss | 0.2008 | the plain-SD baseline is 0.201, so there is no gain in this number |

The sample is small (8 development prompts), so differences of a few thousandths are noise.

**Final test (B, `EVAL_REPORT_M2.md`):** on 15 held-out test prompts, checkpoint 800 at scale 0.8 improved style_margin by +0.037 (95% CI 0.025 to 0.049) over the strongest plain-SD baseline, winning 14 of 15 prompts, with no drop in prompt-following. See the report for the results with and without the near-duplicate holdout painting.

## Limitations

- **Names do not work.** Training captions had no character or person names and only UNet attention was adapted, so prompts naming figures do not map to anything learned. See `NAMES_TEST.md`.
- **Caption mismatch in evaluation.** The evaluation holdout captions end with the v2 style suffix, but this model was trained on v1 captions without it. Test captions include style words, which raises all baselines; the gain over baseline (d) is inflated because its style words were truncated past 77 tokens. The conservative figure is the gain over baseline (a).
- **Possible near-duplicate in the holdout.** One holdout painting (`kadambari_sitar_b04`) is a very close version of a training painting (`a51`). Test results are reported with and without it in `EVAL_REPORT_M2.md`; excluding it changes the gain by less than 0.001.
- **Faces and hands are distorted** in nearly every generated image.
- **Small training set.** 147 images: a small dataset raises the risk of reproducing compositions; copying was monitored and none found.
- **Safety checker is disabled in `generate.py`** because it falsely blocks classical paintings. Any user-facing use needs its own content decision.
- **Limited evidence.** One training seed per run; only 512x512 was tested; there was no human blind test; style is measured with CLIP/DINO proxies; and holdout loss was flat, so it cannot show overfitting.
- **Style, not scholarship.** The output imitates the look of the paintings; it is not an authentic or attributed Ravi Varma work.

## Intended use and licensing

Research and coursework only. Outputs imitate a painting style and must not be passed off as real Raja Ravi Varma works. Varma's paintings depict deities and religious figures, so this model must not be used to generate demeaning content. Museum scans in the training set may carry their own reuse terms (see the provenance log). The base model is released under CreativeML OpenRAIL-M; check its Hugging Face page for the current terms.

## Files in this folder

`pytorch_lora_weights.safetensors`, `training_config.json`, `README.md` (this file), `DATASET_CARD.md`, `training_summary.png`, `loss_curve.png`, `loss_log.csv`, `validation_samples/`, `INFERENCE.md`, `generate.py`, `NAMES_TEST.md`, `VISUAL_VOCAB.md`, `efficiency.md`, `EVAL_REPORT_M2.md`, `TEAM_DECISIONS_M2.md`, `HANDOFF_M3.md`, `requirements_m2.txt`, `ENV_INFO.txt`, `reference/`, `sweep_winner/`, `vocab/`, `names_test/`.
