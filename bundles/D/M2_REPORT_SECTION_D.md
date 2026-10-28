## Member D: dataset freeze and LoRA run D_lr2e-4

**Data.** I froze the M1 collection into two training sets, `train_v1` (plain captions) and `train_v2` (same captions plus a fixed style suffix). A leak check against the evaluation holdout dropped two images (`varma_ladies_moonlight_a01`, `varma_murugan_a62`), leaving 147 paintings. One flagged pair (`kadambari_sitar_a51` vs `b04`) was inspected and kept. Hashes of both metadata files are recorded in `DATA_FROZEN.json`.

**Run.** D_lr2e-4 trains a rank-16 LoRA on SD v1.5 with learning rate 2e-4 for 1600 steps on `train_v1` (effective batch 4, constant schedule with 100 warmup steps, seed 42, bf16 with gradient checkpointing). It ran on a Colab T4 at about 9 s per optimizer step, roughly 5 hours, and was resumed from checkpoints several times after Colab disconnects.

**Findings.** Style was mostly learned by step 400-800. The copying score rose early (0.80 to 0.83 by step 600) and then stayed flat, and prompt alignment (clip_text) peaked at step 400 (0.315). The style measure style_margin at step 800 is 0.123 on dev prompts (0.107 on the final test prompts), and falls to 0.016 without the trigger phrase, so the style needs the trigger. On 15 held-out test prompts, D800 at scale 0.8 improved style_margin by +0.037 (95% CI 0.025-0.049) over the strongest plain-SD baseline, winning 14/15 prompts with no drop in prompt-following (B, final test).

**Selection.** Eleven checkpoints were scored in depth. Step 1600 scored highest, but nine checkpoints sat within the tie band (0.0295), so the pre-set tie rule picked the fewest steps: D_lr2e-4 at step 800. Recommended inference scale is 0.8 (C's sweep).

**Limitations.** The holdout captions carry the v2 suffix while D trained on v1; test captions include style words that raise all baselines, so the conservative figure is the gain over baseline (a). The holdout painting `kadambari_sitar_b04` is a near-copy of a training painting, so B reports test results with and without it. Holdout loss (0.2008) is no better than the plain-SD baseline (0.201). Names of characters do not work as prompts. Faces and hands are distorted in nearly every image.
