## Selection thresholds (B, Oct 3)

Written to kit/thresholds.json:
- tie_band = 0.0295: half-width of the 95% bootstrap CI (2000 resamples, seed 0)
  of per-prompt style_margin for baseline (d) on the 8 dev prompts,
  from bundles/B/eval/baseline.json.
- max_style_bleed = 0.059: 2 x tie_band (rule of thumb).
- max_holdout_loss_rise = 0.01: 5% of baseline holdout loss (0.201) (rule of thumb).
- mem_flag_dino_p95 = 0.85: starting point; flagged checkpoints are also checked by eye.

Sanity check: ceiling (real held-out Varma style_margin) = 0.123, clearly > 0.

Timing: these were filled after the training runs and some eval outputs existed,
but before compare.py was run or any member's results.json was examined.
Values come only from the plain-SD baseline, not from any LoRA results.

Note: the tie band is wide (per-prompt scores vary a lot), so close results
will count as ties and the rule will prefer the fewer-steps checkpoint.

## Model selection (B, Oct 3)

Winner: D_lr2e-4, checkpoint-800 (rank 16, LR 2e-4, train_v1 captions).

How: compare.py with the thresholds above, applied as written.
- All 11 deep-scored checkpoints were eligible.
- Top score: D_lr2e-4 step 1600 (gain_vs_d 0.1155). 9 checkpoints fell within
  the tie band (0.0295). Tie rule: fewest steps (800), then lower rank (16 < 32),
  giving D_lr2e-4 800 (gain_vs_d 0.1058).
- D800 dino_p95 = 0.846, just under the 0.85 flag; memorization sheet checked
  by eye, no near-copies found.
- D800 also has higher clip_text (0.2996) than D1400/1600 (~0.29).

Not done: no selection meeting, no blind test, no combo run.
Recommended LoRA scale: pending C's sweep on this checkpoint.

