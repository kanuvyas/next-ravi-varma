# B_v2caps — Notes

## Run
LoRA rank 16, LR 1e-4, 1600 steps, trained on train_v2 captions (one change vs A_base: caption wording).

## Checkpoint selection
Screened all 8 checkpoints (200-1600). Checkpoint 1600 had the best style_margin (0.0985)
and lowest style_bleed (0.0129) among the deep-scored candidates (1200, 1600). Picked 1600 as best.

## Loss behavior
holdout_loss and train_probe_loss stayed essentially flat (~0.2006, ~0.188) across all
checkpoints — no sign of catastrophic forgetting or overfitting blowup over training.

## Memorization check
Reviewed memorization_ck1200.png by eye. A few generated/nearest-train pairs (notably the
mother-and-child portrait) show fairly close compositional similarity to their matched
training image — similar pose and framing, though not an exact pixel copy. Worth a closer
look / second opinion from another team member before calling this safe; not flagging as
a confirmed leak, but noting it as a point of moderate concern.

## Forgetting check
[[ FILL IN: did you compare against control-art / non-Varma prompts? ]]

## Sweep observations
Scale 0.0 = plain base model (sanity check); scale 1.2 = overshoot test.
Prompts 1 (woman in palace hall) and 3 (king + courtiers, multi-figure) stayed stable and
coherent across all 6 scales, no visible distortion even at 1.2.
Prompt 2 (two figures conversing beside a pillar) is a clear failure mode: the model
rendered the figures as stone statues rather than painted figures, at every scale tested.
This looks like a prompt-interpretation issue rather than a scale issue, since it's
consistent across the whole scale range — worth flagging to the team as a known weak spot
(ambiguous "two figures" prompts without clearer context may need better prompt engineering,
not a LoRA fix).

## Open questions / flags for the team
- Memorization sheet needs a second eye on the mother-and-child pair specifically.
- Row 2 statue failure mode should be mentioned in Failure modes section of EVAL_REPORT_M2.md.
