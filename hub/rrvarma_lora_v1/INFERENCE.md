# Inference
Winner: D_lr2e-4 checkpoint-800 (rank 16). Recommended scale: 0.8. Trigger: `rrvarma style`, placed first in the prompt.

```python
from generate import load, gen
pipe = load("<path to checkpoint-800>", 0.8)
gen(pipe, "rrvarma style, a woman in a silk sari standing in a palace hall", seed=1).save("out.png")
```
Sampling: DPM++ (dpmsolver++), 30 steps, CFG 7.0, 512x512. Negative prompt: `lowres, blurry, deformed, extra limbs, bad anatomy, text, watermark`

## Measured (Tesla T4, fp16, 512x512)
- Peak VRAM: 2.81 GB
- Time per image: 6.08 s (mean of 5 after a warmup image)
- LoRA file size: 6.4 MB

## Prompting notes for M3
- Put the trigger first. Keep the whole prompt at 77 CLIP tokens or fewer (check with n_tokens); longer prompts are silently truncated.
- Training captions had no character names and only UNet attention was adapted, so name-based prompts do not map to anything learned. See NAMES_TEST.md.
- Prefer phrases that worked in VISUAL_VOCAB.md.
- The safety checker is disabled in generate.py because it false-positives on classical paintings. Anything user-facing (Milestone 4 dashboard) needs its own content-policy decision.

## Scale choice
Sweep on D_lr2e-4 checkpoint-800 (3 prompts x 2 seeds per scale; images in sweep_winner/grid.png). Chosen scale: 0.8.

|   scale |   style_margin |   clip_text |
|--------:|---------------:|------------:|
|     0   |         0.023  |      0.2876 |
|     0.4 |         0.0687 |      0.3019 |
|     0.6 |         0.0814 |      0.3114 |
|     0.8 |         0.1081 |      0.2976 |
|     1   |         0.1175 |      0.2996 |
|     1.2 |         0.1186 |      0.2967 |

style_margin rises steeply to 0.8 and is nearly flat from 1.0 to 1.2; clip_text peaks at 0.6 and is roughly flat from 0.8 up. 0.8 gives about 91% of the style gain seen at 1.0 with no loss in text alignment. The sample is small, so differences of a few thousandths are noise.
