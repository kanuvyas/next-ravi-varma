# The Next Ravi Varma

A course project (BITS Pilani, Goa) that fine-tunes **Stable Diffusion v1.5** with a **LoRA** so its images lean toward the look of Raja Ravi Varma's oil paintings: classical Indian academic realism, soft skin rendering, rich drapery and jewelry.

The full project plan is in [`docs/The_Next_Ravi_Varma_AI_Project_Proposal.docx`](docs/The_Next_Ravi_Varma_AI_Project_Proposal.docx). This repository holds the work of Milestones 1 and 2, copied from the team's shared Google Drive.

## Headline result (Milestone 2)

The released adapter is `hub/rrvarma_lora_v1/`: rank 16, UNet attention layers only, 6.4 MB, trained on 147 paintings. Use it with the trigger phrase `rrvarma style` at LoRA scale 0.8.

On 15 held-out test prompts it improved the style score by +0.037 (95% CI 0.025 to 0.049) over the strongest plain-SD baseline and won 14 of 15 prompts, with no drop in prompt-following. Known limits: it does not recognise character names, faces and hands are often distorted, and the evidence is one seed per run at 512x512. See [`hub/rrvarma_lora_v1/README.md`](hub/rrvarma_lora_v1/README.md) and [`EVAL_REPORT_M2.md`](hub/rrvarma_lora_v1/EVAL_REPORT_M2.md).

## Quick start

```python
# from hub/rrvarma_lora_v1/ ; see requirements_m2.txt and INFERENCE.md
from generate import load, gen
pipe = load("hub/rrvarma_lora_v1", 0.8)
gen(pipe, "rrvarma style, a woman in a silk sari standing in a palace hall", seed=1).save("out.png")
```

Sampling used in our tests: DPM++ (dpmsolver++), 30 steps, CFG 7.0, 512x512, prompts of 77 CLIP tokens or fewer. Research and coursework use only; outputs imitate a style and are not real Ravi Varma works.

## Repository layout

| Path | What is in it |
|---|---|
| `docs/` | Project proposal and the team decision logs for Milestone 1 and Milestone 2 |
| `m1_data/` | Milestone 1 data work: provenance log (`DATA_provenance_Sheet1.csv`, `DATA_provenance_Sheet5.csv`), caption instructions, captions (`captions_txt/`, `final_captions.json`), dropped images list |
| `m2_training/kit/` | The Milestone 2 kit: launch, sweep, evaluation and metric scripts, thresholds, eval prompts |
| `m2_training/dataset/` | Frozen training set: `train_v1` (plain captions), `train_v2` list, dataset card, `DATA_FROZEN.json`, decisions |
| `m2_eval/` | Evaluation holdout and the control-art set with its log |
| `bundles/A` to `bundles/D` | One bundle per training run: `A_base`, `B_v2caps`, `C_r32`, `D_lr2e-4`. Each has the launch command, loss log and curve, efficiency numbers, saved weights, evaluation outputs, sweep results and validation samples |
| `hub/rrvarma_lora_v1/` | The final deliverable: weights, model card, inference script, dataset card, evaluation report, handoff notes for Milestone 3 |

## Reproducing a training run

Each `bundles/*/launch_cmd.sh` holds the exact command for that run, and `m2_training/kit/` has the scripts around it (`setup_session.sh`, `launch.sh`, `sweep.py`, `eval_run.py`, `metrics.py`). The kit was written for a Colab T4 and pins the diffusers version in `DIFFUSERS_TAG`. Start with `m2_training/kit/README.txt`.

## What is not in this repository

Some large or duplicate Drive files were left out on purpose: the full-size raw scans (museum reuse terms), regenerable evaluation images, optimizer state from run folders, and TensorBoard logs. The exact list is in [`docs/NOT_IN_REPO.md`](docs/NOT_IN_REPO.md).

## Data and licensing

The paintings are by Raja Ravi Varma (died 1906) and the originals are in the public domain, but some museum scans carry their own reuse terms. Sources and rights notes are in `m1_data/DATA_provenance_Sheet1.csv` and `m2_training/dataset/DATASET_CARD.md`. The base model is released under CreativeML OpenRAIL-M; check its Hugging Face page for the current terms.

## Team

Four-member team, BITS Pilani Goa. Each member's work is under `bundles/` (training runs) and in the decision logs under `docs/`.
