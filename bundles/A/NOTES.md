# NOTES: A_base Run

## Run Configuration
* **Run Name:** A_base
* **Dataset:** train_v1
* **Hyperparameters:** Rank 16, Learning Rate 1e-4, 1600 steps
* **Hardware:** T4 GPU (Google Colab Free Tier)
* **Performance:** 8.71 s/it

## Observations & Anomalies
* **Hardware Compatibility:** The `--enable_xformers_memory_efficient_attention` flag had to be excluded from the launch command. The T4 GPU architecture caused a `NotImplementedError` when attempting to use xformers alongside `bf16` mixed precision.
* **Disconnects & Resumes:** I encountered a Colab free-tier GPU usage timeout midway through the run, which safely paused the training. I later restarted the session, cleared the VRAM (to avoid Out of Memory errors), and successfully completed the run using the `--resume_from_checkpoint=latest` flag from step 600.
* **Evaluation Missing Files:** During the evaluation phase, the script initially crashed because the `control_art` folder was missing from the shared drive, but it ran successfully once the folder was added and the Colab drive connection was force-remounted.

## Checkpoint Selection
* **Selected Winner:** `checkpoint-1400`
* **Reasoning:** After screening all checkpoints and deep-scoring the top candidates, `checkpoint-1400` emerged as the definitive best model. It achieved the highest style margin of **0.1188**. Crucially, it beat the engineered prompt baseline (`d_engineered`, which scored 0.0169) by an impressive margin of +0.1019. Furthermore, `checkpoint-1400` maintained a better holdout loss (0.20064) compared to the runner-up `checkpoint-1000` (0.20068), proving it learned the target style most effectively without losing generalizability.
