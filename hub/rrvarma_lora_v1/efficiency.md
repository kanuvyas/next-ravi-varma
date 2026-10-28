# Hardware & Training Efficiency

| Run | Rank | Trainable Params | % of 860M UNet | LoRA Size (MB) vs Base (~4GB) | Peak VRAM (GB) | s/step | Total Time (min) | GPU | Precision |
|---|---|---|---|---|---|---|---|---|---|
| A_base | 16 | 3,188,736 | 0.371% | 6.4 MB | 9.5 | 8.71 | 232.3 | T4 | fp16 + gradient checkpointing |
| B_v2caps | 16 | 3,188,736 | 0.371% | 12.8 MB | N/A | N/A | N/A | Unknown | fp16 + gradient checkpointing |
| C_r32 | 32 | 6,377,472 | 0.742% | 25.5 MB | N/A | 8.3 | 393.0 | T4 | fp16 + gradient checkpointing |
| D_lr2e-4 | 16 | 3,188,736 | 0.371% | 6.4 MB | 6.0 | 9.0 | 300.0 | T4 | bf16 + gradient checkpointing |

*Note: Missing numbers (N/A) indicate metrics that were not recorded by the team member during their run.*

## Hardware Conclusion
Running this training pipeline on a free-tier Google Colab T4 GPU utilizing mixed precision and gradient checkpointing effectively limits the peak VRAM to well under the 16GB threshold (measured at roughly 6.0 to 9.5 GB). By restricting the trainable parameters to just a fraction of a percent of the UNet via LoRA (around 3 to 6 million parameters depending on rank), a full 1,600-step training cycle can be comfortably completed in approximately 4 to 6.5 hours of active compute time. The resulting adapter weights are highly portable, staying under 30 MB compared to the ~4 GB SD1.5 base model[cite: 1].
*Note on Precision: The `lora_stats.py` script initially hard-coded fp16, but Run D was trained using bf16. The metrics for Run D have been manually corrected to reflect bf16.*
