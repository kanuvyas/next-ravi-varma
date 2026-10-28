# C_r32 notes
Run: rank 32 (A_base uses 16), LR 1e-4, train_v1, 1600 steps, T4 GPU, fp16 + gradient checkpointing. Final checkpoint-1600; checkpoints every 200 steps.

Size and speed: 6,377,472 trainable parameters, file 25.5 MB (A_base: 3,188,736, 6.4 MB). Training took about 8.3 s per optimizer step, about 393 min of active GPU time (estimated from TensorBoard timestamps, disconnect gaps excluded). Peak training VRAM was not measured. launch_cmd.sh shows no resume flag.

What I saw in the validation images: the Ravi Varma look (warm ochre/maroon palette, draped saris, palace columns) is already present at epoch 8 (about step 300). By epochs 32-40 (about steps 1180-1470) reds and golds are more saturated. No washed-out colours and no copying. Faces are small and dark and look weak. One validation image (epoch 32) was blacked out by the safety checker, a false positive on a painting.

Memorization sheets (checkpoints 800, 1000, 1200): no near-copies. Composition, face and setting differ from the nearest training painting in every row. The same four training paintings (standing woman in red sari, woman in blue blouse, blue-background portrait, seated Vishnu group) are the nearest neighbour for most rows, so the DINO scores (0.77-0.87) reflect shared subject type rather than copying. Mother-and-child rows are closest in subject but not layout. Many generated women have a faint moustache-like shadow above the lip.

Eval numbers (screen stage, mean over 8 checkpoints, C_r32 vs A_base): style_margin 0.111 vs 0.110, clip_text 0.299 vs 0.305, max_train_dino_p95 0.817 vs 0.826, drift 0.131 vs 0.121, holdout loss about 0.2007 for both (train-holdout gap 0.013 for both). Per-image std of style_margin is 0.03-0.06, so these gaps are noise. Within C_r32, style_margin reaches its plateau by step 800 (0.118) and stays about 0.116-0.123 to step 1600. clip_text dips to 0.288 at steps 1200-1400 and recovers to 0.299 at 1600. Drift rises steadily from 0.10 to 0.156. Deep-stage numbers: see eval/results.json. Caveat from D's bundle: the eval holdout captions carry the v2 style suffix while A_base and C_r32 trained on v1 captions, which affects both runs equally.

Did rank 32 change anything vs rank 16? Not measurably. Style, text alignment, copying and holdout loss are the same within noise, drift is slightly higher, and the LoRA has twice the parameters and file size. Rank 32 gave a bigger file, not a better result.

Checkpoint I would pick from C_r32 alone: around step 1000 (style_margin 0.121, clip_text 0.302, drift 0.126), since style has plateaued by 800-1000 and later steps add drift. The sweep was run on the final checkpoint-1600, not the screen-best; see sweep_grid.png (rows are prompts, columns are scales 0.0-1.2).
