#!/usr/bin/env bash
# usage: bash launch.sh NAME DATA_DIR RANK LR STEPS [extra flags e.g. --resume_from_checkpoint=latest]
set -e
NAME=$1; DATA=$2; RANK=$3; LR=$4; STEPS=$5; shift 5
OUT=/content/drive/MyDrive/m2/runs/$NAME; mkdir -p "$OUT"
cat > "$OUT/launch_cmd.sh" <<EOF
accelerate launch --mixed_precision="bf16" /content/diffusers/examples/text_to_image/train_text_to_image_lora.py \\
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \\
  --train_data_dir="$DATA" --caption_column="text" \\
  --resolution=512 --center_crop --random_flip \\
  --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing \\
  --dataloader_num_workers=2 --max_train_steps=$STEPS \\
  --learning_rate=$LR --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 \\
  --rank=$RANK --seed=42 --checkpointing_steps=200 \\
  --validation_prompt="rrvarma style, a woman in a silk sari standing in a palace hall" \\
  --num_validation_images=4 --validation_epochs=8 --report_to="tensorboard" \\
  --output_dir="$OUT" $@
EOF
bash "$OUT/launch_cmd.sh"
