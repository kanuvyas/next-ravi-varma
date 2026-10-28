#!/usr/bin/env bash
# usage: bash make_bundle.sh RUN_NAME MEMBER_LETTER   e.g.  bash make_bundle.sh A_base A
set -e
M=/content/drive/MyDrive/m2; RUN=$M/runs/$1; B=$M/bundles/$2
mkdir -p $B/weights $B/eval
cp $RUN/launch_cmd.sh $B/
for c in $RUN/checkpoint-*; do
  n=$(basename $c); mkdir -p $B/weights/$n
  cp $c/pytorch_lora_weights.safetensors $B/weights/$n/
done
[ -f $RUN/pytorch_lora_weights.safetensors ] && mkdir -p $B/weights/final && cp $RUN/pytorch_lora_weights.safetensors $B/weights/final/
[ -d $RUN/logs ] && cp -r $RUN/logs $B/logs
echo "Copied weights, launch_cmd.sh, logs. You still need to add: loss_log.csv, loss_curve.png, validation_samples/, eval/results.json, sweep_grid.png, efficiency.json, NOTES.md"
