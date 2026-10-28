#!/usr/bin/env bash
set -e
KIT=/content/drive/MyDrive/m2/kit
pip install -q -r $KIT/requirements_m2.txt
[ -d /content/diffusers ] || git clone --depth 1 --branch "$(cat $KIT/DIFFUSERS_TAG)" https://github.com/huggingface/diffusers /content/diffusers
pip install -q -r /content/diffusers/examples/text_to_image/requirements.txt
accelerate config default --mixed_precision fp16 || accelerate config default
echo "setup done. If Colab asks to restart the session, do it, then continue from the next cell."
