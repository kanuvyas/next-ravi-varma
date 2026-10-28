import argparse, json, os
from safetensors import safe_open
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("out")
ap.add_argument("--gpu", default="?"); ap.add_argument("--peak_vram_gb", type=float, default=None)
ap.add_argument("--s_per_step", type=float, default=None); ap.add_argument("--wall_clock_min", type=float, default=None)
a = ap.parse_args()
UNET = 859_520_964                                   # SD1.5 UNet parameters (~860M)
path = os.path.join(a.ckpt, "pytorch_lora_weights.safetensors")
n = 0
with safe_open(path, "pt") as f:
    for k in f.keys(): n += f.get_tensor(k).numel()
d = {"trainable_params": n, "pct_of_unet": round(100 * n / UNET, 3), "file_MB": round(os.path.getsize(path) / 1e6, 1),
     "gpu": a.gpu, "peak_vram_gb": a.peak_vram_gb, "s_per_optimizer_step": a.s_per_step, "wall_clock_min": a.wall_clock_min,
     "precision": "fp16 + gradient checkpointing"}
json.dump(d, open(a.out, "w"), indent=1); print(d)
