import os, json, torch
import importlib.metadata as md
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
NEG = "lowres, blurry, deformed, extra limbs, bad anatomy, text, watermark"
ADAPTER = "varma"

def versions():
    return {p: md.version(p) for p in ["torch", "diffusers", "transformers", "peft", "accelerate", "safetensors"]}

def load(lora_dir=None, weight=1.0):
    # safety_checker=None: it often blacks out classical paintings (false positives). Evaluation only.
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++")
    pipe.set_progress_bar_config(disable=True)
    if lora_dir:
        swap_lora(pipe, lora_dir, weight)
    return pipe

def swap_lora(pipe, lora_dir, weight=1.0):
    """Replace whatever LoRA is loaded (or none) with lora_dir. Avoids adapter-name collisions and base reloads."""
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass                                   # nothing loaded yet
    if lora_dir:                               # dir containing pytorch_lora_weights.safetensors
        pipe.load_lora_weights(lora_dir, adapter_name=ADAPTER)
        pipe.set_adapters([ADAPTER], adapter_weights=[weight])
    return pipe

def set_scale(pipe, w):
    pipe.set_adapters([ADAPTER], adapter_weights=[w])

def gen(pipe, prompt, seed, steps=30, cfg=7.0):
    g = torch.Generator("cuda").manual_seed(seed)
    return pipe(prompt, negative_prompt=NEG, num_inference_steps=steps, guidance_scale=cfg,
                height=512, width=512, generator=g).images[0]

def grid(imgs, rows, cols):
    w, h = imgs[0].size
    out = Image.new("RGB", (cols * w, rows * h))
    for i, im in enumerate(imgs):
        out.paste(im, ((i % cols) * w, (i // cols) * h))
    return out

def run_prompts(pipe, prompts, seeds, out_dir, skip_existing=True):
    """Files: p{prompt_idx:02d}_s{seed}.png. Returns paths in (prompt, seed) order. Safe to re-run after a disconnect."""
    os.makedirs(out_dir, exist_ok=True); paths = []
    for pi, p in enumerate(prompts):
        for s in seeds:
            path = f"{out_dir}/p{pi:02d}_s{s}.png"
            if not (skip_existing and os.path.exists(path)):
                gen(pipe, p, s).save(path)
            paths.append(path)
    json.dump({"prompts": prompts, "seeds": list(seeds), "neg": NEG, "versions": versions()},
              open(f"{out_dir}/meta.json", "w"), indent=1)
    return paths

def n_tokens(pipe, prompt):                    # CLIP text limit is 77 including start/end tokens
    return len(pipe.tokenizer(prompt).input_ids)

def peak_vram_gb():
    return torch.cuda.max_memory_allocated() / 1e9

if __name__ == "__main__":                      # self-test:  python generate.py [lora_dir]
    import sys, time
    lora = sys.argv[1] if len(sys.argv) > 1 else None
    torch.cuda.reset_peak_memory_stats()
    pipe = load(lora)
    t = time.time(); im = gen(pipe, "rrvarma style, a woman in a silk sari standing in a palace hall", 1)
    im.save("selftest.png"); print("ok", im.size, f"{time.time()-t:.1f}s (incl. warmup)", f"peak VRAM {peak_vram_gb():.2f} GB")
    if lora:                                    # repeated swaps must not error
        swap_lora(pipe, lora, 0.5); swap_lora(pipe, lora, 1.0); swap_lora(pipe, None); gen(pipe, "a cat", 1); print("swap ok")
