import numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoImageProcessor, AutoModel

dev, norm = "cuda", torch.nn.functional.normalize
_clip  = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
_cproc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
_dino  = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
_dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

def _t(x): return x if torch.is_tensor(x) else x.pooler_output      # survives library version differences
def _load(ps): return [Image.open(p).convert("RGB") for p in ps]
def _batched(fn, items, bs=32):
    items = list(items); return torch.cat([fn(items[i:i+bs]) for i in range(0, len(items), bs)])

@torch.no_grad()
def clip_img(paths):
    def f(ps): return norm(_t(_clip.get_image_features(**_cproc(images=_load(ps), return_tensors="pt").to(dev))).float(), dim=-1).cpu()
    return _batched(f, paths)
@torch.no_grad()
def clip_txt(texts):
    def f(ts): return norm(_t(_clip.get_text_features(**_cproc(text=ts, return_tensors="pt", padding=True,
                        truncation=True, max_length=77).to(dev))).float(), dim=-1).cpu()
    return _batched(f, texts)
@torch.no_grad()
def dino_img(paths):
    def f(ps): return norm(_dino(**_dproc(images=_load(ps), return_tensors="pt").to(dev)).last_hidden_state[:, 0].float(), dim=-1).cpu()
    return _batched(f, paths)

class Refs:
    """holdout = real Varma reference (never trained on); control = non-Varma paintings; train = for memorization."""
    def __init__(self, holdout, control, train):
        self.h, self.c = clip_img(holdout), clip_img(control)
        self.tr_clip, self.tr_dino = clip_img(train), dino_img(train)
        self.h_cent, self.c_cent = norm(self.h.mean(0), dim=-1), norm(self.c.mean(0), dim=-1)

def score(gen_paths, prompts, R):
    g, t, gd = clip_img(gen_paths), clip_txt(prompts), dino_img(gen_paths)
    s_h, s_c = g @ R.h_cent, g @ R.c_cent
    per = {"clip_text": (g * t).sum(-1), "style_sim": s_h, "style_margin": s_h - s_c,
           "max_train_clip": (g @ R.tr_clip.T).max(1).values, "max_train_dino": (gd @ R.tr_dino.T).max(1).values}
    return {k: v.numpy() for k, v in per.items()}           # per-image arrays, keep them

def summarize(per):
    out = {}
    for k, v in per.items():
        out[k] = float(v.mean()); out[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out["max_train_dino_p95"] = float(np.quantile(per["max_train_dino"], .95))
    out["max_train_clip_p95"] = float(np.quantile(per["max_train_clip"], .95))
    return out

def drift(base_paths, lora_paths):
    """Forgetting: 1 - CLIP cosine between base and LoRA output for the SAME prompt+seed (pairs must align)."""
    return 1 - (clip_img(base_paths) * clip_img(lora_paths)).sum(-1).numpy()

def boot_ci(x, n=2000, seed=0):
    """x = per-PROMPT means (average seeds within each prompt first). 95% bootstrap CI over prompts."""
    rng, x = np.random.default_rng(seed), np.asarray(x)
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return tuple(np.percentile(m, [2.5, 97.5]))

def ceiling(R):
    """Style margin of REAL held-out Varma images vs the centroid of the others: the best a generator could look."""
    m = []
    for i in range(len(R.h)):
        cent = norm(torch.cat([R.h[:i], R.h[i+1:]]).mean(0), dim=-1)
        m.append(float(R.h[i] @ cent - R.h[i] @ R.c_cent))
    return float(np.mean(m))
