"""usage: python eval_run.py --run_dir RUN --out OUT [--deep_k 3] [--deep_seeds 1,2,3,4] [--sets dev,controls] [--only_steps 800] [--scale 1.0]
--sets test scores the held-out TEST prompts: use ONCE, on the chosen checkpoint (needs --only_steps)."""
import argparse, glob, json, os, random, re, sys
import numpy as np, torch
KIT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, KIT)
from generate import load, swap_lora, run_prompts, versions
import metrics as M
from sheets import sheet, top_pairs
from holdout_loss import denoise_loss

MD = "/content/drive/MyDrive/m2"
ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--holdout_dir", default=f"{MD}/eval/eval_holdout")
ap.add_argument("--control_dir", default=f"{MD}/eval/control_art")
ap.add_argument("--train_dir", default=f"{MD}/dataset/train_v1")
ap.add_argument("--deep_k", type=int, default=3); ap.add_argument("--deep_seeds", default="1,2,3,4")
ap.add_argument("--only_steps", default=""); ap.add_argument("--sets", default="dev,controls")
ap.add_argument("--scale", type=float, default=1.0)
a = ap.parse_args()
SETS = set(a.sets.split(","))
if "test" in SETS: assert a.only_steps, "--sets test needs --only_steps (use the test set once, on the chosen checkpoint)"

TRIG, SUFFIX = "rrvarma style, ", "oil painting, dramatic lighting, classical Indian attire"
PR = json.load(open(f"{KIT}/eval_prompts.json"))
HOCAP = json.load(open(f"{KIT}/eval_holdout_captions.json"))       # {filename: "rrvarma style, ..."}
if not PR.get("test"): PR["test"] = list(HOCAP.values())
for k in ("dev", "test"): assert all(p.startswith(TRIG) for p in PR[k]), f"{k} prompts must start with the trigger"
OUT = a.out; os.makedirs(OUT, exist_ok=True)
imgs = lambda d: sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg")))
rep = lambda ps, seeds: [p for p in ps for _ in seeds]
R = M.Refs(imgs(a.holdout_dir), imgs(a.control_dir), imgs(a.train_dir))
CEIL = M.ceiling(R); print("ceiling (real Varma style_margin):", round(CEIL, 4))

def variants(p):
    body = p.replace(TRIG, "", 1)
    return {"a_trigger_only": p, "b_named": f"{body}, in the style of Raja Ravi Varma",
            "d_engineered": f"{body}, oil painting, 19th century Indian academic realism, soft skin rendering, "
                            f"chiaroscuro, rich silk drapery, warm palette"}
pipe = load(None)
tmeta = [json.loads(l) for l in open(f"{a.train_dir}/metadata.jsonl")]
random.seed(0); PROBE = {m["file_name"]: m["text"] for m in random.sample(tmeta, min(11, len(tmeta)))}
def probes(): return (denoise_loss(pipe, a.holdout_dir, HOCAP), denoise_loss(pipe, a.train_dir, PROBE))

def gen_score(prompts, seeds, sub):
    paths = run_prompts(pipe, prompts, seeds, f"{OUT}/imgs/{sub}")
    return paths, M.score(paths, rep(prompts, seeds), R)
def summ(per, n_prompts, n_seeds):
    s = M.summarize(per)
    s["per_prompt_style_margin"] = per["style_margin"].reshape(n_prompts, n_seeds).mean(1).tolist(); return s

# ---------- baselines (plain SD1.5, no LoRA) on THIS machine ----------
BF = f"{OUT}/baseline.json"
B = json.load(open(BF)) if os.path.exists(BF) else {}
B.setdefault("gpu", torch.cuda.get_device_name(0)); B.setdefault("versions", versions()); B["ceiling"] = CEIL
SEEDS4 = [1, 2, 3, 4]
for setname in [s for s in ("dev", "test") if s in SETS]:
    for v in ("a_trigger_only", "b_named", "d_engineered"):
        key = f"{setname}:{v}"
        if key not in B:
            ps = [variants(p)[v] for p in PR[setname]]
            _, per = gen_score(ps, SEEDS4, f"base/{setname}_{v}"); B[key] = summ(per, len(ps), 4)
if ("controls" in SETS or "dev" in SETS) and "controls" not in B:
    _, per = gen_score(PR["controls"], SEEDS4, "base/controls"); B["controls"] = summ(per, len(PR["controls"]), 4)
if "holdout_loss" not in B: B["holdout_loss"], B["train_probe_loss"] = probes()
json.dump(B, open(BF, "w"), indent=1)

# ---------- results file (resumable) ----------
RF = f"{OUT}/results.json"
RES = json.load(open(RF)) if os.path.exists(RF) else {"run": os.path.basename(a.run_dir.rstrip("/")), "checkpoints": []}
RES.update({"gpu": B["gpu"], "ceiling": CEIL, "baseline": B, "versions": versions()})
done = lambda s, st: any(r["step"] == s and r["stage"] == st for r in RES["checkpoints"])
def save(): json.dump(RES, open(RF, "w"), indent=1)

step_of = lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1))
cks = sorted([c for c in glob.glob(f"{a.run_dir}/checkpoint-*") if os.path.exists(f"{c}/pytorch_lora_weights.safetensors")], key=step_of)
if a.only_steps:
    keep = {int(x) for x in a.only_steps.split(",")}; cks = [c for c in cks if step_of(c) in keep]
print("checkpoints found:", [step_of(c) for c in cks])

def control_stats(cpaths, seeds):
    """Forgetting on no-trigger prompts vs the base model's images for the SAME prompt+seed (matched by file name)."""
    per = M.score(cpaths, rep(PR["controls"], seeds), R)
    bpaths = [f"{OUT}/imgs/base/controls/{os.path.basename(p)}" for p in cpaths]
    bper = M.score(bpaths, rep(PR["controls"], seeds), R)
    return {"style_bleed": float((per["style_margin"] - bper["style_margin"]).mean()),
            "drift": float(M.drift(bpaths, cpaths).mean()),
            "controls_clip_text_change": float((per["clip_text"] - bper["clip_text"]).mean())}

KEYS = ("style_margin", "style_margin_std", "clip_text", "max_train_dino_p95")
def screen(ck):
    s = step_of(ck); swap_lora(pipe, ck, 1.0); tag = f"ck{s}/screen"
    dp = run_prompts(pipe, PR["dev"], [1, 2], f"{OUT}/imgs/{tag}/dev")
    cp = run_prompts(pipe, PR["controls"], [1], f"{OUT}/imgs/{tag}/controls")
    sm = M.summarize(M.score(dp, rep(PR["dev"], [1, 2]), R))
    row = {"step": s, "stage": "screen", **{k: sm[k] for k in KEYS}, **control_stats(cp, [1])}
    row["holdout_loss"], row["train_probe_loss"] = probes(); return row

def deep(ck):
    s = step_of(ck); swap_lora(pipe, ck, 1.0); seeds = [int(x) for x in a.deep_seeds.split(",")]; tag = f"ck{s}/deep"
    dp = run_prompts(pipe, PR["dev"], seeds, f"{OUT}/imgs/{tag}/dev")
    dper = M.score(dp, rep(PR["dev"], seeds), R); sm = M.summarize(dper)
    cp = run_prompts(pipe, PR["controls"], seeds, f"{OUT}/imgs/{tag}/controls")
    row = {"step": s, "stage": "deep", **{k: sm[k] for k in KEYS}, **control_stats(cp, seeds)}
    row["per_prompt_style_margin"] = dper["style_margin"].reshape(len(PR["dev"]), len(seeds)).mean(1).tolist()
    abl = [p.replace(TRIG, "", 1) for p in PR["dev"]]                       # trigger removed
    ap_ = run_prompts(pipe, abl, seeds, f"{OUT}/imgs/{tag}/no_trigger")
    row["style_margin_no_trigger"] = float(M.score(ap_, rep(abl, seeds), R)["style_margin"].mean())
    nec = [p.replace(TRIG, "", 1) + ", " + SUFFIX for p in PR["dev"]]         # only M1's constant style words, no trigger
    np_ = run_prompts(pipe, nec, seeds, f"{OUT}/imgs/{tag}/suffix_only")
    row["style_margin_suffix_only"] = float(M.score(np_, rep(nec, seeds), R)["style_margin"].mean())
    sheet(top_pairs(dp, R, imgs(a.train_dir)), f"{OUT}/memorization_ck{s}.png")
    row["holdout_loss"], row["train_probe_loss"] = probes(); return row

def test_stage(ck):
    s = step_of(ck); swap_lora(pipe, ck, a.scale); ps = PR["test"]
    paths = run_prompts(pipe, ps, SEEDS4, f"{OUT}/imgs/ck{s}/test_scale{a.scale}")
    per = M.score(paths, rep(ps, SEEDS4), R)
    return {"step": s, "stage": "test", "scale": a.scale, **M.summarize(per),
            "per_prompt_style_margin": per["style_margin"].reshape(len(ps), 4).mean(1).tolist()}

if "dev" in SETS:
    for ck in cks:
        if not done(step_of(ck), "screen"): RES["checkpoints"].append(screen(ck)); save(); print("screened", step_of(ck))
    scr = [r for r in RES["checkpoints"] if r["stage"] == "screen" and r["step"] in {step_of(c) for c in cks}]
    for r in sorted(scr, key=lambda r: r["style_margin"], reverse=True)[:a.deep_k]:
        if not done(r["step"], "deep"):
            RES["checkpoints"].append(deep(next(c for c in cks if step_of(c) == r["step"]))); save(); print("deep-scored", r["step"])
if "test" in SETS:
    for ck in cks:
        if done(step_of(ck), "test"): print("TEST already scored for", step_of(ck), "- not repeating"); continue
        RES["checkpoints"].append(test_stage(ck)); save()

import pandas as pd
df = pd.DataFrame([r for r in RES["checkpoints"] if r["stage"] in ("screen", "deep")])
if len(df):
    cols = ["step", "stage", "style_margin", "clip_text", "max_train_dino_p95", "holdout_loss", "train_probe_loss", "style_bleed", "drift"]
    print(df[cols].sort_values(["stage", "step"]).round(4).to_string(index=False))
