"""usage: python compare.py     (reads m2/bundles/*/eval/results.json)"""
import glob, json, os
import pandas as pd
KIT = os.path.dirname(os.path.abspath(__file__)); MD = "/content/drive/MyDrive/m2"
TH = json.load(open(f"{KIT}/thresholds.json"))
assert all(v is not None for v in TH.values()), "fill thresholds.json before comparing"
rows = []
for p in sorted(glob.glob(f"{MD}/bundles/*/eval/results.json")):
    d = json.load(open(p)); b = d["baseline"]["dev:d_engineered"]["style_margin"]
    cks = [c for c in d["checkpoints"] if c["stage"] in ("screen", "deep")]
    hl_min = min(c["holdout_loss"] for c in cks)
    for c in cks:
        elig = (c["max_train_dino_p95"] <= TH["mem_flag_dino_p95"] and c["style_bleed"] <= TH["max_style_bleed"]
                and c["holdout_loss"] <= hl_min + TH["max_holdout_loss_rise"])
        rows.append({"run": d["run"], "step": c["step"], "stage": c["stage"], "style_margin": c["style_margin"],
                     "gain_vs_d": c["style_margin"] - b, "clip_text": c["clip_text"], "dino_p95": c["max_train_dino_p95"],
                     "holdout_loss": c["holdout_loss"], "bleed": c["style_bleed"], "drift": c["drift"], "eligible": elig})
df = pd.DataFrame(rows)
print("=== all checkpoints ===");   print(df.sort_values(["run", "step"]).round(4).to_string(index=False))
deep = df[(df.stage == "deep") & df.eligible].sort_values("gain_vs_d", ascending=False)
print("\n=== eligible deep-scored candidates, best first ===");  print(deep.round(4).to_string(index=False))
if len(deep):
    top = deep.iloc[0]; ties = deep[deep.gain_vs_d >= top.gain_vs_d - TH["tie_band"]].sort_values("step")
    print(f"\nTop: {top.run} step {top.step} (gain {top.gain_vs_d:.4f}). Within tie band ({TH['tie_band']:.4f}):")
    print(ties[["run", "step", "gain_vs_d"]].round(4).to_string(index=False))
    print("Rule: among ties pick the fewest training steps, then the lower rank. Humans confirm after looking at the images.")
