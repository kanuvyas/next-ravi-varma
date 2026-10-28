import json, os, random, sys
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def sheet(rows, out, t=220):
    """rows = [[(image_path, label), ...], ...]  ->  one contact sheet"""
    W, H = t * len(rows[0]), (t + 18) * len(rows)
    canvas = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(canvas)
    for r, row in enumerate(rows):
        for c, (p, label) in enumerate(row):
            im = Image.open(p).convert("RGB").resize((t, t)); canvas.paste(im, (c * t, r * (t + 18)))
            d.text((c * t + 3, r * (t + 18) + t + 2), label[:38], fill="black")
    canvas.save(out)

def top_pairs(gen_paths, R, train_paths, k=10):
    """The k generated images most similar to a training image (DINO), shown next to that training image."""
    from metrics import dino_img
    gd = dino_img(gen_paths); S = gd @ R.tr_dino.T
    best, idx = S.max(1); order = best.argsort(descending=True)[:k]
    return [[(gen_paths[i], f"gen dino={best[i]:.2f}"), (train_paths[int(idx[i])], "nearest train")] for i in order.tolist()]

def blind_grid(conds, out_png, key_json, seed=0):
    """conds = {"baseline_d": path, "A_ck800": path, ...}. Column order is shuffled; the key is saved separately."""
    from generate import grid
    names = list(conds); random.Random(seed).shuffle(names)
    grid([Image.open(conds[n]).convert("RGB") for n in names], 1, len(names)).save(out_png)
    json.dump(names, open(key_json, "w"))     # keep the key OUT of the shared folder
