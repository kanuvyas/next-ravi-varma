"""usage: python leak_check.py TRAIN_DIR HOLDOUT_DIR [--v2 TRAIN_V2_DIR] [--out leakage_sheet.png]
Run on M1's training_dataset/ vs eval_holdout/ now; D re-runs it on dataset/train_v1 before freezing."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imagehash
from PIL import Image
from metrics import clip_img, dino_img
from sheets import sheet

ap = argparse.ArgumentParser(); ap.add_argument("train"); ap.add_argument("holdout")
ap.add_argument("--v2", default=None); ap.add_argument("--out", default="leakage_sheet.png")
a = ap.parse_args()
ext = (".jpg", ".jpeg", ".png")
lst = lambda d: sorted(f for f in os.listdir(d) if f.lower().endswith(ext))
stem = lambda f: os.path.splitext(f)[0]
tr, ho = lst(a.train), lst(a.holdout)
print(f"{len(tr)} train images, {len(ho)} holdout images")

# Layer 0: v1 and v2 must have identical image lists (only if --v2 given)
if a.v2:
    v2 = {stem(f) for f in lst(a.v2)}
    print("v1/v2 image lists identical:", {stem(f) for f in tr} == v2)

# Layer 1: same filename / stem, or a holdout name inside metadata.jsonl
hs = {stem(f) for f in ho}
print("stem overlap:", {stem(f) for f in tr} & hs)
mp = os.path.join(a.train, "metadata.jsonl")
if os.path.exists(mp):
    meta = [json.loads(l)["file_name"] for l in open(mp)]
    print("holdout named in metadata:", [m for m in meta if stem(m) in hs])

# Layer 2: perceptual hash (lenient on purpose; eyeball everything flagged)
ph = lambda d, fs: {f: imagehash.phash(Image.open(os.path.join(d, f)).convert("RGB")) for f in fs}
pt, phh = ph(a.train, tr), ph(a.holdout, ho)
print("pHash distance <= 8:", [(h, t, int(x - y)) for h, x in phh.items() for t, y in pt.items() if x - y <= 8])

# Layer 3: semantic near-duplicates. For EVERY holdout image, show its 3 nearest training images.
hp, tp = [os.path.join(a.holdout, f) for f in ho], [os.path.join(a.train, f) for f in tr]
hc, tc, hd, td = clip_img(hp), clip_img(tp), dino_img(hp), dino_img(tp)
Sc, Sd = hc @ tc.T, hd @ td.T
rows = []
for i, f in enumerate(ho):
    top = Sd[i].topk(min(3, len(tr))).indices.tolist()
    rows.append([(hp[i], f)] + [(tp[j], f"{tr[j]} dino={Sd[i,j]:.2f} clip={Sc[i,j]:.2f}") for j in top])
    print(f, [(tr[j], round(float(Sd[i, j]), 2), round(float(Sc[i, j]), 2)) for j in top])
sheet(rows, a.out)
print("contact sheet written to", a.out, "(holdout | 3 nearest train). LOOK AT ALL ROWS.")

# Within-train near duplicates (overweight one painting)
S = td @ td.T
pairs = [(tr[i], tr[j], round(float(S[i, j]), 2)) for i in range(len(tr)) for j in range(i + 1, len(tr)) if S[i, j] >= 0.85]
print("within-train pairs with DINO >= 0.85:", pairs)
