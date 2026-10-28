import os
from PIL import Image
from generate import load, gen, grid, set_scale

SCALES = [0.0, 0.4, 0.6, 0.8, 1.0, 1.2]       # 0.0 = plain base (sanity check); 1.2 shows overshoot
PROMPTS = [                                    # 2 simple, 1 multi-figure
    "rrvarma style, a woman in a silk sari standing in a palace hall",
    "rrvarma style, two figures conversing beside a pillar",
    "rrvarma style, a king seated on a throne surrounded by courtiers",
]

def sweep(lora_dir, out_dir, seeds=(1, 2)):
    pipe = load(lora_dir, weight=1.0); os.makedirs(out_dir, exist_ok=True); paths = {}
    for pi, p in enumerate(PROMPTS):
        for s in SCALES:
            set_scale(pipe, s)
            for seed in seeds:
                path = f"{out_dir}/p{pi:02d}_scale{s}_s{seed}.png"
                if not os.path.exists(path): gen(pipe, p, seed).save(path)
                paths[(pi, s, seed)] = path
    imgs = [Image.open(paths[(pi, s, seeds[0])]) for pi in range(len(PROMPTS)) for s in SCALES]
    grid(imgs, rows=len(PROMPTS), cols=len(SCALES)).save(f"{out_dir}/grid.png")   # rows = prompts, cols = scale
    return paths
