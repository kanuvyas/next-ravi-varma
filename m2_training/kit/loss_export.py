# usage: python loss_export.py RUN_DIR OUT_DIR N_TRAIN_IMAGES
# Writes OUT_DIR/loss_log.csv, loss_curve.png, validation_samples/
import io, os, glob, sys
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_run(run_dir):
    dirs = sorted({os.path.dirname(p) for p in glob.glob(f"{run_dir}/logs/**/events.out.tfevents*", recursive=True)})
    frames, images = [], []
    for d in dirs:
        ea = EventAccumulator(d, size_guidance={"images": 0}); ea.Reload(); tags = ea.Tags()
        print(d, "| scalars:", tags["scalars"], "| images:", tags["images"])
        if "train_loss" in tags["scalars"]:
            frames.append(pd.DataFrame([(e.step, e.value) for e in ea.Scalars("train_loss")], columns=["step", "loss"]))
        for t in tags["images"]:
            images += [(t, ev.step, ev.encoded_image_string) for ev in ea.Images(t)]
    df = pd.concat(frames).drop_duplicates("step", keep="last").sort_values("step").reset_index(drop=True)
    return df, images

def export(run_dir, out_dir, n_train_images, eff_batch=4):
    os.makedirs(out_dir, exist_ok=True)
    df, images = load_run(run_dir)
    df["epoch"] = df["step"] / (n_train_images / eff_batch)
    df["loss_smooth"] = df["loss"].ewm(alpha=0.05).mean()
    df.to_csv(f"{out_dir}/loss_log.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.plot(df["epoch"], df["loss"], alpha=0.25, label="raw"); plt.plot(df["epoch"], df["loss_smooth"], label="smoothed")
    plt.xlabel("epoch"); plt.ylabel("train_loss"); plt.title(os.path.basename(run_dir.rstrip("/"))); plt.legend()
    plt.savefig(f"{out_dir}/loss_curve.png", dpi=150, bbox_inches="tight"); plt.close()
    os.makedirs(f"{out_dir}/validation_samples", exist_ok=True)
    for t, step, blob in images:
        Image.open(io.BytesIO(blob)).save(f"{out_dir}/validation_samples/{t.replace('/', '_')}_epoch{step}.png")
    return df

if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2], int(sys.argv[3]))
