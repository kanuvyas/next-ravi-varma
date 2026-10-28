import os, torch
from PIL import Image
from torchvision import transforms
from diffusers import DDPMScheduler
from generate import BASE

@torch.no_grad()
def denoise_loss(pipe, img_dir, text_by_file, timesteps=(50, 150, 300, 450, 600, 750, 900), n_noise=2):
    """Mean epsilon-prediction MSE on fixed noise + fixed timesteps. Compare only like-for-like text."""
    sched = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler")
    tf = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512),
                             transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    losses = []
    for f, text in text_by_file.items():
        x = tf(Image.open(os.path.join(img_dir, f)).convert("RGB")).unsqueeze(0).to("cuda", torch.float16)
        lat = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
        ids = pipe.tokenizer(text, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to("cuda")
        emb = pipe.text_encoder(ids)[0]
        for k in range(n_noise):
            g = torch.Generator("cuda").manual_seed(k)
            noise = torch.randn(lat.shape, generator=g, device="cuda", dtype=lat.dtype)
            for t in timesteps:
                tt = torch.tensor([t], device="cuda")
                pred = pipe.unet(sched.add_noise(lat, noise, tt), tt, encoder_hidden_states=emb).sample
                losses.append(torch.nn.functional.mse_loss(pred.float(), noise.float()).item())
    return sum(losses) / len(losses)
