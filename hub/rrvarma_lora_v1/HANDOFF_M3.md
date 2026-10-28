# Milestone 3 Handoff: Raja Ravi Varma LoRA

## Generation Settings
* **Trigger Phrase:** `rrvarma style`
* **Recommended LoRA Scale:** 0.8
* **Sampling Settings:** DPM++ sampler, 30 steps, CFG 7.0
* **Negative Prompt:** (Use standard Stable Diffusion 1.5 negative prompts, such as "ugly, blurry, poorly drawn, bad anatomy")

## Model Capabilities & Limitations
* **Architecture & Knowledge:** The LoRA adapts the UNet's attention layers only, keeping the text encoder frozen. Because the training captions contained no character names, the model does not know specific identities like "Yudhishthira" or "Karna". Milestone 3's prompt expander must turn names into visual descriptions (e.g., clothing, pose, setting) along with the trigger phrase. Refer to `NAMES_TEST.md` and `VISUAL_VOCAB.md` for guidance.
* **77-Token Limit:** CLIP's text encoder cuts prompts at 77 tokens. LLM-expanded prompts must stay under this limit (with the trigger first), or Milestone 3 will need to implement a long-prompt technique.
* **Milestone 4 Compatibility:** Since the model uses the SD1.5 base, standard ControlNet 1.x models (such as OpenPose and Canny) are natively compatible for future pipeline integration.
* **Safety Checker:** The safety checker was disabled for evaluation because it falsely blocks classical paintings. Milestone 3 and 4's UI should decide its own safety policy.
* **Known Weak Spots:** Faces and hands are heavily distorted in essentially every generated image (e.g., asymmetric eyes, malformed fingers). Milestone 4's pipeline will likely need to rely on pose control (ControlNet) to assist with body structure and may require a dedicated face-restoration step.