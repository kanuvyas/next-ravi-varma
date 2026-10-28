# Dataset card: rrvarma train_v1 / train_v2

**Images:** 147 paintings by Raja Ravi Varma (M1 collected set: 149; dropped after M1: 2). Sources and authorship flags: see the M1 provenance log. Authorship-flagged images: Two paintings were flagged for uncertain authorship (`varma_sankaracharya_a63`, `varma_sankaracharya_b28`); copies were placed in a `flagged_authorship` folder, and both remain in the training set (`train_v1`).
**Holdout:** 15 paintings kept out of training (eval_holdout).

## Preprocessing
Rebuilt from raw originals in `m1/ravi_varma_raw` where available (147 of 147; 0 used M1's already-cropped PNG). EXIF-corrected, RGB, square crop biased toward the top for tall paintings, resized to 512x512 (LANCZOS).
Heavy-crop review: 55 accepted as-is, 0 manual crops, 0 padded, 2 dropped. Decisions: `dataset/decisions/`.
Sources with short side < 768 px: 0.

## Captions
Base: M1 final captions (BLIP-2 draft, hand-corrected; no character or epic names in training captions; names live in scene_metadata only).
- **train_v1:** `rrvarma style, <description>`
- **train_v2:** `rrvarma style, <description>, oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry`
Both variants share the same description, so the only difference is the constant style suffix. Old `<ravivarma_style>` token removed.
Length control: CLIP limit 77 tokens; descriptions trimmed at comma boundaries so that the v2 caption fits (31 trimmed, 27 hand-edited; log in `decisions/caption_trim_log.json`). Global typo fixes: [('holdig', 'holding')].

## Checks
v1/v2 identical file lists; no holdout stems in training; leak check run with `kit/leak_check.py`: two images (`varma_ladies_moonlight_a01`, `varma_murugan_a62`) were dropped; one flagged pair (`kadambari_sitar_a51` vs `b04`) was inspected and kept; name lint clean.
