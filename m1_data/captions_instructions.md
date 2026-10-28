# Caption schema and review instructions (Milestone 1)

> Converted from the team's Google Doc "captions". Formatting is approximate; wording is kept as written. The old trigger token `<ravivarma_style>` shown here was later replaced by `rrvarma style` in Milestone 2 (see `m2_training/dataset/DATASET_CARD.md`).

## Step 1: Run the setup cells you already have

- Run the install cell and the "Load BLIP-2" cell.
- **Skip the fallback model cell** unless you hit a CUDA out-of-memory error. It replaces BLIP-2 with a different model.
- Upload a few of your sample images (`varma_...`) and run the `caption_one` cell and the `test_captions` cell, so `test_captions` is filled in.

## Step 2: Replace the schema cell

Delete the cell between the two `=== EDIT ===` markers (the one with `TRIGGER_TOKEN`, `wrap_caption` and `blank_metadata`) and use this instead:

```python
import re

# === CAPTION SCHEMA ===
TRIGGER_TOKEN = "<ravivarma_style>"
STYLE_SUFFIX = "oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry"

def clean_raw(raw):
    raw = raw.strip().lower().rstrip(".")
    # BLIP often opens with these; "oil painting" is already in the suffix
    raw = re.sub(r"^(a|an|the)?\s*(oil\s+)?(painting|picture|image|photo|drawing)\s+(of|showing)\s+", "", raw)
    return raw

def wrap_caption(raw_caption):
    return f"{TRIGGER_TOKEN}, {clean_raw(raw_caption)}, {STYLE_SUFFIX}"

def blank_metadata(fname=""):
    m = re.search(r"_([abcd])\d+", fname)  # collector letter is in the filename
    collector = f"member_{m.group(1)}" if m else ""
    return {"epic": "", "characters": [], "scene": "", "source_collector": collector}
# === END ===
```

## Step 3: Test it

Run this and **send C the printed output back**:

```python
for fname, raw in test_captions.items():
    print(fname)
    print("  training_caption:", wrap_caption(raw))
    print("  scene_metadata:  ", blank_metadata(fname))
```

C will check that the captions look right before D runs the whole dataset.

## Step 4: Before the full run

In the full captioning cell, make two edits:

- Change `blank_metadata()` to `blank_metadata(fname)`.
- Set `img_dir` to the cleaned folder (`ravi_varma_clean`). Only run it once that folder exists and the eval holdout images have been removed from it.

## Rules for the format

- The trigger token always comes first.
- `training_caption` never names an epic or a character. That goes in `scene_metadata`.
- `scene_metadata` stays blank when captioning. We fill it in during review.

---

Before you send it, confirm that you're happy with the trigger token `<ravivarma_style>` and the suffix wording. Once it goes into LoRA training, changing them means redoing every caption.

## The order of who does what

1. C sends the schema code (the message above).
2. D runs BLIP-2 on the cleaned images, and the code writes `final_captions.json`.
3. D splits it into review batches, one per person.
4. Each member corrects their batch by hand, checking each caption against its image and filling in `scene_metadata`.
5. D merges everyone's corrected batches into the final file.

## Why names stay out of `training_caption`

- We have about 100-150 images across many epics and characters. That's only a handful of examples per character, which isn't enough for the LoRA to reliably learn "this is Shakuntala." It would pick up weak, noisy associations.
- What the LoRA can learn well from that many images is the Ravi Varma look: the soft skin, the drapery, the palette, the composition.
- Scene knowledge doesn't get lost. It goes into `scene_metadata`, and Milestone 3 uses it to expand a user's prompt with the right attire, weapons and setting. The base model already knows roughly what a "woman under a tree" looks like, and the LoRA supplies the style.

If BLIP-2 gives something like "a woman holding an object," the caption is too vague even without names, and that does hurt training. The model needs the caption to describe what's actually in the image, so it can link words to visuals.

That's what your review step should fix. When you correct a caption, add concrete visual details without naming the character:

- **Weak:** a woman holding an object
- **Better:** a woman in a white sari playing a veena, seated on a lotus, holding a book and a string of beads

That describes the Saraswati painting well without saying "Saraswati". The name goes in `scene_metadata`. So when you review, aim for captions that are visually specific but name-free. If the test output from Member D shows BLIP-2 is too vague on most images, tell her, and we can look at ways to get richer raw captions, such as a longer output limit or a prompt.

---

## Sample test output (as pasted in the doc)

```
varma_ayilyam_thirunal_rama_varma_d44.png
  training_caption: <ravivarma_style>, a man in a black outfit with a crown on his head, oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry
  scene_metadata:   {'epic': '', 'characters': [], 'scene': '', 'source_collector': 'member_d'}

varma_jatayu_vadha_d01.jpg
  training_caption: <ravivarma_style>, a man flying with an eagle, oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry
  scene_metadata:   {'epic': '', 'characters': [], 'scene': '', 'source_collector': 'member_d'}

varma_arjuna_subhadra_b03.png
  training_caption: <ravivarma_style>, krishna and lakshmi painting, oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry
  scene_metadata:   {'epic': '', 'characters': [], 'scene': '', 'source_collector': 'member_b'}

varma_begums_bath_b21.png
  training_caption: <ravivarma_style>, three women in traditional clothing, oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry
  scene_metadata:   {'epic': '', 'characters': [], 'scene': '', 'source_collector': 'member_b'}

varma_bharata_bhet_d26.png
  training_caption: <ravivarma_style>, krishna and his wife krishna, oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry
  scene_metadata:   {'epic': '', 'characters': [], 'scene': '', 'source_collector': 'member_d'}

varma_bheeshma_oath_a50.png
  training_caption: <ravivarma_style>, a woman and a man in indian clothing, oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry
  scene_metadata:   {'epic': '', 'characters': [], 'scene': '', 'source_collector': 'member_a'}
```

**To be fixed manually later:**

```
varma_bharani_nal_raja_raja_varma_d31.png
  training_caption: <ravivarma_style>, an old man in a red robe, oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry
  scene_metadata:   {'epic': '', 'characters': [], 'scene': '', 'source_collector': 'member_d'}

varma_ambika_d24.png
  training_caption: <ravivarma_style>, the goddess of the indian goddesses, oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry
  scene_metadata:   {'epic': '', 'characters': [], 'scene': '', 'source_collector': 'member_d'}

varma_bharani_parvathi_bayi_b18.png
  training_caption: <ravivarma_style>, a woman in a white dress and gold jewelry, oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry
  scene_metadata:   {'epic': '', 'characters': [], 'scene': '', 'source_collector': 'member_b'}

varma_ashtabujadevi_c18.png
  training_caption: <ravivarma_style>, a painting depicting the hindu goddess durga and her lion, oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry
  scene_metadata:   {'epic': '', 'characters': [], 'scene': '', 'source_collector': 'member_c'}
```

---

## Manual editing for scene metadata

### Setup

1. Member D splits `final_captions.json` into `review_batch_member_a.json`, `_b.json`, `_c.json`, `_d.json` and sends each person theirs.
2. Everyone also has access to `ravi_varma_clean/` (the actual images), since the JSON only has filenames, not pictures.

### Doing your batch

1. Open your file in a text editor (VS Code, or even Notepad/TextEdit).
2. Open `ravi_varma_clean/` in a file browser or image viewer, sorted by name if possible, so you can match filenames to images quickly.
3. Go entry by entry:
   - Find the image matching the filename key.
   - Read the current `training_caption`. Look at the picture and ask: is this actually what's happening? Is it specific enough?
   - If it's wrong, vague, or has a leftover name/garbled text (like "a figure and a figure"), rewrite it directly in the JSON, describing what you see, no epic/character names.
   - Fill in `scene_metadata`: epic, characters (as a list), scene. If you genuinely don't recognize the scene, leave it blank rather than guess, and note it separately to raise with the group.
4. Save the file after you finish (or periodically, so you don't lose work).

### Example

Before:

```json
"varma_arjuna_subhadra_b03.png": {
  "training_caption": "<ravivarma_style>, a figure and a figure, oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry",
  "scene_metadata": {"epic": "", "characters": [], "scene": "", "source_collector": "member_b"}
}
```

After:

```json
"varma_arjuna_subhadra_b03.png": {
  "training_caption": "<ravivarma_style>, a man and a woman standing together in a palace setting, he in warrior attire and she in a red saree with gold jewelry, oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry",
  "scene_metadata": {"epic": "Mahabharata", "characters": ["Arjuna", "Subhadra"], "scene": "Arjuna and Subhadra together after their marriage", "source_collector": "member_b"}
}
```

Hand your corrected file back to Member D so she can merge all four batches into the final `final_captions.json`.

**Note:** if the scene is not from an epic (Ramayana/Mahabharata), leave the `epic` part blank. This includes portraits and individual paintings of gods/goddesses that are not from these epics.
