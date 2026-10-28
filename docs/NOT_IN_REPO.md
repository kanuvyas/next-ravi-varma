# What was left out when the Drive folder was copied here

The team's Drive folder (`AI PROJECT`) was copied to this repository by script. Everything below was deliberately skipped or could not be copied. Nothing here changes the reported results; the weights and numbers they rest on are in `bundles/` and `hub/`.

| Left out | Files | Why |
|---|---|---|
| Raw full-size scans | 164 | About 4 GB, single files up to 500 MB (GitHub limit is 100 MB), museum reuse terms |
| Regenerable evaluation images | 2,964 | About 1.3 GB, can be rebuilt with `m2_training/kit/eval_run.py`; the scores and grids are kept |
| Training run folders | 412 | Optimizer state and smoke tests; the weights are in `bundles/` |
| TensorBoard event files | 55 | `loss_log.csv` in each bundle holds the same numbers |
| Milestone 1 image copies | 149 | Superseded by `m2_training/dataset/train_v1` |
| `train_v2` images | 149 | Identical copies of `train_v1` (the caption files are kept) |
| Duplicates of `validation_samples` and of `hub/` files, and a zip of `hub/` | 21 | Already present elsewhere in the repo |
| Python cache | 7 | Not needed |
| Google-Sheet copy of `control_art_log.csv` | 1 | The CSV is in the repo |
| Personal member folders | 2 | Not part of the submission |

## Google Docs and Sheets

Google-native files cannot be copied as files, so they were converted to text:

- `#READ- TEAM_DECISIONS` is `docs/TEAM_DECISIONS_M1.md`, with the link to the DATA sheet removed.
- `captions` is `m1_data/captions_instructions.md`.
- `DATA` is `m1_data/DATA_provenance_Sheet1.csv`. Only its first tab (Sheet1, 169 rows) was converted. The second tab, Sheet5 ("no duplicates as of 19th September"), is not included.
- `dropped_images` is `m1_data/dropped_images.csv` (header only, no rows listed).
- Bundle A's `NOTES.md` is not included.
