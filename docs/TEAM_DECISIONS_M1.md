# TEAM_DECISIONS: The Next Ravi Varma (Milestone 1)

> Converted from the team's Google Doc "#READ- TEAM_DECISIONS". Formatting is approximate; wording is unchanged.

## File naming convention

- Pattern: `varma_<subject>_<sourceletter><number>.jpg`
- Example: `varma_shakuntala_a01.jpg`
- `<sourceletter>` = a, b, c, or d based on which member collected it (lets anyone trace a file back to its collector without needing the log)
- `<subject>` = lowercase, underscore-separated (e.g. `saraswati`, `ravana_battle`)
- `<number>` = zero-padded 2-digit sequence per subject per collector (01, 02, ...)

## Resolution rules

- Minimum to keep: 768px on the short side. Discard anything smaller at collection time.
- Target training resolution: 512×512 (training SD1.5)

## Provenance log

- Format: Google Sheet, shared edit access to all 4 members
- Columns (in order): `filename | source_url | source_site | collector | license/rights_note | resolution | date_added`
- Rule: every collector adds a row **immediately** when they save a file, not batched at end of day
- Link: the team's Google Sheet named "DATA" (link omitted here because this repo is public; see `m1_data/`)

## Sign-off

- [ ] Member A
- [ ] Member B
- [ ] Member C
- [ ] Member D
