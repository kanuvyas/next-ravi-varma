# `expander/_shims/`: temporary placeholders

These files stand in for files that **A, C and the team own**, so the expander can be built and tested
before those files exist. They also serve as **proposals** to their owners ("use if helpful"). D removes
this folder at **S2**, once the real files have landed.

| Shim | Stands in for (owner) | Used when |
|---|---|---|
| `m3.default.json` | `config/m3.json` (team; PR + `contracts/CHANGELOG.md`) | `config/m3.json` is missing and `$M3_CONFIG` is unset |
| `tokens.py` | `common/tokens.py` (A) | `from common.tokens import n_tokens` fails |
| `stub_parser.py`, `stub_kb.json` | `kb/stub_parser.py`, `kb/stub_kb.json` (C) | neither `kb.parser` nor `kb.stub_parser` imports |
| `fixtures_scenes.json` | `fixtures/` scenes (A) | tests and offline runs only |

The lookup order lives in `expander/_cfg.py`. Each lookup is one small function, so the S2 swap is a
one-line change there. Nothing outside `expander/_cfg.py` imports from this folder.

Notes for the owners:
- `m3.default.json`: the `style_tail` needs team confirmation on Day 1 (see `_style_tail_note`); `rrv_v1.scale`
  is 0.8 (M2 handoff), not the 0.0 placeholder.
- `tokens.py`: same counter as A's file (CLIP BPE, start/end tokens excluded), but `transformers` is
  imported lazily, so the module imports in CI.
- `stub_parser.py`: C's signature `parse_scene(text, scene_id="", kb_path=None)`, exact `\b` alias match,
  longest first, consuming matches. No fuzzy matching.
