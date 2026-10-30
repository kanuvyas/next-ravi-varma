# Expander (owner D)

`expand(text, scene=None, mode="llm_kb") -> ExpandedPrompt` turns a user's epic-scene sentence into a prompt the M2 model (`rrv_v1`) can render: name-free visual phrases, the trigger first, the style tail last, under the token limit. It runs a **local** instruct LLM (no paid API) with a deterministic template as the fallback.

> **Status: skeleton (Day 1).** Design sections are written. Everything marked `TODO(results)` waits for measured numbers from `outputs/expansions/`, `results/` or B's tables; no number goes in by hand. Contracts referred to: Overview v2 §6 (`config/m3.json`, SceneSpec, ExpandedPrompt, signatures, CLI).

## 1. Design principle

```
<trigger>, <scene phrase 1>, <scene phrase 2>, …, <style_tail>
```

The LLM writes **only the scene phrases**, ordered from most to least important. The **code** adds the trigger (always first) and the style tail (always last), scrubs names, and fits the phrases under the budget. The LLM therefore cannot drop the trigger, leak a name into the final prompt, or exceed the budget. Every record carries `prompt_version`, `kb_version` and `llm_model`, so any expansion can be traced and reproduced from the cache.

## 2. Modes and conditions

| Mode (`expand`) | Condition ID | What the model sees | `source` | `prompt_version` | Run by |
|---|---|---|---|---|---|
| `raw` | `c1_lora_raw` | `trigger, <raw text>, style_tail`: no fitting, no scrub (parity with A's c1 string; names recorded only) | `template` (+ `note: "raw"`) | `raw` | A |
| `template` | `c2_template` | SceneSpec → character `visual`s, then `action`, `setting.visual` and the mood phrase; no LLM | `template` | `tpl1` | D |
| `llm_kb` | `c3_llm_kb` | LLM phrases, user message **with** the `Reference descriptions` block | `llm` / `fallback` | `v1` | D |
| `llm_nokb` | `c4_llm_nokb` | LLM phrases, user message **without** that block (the only c3/c4 difference) | `llm` / `fallback` | `v1` | D |
| `llm_kb_layout` | `c5_llm_kb_layout` (optional) | c3 plus a leading layout phrase from the reply's `layout` JSON | `llm` / `fallback` | `v1_layout` | D |
| `llm_kb` + `--kb kb/kb_llm.json` | `c6_llm_kbllm` | c3 with LLM-written KB descriptions | `llm` / `fallback` | `v1` | C |
| `llm_kb` + `--llm_model Qwen/Qwen2.5-7B-Instruct` | `c3_7b` (optional) | c3 with the larger LLM | `llm` / `fallback` | `v1` | D |
| `llm_kb` + `--scene_negatives` | `c3_neg` (optional) | c3 positive; negative = config negative + 3-6 LLM scene negatives | `llm` / `fallback` | `v1_neg` | D |
| `llm_kb` + `--prompt_version v2` | `c3_v2` (optional; **proposed ID**) | c3 with prompt v2 (SHAP value-per-token ordering) | `llm` / `fallback` | `v2` | D |

`batch_expand.py` sets `condition` from `--condition`, otherwise from the `--out` file name (`<condition>_<split>.json`), otherwise from the mode. Hypotheses for these comparisons: `experiments/HYPOTHESES.md`, section D.

## 3. System prompt v1 (verbatim; `expander/prompts/system_v1.txt`)

```
You write scene descriptions for a text-to-image model trained on 19th-century Indian oil paintings.
Return ONLY valid JSON: {"phrases": ["...", "..."]}
Rules:
- 6 to 12 phrases, each 2 to 6 words, ordered from MOST to LEAST important to the scene.
- NEVER use proper names of people, gods or places. The model does not know names. Describe by
  clothing, ornaments, pose, expression and setting instead.
- If reference descriptions are given, use their wording for the matching figures.
- Cover, in roughly this order: the main figures (attire), what they are doing (pose, gesture),
  the setting, the lighting and mood.
- Do NOT mention style, painting, canvas or artist; that is added separately.
- Use concrete visual words (e.g. "kneeling", "gilded armour", "pillared hall").
```

**User message:** the raw text, plus, in `llm_kb` mode only, `Reference descriptions:` followed by one `<mention> -> <visual>` line per character from the SceneSpec. Variants (`v1_layout`, `v1_neg`, `v2`) are separate files with their own `prompt_version`; v1 is never edited. If JSON failures are frequent, a worked example from a **dev** scene (never held-out) may be added, but only as a new version. `TODO(results)`: the variant texts and the change log of v2.

## 4. Fallback chain (the pipeline never crashes because of the LLM)

1. LLM call (cached) → parse JSON `{"phrases": [...]}` (tolerant: strips code fences, takes the first `{…}`).
2. Parse failure → **one** retry with "Return ONLY valid JSON" appended (cached; `json_retry: true`).
3. Second failure **or any exception** (load error, OOM, timeout) → **template mode** on the same SceneSpec, with `source: "fallback"` and `error` set.
4. Name leak → **one** re-ask: "remove proper names; describe by clothing, pose and setting" (cached; `name_repair: true`). Names still present → drop the offending phrases (`phrases_scrubbed`).
5. `assemble` (§5), then a last re-scan of the final `positive`, excluding trigger and tail. Any hit drops that phrase and the prompt is re-assembled.

## 5. Token fitting

- **Budget:** `max_prompt_tokens` = **70**, counted **excluding** the start/end tokens with the team's one counter `common/tokens.py` `n_tokens(text)` (CLIP BPE of the base model's tokenizer). It stays 70 until A's `pipeline/TOKEN_BUDGET.md` says otherwise. The M2 helper `generate.py n_tokens` counts **including** start/end tokens and is never used here.
- **`assemble(trigger, phrases, style_tail, max_tokens, *, counter=None)`:** adds phrases greedily in priority order and skips any phrase that does not fit, so a shorter later phrase can still be kept. Trigger and tail are always kept, and an empty tail adds no trailing comma. `phrases_dropped` lists **budget drops only**; name removals go to `phrases_scrubbed`.
- **Fixed cost (prep measurement, re-checked by the slow tokenizer test):** trigger 4 tokens, proposed tail 20, `"<trigger>, <tail>"` 25, so about 44 tokens remain for scene phrases. `TODO(results)`: distribution of `token_count` and of the number of dropped phrases per condition (from `check_expansions.py`).

## 6. Name scrub

- **Why:** the LoRA does not know names (M2 `NAMES_TEST.md`), and A's CI fails if a KB alias appears in any `positive` of c2-c6.
- **Name list:** C's `kb/names.json` when present. Until then, the stub KB names together with `expander/names_seed.json`: 190 canonical names and 350 aliases from M1 `scene_metadata` plus the major epic figures and places.
- **Matcher** (`expander/names.py`, shared with A's CI so the two cannot disagree): text normalised with NFKD to ASCII; whole words (`\b…\b`, multi-word aliases joined by `\s+`, longest first); **case-insensitive by default**.
- **Case policy for English-word aliases:** aliases that are also ordinary words are matched **only when capitalised**: `Ram`, `Lava`, `Kush`/`Kusha`, `Bali`, `Kali`, `Tara`, `Kama`, `Hamsa`, `Uma`, `Hari`, `Agni`, `Surya`, `Usha`, `Matsya`, `Varaha`, … (full list in `names_seed_meta.json → risk.case_sensitive_only`). This keeps "a ram", "kusha grass" and "flowing lava" in the prompt. The LLM capitalises proper names. Honorifics and class nouns (`Devi`, `Sri`, `Rakshasa`, `Apsara`, `Gopi`) and `ashoka grove` are not names. In prep, the list had 0 false positives on 178 name-free M2 texts.
- **Record fields:** `names_leaked` = names found **before** repair (B reports how often the LLM leaks). `names_after_repair` = names left in the final phrases, which **must be `[]`**. `phrases_scrubbed` = phrases removed for names.
- **Known cost:** rivers and mounts that are also names (`Ganga`, `Yamuna`, `Sarayu`, `Nandi`, `Garuda`) are scrubbed, which can remove a setting phrase. `TODO(results)`: how often this happened.
- **Template mode:** there is no LLM to re-ask, so the matched name is deleted from the phrase, dangling connectors are tidied, and the phrase is dropped if fewer than 2 content words remain.

## 7. Cache

- **File:** `expander/cache.json`, **committed for dev**, so anyone can reproduce every expansion without loading an LLM (Overview §13.1).
- **Key:** `sha256("prompt_version|model|kb_version|mode|user message")`, one entry per LLM call, retries and repair calls included. The key separates c3 from c4 (`mode`), c3 from c6 (`kb_version`) and v1 from v2 (`prompt_version`). Entry value: `{reply, created, model, prompt_version}`. Entries are only added, never overwritten. Temperature 0.
- **Held-out:** B runs with `--cache <private path>` outside the repo, so held-out text never enters the shared cache.

## 8. Configuration and open team decisions

D reads `trigger`, `style_tail`, `negative`, `max_prompt_tokens` and `llm.*` from `config/m3.json`; nothing is hardcoded. Until the team file exists, `expander/_shims/m3.default.json` holds the proposal below.

| Key | Proposed value | Status |
|---|---|---|
| `trigger` | `rrvarma style` | M2 handoff; agreed |
| `style_tail` | `oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry` (M2's frozen v2 suffix, `DATA_FROZEN.json`) | **OPEN.** The winner (`D_lr2e-4` step 800) was trained on **train_v1**, whose captions have **no fixed tail**, so the overview's definition ("the fixed tail used at the end of training captions") is empty for it. This suffix is the tail behind all M2 test-set numbers for the winner. Alternatives: `""`. A's token-budget run and SHAP `tail` player measure it; change by config PR only. It also changes c1 |
| `negative` | `lowres, blurry, deformed, extra limbs, bad anatomy, text, watermark` | **OPEN, recommended.** `generate.py`'s `NEG`, used for every M2 measurement. The handoff's "ugly, blurry, poorly drawn, bad anatomy" is only an example ("such as") |
| `max_prompt_tokens` | 70 (excluding start/end) | until `TOKEN_BUDGET.md` |
| `llm.model` | `Qwen/Qwen3-4B-Instruct-2507` (primary, Apache-2.0), 4-bit NF4, fp16 compute on T4 | **OPEN** until D's Day-1 T4 load and speed test; pin the revision |
| `llm.model_large` | `Qwen/Qwen2.5-7B-Instruct` (for `c3_7b`) | optional experiment |
| `llm.temperature` / `prompt_version` | `0.0` / `v1` | frozen version posted at S3 |

Other open items: scene ID block `s25`-`s32` for D's dev scenes (pending B's confirmation); `names_leaked` before repair plus the extra field `names_after_repair` (needs B's agreement, issue `needs:B`); A's CI importing `expander.names` (issue `needs:A`); the ID `c3_v2` for prompt v2.

## 9. Ladder results (c2 → c4 → c3)

`TODO(results)`: B's dev table for `c2_template`, `c4_llm_nokb`, `c3_llm_kb` @ `rrv_v1`: human faithfulness, style, quality; checklist judge; `clip_raw`; `style_margin`; paired diffs with 95% CI and win/tie/loss, per tier and seen/unseen. Sources: `results/c{2,3,4}_*@rrv_v1_dev.json`, `eval/EVAL_REPORT.md`. State H-D1/H-D2 outcomes with the pre-registered rule. Held-out rows are added after B's Day-5 run.

`TODO(results)`: process table per file from `check_expansions.py`: % valid, leaks before and after repair, fallback rate, mean `token_count`, mean number of dropped and scrubbed phrases.

## 10. Optional experiments

`TODO(results)`: c5 (H-D3), c3_7b (H-D4), c3_neg (H-D5), prompt v2 (H-D6), each with the result, or "not run" and the reason (Overview §14 cut order: negatives → LLM size → c5).

## 11. Failure modes with examples

`TODO(results)`: from B's Day-4 failure notes (`needs:D`) and `phrases_dropped`: for 3-5 bad images, the scene, the expansion, the image code and the diagnosed cause (figure phrase dropped by budget, KB description too vague, tail pushed out, name scrub removed a setting phrase, layout ignored). Include failures, not only successes.

## 12. Measured latency

`TODO(results)`: LLM load time, seconds per scene (mean/median `latency_s`) for each LLM model on a T4, peak VRAM from `python -m expander.llm --smoke`, and the template-mode time. Expansion runs alone in the `llm` session: expand everything, `close()`, then generate in a new `gen` session.

## 13. Commands

Dev (D, `llm` session):
```
python expander/batch_expand.py --scenes eval/test_scenes_dev.json --mode template --out outputs/expansions/c2_template_dev.json
python expander/batch_expand.py --scenes eval/test_scenes_dev.json --mode llm_kb   --out outputs/expansions/c3_llm_kb_dev.json
python expander/batch_expand.py --scenes eval/test_scenes_dev.json --mode llm_nokb --out outputs/expansions/c4_llm_nokb_dev.json
python expander/check_expansions.py outputs/expansions/c2_template_dev.json outputs/expansions/c3_llm_kb_dev.json outputs/expansions/c4_llm_nokb_dev.json
```

Held-out (B, Day 5, frozen S3 sha, private cache): `TODO(results)`: confirm the frozen sha, the final `prompt_version` and the exact flags at S3. Draft:
```
git checkout <S3 sha>
python expander/batch_expand.py --scenes <private>/test_scenes_heldout.json --mode llm_kb --split heldout \
    --cache <private>/heldout_cache.json --out outputs/expansions/c3_llm_kb_heldout.json
# same for --mode template (c2) and --mode llm_nokb (c4), plus any optional condition that ran on dev
```
