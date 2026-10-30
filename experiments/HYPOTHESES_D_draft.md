# Section D: Expander (owner D), DRAFT

> **Draft.** This is D's section of the shared `experiments/HYPOTHESES.md` (Overview §5.2.5). It is pasted into that file by PR once A creates it. The section counts as pre-registered only from the **git timestamp of that merge**, which must come **before D's first scored run** (the first `results/c2_template@rrv_v1_dev.json`). Nothing below may be edited after results exist; changes go under **Deviations** as dated lines.

## D.0 Common setup (applies to every hypothesis below)

| Item | Value |
|---|---|
| Data | Dev set `eval/test_scenes_dev.json`: 32 scenes (8 per member), tiers 1-5. Seeds 1-2 for automated scores; human ratings use seed 1 only (Overview §10). Held-out (12 scenes, seeds 1-4) is run by B after the S3 freeze and is **reported, never used to choose** |
| Model | `rrv_v1` (M2 winner `D_lr2e-4` step 800, LoRA scale from `config/m3.json`) for every condition. `@rrv_names` comparisons belong to C |
| Unit of analysis | The **scene**. Automated scores are averaged over seeds within a scene; ratings are averaged over raters within a scene |
| Primary metric | **Human scene faithfulness** (1-5, ≥ 2 raters, blind codes) |
| Supporting metric | Checklist judge score (`checklist` in `results/`: weighted mean P(yes)), interpreted only to the extent B's judge-validation result in section B allows |
| Secondary metrics | `clip_raw`, style rating, `style_margin`, visual quality rating, failure-taxonomy tags, `max_train_dino` p95 (guardrail), expansion process metrics (D.7) |
| Comparison | B's `eval/stats.py` `paired(a, b)`: per-scene difference b − a, its mean, a **paired bootstrap 95% CI over scenes** (5000 resamples, seed 0), and **win/tie/loss** counts |
| Tie bands | As fixed in section B: 0.25 for ratings; M2's `tie_band` (0.0295) for automated scores |
| Decision rule | **Supported**: mean difference in the predicted direction and the 95% CI excludes 0. **No clear difference**: the CI includes 0. **Contradicted**: the CI excludes 0 in the opposite direction. No significance tests, no p-values (Overview §10) |
| Results files | `results/<condition>@rrv_v1_dev.json` (automated), B's rating tables in `eval/EVAL_REPORT.md` (human) |
| Expansion files | `outputs/expansions/<condition>_dev.json`, produced from the committed `expander/cache.json` |

Conditions are named with their exact IDs. `prompt_version` is `v1` for every LLM condition unless stated otherwise; the KB is whatever `kb_version` C ships at S2 (every record carries it). Only images generated from the **real KB** (after S2) are scored for these hypotheses; stub-KB runs are previews.

---

## H-D1 (core): the LLM beats the template

**Hypothesis.** `c3_llm_kb` is more faithful than `c2_template` (the LLM adds composition the template lacks).

- **Conditions:** `c2_template@rrv_v1` (a) vs `c3_llm_kb@rrv_v1` (b), all 32 dev scenes.
- **Primary:** human faithfulness, `paired(c2, c3)`; prediction: diff > 0.
- **Supporting:** checklist judge, same pairing. **Secondary:** `clip_raw`; style rating and `style_margin` (prediction: no clear difference); failure tags "wrong action" and "wrong figure count" (prediction: fewer in c3).
- **Decision:** the D.0 rule on the primary metric. If the judge disagrees with the humans, the human result stands and the disagreement is reported.
- **Null result:** the CI includes 0. Reported as "no clear benefit of the LLM over a deterministic template at n = 32", which is a valid answer to the ladder question (Overview §14: never cut).
- **Will NOT claim:** that LLMs in general beat templates; anything about the held-out set before B's Day-5 run; a gain on any tier subset unless it is listed here (per-tier numbers are descriptive).

## H-D2 (core): the KB matters most for rare characters

**Hypothesis.** `c3_llm_kb` beats `c4_llm_nokb`, and the gain is concentrated on **tier 4** (rare or absent characters, spelling variants), where the LLM's own knowledge is weakest.

- **Conditions:** `c4_llm_nokb@rrv_v1` (a) vs `c3_llm_kb@rrv_v1` (b). The only difference between them is the `Reference descriptions` block in the user message.
- **Primary:** human faithfulness. Two numbers, both from `paired(c4, c3)`:
  1. the all-scene difference (prediction > 0);
  2. the tier-4 difference minus the tiers 1-3 difference, with a bootstrap CI that resamples scenes **within** each stratum (prediction > 0).
- **Supporting:** checklist judge, same split. **Secondary:** the failure tags "generic figure (name ignored)" and "wrong/missing figure identity" per tier; `clip_raw`; C's `coverage` of the scene's characters (`none`/`low` vs `medium`/`high`) as a second split, descriptive only.
- **Decision:** H-D2 is supported only if (2) is in the predicted direction with a CI excluding 0. If (1) holds but (2) does not, we report "the KB helps overall, not specifically on tier 4".
- **Null result:** either CI includes 0. With about 4-6 tier-4 scenes team-wide, the tier-4 CI is expected to be wide; a null here means "not resolvable at this n", **not** "the KB does not matter for rare characters".
- **Will NOT claim:** that the KB fixes names in general; any per-character effect; anything from tier 5 (vague texts often have no characters for the KB to describe).

## H-D3 (core if c5 runs; c5 itself is optional): layout phrases fix counts and poses

**Hypothesis.** `c5_llm_kb_layout` beats `c3_llm_kb` on **count and pose** checklist items in **tiers 2-3**.

- **Conditions:** `c3_llm_kb@rrv_v1` (a) vs `c5_llm_kb_layout@rrv_v1` (b). c5 = c3 plus a leading layout phrase built from the LLM's `layout` JSON (figure count, pose, left/right position). Runs only if c2-c4 are done by the Day 4 checkpoint (Overview §9.1).
- **Primary for this item-level hypothesis:** the judge's P(yes) averaged over the checklist items with `type` ∈ {`count`, `pose`}, per scene, on tier 2-3 scenes. Humans do not rate items, so the human check is:
  - **Supporting (human):** faithfulness on tier 2-3 scenes, plus the share of images tagged "wrong figure count" or "wrong action".
  - If B's judge **fails** validation (section B threshold), H-D3 is decided on the human measures alone, and that is stated.
- **Secondary:** all-scene human faithfulness (prediction: no clear difference); style rating (guard: c5 must not fall below c1's mean − 0.3); `token_count` and `phrases_dropped` (the layout phrase costs tokens and may push out a description).
- **Decision:** the D.0 rule on the primary measure.
- **Null result:** the CI includes 0, reported as "layout words did not measurably change counts or poses". That outcome matters for M4, because it points at ControlNet rather than prompt wording.
- **Will NOT claim:** spatial correctness (left/right) because no checklist item measures position reliably; anything on tiers 1, 4 or 5.

## H-D4 (optional, diagnostic): a larger LLM is not needed once the KB supplies the facts

**Hypothesis.** `c3_7b` (Qwen/Qwen2.5-7B-Instruct, 4-bit) shows **no clear faithfulness gain** over `c3_llm_kb` (Qwen/Qwen3-4B-Instruct-2507, 4-bit) and costs **≥ 2×** the LLM seconds per scene.

- **Conditions:** `c3_llm_kb@rrv_v1` (a) vs `c3_7b@rrv_v1` (b); same KB, same prompt version, same `rrv_v1`.
- **Primary:** human faithfulness, `paired(c3, c3_7b)`. **Supporting:** checklist judge. **Cost:** mean and median `latency_s` per scene from the two expansion files (same GPU type, cache disabled for the timing run), plus JSON-failure (`json_retry`), fallback and pre-repair leak rates.
- **Decision:** "supported" if the faithfulness CI includes 0 **and** the latency ratio is ≥ 2. If c3_7b is clearly better (CI excludes 0), we report the gain together with its cost.
- **Null result:** the prediction is itself a "no clear difference". Such a result is **not evidence of equivalence**; we will say "no clear difference at n = 32".
- **Will NOT claim:** that model size never matters; anything about models other than these two checkpoints. **Eligibility:** `c3_7b` is diagnostic and **not eligible as the default pipeline** unless the team adds it to Overview §10's simplicity order before S3.

## H-D5 (optional, diagnostic): scene negatives reduce anachronisms

**Hypothesis.** `c3_neg` (c3 plus 3-6 LLM-written scene-specific negative phrases appended to the config negative) produces **fewer anachronism failures** than `c3_llm_kb`, with no loss of faithfulness or style.

- **Conditions:** `c3_llm_kb@rrv_v1` (a) vs `c3_neg@rrv_v1` (b); positives identical by construction (same cache entries), only the negative differs.
- **Primary:** the per-scene count of seed-1 images with an **anachronism failure**, defined before rating as: a rating tagged "wrong setting" or "other" whose free text mentions modern, Western or European clothing, armour or architecture. B codes these from the free text **before** the blind key is revealed.
- **Supporting:** human faithfulness (prediction: no clear difference); judge `setting` and `attire` items.
- **Secondary:** style rating and `style_margin`, because negatives such as "european" could push style; `negatives_added` content, checked for names.
- **Decision:** the D.0 rule on the anachronism count (b − a, prediction < 0).
- **Null result:** the CI includes 0. If c3 itself has almost no anachronism failures (for example ≤ 2 of 32 images), H-D5 is reported as **untestable on this set** rather than as null.
- **Will NOT claim:** a general anachronism fix; any effect on faithfulness beyond "no clear difference". **Eligibility:** diagnostic and not eligible as default, as for H-D4.

## H-D6 (optional): prompt v2 ordered by SHAP value per token

**Hypothesis.** Ordering phrases by A's SHAP **value per token** (drop the lowest-value phrases first) improves faithfulness on the scenes where v1 had to drop phrases for the token budget.

- **Conditions:** `c3_llm_kb@rrv_v1` (prompt `v1`) (a) vs `c3_v2@rrv_v1` (prompt `v2`, the proposed condition ID; same model, same KB) (b).
- **Primary:** human faithfulness on the subset of scenes where the v1 record has a non-empty `phrases_dropped`. Scenes whose v1 and v2 `positive` strings are identical produce identical images with the same seed; they are ties by construction, so we exclude them and report how many there were.
- **Supporting:** checklist judge on the same subset. **Secondary:** all-scene faithfulness (prediction: no clear difference), style rating (guard), `phrases_dropped` counts.
- **Decision:** the D.0 rule on the subset. If the subset has fewer than 5 scenes, H-D6 is reported as **descriptive only**.
- **Null result:** the CI includes 0. Reported as "SHAP ordering did not change faithfulness measurably".
- **Will NOT claim:** that SHAP values transfer to other prompt formats; A's values describe this format on 8 scenes.
- **Note:** A's 8 SHAP scenes are dev scenes, so v2 is tuned on dev. That is allowed (Overview §13.3); only B's held-out run can show whether it generalises. If v2 is adopted, the frozen `prompt_version` is posted in the Status issue at S3, and the change is recorded below as a dated deviation. H-D1 and H-D2 stay reported on v1, the pre-registered version.

---

## D.7 Expander validity targets (process; checked by `expander/check_expansions.py`)

Per expansion file `outputs/expansions/<condition>_<split>.json`, for `c2_template`, `c3_llm_kb`, `c4_llm_nokb` and every optional D condition:

| Target | Definition | Threshold |
|---|---|---|
| Valid records | `trigger_first` is true **and** `token_count ≤ max_prompt_tokens` (70, counted **excluding** start/end tokens with `common/tokens.py`) | **≥ 95%** of records |
| Names after repair | `names_after_repair == []`, and a re-scan of `positive` minus trigger and tail finds no name from `kb/names.json` (or the fallback list) | **zero** across the file |
| Leak rate before repair | share of records with a non-empty `names_leaked` | **reported**, no threshold. Expectation (descriptive, not a test): c4 leaks more often than c3, because c3 can copy the name-free KB wording |
| Fallback rate | share of records with `source == "fallback"` | reported, no threshold (D's guide only requires that the pipeline never crashes); each fallback is listed with its `error` |
| Latency | mean and median `latency_s` | reported in `EXPANDER.md` |

A file that misses a threshold is fixed **before** its images are scored. If that is impossible before the Day-4 10 am deadline, the miss is reported next to the results and recorded as a deviation.

## Deviations

Rule: after results exist, the text above is **never edited**. Each change gets one dated line here, in the form `YYYY-MM-DD: <what changed>, <why>, <which results it affects>`. Lines are appended and never removed.

<!-- no deviations yet -->
