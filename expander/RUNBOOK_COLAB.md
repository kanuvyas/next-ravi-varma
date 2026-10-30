# Expander runbook: Colab, free T4 (owner D)

Copy-paste cells for the `llm` session. The local LLM, Stable Diffusion and the judge do not fit on a T4 together, so the **LLM runs alone**: expand everything, write the files, then generate in a **new** runtime (Overview §5.3). Expansions are reproducible from `expander/cache.json` without a GPU.

**Outputs (fixed names):** `outputs/expansions/<condition>_<split>.json`, e.g. `c2_template_dev.json`, `c3_llm_kb_dev.json`, `c4_llm_nokb_dev.json`; optional `c5_llm_kb_layout_dev.json`, `c3_7b_dev.json`. `batch_expand.py` takes the condition from this file name, so keep the names exact.

## 0. Runtime

*Runtime → Change runtime type → T4 GPU.* Start from a fresh runtime, with nothing else loaded.

## 1. Drive, clone, pull

```python
from google.colab import drive, userdata
import os, subprocess
drive.mount('/content/drive')
REPO = "<ORG>/<REPO>"          # e.g. the team's M3 repo
tok = userdata.get('GH_TOKEN')  # Colab Secrets; never print it
if not os.path.exists('/content/m3'):
    subprocess.run(["git", "clone", f"https://x-access-token:{tok}@github.com/{REPO}.git", "/content/m3"], check=True)
os.chdir('/content/m3'); subprocess.run(["git", "pull"], check=True)
# Optional, fewer HF rate limits: os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

## 2. Install (`llm` profile)

```python
import os
if os.path.exists("pipeline/setup_session.sh"):
    !bash pipeline/setup_session.sh llm        # A's script; it should install expander/requirements_llm.txt
else:
    !pip install -q -r expander/requirements_llm.txt
import sys; sys.path.insert(0, "/content/m3")  # the repo has no packaging: no `pip install -e .`
!nvidia-smi --query-gpu=name,memory.total --format=csv
!python -c "import torch, transformers, bitsandbytes, accelerate; print(torch.__version__, torch.version.cuda, transformers.__version__, bitsandbytes.__version__, accelerate.__version__)"
```

The scripts add the repo root to `sys.path` themselves, so always run them from `/content/m3`.

## 3. Smoke test (Day 1: pin the model)

```python
!python -m expander.llm --smoke                 # config llm.model + llm.revision, 4-bit NF4, fp16 compute
```

This loads the model, sends one hard-coded scene (with a reference block) twice and prints:
- the **raw reply**;
- `valid JSON`, `names leaked`;
- `load` (s);
- `s/call` (first and warm);
- `peak VRAM`.

Paste these numbers into the Status issue and into `EXPANDER.md` §12. Then:
- **Valid JSON and under about 10 s/call:** keep the model and pin `llm.model` plus `llm.revision` in `config/m3.json` (PR + CHANGELOG).
- **Prose instead of JSON:** run `!python -m expander.llm --smoke --prompt_version v1ex`, which adds one worked example from dev scene s25. If that fixes it, use `--prompt_version v1ex` for every LLM run (it is recorded in each record and is part of the cache key). Note that s25's own expansion is then not independent of the prompt.
- **Larger model, for the optional `c3_7b`:** `!python -m expander.llm --smoke --size large`.

Then re-freeze the libraries (see the TODO in `expander/requirements_llm.txt`):

```python
!pip freeze | grep -iE '^(torch|transformers|accelerate|bitsandbytes|tokenizers|safetensors|huggingface.hub|jinja2)=='
```

Token counter check (CLIP tokenizer; expected `4 25`):

```python
!python -c "from expander._cfg import get_counter, load_config as L; n=get_counter(); c=L(); print(n(c['trigger']), n(c['trigger']+', '+c['style_tail']))"
```

## 4. Scene file

```python
import json, os
SCENES = "eval/test_scenes_dev.json"
if not os.path.exists(SCENES):   # until B merges the dev scenes: D's 8 + the 3 fixtures (previews, re-run later)
    SCENES = "/content/dev_scenes_preview.json"
    scenes = json.load(open("expander/dev_scenes_D.json")) + json.load(open("expander/_shims/fixtures_scenes.json"))
    json.dump(scenes, open(SCENES, "w"), indent=1)
print(SCENES, len(json.load(open(SCENES))), "scenes")
```

Expansions made from the preview file are still written to `outputs/expansions/<condition>_dev.json`. Commit them, then re-run with `--overwrite` once B's file lands, and again after S2 with the real KB. Git history keeps the old versions, and every record carries `kb_version`.

## 5. c2, c3, c4 on dev

```python
!python expander/batch_expand.py --scenes $SCENES --mode template --split dev --out outputs/expansions/c2_template_dev.json
!python expander/batch_expand.py --scenes $SCENES --mode llm_kb   --split dev --out outputs/expansions/c3_llm_kb_dev.json
!python expander/batch_expand.py --scenes $SCENES --mode llm_nokb --split dev --out outputs/expansions/c4_llm_nokb_dev.json
```

How these runs behave:
- Each LLM run loads the model **once**, and only if a scene misses the cache. It closes the model at the end and prints one summary line with `llm_calls`, `cache_hits`, `model_loads` and the mean latency.
- **Resumable:** after a disconnect, re-run the same line. Scenes already in `--out` are skipped, and cached LLM replies are free.
- Add `--kb kb/kb.json` after S2 if C's parser does not pick the KB up by default. Add `--prompt_version v1ex` if step 3 required it.
- `latency_s` is the wall time of `expand()`. The first scene that loads the model also records `model_load_s`; subtract it for s/scene.

## 6. Check (target: at least 95% valid, zero leaks after repair)

```python
!python expander/check_expansions.py outputs/expansions/c2_template_dev.json outputs/expansions/c3_llm_kb_dev.json outputs/expansions/c4_llm_nokb_dev.json --strict
```

The script prints a markdown table: valid, trigger-first, within budget, leaks before and after repair, fallback rate, latency, tokens and drops. It also prints one `INVALID` line per bad record. Token counts are recomputed with the CLIP counter. `--strict` exits 1 if a file is below 95% valid or still has a name after repair. Paste the table into `EXPANDER.md` §9. If the fallback rate is high, look at `error` in the records and at the cache replies.

## 7. Commit outputs and cache

```python
!git config user.email "<you>@users.noreply.github.com" && git config user.name "<you>"
!git checkout -B d/expander-runs
!git add expander/cache.json outputs/expansions/c2_template_dev.json outputs/expansions/c3_llm_kb_dev.json outputs/expansions/c4_llm_nokb_dev.json
!git commit -m "[expander] c2-c4 dev expansions (kb <kb_version>, prompt v1, tpl2)"
!git push -u origin d/expander-runs     # then open a PR; self-merge (your folders only)
```

`expander/cache.json` is committed for **dev only**. Never commit held-out text (step 10).

## 8. Hand-off to generation (A's `gen` profile)

1. Close the LLM: *Runtime → Disconnect and delete runtime*. `batch_expand.py` already closed the model; a new runtime makes the freed VRAM certain.
2. Start a new T4 runtime, then run step 1 and the following cell:

```python
!bash pipeline/setup_session.sh gen
IMG = "/content/drive/MyDrive/m3/images"
!python pipeline/run_batch.py --expansions outputs/expansions/c2_template_dev.json --model rrv_v1 --seeds 1 2 --out {IMG}/c2_template@rrv_v1/dev
!python pipeline/run_batch.py --expansions outputs/expansions/c3_llm_kb_dev.json   --model rrv_v1 --seeds 1 2 --out {IMG}/c3_llm_kb@rrv_v1/dev
!python pipeline/run_batch.py --expansions outputs/expansions/c4_llm_nokb_dev.json --model rrv_v1 --seeds 1 2 --out {IMG}/c4_llm_nokb@rrv_v1/dev
```

3. Scoring runs in a third runtime (`eval` profile): `python eval/score.py --manifest {IMG}/<cond>@rrv_v1/dev/manifest.json --out results/<cond>@rrv_v1_dev.json`. `score.py` copies `token_count`, `trigger_first`, `names_leaked` and `source` from the expansion file.

## 9. Optional experiments (Day 4; write the hypothesis first; cut order: negatives → LLM size → c5)

```python
!python expander/batch_expand.py --scenes $SCENES --mode llm_kb_layout --split dev --out outputs/expansions/c5_llm_kb_layout_dev.json
!python expander/batch_expand.py --scenes $SCENES --mode llm_kb --llm large --split dev --out outputs/expansions/c3_7b_dev.json
!python expander/check_expansions.py outputs/expansions/c5_llm_kb_layout_dev.json outputs/expansions/c3_7b_dev.json --strict
```

- `--llm large` uses `llm.model_large` and `llm.model_large_revision`. `--llm_model X [--llm_revision R]` overrides the model id.
- `c5` records carry `layout` and use prompt `v1_layout` (or `v1ex_layout` with `--prompt_version v1ex`).
- Watch `phrases_dropped` in c5: the leading layout phrases cost tokens and can push figure phrases out.

## 10. Held-out (B, Day 5, frozen S3 sha, private cache)

```python
!git checkout <S3 sha>
PRIV = "/content/drive/MyDrive/<B's private folder>"   # outside the repo and outside the shared Drive
!python expander/batch_expand.py --scenes {PRIV}/test_scenes_heldout.json --mode template --split heldout --out outputs/expansions/c2_template_heldout.json
!python expander/batch_expand.py --scenes {PRIV}/test_scenes_heldout.json --mode llm_kb   --split heldout --cache {PRIV}/heldout_cache.json --out outputs/expansions/c3_llm_kb_heldout.json
!python expander/batch_expand.py --scenes {PRIV}/test_scenes_heldout.json --mode llm_nokb --split heldout --cache {PRIV}/heldout_cache.json --out outputs/expansions/c4_llm_nokb_heldout.json
```

Use the frozen `--prompt_version` that D posted at S3. Do **not** `git add expander/cache.json` from this session.

## 11. If things go wrong

| Symptom | Move |
|---|---|
| `CUDA out of memory` on load | Restart the runtime and run nothing else in it. Re-run: the batch resumes. Still OOM: a smaller model with `--llm_model Qwen/Qwen2.5-3B-Instruct` (pin its revision), or lower `llm.max_new_tokens` (default 256) |
| `the LLM failed to load` (batch stops) | Read the error. The batch stops on purpose so that a file of template fallbacks is never labelled c3. Fix the cause and re-run the same line |
| `bitsandbytes` import or CUDA-kernel error | `!pip install -U bitsandbytes`, restart the runtime, smoke test again, then re-freeze the requirements |
| Prose instead of JSON (high `fallback`, `json_retry: true`) | Read the raw replies in the smoke test or in `expander/cache.json`. Use `--prompt_version v1ex` (worked example). Report the fallback rate; never hide it |
| Names after repair > 0 (`--strict` fails) | Check that `kb/names.json` is current (`check_expansions.py --names kb/names.json`), then re-run. Misspelled names not in the list pass both the scrub and A's CI: report them to C (`needs:C`) |
| Slow (much more than about 10 s/scene) | `torch.cuda.is_available()` must be True and the smoke test must show a T4. First-call latency includes the model load (`model_load_s`). Lower `max_new_tokens` |
| `token counter unavailable ... approximate counter` warning | `transformers` is missing or the CLIP tokenizer could not download. Install the requirements (or set `HF_TOKEN`) and re-run with `--overwrite`. Approximate counts are for dry runs only, and `check_expansions.py` flags them |
| `holds condition=... use --overwrite` | The `--out` file belongs to another condition or mode. Use the right file name, or pass `--overwrite` on purpose |
| Disconnected mid-run | Re-run the same command. The file is rewritten atomically after every scene, and the cache after every call |

CPU dry run, with no GPU and no model (CI does the same): `python expander/batch_expand.py --scenes expander/dev_scenes_D.json --mode llm_kb --llm fake --out /tmp/c3_llm_kb_dev.json`. With `--llm fake`, the cache stays in memory unless `--cache` is given.
