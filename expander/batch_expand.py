"""Expand a scene file into ``outputs/expansions/<condition>_<split>.json`` (a JSON list of ExpandedPrompt).

    python expander/batch_expand.py --scenes eval/test_scenes_dev.json --mode llm_kb \\
        --out outputs/expansions/c3_llm_kb_dev.json [--kb kb/kb.json] [--cache expander/cache.json]

- Scenes: a JSON list (or JSONL, or ``{"scenes": [...]}``) in the Overview §6.2 schema; ``text`` is the input,
  ``id`` the scene id (``raw`` / ``scene_id`` are accepted too). Each scene is parsed with C's parser
  (``parse_scene(text, scene_id=id, kb_path=--kb)``; the stub until S2), except in ``raw`` mode.
- Condition: ``--condition`` > the ``--out`` file name (``c3_7b_dev.json`` -> ``c3_7b``) > the mode.
- LLM modes: ONE client, created only if the mode needs it and loaded lazily on the first cache miss (a
  warm cache never loads the model), closed in ``finally``. ``--llm fake`` is a deterministic CPU stand-in.
- Resumable: records already in ``--out`` (same condition and mode) are kept and their scenes skipped; the file
  is rewritten atomically after every scene, sorted by ``scene_id``. ``--overwrite`` starts afresh.
- ``latency_s`` is the wall time of ``expand()``; the scene that triggered the lazy model load also carries
  ``model_load_s`` (subtract it for s/scene).
- Cache: ``expander/cache.json`` by default (committed for dev). B's held-out run passes a private path.
  With ``--llm fake`` and no ``--cache``, the cache stays in memory so fake replies never enter the real one.

Standard library only at import time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):  # `python expander/batch_expand.py`: import `expander` from the repo root
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
    sys.path.insert(0, os.path.dirname(_HERE))

from expander._cfg import get_counter, get_parser, load_config, repo_root
from expander.cache import Cache
from expander.expand import LLM_MODES, MODES, expand, resolve_condition
from expander.llm_base import load_system_prompt

DEFAULT_CACHE = repo_root() / "expander" / "cache.json"
LLM_CHOICES = ("default", "large", "fake")


def _log(msg: str) -> None:
    print(f"[batch_expand] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- inputs

def load_scenes(path: str | os.PathLike) -> list[dict]:
    """Scenes as ``[{"id", "text", "split"?, ...}]`` from a JSON list, ``{"scenes": [...]}`` or JSONL."""
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = data.get("scenes", [data])
    if not isinstance(data, list):
        raise TypeError(f"{path}: expected a JSON list of scenes")
    scenes, seen = [], set()
    for i, s in enumerate(data):
        if not isinstance(s, dict):
            raise TypeError(f"{path}: scene #{i} is not an object")
        sid = s.get("id", s.get("scene_id"))
        text = s.get("text", s.get("raw"))
        if not isinstance(sid, str) or not sid.strip():
            raise ValueError(f"{path}: scene #{i} has no string 'id'")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: scene {sid!r} has no 'text'")
        if sid in seen:
            raise ValueError(f"{path}: duplicate scene id {sid!r}")
        seen.add(sid)
        scenes.append({**s, "id": sid, "text": text})
    return scenes


# --------------------------------------------------------------------------- token counter

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")


def approx_n_tokens(text: str) -> int:
    """Conservative stand-in for CLIP BPE (no ``transformers``): ceil(len/5) per word, 1 per punctuation mark.

    It over-counts real CLIP (``"rrvarma style"`` -> 4, the v2 tail -> 27 vs 20), so prompts stay in budget.
    Only for dry runs and CI; records carry ``token_counter: "approx"``.
    """
    return sum(math.ceil(len(t) / 5) if t[0].isalnum() else 1 for t in _TOKEN_RE.findall(text))


def resolve_counter(choice: str = "auto"):
    """``(count_fn, name)``: ``clip`` = A's ``common.tokens.n_tokens`` (or its shim); ``approx`` = above.

    ``auto`` uses ``clip`` when it works and falls back to ``approx`` with a warning.
    """
    if choice == "approx":
        return approx_n_tokens, "approx"
    try:
        fn = get_counter()
        fn("rrvarma style")
        return fn, "clip"
    except Exception as e:  # missing transformers, offline tokenizer download, ...
        if choice == "clip":
            raise SystemExit(f"[batch_expand] the CLIP token counter is unavailable: {e}") from e
        _log(f"WARNING: CLIP token counter unavailable ({type(e).__name__}: {e}); using the approximate "
             "counter. Records get token_counter='approx'; do not score them.")
        return approx_n_tokens, "approx"


# --------------------------------------------------------------------------- output

def write_records(path: str | os.PathLike, records: list[dict]) -> None:
    """Atomic write (temp file + ``os.replace``) of the records sorted by ``scene_id``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = sorted(records, key=lambda r: str(r.get("scene_id", "")))
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_existing(path: Path, condition: str, mode: str) -> dict[str, dict]:
    """Records already in ``path`` keyed by ``scene_id``; refuses a file of another condition or mode."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"[batch_expand] {path} is not a JSON list; use --overwrite or another --out")
    done: dict[str, dict] = {}
    for r in data:
        if not isinstance(r, dict) or not isinstance(r.get("scene_id"), str):
            raise SystemExit(f"[batch_expand] {path} has a record without scene_id; use --overwrite")
        if r.get("condition") != condition or r.get("mode", mode) != mode:
            raise SystemExit(f"[batch_expand] {path} holds condition={r.get('condition')!r} mode={r.get('mode')!r}, "
                             f"not {condition!r}/{mode!r}; use --overwrite or another --out")
        done[r["scene_id"]] = r
    return done


# --------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="batch_expand.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenes", required=True, help="JSON list / JSONL of scenes (Overview §6.2)")
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--out", required=True, help="outputs/expansions/<condition>_<split>.json")
    ap.add_argument("--kb", default=None, help="KB file for the parser and the mood lookup (default: kb/kb.json, "
                    "kb/stub_kb.json, then the shim)")
    ap.add_argument("--cache", default=None, help=f"LLM cache file (default {DEFAULT_CACHE.relative_to(repo_root())}; "
                    "'none' = in memory). Held-out: a private path outside the repo")
    ap.add_argument("--condition", default=None, help="condition ID (default: from --out, else from --mode)")
    ap.add_argument("--split", default=None, help="dev | heldout (default: each scene's 'split', else '')")
    ap.add_argument("--llm", choices=LLM_CHOICES, default="default",
                    help="config llm.model (default), llm.model_large (large), or the CPU echo fake (fake)")
    ap.add_argument("--llm_model", default=None, help="override the model id (e.g. Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--llm_revision", default=None, help="HF revision for --llm_model")
    ap.add_argument("--no_4bit", action="store_true", help="load the HF model without bitsandbytes (CPU previews)")
    ap.add_argument("--prompt_version", default=None, help="v1 (config default), v1ex (+ worked example), ...")
    ap.add_argument("--limit", type=int, default=None, help="only the first N scenes of the file")
    ap.add_argument("--config", default=None, help="config file (default: $M3_CONFIG > config/m3.json > shim)")
    ap.add_argument("--overwrite", action="store_true", help="ignore an existing --out instead of resuming")
    ap.add_argument("--counter", choices=("auto", "clip", "approx"), default="auto",
                    help="token counter: clip = common.tokens.n_tokens; approx = CPU stand-in for dry runs")
    return ap


def run(argv: list[str] | None = None) -> dict:
    """Run the batch; returns the summary dict (``main`` prints it)."""
    a = build_parser().parse_args(argv)
    cfg = load_config(a.config)
    if a.no_4bit:
        cfg.setdefault("llm", {})["load_in_4bit"] = False
    out = Path(a.out)
    condition = resolve_condition(a.mode, a.condition, out)
    scenes = load_scenes(a.scenes)
    if a.limit is not None:
        scenes = scenes[: max(a.limit, 0)]
    done = {} if a.overwrite else load_existing(out, condition, a.mode)
    pending = [s for s in scenes if s["id"] not in done]
    needs_llm = a.mode in LLM_MODES

    pv = a.prompt_version
    if needs_llm:
        eff = pv or cfg["llm"]["prompt_version"]
        if a.mode == "llm_kb_layout" and not eff.endswith("_layout"):
            eff += "_layout"
        try:
            load_system_prompt(eff)
        except KeyError as e:
            raise SystemExit(f"[batch_expand] {e.args[0]}") from e

    count_fn, counter_name = resolve_counter(a.counter) if pending else (approx_n_tokens, "n/a")
    client = cache = None
    if needs_llm and pending:
        from expander.llm import make_client

        backend = "fake" if a.llm == "fake" else None
        size = "large" if a.llm == "large" else "default"
        client = make_client(cfg, size, backend, model=a.llm_model, revision=a.llm_revision, lazy=True)
        cache_path = a.cache if a.cache is not None else (None if a.llm == "fake" else str(DEFAULT_CACHE))
        cache = Cache(None if cache_path in (None, "", "none") else cache_path)
    parse = get_parser() if a.mode != "raw" and pending else None

    records = dict(done)
    new: list[dict] = []
    try:
        for i, s in enumerate(pending, 1):
            sid, text = s["id"], s["text"]
            split = a.split if a.split is not None else (s.get("split") or "")
            scene = parse(text, scene_id=sid, kb_path=a.kb) if parse else {"id": sid}
            loads_before = getattr(client, "loads", 0)
            rec = expand(text, scene, a.mode, llm=client, cache=cache, kb_path=a.kb, condition=condition,
                         split=split, count_fn=count_fn, cfg=cfg, prompt_version=pv)
            if getattr(client, "load_error", None) is not None:  # a whole file of template fallbacks is not c3
                raise SystemExit(f"[batch_expand] the LLM failed to load ({client.load_error!r}); stopped before "
                                 f"{sid}. {len(new)} new records were written. Fix the load (see RUNBOOK_COLAB.md) "
                                 "and re-run the same command to resume.")
            rec["scene_id"] = sid
            rec["token_counter"] = counter_name
            if needs_llm and client is not None:
                rec["llm_revision"] = getattr(client, "revision", None)
                if getattr(client, "loads", 0) > loads_before:  # this scene's latency_s includes the model load
                    rec["model_load_s"] = client.load_s
            records[sid] = rec
            new.append(rec)
            write_records(out, list(records.values()))
            _log(f"{i}/{len(pending)} {sid}: source={rec['source']} tokens={rec['token_count']} "
                 f"calls={rec['llm_calls']} hits={rec['cache_hits']} {rec['latency_s']:.2f}s")
    finally:
        if client is not None:
            client.close()
    if not pending and not out.exists():
        write_records(out, [])

    allr = list(records.values())
    src = Counter(r["source"] for r in allr)
    lat = [r["latency_s"] for r in new]
    toks = [r["token_count"] for r in allr if isinstance(r.get("token_count"), int)]
    return {
        "out": str(out), "condition": condition, "mode": a.mode, "n": len(allr), "new": len(new),
        "kept": len(allr) - len(new), "source": dict(src),
        "llm_calls": sum(r.get("llm_calls", 0) for r in new), "cache_hits": sum(r.get("cache_hits", 0) for r in new),
        "model_loads": getattr(client, "loads", 0) if client is not None else 0,
        "llm_model": next((r["llm_model"] for r in new), None), "counter": counter_name,
        "mean_latency_s": round(statistics.fmean(lat), 4) if lat else None,
        "mean_tokens": round(statistics.fmean(toks), 1) if toks else None, "max_tokens": max(toks) if toks else None,
        "trigger_first": sum(bool(r.get("trigger_first")) for r in allr),
        "names_after_repair": sum(bool(r.get("names_after_repair")) for r in allr if r.get("mode") != "raw"),
    }


def format_summary(s: dict) -> str:
    src = " ".join(f"{k}={s['source'].get(k, 0)}" for k in ("llm", "fallback", "template"))
    lat = "n/a" if s["mean_latency_s"] is None else f"{s['mean_latency_s']:.2f}s"
    return (f"[batch_expand] {s['condition']}: {s['n']} records -> {s['out']} ({s['new']} new, {s['kept']} resumed) | "
            f"{src} | llm_calls={s['llm_calls']} cache_hits={s['cache_hits']} model_loads={s['model_loads']} | "
            f"mean latency {lat} | tokens mean {s['mean_tokens']} max {s['max_tokens']} ({s['counter']}) | "
            f"trigger_first {s['trigger_first']}/{s['n']} | leaks after repair {s['names_after_repair']}")


def main(argv: list[str] | None = None) -> int:
    print(format_summary(run(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
