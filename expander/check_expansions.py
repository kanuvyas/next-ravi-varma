"""Validate expansion files (D guide Day 2.5): per file, % valid, trigger-first, within budget, name leaks
before/after repair, fallback rate, latency. Prints a markdown table.

    python expander/check_expansions.py outputs/expansions/c2_template_dev.json outputs/expansions/c3_llm_kb_dev.json \\
        [--max_tokens 70] [--json results/check_expansions_dev.json] [--strict]

A record is **valid** when all 16 contract fields are present, the positive starts with the trigger, its token
count is within the budget, and (except in ``raw`` mode, which keeps the user's text on purpose) no name is left
in the positive. Token counts are recomputed with the CLIP counter (``common.tokens.n_tokens``) when it is
available; otherwise the file's ``token_count`` is trusted and the output says so.
``--strict`` exits 1 if any file is below 95% valid or has a leak after repair (target: >= 95%, zero leaks).

Standard library only at import time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python expander/check_expansions.py`
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
    sys.path.insert(0, os.path.dirname(_HERE))

from expander._cfg import get_counter, load_config
from expander.expand import CONTRACT_FIELDS
from expander.names import NameMatcher, default_matcher, load_names

VALID_TARGET = 95.0
# Report only. "oil" is NOT a style word: "lit by an oil lamp" is a visual phrase.
STYLE_WORDS = re.compile(r"\b(painting|painted|canvas|artist|style|brushwork|masterpiece)\b", re.IGNORECASE)


def is_raw(rec: dict) -> bool:
    """``raw`` mode (c0/c1): names are expected in the positive."""
    if "mode" in rec:
        return rec["mode"] == "raw"
    return rec.get("prompt_version") == "raw" or str(rec.get("condition", "")).startswith(("c0_", "c1_"))


def check_record(rec: dict, *, trigger: str, max_tokens: int, matcher: NameMatcher, counter=None) -> dict:
    """Per-record verdict: ``{"valid", "reasons", "tokens", "leaks_after", ...}``."""
    reasons = []
    missing = [f for f in CONTRACT_FIELDS if f not in rec]
    if missing:
        reasons.append(f"missing fields {missing}")
    pos = rec.get("positive") if isinstance(rec.get("positive"), str) else ""
    trig = bool(pos) and pos.startswith(trigger) and rec.get("trigger_first", True) is not False
    if not trig:
        reasons.append("trigger not first")
    tokens = counter(pos) if (counter is not None and pos) else rec.get("token_count")
    within = isinstance(tokens, int) and 0 <= tokens <= max_tokens
    if not within:
        reasons.append(f"tokens {tokens} > {max_tokens}" if isinstance(tokens, int) and tokens > max_tokens
                       else f"token_count {tokens!r} unknown")
    raw = is_raw(rec)
    after: list[str] = []
    if not raw:
        after = list(dict.fromkeys([*matcher.find(pos), *(rec.get("names_after_repair") or [])]))
        if after:
            reasons.append(f"names after repair {after}")
    phrases = " ; ".join(p for p in rec.get("phrases_kept") or [] if isinstance(p, str))
    return {"scene_id": rec.get("scene_id"), "valid": not reasons, "reasons": reasons, "tokens": tokens,
            "trigger_first": trig, "within_budget": within, "raw": raw, "leaks_after": after,
            "leaked_before": bool(rec.get("names_leaked")), "fallback": rec.get("source") == "fallback",
            "style_words": len(STYLE_WORDS.findall(phrases)) if not raw else 0}


def _pct(k: int, n: int) -> float:
    return round(100.0 * k / n, 1) if n else 0.0


def summarize(records: list[dict], *, trigger: str, max_tokens: int, matcher: NameMatcher | None = None,
              counter=None) -> dict:
    """Summary of one file (see the module docstring for the definitions)."""
    m = matcher or default_matcher()
    checks = [check_record(r, trigger=trigger, max_tokens=max_tokens, matcher=m, counter=counter) for r in records]
    n = len(records)
    non_raw = [c for c in checks if not c["raw"]]
    lat = [r["latency_s"] for r in records if isinstance(r.get("latency_s"), (int, float))]
    toks = [c["tokens"] for c in checks if isinstance(c["tokens"], int)]
    return {
        "n": n,
        "n_valid": sum(c["valid"] for c in checks),
        "pct_valid": _pct(sum(c["valid"] for c in checks), n),
        "pct_trigger_first": _pct(sum(c["trigger_first"] for c in checks), n),
        "pct_within_budget": _pct(sum(c["within_budget"] for c in checks), n),
        "leak_rate_before": _pct(sum(c["leaked_before"] for c in checks), n),
        "n_leaks_after": sum(bool(c["leaks_after"]) for c in non_raw),
        "leak_rate_after": _pct(sum(bool(c["leaks_after"]) for c in non_raw), len(non_raw)),
        "n_raw": n - len(non_raw),
        "fallback_rate": _pct(sum(c["fallback"] for c in checks), n),
        "mean_latency_s": round(statistics.fmean(lat), 3) if lat else None,
        "median_latency_s": round(statistics.median(lat), 3) if lat else None,
        "mean_tokens": round(statistics.fmean(toks), 1) if toks else None,
        "max_tokens_seen": max(toks) if toks else None,
        "mean_n_dropped": round(statistics.fmean(len(r.get("phrases_dropped") or []) for r in records), 2) if n else None,
        "mean_n_scrubbed": round(statistics.fmean(len(r.get("phrases_scrubbed") or []) for r in records), 2) if n else None,
        "style_words": sum(c["style_words"] for c in checks),
        "n_approx_counter": sum(r.get("token_counter") == "approx" for r in records),
        "invalid": [{"scene_id": c["scene_id"], "reasons": c["reasons"]} for c in checks if not c["valid"]],
    }


def resolve_counter(recount: bool):
    """``(counter or None, label)``; ``None`` means "trust token_count"."""
    if not recount:
        return None, "token_count from the file (--no_recount)"
    try:
        fn = get_counter()
        fn("rrvarma style")
        return fn, "recomputed with the CLIP counter (common.tokens.n_tokens)"
    except Exception as e:  # noqa: BLE001 - no transformers / offline
        return None, f"token_count trusted from the file (CLIP counter unavailable: {type(e).__name__})"


def load_records(path: str | os.PathLike) -> list[dict]:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
        raise ValueError(f"{path}: expected a JSON list of ExpandedPrompt records")
    return data


def markdown_table(summaries: dict[str, dict]) -> str:
    def f(v, suffix=""):
        return "n/a" if v is None else f"{v}{suffix}"

    rows = [("| file | n | valid | trigger-first | within budget | leaks before | leaks after | fallback "
             "| latency mean / median (s) | tokens mean / max | dropped / rec | style words |"),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for path, s in summaries.items():
        after = "n/a (raw)" if s["n_raw"] == s["n"] and s["n"] else f"{s['n_leaks_after']} ({s['leak_rate_after']}%)"
        rows.append(f"| {Path(path).name} | {s['n']} | {s['pct_valid']}% | {s['pct_trigger_first']}% "
                    f"| {s['pct_within_budget']}% | {s['leak_rate_before']}% | {after} | {s['fallback_rate']}% "
                    f"| {f(s['mean_latency_s'])} / {f(s['median_latency_s'])} | {f(s['mean_tokens'])} / "
                    f"{f(s['max_tokens_seen'])} | {f(s['mean_n_dropped'])} | {s['style_words']} |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="check_expansions.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", metavar="FILE")
    ap.add_argument("--max_tokens", type=int, default=None, help="budget (default: config max_prompt_tokens)")
    ap.add_argument("--json", default=None, metavar="OUT", help="also write the summaries as JSON")
    ap.add_argument("--strict", action="store_true", help=f"exit 1 if valid < {VALID_TARGET}%% or any leak remains")
    ap.add_argument("--config", default=None)
    ap.add_argument("--names", default=None, help="names file (default: kb/names.json, else expander/names_seed.json)")
    ap.add_argument("--no_recount", action="store_true", help="trust token_count instead of recounting")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    max_tokens = a.max_tokens if a.max_tokens is not None else int(cfg["max_prompt_tokens"])
    matcher = NameMatcher(load_names(a.names)) if a.names else default_matcher()
    counter, token_note = resolve_counter(not a.no_recount)

    summaries: dict[str, dict] = {}
    for path in a.files:
        try:
            records = load_records(path)
        except (OSError, ValueError, TypeError) as e:
            print(f"check_expansions: cannot read {path}: {e}", file=sys.stderr)
            return 2
        summaries[path] = summarize(records, trigger=cfg["trigger"], max_tokens=max_tokens, matcher=matcher,
                                    counter=counter)

    print(markdown_table(summaries))
    print(f"\nBudget {max_tokens} tokens (excluding start/end); {token_note}. Valid = contract fields + trigger first "
          f"+ within budget + no names after repair (raw mode exempt from names). Target >= {VALID_TARGET}% valid, "
          "0 leaks after repair.")
    failed = []
    for path, s in summaries.items():
        if s["n_approx_counter"]:
            print(f"NOTE {Path(path).name}: {s['n_approx_counter']} records were fitted with the APPROXIMATE counter "
                  "(dry run); re-run with transformers installed before scoring.")
        for bad in s["invalid"][:5]:
            print(f"INVALID {Path(path).name} {bad['scene_id']}: {'; '.join(bad['reasons'])}")
        if s["pct_valid"] < VALID_TARGET or s["n_leaks_after"]:
            failed.append(Path(path).name)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"max_tokens": max_tokens, "token_source": token_note,
                                            "files": summaries}, indent=1, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    if a.strict and failed:
        print(f"STRICT: FAILED for {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
