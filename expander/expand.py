"""``expand(text, scene=None, mode="llm_kb") -> ExpandedPrompt`` (Overview v2 §6.4, §6.5).

Design principle: the LLM (or the template) writes only the scene phrases. This code adds the trigger
(always first) and the style tail (always last), fits everything under ``max_prompt_tokens`` and
removes proper names. The LLM can never drop the trigger, blow the budget or crash the pipeline.

Modes:
- ``raw``: ``"<trigger>, <text>, <tail>"`` exactly like A's c1 (no fitting, no scrub; may hold names).
- ``template``: deterministic phrases from the SceneSpec (``template.py``). No LLM.
- ``llm_kb`` / ``llm_nokb``: LLM phrases. The ONLY difference is the ``Reference descriptions:`` block
  appended to the user message in ``llm_kb``.
- ``llm_kb_layout``: ``llm_kb`` plus an optional ``layout`` object, turned into leading layout phrases.

LLM validation chain: reply -> JSON (tolerates code fences and prose around the first ``{...}``) ->
one retry with "Return ONLY valid JSON" -> template fallback (``source="fallback"``). Any exception in
the LLM path ends in the fallback. Names: re-ask once with the repair instruction, then drop the
phrases that still contain names. ``names_leaked`` is recorded BEFORE repair; ``names_after_repair``
(extra field) must be ``[]``.

Standard library only at import time.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from expander._cfg import get_counter, get_parser, load_config, load_kb
from expander.assemble import CountFn, assemble, clean_phrases
from expander.cache import Cache, cache_key
from expander.llm_base import LLMClient, load_system_prompt
from expander.names import NameMatcher, default_matcher, scrub_phrases
from expander.template import TEMPLATE_PROMPT_VERSION, template_prompt

MODES = ("raw", "template", "llm_kb", "llm_nokb", "llm_kb_layout")
LLM_MODES = frozenset({"llm_kb", "llm_nokb", "llm_kb_layout"})
KB_MODES = frozenset({"llm_kb", "llm_kb_layout"})
MODE_TO_CONDITION = {"raw": "c1_lora_raw", "template": "c2_template", "llm_kb": "c3_llm_kb",
                     "llm_nokb": "c4_llm_nokb", "llm_kb_layout": "c5_llm_kb_layout"}
SPLITS = ("dev", "heldout", "test")
# Overview v2 §6.4, in contract order. Extra fields follow them in every record.
CONTRACT_FIELDS = ("scene_id", "condition", "split", "positive", "negative", "phrases_kept", "phrases_dropped",
                   "token_count", "trigger_first", "names_leaked", "source", "layout", "llm_model",
                   "prompt_version", "kb_version", "latency_s")

RETRY_INSTRUCTION = "Return ONLY valid JSON"
REPAIR_INSTRUCTION = "Remove proper names; describe by clothing, pose and setting."

POSES = ("standing", "seated", "kneeling", "lying", "reclining", "arms_raised", "walking", "flying")
POSITIONS = ("left", "center", "right", "foreground", "background")
_POSE_ALIASES = {"sitting": "seated", "arms raised": "arms_raised", "arms-raised": "arms_raised",
                 "lying down": "lying"}
_POSITION_ALIASES = {"centre": "center", "middle": "center"}
_POSITION_PHRASE = {"left": "on the left", "center": "in the center", "right": "on the right",
                    "foreground": "in the foreground", "background": "in the background"}
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
                 9: "nine", 10: "ten"}
MAX_LAYOUT_FIGURES = 4

_DEFAULT_LLM: LLMClient | None = None


class LLMUnavailableError(RuntimeError):
    """An LLM mode was requested but no client was given and the config backend is not ``"none"``."""


class _InvalidReply(ValueError):
    pass


# --------------------------------------------------------------------------- client lifecycle

def set_default_llm(client: LLMClient | None) -> None:
    """Register the client ``expand()`` uses when called without ``llm=`` (e.g. from ``text_to_image``)."""
    global _DEFAULT_LLM
    _DEFAULT_LLM = client


def close() -> None:
    """Close and forget the default client (frees GPU memory before diffusion). Safe to call twice."""
    global _DEFAULT_LLM
    client, _DEFAULT_LLM = _DEFAULT_LLM, None
    if client is not None:
        client.close()


def _resolve_llm(llm: LLMClient | None, cfg: dict) -> LLMClient | None:
    """The client to use; ``None`` means "backend none: use the template fallback"."""
    client = llm if llm is not None else _DEFAULT_LLM
    if client is not None:
        return client
    if str(cfg.get("llm", {}).get("backend", "")).lower() == "none":
        return None
    raise LLMUnavailableError(
        "this mode needs an LLM: pass expand(..., llm=<LLMClient>) or call set_default_llm(client); "
        "set config llm.backend to \"none\" to use the template fallback instead")


# --------------------------------------------------------------------------- helpers

def resolve_condition(mode: str, condition: str | None = None, out_path: str | Path | None = None) -> str:
    """Condition ID: explicit ``condition`` > ``cN_...`` from the ``out_path`` file name > by ``mode``.

    ``outputs/expansions/c6_llm_kbllm_dev.json`` -> ``c6_llm_kbllm``; ``c3_7b_dev.json`` -> ``c3_7b``.
    """
    if condition:
        return condition
    if out_path:
        stem = Path(out_path).name
        stem = stem.removesuffix(".json")
        if re.match(r"^c\d+[a-z]*(?:_|$)", stem):
            return re.sub("_(?:" + "|".join(SPLITS) + ")$", "", stem)
    if mode not in MODE_TO_CONDITION:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    return MODE_TO_CONDITION[mode]


def build_user_message(text: str, scene: dict | None, mode: str) -> str:
    """The raw text, plus (KB modes only) ``Reference descriptions:`` with ``<mention> -> <visual>`` lines."""
    msg = text.strip()
    if mode not in KB_MODES or not scene:
        return msg
    refs = []
    for ch in scene.get("characters") or []:
        if not isinstance(ch, dict):
            continue
        who = (ch.get("mention") or ch.get("canonical") or "").strip()
        visual = (ch.get("visual") or "").strip()
        if who and visual:
            refs.append(f"{who} -> {visual}")
    setting = scene.get("setting")
    if isinstance(setting, dict) and (setting.get("visual") or "").strip():
        refs.append(f"setting -> {setting['visual'].strip()}")
    if refs:
        msg += "\n\nReference descriptions:\n" + "\n".join(refs)
    return msg


def _brace_end(s: str, start: int) -> int | None:
    """Index of the ``}`` closing the ``{`` at ``start`` (string-aware), or ``None``."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def extract_json(reply: str) -> dict | None:
    """First JSON object in ``reply``; tolerates code fences and prose. Prefers objects with ``phrases``."""
    if not isinstance(reply, str):
        return None
    s = reply.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except ValueError:
        pass
    first = None
    i = s.find("{")
    while i != -1:
        end = _brace_end(s, i)
        if end is not None:
            try:
                obj = json.loads(s[i:end + 1])
            except ValueError:
                obj = None
            if isinstance(obj, dict):
                if "phrases" in obj:
                    return obj
                first = first or obj
        i = s.find("{", i + 1)
    return first


def parse_reply(reply: str) -> dict | None:
    """``{"phrases": [non-empty clean strings], ...other keys}`` or ``None`` if the reply is unusable."""
    obj = extract_json(reply)
    if not obj or not isinstance(obj.get("phrases"), list):
        return None
    phrases = clean_phrases(obj["phrases"])
    if not phrases:
        return None
    return {**obj, "phrases": phrases}


def _vocab(value, allowed: tuple[str, ...], aliases: dict[str, str]) -> str | None:
    if not isinstance(value, str):
        return None
    v = " ".join(value.strip().lower().split())
    v = aliases.get(v, v).replace(" ", "_")
    return v if v in allowed else None


def parse_layout(obj, matcher: NameMatcher | None = None) -> dict | None:
    """Validate an LLM ``layout`` object. Invalid poses/positions are dropped from their figure; a figure
    with neither a valid pose nor a valid position is dropped; roles with names become ``"figure"``.
    Returns ``{"n_figures": int, "figures": [{"role", "pose"?, "position"?}]}`` or ``None``."""
    if not isinstance(obj, dict):
        return None
    m = matcher or default_matcher()
    figures = []
    raw_figs = obj.get("figures") if isinstance(obj.get("figures"), list) else []
    for f in raw_figs:
        if not isinstance(f, dict):
            continue
        pose = _vocab(f.get("pose"), POSES, _POSE_ALIASES)
        position = _vocab(f.get("position"), POSITIONS, _POSITION_ALIASES)
        if pose is None and position is None:
            continue
        role = f.get("role") if isinstance(f.get("role"), str) else ""
        role = " ".join(re.sub(r"[^A-Za-z\s-]", " ", role).split()[:4]).lower()
        if not role or m.find(role):
            role = "figure"
        fig = {"role": role}
        if pose:
            fig["pose"] = pose
        if position:
            fig["position"] = position
        figures.append(fig)
    n = obj.get("n_figures")
    if not (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 20):
        n = len(figures) or None
    if n is None:
        return None
    return {"n_figures": n, "figures": figures}


def layout_phrases(layout: dict) -> list[str]:
    """``["two figures", "a kneeling king on the left", ...]`` (deterministic; at most 4 figure phrases)."""
    n = layout["n_figures"]
    out = [f"{_NUMBER_WORDS.get(n, 'many')} figure{'' if n == 1 else 's'}"]
    for f in layout.get("figures", [])[:MAX_LAYOUT_FIGURES]:
        role, pose, pos = f.get("role", "figure"), f.get("pose"), f.get("position")
        if pose == "arms_raised":
            body = f"{role} with arms raised"
        elif pose:
            body = f"{pose} {role}"
        else:
            body = role
        article = "an" if body[:1].lower() in "aeiou" else "a"
        out.append(f"{article} {body}" + (f" {_POSITION_PHRASE[pos]}" if pos else ""))
    return out


# --------------------------------------------------------------------------- expand

def _llm_phrases(text: str, scene: dict, mode: str, client: LLMClient, cache: Cache | None, pv: str,
                 model: str, kb_version: str, matcher: NameMatcher, state: dict) -> dict:
    """Run the LLM chain. Returns phrases/layout/leaks; raises on any failure (caller falls back)."""
    system = load_system_prompt(pv)
    user = build_user_message(text, scene, mode)

    def call(u: str) -> str:
        key = cache_key(pv, model, kb_version, mode, u)
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                state["cache_hits"] += 1
                return hit
        reply = client.complete(system, u)
        state["llm_calls"] += 1
        if not isinstance(reply, str):
            raise _InvalidReply(f"LLM returned {type(reply).__name__}, not str")
        if cache is not None:
            cache.put(key, reply, model=model, prompt_version=pv, mode=mode, kb_version=kb_version)
        return reply

    obj = parse_reply(call(user))
    if obj is None:
        state["json_retry"] = True
        obj = parse_reply(call(f"{user}\n\n{RETRY_INSTRUCTION}."))
        if obj is None:
            raise _InvalidReply("no valid JSON after one retry")

    phrases = obj["phrases"]
    raw_layout = obj.get("layout") if mode == "llm_kb_layout" else None
    figs = raw_layout.get("figures") if isinstance(raw_layout, dict) else None
    roles = [f["role"] for f in figs if isinstance(f, dict) and isinstance(f.get("role"), str)] \
        if isinstance(figs, list) else []
    leaked = matcher.find(" ; ".join([*phrases, *roles]))
    state["leaked"] = leaked

    if leaked:
        state["name_repair"] = True
        repair = (f"{user}\n\nYour previous phrases were: {json.dumps(phrases, ensure_ascii=False)}\n"
                  f"They contain proper names ({', '.join(leaked)}). {REPAIR_INSTRUCTION} {RETRY_INSTRUCTION}.")
        try:
            fixed = parse_reply(call(repair))
        except Exception:  # noqa: BLE001 - a failed repair keeps the first reply; offending phrases are dropped below
            fixed = None
        if fixed is not None:
            phrases = fixed["phrases"]
            if mode == "llm_kb_layout" and isinstance(fixed.get("layout"), dict):
                raw_layout = fixed["layout"]

    phrases, scrubbed = scrub_phrases(phrases, matcher)
    if not phrases:
        raise _InvalidReply("every LLM phrase contained a proper name")
    layout = parse_layout(raw_layout, matcher) if mode == "llm_kb_layout" else None
    lead = [p for p in layout_phrases(layout) if not matcher.find(p)] if layout else []
    return {"phrases": phrases, "lead": lead, "layout": layout, "leaked": leaked, "scrubbed": scrubbed}


def expand(text: str, scene: dict | None = None, mode: str = "llm_kb", *, llm: LLMClient | None = None,
           cache: Cache | None = None, kb_path: str | None = None, condition: str | None = None,
           split: str = "", count_fn: CountFn | None = None, cfg: dict | None = None,
           prompt_version: str | None = None, matcher: NameMatcher | None = None) -> dict:
    """Expand one scene into an ``ExpandedPrompt`` dict (the 16 contract fields of §6.4 plus extras).

    - ``scene``: C's SceneSpec; parsed from ``text`` with the resolved parser when ``None`` (not for raw).
    - ``llm``: an ``LLMClient`` (else the default set by ``set_default_llm``); required by LLM modes
      unless config ``llm.backend == "none"``, which falls back to the template.
    - ``cache``: an ``expander.cache.Cache``; ``None`` disables caching.
    - ``kb_path``: KB file for the parser and the mood lookup (default: ``_cfg.default_kb_path()``).
    - ``condition``: condition ID (default from ``mode``, see ``resolve_condition``); ``split``: "dev"/"heldout".
    - ``count_fn``: token counter (default: the ONE counter, ``common.tokens.n_tokens``).
    - ``cfg``: config dict (default: ``_cfg.load_config()``); ``prompt_version`` overrides ``cfg.llm``.
    """
    t0 = time.perf_counter()
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("expand() needs non-empty text")
    cfg = cfg if cfg is not None else load_config()
    count = count_fn or get_counter()
    matcher = matcher or default_matcher()
    if scene is None and mode != "raw":
        scene = get_parser()(text, kb_path=kb_path)
    scene = scene or {}
    trigger, tail = cfg["trigger"], cfg.get("style_tail", "")

    rec: dict = {
        "scene_id": scene.get("id", "") or "",
        "condition": resolve_condition(mode, condition),
        "split": split or "",
        "negative": cfg["negative"],
        "layout": None,
        "kb_version": scene.get("kb_version") or "none",
        "mode": mode,
        "phrases_scrubbed": [],
        "json_retry": False,
        "name_repair": False,
        "cache_hits": 0,
        "llm_calls": 0,
        "error": None,
        "note": None,
    }

    if mode == "raw":
        positive = ", ".join([trigger, text.strip()] + ([tail] if tail else []))
        rec.update(positive=positive, phrases_kept=[text.strip()], phrases_dropped=[], source="template",
                   llm_model="none", prompt_version="raw", names_leaked=matcher.find(text), note="raw")
    elif mode == "template":
        tpl = template_prompt(scene, cfg, load_kb(kb_path), count_fn=count, matcher=matcher)
        rec.update(positive=tpl["positive"], phrases_kept=tpl["phrases_kept"],
                   phrases_dropped=tpl["phrases_dropped"], names_leaked=tpl["names_leaked"],
                   phrases_scrubbed=tpl["phrases_scrubbed"], source="template", llm_model="none",
                   prompt_version=TEMPLATE_PROMPT_VERSION)
    else:
        client = _resolve_llm(llm, cfg)
        pv = prompt_version or cfg["llm"]["prompt_version"]
        if mode == "llm_kb_layout" and not pv.endswith("_layout"):
            pv += "_layout"
        model = (getattr(client, "model_id", None) or cfg["llm"]["model"]) if client is not None else "none"
        rec.update(llm_model=model, prompt_version=pv)
        state = {"json_retry": False, "name_repair": False, "cache_hits": 0, "llm_calls": 0, "leaked": []}
        out = None
        try:
            if client is None:
                raise LLMUnavailableError('config llm.backend is "none"')
            out = _llm_phrases(text, scene, mode, client, cache, pv, model, rec["kb_version"], matcher, state)
        except Exception as e:  # noqa: BLE001 - never crash because of the LLM: template fallback
            rec["error"] = f"{type(e).__name__}: {e}"
        llm_leaked = state.pop("leaked")
        rec.update(state)
        if out is not None:
            positive, kept, dropped = assemble(trigger, [*out["lead"], *out["phrases"]], tail,
                                               int(cfg["max_prompt_tokens"]), count_fn=count)
            rec.update(positive=positive, phrases_kept=kept, phrases_dropped=dropped, source="llm",
                       names_leaked=out["leaked"], phrases_scrubbed=out["scrubbed"], layout=out["layout"])
        else:
            tpl = template_prompt(scene, cfg, load_kb(kb_path), count_fn=count, matcher=matcher)
            rec.update(positive=tpl["positive"], phrases_kept=tpl["phrases_kept"],
                       phrases_dropped=tpl["phrases_dropped"], phrases_scrubbed=tpl["phrases_scrubbed"],
                       names_leaked=list(dict.fromkeys([*llm_leaked, *tpl["names_leaked"]])), source="fallback")

    rec["token_count"] = count(rec["positive"])
    rec["trigger_first"] = rec["positive"].startswith(trigger)
    rec["names_after_repair"] = matcher.find(rec["positive"])
    rec["latency_s"] = round(time.perf_counter() - t0, 4)
    return {**{k: rec[k] for k in CONTRACT_FIELDS}, **{k: v for k, v in rec.items() if k not in CONTRACT_FIELDS}}
