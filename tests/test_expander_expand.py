"""expand() end to end with FakeLLM: validation chain, fallback, name repair, KB block, layout, records."""

import importlib.util
import json
import re

import pytest

from expander import _cfg
from expander import expand as expand_mod
from expander.cache import Cache
from expander.expand import (
    CONTRACT_FIELDS,
    LLMUnavailableError,
    expand,
    parse_layout,
    set_default_llm,
)
from expander.llm_base import FakeLLM
from expander.names import SEED_PATH, NameMatcher, load_names

CFG = json.loads((_cfg.SHIM_DIR / "m3.default.json").read_text())
KB_PATH = str(_cfg.SHIM_DIR / "stub_kb.json")
SCHEMA = json.loads((_cfg.repo_root() / "expander" / "schemas" / "expanded_prompt.schema.json").read_text())
SEED = NameMatcher(load_names(SEED_PATH))
TEXT = "Hanuman kneeling before Rama and Sita in the royal hall"
SCENE = {"id": "t1", "raw": TEXT, "epic": "Ramayana",
         "characters": [{"mention": "Hanuman", "canonical": "Hanuman", "visual": "monkey-faced figure, gold crown",
                         "coverage": "medium"},
                        {"mention": "Rama", "canonical": "Rama", "visual": "blue-skinned crowned prince",
                         "coverage": "medium"},
                        {"mention": "Sita", "canonical": "Sita", "visual": "", "coverage": "none"}],
         "setting": {"name": "palace", "visual": "pillared royal hall"}, "action": "kneeling before and in the royal hall",
         "mood": "devotion", "unknown_entities": [], "kb_version": "stub0"}
GOOD = '{"phrases": ["a monkey-faced figure kneeling", "a blue-skinned crowned prince", "pillared royal hall"]}'

TYPES = {"scene_id": str, "condition": str, "split": str, "positive": str, "negative": str, "phrases_kept": list,
         "phrases_dropped": list, "token_count": int, "trigger_first": bool, "names_leaked": list, "source": str,
         "llm_model": str, "prompt_version": str, "kb_version": str, "latency_s": float, "names_after_repair": list}


def fake_count(s: str) -> int:
    return sum(-(-len(w) // 6) if w[0].isalnum() else 1 for w in re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", s))


def run(mode="llm_kb", replies=(GOOD,), scene=SCENE, **kw):
    llm = kw.pop("llm", None) or FakeLLM(list(replies))
    rec = expand(TEXT, scene, mode, llm=llm, cfg=CFG, count_fn=fake_count, matcher=SEED, kb_path=KB_PATH, **kw)
    check_record(rec)
    return rec, llm


def check_record(rec):
    for field in (*CONTRACT_FIELDS, "names_after_repair"):
        assert field in rec, field
    for field, t in TYPES.items():
        assert isinstance(rec[field], t), (field, rec[field])
    assert rec["layout"] is None or isinstance(rec["layout"], dict)
    assert rec["source"] in {"llm", "template", "fallback"}
    assert rec["trigger_first"] is True and rec["positive"].startswith(CFG["trigger"])
    assert rec["negative"] == CFG["negative"]
    if rec["mode"] != "raw":
        assert rec["token_count"] <= CFG["max_prompt_tokens"]
        assert rec["names_after_repair"] == []
        assert rec["positive"].endswith(CFG["style_tail"])
    if importlib.util.find_spec("jsonschema"):
        import jsonschema
        jsonschema.validate(rec, SCHEMA)


def test_valid_json_is_used():
    rec, _ = run()
    assert rec["source"] == "llm" and rec["llm_calls"] == 1 and not rec["json_retry"] and rec["error"] is None
    assert rec["phrases_kept"] == json.loads(GOOD)["phrases"]
    assert rec["condition"] == "c3_llm_kb" and rec["llm_model"] == CFG["llm"]["model"]
    assert rec["prompt_version"] == "v1" and rec["kb_version"] == "stub0"
    assert rec["positive"] == ", ".join([CFG["trigger"], *rec["phrases_kept"], CFG["style_tail"]])


def test_code_fence_and_prose_parsed_without_retry():
    rec, _ = run(replies=["Here you go:\n```json\n" + GOOD + "\n```\nHope this helps {not json}"])
    assert rec["source"] == "llm" and rec["llm_calls"] == 1 and not rec["json_retry"]


def test_prose_then_valid_json_retries_once():
    rec, llm = run(replies=["Sure, here are some phrases: a kneeling figure.", GOOD])
    assert rec["source"] == "llm" and rec["json_retry"] and rec["llm_calls"] == 2
    assert llm.calls[1][1].endswith("Return ONLY valid JSON.")


def test_garbage_twice_falls_back_to_template():
    rec, _ = run(replies=["no json here", '{"phrases": []}'])
    assert rec["source"] == "fallback" and rec["json_retry"] and "no valid JSON" in rec["error"]
    assert rec["llm_model"] == CFG["llm"]["model"] and rec["prompt_version"] == "v1"
    assert "monkey-faced figure" in rec["phrases_kept"]


def test_raising_llm_falls_back():
    rec, _ = run(replies=[RuntimeError("CUDA out of memory")])
    assert rec["source"] == "fallback" and "CUDA out of memory" in rec["error"]


def test_leak_repaired_by_reask():
    leaky = '{"phrases": ["Hanuman kneeling", "a blue-skinned crowned prince"]}'
    rec, llm = run(replies=[leaky, GOOD])
    assert rec["source"] == "llm" and rec["name_repair"]
    assert rec["names_leaked"] == ["Hanuman"] and rec["names_after_repair"] == [] and rec["phrases_scrubbed"] == []
    repair_msg = llm.calls[1][1]
    assert "Remove proper names; describe by clothing, pose and setting." in repair_msg and "Hanuman" in repair_msg


def test_leak_not_repaired_drops_phrases():
    leaky = '{"phrases": ["Hanuman kneeling", "a blue-skinned crowned prince", "Sita smiling"]}'
    rec, _ = run(replies=[leaky, leaky])
    assert rec["source"] == "llm" and rec["name_repair"]
    assert rec["names_leaked"] == ["Hanuman", "Sita"] and rec["names_after_repair"] == []
    assert rec["phrases_scrubbed"] == ["Hanuman kneeling", "Sita smiling"]
    assert rec["phrases_kept"] == ["a blue-skinned crowned prince"] and rec["phrases_dropped"] == []


def test_repair_failure_keeps_first_reply_minus_names():
    leaky = '{"phrases": ["Hanuman kneeling", "a blue-skinned crowned prince"]}'
    rec, _ = run(replies=[leaky, RuntimeError("boom")])
    assert rec["source"] == "llm" and rec["phrases_kept"] == ["a blue-skinned crowned prince"]


def test_all_phrases_named_falls_back():
    leaky = '{"phrases": ["Hanuman kneeling", "Rama standing"]}'
    rec, _ = run(replies=[leaky, leaky])
    assert rec["source"] == "fallback" and rec["names_leaked"][:2] == ["Hanuman", "Rama"]


def test_kb_and_nokb_differ_only_by_reference_block():
    _, kb_llm = run("llm_kb")
    _, nokb_llm = run("llm_nokb")
    (sys_kb, user_kb), (sys_nokb, user_nokb) = kb_llm.calls[0], nokb_llm.calls[0]
    assert sys_kb == sys_nokb
    assert user_nokb == TEXT
    assert user_kb.startswith(user_nokb)
    block = user_kb[len(user_nokb):]
    assert block == ("\n\nReference descriptions:\nHanuman -> monkey-faced figure, gold crown\n"
                     "Rama -> blue-skinned crowned prince\nsetting -> pillared royal hall")


def test_layout_parsed_and_prepended():
    reply = json.dumps({"phrases": ["a monkey-faced figure", "pillared royal hall"],
                        "layout": {"n_figures": 2, "figures": [
                            {"role": "king", "pose": "kneeling", "position": "left"},
                            {"role": "queen", "pose": "dancing", "position": "centre"},
                            {"role": "Rama", "pose": "standing", "position": "upstairs"},
                            {"role": "ghost", "pose": "hovering", "position": "nowhere"}]}})
    rec, llm = run("llm_kb_layout", replies=[reply, reply])
    assert rec["condition"] == "c5_llm_kb_layout" and rec["prompt_version"] == "v1_layout"
    assert "pose: one of" in llm.calls[0][0]
    assert rec["layout"] == {"n_figures": 2, "figures": [
        {"role": "king", "pose": "kneeling", "position": "left"},
        {"role": "queen", "position": "center"},
        {"role": "figure", "pose": "standing"}]}
    assert rec["positive"].startswith("rrvarma style, two figures, a kneeling king on the left, "
                                      "a queen in the center, a standing figure, a monkey-faced figure")
    assert rec["names_leaked"] == ["Rama"]


def test_parse_layout_rejects_garbage():
    assert parse_layout(None) is None and parse_layout({"figures": "x"}) is None
    assert parse_layout({"n_figures": True, "figures": []}) is None
    assert parse_layout({"n_figures": 3}, SEED) == {"n_figures": 3, "figures": []}


def test_layout_absent_in_other_modes():
    reply = json.dumps({"phrases": ["a kneeling figure"], "layout": {"n_figures": 1, "figures": []}})
    rec, _ = run("llm_kb", replies=[reply])
    assert rec["layout"] is None


def test_cache_makes_second_run_free(tmp_path):
    cache = Cache(tmp_path / "cache.json")
    first, llm1 = run(cache=cache, replies=["prose", GOOD])
    second, llm2 = run(cache=Cache(tmp_path / "cache.json"), replies=[])
    assert llm1.calls and llm2.calls == []
    assert second["positive"] == first["positive"] and second["cache_hits"] == 2 and second["llm_calls"] == 0
    _, llm3 = run("llm_nokb", cache=cache, replies=[GOOD])
    assert len(llm3.calls) == 1  # mode is part of the key


def test_template_raw_and_overrides():
    tpl, _ = run("template", replies=[], split="dev", condition="c2_custom")
    assert tpl["source"] == "template" and tpl["llm_model"] == "none" and tpl["prompt_version"] == "tpl2"
    assert tpl["template_version"] == "tpl2"
    assert tpl["condition"] == "c2_custom" and tpl["split"] == "dev"
    assert tpl["names_leaked"] == [] and "pillared royal hall" in tpl["phrases_kept"]
    raw, _ = run("raw", replies=[], scene=None)
    assert raw["positive"] == f"{CFG['trigger']}, {TEXT}, {CFG['style_tail']}"
    assert raw["names_leaked"] == ["Hanuman", "Rama", "Sita"] and raw["note"] == "raw"
    assert raw["condition"] == "c1_lora_raw" and raw["prompt_version"] == "raw" and raw["scene_id"] == ""


def test_scene_none_is_parsed():
    rec, _ = run("template", replies=[], scene=None)
    assert rec["kb_version"] != "none" and rec["names_after_repair"] == []


def test_missing_llm_raises_unless_backend_none():
    with pytest.raises(LLMUnavailableError):
        expand(TEXT, SCENE, "llm_kb", cfg=CFG, count_fn=fake_count, matcher=SEED)
    cfg_none = {**CFG, "llm": {**CFG["llm"], "backend": "none"}}
    rec = expand(TEXT, SCENE, "llm_kb", cfg=cfg_none, count_fn=fake_count, matcher=SEED, kb_path=KB_PATH)
    check_record(rec)
    assert rec["source"] == "fallback" and rec["llm_model"] == "none"


def test_default_llm_and_close():
    llm = FakeLLM([GOOD], model_id="fake-model")
    set_default_llm(llm)
    try:
        rec = expand(TEXT, SCENE, "llm_kb", cfg=CFG, count_fn=fake_count, matcher=SEED)
        assert rec["source"] == "llm" and rec["llm_model"] == "fake-model"
    finally:
        import expander
        expander.close()
    assert llm.closed and expand_mod._DEFAULT_LLM is None


def test_bad_inputs():
    with pytest.raises(ValueError):
        expand("   ", SCENE, "template", cfg=CFG, count_fn=fake_count)
    with pytest.raises(ValueError):
        expand(TEXT, SCENE, "llm_magic", cfg=CFG, count_fn=fake_count)


def test_budget_respected_with_many_phrases():
    many = json.dumps({"phrases": [f"a richly ornamented figure number {i}" for i in range(15)]})
    rec, _ = run(replies=[many])
    assert rec["phrases_dropped"] and fake_count(rec["positive"]) <= CFG["max_prompt_tokens"]
