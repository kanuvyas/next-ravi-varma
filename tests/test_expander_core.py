"""Deterministic expander machinery: config/shims, assemble, template, cache, conditions, stub, LLM base."""

import importlib.util
import json
import re

import pytest

from expander import _cfg
from expander.assemble import assemble, clean_phrases, token_count
from expander.cache import Cache, cache_key
from expander.expand import MODE_TO_CONDITION, resolve_condition
from expander.llm_base import SYSTEM_PROMPT_V1, FakeLLM, LLMClient, load_system_prompt
from expander.names import SEED_PATH, NameMatcher, load_names
from expander.template import (
    FALLBACK_MOOD_LIGHTING,
    mood_phrase,
    template_phrases,
    template_prompt,
)

CFG = json.loads((_cfg.SHIM_DIR / "m3.default.json").read_text())
KB = json.loads((_cfg.SHIM_DIR / "stub_kb.json").read_text())
SEED = NameMatcher(load_names(SEED_PATH))
TRIGGER, TAIL = CFG["trigger"], CFG["style_tail"]


def fake_count(s: str) -> int:
    """Deterministic stand-in for CLIP: 1 per punctuation mark, 1 per 6 letters of each word (rounded up)."""
    return sum(-(-len(w) // 6) if w[0].isalnum() else 1 for w in re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", s))


SCENE = {"id": "t1", "raw": "Karna weeping over the fallen body of Arjun on the battlefield", "epic": "Mahabharata",
         "characters": [{"mention": "Karna", "canonical": "Karna", "visual": "bare-chested warrior, golden breastplate",
                         "coverage": "medium"},
                        {"mention": "Arjun", "canonical": "Arjuna", "visual": "young warrior, long bow",
                         "coverage": "low"}],
         "setting": {"name": "kurukshetra", "visual": "open battlefield, chariots and banners"},
         "action": "weeping over the fallen body of Arjun", "mood": "grief", "unknown_entities": [],
         "kb_version": "stub0"}


# ------------------------------------------------------------------ config and shims

def test_config_shim_values():
    assert CFG["trigger"] == "rrvarma style"
    assert CFG["style_tail"].startswith("oil painting, classical Indian academic realism") and "_style_tail_note" in CFG
    assert CFG["negative"] == "lowres, blurry, deformed, extra limbs, bad anatomy, text, watermark"
    assert CFG["max_prompt_tokens"] == 70
    m = CFG["models"]["rrv_v1"]
    assert m["scale"] == 0.8 and m["sha256"].startswith("067ee19f") and "D_lr2e-4 step 800" in m["source"]
    assert CFG["llm"]["model"] == "Qwen/Qwen3-4B-Instruct-2507" and CFG["llm"]["temperature"] == 0.0
    assert CFG["llm"]["model_large"] == "Qwen/Qwen2.5-7B-Instruct" and CFG["llm"]["prompt_version"] == "v1"


def test_config_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("M3_CONFIG", raising=False)
    expected = _cfg.repo_root() / "config" / "m3.json"
    assert _cfg.config_path() == (expected if expected.exists() else _cfg.SHIM_DIR / "m3.default.json")
    p = tmp_path / "m3.json"
    p.write_text(json.dumps({**CFG, "max_prompt_tokens": 50}))
    assert _cfg.load_config(p)["max_prompt_tokens"] == 50
    monkeypatch.setenv("M3_CONFIG", str(p))
    assert _cfg.load_config()["max_prompt_tokens"] == 50
    cfg = _cfg.load_config()
    cfg["trigger"] = "mutated"
    assert _cfg.load_config()["trigger"] == "rrvarma style"  # callers get a fresh dict


def test_parser_and_kb_resolution():
    parse = _cfg.get_parser()
    assert callable(parse)
    assert _cfg.default_kb_path().exists()
    spec = parse("Rama and Sita in the forest, from the Ramayana", "x1")
    canon = [c["canonical"] for c in spec["characters"]]
    assert spec["id"] == "x1" and {"Rama", "Sita"} <= set(canon) and len(canon) == len(set(canon))
    for key in ("id", "raw", "epic", "characters", "setting", "action", "mood", "unknown_entities", "kb_version"):
        assert key in spec


def test_shim_parser_on_fixtures():
    from expander._shims.stub_parser import parse_scene
    fixtures = json.loads((_cfg.SHIM_DIR / "fixtures_scenes.json").read_text())
    assert len(fixtures) == 3
    for f in fixtures:
        spec = parse_scene(f["text"], f["id"])
        assert [c["canonical"] for c in spec["characters"]] == f["expected_characters"]
        assert spec["epic"] == f["expected_epic"] and spec["kb_version"] == "stub0"
        assert SEED.find(spec["action"]) == []


def test_shim_counter_is_lazy():
    import expander._shims.tokens as t  # importing must not need transformers
    assert callable(t.n_tokens)


# ------------------------------------------------------------------ assemble

def test_assemble_trigger_first_tail_last_budget():
    phrases = ["a crowned king", "kneeling in prayer", "a pillared hall", "dim light"]
    pos, kept, dropped = assemble(TRIGGER, phrases, TAIL, 70, count_fn=fake_count)
    assert pos.startswith(TRIGGER + ", ") and pos.endswith(", " + TAIL)
    assert fake_count(pos) <= 70 and kept == phrases and dropped == []


def test_assemble_skips_long_phrase_keeps_later_short_ones():
    base = fake_count(f"{TRIGGER}, {TAIL}")
    long = "an extraordinarily elaborate description " * 4
    pos, kept, dropped = assemble(TRIGGER, ["a king", long.strip(), "a bow"], TAIL, base + 6, count_fn=fake_count)
    assert kept == ["a king", "a bow"] and dropped == [long.strip()]
    assert fake_count(pos) <= base + 6


def test_assemble_partition_preserves_order():
    phrases = [f"phrase number {i}" for i in range(20)]
    _, kept, dropped = assemble(TRIGGER, phrases, TAIL, 50, count_fn=fake_count)
    assert sorted(kept + dropped, key=phrases.index) == phrases
    assert kept == [p for p in phrases if p in kept] and dropped == [p for p in phrases if p in dropped]
    assert dropped


def test_assemble_empty_tail():
    pos, _, _ = assemble(TRIGGER, ["a king"], "", 70, count_fn=fake_count)
    assert pos == "rrvarma style, a king"
    assert assemble(TRIGGER, [], "", 70, count_fn=fake_count)[0] == "rrvarma style"


def test_assemble_asserts_when_trigger_and_tail_too_long():
    with pytest.raises(AssertionError, match="trigger \\+ tail"):
        assemble(TRIGGER, ["a king"], TAIL, 5, count_fn=fake_count)


def test_token_count_and_clean_phrases():
    assert token_count("a b c", count_fn=fake_count) == 3
    assert clean_phrases(["  a  king. ", "A King", "", None, 3, '"bow"', "x;"]) == ["a king", "bow", "x"]


# ------------------------------------------------------------------ template

def test_template_order_and_mood_from_kb():
    phrases = template_phrases(SCENE, KB)
    assert phrases == ["bare-chested warrior", "golden breastplate", "young warrior", "long bow",
                       "weeping over the fallen body of Arjun", "open battlefield", "chariots and banners",
                       "dim dramatic lighting", "shadowed faces"]


def test_template_mood_sources():
    assert mood_phrase({**SCENE, "mood_visual": "lamp-lit night"}, KB) == "lamp-lit night"
    assert mood_phrase(SCENE, None) == FALLBACK_MOOD_LIGHTING["grief"] == "dim somber light"
    assert mood_phrase({**SCENE, "mood": "Grief"}, {"moods": {"grief": {"lighting": "grey dawn"}}}) == "grey dawn"
    assert mood_phrase({**SCENE, "mood": None}, KB) == ""
    assert mood_phrase({**SCENE, "mood": "unheard-of"}, KB) == ""


def test_template_handles_missing_fields():
    sparse = {"id": "t2", "characters": [{"mention": "X", "visual": ""}], "setting": None, "action": "",
              "mood": "", "kb_version": "none"}
    assert template_phrases(sparse, KB) == []
    pos = template_prompt(sparse, CFG, KB, count_fn=fake_count, matcher=SEED)["positive"]
    assert pos == f"{TRIGGER}, {TAIL}"


def test_template_strips_names_and_is_deterministic():
    a = template_prompt(SCENE, CFG, KB, count_fn=fake_count, matcher=SEED)
    b = template_prompt(json.loads(json.dumps(SCENE)), CFG, KB, count_fn=fake_count, matcher=SEED)
    assert a == b
    assert a["names_leaked"] == ["Arjuna"]
    assert a["phrases_scrubbed"] == ["weeping over the fallen body of Arjun"]
    assert "weeping over the fallen body" in a["phrases_kept"]
    assert SEED.find(a["positive"]) == []
    assert a["positive"].startswith(TRIGGER) and a["positive"].endswith(TAIL)
    assert fake_count(a["positive"]) <= CFG["max_prompt_tokens"]


# ------------------------------------------------------------------ cache

def test_cache_key_is_stable_and_field_sensitive():
    k = cache_key("v1", "m", "kb0.2", "llm_kb", "hello")
    assert k == cache_key("v1", "m", "kb0.2", "llm_kb", "hello")
    assert re.fullmatch(r"[0-9a-f]{64}", k)
    variants = [("v2", "m", "kb0.2", "llm_kb", "hello"), ("v1", "m2", "kb0.2", "llm_kb", "hello"),
                ("v1", "m", "kb0.3", "llm_kb", "hello"), ("v1", "m", "kb0.2", "llm_nokb", "hello"),
                ("v1", "m", "kb0.2", "llm_kb", "hello!")]
    assert len({k, *(cache_key(*v) for v in variants)}) == 6


def test_cache_key_golden():
    import hashlib
    assert cache_key("v1", "m", "kb", "llm_kb", "u") == hashlib.sha256(b"v1|m|kb|llm_kb|u").hexdigest()


def test_cache_roundtrip_atomic_sorted(tmp_path):
    p = tmp_path / "sub" / "cache.json"
    c = Cache(p)
    assert c.get("b") is None and c.misses == 1
    c.put("b", "reply-b", model="m")
    c.put("a", "reply-a")
    assert p.exists() and [x.name for x in p.parent.iterdir()] == ["cache.json"]  # no temp files left
    text = p.read_text()
    assert text.index('"a"') < text.index('"b"')  # sorted keys
    c2 = Cache(p)
    assert len(c2) == 2 and "a" in c2 and c2.get("b") == "reply-b" and c2.hits == 1
    assert json.loads(text)["b"]["model"] == "m"


def test_cache_in_memory_and_manual_save(tmp_path):
    mem = Cache(None)
    mem.put("k", "v")
    mem.save()
    assert mem.get("k") == "v"
    p = tmp_path / "c.json"
    lazy = Cache(p, autosave=False)
    lazy.put("k", "v")
    assert not p.exists()
    lazy.save()
    assert Cache(p).get("k") == "v"


# ------------------------------------------------------------------ conditions, stub, LLM base

@pytest.mark.parametrize("args, expected", [
    (("llm_kb", "c3_custom"), "c3_custom"),
    (("llm_kb", None, "outputs/expansions/c6_llm_kbllm_dev.json"), "c6_llm_kbllm"),
    (("llm_kb", None, "outputs/expansions/c3_7b_dev.json"), "c3_7b"),
    (("llm_kb", None, "c3_neg_heldout.json"), "c3_neg"),
    (("llm_kb", None, "c6s_llm_kbshort_dev.json"), "c6s_llm_kbshort"),
    (("llm_kb", None, "/tmp/smoke.json"), "c3_llm_kb"),
    (("raw",), "c1_lora_raw"), (("template",), "c2_template"), (("llm_kb",), "c3_llm_kb"),
    (("llm_nokb",), "c4_llm_nokb"), (("llm_kb_layout",), "c5_llm_kb_layout"),
])
def test_resolve_condition(args, expected):
    assert resolve_condition(*args) == expected


def test_resolve_condition_rejects_unknown_mode():
    with pytest.raises(ValueError):
        resolve_condition("nope")
    assert set(MODE_TO_CONDITION) == {"raw", "template", "llm_kb", "llm_nokb", "llm_kb_layout"}


def test_stub_expand():
    from expander.stub_expand import CFG as STUB_CFG
    from expander.stub_expand import expand as stub
    r = stub("Karna on the battlefield", {"id": "s01"})
    assert len(r) == 16 and r["scene_id"] == "s01" and r["token_count"] == -1
    assert r["positive"] == f"{STUB_CFG['trigger']}, Karna on the battlefield, {STUB_CFG['style_tail']}"
    assert r["positive"].startswith("rrvarma style")


def test_fake_llm_and_protocol():
    f = FakeLLM(["a", ValueError("boom")])
    assert isinstance(f, LLMClient)
    assert f.complete("s", "u") == "a"
    with pytest.raises(ValueError):
        f.complete("s", "u2")
    with pytest.raises(RuntimeError):
        f.complete("s", "u3")
    assert f.calls == [("s", "u"), ("s", "u2"), ("s", "u3")]
    g = FakeLLM(lambda s, u: u.upper(), model_id="fake-1")
    assert g.complete("s", "hi") == "HI" and g.model_id == "fake-1"
    g.close()
    assert g.closed


def test_load_system_prompt(tmp_path):
    assert load_system_prompt("v1", tmp_path) == SYSTEM_PROMPT_V1
    assert SYSTEM_PROMPT_V1.startswith("You write scene descriptions") and '{"phrases"' in SYSTEM_PROMPT_V1
    assert "pose: one of" in load_system_prompt("v1_layout", tmp_path)
    (tmp_path / "system_v9.txt").write_text("custom prompt\n")
    assert load_system_prompt("v9", tmp_path) == "custom prompt"
    with pytest.raises(KeyError):
        load_system_prompt("v404", tmp_path)


@pytest.mark.slow
def test_real_clip_counter():
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers not installed")
    from expander._shims.tokens import n_tokens
    try:
        n_tokens("warm up")
    except Exception as e:  # noqa: BLE001 - offline environments
        pytest.skip(f"tokenizer unavailable: {e}")
    assert n_tokens("rrvarma style") == 4
    assert n_tokens(TAIL) == 20
    assert n_tokens(f"{TRIGGER}, {TAIL}") == 25
