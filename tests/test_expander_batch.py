"""batch_expand.py and check_expansions.py end to end on CPU (``--llm fake``, approximate counter, no downloads)."""

import importlib.util
import json
from pathlib import Path

import pytest

from expander import _cfg
from expander import batch_expand as be
from expander import check_expansions as ce
from expander.expand import CONTRACT_FIELDS, LLM_MODES, MODES
from expander.llm import FAKE_MODEL_ID, EchoFakeLLM, LazyClient
from expander.names import default_matcher

ROOT = _cfg.repo_root()
FIXTURES = ROOT / "expander" / "_shims" / "fixtures_scenes.json"
DEV_D = ROOT / "expander" / "dev_scenes_D.json"
CFG = json.loads((_cfg.SHIM_DIR / "m3.default.json").read_text())
SCHEMA = json.loads((ROOT / "expander" / "schemas" / "expanded_prompt.schema.json").read_text())
CONDITION = {"raw": "c1_lora_raw", "template": "c2_template", "llm_kb": "c3_llm_kb", "llm_nokb": "c4_llm_nokb",
             "llm_kb_layout": "c5_llm_kb_layout"}


@pytest.fixture(autouse=True)
def _no_env_config(monkeypatch):
    monkeypatch.delenv("M3_CONFIG", raising=False)


def run(tmp_path, scenes, mode, out_name=None, *extra):
    out = tmp_path / (out_name or f"{CONDITION[mode]}_dev.json")
    argv = ["--scenes", str(scenes), "--mode", mode, "--out", str(out), "--llm", "fake", "--counter", "approx",
            "--cache", str(tmp_path / "cache.json"), *extra]
    return be.run(argv), out


def load(path):
    return json.loads(Path(path).read_text())


def check_file(path, scenes, mode):
    recs = load(path)
    ids = [s["id"] for s in json.loads(Path(scenes).read_text())]
    assert [r["scene_id"] for r in recs] == sorted(ids)
    for r in recs:
        for f in (*CONTRACT_FIELDS, "names_after_repair"):
            assert f in r, f
        assert r["mode"] == mode and r["condition"] == CONDITION[mode]
        assert r["trigger_first"] and r["positive"].startswith(CFG["trigger"])
        assert r["token_counter"] == "approx"
        if mode != "raw":
            assert r["names_after_repair"] == [] and default_matcher().find(r["positive"]) == []
            assert r["token_count"] <= CFG["max_prompt_tokens"] and r["positive"].endswith(CFG["style_tail"])
        if mode in LLM_MODES:
            assert r["source"] == "llm" and r["llm_model"] == FAKE_MODEL_ID
        if mode == "template":
            assert r["prompt_version"] == "tpl2" and r["template_version"] == "tpl2"
        if mode == "llm_kb_layout":
            assert r["layout"] and r["prompt_version"] == "v1_layout"
        if importlib.util.find_spec("jsonschema"):
            import jsonschema
            jsonschema.validate(r, SCHEMA)
    return recs


# ------------------------------------------------------------------ batch_expand

@pytest.mark.parametrize("scenes", [FIXTURES, DEV_D], ids=["fixtures", "dev_scenes_D"])
@pytest.mark.parametrize("mode", MODES)
def test_every_mode_end_to_end(tmp_path, scenes, mode):
    summary, out = run(tmp_path, scenes, mode)
    recs = check_file(out, scenes, mode)
    assert summary["n"] == summary["new"] == len(recs) and summary["kept"] == 0
    assert summary["condition"] == CONDITION[mode]
    assert {r["split"] for r in recs} == {"dev"}  # from each scene's "split"
    if mode in LLM_MODES:
        assert summary["llm_calls"] >= len(recs)
    report = ce.summarize(recs, trigger=CFG["trigger"], max_tokens=CFG["max_prompt_tokens"])
    if mode != "raw":
        assert report["pct_valid"] == 100.0 and report["n_leaks_after"] == 0


def test_main_prints_one_summary_line(tmp_path, capsys):
    out = tmp_path / "c2_template_dev.json"
    assert be.main(["--scenes", str(FIXTURES), "--mode", "template", "--out", str(out), "--counter", "approx"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("[batch_expand] c2_template: 3 records")


def test_resume_skips_done_scenes_and_cache_hits(tmp_path):
    s1, out = run(tmp_path, DEV_D, "llm_kb", None, "--limit", "3")
    assert s1["new"] == 3 and s1["llm_calls"] >= 3
    first = {r["scene_id"]: r for r in load(out)}
    s2, _ = run(tmp_path, DEV_D, "llm_kb")  # resumes: only the 5 missing scenes
    assert s2["kept"] == 3 and s2["new"] == 5 and s2["n"] == 8
    again = {r["scene_id"]: r for r in load(out)}
    assert all(again[k] == v for k, v in first.items())
    s3, _ = run(tmp_path, DEV_D, "llm_kb")  # nothing left to do: no LLM, no calls
    assert s3["new"] == 0 and s3["llm_calls"] == 0 and s3["cache_hits"] == 0 and s3["kept"] == 8
    s4, out4 = run(tmp_path, DEV_D, "llm_kb", "c3_copy_dev.json")  # same cache, new file: all cache hits
    assert s4["new"] == 8 and s4["llm_calls"] == 0 and s4["cache_hits"] >= 8
    assert [r["positive"] for r in load(out4)] == [r["positive"] for r in load(out)]
    s5, _ = run(tmp_path, DEV_D, "llm_kb", None, "--overwrite")
    assert s5["new"] == 8 and s5["kept"] == 0 and s5["llm_calls"] == 0


def test_warm_cache_never_loads_the_model(tmp_path, monkeypatch):
    run(tmp_path, FIXTURES, "llm_nokb")  # warms the cache under model_id FAKE_MODEL_ID
    loads = []

    def fake_make_client(cfg, size="default", backend=None, **kw):
        return LazyClient(lambda: loads.append(1) or EchoFakeLLM(), FAKE_MODEL_ID)

    monkeypatch.setattr("expander.llm.make_client", fake_make_client)
    summary, out = run(tmp_path, FIXTURES, "llm_nokb", "c4_again_dev.json")
    assert loads == [] and summary["model_loads"] == 0 and summary["cache_hits"] == 3
    assert not any("model_load_s" in r for r in load(out))
    summary, out = run(tmp_path, FIXTURES, "llm_kb", "c3_cold_dev.json")  # cold cache: loads once, on scene 1
    assert loads == [1] and summary["model_loads"] == 1
    assert [("model_load_s" in r) for r in load(out)] == [True, False, False]


def test_load_failure_stops_the_batch(tmp_path, monkeypatch):
    def boom():
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("expander.llm.make_client", lambda *a, **k: LazyClient(boom, "some/model"))
    with pytest.raises(SystemExit, match="failed to load"):
        run(tmp_path, FIXTURES, "llm_kb")
    out = tmp_path / "c3_llm_kb_dev.json"
    assert not out.exists() or load(out) == []  # no fallback record is passed off as c3


def test_condition_from_out_name_and_flag(tmp_path):
    s, out = run(tmp_path, FIXTURES, "llm_kb", "c6_llm_kbllm_dev.json")
    assert s["condition"] == "c6_llm_kbllm" and {r["condition"] for r in load(out)} == {"c6_llm_kbllm"}
    s, out = run(tmp_path, FIXTURES, "template", "anything.json", "--condition", "c2_custom", "--split", "heldout")
    assert {(r["condition"], r["split"]) for r in load(out)} == {("c2_custom", "heldout")}


def test_refuses_to_mix_conditions(tmp_path):
    _, out = run(tmp_path, FIXTURES, "template", "mixed.json")
    with pytest.raises(SystemExit, match="use --overwrite"):
        run(tmp_path, FIXTURES, "llm_kb", "mixed.json")
    s, _ = run(tmp_path, FIXTURES, "llm_kb", "mixed.json", "--overwrite")
    assert s["new"] == 3 and {r["mode"] for r in load(out)} == {"llm_kb"}


def test_prompt_versions(tmp_path):
    with pytest.raises(SystemExit, match="no system prompt"):
        run(tmp_path, FIXTURES, "llm_kb", None, "--prompt_version", "v404")
    _, out = run(tmp_path, FIXTURES, "llm_kb", "c3_ex_dev.json", "--prompt_version", "v1ex")
    assert {r["prompt_version"] for r in load(out)} == {"v1ex"}
    _, out = run(tmp_path, FIXTURES, "llm_kb_layout", "c5_ex_dev.json", "--prompt_version", "v1ex")
    assert {r["prompt_version"] for r in load(out)} == {"v1ex_layout"}


def test_scene_inputs(tmp_path):
    scenes = json.loads(FIXTURES.read_text())
    jsonl = tmp_path / "scenes.jsonl"
    jsonl.write_text("\n".join(json.dumps(s) for s in scenes) + "\n")
    assert [s["id"] for s in be.load_scenes(jsonl)] == ["f01", "f02", "f03"]
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"scenes": [{"scene_id": "x1", "raw": "a sad king"}]}))
    assert be.load_scenes(wrapped) == [{"scene_id": "x1", "raw": "a sad king", "id": "x1", "text": "a sad king"}]
    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps([scenes[0], scenes[0]]))
    with pytest.raises(ValueError, match="duplicate"):
        be.load_scenes(dup)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "x"}]))
    with pytest.raises(ValueError, match="no 'text'"):
        be.load_scenes(bad)


def test_fake_runs_never_touch_the_shared_cache(tmp_path):
    shared = be.DEFAULT_CACHE
    before = shared.read_bytes() if shared.exists() else None
    out = tmp_path / "c3_llm_kb_dev.json"
    be.run(["--scenes", str(FIXTURES), "--mode", "llm_kb", "--out", str(out), "--llm", "fake", "--counter", "approx"])
    assert (shared.read_bytes() if shared.exists() else None) == before


def test_approx_counter():
    assert be.approx_n_tokens("rrvarma style") == 3
    assert be.approx_n_tokens(CFG["style_tail"]) >= 20  # over-counts real CLIP (20)
    assert be.approx_n_tokens("") == 0
    fn, name = be.resolve_counter("approx")
    assert fn is be.approx_n_tokens and name == "approx"


# ------------------------------------------------------------------ check_expansions

def _good(**kw):
    rec = {"scene_id": "g1", "condition": "c3_llm_kb", "split": "dev",
           "positive": f"{CFG['trigger']}, a kneeling monkey-faced figure, {CFG['style_tail']}",
           "negative": CFG["negative"], "phrases_kept": ["a kneeling monkey-faced figure"], "phrases_dropped": [],
           "token_count": 40, "trigger_first": True, "names_leaked": [], "source": "llm", "layout": None,
           "llm_model": "m", "prompt_version": "v1", "kb_version": "stub0", "latency_s": 1.0,
           "names_after_repair": [], "mode": "llm_kb"}
    rec.update(kw)
    return rec


BAD = [
    _good(),
    _good(scene_id="b_leak", positive=f"{CFG['trigger']}, Rama kneeling, {CFG['style_tail']}", names_leaked=["Rama"],
          source="fallback", latency_s=3.0),
    _good(scene_id="b_budget", token_count=90),
    _good(scene_id="b_trigger", positive=f"a kneeling figure, {CFG['trigger']}, {CFG['style_tail']}"),
]


def test_check_expansions_on_produced_files(tmp_path, capsys):
    outs = [run(tmp_path, DEV_D, m)[1] for m in ("template", "llm_kb", "llm_nokb", "llm_kb_layout")]
    js = tmp_path / "check.json"
    assert ce.main([*map(str, outs), "--strict", "--no_recount", "--json", str(js)]) == 0
    text = capsys.readouterr().out
    assert "| c3_llm_kb_dev.json | 8 | 100.0% |" in text and "APPROXIMATE counter" in text
    files = load(js)["files"]
    assert all(s["pct_valid"] == 100.0 and s["n_leaks_after"] == 0 for s in files.values())


def test_check_expansions_flags_bad_records(tmp_path, capsys):
    s = ce.summarize(BAD, trigger=CFG["trigger"], max_tokens=70)
    assert s["n"] == 4 and s["n_valid"] == 1 and s["pct_valid"] == 25.0
    assert s["pct_trigger_first"] == 75.0 and s["pct_within_budget"] == 75.0
    assert s["leak_rate_before"] == 25.0 and s["n_leaks_after"] == 1 and s["fallback_rate"] == 25.0
    assert s["mean_latency_s"] == 1.5 and s["median_latency_s"] == 1.0
    assert {b["scene_id"] for b in s["invalid"]} == {"b_leak", "b_budget", "b_trigger"}
    # a recount overrides the stored token_count
    assert ce.summarize(BAD, trigger=CFG["trigger"], max_tokens=70, counter=lambda t: 10)["pct_within_budget"] == 100.0

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(BAD))
    assert ce.main([str(bad), "--no_recount"]) == 0
    assert ce.main([str(bad), "--no_recount", "--strict"]) == 1
    out = capsys.readouterr().out
    assert "INVALID bad.json b_leak: names after repair ['Rama']" in out and "| bad.json | 4 | 25.0% |" in out
    only_leak = tmp_path / "leak.json"  # 99% valid would pass, but a single leak fails --strict
    only_leak.write_text(json.dumps([_good(scene_id=f"g{i}") for i in range(99)] + [BAD[1]]))
    assert ce.main([str(only_leak), "--no_recount", "--strict"]) == 1
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps([_good(token_count=60)]))
    assert ce.main([str(budget), "--no_recount", "--strict", "--max_tokens", "50"]) == 1
    assert ce.main([str(budget), "--no_recount", "--strict", "--max_tokens", "60"]) == 0


def test_check_expansions_raw_and_errors(tmp_path):
    raw = _good(mode="raw", condition="c1_lora_raw", positive=f"{CFG['trigger']}, Rama in the forest, x",
                names_leaked=["Rama"])
    s = ce.summarize([raw], trigger=CFG["trigger"], max_tokens=70)
    assert s["n_raw"] == 1 and s["n_leaks_after"] == 0 and s["pct_valid"] == 100.0
    missing = {k: v for k, v in _good().items() if k != "kb_version"}
    assert ce.summarize([missing], trigger=CFG["trigger"], max_tokens=70)["pct_valid"] == 0.0
    nope = tmp_path / "nope.json"
    nope.write_text('{"not": "a list"}')
    assert ce.main([str(nope)]) == 2
    assert ce.main([str(tmp_path / "missing.json")]) == 2
