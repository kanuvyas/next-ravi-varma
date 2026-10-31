"""expander/llm.py and the prompt files, CPU only: no torch, no transformers, no downloads (fakes stand in)."""

import hashlib
import json
import subprocess
import sys
import types

import pytest

from expander import _cfg
from expander import llm as llm_mod
from expander.expand import parse_layout, parse_reply
from expander.llm import (
    FAKE_MODEL_ID,
    EchoFakeLLM,
    GeminiClient,
    HFLocalClient,
    LazyClient,
    make_client,
    resolve_model,
)
from expander.llm_base import (
    BUILTIN_PROMPTS,
    PROMPTS_DIR,
    SYSTEM_PROMPT_V1,
    LLMClient,
    load_system_prompt,
)
from expander.names import default_matcher

ROOT = _cfg.repo_root()
CFG = _cfg.load_config()  # what batch_expand itself loads: team config, gaps filled from the shim
# Pinned: changing system_v1.txt without a new prompt_version breaks the cache's meaning.
SYSTEM_V1_SHA256 = "80219a5415fe89cd7804637f4046742b98cdf8e0821313a72ff6dbadd7a96586"


# ------------------------------------------------------------------ import hygiene

def test_import_needs_no_torch():
    code = ("import sys, expander.llm, expander.batch_expand, expander.check_expansions; "
            "bad = [m for m in ('torch', 'transformers', 'bitsandbytes', 'accelerate') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_scripts_run_from_any_directory(tmp_path):
    for script in ("batch_expand.py", "check_expansions.py", "llm.py"):
        r = subprocess.run([sys.executable, str(ROOT / "expander" / script), "--help"], check=False, cwd=tmp_path,
                           capture_output=True, text=True)
        assert r.returncode in (0, 2) and "usage" in (r.stdout + r.stderr).lower(), (script, r.stderr)


def test_hf_client_without_torch_raises_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    with pytest.raises(RuntimeError, match="requirements_llm.txt"):
        HFLocalClient("Qwen/Qwen3-4B-Instruct-2507")


# ------------------------------------------------------------------ HFLocalClient against fake torch/transformers

class _T:
    """Minimal tensor: 2-D list of ints with .to/.shape and [0, a:] slicing."""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def to(self, device):
        return self

    def __getitem__(self, idx):
        r, c = idx
        return self.rows[r][c]


@pytest.fixture
def fake_stack(monkeypatch):
    log = {"empty_cache": 0, "alloc": 3 * 1024 ** 3}

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def reset_peak_memory_stats():
            pass

        @staticmethod
        def memory_allocated():
            return log["alloc"]

        @staticmethod
        def max_memory_allocated():
            return 4 * 1024 ** 3

        @staticmethod
        def empty_cache():
            log["empty_cache"] += 1
            log["alloc"] = 0

    class Inference:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    torch = types.SimpleNamespace(cuda=Cuda, float16="fp16", float32="fp32", inference_mode=Inference)

    class Tok:
        pad_token_id = None
        eos_token_id = 0

        @classmethod
        def from_pretrained(cls, model_id, revision=None):
            log["tok"] = (model_id, revision)
            return cls()

        def apply_chat_template(self, messages, **kw):
            log["chat"] = (messages, kw)
            return {"input_ids": _T([[5, 6, 7]]), "attention_mask": _T([[1, 1, 1]])}

        def decode(self, ids, skip_special_tokens=False):
            log["decoded"] = list(ids)
            return ' {"phrases": ["a kneeling figure", "pillared hall"]} '

    class Model:
        device = "cuda:0"

        @classmethod
        def from_pretrained(cls, model_id, **kw):
            log["model"] = (model_id, kw)
            return cls()

        def eval(self):
            return self

        def generate(self, **kw):
            log["generate"] = kw
            return _T([[5, 6, 7, 42, 43]])

    class BnB:
        def __init__(self, **kw):
            self.kw = kw

    transformers = types.SimpleNamespace(AutoTokenizer=Tok, AutoModelForCausalLM=Model, BitsAndBytesConfig=BnB)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
    return log


def test_hf_client_4bit_greedy_new_tokens_and_close(fake_stack):
    c = HFLocalClient("Qwen/Qwen3-4B-Instruct-2507", revision="abc", max_new_tokens=99)
    assert isinstance(c, LLMClient) and c.model_id == "Qwen/Qwen3-4B-Instruct-2507"
    mid, kw = fake_stack["model"]
    assert kw["revision"] == "abc" and kw["device_map"] == "auto" and fake_stack["tok"] == (mid, "abc")
    q = kw["quantization_config"].kw
    assert q == {"load_in_4bit": True, "bnb_4bit_compute_dtype": "fp16", "bnb_4bit_quant_type": "nf4"}
    reply = c.complete("SYS", "USER")
    assert reply == '{"phrases": ["a kneeling figure", "pillared hall"]}'
    msgs, ckw = fake_stack["chat"]
    assert msgs == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert ckw["add_generation_prompt"] is True and ckw["return_dict"] is True
    g = fake_stack["generate"]
    assert g["do_sample"] is False and g["max_new_tokens"] == 99 and g["temperature"] is None
    assert g["pad_token_id"] == 0
    assert fake_stack["decoded"] == [42, 43]  # only the new tokens
    assert c.n_calls == 1
    c.close()
    assert c.model is None and fake_stack["empty_cache"] == 1 and c.freed_gb == 3.0
    c.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        c.complete("s", "u")


def test_hf_client_without_4bit(fake_stack):
    HFLocalClient("tiny/model", load_in_4bit=False)
    _, kw = fake_stack["model"]
    assert "quantization_config" not in kw and kw["dtype"] == "fp16"


# ------------------------------------------------------------------ factory, lazy, fake

def test_make_client_fake_and_none():
    c = make_client(CFG, backend="fake")
    assert isinstance(c, EchoFakeLLM) and isinstance(c, LLMClient) and c.model_id == FAKE_MODEL_ID
    assert make_client(CFG, backend="none") is None
    assert make_client({**CFG, "llm": {**CFG["llm"], "backend": "fake"}}).model_id == FAKE_MODEL_ID
    with pytest.raises(ValueError, match="unknown llm backend"):
        make_client(CFG, backend="openai")


def test_make_client_lazy_does_not_load():
    c = make_client(CFG, lazy=True)
    assert isinstance(c, LazyClient) and not c.loaded
    assert (c.model_id, c.revision) == (CFG["llm"]["model"], CFG["llm"]["revision"])
    big = make_client(CFG, "large", lazy=True)
    assert (big.model_id, big.revision) == (CFG["llm"]["model_large"], CFG["llm"]["model_large_revision"])
    c.close()  # never loaded: nothing to do


def test_resolve_model():
    assert resolve_model(CFG) == (CFG["llm"]["model"], CFG["llm"]["revision"])
    assert resolve_model(CFG, model="other/m") == ("other/m", None)  # the config revision belongs to another model
    assert resolve_model(CFG, model="other/m", revision="r1") == ("other/m", "r1")
    assert resolve_model(CFG, model=CFG["llm"]["model"]) == (CFG["llm"]["model"], CFG["llm"]["revision"])
    with pytest.raises(ValueError):
        resolve_model(CFG, size="huge")
    with pytest.raises(KeyError):
        resolve_model({"llm": {}})


def test_lazy_client_loads_once_and_remembers_failure():
    built = []
    lc = LazyClient(lambda: built.append(1) or EchoFakeLLM(), "m")
    lc.complete("s", "a weeping woman alone")
    lc.complete("s", "a weeping woman alone")
    assert built == [1] and lc.loads == 1 and lc.loaded
    lc.close()
    assert not lc.loaded
    tries = []

    def boom():
        tries.append(1)
        raise MemoryError("OOM")

    bad = LazyClient(boom, "m")
    with pytest.raises(MemoryError):
        bad.complete("s", "u")
    with pytest.raises(RuntimeError, match="load failed earlier"):
        bad.complete("s", "u")
    assert tries == [1] and isinstance(bad.load_error, MemoryError)


def test_echo_fake_is_deterministic_and_valid():
    f = EchoFakeLLM()
    user = ("Draupadi pleading with krishn in the palace court\n\nReference descriptions:\n"
            "Draupadi -> woman in a dark blue saree, gold jewelry\nsetting -> pillared royal hall")
    a, b = f.complete(load_system_prompt("v1"), user), EchoFakeLLM().complete(load_system_prompt("v1"), user)
    assert a == b
    obj = parse_reply(a)
    assert obj["phrases"][:3] == ["woman in a dark blue saree", "gold jewelry", "pillared royal hall"]
    assert "layout" not in json.loads(a)
    assert default_matcher().find(" ; ".join(obj["phrases"])) == ["Krishna"]  # lower-case name slips through
    repaired = parse_reply(f.complete("s", f"{user}\n\nThey contain proper names (Krishna). Remove proper names."))
    assert default_matcher().find(" ; ".join(repaired["phrases"])) == []
    lay = json.loads(f.complete(load_system_prompt("v1_layout"), user))["layout"]
    assert parse_layout(lay) == lay and lay["n_figures"] == 2
    assert parse_reply(f.complete("s", "?")) == {"phrases": ["a lone figure standing"]}


def test_gemini_client(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiClient()
    monkeypatch.setenv("GEMINI_API_KEY", "sekret-123")
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"candidates": [{"content": {"parts": [{"text": '{"phrases": ["a sage"]}'}]}}]}).encode()

    def urlopen(req, timeout):
        seen["req"], seen["timeout"] = req, timeout
        return Resp()

    g = GeminiClient("gemini-x", urlopen=urlopen)
    assert g.complete("SYS", "USER") == '{"phrases": ["a sage"]}'
    body = json.loads(seen["req"].data)
    assert body["generationConfig"]["temperature"] == 0.0 and body["system_instruction"]["parts"][0]["text"] == "SYS"
    assert "gemini-x:generateContent" in seen["req"].full_url and "sekret" not in seen["req"].full_url
    assert seen["req"].get_header("X-goog-api-key") == "sekret-123"
    assert "sekret" not in repr(g.__dict__) and "sekret" not in capsys.readouterr().err
    g.close()


def test_smoke_cli_with_fake(capsys):
    assert llm_mod._smoke(["--smoke", "--backend", "fake", "--calls", "1"]) == 0
    out = capsys.readouterr().out
    assert "----- raw reply -----" in out and "valid JSON  : True" in out and FAKE_MODEL_ID in out
    assert llm_mod._smoke([]) == 2


# ------------------------------------------------------------------ prompt files

def test_system_v1_file_matches_builtin_and_is_pinned():
    text = (PROMPTS_DIR / "system_v1.txt").read_text(encoding="utf-8")
    assert text == SYSTEM_PROMPT_V1 + "\n"
    assert load_system_prompt("v1") == SYSTEM_PROMPT_V1
    assert hashlib.sha256(SYSTEM_PROMPT_V1.encode()).hexdigest() == SYSTEM_V1_SHA256


def test_system_v1_layout_file_matches_builtin():
    assert (PROMPTS_DIR / "system_v1_layout.txt").read_text(encoding="utf-8") == BUILTIN_PROMPTS["v1_layout"] + "\n"
    text = load_system_prompt("v1_layout")
    assert '"layout"' in text and "arms_raised" in text and "foreground" in text


@pytest.mark.parametrize("version,base", [("v1ex", "v1"), ("v1ex_layout", "v1_layout")])
def test_worked_example_prompts(version, base):
    text = load_system_prompt(version)
    assert text.startswith(load_system_prompt(base) + "\n\nExample")
    answer = text.rsplit("Answer:\n", 1)[1]
    obj = parse_reply(answer)
    assert obj and 6 <= len(obj["phrases"]) <= 12
    assert all(2 <= len(p.split()) <= 6 for p in obj["phrases"])
    assert default_matcher().find(" ; ".join(obj["phrases"])) == []
    if base.endswith("_layout"):
        assert parse_layout(obj["layout"]) == obj["layout"]
    scene_text = text.split("Scene:\n", 1)[1].split("\n", 1)[0]
    dev = {s["text"]: s for s in json.loads((ROOT / "expander" / "dev_scenes_D.json").read_text())}
    assert scene_text in dev and dev[scene_text]["split"] == "dev"  # a dev scene, never held-out
    with pytest.raises(KeyError):
        load_system_prompt("v9ex")
