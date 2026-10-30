"""LLM backends behind ``expander.llm_base.LLMClient``: a local 4-bit HF model, a deterministic echo fake,
an optional Gemini REST client, a lazy wrapper, and ``make_client(cfg)``.

Importing this module is stdlib-only: ``torch``, ``transformers`` and ``bitsandbytes`` are imported inside
``HFLocalClient.__init__``, so CI (no torch) can import it.

Smoke test on a Colab T4 (prints the raw reply, load time, s/call and peak VRAM)::

    python -m expander.llm --smoke                      # config llm.model (+ llm.revision)
    python -m expander.llm --smoke --size large         # config llm.model_large
    python -m expander.llm --smoke --model Qwen/Qwen2.5-1.5B-Instruct --no_4bit   # CPU check, no bitsandbytes
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections.abc import Callable

if __package__ in (None, ""):  # `python expander/llm.py`: make `import expander` work, never shadow by file name
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
    sys.path.insert(0, _ROOT)

from expander.llm_base import FakeLLM, LLMClient, load_system_prompt

__all__ = ["FAKE_MODEL_ID", "EchoFakeLLM", "FakeLLM", "GeminiClient", "HFLocalClient", "LLMClient", "LazyClient",
           "make_client", "resolve_model"]

FAKE_MODEL_ID = "fake-echo-v1"
GIB = 1024 ** 3


def _log(msg: str) -> None:
    print(f"[llm] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- local HF model

class HFLocalClient:
    """A local instruct model via ``transformers`` (4-bit NF4 with fp16 compute by default; the T4 has no bf16).

    Greedy decoding (``do_sample=False``: temperature 0), chat template with ``add_generation_prompt=True``;
    ``complete`` returns only the newly generated text. ``close()`` frees the GPU before diffusion.
    """

    def __init__(self, model_id: str, revision: str | None = None, load_in_4bit: bool = True,
                 max_new_tokens: int = 256, device_map: str | dict | None = "auto") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("HFLocalClient needs torch + transformers: "
                               "pip install -r expander/requirements_llm.txt") from e
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = int(max_new_tokens)
        self.load_in_4bit = bool(load_in_4bit)
        self.n_calls = 0
        self.total_s = 0.0
        self.last_s = 0.0
        self._torch = torch
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        kwargs: dict = {"revision": revision}
        if device_map is not None:
            kwargs["device_map"] = device_map
        if self.load_in_4bit:
            if not cuda:
                raise RuntimeError("load_in_4bit needs a CUDA GPU (bitsandbytes); use load_in_4bit=False on CPU")
            try:
                import bitsandbytes  # noqa: F401  - fail here with a clear message, not deep inside from_pretrained
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise RuntimeError("4-bit loading needs bitsandbytes: pip install -r expander/requirements_llm.txt") from e
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        else:
            kwargs["dtype"] = torch.float16 if cuda else torch.float32
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.load_s = round(time.perf_counter() - t0, 2)
        _log(f"loaded {model_id}@{revision or 'main'} ({'4-bit nf4' if self.load_in_4bit else kwargs['dtype']}) "
             f"in {self.load_s} s; VRAM {self.vram_gb():.2f} GiB")

    # -- helpers
    def vram_gb(self, peak: bool = False) -> float:
        t = self._torch
        if not t.cuda.is_available():
            return 0.0
        return (t.cuda.max_memory_allocated() if peak else t.cuda.memory_allocated()) / GIB

    def prompt_text(self, system: str, user: str) -> str:
        """The exact text the model sees (for debugging the chat template)."""
        return self.tokenizer.apply_chat_template(self._messages(system, user), tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)

    @staticmethod
    def _messages(system: str, user: str) -> list[dict]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # -- LLMClient
    def complete(self, system: str, user: str) -> str:
        if getattr(self, "model", None) is None:
            raise RuntimeError("HFLocalClient is closed")
        t = self._torch
        t0 = time.perf_counter()
        # enable_thinking=False: Qwen3 hybrid templates skip <think>; other templates ignore unknown variables.
        enc = self.tokenizer.apply_chat_template(self._messages(system, user), add_generation_prompt=True,
                                                 return_tensors="pt", return_dict=True, enable_thinking=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        with t.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None, pad_token_id=pad)
        new = out[0, enc["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        self.last_s = time.perf_counter() - t0
        self.total_s += self.last_s
        self.n_calls += 1
        return text

    def close(self) -> None:
        """Delete the model, ``gc.collect()``, ``torch.cuda.empty_cache()``; logs the VRAM freed. Idempotent."""
        if getattr(self, "model", None) is None:
            return
        before = self.vram_gb()
        peak = self.vram_gb(peak=True)
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        after = self.vram_gb()
        self.freed_gb = round(before - after, 3)
        _log(f"closed {self.model_id}: freed {self.freed_gb:.2f} GiB VRAM (peak {peak:.2f}, now {after:.2f} GiB); "
             f"{self.n_calls} calls, {self.total_s / max(self.n_calls, 1):.2f} s/call")


# --------------------------------------------------------------------------- deterministic fake

_FAKE_STOP = frozenset(("a", "an", "the", "of", "and", "or", "with", "to", "in", "on", "at", "by", "for",
                        "from", "his", "her", "their", "its", "is", "are", "was", "were", "be", "who", "which",
                        "that", "this", "these", "those", "as", "into", "onto", "over", "under", "while",
                        "just", "before", "after", "show", "draw", "make", "me", "please", "can", "you", "i",
                        "it", "so"))
_FAKE_POSES = (("arms raised", "arms_raised"), ("flying", "flying"), ("kneel", "kneeling"),
               ("seated", "seated"), ("sitting", "seated"), ("lying", "lying"), ("reclin", "reclining"),
               ("walk", "walking"), ("stand", "standing"))
_FAKE_POSITIONS = ("left", "right", "center", "foreground", "background")


class EchoFakeLLM(FakeLLM):
    """Deterministic offline LLM for dry runs and CI: builds valid JSON phrases from the user message.

    Phrases = the ``Reference descriptions`` visuals (if any) + 2-4 word chunks of the scene text with
    capitalised words removed. Lower-case names can slip through (like a real LLM); on the repair
    re-ask it strips every name (``names.strip_names``). Layout prompts also get a ``layout`` object.
    """

    def __init__(self, model_id: str = FAKE_MODEL_ID) -> None:
        super().__init__(self._reply, model_id=model_id)

    @staticmethod
    def _split(user: str) -> tuple[str, list[str], list[str]]:
        text = user.split("\n\n", 1)[0]
        refs: list[str] = []
        m = re.search(r"Reference descriptions:\n((?:.+\n?)+)", user)
        if m:
            for line in m.group(1).splitlines():
                if "->" in line:
                    refs.append(line.split("->", 1)[1].strip())
        names = []
        m = re.search(r"contain proper names \(([^)]*)\)", user)
        if m:
            names = [n.strip() for n in m.group(1).split(",") if n.strip()]
        return text, refs, names

    @staticmethod
    def _chunks(text: str) -> list[str]:
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", text) if not w[0].isupper()]
        words = [w for w in words if w.casefold() not in _FAKE_STOP]
        return [" ".join(words[i:i + 3]) for i in range(0, len(words), 3) if len(words[i:i + 3]) >= 2]

    def _reply(self, system: str, user: str) -> str:
        text, refs, repair_names = self._split(user)
        phrases = [p.strip() for r in refs for p in r.split(",")][:6] + self._chunks(text)
        if repair_names:  # the re-ask: behave like a model that follows the instruction
            from expander.names import strip_names

            phrases = [strip_names(p) for p in phrases]
        phrases = [p for p in dict.fromkeys(phrases) if len(p.split()) >= 2][:12] or ["a lone figure standing"]
        out: dict = {"phrases": phrases}
        if "layout" in system.lower() and "n_figures" in system:
            low = text.lower()
            pose = next((p for k, p in _FAKE_POSES if k in low), "standing")
            n = max(1, min(len(refs), 4))
            out["layout"] = {"n_figures": n, "figures": [
                {"role": "figure", "pose": pose, "position": _FAKE_POSITIONS[i % 3 if n > 1 else 2]}
                for i in range(n)]}
        return json.dumps(out, ensure_ascii=False)


# --------------------------------------------------------------------------- optional hosted backend

class GeminiClient:
    """Optional free-tier hosted backend (Google Gemini REST API, stdlib ``urllib``). Not required.

    The key comes ONLY from ``$GEMINI_API_KEY`` (Colab: ``os.environ["GEMINI_API_KEY"] = userdata.get(...)``);
    it is sent in a header, never logged, never stored on disk. Untested against the live API.
    """

    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model_id: str = "gemini-2.5-flash", timeout_s: float = 60.0,
                 urlopen: Callable | None = None) -> None:
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GeminiClient needs the GEMINI_API_KEY environment variable (Colab Secrets)")
        self.model_id = model_id
        self.timeout_s = timeout_s
        self._urlopen = urlopen

    def complete(self, system: str, user: str) -> str:
        import urllib.request

        body = {"system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}}
        req = urllib.request.Request(self.ENDPOINT.format(model=self.model_id), method="POST",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        with (self._urlopen or urllib.request.urlopen)(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"unexpected Gemini response shape: {list(data)[:5]}") from e

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- lazy wrapper and factory

class LazyClient:
    """Defers loading until the first ``complete``. With a warm cache a batch run never loads the model.

    ``model_id`` must be known up front: it is part of the cache key.
    """

    def __init__(self, factory: Callable[[], LLMClient], model_id: str, revision: str | None = None) -> None:
        self._factory = factory
        self.model_id = model_id
        self.revision = revision
        self.client: LLMClient | None = None
        self.loads = 0
        self.load_s: float | None = None
        self.load_error: BaseException | None = None

    @property
    def loaded(self) -> bool:
        return self.client is not None

    def complete(self, system: str, user: str) -> str:
        if self.load_error is not None:  # never retry a failed load once per scene
            raise RuntimeError(f"LLM load failed earlier: {self.load_error}") from self.load_error
        if self.client is None:
            t0 = time.perf_counter()
            try:
                self.client = self._factory()
            except Exception as e:
                self.load_error = e
                raise
            self.loads += 1
            self.load_s = round(time.perf_counter() - t0, 2)
        return self.client.complete(system, user)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None


def resolve_model(cfg: dict, size: str = "default", model: str | None = None,
                  revision: str | None = None) -> tuple[str, str | None]:
    """``(model_id, revision)`` from ``cfg["llm"]``: ``model``/``revision`` or ``model_large``/``model_large_revision``.

    An explicit ``model`` that differs from the config's keeps only an explicit ``revision``.
    """
    llm = cfg.get("llm", {})
    if size not in ("default", "large"):
        raise ValueError(f"size must be 'default' or 'large', not {size!r}")
    key, rkey = ("model_large", "model_large_revision") if size == "large" else ("model", "revision")
    cfg_model = llm.get(key)
    if model is None:
        if not cfg_model:
            raise KeyError(f"config llm.{key} is not set")
        model = cfg_model
    if revision is None and model == cfg_model:
        revision = llm.get(rkey)
    return model, revision


def make_client(cfg: dict, size: str = "default", backend: str | None = None, *, model: str | None = None,
                revision: str | None = None, lazy: bool = False) -> LLMClient | None:
    """Build the client from ``cfg["llm"]``. ``backend``: ``hf_local`` (default from config), ``fake``,
    ``gemini`` or ``none`` (returns ``None``: ``expand()`` then uses the template fallback when the config
    backend is ``none``). ``lazy=True`` returns a ``LazyClient`` that loads on the first call.
    """
    llm = cfg.get("llm", {})
    backend = (backend or llm.get("backend") or "hf_local").lower()
    if backend == "fake":
        return EchoFakeLLM()
    if backend == "none":
        return None
    if backend == "gemini":
        mid = model or llm.get("gemini_model", "gemini-2.5-flash")
        return LazyClient(lambda: GeminiClient(mid), mid) if lazy else GeminiClient(mid)
    if backend != "hf_local":
        raise ValueError(f"unknown llm backend {backend!r}; expected hf_local, fake, gemini or none")
    mid, rev = resolve_model(cfg, size, model, revision)

    def build() -> HFLocalClient:
        return HFLocalClient(mid, rev, load_in_4bit=bool(llm.get("load_in_4bit", True)),
                             max_new_tokens=int(llm.get("max_new_tokens", 256)))

    return LazyClient(build, mid, rev) if lazy else build()


# --------------------------------------------------------------------------- smoke test

SMOKE_USER = ("Rama and Sita seated on a throne in the royal hall, Hanuman kneeling at their feet with folded hands"
              "\n\nReference descriptions:\n"
              "Rama -> blue-skinned crowned prince, topknot, yellow silk dhoti, long bow and arrows\n"
              "Sita -> young woman in a red saree with a gold border, layered pearl necklaces, gold bangles\n"
              "Hanuman -> monkey-faced figure, white beard, gold crown, red garment, mace over shoulder\n"
              "setting -> pillared royal hall, carved stone columns")


def _smoke(argv: list[str] | None = None) -> int:
    from expander._cfg import load_config
    from expander.expand import parse_reply
    from expander.names import find_names

    ap = argparse.ArgumentParser(prog="python -m expander.llm", description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true", help="load the model, make one call, report timings")
    ap.add_argument("--config", default=None)
    ap.add_argument("--size", choices=["default", "large"], default="default")
    ap.add_argument("--model", default=None, help="override llm.model / llm.model_large")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--backend", default=None, help="hf_local (default), fake, gemini")
    ap.add_argument("--no_4bit", action="store_true", help="fp16 on GPU / fp32 on CPU (no bitsandbytes)")
    ap.add_argument("--prompt_version", default=None)
    ap.add_argument("--calls", type=int, default=2, help="repeat the call to measure warm s/call")
    a = ap.parse_args(argv)
    if not a.smoke:
        ap.print_help()
        return 2
    cfg = load_config(a.config)
    if a.no_4bit:
        cfg.setdefault("llm", {})["load_in_4bit"] = False
    pv = a.prompt_version or cfg["llm"].get("prompt_version", "v1")
    system = load_system_prompt(pv)
    t0 = time.perf_counter()
    client = make_client(cfg, a.size, a.backend, model=a.model, revision=a.revision)
    load_s = time.perf_counter() - t0
    try:
        times, reply = [], ""
        for _ in range(max(1, a.calls)):
            t1 = time.perf_counter()
            reply = client.complete(system, SMOKE_USER)
            times.append(time.perf_counter() - t1)
        parsed = parse_reply(reply)
        print("----- raw reply -----")
        print(reply)
        print("---------------------")
        print(f"model       : {client.model_id} @ {getattr(client, 'revision', None) or 'main'}")
        print(f"prompt      : {pv}")
        print(f"load        : {load_s:.1f} s")
        print(f"s/call      : first {times[0]:.2f}, warm {min(times):.2f}")
        print(f"valid JSON  : {parsed is not None}" + (f" ({len(parsed['phrases'])} phrases)" if parsed else ""))
        if parsed:
            print(f"names leaked: {find_names(' ; '.join(parsed['phrases']))}")
        if isinstance(client, HFLocalClient):
            print(f"peak VRAM   : {client.vram_gb(peak=True):.2f} GiB")
        return 0 if parsed is not None else 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(_smoke())
