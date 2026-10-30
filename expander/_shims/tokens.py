"""TEMPORARY stand-in for A's ``common/tokens.py`` (the ONE token counter). Removed at S2.

Same semantics as A's file: CLIP BPE token count of the base model's tokenizer, **excluding**
start/end tokens, no truncation. Differences, all deliberate:
- ``transformers`` is imported lazily (on the first call), so importing this module is stdlib-only;
- the config is resolved through ``expander._cfg`` instead of a relative ``open("config/m3.json")``.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _tok():
    try:
        from transformers import CLIPTokenizer
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "n_tokens needs `transformers` (CLIP tokenizer). Install it, or pass count_fn= to "
            "assemble()/expand() in tests."
        ) from e
    from expander._cfg import load_config

    return CLIPTokenizer.from_pretrained(load_config()["base_model"], subfolder="tokenizer")


def n_tokens(text: str) -> int:
    """Number of CLIP tokens in ``text``, excluding start/end tokens (limit = config max_prompt_tokens)."""
    return len(_tok()(text, add_special_tokens=False).input_ids)
