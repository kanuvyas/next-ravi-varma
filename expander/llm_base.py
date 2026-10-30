"""LLM interface, a scripted fake for tests/offline runs, and system-prompt loading. Stdlib only.

Real backends (``expander/llm.py``, e.g. a 4-bit local HF model) implement ``LLMClient``:
``complete(system, user) -> str`` and ``close()``. A backend may expose ``model_id: str``; ``expand()``
records it as ``llm_model`` and uses it in the cache key (else ``config["llm"]["model"]``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# D guide v2 §3.1, verbatim.
SYSTEM_PROMPT_V1 = """You write scene descriptions for a text-to-image model trained on 19th-century Indian oil paintings.
Return ONLY valid JSON: {"phrases": ["...", "..."]}
Rules:
- 6 to 12 phrases, each 2 to 6 words, ordered from MOST to LEAST important to the scene.
- NEVER use proper names of people, gods or places. The model does not know names. Describe by
  clothing, ornaments, pose, expression and setting instead.
- If reference descriptions are given, use their wording for the matching figures.
- Cover, in roughly this order: the main figures (attire), what they are doing (pose, gesture),
  the setting, the lighting and mood.
- Do NOT mention style, painting, canvas or artist; that is added separately.
- Use concrete visual words (e.g. "kneeling", "gilded armour", "pillared hall")."""

# Appended for the layout variant (prompt version "<v>_layout", condition c5). Vocabulary: D guide Day 4.3.
LAYOUT_ADDENDUM = """
Also return the figure layout. Return ONLY valid JSON:
{"phrases": ["...", "..."], "layout": {"n_figures": 2, "figures": [{"role": "king", "pose": "kneeling", "position": "left"}]}}
- role: 1 to 3 plain words, no names (e.g. "king", "young woman", "sage").
- pose: one of standing, seated, kneeling, lying, reclining, arms_raised, walking, flying.
- position: one of left, center, right, foreground, background."""

BUILTIN_PROMPTS: dict[str, str] = {
    "v1": SYSTEM_PROMPT_V1,
    "v1_layout": SYSTEM_PROMPT_V1 + "\n" + LAYOUT_ADDENDUM,
}


@runtime_checkable
class LLMClient(Protocol):
    """What ``expand()`` needs from an LLM backend."""

    def complete(self, system: str, user: str) -> str:
        """Return the model's reply to one system + user turn (temperature 0)."""
        ...

    def close(self) -> None:
        """Free GPU memory before diffusion."""
        ...


class FakeLLM:
    """Scripted ``LLMClient`` for tests and offline runs.

    ``replies`` is either a list consumed in order (an ``Exception`` item is raised instead of returned;
    running out raises ``RuntimeError``) or a callable ``(system, user) -> str``. Every call is recorded
    in ``calls`` as ``(system, user)``.
    """

    def __init__(self, replies: Sequence[str | BaseException] | Callable[[str, str], str], *,
                 model_id: str | None = None) -> None:
        self._fn = replies if callable(replies) else None
        self._queue = [] if callable(replies) else list(replies)
        self.model_id = model_id
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self._fn is not None:
            return self._fn(system, user)
        if not self._queue:
            raise RuntimeError("FakeLLM: no scripted replies left")
        r = self._queue.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r

    def close(self) -> None:
        self.closed = True


def load_system_prompt(version: str = "v1", prompts_dir: str | Path | None = None) -> str:
    """``expander/prompts/system_<version>.txt`` if present, else the built-in text for ``version``.

    Raises ``KeyError`` for an unknown version with no file.
    """
    p = Path(prompts_dir or PROMPTS_DIR) / f"system_{version}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    if version in BUILTIN_PROMPTS:
        return BUILTIN_PROMPTS[version]
    raise KeyError(f"no system prompt for version {version!r} (looked for {p})")
