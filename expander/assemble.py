"""Token fitting: ``<trigger>, <phrase 1>, ..., <phrase n>, <style_tail>`` under the CLIP budget.

The LLM (or the template) only proposes scene phrases, most important first. This module adds the
trigger (always first) and the style tail (always last) and keeps as many phrases as fit. A phrase
that does not fit is skipped, but shorter phrases after it can still be kept.

Token counts exclude start/end tokens (``common.tokens.n_tokens``). Tests inject ``count_fn``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

CountFn = Callable[[str], int]

_EDGE = " \t\r\n,.;:\"'`"


def _resolve(count_fn: CountFn | None) -> CountFn:
    if count_fn is not None:
        return count_fn
    from expander._cfg import get_counter

    return get_counter()


def clean_phrases(phrases: Iterable) -> list[str]:
    """Strip, collapse whitespace, trim edge punctuation/quotes, drop empties and non-strings,
    and remove case-insensitive duplicates (first occurrence wins). Order is preserved."""
    out: list[str] = []
    seen: set[str] = set()
    for p in phrases:
        if not isinstance(p, str):
            continue
        s = re.sub(r"\s+", " ", p).strip(_EDGE).strip()
        key = s.casefold()
        if s and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _join(trigger: str, phrases: list[str], style_tail: str) -> str:
    return ", ".join([trigger, *phrases] + ([style_tail] if style_tail else []))


def assemble(trigger: str, phrases: list[str], style_tail: str, max_tokens: int, *,
             count_fn: CountFn | None = None) -> tuple[str, list[str], list[str]]:
    """Fit ``phrases`` (ordered MOST -> LEAST important) between the trigger and the tail.

    Returns ``(positive, kept, dropped)``; ``kept + dropped`` is a partition of ``phrases`` and both keep
    input order. An empty ``style_tail`` adds no trailing ``", "``.
    Raises ``AssertionError`` if the trigger and the tail alone exceed ``max_tokens``.
    """
    n = _resolve(count_fn)
    base = n(_join(trigger, [], style_tail))
    if base > max_tokens:
        raise AssertionError(f"trigger + tail alone exceed the budget ({base} > {max_tokens} tokens)")
    kept: list[str] = []
    dropped: list[str] = []
    for p in phrases:
        if n(_join(trigger, [*kept, p], style_tail)) <= max_tokens:
            kept.append(p)
        else:
            dropped.append(p)  # a long phrase is skipped; shorter later ones can still fit
    return _join(trigger, kept, style_tail), kept, dropped


def token_count(positive: str, *, count_fn: CountFn | None = None) -> int:
    """Tokens in a finished prompt, excluding start/end (the ``token_count`` field)."""
    return _resolve(count_fn)(positive)
