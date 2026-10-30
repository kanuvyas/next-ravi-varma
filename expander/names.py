"""Proper-name matcher shared by the expander's scrub and A's CI leak check.

Policy (keep identical in every consumer; import this module rather than copying the regex):

1. Text and names are normalised with NFKD; combining marks are dropped (``Rāma`` -> ``Rama``) and any
   other non-ASCII character becomes a space (so ``Rama’s`` still matches ``Rama``).
2. A name matches as whole words: ``(?<!\\w)`` + its words joined by ``\\s+`` + ``(?!\\w)`` (the ``\\b``
   rule, robust to names that end in punctuation). Longest names win; matches never overlap.
   ``panorama``, ``sitar``, ``Ramayana`` (for Rama) and the trigger ``rrvarma`` do not match.
3. Case-insensitive by default. The names in ``CASE_SENSITIVE_ONLY`` are also ordinary words
   (``a ram``, ``lava flow``, ``kusha grass``, ``a hamsa``), so they match only when capitalised as
   written there (``Ram``, ``Lava``). An LLM capitalises proper names.
4. Rivers (Ganga policy): ``Ganga``, ``Yamuna``/``Jamuna`` and ``Sarayu`` are both names and places, and
   the system prompt forbids place names. They are case-sensitive: ``the banks of the Ganga`` is a leak
   and is scrubbed; lowercase ``the river ganga`` is treated as a common noun and kept.
5. Name sources: ``kb/names.json`` (C) if it exists, else ``expander/names_seed.json``. Matches are
   reported as canonical names (the seed's alias map is used when the source is a flat list).

Standard library only.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path

SEED_PATH = Path(__file__).resolve().parent / "names_seed.json"

# names_seed_meta.json -> risk.case_sensitive_only, plus the rivers (policy 4).
CASE_SENSITIVE_ONLY: frozenset[str] = frozenset({
    "Ram", "Lava", "Lav", "Kush", "Kusha", "Bali", "Kali", "Tara", "Kama", "Hamsa", "Uma", "Hari", "Shyam",
    "Amba", "Datta", "Nal", "Karn", "Bhim", "Ananta", "Sakra", "Gita", "Purana", "Matsya", "Varaha", "Rati",
    "Usha", "Agni", "Surya", "Sharada", "Indira", "Malati", "Siddhi", "Riddhi",
    "Ganga", "Yamuna", "Jamuna", "Sarayu",
})

_STOPWORDS = frozenset({"a", "an", "the", "of", "and", "or", "with", "to", "in", "on", "at", "by", "for", "from",
                        "his", "her", "their", "its", "s", "as", "into", "over", "near", "beside"})


def normalize(text: str) -> str:
    """NFKD, drop combining marks, map every other non-ASCII character to a space."""
    out = []
    for ch in unicodedata.normalize("NFKD", text):
        if unicodedata.combining(ch):
            continue
        out.append(ch if ord(ch) < 128 else " ")
    return "".join(out)


def _key(name: str) -> str:
    return " ".join(normalize(name).split()).casefold()


def _pattern(names: Iterable[str]) -> str:
    alts = [r"\s+".join(map(re.escape, normalize(n).split())) for n in names]
    return r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)"


def load_names(source) -> dict[str, str]:
    """Return ``{name or alias: canonical}`` from any accepted shape.

    ``source`` may be a path or an already-loaded object:
    - a flat list of strings (C's ``kb/names.json``); canonicals come from the seed's alias map when known;
    - ``{canonical: [aliases]}`` (``names_seed.json``);
    - ``{"names": [...]}``;
    - a KB dict (C's schema): characters' ``canonical`` + ``aliases`` only. Setting aliases are generic
      words ("battlefield", "palace") and are deliberately ignored.
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, encoding="utf-8") as f:
            source = json.load(f)
    out: dict[str, str] = {}
    if isinstance(source, dict) and isinstance(source.get("names"), list):
        source = source["names"]
    if isinstance(source, list):
        seed = _seed_map()
        for n in source:
            if isinstance(n, str) and n.strip():
                out[n.strip()] = seed.get(_key(n), n.strip())
    elif isinstance(source, dict) and isinstance(source.get("characters"), dict):
        for key, ch in source["characters"].items():
            canon = ch.get("canonical") or key
            for a in [canon, *ch.get("aliases", [])]:
                if isinstance(a, str) and a.strip():
                    out[a.strip()] = canon
    elif isinstance(source, dict):
        for canon, aliases in source.items():
            if canon.startswith("_"):
                continue
            for a in [canon, *(aliases or [])]:
                if isinstance(a, str) and a.strip():
                    out[a.strip()] = canon
    else:
        raise TypeError(f"unsupported names source: {type(source).__name__}")
    return out


@lru_cache(maxsize=1)
def _seed_map() -> dict[str, str]:
    """``{casefolded alias: canonical}`` from the shipped seed."""
    with open(SEED_PATH, encoding="utf-8") as f:
        seed = json.load(f)
    return {_key(a): canon for canon, aliases in seed.items() for a in [canon, *aliases]}


def default_names(kb_names_path: str | os.PathLike | None = None) -> dict[str, str]:
    """``kb/names.json`` (C) if it exists (or the given path), else ``expander/names_seed.json``."""
    if kb_names_path:
        return load_names(kb_names_path)
    real = Path(__file__).resolve().parent.parent / "kb" / "names.json"
    return load_names(real if real.exists() else SEED_PATH)


class NameMatcher:
    """Finds proper names in text under the module policy."""

    def __init__(self, names: Mapping[str, str] | Iterable[str],
                 case_sensitive: Iterable[str] = CASE_SENSITIVE_ONLY) -> None:
        mapping = dict(names) if isinstance(names, Mapping) else {n: n for n in names}
        cs_form = {_key(c): " ".join(normalize(c).split()) for c in case_sensitive}
        self._canon_ci: dict[str, str] = {}
        self._canon_cs: dict[str, str] = {}
        for name, canon in mapping.items():
            k = _key(name)
            if not k:
                continue
            if k in cs_form:
                self._canon_cs.setdefault(cs_form[k], canon)
            else:
                self._canon_ci.setdefault(k, canon)
        by_len = lambda xs: sorted(xs, key=len, reverse=True)
        self._ci = re.compile(_pattern(by_len(self._canon_ci)), re.IGNORECASE) if self._canon_ci else None
        self._cs = re.compile(_pattern(by_len(self._canon_cs))) if self._canon_cs else None

    def spans(self, text: str) -> list[tuple[int, int, str]]:
        """Non-overlapping ``(start, end, canonical)`` over ``normalize(text)``, in order."""
        t = normalize(text)
        cands: list[tuple[int, int, str]] = []
        if self._ci:
            cands += [(m.start(), m.end(), self._canon_ci[_key(m.group(0))]) for m in self._ci.finditer(t)]
        if self._cs:
            cands += [(m.start(), m.end(), self._canon_cs[" ".join(m.group(0).split())]) for m in self._cs.finditer(t)]
        cands.sort(key=lambda c: (c[0], c[0] - c[1]))
        out: list[tuple[int, int, str]] = []
        for c in cands:
            if not out or c[0] >= out[-1][1]:
                out.append(c)
        return out

    def find(self, text: str) -> list[str]:
        """Canonical names in order of first appearance, de-duplicated."""
        return list(dict.fromkeys(c for _, _, c in self.spans(text)))


@lru_cache(maxsize=1)
def default_matcher() -> NameMatcher:
    """The matcher over ``default_names()`` (cached; ``default_matcher.cache_clear()`` to reload)."""
    return NameMatcher(default_names())


def find_names(text: str, matcher: NameMatcher | None = None) -> list[str]:
    """Canonical names found in ``text`` (deduplicated, in order of appearance)."""
    return (matcher or default_matcher()).find(text)


def scrub_phrases(phrases: Iterable[str], matcher: NameMatcher | None = None) -> tuple[list[str], list[str]]:
    """Split phrases into ``(kept, dropped)``: a phrase containing any name is dropped whole."""
    m = matcher or default_matcher()
    kept: list[str] = []
    dropped: list[str] = []
    for p in phrases:
        (dropped if m.find(p) else kept).append(p)
    return kept, dropped


def strip_names(phrase: str, matcher: NameMatcher | None = None) -> str:
    """Template-mode repair: delete name substrings and tidy the connectors left behind.

    Returns the phrase unchanged if it has no names, and ``""`` if fewer than 2 content words remain.
    """
    m = matcher or default_matcher()
    spans = m.spans(phrase)
    if not spans:
        return phrase
    t = normalize(phrase)
    for start, end, _ in reversed(spans):
        tail = t[end:]
        tail = re.sub(r"^'s\b", "", tail)  # possessive left by the name
        t = t[:start] + " " + tail
    t = re.sub(r"\s+([,.;:])", r"\1", re.sub(r"\s+", " ", t))
    t = re.sub(r"(?:\s*,\s*){2,}", ", ", t).strip(" ,;:")
    words = t.split()
    while words and words[0].casefold().strip(",;:") in {"and", "or", "with", "of", "s"}:
        words.pop(0)
    while words and words[-1].casefold().strip(",;:") in _STOPWORDS:
        words.pop()
    t = " ".join(words).strip(" ,;:")
    content = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", t) if w.casefold() not in _STOPWORDS]
    return t if len(content) >= 2 else ""
