"""TEMPORARY stand-in for C's ``kb/stub_parser.py``. Removed at S2.

Mirrors C's interface exactly: ``parse_scene(text, scene_id="", kb_path=None) -> SceneSpec``
(Overview v2 §6.3). Deliberately minimal and stdlib-only (no rapidfuzz): exact, word-bounded,
case-insensitive alias lookup over the KB, longest alias first, consuming matched text so that
"Rama" never matches inside "Ramayana". Mood comes from a small keyword lexicon.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

DEFAULT_KB = Path(__file__).resolve().parent / "stub_kb.json"

MOOD_LEXICON: dict[str, tuple[str, ...]] = {
    "grief": ("weeping", "weeps", "weep", "tears", "mourning", "mourns", "grief", "lament", "lamenting",
              "sorrow", "sad", "despair", "dying", "fallen body"),
    "devotion": ("praying", "prays", "prayer", "devotion", "worship", "worshipping", "folded hands", "bowing",
                 "offering", "blessing"),
    "anger": ("angry", "rage", "fury", "furious", "shouts", "shouting", "attacks", "fighting", "battle"),
    "joy": ("dancing", "dance", "celebration", "celebrating", "joy", "happy", "laughing", "coronation", "wedding"),
}

_STOP = {"The", "A", "An", "And", "In", "On", "At", "Of", "With", "His", "Her", "Their", "One", "Then", "So",
         "This", "That", "He", "She", "They", "It", "I", "My", "Lord", "King", "Queen", "Prince", "Princess"}


def _ascii(s: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFKD", s):
        if unicodedata.combining(ch):
            continue
        out.append(ch if ord(ch) < 128 else " ")
    return "".join(out)


def _pattern(alias: str) -> re.Pattern:
    words = _ascii(alias).split()
    return re.compile(r"\b" + r"\s+".join(map(re.escape, words)) + r"\b", re.IGNORECASE)


def _load_kb(kb_path: str | None) -> dict:
    with open(kb_path or DEFAULT_KB, encoding="utf-8") as f:
        return json.load(f)


def _tidy(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,;:])\1+", r"\1", s)
    s = re.sub(r"^(?:\s*(?:and|,|;)\s*)+", "", s, flags=re.IGNORECASE)
    return s.strip(" ,;:")


def parse_scene(text: str, scene_id: str = "", kb_path: str | None = None) -> dict:
    """Return a SceneSpec for ``text`` using the stub KB (or the KB at ``kb_path``)."""
    kb = _load_kb(kb_path)
    norm = _ascii(text)
    work = norm  # matched spans are blanked out here so they cannot match twice

    aliases: list[tuple[str, str]] = []  # (alias, character key)
    for key, ch in kb.get("characters", {}).items():
        for a in [ch.get("canonical", key), *ch.get("aliases", [])]:
            aliases.append((a, key))
    aliases.sort(key=lambda t: len(t[0]), reverse=True)

    hits: list[tuple[int, int, str, str]] = []  # (start, end, mention, key)
    for alias, key in aliases:
        for m in _pattern(alias).finditer(work):
            hits.append((m.start(), m.end(), norm[m.start():m.end()], key))
            work = work[: m.start()] + " " * (m.end() - m.start()) + work[m.end():]
    hits.sort()

    characters, seen = [], set()
    for _, _, mention, key in hits:
        if key in seen:
            continue
        seen.add(key)
        ch = kb["characters"][key]
        characters.append({"mention": mention, "canonical": ch.get("canonical", key),
                           "visual": ch.get("visual", ""), "coverage": ch.get("coverage", "none")})

    # epic = majority of matched characters; ties go to the first-mentioned; none -> "Unknown"
    order = [kb["characters"][key].get("epic", "Unknown") for key in dict.fromkeys(h[3] for h in hits)]
    counts = Counter(order)
    epic = max(order, key=lambda e: counts[e]) if order else "Unknown"

    setting = None
    best: tuple[int, str] | None = None  # (position of the earliest alias hit, setting key)
    for key, st in kb.get("settings", {}).items():
        for a in [key, *st.get("aliases", [])]:
            m = _pattern(a).search(norm)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), key)
    if best:
        setting = {"name": best[1], "visual": kb["settings"][best[1]].get("visual", "")}

    low = norm.lower()
    mood = ""
    for label, words in MOOD_LEXICON.items():
        if any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in words):
            mood = label
            break

    action = norm
    for start, end, _, _ in sorted(hits, reverse=True):
        action = action[:start] + action[end:]
    action = _tidy(action)

    matched_spans = [(s, e) for s, e, _, _ in hits]
    unknown = []
    for m in re.finditer(r"\b[A-Z][a-z]+\b", norm):
        if m.start() == 0 or norm[: m.start()].rstrip().endswith((".", "!", "?")):
            continue
        if m.group(0) in _STOP or any(s <= m.start() < e for s, e in matched_spans):
            continue
        if m.group(0) not in unknown:
            unknown.append(m.group(0))

    return {"id": scene_id, "raw": text, "epic": epic, "characters": characters, "setting": setting,
            "action": action, "mood": mood, "unknown_entities": unknown,
            "kb_version": kb.get("version", "none")}
