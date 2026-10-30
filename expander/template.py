"""Template mode (c2, and the offline fallback): deterministic phrases from a SceneSpec. No LLM.

Phrase priority (most important first):
1. each character's ``visual``, split on commas into short phrases (characters in SceneSpec order);
2. ``action``;
3. ``setting.visual`` (split on commas);
4. the mood's lighting phrase: ``scene["mood_visual"]`` > the KB ``moods`` entry > ``FALLBACK_MOOD_LIGHTING``.

Names are removed from every phrase (``names.strip_names``) before token fitting, because ``action``
is the user's text minus *matched* names and can still contain unmatched ones.
"""

from __future__ import annotations

import re

from expander.assemble import CountFn, assemble, clean_phrases
from expander.names import NameMatcher, default_matcher, strip_names

TEMPLATE_PROMPT_VERSION = "tpl1"

FALLBACK_MOOD_LIGHTING: dict[str, str] = {
    "grief": "dim somber light",
    "sorrow": "dim somber light",
    "devotion": "soft warm glow",
    "serene": "soft even daylight",
    "calm": "soft even daylight",
    "joy": "bright warm light",
    "celebration": "bright warm light",
    "love": "soft golden light",
    "romance": "soft golden light",
    "anger": "harsh dramatic light",
    "fear": "dark stormy light",
    "battle": "dusty dramatic light",
    "heroic": "bright dramatic light",
    "divine": "radiant golden glow",
    "night": "moonlit night",
}


def split_visual(visual: str | None) -> list[str]:
    """``"a, b; c"`` -> ``["a", "b", "c"]`` (empty parts removed)."""
    if not visual or not isinstance(visual, str):
        return []
    return [p.strip() for p in re.split(r"[,;]", visual) if p.strip()]


def mood_phrase(scene: dict, kb: dict | None = None) -> str:
    """Lighting phrase for the scene's mood ("" if there is no usable mood)."""
    if isinstance(scene.get("mood_visual"), str) and scene["mood_visual"].strip():
        return scene["mood_visual"].strip()
    label = scene.get("mood")
    if not isinstance(label, str) or not label.strip():
        return ""
    label = label.strip().lower()
    entry = ((kb or {}).get("moods") or {}).get(label)
    if isinstance(entry, dict):  # tolerate {"lighting": ...} / {"visual": ...}
        entry = entry.get("lighting") or entry.get("visual") or entry.get("phrase")
    if isinstance(entry, str) and entry.strip():
        return entry.strip()
    return FALLBACK_MOOD_LIGHTING.get(label, "")


def template_phrases(scene: dict, kb: dict | None = None) -> list[str]:
    """Raw template phrases in priority order (before name stripping and cleaning).

    ``kb`` is only used for the mood lookup; ``None`` means the built-in fallback map only.
    """
    phrases: list[str] = []
    for ch in scene.get("characters") or []:
        if isinstance(ch, dict):
            phrases += split_visual(ch.get("visual"))
    action = scene.get("action")
    if isinstance(action, str) and action.strip():
        phrases.append(action.strip())
    setting = scene.get("setting")
    if isinstance(setting, dict):
        phrases += split_visual(setting.get("visual"))
    mood = mood_phrase(scene, kb)
    if mood:
        phrases += split_visual(mood)
    return phrases


def scrub_template_phrases(phrases: list[str], matcher: NameMatcher | None = None
                           ) -> tuple[list[str], list[str], list[str]]:
    """Strip names from each phrase. Returns ``(clean_phrases, names_found, phrases_changed)``."""
    m = matcher or default_matcher()
    out: list[str] = []
    found: list[str] = []
    changed: list[str] = []
    for p in phrases:
        names = m.find(p)
        if names:
            found += names
            changed.append(p)
            p = strip_names(p, m)
        if p:
            out.append(p)
    return clean_phrases(out), list(dict.fromkeys(found)), changed


def template_prompt(scene: dict, cfg: dict, kb: dict | None = None, *, count_fn: CountFn | None = None,
                    matcher: NameMatcher | None = None, lead: list[str] | None = None) -> dict:
    """Build the template prompt.

    Returns ``{"positive", "phrases_kept", "phrases_dropped", "names_leaked", "phrases_scrubbed"}``.
    ``lead`` phrases (e.g. a layout phrase) are placed before the template phrases.
    """
    phrases, leaked, scrubbed = scrub_template_phrases([*(lead or []), *template_phrases(scene, kb)], matcher)
    positive, kept, dropped = assemble(cfg["trigger"], phrases, cfg.get("style_tail", ""),
                                       int(cfg["max_prompt_tokens"]), count_fn=count_fn)
    return {"positive": positive, "phrases_kept": kept, "phrases_dropped": dropped,
            "names_leaked": leaked, "phrases_scrubbed": scrubbed}
