"""Checks for Member D's 8 dev scenes (expander/dev_scenes_D.json).

CPU only, standard library plus pytest; jsonschema is optional.
The rules come from the Team Overview v2, section 6.2 (scene schema, tiers,
checklist rules) and section 10 (coverage).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCENES_PATH = REPO / "expander" / "dev_scenes_D.json"
NAMES_SEED_PATH = REPO / "expander" / "names_seed.json"

ID_RE = re.compile(r"^s(2[5-9]|3[0-2])$")  # D's proposed block: s25-s32
TIERS = {1, 2, 3, 4, 5}
EPICS = {"Mahabharata", "Ramayana", "Puranic", "Other", "Unknown"}
ITEM_TYPES = {"count", "figure", "pose", "attire", "object", "setting", "mood"}
WEIGHTS = {1.0, 0.5}  # central items 1.0, background items 0.5

REQUIRED = {
    "id": str,
    "split": str,
    "text": str,
    "tier": int,
    "expected_epic": str,
    "expected_characters": list,
    "seen_in_training": bool,
    "contributed_by": str,
    "checklist": list,
}
ITEM_REQUIRED = {"id": str, "q": str, "type": str, "weight": (int, float)}

# Style is measured separately, so questions must not ask about it.
STYLE_RE = re.compile(
    r"\b(paint\w*|style\w*|brush\w*|realis\w*|varma|canvas\w*|artist\w*|artwork\w*|"
    r"masterpiece\w*|academic|oil\s+painting)\b",
    re.IGNORECASE,
)
# Things a 512x512 SD1.5 image cannot plausibly show.
UNANSWERABLE_RE = re.compile(
    r"\b(fingers?|toes?|text|writing|written|letters?|words?|inscriptions?|signature|caption)\b",
    re.IGNORECASE,
)
YES_NO_START = ("Is ", "Are ", "Does ", "Do ", "Has ", "Have ", "Can ")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip()


def _all_items(scenes):
    for s in scenes:
        for item in s["checklist"]:
            yield s["id"], item


@pytest.fixture(scope="module")
def scenes():
    with SCENES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "the scenes file must be a JSON list (Overview 6.2)"
    return data


@pytest.fixture(scope="module")
def names_seed():
    if not NAMES_SEED_PATH.exists():
        pytest.skip(f"{NAMES_SEED_PATH.relative_to(REPO)} not present yet (shipped with the expander code)")
    with NAMES_SEED_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _flat_names(seed) -> list[str]:
    if isinstance(seed, dict):
        out = []
        for canon, aliases in seed.items():
            out.append(canon)
            out.extend(aliases or [])
        return out
    if isinstance(seed, list):
        return list(seed)
    raise TypeError(f"unexpected names seed type {type(seed).__name__}")


# --- schema ---------------------------------------------------------------


def test_eight_scenes(scenes):
    assert len(scenes) == 8


def test_required_fields_and_types(scenes):
    for s in scenes:
        for key, typ in REQUIRED.items():
            assert key in s, f"{s.get('id')}: missing {key}"
            assert isinstance(s[key], typ), f"{s['id']}: {key} should be {typ.__name__}"
        assert not isinstance(s["tier"], bool)
        assert s["text"].strip(), f"{s['id']}: empty text"
        assert s["split"] == "dev"
        assert s["contributed_by"] == "D"
        assert s["tier"] in TIERS, f"{s['id']}: tier {s['tier']}"
        assert s["expected_epic"] in EPICS, f"{s['id']}: epic {s['expected_epic']}"
        chars = s["expected_characters"]
        assert all(isinstance(c, str) and c.strip() for c in chars)
        assert len(chars) == len(set(chars)), f"{s['id']}: duplicate characters"
        assert isinstance(s.get("_seen_note"), str) and s["_seen_note"], f"{s['id']}: explain seen_in_training"
        assert isinstance(s.get("_tags", []), list)
        assert isinstance(s.get("_groups", []), list)


def test_ids_unique_and_in_block(scenes):
    ids = [s["id"] for s in scenes]
    assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"
    for i in ids:
        assert ID_RE.match(i), f"{i} is outside D's block s25-s32"


def test_checklist_items(scenes):
    for s in scenes:
        cl = s["checklist"]
        assert 3 <= len(cl) <= 6, f"{s['id']}: {len(cl)} checklist items (need 3-6)"
        item_ids = [it.get("id") for it in cl]
        assert len(item_ids) == len(set(item_ids)), f"{s['id']}: duplicate item ids"
        for it in cl:
            for key, typ in ITEM_REQUIRED.items():
                assert key in it, f"{s['id']}/{it.get('id')}: missing {key}"
                assert isinstance(it[key], typ), f"{s['id']}/{it['id']}: bad {key}"
            assert it["type"] in ITEM_TYPES, f"{s['id']}/{it['id']}: type {it['type']}"
            assert float(it["weight"]) in WEIGHTS, f"{s['id']}/{it['id']}: weight {it['weight']}"


def test_jsonschema_valid(scenes):
    jsonschema = pytest.importorskip("jsonschema")
    item = {
        "type": "object",
        "required": list(ITEM_REQUIRED),
        "properties": {
            "id": {"type": "string"},
            "q": {"type": "string", "minLength": 5},
            "type": {"enum": sorted(ITEM_TYPES)},
            "weight": {"type": "number", "exclusiveMinimum": 0},
        },
    }
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": list(REQUIRED),  # extra fields are allowed
            "properties": {
                "id": {"type": "string"},
                "split": {"enum": ["dev"]},
                "text": {"type": "string", "minLength": 1},
                "tier": {"type": "integer", "minimum": 1, "maximum": 5},
                "expected_epic": {"enum": sorted(EPICS)},
                "expected_characters": {"type": "array", "items": {"type": "string"}},
                "seen_in_training": {"type": "boolean"},
                "contributed_by": {"type": "string"},
                "checklist": {"type": "array", "minItems": 3, "maxItems": 6, "items": item},
            },
        },
    }
    jsonschema.validate(scenes, schema)


# --- coverage -------------------------------------------------------------


def test_every_tier_present(scenes):
    tiers = Counter(s["tier"] for s in scenes)
    assert set(tiers) == TIERS, f"tiers present: {sorted(tiers)}"
    # one scene per tier, the remaining three in tiers 2-3 (where the experiments bite)
    assert tiers[2] + tiers[3] >= 5, tiers


def test_epic_mix(scenes):
    epics = Counter(s["expected_epic"] for s in scenes)
    # team-wide target is >= 10 Ramayana and >= 10 Mahabharata from 32 scenes
    assert epics["Ramayana"] >= 3, epics
    assert epics["Mahabharata"] >= 3, epics


def test_tier_2_3_have_count_or_pose_items(scenes):
    # H-D3 (layout, c5) is scored on count/pose items in tiers 2-3
    for s in scenes:
        if s["tier"] in (2, 3):
            types = {it["type"] for it in s["checklist"]}
            assert types & {"count", "pose"}, f"{s['id']}: no count/pose item"


def test_several_item_types_per_scene(scenes):
    for s in scenes:
        types = {it["type"] for it in s["checklist"]}
        assert len(types) >= 3, f"{s['id']}: only {sorted(types)}"
        assert any(it["weight"] == 1.0 for it in s["checklist"]), f"{s['id']}: no central item"


def test_stress_tags(scenes):
    def tagged(tag):
        return [s for s in scenes if tag in s.get("_tags", [])]

    miss = [s for s in tagged("misspelling") if s["tier"] in (4, 5)]
    assert miss, "need a misspelling-flagged scene in tier 4 or 5"
    vague = [s for s in tagged("vague") if s["tier"] == 5]
    assert vague, "need a vague tier-5 scene"
    long_ = [s for s in tagged("long_text") if s["tier"] == 5]
    assert long_, "need a long tier-5 scene"
    for s in long_:
        assert len(s["text"].split()) >= 100, f"{s['id']}: long_text has {len(s['text'].split())} words"


# --- checklist content rules -----------------------------------------------


def test_questions_are_yes_no(scenes):
    for sid, it in _all_items(scenes):
        q = it["q"].strip()
        assert q.endswith("?"), f"{sid}/{it['id']}: not a question"
        assert q.startswith(YES_NO_START), f"{sid}/{it['id']}: not a yes/no question: {q!r}"


def test_no_style_words_in_questions(scenes):
    for sid, it in _all_items(scenes):
        m = STYLE_RE.search(it["q"])
        assert not m, f"{sid}/{it['id']}: style word {m.group(0)!r} in {it['q']!r}"


def test_no_unanswerable_details(scenes):
    for sid, it in _all_items(scenes):
        m = UNANSWERABLE_RE.search(it["q"])
        assert not m, f"{sid}/{it['id']}: {m.group(0)!r} is not judgeable at 512x512"


def test_no_names_in_questions(scenes, names_seed):
    names = sorted({_norm(n) for n in _flat_names(names_seed) if n and n.strip()}, key=len, reverse=True)
    assert names, "names seed is empty"
    pattern = re.compile(
        r"\b(" + "|".join(r"\s+".join(map(re.escape, n.split())) for n in names) + r")\b",
        re.IGNORECASE,  # stricter than the scrub's mixed case policy, on purpose
    )
    hits = []
    for sid, it in _all_items(scenes):
        m = pattern.search(_norm(it["q"]))
        if m:
            hits.append(f"{sid}/{it['id']}: {m.group(0)!r} in {it['q']!r}")
    assert not hits, "names in checklist questions:\n" + "\n".join(hits)


def test_expected_characters_are_canonical(scenes, names_seed):
    if not isinstance(names_seed, dict):
        pytest.skip("names seed is a flat list; canonical names cannot be told apart from aliases")
    canon = set(names_seed)
    for s in scenes:
        for c in s["expected_characters"] + s.get("_groups", []):
            assert c in canon, f"{s['id']}: {c!r} is not a canonical name in the names seed"
