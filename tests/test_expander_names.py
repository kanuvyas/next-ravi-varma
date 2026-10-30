"""Name matcher (expander/names.py): shared policy for the scrub and A's CI leak check."""

import json
from pathlib import Path

import pytest

from expander.names import (
    SEED_PATH,
    NameMatcher,
    default_matcher,
    default_names,
    find_names,
    load_names,
    normalize,
    scrub_phrases,
    strip_names,
)

ROOT = Path(__file__).resolve().parent.parent
TRAIN_V1 = ROOT / "m2_training" / "dataset" / "train_v1" / "metadata.jsonl"
SEED = NameMatcher(load_names(SEED_PATH))


@pytest.mark.parametrize("text, expected", [
    ("Rama lifts the bow", ["Rama"]),
    ("Sri Rama and Laxman in exile", ["Rama", "Lakshmana"]),
    ("Seetha in the grove", ["Sita"]),
    ("Rāma’s bow", ["Rama"]),                      # NFKD + curly apostrophe
    ("Lord Ram blesses", ["Rama"]),                # case-sensitive alias, capitalised
    ("Bheem and Yudhisthira", ["Bhima", "Yudhishthira"]),
    ("Ghatotkach lifts a chariot", ["Ghatotkacha"]),
    ("Kusha and Lava sing", ["Kusha", "Lava"]),
    ("Ravana-like demon", ["Ravana"]),
    ("the banks of the Ganga", ["Ganga"]),         # capitalised river = place name: scrubbed
    ("KRISHNA playing a flute", ["Krishna"]),      # case-insensitive by default
    ("Laxman, Rama and Lakshmana again", ["Lakshmana", "Rama"]),   # canonical, deduped, in order
    ("Hanuman\n  carrying a mountain", ["Hanuman"]),
])
def test_finds_names_across_aliases(text, expected):
    assert SEED.find(text) == expected


@pytest.mark.parametrize("text", [
    "a panorama of hills", "a woman playing a sitar", "rrvarma style, a woman in a red saree",
    "near an ashoka grove", "a ram tied near the altar", "a lava flow at dusk", "the river ganga at dawn",
    "a kusha grass mat", "a white swan, a hamsa", "a magnificent palace", "an agitated crowd",
    "a silk cushion", "an ambassador", "a canal", "a bull beside a snowy mountain",
])
def test_no_false_positives(text):
    assert SEED.find(text) == []


def test_ramayana_is_not_rama():
    assert SEED.find("a scene from the Ramayana") == ["Ramayana"]  # the epic's own name, never "Rama"


def test_train_v1_captions_are_name_free():
    if not TRAIN_V1.exists():
        pytest.skip("m2_training/dataset/train_v1/metadata.jsonl not present")
    texts = [json.loads(line)["text"] for line in TRAIN_V1.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(texts) > 100
    hits = {t: SEED.find(t) for t in texts if SEED.find(t)}
    assert hits == {}
    assert all(find_names(t) == [] for t in texts)  # the default matcher too


def test_normalize():
    assert normalize("Rāmāyaṇa") == "Ramayana"
    assert normalize("Rama’s") == "Rama s"


def test_load_names_shapes(tmp_path):
    flat = load_names(["Laxman", "Someone New"])
    assert flat == {"Laxman": "Lakshmana", "Someone New": "Someone New"}
    assert load_names({"names": ["Arjun"]}) == {"Arjun": "Arjuna"}
    assert load_names({"Karna": ["Radheya"]}) == {"Karna": "Karna", "Radheya": "Karna"}
    kb = {"version": "kb0.1",
          "characters": {"karna": {"canonical": "Karna", "aliases": ["Radheya"]}},
          "settings": {"kurukshetra": {"aliases": ["battlefield"], "visual": "open battlefield"}}}
    assert load_names(kb) == {"Karna": "Karna", "Radheya": "Karna"}  # setting aliases are generic words
    p = tmp_path / "names.json"
    p.write_text(json.dumps(["Karna", "Radheya"]))
    assert load_names(p) == {"Karna": "Karna", "Radheya": "Karna"}
    assert default_names(p) == {"Karna": "Karna", "Radheya": "Karna"}


def test_case_sensitive_policy_applies_to_any_source():
    m = NameMatcher(["ram", "Karna"])  # C might list a lowercase alias: it still needs a capital to match
    assert m.find("a ram and Karna") == ["Karna"]
    assert m.find("Ram and Karna") == ["ram", "Karna"]


def test_scrub_and_strip():
    kept, dropped = scrub_phrases(["a crowned king", "Sita weeping", "golden deer"], SEED)
    assert kept == ["a crowned king", "golden deer"] and dropped == ["Sita weeping"]
    assert strip_names("weeping over the fallen body of Karna", SEED) == "weeping over the fallen body"
    assert strip_names("Rama's long bow", SEED) == "long bow"
    assert strip_names("Karna", SEED) == ""
    assert strip_names("a golden deer", SEED) == "a golden deer"


def test_default_matcher_is_cached():
    assert default_matcher() is default_matcher()
