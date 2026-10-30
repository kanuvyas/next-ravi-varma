"""S1 stub of ``expand()`` (D guide v2 Day 1.2): returns ``"<trigger>, <raw text>, <style_tail>"``.

A's ``text_to_image`` imports this until S2, then switches to ``from expander.expand import expand``.
Same as the guide, except that the config comes from ``expander._cfg`` (works from any working
directory, and before ``config/m3.json`` exists) and an empty tail adds no trailing ``", "``.
"""

from expander._cfg import load_config

CFG = load_config()


def expand(text, scene=None, mode="llm_kb"):
    positive = ", ".join([CFG["trigger"], text] + ([CFG["style_tail"]] if CFG["style_tail"] else []))
    return {"scene_id": (scene or {}).get("id", ""), "condition": mode, "split": "",
            "positive": positive, "negative": CFG["negative"],
            "phrases_kept": [text], "phrases_dropped": [], "token_count": -1, "trigger_first": True,
            "names_leaked": [], "source": "fallback", "layout": None, "llm_model": "none",
            "prompt_version": "stub", "kb_version": "none", "latency_s": 0.0}
