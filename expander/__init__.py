"""Member D's prompt expander (Milestone 3): ``expander.expand.expand(text, scene=None, mode="llm_kb")``.

Importing the package is cheap and stdlib-only. ``expander.close()`` frees the default LLM client
(A's ``text_to_image`` calls it before the first ``generate``).
"""


def close() -> None:
    """Close the default LLM client registered with ``expander.expand.set_default_llm`` (if any)."""
    from expander.expand import close as _close

    _close()


__all__ = ["close"]
