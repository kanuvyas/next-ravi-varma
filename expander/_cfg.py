"""Where the expander finds its settings and its collaborators' code.

Each lookup is ONE small function, so the S2 swap (shim -> real file) is a one-line change:

- ``config_path`` / ``load_config``: ``$M3_CONFIG`` > argument > ``config/m3.json`` > ``_shims/m3.default.json``
- ``get_counter``: ``common.tokens.n_tokens`` (A) > ``_shims/tokens.py``
- ``get_parser``: ``kb.parser.parse_scene`` (C, S2) > ``kb.stub_parser.parse_scene`` (C, S1) > ``_shims/stub_parser.py``
- ``default_kb_path``: ``kb/kb.json`` > ``kb/stub_kb.json`` > ``_shims/stub_kb.json``

Standard library only at import time (CI has no ``transformers``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

SHIM_DIR = Path(__file__).resolve().parent / "_shims"


def repo_root() -> Path:
    """The repository root (the folder that holds ``expander/``)."""
    return Path(__file__).resolve().parent.parent


def config_path(path: str | os.PathLike | None = None) -> Path:
    """Resolve the config file: ``$M3_CONFIG`` > ``path`` > ``config/m3.json`` > the shim."""
    env = os.environ.get("M3_CONFIG")
    if env:
        return Path(env)
    if path:
        return Path(path)
    real = repo_root() / "config" / "m3.json"
    return real if real.exists() else SHIM_DIR / "m3.default.json"


@lru_cache(maxsize=16)
def _read_cached(resolved: str, mtime_ns: int) -> str:
    with open(resolved, encoding="utf-8") as f:
        return f.read()


def _read_json(path: Path) -> dict:
    """Parse a JSON file; the text is cached per (path, mtime), the returned dict is always fresh."""
    return json.loads(_read_cached(str(path), path.stat().st_mtime_ns))


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Return a fresh dict with the team config (callers may mutate it safely).

    Keys the team config lacks (e.g. ``llm`` before D's config PR lands) are filled from the
    shim and listed in ``_defaults_from_shim``; keys the team config has always win.
    """
    resolved = config_path(path)
    cfg = _read_json(resolved)
    shim = SHIM_DIR / "m3.default.json"
    if resolved.resolve() == shim.resolve():
        return cfg
    defaults = _read_json(shim)
    filled = []
    for key, value in defaults.items():
        if key.startswith("_") or key == "models":
            continue
        if key not in cfg:
            cfg[key] = value
            filled.append(key)
        elif key == "llm" and isinstance(cfg[key], dict) and isinstance(value, dict):
            for sub, v in value.items():
                if sub not in cfg[key]:
                    cfg[key][sub] = v
                    filled.append(f"llm.{sub}")
    if filled:
        cfg["_defaults_from_shim"] = filled
    return cfg


def get_counter() -> Callable[[str], int]:
    """The ONE token counter (excludes start/end tokens). A's ``common/tokens.py`` once it exists."""
    # A's file imports transformers at import time; the shim is the same algorithm with a lazy import.
    try:
        from common.tokens import n_tokens
    except (ImportError, OSError):
        from expander._shims.tokens import n_tokens
    return n_tokens


def get_parser() -> Callable[..., dict]:
    """C's ``parse_scene(text, scene_id="", kb_path=None) -> SceneSpec``: real > stub > shim."""
    try:
        from kb.parser import parse_scene
    except ImportError:
        try:
            from kb.stub_parser import parse_scene
        except ImportError:
            from expander._shims.stub_parser import parse_scene
    return parse_scene


def default_kb_path() -> Path:
    """The KB file for mood lookups when no ``kb_path`` is given."""
    root = repo_root()
    for p in (root / "kb" / "kb.json", root / "kb" / "stub_kb.json"):
        if p.exists():
            return p
    return SHIM_DIR / "stub_kb.json"


def load_kb(kb_path: str | os.PathLike | None = None) -> dict:
    """Load a KB file (C's schema). ``None`` means ``default_kb_path()``."""
    return _read_json(Path(kb_path) if kb_path else default_kb_path())
