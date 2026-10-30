"""JSON file cache for LLM calls, so expansions are reproducible without loading an LLM.

Key: ``sha256("prompt_version|model|kb_version|mode|user_message")`` (D guide Day 2.3). ``kb_version``
separates c3 from c6, ``mode`` separates c3 from c4. One entry per LLM call; retries and repairs have
their own user message and therefore their own entry.

The file is written atomically (temp file + ``os.replace``) with sorted keys for stable diffs.
Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def cache_key(prompt_version: str, model: str, kb_version: str, mode: str, user_message: str) -> str:
    """sha256 hex digest of ``prompt_version|model|kb_version|mode|user_message``."""
    raw = "|".join([str(prompt_version), str(model), str(kb_version), str(mode), user_message])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Cache:
    """``{key: {"reply": str, "created": iso8601, **meta}}`` stored at ``path``.

    ``path=None`` keeps the cache in memory only. With ``autosave=True`` (default) every ``put`` writes
    the file, so an interrupted batch run resumes where it stopped.
    """

    def __init__(self, path: str | os.PathLike | None, *, autosave: bool = True) -> None:
        self.path = Path(path) if path else None
        self.autosave = autosave
        self.hits = 0
        self.misses = 0
        self._data: dict[str, dict] = {}
        if self.path and self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str) -> str | None:
        """The cached reply, or ``None``. Counts hits and misses."""
        entry = self._data.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry["reply"]

    def put(self, key: str, reply: str, **meta) -> None:
        """Store a reply (plus JSON-serialisable metadata such as model and prompt_version)."""
        self._data[key] = {"reply": reply, "created": datetime.now(UTC).isoformat(timespec="seconds"),
                           **meta}
        if self.autosave:
            self.save()

    def save(self) -> None:
        """Atomic write: temp file in the same folder, then ``os.replace``. No-op when in memory."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, sort_keys=True, indent=1, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
