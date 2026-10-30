"""Make the repository root importable when CI runs ``pytest -q tests``; register the ``slow`` marker."""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs optional heavy dependencies or downloads; skipped otherwise")
