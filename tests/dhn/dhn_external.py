"""Helpers for locating and importing the official DHN repository."""

from __future__ import annotations

import sys
from pathlib import Path

DHN_REPO_URL = "https://github.com/gear/dhn.git"
# Pin for reproducible baseline numbers in this repo.
DHN_PINNED_COMMIT = "c1084b37f303d14952d514b7c65e53dc1c1df59a"


def repo_root_from_here() -> Path:
    """Return parent repo root from files under tests/dhn."""
    return Path(__file__).resolve().parents[3]


def external_dhn_root() -> Path:
    """Return expected checkout location for official DHN source."""
    return repo_root_from_here() / "_external" / "dhn"


def ensure_external_dhn_on_path() -> Path:
    """Add `_external/dhn` to sys.path and return that path."""
    root = external_dhn_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root

