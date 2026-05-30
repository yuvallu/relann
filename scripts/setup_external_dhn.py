"""Clone/update official DHN repo at a pinned commit under `_external/dhn`.

Usage:
  uv run python scripts/setup_external_dhn.py
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/gear/dhn.git"
PINNED_COMMIT = "c1084b37f303d14952d514b7c65e53dc1c1df59a"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--commit", default=PINNED_COMMIT)
    parser.add_argument("--target", default="_external/dhn")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    target = (root / args.target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.exists():
        _run(["git", "clone", args.repo_url, str(target)])
    else:
        _run(["git", "fetch", "--all", "--tags"], cwd=target)

    _run(["git", "checkout", args.commit], cwd=target)
    print(f"Ready: {target}")
    print(f"Pinned commit: {args.commit}")


if __name__ == "__main__":
    main()

