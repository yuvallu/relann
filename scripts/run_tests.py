"""Manual test profile runner for relann.

Profiles:
- smoke   (~5s)   — pytest tests/smoke
- quick   (~15s)  — pytest tests/smoke tests/feature
- hgt             — pytest tests/slow (HGT scripts)
- dhn     (~60s)  — pytest tests/dhn
- full    (~6min) — pytest tests/

This runner does NOT call `nbdev_export` or `nbdev_prepare` — the project
migrated from nbdev to juplit. To regenerate paired notebooks, run
`uv run poe sync` (or `juplit sync`).

Prefer running via poe: `uv run poe smoke`, `uv run poe quick`, etc.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SMOKE = [Path("tests/smoke")]
QUICK = [Path("tests/smoke"), Path("tests/feature")]
HGT   = [Path("tests/slow")]
DHN   = [Path("tests/dhn")]
FULL  = [Path("tests")]


def _run_pytest(paths: list[Path], dry_run: bool) -> int:
    cmd = [sys.executable, "-m", "pytest", *[str(p) for p in paths]]
    print("$", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def run_profile(profile: str, dry_run: bool) -> int:
    return _run_pytest({
        "smoke": SMOKE,
        "quick": QUICK,
        "hgt":   HGT,
        "dhn":   DHN,
        "full":  FULL,
    }[profile], dry_run=dry_run)


def _print_profiles() -> None:
    print("Profiles:")
    print("  smoke:  (~5s)   pytest tests/smoke")
    print("  quick:  (~15s)  pytest tests/smoke tests/feature")
    print("  hgt:            pytest tests/slow")
    print("  dhn:    (~60s)  pytest tests/dhn")
    print("  full:   (~6min) pytest tests")


def main() -> int:
    p = argparse.ArgumentParser(description="Run a named test profile.")
    p.add_argument("profile", choices=["smoke", "quick", "hgt", "dhn", "full"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list:
        _print_profiles()
        return 0
    return run_profile(args.profile, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
