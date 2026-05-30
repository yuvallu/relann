"""pytest wrappers for the `run_*.py` slow scripts.

These exist so pytest at least *discovers* the slow scripts and reports
their pass/fail in a normal CI/dev workflow. Each test is marked `slow`
so it's skipped by default; opt in with::

    pytest -m slow tests/slow/

Each wrapper invokes the underlying script in a subprocess and asserts a
clean exit. A "SKIP: <data> not found" line in the script's output (printed
when the script bails because Cora / DBLP / pyHGT aren't present) is treated
as a soft skip via ``pytest.skip``, NOT a failure.

Why subprocess instead of importing-and-calling
-----------------------------------------------
The slow scripts are top-level modules with no `def main():`. Importing
them would execute every line at top level, including model training,
which we cannot retract cleanly between tests. Subprocesses give clean
isolation and accurate per-script timing.

What this does NOT do
---------------------
- It does NOT promote the assertions inside each script to pytest-level
  test functions. If you want finer-grained assertions ("param count
  matches", "weight-synced forward diff < 1e-5"), refactor the target
  script to expose them as helpers and write per-assertion tests against
  those helpers. The current wrappers only catch "did the script crash?".
- It does NOT install `_external/pyHGT` for you. See CHEATSHEET.md.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SLOW_DIR = _REPO_ROOT / "tests" / "slow"


def _run(script_name: str, timeout: int = 900) -> None:
    """Invoke the script and assert it exits cleanly.

    A ``SKIP:`` line in the script's stdout is treated as a pytest.skip;
    a non-zero exit code or any other error is a test failure.
    """
    script = _SLOW_DIR / script_name
    assert script.exists(), f"missing script: {script}"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(_REPO_ROOT),
    )
    combined = (result.stdout or "") + (result.stderr or "")
    if "SKIP:" in combined:
        # Script bailed because required data / external dep is missing.
        # Surface the line so the skip message is useful.
        skip_line = next(
            (line for line in combined.splitlines() if line.startswith("SKIP:")),
            "SKIP: (no details)",
        )
        pytest.skip(skip_line)
    assert result.returncode == 0, (
        f"{script_name} exited with code {result.returncode}.\n"
        f"--- stdout tail ---\n{(result.stdout or '')[-2000:]}\n"
        f"--- stderr tail ---\n{(result.stderr or '')[-2000:]}"
    )


# Tag every test with `slow` so they're opt-in. Each wrapper is one line.

@pytest.mark.slow
def test_run_compare_dblp_hgt():
    _run("run_compare_dblp_hgt.py")


@pytest.mark.slow
def test_run_compare_dblp_hgt_generic():
    """Canonical RelNN HGT vs PyTorch HGT 1-to-1 comparison.
    Produced the published 2026-03-30 DBLP benchmark."""
    _run("run_compare_dblp_hgt_generic.py")


@pytest.mark.slow
def test_run_compare_dblp_hgt_multirun():
    _run("run_compare_dblp_hgt_multirun.py", timeout=1500)  # 5-run, ~15 min


@pytest.mark.slow
def test_run_compare_dblp_original_hgt():
    _run("run_compare_dblp_original_hgt.py")


@pytest.mark.slow
def test_run_hgt_generic_2l_accuracy():
    """Known to FAIL on `main` AND juplit with a long-standing NaN-at-epoch-0
    bug in the 2-layer attention. Wrapped here for visibility; mark
    `xfail` once a tracking issue is filed."""
    _run("run_hgt_generic_2l_accuracy.py")


@pytest.mark.slow
def test_run_hgt_template_cora():
    _run("run_hgt_template_cora.py")


@pytest.mark.slow
def test_run_match_hgt_2l_accuracy():
    _run("run_match_hgt_2l_accuracy.py")


@pytest.mark.slow
def test_run_match_hgt_accuracy():
    _run("run_match_hgt_accuracy.py")
