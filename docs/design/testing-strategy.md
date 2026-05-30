# Testing Strategy

This document defines test taxonomy, placement, and runtime tiers.

> Last updated: 2026-05-23, after the migration from `nbs/tests/` to root-level `tests/`.

Design refs:
- `docs/design/repo-structure.md`

## Goals

- Day-to-day validation fast enough to run manually before commits.
- Deeper feature and regression checks for focused runs.
- Full validation path (everything under `tests/`) when needed.

## Taxonomy

- `smoke`: very fast checks that verify critical paths and shape contracts.
- `feature`: targeted behavior tests for specific operators/subsystems.
- `integration`: broader flows that compose multiple subsystems.
- `slow`: expensive tests (dataset-heavy, long training loops, full-db scenarios).
- `download`: tests that may require cached downloads (e.g. Planetoid).
- `repro`: issue reproductions and historical regression cases.

Markers are declared in `pyproject.toml` under `[tool.pytest.ini_options].markers`.

## Placement Policy

- `tests/` is the single test root.
- Organize by bucket: `tests/smoke/`, `tests/feature/`, `tests/dhn/`, `tests/slow/`, `tests/repro/`, `tests/scaffold/`.
- `scaffold/` contains end-to-end scenario tests converted from the original notebook scaffolds (`902_*`, `912_*`, `913_*`).

## Profiles (the runner)

Use `scripts/run_tests.py` or the equivalent `poe` task:

| Profile | Command | Time | What it runs |
|---|---|---|---|
| smoke | `uv run poe smoke` | ~5s | `tests/smoke` |
| quick | `uv run poe quick` | ~15s | `tests/smoke` + `tests/feature` |
| hgt | `uv run python scripts/run_tests.py hgt` | varies | `tests/slow` (HGT scripts) |
| dhn | `uv run python scripts/run_tests.py dhn` | ~60s | `tests/dhn` |
| full | `uv run poe test` | ~6min | all of `tests/` |

## Inline `if test():` blocks

Test files (under `tests/`) may use `from juplit import test` and gate code with `if test():`. During pytest collection, `juplit.test()` returns `True` (because `pytest` is in `sys.modules`), so the block runs. Outside pytest the block is dormant. Pytest discovers them via `python_files = ["*.py"]` in `pyproject.toml`.

## Source modules (`relann/*.py`)

Source modules use `if __name__ == "__main__":` (NOT `if test():`) for their notebook-demo cells. Reason: under nbdev, these were `#| hide` / `#| eval: false` cells skipped by automated test runs. Using `if test():` would have fired them under pytest and caused circular imports between `term_graph`, `parser`, and `engine`. Now they only fire when the file is run as a script (e.g., `python relann/engine.py`) or opened as a notebook in Jupyter.

Some of these inline demos are ordering-dependent and break under "Run All Cells" or `python relann/engine.py`. This was the case under nbdev too — see `docs/design/notebook-demos.md` for the known-broken list and the decision-tree on how to handle them.

## Naming And Stability

- `test_*.py` for normal pytest tests.
- `run_*.py` for scenario runners (datasets, benchmarks).
- `repro/repro_*` and `repro/test_*_<date>.py` for issue reproductions.
- Smoke must stay small and predictable so it's safe to run often.
