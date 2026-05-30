# Testing Workflow

This repository uses pytest. Tests live under `tests/` organised by bucket. Inline literate-style asserts inside test files can be gated with `if test():`. Source modules under `relann/*.py` use `if __name__ == "__main__":` for their notebook-demo cells (so they run interactively in Jupyter but **not** under pytest — that avoids circular imports between term_graph ↔ parser ↔ engine).

## First-time setup

See [README.md](README.md) for the full install. Short version:

```bash
uv sync
uv pip install --no-build-isolation \
    torch-scatter torch-sparse torch-cluster torch-geometric \
    -f "https://data.pyg.org/whl/torch-2.6.0+cu124.html"     # CUDA host
uv pip install scipy nbdev stringdale                         # transitive deps
```

## Runner

Prefer poe via uv:

```bash
uv run poe smoke         # ~5s    pytest tests/smoke
uv run poe quick         # ~15s   pytest tests/smoke tests/feature
uv run poe test          # full   pytest tests/
```

Or the script:

```bash
uv run python scripts/run_tests.py smoke
uv run python scripts/run_tests.py quick
uv run python scripts/run_tests.py hgt
uv run python scripts/run_tests.py dhn
uv run python scripts/run_tests.py full
```

## Profiles

| Profile | Time   | What it runs                                  | When to use                                       |
|---------|--------|-----------------------------------------------|---------------------------------------------------|
| `smoke` | ~5s    | `pytest tests/smoke`                          | Quick sanity before any commit                    |
| `quick` | ~15s   | `pytest tests/smoke tests/feature`            | Default pre-commit check                          |
| `hgt`   | varies | `pytest tests/slow` (HGT scripts)             | After HGT attention / message-passing changes     |
| `dhn`   | ~60s   | `pytest tests/dhn`                            | After DHN / homomorphism changes                  |
| `full`  | ~6min  | `pytest tests/`                               | Before merging PRs to `main`                      |

## Markers

Defined in `pyproject.toml` under `[tool.pytest.ini_options].markers`:

- `smoke` — fast critical-path validation
- `feature` — targeted subsystem behavior checks
- `integration` — multi-component workflow checks
- `slow` — expensive and dataset-heavy tests
- `download` — may require cached downloads (e.g. Planetoid)
- `repro` — issue reproduction and historical regressions

Skip slow tests: `uv run pytest -m "not slow" tests/`.

## `if test():` inside test files

Some test files (mostly the ones converted from notebooks under `tests/scaffold/`, `tests/feature/test_903_*`, `tests/feature/test_904_*`, `tests/slow/test_hygnn_*`) use `if test():` from `juplit` to gate test code:

```python
from juplit import test

def my_helper():
    return 42

if test():
    assert my_helper() == 42
```

During pytest collection, `juplit.test()` evaluates to `True` (because `pytest` is in `sys.modules`), so the block runs. Outside pytest (plain `python -m tests.X`), it returns `False`.

## Source modules (`relann/*.py`)

These use `if __name__ == "__main__":` for their notebook-demo cells, not `if test():`. This intentionally hides those cells from pytest so that loading the source modules during `import relann` doesn't trigger the circular dependency between `term_graph`, `parser`, and `engine`. Open the file as a Jupyter notebook (via `uv run poe nb`) to run the demos interactively.

## Test layout

See [tests/README.md](tests/README.md).
