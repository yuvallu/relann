# tests/ Layout

Standalone pytest files for the relann package. Inline unit-style asserts that live next to the code they cover (under `if test():` blocks inside `relann/*.py`) are handled separately by pytest via `python_files = ["*.py"]`.

## Buckets

- **`tests/smoke/`** — fastest sanity checks (~5s) — session creation, DSL parsing, template specialization.
- **`tests/feature/`** — feature validation (~15s) — joins, aggregations, transformations, encode/decode, smart ops, tensor compilation, data sources.
- **`tests/dhn/`** — Deep Hypergraph Networks unit tests and benchmarks (~60s). Datasets live in `tests/dhn/data/`.
- **`tests/slow/`** — long-running comparison scripts: HGT vs PyG HGTConv, Cora PyTorch reference, DBLP HGT, etc.
- **`tests/repro/`** — regression / reproduction tests for specific bug fixes.
- **`tests/scaffold/`** — scaffold-style end-to-end tests migrated from notebook form (`test_902_gcn_relnn.py`, `test_912_scaffold_gcn_cora.py`).

## Running

```bash
uv run poe smoke                     # ~5s
uv run poe quick                     # smoke + feature (~15s)
uv run pytest tests/dhn -v           # ~60s
uv run pytest tests/                 # full sweep
uv run pytest -m "not slow" tests/   # skip slow bucket
```

## Tests inside `relann/*.py`

Pytest also collects `if test():` blocks inside the relann source modules via `python_files = ["*.py"]` in `pyproject.toml`. Those are the literate-programming-style inline asserts.
