# Repository Architecture

This document is the long-lived architecture reference for repository layout and ownership.

> Last updated: 2026-05-23, after the migration from nbdev/conda/cursor to juplit/uv/claude.

## Source Of Truth

- **`.py` files under `relann/` and `examples/` are the source of truth.** They are juplit-paired with `.ipynb` notebooks via jupytext (`formats: ipynb,py:percent` header).
- `.ipynb` files are **generated on demand** by `uv run poe nb` (or `juplit nb`) and are **gitignored**.
- Edit the `.py` directly. If you also opened it in Jupyter, Jupytext keeps the `.ipynb` in sync on save. If you edit the `.py` from outside Jupyter, run `uv run poe sync` before reopening the notebook so the on-disk `.ipynb` doesn't go stale.
- There is no separate notebook source directory anymore (no `nbs/`). Each module under `relann/` *is* its own paired notebook.

## Directory Responsibilities

| Path | Purpose |
|---|---|
| `relann/` | The Python package. Top-level files (`engine.py`, `parser.py`, etc.) are juplit-paired modules. Subpackages (`utils/`) are plain Python — no notebooks. |
| `relann/relann_grammar.lark` | Lark grammar for the RelaNN DSL. Shipped as package data via `pyproject.toml`. |
| `tests/` | Single test root. Organised into buckets: `smoke/`, `feature/`, `dhn/`, `slow/`, `repro/`, `scaffold/`. See `tests/README.md`. |
| `examples/` | User-facing demos (juplit `.py`). Each demo is a paired notebook a user opens to learn the framework. |
| `research/` | Reproducibility artefacts not part of the installable package: `paper_experiments/` (per-paper benchmark drivers), `_drafts/` (in-progress sketches). |
| `docs/` | Long-lived design and reference documentation. See `docs/design/` for ADR-style decisions, `docs/_archive/` for historical content. |
| `.claude/skills/` | Claude Code skills (DSL reference, repo overview, conventions, etc.). |
| `scripts/` | Developer automation: `run_tests.py` profile runner, `setup_external_dhn.py`. |
| `pyproject.toml` | Single source of truth for the build (uv backend), dependencies, poe tasks, juplit config, jupytext format, pytest config and markers. |
| `data/` | Datasets (gitignored). PyG `Planetoid` and others land here on first use. |
| `_deprecated/`, `_external/` | Untouched legacy / external comparison code. Both gitignored. |

## Source Code Editing

- **Top-level `relann/*.py`** uses **absolute** imports (`from relann.term_graph import TermGraph`) — these files are also opened as Jupyter notebooks, and Jupyter doesn't load notebooks as package members so relative imports break.
- **Subpackage files** (`relann/utils/*.py`) use **relative** imports (`from .algebra import Op`) — conventional for intra-package code; they are never opened as notebooks.
- See `.claude/skills/relann-conventions.md` for the full convention list.

## Plan And Design Separation

Two-layer model:

1. **Design docs** (`docs/design/*.md`) — durable architecture decisions and rationale.
2. **Execution plans** — short-lived task plans. They can live anywhere convenient (or use Claude's plan files in `.claude/plans/` which are gitignored).

Every plan should include a "Design refs" line pointing at the design docs it touches.

## Test Placement Rules

- All repository tests live under `tests/` (root).
- Organise by runtime/intent bucket: `smoke`, `feature`, `dhn`, `slow`, `repro`, `scaffold`.
- Keep experiment-heavy comparisons outside `smoke` and `feature` — they belong in `slow` or `research/paper_experiments/`.
- Inline `if test():` blocks inside `tests/*.py` are also picked up by pytest via `python_files = ["*.py"]` in `pyproject.toml`. See `docs/design/testing-strategy.md`.

## Naming And Stability

- `test_*.py` for normal pytest tests, `run_*.py` for explicit scenario runners (datasets, benchmarks).
- `repro/` for issue-focused reproductions that may become historical artifacts.
- Smoke must stay small and predictable so it's safe to run often.
- RelaNN operator access is node-based: use `module_for_node(node_id)` (Python) and `node.<node_id>` (scaffold hook paths). Treat `ops` keys as internal implementation detail only.
