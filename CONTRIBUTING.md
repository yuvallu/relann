# Contributing to relann

Welcome. This is the short guide for picking up work in this repo without footguns.

If you're an AI agent, also read `AGENTS.md` and the skills under `.claude/skills/` first.

## One-time setup

```bash
# 1. Install uv (https://astral.sh/uv/)
pip install uv

# 2. Clone + sync
git clone <this-repo> && cd relann
uv sync                                  # creates .venv with base deps

# 3. PyG sparse stack (needed at import time by relann.era_operations)
uv pip install --no-build-isolation \
    torch-scatter torch-sparse torch-cluster torch-geometric \
    -f "https://data.pyg.org/whl/torch-2.6.0+cu124.html"   # cu124 → cpu for laptop

# 4. Git hooks + initial notebook generation
uv run poe init
uv run poe nb
```

For GPU / RAPIDS / cuDF, see `docs/install-gpu.md`. The base setup above is enough for tests and most demos.

If your `uv sync` ever removes the PyG packages (it will, because they're not in `pyproject.toml`), just rerun the `uv pip install` line from step 3.

## Daily commands

See **`CHEATSHEET.md`** at repo root — it has the canonical list. Key ones:

```bash
uv run poe smoke         # ~5s    quick sanity
uv run poe quick         # ~15s   smoke + feature
uv run poe test          # full   pytest tests/
uv run poe sync          # sync .py ↔ .ipynb
uv run poe nb            # regenerate .ipynb from .py
uv run poe clean         # sync, then delete all .ipynb (prep for AI agents)
```

## Adding a new source module

1. Create `relann/<my_module>.py` with the jupytext header (copy from `relann/log_utils.py` for a template).
2. Write your code as cells: `# %%` for code, `# %% [markdown]` for prose.
3. Use **absolute** imports (`from relann.other_module import foo`), not relative (`from .other_module import foo`). Reason: top-level `relann/*.py` files are also opened as Jupyter notebooks, and Jupyter doesn't load them as package members — relative imports break.
4. Inline tests / examples: wrap in `if test():` (test files only) or `if __name__ == "__main__":` (notebook-demo cells in source modules).
5. Run `uv run poe sync` after editing to keep the `.ipynb` consistent.

See `.claude/skills/relann-conventions.md` for full conventions.

## Adding a new test

- Pick a bucket under `tests/`: `smoke/`, `feature/`, `dhn/`, `optimizer/`, `optimizer_v2/`, `slow/`, `repro/`, or `scaffold/`. See `tests/README.md` for what each bucket is for.
- Name your file `test_<thing>.py`.
- Import from `relann.*` directly (no `sys.path` hacks needed — the venv has `relann` installed editable).
- Run `uv run pytest tests/<bucket>/test_<thing>.py -v` to verify.

If your test uses `if test():` blocks, import `test` from `juplit` first:

```python
from juplit import test

def my_thing(): ...

if test():
    assert my_thing() == ...

def test_my_thing():
    assert my_thing() == ...
```

Pytest collects both the `if test():` body (via `python_files = ["*.py"]`) and the `def test_*` function.

## Adding a new example (demo)

Create `examples/<NNN>_<name>.py` (zero-padded number for sort order). Same juplit cell format as source modules. Examples are designed to "Run All Cells" end-to-end in Jupyter — make sure yours does.

## Editing notebooks safely

The `.py` is the source of truth. The `.ipynb` is generated.

- **In VS Code**: open the `.py` directly. The Python + Jupyter extensions give you cell-by-cell "Run Cell" buttons in the `.py` view. No `.ipynb` needed.
- **In JupyterLab**: `uv run poe nb` (once after clone), `uv run jupyter lab`, open the `.ipynb`. Save in Jupyter (Jupytext writes back to `.py` automatically because of the `formats: ipynb,py:percent` header).
- **Never edit the `.py` and the `.ipynb` simultaneously** in different editors — that's the only way to create divergence.

If a notebook ever shows code that doesn't match the `.py` (stale `.ipynb`):

```bash
rm -f relann/*.ipynb examples/*.ipynb .sync_hashes.json
uv run poe nb
```

## Submitting changes

- Branch from `main` (or `juplit` while the migration branch is still active).
- Don't push your branch unless asked. Commit when asked but don't auto-push (`.claude/skills/planning-and-git-policy.md`).
- After grammar / parser changes, also update `.claude/skills/relann-dsl-reference.md`, `.claude/skills/relann-repo-overview.md`, and `.claude/skills/write-relnn-program.md`.
- After test taxonomy or repo-layout changes, also update `TESTING.md` and `tests/README.md`.

## When in doubt

- For DSL syntax: `.claude/skills/relann-dsl-reference.md`
- For repo orientation: `.claude/skills/relann-repo-overview.md`
- For workflow conventions (juplit, imports, debugging): `.claude/skills/relann-conventions.md`
- For commands: `CHEATSHEET.md`
- For the paper / motivation: *Incorporating Deep Learning Design in Database Queries* (VLDB TaDA 2026) — [arXiv:2605.24207](https://arxiv.org/abs/2605.24207).
- Open an issue / discuss with `@yuvallu` for anything else.
