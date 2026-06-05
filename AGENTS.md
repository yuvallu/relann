# AGENTS.md

Conventions for AI coding agents (Cursor, Claude Code, Codex, etc.) working in this repo.

## Project at a glance

**RelaNN** — declarative query language for neural networks over relational databases. See `README.md` for the high-level pitch, `.claude/skills/relann-repo-overview.md` for repo orientation, and `.claude/skills/relann-dsl-reference.md` for DSL syntax.

The Python package is named `relann` (lowercase, with an 'a'). The paper rebrands it as `RelaNN` in prose.

## Environment

The project uses **uv** (Astral) — no conda, no nbdev. Python ≥3.12 in a uv-managed venv.

```bash
uv sync                          # create .venv + install all deps
uv run poe init                  # install git hooks
uv run poe nb                    # generate .ipynb from juplit .py files (once after cloning)
uv run python -c "import relann; print(relann.__version__)"
```

For GPU / cuDF / RAPIDS, install the heavy CUDA stack separately on a GPU host. The base `uv sync` is CPU-only and sufficient for tests and most demos.

## Running tests

```bash
uv run poe smoke                 # ~5s   smoke bucket
uv run poe quick                 # ~15s  smoke + feature
uv run pytest tests/dhn -v       # ~60s  DHN unit tests
uv run pytest tests/             # full sweep
uv run python scripts/run_tests.py smoke   # alternative runner
```

All tests live under `tests/` (root). pytest's `testpaths = ["tests"]` and `python_files = ["test_*.py"]` mean source modules under `relann/` aren't scanned at collection time — they keep their `if __name__ == "__main__":` demo cells but those don't appear in pytest.

## Editing source

Source modules under `relann/` and demos under `examples/` are **juplit-paired** `.py` files: each one has a jupytext header pairing it with a (gitignored) `.ipynb`. Edit only the `.py`:

```python
# %%                              # code cell
def my_thing(): ...

# %%
if test():                        # inline test, wrapped so it doesn't run on import
    assert my_thing() == ...

# %% [markdown]                   # markdown cell, each line `# `-prefixed
# ## Section
```

After editing run `uv run poe sync` so `.ipynb` stays in sync if you want to open the notebook in Jupyter.

There are NO `#|export` / `#|hide` directives. Everything in a `.py` file is exported by default. The `if test():` block is what separates production code from tests.

## Key gotchas

- **PyG sparse extensions** (`torch-scatter`, `torch-sparse`, etc.) must match the exact PyTorch version. Install from `https://data.pyg.org/whl/torch-<version>+cpu.html`.
- **HGT slow tests** (`tests/slow/run_hgt_fit_*.py`) download the DBLP dataset on first run.
- **Don't ever edit `relann/*.ipynb` directly** — that file is generated. Edit the `.py`.

## Workflow conventions

- Tests must hit a real engine/session, not a mocked one (we've been bitten by mocks diverging from prod). See `.claude/skills/design-and-testing-doc-discipline.md`.
- Never push unless explicitly asked. Create commits only when the user asks.
- After grammar/parser changes, also update:
  - `.claude/skills/relann-dsl-reference.md`
  - `.claude/skills/relann-repo-overview.md`
  - `.claude/skills/write-relnn-program.md`

## Capturing learnings (how this repo stays sharp)

When you correct an agent, find a reusable pattern, or hit the same gotcha twice — capture it so the next session starts knowing it. Route by *kind*:

| What you learned                         | Where it goes                                          |
|------------------------------------------|--------------------------------------------------------|
| A durable fact or preference             | agent memory (Claude: `/remember`)                     |
| A repeatable procedure ("how we do X")   | a skill in `.claude/skills/` (see its README)          |
| An always-on rule for the repo           | `AGENTS.md` (all agents) / `CLAUDE.md` (Claude-only)   |
| A deterministic, checkable rule          | a hook / lint (enforced, not remembered)               |

Prune as you go — stale skills and memory dilute attention; retire notes when the project moves on. Rule of thumb: judgment call → skill; hard rule → hook.

## Skills (deep dive on demand)

Skills live under `.claude/skills/` and are loaded by agents that support skill discovery (Claude Code does this automatically). Read them when you need depth on a particular topic:

| Skill | When to use |
|---|---|
| `relann-repo-overview` | First-time orientation — repo layout, components, execution pipeline |
| `relann-conventions` | Code style, debugging via `checkLogs`, row-first tensors, HGT shape invariants, import rules |
| `relann-dsl-reference` | Authoring or reading RelaNN DSL — full grammar quick-ref |
| `write-relnn-program` | Step-by-step authoring of a new RelaNN program from a target architecture |
| `juplit-programming` | The `.py` ↔ `.ipynb` workflow, `if test():` blocks, poe commands |
| `juplit-migrate-from-nbdev` | Only relevant if you encounter nbdev artefacts in this or another repo |
| `planning-and-git-policy` | Ask before building; never push; commit only when asked |

A `.claude/skills/README.md` lists them with one-line descriptions.

## Quick links

- One-page command cheatsheet: `CHEATSHEET.md`
- Test layout: `tests/README.md`
- Testing strategy: `TESTING.md`
- GPU / PyG / RAPIDS install: `docs/install-gpu.md`
- Paper (motivation): *Incorporating Deep Learning Design in Database Queries* (VLDB TaDA 2026) — [arXiv:2605.24207](https://arxiv.org/abs/2605.24207).
