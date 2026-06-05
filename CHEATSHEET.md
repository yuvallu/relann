# RelaNN Cheatsheet

> **TL;DR**: install with `uv`, edit `.py` files, run via `poe`. The `.ipynb` files are throwaway.

```bash
uv sync                                   # install env (.venv/)
uv pip install --no-build-isolation \
    torch-scatter torch-sparse torch-cluster torch-geometric \
    -f "https://data.pyg.org/whl/torch-2.6.0+cu124.html"     # CUDA — change cu124->cpu for laptop
uv pip install scikit-learn relbench scipy nbdev stringdale matplotlib  # demo + transitive deps
uv run poe init                           # install pre-commit hooks
uv run poe nb                             # generate paired .ipynb files
```

That's the one-time clone-to-running pipeline.

### Optional: external pyHGT clone (for HGT comparison scripts)

The HGT comparison scripts under `tests/slow/run_compare_dblp_hgt*.py` and
`tests/slow/run_match_hgt_*.py` import upstream pyHGT for side-by-side
parity checks. It is **not** in `pyproject.toml` because it's not on PyPI
and is only needed by the slow comparison scripts:

```bash
git clone https://github.com/acbull/pyHGT.git _external/pyHGT
```

Without this clone you'll see `ModuleNotFoundError: No module named 'pyHGT'`
on 4 of the 8 slow HGT scripts. The pytest suite does NOT need it.

---

## Daily commands

### Tests

| Command | What it does | Time |
|---|---|---|
| `uv run poe smoke` | fast sanity (`tests/smoke`) | ~5s |
| `uv run poe quick` | smoke + feature | ~15s |
| `uv run poe test` | everything in `tests/` | ~6 min |
| `uv run pytest tests/dhn -v` | DHN bucket only | ~25s |
| `uv run pytest tests/<bucket> -k <pattern>` | filter by name | varies |
| `uv run pytest -m "not slow" tests/` | skip slow bucket | ~1 min |
| `uv run python scripts/run_tests.py <profile>` | alt runner: `smoke`/`quick`/`hgt`/`dhn`/`full` | varies |

### Notebooks (juplit / jupytext)

| Command | What it does |
|---|---|
| `uv run poe nb` | generate `.ipynb` from `.py` (after clone / after bulk `.py` edits) |
| `uv run poe sync` | sync `.py` ↔ `.ipynb` (use after editing either side) |
| `uv run poe clean` | sync, then delete all `.ipynb` (prep before AI agent sessions) |
| `uv run jupyter lab` | launch JupyterLab from the uv venv |

**Golden rule**: edit the `.py`, not the `.ipynb`. If you must edit in Jupyter, save in Jupyter (Jupytext auto-syncs), then close the notebook before doing anything else.

If a notebook shows old code that doesn't match the `.py`:
```bash
rm -f relann/*.ipynb examples/*.ipynb .sync_hashes.json
uv run poe nb
```

### Imports

```python
import relann                                       # version, package init
from relann.session import Session                  # high-level API
from relann.engine import Engine                    # core engine
from relann.parser import get_relnn_grammar_parser  # parse DSL strings
from relann.log_utils import checkLogs              # temporary debug logging
```

### DSL quick reference

```relann
in_channels  = 1433 .
hidden       = 16 .
out_channels = 7 .

PapersEmb1(pid; Linear(in_channels, hidden, False)(z)) :- Papers(pid; z) .
Agg1(target; sum(z * w)) :- PapersEmb1(src; z), Citation(src, target; w) .
?fit  <epochs=200, lr=0.01> Loss(; CrossEntropyLoss()(p, y)) :- Pred(id; p), Labels(id; y) .
?pred Output(id; ArgMax()(z))                              :- Pred(id; z) .
```

- `,` = join · `|` = union · `;` separates content attrs from embedding
- Templates use `<>`; example: `Linear<16,32>` (template_args), `Linear(16, 32)` (hyper_params)
- See `.claude/skills/relann-dsl-reference.md` for the full reference.

### Debugging

```python
from relann.log_utils import checkLogs
with checkLogs(name='relann.engine'):       # or 'relann' for all modules
    session.run(my_program)                  # debug logs from that namespace will print
```

---

## Working with Jupyter

### Selecting a kernel

In VS Code / JupyterLab, pick **`.venv (Python 3.13.x)`** (the uv-managed venv) — usually marked "Recommended". Do NOT pick any old conda env (`parent`, `base`, etc.) — those are stale.

Sanity-check inside the notebook:
```python
import sys, relann
print(sys.executable)        # should end in   parent\.venv\Scripts\python.exe
print(relann.__version__)    # should print    0.1.0
```

### Running a source-module notebook (`relann/engine.ipynb` etc.)

- "Run All" will hit ordering bugs in a few demo cells (engine.py line ~287, ~829; parser.py line ~2223). This is **not new** — it predates the migration.
- Run cells selectively: skip the inline demos that reference methods defined far below in the same file.
- Or open the file in VS Code with the Python + Jupyter extensions — you get cell-by-cell "Run Cell" buttons in the `.py` view directly, no `.ipynb` needed.

### Running an example notebook (`examples/001_relnn_hello_world.ipynb` etc.)

Examples are designed to "Run All Cells" end-to-end. They will:
- Download datasets to `data/` on first run (Cora, Planetoid).
- Some need `scikit-learn`, `relbench`, `matplotlib` — install per the top-of-file.

---

## Repo layout (one-line per dir)

```
relann/                # the Python package (paired .py notebooks)
tests/                 # pytest files in buckets: smoke/feature/dhn/slow/repro/scaffold
examples/              # user-facing demos (paired .py notebooks)
research/              # paper reproducibility artefacts (not part of the package)
docs/                  # design docs, references; _archive/ for historical
.claude/skills/        # Claude Code skills the agent invokes automatically
scripts/               # run_tests.py + setup_external_dhn.py
data/                  # datasets (gitignored)
pyproject.toml         # build / deps / poe tasks / pytest config — single source of truth
uv.lock                # exact resolved versions — committed
```

---

## Git

- **Never push** without explicit ask.
- Commit only when asked.
- `.ipynb`, `.venv/`, `data/`, `.cursor/`, `_external/` are gitignored.

---

## When something breaks

- `import relann` errors → run `uv sync`, install PyG sparse stack (see top of this file).
- Notebook shows old code → `rm -f relann/*.ipynb examples/*.ipynb .sync_hashes.json && uv run poe nb`.
- Test fails on Windows with cp1252 encoding error → already fixed (`relann/__init__.py` reconfigures stdout); make sure your venv is current.
- `FileNotFoundError: Could not find project root` → already fixed (`get_project_root()` now looks for `pyproject.toml`); update your branch.
- Slow tests (`tests/slow/run_compare_dblp_hgt.py` etc.) fail with `ModuleNotFoundError: pyHGT` → these need the external `pyHGT` repo cloned into `_external/`; not required for normal dev.
- `tests/feature/test_904_self_join_h0_term_graph.py` fails to collect → has a transitive nbdev `settings.ini` lookup quirk via stringdale; ignore for now or run with `--ignore=tests/feature/test_904_self_join_h0_term_graph.py`.

---

## Skills available (Claude Code)

The repo ships skills under `.claude/skills/`. They're invoked automatically when relevant. The ones you'll see most:

- `relann-repo-overview` — orient yourself in the codebase
- `relann-conventions` — code style, debug, import rules
- `relann-dsl-reference` — full DSL grammar
- `write-relnn-program` — how to author a new RelaNN program
- `juplit-programming` — the .py/.ipynb workflow
- `planning-and-git-policy` — ask before building; never push
- `design-and-testing-doc-discipline` — keep docs aligned

---

## More

- README: `README.md`
- Testing reference: `TESTING.md`, `tests/README.md`
- AI agent guide: `AGENTS.md` (vendor-neutral), `CLAUDE.md` (Claude-specific, gitignored)
- GPU / PyG / RAPIDS install: `docs/install-gpu.md`
- Paper: *Incorporating Deep Learning Design in Database Queries* (VLDB TaDA 2026) — [arXiv:2605.24207](https://arxiv.org/abs/2605.24207).
