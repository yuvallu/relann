# relann

> **RelaNN** — A declarative query language for neural networks over relational databases.

RelaNN lets you express deep neural networks directly over relational data using a Datalog-style query language with embedding semantics. Each tuple carries a learnable vector embedding; joins compose embeddings; group-by projections aggregate them. Programs are compiled to PyTorch + cuDF + SQL physical plans.

This codebase is the open-source proof-of-concept implementation accompanying the paper *"Incorporating Deep Learning Design in Database Queries"* (VLDB TaDA 2026).

## Why RelaNN

- **40 % of the world's data lives in relational databases** — RelaNN keeps the model definition *inside* the relational paradigm rather than round-tripping through external graph libraries.
- **Declarative syntax similar to SQL/Datalog** — define the architecture, not the tensor plumbing.
- **No graph-conversion boilerplate** — joins, aggregations and transformations have well-defined embedding semantics built into the language.
- **GPU acceleration via cuDF**, with pandas fallback on CPU.
- **Mirrors the math of the original papers** — implementations of GCN, R-GCN, HGT, HyGNN and DHN are 3–10× shorter than their PyTorch/PyG references.

## Example

The two rules below implement query-key-value attention as used by Heterogeneous Graph Transformers over a `Patients × Treatments` schema:

```relann
Score(p, t; q*k)         :- Treat(p, t), Queries(p; q), Keys(t; k) .
Attention(p; sum(a*v))   :- Score(p, t; a), Values(t; v) .
```

The first rule joins `Treat`, `Queries`, `Keys` on `p` and `t`, composing the embeddings as `q*k`. The second joins `Score` with `Values`, weights each value by its attention score, and aggregates with `sum` after projecting away `t`.

## Install

```bash
# 1. Install uv (https://astral.sh) if you don't have it
pip install uv                       # or: brew install uv (macOS), curl -LsSf https://astral.sh/uv/install.sh | sh (Linux)

# 2. Clone & sync the environment
git clone https://github.com/yuvallu/relann.git && cd relann
uv sync                              # creates .venv with the base deps

# 3. Install PyG sparse extensions (required at module import time)
#    Substitute your torch + cuda/cpu tag — see docs/install-gpu.md for details.
uv pip install --no-build-isolation \
    torch-scatter torch-sparse torch-cluster torch-geometric \
    -f "https://data.pyg.org/whl/torch-2.6.0+cu124.html"     # CUDA 12.4 host
#    Replace cu124 with cpu for a CPU-only laptop install.

# 4. (Optional) install heavier test-time deps that some tests need
uv pip install scipy nbdev stringdale                         # transitive for some demo/scaffold tests

# 5. Install git hooks + generate Jupyter notebooks from .py files
uv run poe init
uv run poe nb
```

For **GPU / cuDF / RAPIDS** support, see `docs/install-gpu.md`. The base `uv sync` plus the PyG step above gives you everything needed to run the smoke + feature test suites.

## Workflow

This project uses [juplit](https://github.com/DeanLight/juplit) for literate programming: every `.py` file in `relann/` and `examples/` is paired with a `.ipynb` notebook via [jupytext](https://jupytext.readthedocs.io/). The `.py` is the source of truth; `.ipynb` is generated on demand and gitignored.

```bash
uv run poe sync         # sync .py ↔ .ipynb after edits
uv run poe nb           # generate .ipynb from .py (run after cloning)
uv run poe clean        # sync then delete all .ipynb files (clean for AI agents)
uv run poe smoke        # fastest sanity check (~5s)
uv run poe quick        # smoke + feature (~15s)
uv run poe test         # full pytest sweep
```

## Repository layout

```
relann/                # Python package — paired juplit .py notebooks (jupytext header + if test():)
├── parser.py, engine.py, session.py, term_graph.py, era_operations.py, …
├── utils/
└── relnn_grammar.lark # Lark DSL grammar

tests/                 # standalone pytest files
├── smoke/  feature/  dhn/  slow/  repro/  scaffold/

examples/              # user-facing demos (juplit .py)
research/
├── paper_experiments/ # reproducibility artefacts for the paper
└── _drafts/           # in-progress research notebooks

docs/                  # design notes, architecture, historical reference
.claude/skills/        # Claude-Code skills (juplit-programming, write-relnn-program)
scripts/run_tests.py   # convenience wrapper around pytest profiles
```

## Tests

After the install steps above, all of these should pass on a CPU-only host:

```bash
uv run poe smoke                                   # ~5s   — 34 tests
uv run poe quick                                   # ~15s  — smoke + 274 feature tests
uv run pytest tests/repro                          # ~5s
uv run pytest tests/dhn -v                         # ~25s  — 51 tests
uv run pytest tests/                               # full sweep
```

Or via the runner script:

```bash
uv run python scripts/run_tests.py smoke           # ~5s
uv run python scripts/run_tests.py quick           # ~15s
uv run python scripts/run_tests.py hgt             # HGT slow scripts
uv run python scripts/run_tests.py dhn             # ~60s
uv run python scripts/run_tests.py full            # ~6min
```

**Test collection** is scoped by `pyproject.toml` to `testpaths = ["tests"]` and `python_files = ["test_*.py"]`. Source modules under `relann/*.py` keep `if __name__ == "__main__":` demo cells that run only when opened interactively in Jupyter — they aren't picked up at pytest collection time.

## Paper & citation

- Paper: *Incorporating Deep Learning Design in Database Queries.* Yuval Lev Lubarsky, Dean Light, Boaz Berger, Shunit Agmon, Benny Kimelfeld. VLDB TaDA 2026. [arXiv:2605.24207](https://arxiv.org/abs/2605.24207)
- Source: https://github.com/yuvallu/relann

## License

Apache-2.0. See `LICENSE`.
