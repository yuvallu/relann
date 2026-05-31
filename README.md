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

relann is **GPU-first** (PyTorch + optional cuDF/RAPIDS). The steps below install the
CUDA build; a CPU-only fallback is noted underneath.

```bash
# 1. Install the CUDA build of PyTorch FIRST. On Windows the default PyPI torch is
#    CPU-only, so pull it from PyTorch's CUDA index. Swap cu124 for the tag matching
#    your NVIDIA driver (cu118 / cu121 / cu124 / cu126 / ...).
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. Install relann (torch is already satisfied, so pip won't pull a CPU wheel).
pip install relann                       # core library
pip install "relann[examples]"           # + relbench, scikit-learn, matplotlib (to run the examples)
#   extras: [examples] [viz] [benchmarks] [sql] — combine like "relann[examples,viz]"

# 3. Install the PyG sparse stack (torch-scatter & friends) used by relann's
#    scatter/aggregation operators. These are prebuilt wheels matched to your torch
#    version + CUDA tag, so they are NOT installed by `pip install relann`.
pip install --no-build-isolation torch-scatter torch-sparse torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

> **`import relann` works without step 3** — only the scatter aggregation operators
> require the PyG stack, and they raise a clear install hint if it's missing.
>
> **CPU-only fallback:** use `--index-url https://download.pytorch.org/whl/cpu` in step 1
> and `+cpu` in step 3.
>
> **Why is step 3 separate?** `torch-scatter` / `torch-sparse` / `torch-cluster` are
> compiled C++/CUDA extensions that must match your exact torch + CUDA build, so pip
> can't resolve them from PyPI — they ship as prebuilt wheels on the PyG index. For full
> **GPU / cuDF / RAPIDS** setup, see
> [docs/install-gpu.md](https://github.com/yuvallu/relann/blob/main/docs/install-gpu.md).

## Develop from source

Working on relann itself uses [uv](https://astral.sh) and the juplit notebook workflow:

```bash
pip install uv      # or: brew install uv (macOS) · curl -LsSf https://astral.sh/uv/install.sh | sh (Linux)

git clone https://github.com/yuvallu/relann.git && cd relann
uv run poe full-setup    # uv sync + PyG sparse stack (CPU by default) + generate notebooks
uv run poe init          # install git hooks
```

For a CUDA dev box, set `TORCH_PYG_URL` before `full-setup` (see
[docs/install-gpu.md](https://github.com/yuvallu/relann/blob/main/docs/install-gpu.md)).

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
