# Installing GPU / PyG / RAPIDS dependencies

The base `uv sync` gives you a CPU-only environment with `torch`, `pandas`, `numpy`, `pydantic`, `networkx`, `lark`, `juplit`, and `fastcore`. That's enough to import most of `relann` *except* the operators that depend on `torch-scatter` — namely `relann.era_operations` and anything that pulls it in transitively (including `relann.relnn`, `relann.engine`, `relann.session`, the top-level `import relann`).

To run tests you need to install the **PyG sparse extension stack** matched to your torch + CUDA combo, plus optionally **RAPIDS** for GPU-accelerated dataframes.

## 1. After `uv sync`, install the PyG sparse stack

```bash
# Discover what torch you got
TORCH=$(uv run python -c "import torch; print(torch.__version__.split('+')[0])")
# e.g. TORCH=2.6.0

# CPU build (Linux/macOS):
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    --no-build-isolation \
    -f "https://data.pyg.org/whl/torch-${TORCH}+cpu.html"

# CUDA 12.4 build (Linux):
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    --no-build-isolation \
    -f "https://data.pyg.org/whl/torch-${TORCH}+cu124.html"
```

`--no-build-isolation` is required because PyG sparse extensions import `torch` at *build* time but don't declare it in their `build-system.requires`.

## 2. (Optional) RAPIDS / cuGraph / cuML / cuDF for GPU dataframes

RAPIDS doesn't have PyPI wheels — it lives on conda-forge and the rapidsai channel. Two paths:

### Option A: separate conda env for RAPIDS, uv for everything else
Keep `.venv/` from `uv sync` for the relann package. Run cuDF-heavy scripts inside a sibling conda env that has `rapids=25.06` installed:

```bash
conda create -n relann-gpu -c rapidsai -c conda-forge -c nvidia \
    rapids=25.06 python=3.12 cuda-version=12.4
conda activate relann-gpu
pip install -e .                  # editable install of relann
```

### Option B: install RAPIDS into your uv venv
RAPIDS provides `pylibcudf` wheels via the rapidsai PyPI index (preview). Check the [official RAPIDS PyPI guide](https://docs.rapids.ai/install/) for the current command — it's something like:

```bash
uv pip install --extra-index-url https://pypi.nvidia.com cudf-cu12 cugraph-cu12 cuml-cu12
```

## 3. Verify

```bash
uv run python -c "import relann; print(relann.__version__)"
uv run poe smoke      # ~5s
uv run poe quick      # ~15s
```

## 4. Versions the codebase has been tested with

From the original conda lockfile, the project worked with:

- `torch==2.5.0+cu124`
- `torch-scatter==2.1.2+pt25cu124`
- `torch-geometric==2.7.0`
- `torch-sparse==0.6.18+pt25cu124`
- `torch-cluster==1.6.3+pt25cu124`
- `pyg-lib==0.4.0+pt25cu124`
- `rapids==25.06.00 / cuda12_py310`

Newer combinations should work but pin to those if you hit a regression.
