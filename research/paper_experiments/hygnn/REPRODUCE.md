# HyGNN (DDI hypergraph)

RelNN first-order HyGNN vs a PyTorch reference on DrugBank / TWOSIDES-style DDI data, using precomputed hypergraphs from the [HyGNN reference repo](https://github.com/shouhengtuo/HyGNN-Drug-Drug-Interaction-Prediction-via-Hypergraph-Neural-Network) (`load_hygnn_dataset` in `parent.datasets`).

## Prerequisites

- `parent` conda env with `torch`, `torch-scatter`, `sklearn`
- Network (first run downloads `.pt` / `.txt` files into `data/HyGNN/`)

## Paper numbers (TWOSIDES, k-mer $k=9$, MLP decoder)

The experiments section in `main.tex` uses this run (500 epochs, seed 42). Canonical metrics are committed as:

`research/paper_experiments/hygnn/results/hygnn_twosides_kmer_mlp.json`

```powershell
$env:PYTHONUTF8='1'
$py = "python"
& $py research/paper_experiments/hygnn/run_compare_hygnn.py --source TWOSIDES --substructure kmer --decoder mlp
```

On Windows, set `PYTHONUTF8=1` or `PYTHONIOENCODING=utf-8` so the dataset summary prints cleanly (otherwise `cp1252` may error on Unicode box-drawing characters).

## Canonical comparison (full grid)

Default run exercises all combinations: TWOSIDES + DrugBank; k-mer (k=9) + ESPF; MLP + dot decoders.

```powershell
$py = "python"
& $py research/paper_experiments/hygnn/run_compare_hygnn.py
```

## Focused runs

```powershell
& $py research/paper_experiments/hygnn/run_compare_hygnn.py --decoder mlp
& $py research/paper_experiments/hygnn/run_compare_hygnn.py --espf-support 10
```

## Reference PyTorch only

```powershell
& $py research/paper_experiments/hygnn/run_hygnn_pytorch_ref.py --decoder mlp
```

## Demos / DSL

- First-order: `examples/004_relnn_hygnn.py`
- High-order notebook: `tests/slow/test_hygnn_relnn_high_order.py`

## Notes

- **ESPF on TWOSIDES:** use `--espf-support 10` (support 5 files are not published for TWOSIDES in the upstream repo).
- **Weights:** `SessionComparison.sync_weights` aligns PyTorch and RelNN before training so RNG ordering does not skew metrics.
- PR #42 originally lived under `tests/slow/`; scripts were moved here to match other paper experiments.
