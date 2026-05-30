# HGT on DBLP — Reproduction Guide

## Paper table rows populated

| Table | Row |
|-------|-----|
| `tab:unified-results` | HGT 1L / DBLP — pyHGT, PyG, RelNN columns |
| `tab:unified-results` | HGT 2L / DBLP — pyHGT, PyG, RelNN columns |
| `tab:loc-comparison` | HGT 1L and HGT 2L rows |

## Prerequisites

- CUDA GPU (timing comparisons are GPU-based)
- `_external/pyHGT` cloned (original paper's DGL-based HGT)
- DBLP dataset (auto-downloaded on first run via `torch_geometric.datasets`)
- Python environment with `parent`, `torch`, `torch_geometric`, `dgl` installed

## Scripts

| Script | Purpose | Runtime |
|--------|---------|---------|
| `run_compare_dblp_hgt_multirun.py` | Main 5-seed benchmark (accuracy + GPU timing) | ~18 min |
| `run_compare_dblp_original_hgt.py` | Per-seed implementations (pyHGT, PyG, RelNN) | Called by multirun |
| `run_match_hgt_accuracy.py` | 1L weight-sync parity check | ~5 min |
| `run_match_hgt_2l_accuracy.py` | 2L weight-sync parity check | ~10 min |

## How to run

### Step 1: Main timing + accuracy table

```powershell
$py = "python"
& $py -u research/paper_experiments/hgt/run_compare_dblp_hgt_multirun.py --runs 5
```

**Outputs:**
- `research/paper_experiments/hgt/results/hgt_dblp_5run_results.json` — raw per-run data
- `research/paper_experiments/hgt/results/hgt_dblp_5run_summary.md` — mean ± std table

### Step 2: Parity checks (optional, for footnotes)

```powershell
# 1L: proves RelNN matches PyG when weights are synchronized
& $py -u research/paper_experiments/hgt/run_match_hgt_accuracy.py --runs 5 --epochs 100

# 2L: proves RelNN matches PyG when weights are synchronized
& $py -u research/paper_experiments/hgt/run_match_hgt_2l_accuracy.py --runs 5 --epochs 100
```

These verify that the RelNN DSL implements the same architecture as PyG HGTConv
by showing machine-precision identical forward passes with synchronized weights.

## Timing protocol

All timing uses `torch.cuda.synchronize()` before and after each run.
Data is pre-loaded to GPU at module scope — no per-epoch CPU→GPU transfers.
Each "run" includes model construction + training + prediction.

## Expected results

### 1-Layer (weight-synced)
- PyG and RelNN produce identical test accuracy (~78.5%)
- pyHGT differs architecturally (no weight sync possible)

### 2-Layer
- PyG and RelNN produce identical forward passes (machine precision) when weights synced
- Training accuracy may differ across seeds due to PyG instability on certain seeds

## LOC counting

| Implementation | What is counted | LOC |
|----------------|-----------------|-----|
| RelNN 1L | `RELNN_DEFINE_DSL` in `run_compare_dblp_original_hgt.py` | 41 |
| RelNN 2L | `RELNN_DEFINE_DSL_2L` in `run_compare_dblp_original_hgt.py` | 117 |
| PyG HGTConv | Full `torch_geometric.nn.HGTConv` source | 217 |
| pyHGT | `pyHGT/conv.py` + model wrapper | 143 |

## Source scripts

The canonical sources are in `tests/slow/`. There are no script copies
in `research/paper_experiments/hgt/`; run the scripts directly from `tests/slow/`.

## Correspondence figure

`hgt_equations_vs_relnn.pdf` in this directory is the paper's equation-to-rule
correspondence figure: the HGT paper's LaTeX equations side by side with the
RelNN DSL that transcribes them. It is supplementary material for the paper
(the target venue does not allow an appendix).
