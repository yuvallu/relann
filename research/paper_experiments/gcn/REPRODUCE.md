# GCN on Cora — Reproduction Guide

## Paper table rows populated

| Table | Row |
|-------|-----|
| `tab:unified-results` | GCN / Cora — RelNN and PyG columns |
| `tab:loc-comparison` | GCN row — RelNN LOC and PyG LOC |

## Prerequisites

- Python environment with `parent`, `torch`, `torch_geometric` installed
- CPU is sufficient (Cora is small)

## How to run

```powershell
$py = "python"
& $py research/paper_experiments/gcn/run_compare_cora_pytorch.py
```

Runtime: ~2 minutes on CPU.

## What it does

The script (`run_compare_cora_pytorch.py`) uses `SessionComparison` to run:
1. **RelNN GCN** — templated DSL rules at three complexity levels
2. **PyG GCNConv** — standard PyTorch Geometric implementation

For each, it trains on Cora's train/val/test split and reports test accuracy.

## Expected output

- RelNN and PyG GCN achieve comparable test accuracy on Cora (~80-82%)
- Timing comparison printed to stdout

## LOC counting

| Implementation | What is counted | LOC |
|----------------|-----------------|-----|
| RelNN | define + fit + predict rules in the script | 21 |
| PyG GCNConv | Full `torch_geometric.nn.GCNConv` source | 217 |

RelNN LOC = the DSL program only (framework is shared infrastructure).
PyG LOC = the full library implementation of `GCNConv`, reflecting the true
complexity a user must understand to modify the architecture.

## Source script

The canonical source is `tests/slow/run_compare_cora_pytorch.py`.
The copy here is for convenience; keep them in sync.
