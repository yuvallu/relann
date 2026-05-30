# RelBench official `gnn_node.py` baseline (5 epochs, CPU)

Source: `_external/relbench/examples/gnn_node.py` (tag `v1.1.0`)

Environment notes:
- Installed `pyg-lib` in `parent` conda env to satisfy `NeighborSampler` backend.
- Command used Python from `python`.

## Commands

```powershell
& $py examples/gnn_node.py --dataset rel-f1 --task driver-position --epochs 5 --num_workers 0 --seed 42
& $py examples/gnn_node.py --dataset rel-f1 --task driver-dnf --epochs 5 --num_workers 0 --seed 42
```

## Best test metrics

- `driver-position`: `r2=0.0042`, `mae=4.2444`, `rmse=5.1994`
- `driver-dnf`: `average_precision=0.8205`, `accuracy=0.7051`, `f1=0.8271`, `roc_auc=0.7045`

These are short-run baseline numbers (5 epochs), intended for directional comparison with RelNN runs.

## 30-epoch run (executed in-repo, seed=42)

From `_external/relbench` root, after `pyg-lib` is available:

```powershell
$py = "python"
& $py examples/gnn_node.py --dataset rel-f1 --task driver-position --epochs 30 --num_workers 0 --seed 42
& $py examples/gnn_node.py --dataset rel-f1 --task driver-dnf --epochs 30 --num_workers 0 --seed 42
```

Best metrics observed from the 30-epoch runs:

- `driver-position`:
  - best val: `r2=0.3056`, `mae=3.1006`, `rmse=3.8633`
  - best test: `r2=0.0836`, `mae=4.0943`, `rmse=4.9878`
- `driver-dnf`:
  - best val: `average_precision=0.9235`, `accuracy=0.7845`, `f1=0.8674`, `roc_auc=0.7686`
  - best test: `average_precision=0.8507`, `accuracy=0.7279`, `f1=0.8200`, `roc_auc=0.7280`

Interpretation: at 30 epochs, the official baseline remains stronger on
`driver-dnf` than RelNN (~0.728 vs ~0.612 AUROC), while RelNN still wins on
`driver-position` MAE (3.99 vs 4.09).
