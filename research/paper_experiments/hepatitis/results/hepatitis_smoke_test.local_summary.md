# CTU Hepatitis multi-run (canonical paper results)

- UTC: 2026-04-03T22:47:10+00:00
- Device: cpu
- epochs=5, runs=1, seed_start=100
- hidden=32, lr=0.01, wd=0.0001

## Results

- val_roc_auc: 0.860115 ± 0.000000
- test_roc_auc: 0.899573 ± 0.000000

## Paper table values

| Task | Metric | RelNN | ReDeLEx GNN baseline |
|---|---|---|---|
| Hepatitis (HBV vs HCV) | AUC ROC ↑ | **0.900 ± 0.000** | 1.000 (ResNet SAGE) |

Baselines from ReDeLEx Table 1 (arXiv:2506.22199):
  LightGBM test AUC: 0.626
  GraphSAGE (Linear/ResNet) test AUC: 0.997 / 1.000
  DBFormer test AUC: 0.996
