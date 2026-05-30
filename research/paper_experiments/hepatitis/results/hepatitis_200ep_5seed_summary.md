# CTU Hepatitis multi-run (canonical paper results)

- UTC: 2026-04-03T22:51:15+00:00
- Device: cpu
- epochs=200, runs=5, seed_start=100
- hidden=64, lr=0.02, wd=0.0

## Results

- val_roc_auc: 0.888522 ± 0.004720
- test_roc_auc: 0.876068 ± 0.006508

## Paper table values

| Task | Metric | RelNN | ReDeLEx GNN baseline |
|---|---|---|---|
| Hepatitis (HBV vs HCV) | AUC ROC ↑ | **0.876 ± 0.007** | 1.000 (ResNet SAGE) |

Baselines from ReDeLEx Table 1 (arXiv:2506.22199):
  LightGBM test AUC: 0.626
  GraphSAGE (Linear/ResNet) test AUC: 0.997 / 1.000
  DBFormer test AUC: 0.996
