# RelBench rel-f1 multi-run (canonical paper results)

- Canonical artifact: relbench_f1_multirun_200ep_5seed.json
- Device: cpu
- uniform_hparams=False, extra_tables=False
- epochs=200, runs=5, seed_start=100

## Tuned hyperparameters

- driver-position: hidden=32, lr=0.01, wd=0.0001
- driver-dnf: hidden=16, lr=0.003, wd=0.0

## driver-position

- r2: 0.153781 ± 0.005688
- mae: 3.991875 ± 0.018817
- rmse: 4.793021 ± 0.016118

## driver-dnf

- average_precision: 0.806479 ± 0.004752
- accuracy: 0.722222 ± 0.009556
- f1: 0.823694 ± 0.003937
- roc_auc: 0.612231 ± 0.009395

## Paper table values

| Task | Metric | RelNN | Official GNN baseline (5 ep) |
|---|---|---|---|
| driver-position | MAE↓ | **3.99 ± 0.02** | 4.24 |
| driver-dnf | AUROC↑ | 0.61 ± 0.01 | **0.70** |

RelNN beats the baseline on driver-position. Gap remains on driver-dnf (see PROBLEMS_WE_DIDNT_SOLVE.md).
