# RelBench rel-f1 multi-run

- UTC: 2026-04-01T14:21:53+00:00
- Device: cpu
- uniform_hparams=False, extra_tables=False
- epochs=200

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

