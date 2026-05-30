# R-GCN Entities (AIFB / MUTAG)

- Timestamp (UTC): 2026-05-09T19:57:52+00:00
- Device: cpu
- Runs per dataset: 5

| Dataset | Impl | Test Acc (mean ± std) | Time (s) (mean ± std) | Params | DSL LOC |
|---|---:|---:|---:|---:|---:|
| AIFB | torch-rgcn | 93.33% ± 4.21% | 25.54 ± 5.92 | 24004964 | — |
| AIFB | PyG FastRGCN | 92.22% ± 1.24% | 8.30 ± 3.05 | 4116764 | — |
| AIFB | RelNN (full, ↔ torch-rgcn) | 92.78% ± 2.48% | 178.53 ± 43.21 | 24004964 | 20 |
