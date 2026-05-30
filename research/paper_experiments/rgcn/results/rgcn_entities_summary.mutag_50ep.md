# R-GCN Entities (AIFB / MUTAG)

- Timestamp (UTC): 2026-04-01T18:42:38+00:00
- Device: cpu
- Runs per dataset: 5

| Dataset | Impl | Test Acc (mean ± std) | Time (s) (mean ± std) | Params | DSL LOC |
|---|---:|---:|---:|---:|---:|
| MUTAG | torch-rgcn | 70.59% ± 2.94% | 32.75 ± 1.85 | 11355678 | — |
| MUTAG | PyG FastRGCN | 70.88% ± 3.01% | 14.64 ± 2.37 | 11731194 | — |
| MUTAG | RelNN | 55.29% ± 12.29% | 1139.57 ± 61.77 | 389992 | 31 |
