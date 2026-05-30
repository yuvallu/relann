# HGT DBLP 5-Run Summary

- Timestamp (UTC): 2026-03-30T23:46:56+00:00
- Device: cuda
- Runs: 5
- Seeds: [42, 43, 44, 45, 46]
- Epochs per run: 100

## Aggregated Results (mean +- std)

| Scope | Implementation | #Params | Train | Val | Test | Time (s) |
|---|---|---:|---:|---:|---:|---:|
| FULL_GRAPH_1L | original_pyHGT | 387604 | 100.0% +- 0.0% | 77.8% +- 0.6% | 79.5% +- 0.5% | 37.6 +- 1.0 |
| FULL_GRAPH_1L | pyg_hgtconv | 387092 | 100.0% +- 0.0% | 75.1% +- 1.8% | 76.6% +- 2.4% | 11.2 +- 0.1 |
| FULL_GRAPH_1L | relnn | 313287 | 100.0% +- 0.0% | 76.3% +- 1.0% | 78.0% +- 0.5% | 26.8 +- 1.1 |
| FULL_GRAPH_2L | original_pyHGT | 479268 | 100.0% +- 0.0% | 77.6% +- 0.8% | 79.4% +- 0.5% | 71.1 +- 0.8 |
| FULL_GRAPH_2L | pyg_hgtconv | 478244 | 99.8% +- 0.4% | 66.8% +- 2.3% | 69.9% +- 2.5% | 21.1 +- 0.1 |
| FULL_GRAPH_2L | relnn | 382993 | 100.0% +- 0.0% | 72.2% +- 1.3% | 74.9% +- 1.2% | 80.1 +- 0.3 |

## Notes

- All timed blocks are measured with `torch.cuda.synchronize()` before and after timing when CUDA is available.
- Full-graph and PA-path rows are different compute scopes; compare runtimes within scope.
