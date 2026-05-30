# R-GCN on AIFB (Entities)

**MUTAG status:** MUTAG results are archived in `results/rgcn_entities_results.mutag_50ep.json` but are **not included in the paper** (severe overfitting: 55.3% ± 12.3%). Only AIFB is in the paper tables.

## Paper mapping

| Table | Row |
|-------|-----|
| `tab:exp-roadmap` | R-GCN / Entities AIFB |
| `tab:unified-results` | Test accuracy + time (PyG `FastRGCNConv` vs RelNN per-relation lookup) |
| `tab:loc-comparison` | RelNN 22 LOC vs PyG 186 library LOC (8.5× ratio) |

## Prerequisites

- `torch`, `torch_geometric`, `rdflib` (PyG `Entities` loader)
- CPU or CUDA

## Command (canonical — AIFB only, 5 seeds)

```powershell
$py = "python"
& $py research/paper_experiments/rgcn/run_compare_entities_rgcn.py --datasets AIFB --runs 5 --seed-start 42 --cpu-only --num-threads 1 --out-json research/paper_experiments/rgcn/results/rgcn_entities_results.aifb_5run.json --out-md research/paper_experiments/rgcn/results/rgcn_entities_summary.aifb_5run.md
```

> `--out-md` is passed explicitly so re-runs overwrite the stable tracked summary path. If omitted, the script writes to `<json_stem>_summary.md` in the same directory.

## Canonical outputs

- `results/rgcn_entities_results.aifb_5run.json` — canonical 5-seed AIFB paper results
- `results/rgcn_entities_results.mutag_50ep.json` — archived MUTAG results (NOT in paper)

## Protocol notes

- **torch-rgcn baseline**: cloned under `_external/torch-rgcn`, `NodeClassifier` path (Schlichtkrull reproduction code path).
- **PyG baseline**: `FastRGCNConv`, `num_bases=30`, Adam `lr=0.01`, `weight_decay=5e-4`, 50 epochs (same family as PyG `examples/rgcn.py`).
- **RelNN**: template-based DSL with bounded-set expansion over `MetaRel(ts, pe, tt)`, per-relation operators, and root transform. **AIFB** uses per-relation `NodeLookup<pe>` embedding tables (mathematically identical to featureless R-GCN's `W_r @ one_hot(s)` but expressed as a gather; same param count, no 8285×8285 identity materialized). **MUTAG** uses a shared `NodeLookup` plus **basis decomposition in layers 1 and 2** (RelBench-sized graphs cannot materialize a full one-hot).
- **Memory safety**: run with `--cpu-only --num-threads 1` on large graphs.

## Timing

RelNN `time_s` is end-to-end for `Session.run` (compile/instantiate + training epochs). PyG and torch-rgcn times measure the training loop only. Do not compare raw seconds as a fair “framework speed” benchmark without separating compile from train.

## LOC

Non-blank, non-comment DSL lines are counted by `_count_dsl_loc()` in `run_compare_entities_rgcn.py` (rules only, excluding `#` lines).
