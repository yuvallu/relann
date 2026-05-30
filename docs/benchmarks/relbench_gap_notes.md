# RelBench integration: framework gaps (RelNN)

This log records what required **Python glue** outside the RelNN DSL for the rel-f1 paper experiments, and what would ideally become **native** RelNN capabilities for “eat the database” workflows.

## What stayed in Python

| Concern | Current approach |
|---------|------------------|
| **Temporal / task splits** | RelBench `task` objects provide train/val/test tables; we only materialized train labels into `TrainLabels` and evaluated on `test_table` via `task.evaluate` in the driver script. |
| **Mixed-type feature encoding** | `parent.datasets.load_relbench_f1_dataset` encodes drivers, constructors, and results into numeric tensors before RelNN sees them. |
| **Official metrics** | `task.evaluate(pred, test_table)` runs RelBench’s metric code—not reimplemented in DSL. |
| **Orchestration** | Multi-seed loops, JSON/markdown artifacts, and tuning grids are plain Python scripts under `nbs/paper_experiments/relbench/`. |

## What would be “native” later

1. **Declarative ingestion**: Auto-map SQL/Pandas schemas to ER schemas with typed columns and optional learned encoders (categorical → embedding, numeric → normalization) declared in DSL or a thin config layer.
2. **Task objects as first-class outputs**: Attach evaluation hooks (`evaluate(pred)`) to named `?pred` rules without hand-writing NumPy glue in scripts.
3. **Mini-batch / neighbor sampling**: For large graphs, integrate dataloader abstractions that still compile to the same relational term graph (today’s `fit` loop is full-batch for these experiments).
4. **Integer vs float embedding columns**: Uniform handling of index-like columns in transformations (see R-GCN experiment notes: packing indices into float tensors or small wrapper modules was needed where a single ER has both discrete indices and continuous weights).

## Cross-links

- RelBench experiment README: `nbs/paper_experiments/relbench/REPRODUCE.md`
- Dataset loader: `parent.datasets.load_relbench_f1_dataset`
- Demo notebooks: `nbs/demos/005_relnn_relbench_f1.ipynb`, `nbs/demos/006_relnn_relbench_f1_dnf.ipynb`
