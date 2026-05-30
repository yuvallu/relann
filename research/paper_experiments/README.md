# Paper Experiments

Reproducible experiments for the RelNN paper. Each subdirectory corresponds to
one architecture / benchmark family.

## Structure

```
paper_experiments/
  gcn/           GCN on Cora (Table 2, Table 3)
  hgt/           HGT on DBLP — 1L and 2L (Table 2, Table 3)
  dhn/           DHN on CSL, EXP, SR25, ENZYMES, PROTEINS (Table 2, Table 3)
  rgcn/          R-GCN on AIFB / MUTAG (Entities) — PyG vs RelNN
  relbench/      RelBench rel-f1 (driver-position, driver-dnf, driver-top3)
  hepatitis/     CTU Hepatitis HBV vs HCV binary classification
```

Table 2 = `tab:unified-results` (accuracy + time).
Table 3 = `tab:loc-comparison` (lines of code).

## Canonical RelNN DSL paths

Use these locations as the source of truth when you need to inspect the exact
RelNN program, count LOC, or explain which DSL generated a paper row.

| Experiment | Canonical RelNN DSL location | Form | Notes |
|---|---|---|---|
| GCN (Cora) | `tests/slow/run_compare_cora_pytorch.py` (`GCN_DSL`) | inline string | Used for GCN paper rows and LOC counting |
| HGT 1L / 2L (DBLP) | `tests/slow/run_compare_dblp_original_hgt.py` (`RELNN_DEFINE_DSL`, `RELNN_DEFINE_DSL_2L`) | inline strings | Used for main HGT accuracy/time/LOC rows |
| DHN | `tests/dhn/dhn_C2_8_templated.relnn` | `.relnn` file | Paper LOC claim uses this templated version (55 LOC) |
| R-GCN (AIFB, MUTAG) | `research/paper_experiments/rgcn/run_compare_entities_rgcn.py` (`_generate_relnn_rgcn_dsl`) | generated string | RelNN DSL is produced from a parameterized generator |
| RelBench rel-f1 | `research/paper_experiments/relbench/run_relbench_f1_multirun.py` (`_build_dsl`) | generated string | Same DSL shape across tasks (position, dnf, top3); dimensions are dataset-driven |
| CTU Hepatitis | `research/paper_experiments/hepatitis/run_hepatitis_multirun.py` (`HEPATITIS_DSL`) | inline string | 9 DSL lines; Patients + Biopsies + Labs tables |

## Quick start

```powershell
$py = "python"

# GCN — Cora 3-way comparison (~2 min, CPU)
& $py research/paper_experiments/gcn/run_compare_cora_pytorch.py

# HGT — DBLP 5-seed benchmark (~18 min, GPU recommended)
& $py research/paper_experiments/hgt/run_compare_dblp_hgt_multirun.py --runs 5

# DHN — full benchmark suite (~30 min, CPU)
& $py research/paper_experiments/dhn/run_pure_benchmarks.py

# R-GCN — AIFB + MUTAG (long on CPU; use --cpu-only --num-threads 1 for stability)
& $py research/paper_experiments/rgcn/run_compare_entities_rgcn.py --datasets AIFB MUTAG --runs 5 --cpu-only --num-threads 1

# RelBench rel-f1 (requires: pip install relbench; per-task tuned hparams by default)
& $py research/paper_experiments/relbench/run_relbench_f1_multirun.py --runs 5 --epochs 200 --cpu-only

# CTU Hepatitis (requires: pip install pymysql scikit-learn; downloads ~2.2 MB on first run)
& $py research/paper_experiments/hepatitis/run_hepatitis_multirun.py --runs 5 --epochs 200 --cpu-only
```

Each subdirectory has a `REPRODUCE.md` with detailed instructions, expected
outputs, and the mapping from scripts to paper table rows.

## Consolidated results

See `docs/paper_experiment_results.md` for the master results table used
when filling in `main.tex`.

## Note

These scripts are **not** part of the `pytest` test suite. The automated tests
live in `tests/` (smoke, feature, slow). The scripts here are for paper
reproduction only and may require a GPU or long runtimes.
