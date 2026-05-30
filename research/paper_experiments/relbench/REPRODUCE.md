# RelBench rel-f1 (driver-position, driver-dnf, driver-top3)

## Paper mapping

| Table | Row |
|-------|-----|
| `tab:exp-roadmap` | RelBench rel-f1 tasks |
| `tab:unified-results` | driver-position (MAE), driver-dnf (AUROC), driver-top3 (AUROC) |
| Appendix | Secondary task metrics |

## Prerequisites

```bash
pip install relbench
```

## Scripts

| Script | Purpose |
|--------|---------|
| `run_relbench_f1_multirun.py` | Fixed hyperparameters, multi-seed, official `task.evaluate` |
| `run_relbench_f1_tuning.py` | Coarse grid (stage A) + confirmation runs (stage B) |

## Commands

```powershell
$py = "python"

# --- CANONICAL PAPER RUN (all 3 tasks) ---
# Tuned hparams per task:
#   driver-position: h=32, lr=0.01, wd=1e-4
#   driver-dnf:      h=16, lr=0.003, wd=0
#   driver-top3:     h=64, lr=0.005, wd=5e-4
& $py research/paper_experiments/relbench/run_relbench_f1_multirun.py --runs 5 --epochs 200 --cpu-only --out-json research/paper_experiments/relbench/results/relbench_f1_multirun_200ep_5seed.json

# --- driver-top3 only (separate output file) ---
& $py research/paper_experiments/relbench/run_relbench_f1_multirun.py --tasks driver-top3 --runs 5 --epochs 200 --cpu-only --out-json research/paper_experiments/relbench/results/relbench_f1_top3_200ep_5seed.json

# --- RE-TUNING (if needed) ---
& $py research/paper_experiments/relbench/run_relbench_f1_tuning.py --tasks driver-top3 --confirm-epochs 200 --seeds 5 --cpu-only

# --- EXTRA-TABLES VARIANT (experimental, not in paper) ---
# Appends qualifying + standings columns. Shows higher single-seed perf but degrades across seeds.
# Extra-tables tuning found high single-seed perf but poor multi-seed generalization.
& $py research/paper_experiments/relbench/run_relbench_f1_multirun.py --runs 5 --epochs 200 --cpu-only --extra-tables --out-json research/paper_experiments/relbench/results/relbench_f1_multirun_200ep_5seed_extra.local.json
```

> The `--out-md` summary file defaults to `<out-json-stem>_summary.md` in the same directory.

## Canonical outputs

- `results/relbench_f1_multirun_200ep_5seed.json` — **canonical paper results** (5 seeds 100–104, 200 ep, no extra-tables)
- `results/relbench_f1_multirun_summary.md` — human-readable summary of canonical results
- Tuning/extra-table outputs are intentionally not tracked in Git; generate locally with `--out-json` when needed.

## Model

Relational model with race context:

- `DriverEmb`, `ConsEmb`, `RaceEmb` projections
- `DriverHistory` from `Results` joined with constructor and race embeddings
- `Score` head over `Concat(driver, history)`
- Task-specific loss:
  - `driver-position`: `MSELoss`
  - `driver-dnf`: `BCEWithLogitsLoss`
  - `driver-top3`: `BCEWithLogitsLoss`

The loader exposes `Drivers`, `Constructors`, `Races`, `Results`, `TrainLabels`. Pass `extra_tables=True` to `load_relbench_f1_dataset` (or `--extra-tables` on the multirun script) to append normalized qualifying + driver-standings features to each result row.

## Gap notes

See `docs/benchmarks/relbench_gap_notes.md` for framework vs glue code.
