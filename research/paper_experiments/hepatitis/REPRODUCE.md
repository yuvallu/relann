# CTU Hepatitis (HBV vs HCV binary classification)

## Paper mapping

| Table | Row |
|-------|-----|
| `tab:unified-results` | Hepatitis row (AUC ROC) |
| `tab:loc-comparison` | Hepatitis row (9 DSL lines) |

## Dataset

**CTU Hepatitis_std** from the CTU Relational Learning Repository (Hepatitis B vs C classification).

- Source: `relational.fel.cvut.cz` (public MariaDB, guest credentials)
- 500 patients across 3 tables: `dispat` (patient features), `Bio` (biopsies), `indis` (lab tests)
- Task: binary classification of Hepatitis B (0) vs Hepatitis C (1)
- Split: random 70/15/15 on patient ID (seed=42, matching ReDeLEx "orig." protocol)
- Metric: AUC ROC (higher is better)

## Baselines

From ReDeLEx Table 1 (arXiv:2506.22199, ECML PKDD 2025):

| Model | Type | Test AUC |
|-------|------|----------|
| LightGBM | Single-table GBDT | 0.626 |
| GraphSAGE (ResNet SAGE) | RDL/GNN | 1.000 |
| DBFormer | RDL/GNN | 0.996 |

**Key finding**: Flat methods (~0.63) are far below GNN methods (~1.0), showing that relational
structure is essential for this task.

## Canonical RelNN DSL

The DSL is inline in `run_hepatitis_multirun.py` (`HEPATITIS_DSL` string constant):

```relnn
d_patient = 2 .
d_biopsy  = 2 .
d_lab     = 10 .
hidden    = {hidden} .

PatientEmb(m_id; ReLU()(Linear(d_patient, hidden)(z))) :- Patients(m_id; z) .
BiopsyEmb(m_id; mean(ReLU()(Linear(d_biopsy, hidden)(z)))) :- Biopsies(biopsy_id, m_id; z) .
LabEmb(m_id; mean(ReLU()(Linear(d_lab, hidden)(z)))) :- Labs(lab_id, m_id; z) .
Score(m_id; Linear(hidden * 3, 1)(Concat(z_p, z_b, z_l))) :-
    PatientEmb(m_id; z_p), BiopsyEmb(m_id; z_b), LabEmb(m_id; z_l) .
```

**9 meaningful DSL lines** (4 dimension declarations + 4 rules + Score rule).
LOC count for paper: **9**.

## Prerequisites

```bash
pip install pymysql scikit-learn
```

No GPU required (CPU training is fast: ~0.1s/epoch for this dataset).

First run downloads all tables from the CTU MariaDB server (~2.2 MB) and caches
them as parquet files in `~/.cache/ctu_hepatitis/`. Subsequent runs load from cache.

## Commands

```powershell
$py = "python"

# --- TUNING (run once to find best hparams) ---
& $py research/paper_experiments/hepatitis/run_hepatitis_tuning.py --stage-a-epochs 30 --confirm-epochs 100 --seeds 3 --cpu-only

# --- CANONICAL PAPER RUN ---
# Update TUNED_HPARAMS in run_hepatitis_multirun.py with tuning output, then:
& $py research/paper_experiments/hepatitis/run_hepatitis_multirun.py --runs 5 --epochs 200 --cpu-only --out-json research/paper_experiments/hepatitis/results/hepatitis_200ep_5seed.json
```

## Canonical outputs

- `results/hepatitis_200ep_5seed.json` -- canonical paper results (5 seeds 100–104, 200 ep)
- `results/hepatitis_200ep_5seed_summary.md` -- human-readable summary

## Model architecture

3-table relational architecture with mean aggregation:

1. `PatientEmb(m_id)` -- project patient demographics (sex, age) to hidden dim
2. `BiopsyEmb(m_id)` -- mean-pool biopsy features (fibros, activity) per patient
3. `LabEmb(m_id)` -- mean-pool 10 lab measurements per patient visit
4. `Score(m_id)` -- MLP over concatenated [patient, biopsy, lab] embeddings

Loss: `BCEWithLogitsLoss` (binary classification).
Evaluation: `sklearn.metrics.roc_auc_score`.
