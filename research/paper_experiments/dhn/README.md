# DHN RelNN Experiments

**Canonical configuration:** C2:4 (cycles C2, C3, C4) using edge-join enumeration.

This directory contains RelNN programs and runners to reproduce DHN results from the paper.
For the equations-vs-code comparison, see `dhn_ghl_csl_equations.ipynb` and
`dhn_equations_vs_relnn.pdf` — the paper's equation-to-rule correspondence figure
(paper LaTeX equations side by side with the RelNN DSL).

## Quick Start

Run RelNN implementation on CSL:

```bash
python research/paper_experiments/dhn/run_dhn_ghl.py --dataset CSL --config c2_4 --epochs 500
```

Run official DHN on CSL:

```bash
python scripts/setup_external_dhn.py  # one-time setup
python research/paper_experiments/dhn/run_official_dhn.py --dataset CSL --max-cycle 4
```

Run both and compare timings:

```bash
python research/paper_experiments/dhn/run_compare.py
```

## Experiments

- **RelNN C2:4:** `run_dhn_ghl.py --dataset {CSL,EXP} --config c2_4`
- **RelNN C2:10:** `run_dhn_ghl.py --dataset {CSL,EXP} --config c2_10` (requires simple-cycles precompute)
- **Official DHN C2:4:** `run_official_dhn.py --dataset {CSL,EXP} --max-cycle 4`
- **Official DHN C2:10:** `run_official_dhn.py --dataset {CSL,EXP} --max-cycle 10`

## Files

- `dhn_ghl_csl_c2_4.relnn` — RelNN C2:4 (edge-join), canonical for paper
- `dhn_ghl_csl_c2_10.relnn` — RelNN C2:10 (simple-cycles precompute)
- `run_dhn_ghl.py` — Runner for RelNN
- `run_official_dhn.py` — Runner for official DHN
- `run_compare.py` — Side-by-side benchmark (RelNN vs official)
- `dhn_ghl_csl_equations.ipynb` — Mathematical notation → RelNN DSL correspondence
- `dhn_equations_vs_relnn.pdf` — Paper equation-to-rule correspondence figure (LaTeX equations vs RelNN DSL)
- `results/` — JSON outputs with timing and accuracy

## Notes

- **Default seed:** 42
- **Default epochs:** 500
- **Default device:** CPU (for fair timing comparison)
- **Simple cycles:** C2:10 precompute uses `nx.simple_cycles` (matches official gear/dhn strategy)
- **Paper tables:** See `results/dhn_table_timing.json` for consolidated results
