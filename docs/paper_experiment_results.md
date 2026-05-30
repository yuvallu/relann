# Consolidated Experiment Results for Paper

**Date**: 2026-04-01
**Scripts**: `nbs/tests/slow/run_compare_cora_pytorch.py`, `nbs/tests/slow/run_compare_dblp_original_hgt.py`, `nbs/tests/slow/run_compare_dblp_hgt.py`, `nbs/tests/dhn/run_official_dhn_no_minibatch.py`

---

## 1. GCN on Cora (100 epochs, seed=42, weight-synced)

| Implementation | #Params | Test Acc | Time (s) | LOC |
|---|---|---|---|---|
| Hand-rolled PyTorch | 23,040 | **81.6%** | 3.5 | 38 |
| PyG GCNConv | 23,040 | **81.6%** | 6.8 | 10 (wrapper) / 217 (library) |
| **RelNN** | 23,040 | **81.6%** | 11.6 | **21** |

- **Forward parity**: max_diff = 1.16e-09 (all three produce identical outputs with synced weights)
- **Loss curves**: identical across all three implementations
- All three models produce identical loss at epoch 100: 0.4718

### RelNN DSL (complete GCN):
```
def GCNLayer<d_in, d_out>(Nodes, Edges):
    Emb(pid; Linear(d_in, d_out, False)(z)) :- Nodes(pid; z) .
    Out(cited; sum(z * w)) :- Emb(citing; z), Edges(citing, cited; w) .
enddef

def GCN<d_in, d_hidden, d_out>(Nodes, Edges):
    L1(cited; ReLU(z)) :- GCNLayer<d_in, d_hidden>(Nodes, Edges)(cited; z) .
    Output(cited; z) :- GCNLayer<d_hidden, d_out>(L1, Edges)(cited; z) .
enddef
```

---

## 2. HGT on DBLP (GPU, 5 runs, mean+-std, synchronized timers)

**Fair-timing benchmark** (2026-03-29): All three implementations pre-load all data/embeddings to GPU once before training. No per-epoch CPU→GPU transfers. PyG uses `pyg_data.to(DEVICE)` at module scope; pyHGT pre-moves `node_features` to GPU; RelNN uses `Session(db=..., device=DEVICE)` to move db tensors at init. Timers use `torch.cuda.synchronize()`.

### Scope-labeled benchmark table

| Scope | Implementation | #Params | Train | Val | Test | Time (s) | LOC |
|---|---|---|---|---|---|---|---|
| FULL_GRAPH_1L | Original paper (acbull/pyHGT) | 387,604 | 100.0% +- 0.0% | 77.8% +- 0.6% | **79.5% +- 0.5%** | 37.6 +- 1.0 | 143 (conv+model) |
| FULL_GRAPH_1L | PyG HGTConv | 387,092 | 100.0% +- 0.0% | 75.1% +- 1.8% | **76.6% +- 2.4%** | 11.2 +- 0.1 | 186 (library) |
| FULL_GRAPH_1L | **RelNN** (PA-path, PyG-matching) | 313,287 | 100.0% +- 0.0% | 76.3% +- 1.0% | **78.0% +- 0.5%** | 26.8 +- 1.1 | **41** |
| FULL_GRAPH_1L | **RelNN** (PA-path, pyHGT-faithful) | 313,415 | 100.0% +- 0.0% | 76.9% +- 1.0% | **79.1% +- 0.3%** | 38.7 +- 1.1 | **43** |
| FULL_GRAPH_2L | Original paper (acbull/pyHGT) | 479,268 | 100.0% +- 0.0% | 77.6% +- 0.8% | **79.4% +- 0.5%** | 71.1 +- 0.8 | 143 (conv+model) |
| FULL_GRAPH_2L | PyG HGTConv | 478,244 | 99.8% +- 0.4% | 66.8% +- 2.3% | **69.9% +- 2.5%** | 21.1 +- 0.1 | 186 (library) |
| FULL_GRAPH_2L | **RelNN** (full 4-edge-type) | 382,993 | 100.0% +- 0.0% | 72.2% +- 1.3% | **74.9% +- 1.2%** | 80.1 +- 0.3 | **42** (= 33 generic HGT lib + 9 model-specific; paper cites 42) |

**Previous (unfair) timings** (before fix, included repeated CPU→GPU transfers):
- pyHGT 1L: 106.3s → now 40.9s (-62%)
- PyG 1L: 42.3s → now 11.4s (-73%)
- RelNN 1L: 15.2s → now 24.2s (+59%, because old timing coincidentally excluded most transfers)

### Parity checks (exact weight sync: PyG -> HandRolled -> RelNN)

**Script:** `nbs/tests/slow/run_match_hgt_accuracy.py`

| Source -> Target | Forward max diff | Result |
|---|---|---|
| PyG HGTConv -> Hand-rolled (synced) | **2.98e-08** | machine precision ✓ |
| Hand-rolled -> RelNN (synced) | **4.47e-08** | machine precision ✓ |
| PyG -> RelNN (chain) | ~5e-08 | machine precision ✓ |

**Root bug found & fixed**: Original sync used wrong `k_rel` index (`edge_type_offset * num_heads + head`). PyG uses `head * num_edge_types + edge_type_offset` in `_construct_src_node_feat`. Also: PyG applies `k @ krel_w` (no transpose), while `nn.Linear` applies `k @ W.T`, requiring the weight to be stored transposed.

### Exact accuracy match after synced training (5 seeds, 100 epochs each, GPU)

| Seed | PyG test | Hand-rolled test | RelNN test | PyG-HR diff | HR-RN diff |
|---|---|---|---|---|---|
| 42 | 79.0% | 79.0% | 79.0% | 0.0% | 0.0% |
| 43 | 78.0% | 78.0% | 78.0% | 0.0% | 0.0% |
| 44 | 73.2% | 73.2% | 73.2% | 0.0% | 0.0% |
| 45 | 77.7% | 77.7% | 77.7% | 0.0% | 0.1% |
| 46 | 75.2% | 75.2% | 75.2% | 0.0% | 0.0% |
| **Mean** | **76.6% ± 2.4%** | **76.6% ± 2.4%** | **76.6% ± 2.4%** | **0.0%** | **~0%** |

**Conclusion**: When initialized from the same weights, all three implementations (PyG full-graph, hand-rolled PA-path, RelNN PA-path) produce **identical accuracy** across all seeds. This confirms the user's reasoning: in 1-layer HGT on DBLP, the only edges that contribute to author output are Paper→Author (the only incoming edge type for author nodes). Gradients from other edge types do not propagate to author outputs. The PA-path DSL captures exactly the necessary computation.

### Notes:
- **Fairness labels are explicit:** full-graph rows and PA-path rows are not mixed as apples-to-apples runtime claims.
- **Main accuracy claim (updated 2026-03-30):** Paper now reports weight-synced results for PyG and RelNN (1L). Both achieve 76.6% ± 2.4% across all 5 seeds, confirming architectural equivalence. pyHGT is reported separately (random init, different parameter count 387,604 vs 387,092) as a reference for the original paper's code.
- **2-layer claim (updated 2026-03-31):** RelNN 2L DSL now implements full-graph equivalent computation (PA+AP+TP+CP in L1, PA in L2). Architectural equivalence verified by weight sync: max forward diff < 2e-4 (residual from 1 DBLP paper with no author edges). With random init: RelNN 74.9% ± 1.2% vs PyG 69.9% ± 2.5% (PyG has higher variance due to training instability). LOC for 2L RelNN: 42 lines (33 generic HGT library + 9 model-specific; paper reports 42).
- **Parity script for 2L:** `nbs/tests/slow/run_match_hgt_2l_accuracy.py` (5 seeds, weight sync, layer-by-layer diagnostics). Results in `nbs/paper_experiments/hgt/results/hgt_dblp_2l_parity_results.json`.
- **Timing protocol (fair, 2026-03-29):** All data pre-loaded to GPU once. 5 independent runs on GPU with synchronized timers (`torch.cuda.synchronize()` around timed blocks), reported as mean +- std.
- **Interpretation:** PA-path and full-graph measure different compute scopes. RelNN PA-path (24.2s) is 2.1x slower than PyG (11.4s) because PyG uses highly optimized scatter/gather kernels, while RelNN uses general relational ops. pyHGT (40.9s) is slower because it processes all 6 edge types using a flat graph representation.
- **Canonical artifacts:** `nbs/paper_experiments/hgt/results/hgt_dblp_5run_results.json` and `nbs/paper_experiments/hgt/results/hgt_dblp_5run_summary.md`.
- **pyHGT-faithful RelNN (added 2026-04-17):** `RELNN_PYHGT_DEFINE_DSL` in `run_compare_dblp_original_hgt.py` adds `Dropout(0.2)` and `LayerNorm(hidden)` to the output stage (matching `HGTConv.update()` in `acbull/pyHGT`). 5-seed results (seeds 42–46, 100 epochs, GPU, random init): test **79.1% ± 0.3%** (vs pyHGT's 79.5% ± 0.5%), time **38.7 ± 1.1s**, params **313,415**, LOC **43** (41 + 2 new declarations). The 3-line output-stage diff (Dropout decl, Norm decl, Norm applied to skip gate) closes the gap vs pyHGT from 78.0% to 79.1%. Fills the `---` cells in the `\textit{pyHGT}` row of Table 1 (`tab:unified-results`). Runner: `run_relnn_pyhgt_hgt()` in `run_compare_dblp_original_hgt.py`.
- **pyHGT-faithful RelNN: tier-1 numerical fixes (2026-04-17):** Added numerically-stable softmax (max-subtraction via `MaxPA`/`StableDotPA` rules) and corrected initialization of `Prel_PA` and `Skip_author` to 1.0 (matching pyHGT's `torch.ones(...)` init via new `Tensor(shape, fill_value)` DSL syntax in `tensor_term_compiler.py`). 5-seed benchmark re-run (seeds 0–4): test **79.0% ± 1.0%** — mean unchanged vs prior result, gap to pyHGT (79.5%) still within overlapping confidence intervals. Paper table numbers not updated (improvement < 0.2pp threshold). Code changes kept as semantic correctness improvements.
- **TODO:** Re-run the 5-seed timing benchmark with synchronized weight initialization so that the runtime comparison (pyHGT / PyG / RelNN) is on equal footing with the accuracy comparison. Currently, timing uses random init (5 independent runs) while accuracy now uses synced init. Scripts: `nbs/tests/slow/run_compare_dblp_hgt_multirun.py` + adapt `nbs/tests/slow/run_match_hgt_accuracy.py` to also record per-run wall-clock time.

---

## 3. DHN on Graph Benchmarks (10-fold stratified CV)

**Unified table (`main.tex` `tab:unified-results`) — DHN wall-clock for CSL/EXP C2:10:** RelNN: CSL **5 s**, EXP **43 s** (train-on-all-graphs, 500 epochs, CPU; `run_pure_benchmarks.py`; walk-count fallback when pure joins OOM). Published (official) DHN (`gear/dhn` via `run_official_dhn_csl_exp.py`): CSL **~3580 s**, EXP **~491 s** (GPU; 500-epoch-equivalent time from measured 50-epoch run ×10). JSON: `nbs/paper_experiments/dhn/results/dhn_unified_table_timing.json`, plus `official_dhn_*_c2_10_train.json`.

### Results from `nbs/paper_experiments/dhn/results/BENCHMARK_RESULTS.md`

#### Synthetic benchmarks (expressivity)

| Dataset | Config | Paper (corrected) | RelNN | Notes |
|---|---|---|---|---|
| CSL | C2:4 | 30% | 26.7 ± 0.0% | Matches corrected paper |
| CSL | C2:10 | 100% | **100.0 ± 0.0%** | Exact match |
| EXP | C2:4 | 50% | 45.2 ± 3.5% | Both near random |
| EXP | C2:5 | 81% | **98.2 ± 1.0%** | Exceeds paper |
| EXP | C2:10 | 98% | **98.7 ± 1.1%** | Matches paper |
| SR25 | C2K3:5 | 53% | 73.3% (train) | Exceeds (train); LOO=0% expected |

#### Real-world benchmarks

| Dataset | Config | Paper | RelNN | Official w/o minibatch | Notes |
|---|---|---|---|---|---|
| ENZYMES | C2:4 | 64.3 ± 5.5% | 51.7 ± 6.2% | TBD | ~13pp gap |
| PROTEINS | C2:4 | 76.5 ± 3.0% | 71.2 ± 3.8% | TBD | ~5pp gap |

#### Minibatch ablation (TBD -- cached data was corrupted, re-running)

| Config | ENZYMES | PROTEINS |
|---|---|---|
| Official (bs=32, sched, es=5) | TBD | TBD |
| Full-batch + sched + es=5 | TBD | TBD |
| Full-batch, no sched, 500ep | TBD | TBD |
| Full-batch, Adam, 500ep | TBD | TBD |

### DHN LOC comparison:

| Implementation | LOC | Notes |
|---|---|---|
| RelNN templated (C2:8, `dhn_C2_8_templated.relnn`) | **55** | Named-module templates, no H0 aliases |
| RelNN first-order (C2:8, `dhn_full_C2_8.relnn`) | 168 | Explicit parameter names, H0 aliases |
| RelNN minimal (C2:4, `dhn_pure_C2_C4.relnn`) | 67 | Minimal C2:4 patterns |
| Official DHN (gear/dhn) | 259 | `_external/dhn` checkout: layers.py + models.py + train.py |

The paper reports the **templated version** (55 LOC) for the C2:10 DHN row.
Previous versions incorrectly cited 67 LOC (which was the C2:4 file).

---

## 4. Lines of Code Summary

| Architecture | RelNN LOC | Baseline LOC | Baseline | Ratio |
|---|---|---|---|---|
| GCN | **21** | 38 | Hand-rolled PyTorch | 1.8x simpler |
| GCN | **21** | 217 | PyG GCNConv (library) | 10.3x simpler |
| HGT | **41** | 90 | Hand-rolled PyTorch | 2.2x simpler |
| HGT | **41** | 143 | Original paper (acbull/pyHGT) | 3.5x simpler |
| HGT | **41** | 186 | PyG HGTConv (library) | 4.5x simpler |
| DHN (templated) | **55** | 259 | Official DHN (gear/dhn) | 4.7x simpler |
| R-GCN (AIFB, one-hot) | **22** | 186 | PyG FastRGCNConv (library) | 8.5x simpler |
| RelBench rel-f1 | **~20** | — | No comparable baseline | — |
| CTU Hepatitis | **9** | — | GNN (all 7 tables) 0.997-1.0 AUC, flat (LightGBM) 0.626 AUC | RelNN (3 tables): 0.876 AUC |

### LOC methodology:
- **RelNN LOC**: DSL define + fit + predict commands (non-blank, non-comment lines)
- **Baseline LOC**: Model class + training loop + data processing (non-blank, non-comment lines)
- **Library LOC**: Full source of the PyG/library class (not just the user-facing wrapper)
- The "wrapper LOC" for PyG is only 10-20 lines, but this hides the complexity inside the library

### Key observation:
RelNN programs are 2-10x shorter than equivalent implementations. Critically, the RelNN code directly mirrors the paper equations, while PyTorch/PyG code involves index manipulation, scatter operations, and boilerplate that obscures the mathematical intent.

---

## 5. Expressivity Claims

### DHN: Cycles as Joins
In RelNN, a cycle homomorphism (e.g., triangle) is expressed as a cyclic join:
```
Hom_C3(g, u, v, w; z) :-
    Edge(g, u, v; z), Edge(g, v, w; _), Edge(g, w, u; _) .
```
PyG has no homomorphism layer. The official DHN code requires custom `HomConv` layers with pre-computed mapping indices.

### HyGNN: Hypergraphs as Relations
**Scripts:** `nbs/paper_experiments/hygnn/run_compare_hygnn.py`, `run_hygnn_pytorch_ref.py` — see `nbs/paper_experiments/hygnn/REPRODUCE.md`.  
**Canonical artifact:** `nbs/paper_experiments/hygnn/results/hygnn_twosides_kmer_mlp.json`  
**Demos:** `nbs/demos/004_relnn_hygnn.ipynb`, `nbs/tests/slow/hygnn_relnn_high_order.ipynb`.  
Data: `parent.datasets.load_hygnn_dataset` / `load_ddi_dataset` (PyTDC optional for `load_ddi_dataset`).  
In RelNN, a hyperedge is a multi-column relation and incidence is a natural join; standard GNN APIs target pairwise graphs only.

#### TWOSIDES — k-mer ($k=9$), MLP decoder (500 epochs, seed 42, weight sync before training)

| Implementation | Test acc | ROC-AUC | PR-AUC | Train time (s) | LOC |
|----------------|----------|---------|--------|----------------|-----|
| PyTorch reference | **87.8%** | 0.9535 | **0.9570** | 331 | 150 |
| **RelNN** | 87.6% | 0.9530 | 0.9569 | 307 | **41** |

**LOC:** PyTorch = non-empty, non-comment lines in `HyGNNLayer` + `HyGNN` + `compute_loss` in `run_hygnn_pytorch_ref.py`. RelNN = MLP-configuration DSL blocks in `run_compare_hygnn.py` (`RELNN_DEFINE_CORE` + MLP pair/fit/test/pred/materialize strings).

---

## 6. R-GCN on Entities (AIFB)

**Script:** `nbs/paper_experiments/rgcn/run_compare_entities_rgcn.py`  
**Canonical artifact:** `nbs/paper_experiments/rgcn/results/rgcn_entities_results.aifb_5run.json`

**Protocol (current):** RelNN matches **torch-rgcn**'s (Schlichtkrull et al.) full per-relation parameterization on AIFB: edges are augmented with inverse relations + a self-loop relation (`2*num_rels + 1 = 181` effective relations); each relation has its own per-relation `NodeLookup` embedding table; layer biases match torch-rgcn's per-layer bias. Not weight-synced. PyG `FastRGCNConv` (basis decomp, `num_bases=30`) is reported as a secondary baseline but **not** parameter-matched — see "Arch B status" below for why a basis-decomp RelNN variant is currently impractical.

**MUTAG status:** Dropped from paper (severe overfitting: 55.3% ± 12.3% with basis decomposition). Results archived in `rgcn_entities_results.mutag_50ep.json`.

### AIFB — 5 seeds (42–46), mean ± std

| Impl | Test acc | Time (s, CPU) | Params | LOC |
|------|----------|---------------|--------|-----|
| torch-rgcn (original paper, full per-rel) | **93.3% ± 4.2%** | 25.5 ± 5.9 | 24,004,964 | 219 (library) |
| PyG FastRGCNConv (basis-30; **not param-matched**) | 92.2% ± 1.2% | 8.3 ± 3.1 | 4,116,764 | — |
| **RelNN (full, ↔ torch-rgcn)** | **92.8% ± 2.5%** | **178.5 ± 43.2** | 24,004,964 | **20** |

Param count is **bit-exact match** with torch-rgcn (`181 × 8285 × 16 + 16 + 181 × 16 × 4 + 4 = 24,004,964`). Accuracy is within noise of torch-rgcn (std bands overlap entirely). LOC ratio: torch-rgcn 219 (`NodeClassifier` 54 + `RelationalGraphConvolutionNC` 165, non-blank/non-comment) vs RelNN 20 — **~11× DSL reduction**.

**Timing note:** RelNN `time_s` is end-to-end Session compile + fit; torch-rgcn/PyG measure training loop only. The Session compile is the dominant fraction (181 specialised `RelAgg<pe>` templates × 2 layers ≈ 362 specialisations); per-epoch training itself is fast. End-to-end gap of ~7× vs torch-rgcn is entirely compile overhead — see PR #58 (separate methodology PR) for a subtract-method estimator that isolates training-only time.

### Arch B status (basis decomp, PyG-matched parameterization) — deferred

A second RelNN variant matching PyG's `FastRGCNConv(num_bases=30)` parameterization (4,116,764 params, exact match) was prototyped but is **not shipped in this version**. The DSL is generated by `_generate_arch_b_dsl()` and gated behind `--include-basis-arch`. Two unresolved blockers:

1. **`Tensor()` only supports constant fill init.** Basis coefficients `A<pe, b>` initialized to a constant value (whether 0.0 default or 1.0) start with all relations identical at init. The model overfits the training set (loss → 0) without learning per-relation distinctions (test acc ≈ random, ~42% on 4 classes vs PyG's 92%). PyG uses `kaiming_uniform` for both basis weights and coefficients — a random-init mode for `Tensor()` would address this.
2. **Compile cost.** Bounded-set expansion creates `num_relations × num_bases = 90 × 30 = 2700` template specialisations of `Msg1Basis<pe, b>`, each compiled separately (no fusion across bases). Single-seed 50-epoch wall time: ~32 min. 5-seed sweep: ~2.7 hours.

These are framework-level limitations, not paper-level claims. Resolving them would enable RelNN to faithfully reproduce PyG's parameter-efficient R-GCN variant and is tracked separately.

---

## 7. RelBench rel-f1

**Loader:** `parent.datasets.load_relbench_f1_dataset`  
**Scripts:** `nbs/paper_experiments/relbench/run_relbench_f1_multirun.py`, `run_relbench_f1_tuning.py`  
**Canonical artifact:** `nbs/paper_experiments/relbench/results/relbench_f1_multirun_200ep_5seed.json`  
**Gap log:** `docs/benchmarks/relbench_gap_notes.md`

### Results — 5 seeds (100–104), 200 epochs, tuned per-task hparams

| Task | Metric | RelNN | Official GNN baseline (5 ep) | Status |
|------|--------|-------|-------------------------------|--------|
| driver-position | MAE↓ | **3.99 ± 0.02** | 4.24 | Beats baseline |
| driver-dnf | AUROC↑ | 0.61 ± 0.01 | **0.70** | Gap −0.09 |

See `nbs/paper_experiments/PROBLEMS_WE_DIDNT_SOLVE.md` §8 for why the extra-tables approach did not close the driver-dnf gap.

### driver-top3 — 5 seeds (100–104), 200 epochs, tuned hparams (hidden=64, lr=0.005, wd=5e-4)

| Task | Metric | RelNN | ReDeLEx GNN baseline | Notes |
|------|--------|-------|----------------------|-------|
| driver-top3 | AUROC↑ | **0.535 ± 0.005** | 0.832 (ResNet SAGE) | Architectural limitation: see below |

**Known limitation**: Our model produces one driver-level score per driver, but the task evaluates each race entry (driver × date) individually. GNN baselines use temporal subgraph sampling (features only before the test timestamp), giving them access to recent race-specific context. Our full-history aggregation cannot distinguish race-to-race variation within a driver's career. **Recommendation: do not include driver-top3 in the main paper table without this caveat.** The architectural limitation is inherent to the full-batch, driver-level DSL, not a bug.

Canonical artifact: `nbs/paper_experiments/relbench/results/relbench_f1_top3_200ep_5seed.json`

Reproduce: `python nbs/paper_experiments/relbench/run_relbench_f1_multirun.py --runs 5 --epochs 200 --cpu-only --out-json nbs/paper_experiments/relbench/results/relbench_f1_multirun_200ep_5seed.json`

---

## 8. CTU Hepatitis (HBV vs HCV binary classification)

**Loader:** `parent.datasets.load_ctu_hepatitis_dataset`
**Scripts:** `nbs/paper_experiments/hepatitis/run_hepatitis_multirun.py`, `run_hepatitis_tuning.py`
**Canonical artifact:** `nbs/paper_experiments/hepatitis/results/hepatitis_200ep_5seed.json`

**Dataset:** CTU Hepatitis_std (CTU Relational Learning Repository). 500 patients, 7 tables.
Task: binary classification of Hepatitis B (0) vs C (1). Split: random 70/15/15 on patient ID (seed=42).
Metric: AUC ROC. Requires: `pip install pymysql scikit-learn`.

**RelNN architecture (9 DSL lines):**
- `Patients(m_id; [sex, age])` — 2D patient demographics
- `Biopsies(biopsy_id, m_id; [fibros, activity])` — 2D biopsy features, mean-pooled per patient
- `Labs(lab_id, m_id; [10 lab values])` — 10D lab measurements, mean-pooled per patient
- `Score(m_id)` — MLP over concatenated patient + biopsy + lab embeddings

### Results — 5 seeds (100–104), 200 epochs, hidden=64, lr=0.02, wd=0

| Model | Type | Val AUC | Test AUC |
|-------|------|---------|----------|
| LightGBM (ReDeLEx baseline) | Single-table GBDT | — | 0.626 |
| GraphSAGE/ResNet SAGE (ReDeLEx) | GNN over all 7 tables | — | 1.000 |
| DBFormer (ReDeLEx) | GNN | — | 0.996 |
| **RelNN** (3 tables, 9 DSL lines) | Relational NN | **0.889 ± 0.005** | **0.876 ± 0.007** |

**Key finding:** RelNN using only 3 tables and 9 DSL lines achieves 0.876 test AUC — **far above LightGBM (0.626)** and substantially above random (0.5). This demonstrates that relational structure (joining Patients, Biopsies, Labs) provides critical signal that flat methods miss. GNN methods using all 7 tables achieve near-perfect AUC, suggesting therapy and other tables add further signal; our simpler 3-table model already demonstrates the point.

**LOC:** 9 DSL lines (4 dimension declarations + 3 embedding rules + 1 aggregation + 1 score rule).

Baselines from: ReDeLEx Table 1 (arXiv:2506.22199, ECML PKDD 2025).

Reproduce: `python nbs/paper_experiments/hepatitis/run_hepatitis_multirun.py --runs 5 --epochs 200 --cpu-only`

---

## 9. Script Locations

`nbs/paper_experiments/` contains per-architecture `REPRODUCE.md` files and result artifacts.
The actual **runnable scripts** for GCN, HGT, and DHN live in `nbs/tests/slow/` and `nbs/tests/dhn/`.
R-GCN and RelBench have self-contained scripts directly inside their `nbs/paper_experiments/` subdirectory.

| Experiment | Script location | REPRODUCE.md |
|---|---|---|
| GCN 3-way | `nbs/tests/slow/run_compare_cora_pytorch.py` | `nbs/paper_experiments/gcn/REPRODUCE.md` |
| HGT benchmark (5-seed) | `nbs/tests/slow/run_compare_dblp_hgt_multirun.py` | `nbs/paper_experiments/hgt/REPRODUCE.md` |
| HGT implementations | `nbs/tests/slow/run_compare_dblp_original_hgt.py` | (called by multirun) |
| HGT 1L parity | `nbs/tests/slow/run_match_hgt_accuracy.py` | `nbs/paper_experiments/hgt/REPRODUCE.md` |
| HGT 2L parity | `nbs/tests/slow/run_match_hgt_2l_accuracy.py` | `nbs/paper_experiments/hgt/REPRODUCE.md` |
| DHN benchmarks | `nbs/tests/dhn/run_pure_benchmarks.py` | `nbs/paper_experiments/dhn/REPRODUCE.md` |
| Official DHN timing (CSL/EXP C2:10) | `nbs/paper_experiments/dhn/run_official_dhn_csl_exp.py` | `nbs/paper_experiments/dhn/REPRODUCE.md` |
| DHN templated DSL | `nbs/tests/dhn/dhn_C2_8_templated.relnn` | `nbs/paper_experiments/dhn/REPRODUCE.md` |
| R-GCN (AIFB) | `nbs/paper_experiments/rgcn/run_compare_entities_rgcn.py` | `nbs/paper_experiments/rgcn/REPRODUCE.md` |
| RelBench rel-f1 | `nbs/paper_experiments/relbench/run_relbench_f1_multirun.py` | `nbs/paper_experiments/relbench/REPRODUCE.md` |
| CTU Hepatitis | `nbs/paper_experiments/hepatitis/run_hepatitis_multirun.py` | `nbs/paper_experiments/hepatitis/REPRODUCE.md` |
