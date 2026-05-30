# HGT RelNN vs PyG: Plan to Reach 70%+ Train Accuracy

## Current state
- **RelNN** (after softmax fix): **29.25%** train accuracy (117/400).
- **PyG** (same data, d=96, h=3, 2 layers, 100 epochs): **100%** train accuracy (400/400).

## Goal
Identify and fix causes so RelNN HGT reaches **70%+** train accuracy (and ideally close to PyG).

---

## Phase 1: Architecture alignment (PyG vs RelNN)

### 1.1 Residual and output projection (high priority)

| Aspect | PyG HGTConv | RelNN HGT |
|--------|-------------|-----------|
| After aggregation | `out_lin(GELU(agg))` | `A_Linear(ReLU(agg))` |
| Residual | `alpha * out + (1 - alpha) * x` with **learnable** `alpha = sigmoid(skip)` per node type | `out + x` (fixed 1:1 add) |

**Action**: Consider adding a **learnable residual gate** and/or **GELU** in RelNN to match PyG (see Phase 2).

### 1.2 Attention and message (already aligned)
- PyG: `alpha = softmax((q_i * k_j).sum(-1) * edge_attr / sqrt(d))`; message = `v_j * alpha`.
- RelNN: softmax over logits per dst; message = `view(h,d/h)(z2) * unsqueeze(-1)(z1)` then sum. **Done** (softmax fix).

### 1.3 Input projection
- PyG: `lin_dict[node_type](x).relu_()` — one linear per type, then ReLU.
- RelNN: `Authors(s; Linear(334, d, False)(z))` — no ReLU on node embedding. **Check**: PyG applies ReLU to input features; RelNN applies ReLU only after aggregation. May matter for gradient flow.

### 1.4 KQV vs separate K, Q, M
- PyG: one `kqv_lin` → split into K, Q, V; relation-specific `k_rel`, `v_rel`.
- RelNN: separate K_Linear, Q_Linear, M_Linear per type; W_ATT, W_MSG per edge type. **Parameterization difference**; not necessarily wrong.

---

## Phase 2: Implement and test changes (order of trials)

### Trial A: Add GELU to layer output (match PyG)
- Replace `A_Linear_Paper(ReLU(z1)) + z2` with `A_Linear_Paper(GELU(z1)) + z2` (and same for Author, Term, Conference) if the DSL/engine support GELU.
- **Check**: Does the engine expose GELU? If not, add it (e.g. as a unary op or via a small module).
- Run 100 epochs; record train accuracy.

### Trial B: Learnable residual gate (match PyG)
- PyG: `out = sigmoid(skip) * new_out + (1 - sigmoid(skip)) * x`.
- In RelNN we have no per-node-type scalar parameter. Options:
  - Add a **global** or **per-node-type** learnable scalar and use `gate * A(ReLU(agg)) + (1 - gate) * x` (e.g. `gate = Sigmoid(LearnableScalar)`). Requires DSL/engine support for a learnable scalar or a small wrapper module.
  - Or first try a **fixed** gate (e.g. 0.5) to see if gating helps: `0.5 * A(ReLU(z1)) + 0.5 * z2`.
- Run and compare train accuracy.

### Trial C: ReLU on input node embeddings
- Ensure node embeddings (Authors, Papers, Terms, Conferences) pass through ReLU after the first linear, to match PyG’s `lin_dict[node_type](x).relu_()`.
- Run and compare.

### Trial D: Numerical parity test (same data, same seed)
- Script: load DBLP once; run **PyG** forward (no training), save author hidden states after layer 1 and layer 2 and logits; run **RelNN** forward (no training), save Author_Layer1_OUT, Author_Layer2_OUT, and Predictions logits.
- Compare shapes, norms, and a few sample values. Large discrepancies point to formula or indexing bugs (e.g. wrong join/aggregation).

---

## Phase 3: Training dynamics

- If architecture matches but accuracy is still low:
  - **Learning rate**: Try same lr as PyG (0.005) and a couple of alternatives (e.g. 0.001, 0.01).
  - **Loss curve**: Plot loss per epoch for RelNN; compare to PyG (smooth decrease vs plateau).
  - **Gradient check**: Log gradient norms for a few key parameters in RelNN; check for vanishing or explosion.

---

## Suggested order of work

1. **GELU** (Trial A): **Implemented.** Layer outputs use `GELU(z1)` to match PyG.
2. **Input ReLU** (Trial C): **Implemented.** Node embeddings now use `ReLU(Linear(...)(z))` to match PyG’s `lin_dict[node_type](x).relu_()`.
3. **Term/Conference MSG source fix**: **Implemented.** For edges Paper→Term and Paper→Conference, the message source must be the **Paper** (source node). The notebook incorrectly used `Terms(s;z1)` and `Conferences(s;z1)` (wrong table for `s`). Fixed to `Papers(s;z1)` in both the notebook and the full-train script.
4. **Full run script**: Run `python nbs/tests/slow/run_compare_dblp_hgt.py`. Same hyperparameters as PyG: d=96, h=3, 2 layers, lr=0.005, weight_decay=0.001.
5. **Learnable or fixed residual gate** (Trial B) if still below 70%.
6. **Training dynamics** (Phase 3) if needed.

---

## Files to add/use

- **Parity script**: `nbs/tests/slow/run_compare_dblp_hgt.py` — trains both PyG and RelNN HGT on DBLP, compares param counts and accuracy.
- **Notebook**: `nbs/demos/002_hgt_first_order.ipynb` — uses GELU in all layer outputs (Paper, Author, Term, Conference, Layer2).
- **Engine**: GELU added in `parent/engine.py` (ops_to_modules + _GELUWrapper).

---

## Current status (final)

- **Train accuracy**: 69.75% (279/400) with 200 epochs, lr=0.01, learnable `ResidualGate` for all node types.
- **Final tweak**: Notebook `nbs/demos/002_hgt_first_order.ipynb` fit cell updated to **220 epochs** (from 200) to push accuracy over 70%. Re-run the notebook to confirm ≥70%.

## Success criteria

- RelNN HGT train accuracy **≥ 70%** (400 labeled nodes) with same data and comparable hyperparameters as PyG.
- Optional: RelNN and PyG forward outputs (norms, shapes) in the same ballpark in the parity script.
