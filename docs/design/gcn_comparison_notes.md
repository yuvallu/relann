# GCN Comparison Notes: RelNN vs PyG vs Hand-rolled

**Date**: 2026-02-25
**Script**: `nbs/tests/slow/run_compare_cora_pytorch.py`

## Final Results (100 epochs, seed=42, weight-synced init, no final ReLU)

| Model                        | Params | Test Accuracy | Loss @ Epoch 100 |
|------------------------------|--------|---------------|-------------------|
| Hand-rolled PyTorch          | 23,040 | **81.6%**     | 0.4718            |
| PyG GCNConv (synced init)    | 23,040 | **81.6%**     | 0.4718            |
| RelNN Templated (synced init)| 23,040 | **81.6%**     | 0.4718            |

All three models produce **identical loss curves** and **identical final accuracy**
when starting from the same weights. This definitively proves:

1. **Normalization equivalence**: hand-rolled `D̂^{-1/2} Â D̂^{-1/2}` == PyG's
   `gcn_norm` == RelNN's pre-computed Citation weights.
2. **Full training equivalence**: gradients, optimizer updates, and convergence
   are identical across all three implementations.
3. **Forward pass equivalence**: weight-synced forward comparison shows
   `max_diff = 1.16e-09` (float precision).

## Architecture: Standard 2-Layer GCN (no final ReLU)

```
Layer 1: Linear(1433 → 16, bias=False) → Aggregate(D̂^{-1/2} Â D̂^{-1/2}) → ReLU
Layer 2: Linear(16 → 7, bias=False)    → Aggregate(D̂^{-1/2} Â D̂^{-1/2})
                                                      ↑ NO activation on output
```

The output layer produces raw logits for `CrossEntropyLoss`. This matches PyG's
standard GCN usage pattern (activation between layers, not after the final layer).

## Previous Bug: Final ReLU on Output Layer

Before this fix, all three models applied `ReLU` after the output layer. This:
- Clipped negative logits to 0, reducing the model's ability to express
  "definitely not this class" during softmax.
- Caused accuracy differences between models due to initialization sensitivity:
  `nn.Linear` (Kaiming) vs `GCNConv` (Glorot/Xavier) produced different starting
  weights, leading to different convergence with only 100 epochs.

| Model (old, with final ReLU) | Accuracy | Init Scheme     |
|------------------------------|----------|-----------------|
| Hand-rolled                  | 82.2%    | Kaiming uniform |
| PyG GCNConv                  | 75.6%    | Glorot uniform  |
| RelNN                        | 82.2%    | Kaiming uniform |

The 6.6% gap between PyG and hand-rolled was entirely due to the interaction of
the non-standard final ReLU with different weight initialization schemes, NOT a
normalization or architecture difference.

## How Weight Sync Works

1. Create hand-rolled model (`nn.Linear`, Kaiming init) with `full_seed(42)`.
2. Copy `lin1.weight` → `PyGGCN.conv1.lin.weight` and `lin2.weight` → `conv2.lin.weight`.
3. For RelNN: define the model, run a `?pred` to compile parameters, then
   copy weights into `parameter_store` entries (found by shape matching).
4. Set `full_seed(42)` before each training run for deterministic Adam.

## RelNN GCN DSL (standard formulation)

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

`GCNLayer` is a pure Linear + Aggregate block. The `GCN` template applies `ReLU`
between layers but NOT after the output — the caller decides activation policy.
