# RelNN Runtime Optimization Opportunities

**Date**: 2026-03-29  
**Context**: Identified while analyzing why RelNN PA-path (24.2s / 100 epochs on GPU) is ~2× slower than PyG HGTConv (11.4s) despite processing 12× fewer edges (19,645 PA edges vs 239,566 full-graph edges). This implies RelNN is ~24× slower per edge, a significant gap to close.

---

## Benchmark baseline (fair GPU, 5 seeds, DBLP HGT 100 epochs)

| Implementation | Edges processed | Time (s) | Time/edge (relative) |
|---|---|---|---|
| PyG HGTConv 1L | 239,566 (full graph) | 11.4 | 1× |
| RelNN PA-path 1L | 19,645 (PA only) | 24.2 | ~24× |
| original pyHGT 1L | 239,566 (flat graph) | 40.9 | ~3.6× vs PyG |

---

## Optimization 1: Batch multi-head attention (highest impact)

**Current behavior**: Each attention head is a separate template instantiation — two independent computation graphs that run sequentially:
```relnn
PaperK<1>(paper_id; ...) :- PaperEmb(paper_id; z) .
PaperK<2>(paper_id; ...) :- PaperEmb(paper_id; z) .
```
This produces 6 separate `nn.Linear` forward passes (K, Q, V × 2 heads) instead of 3 batched ones.

**What PyG does**: A single matmul over all heads at once:
```python
k = self.k_lin(x).view(-1, H, D//H)  # shape (N, H, d/h) in one call
```

**Fix**: Add a `MultiHead<H>` or `Batched<H>` template primitive that applies a linear layer and splits the output into H heads as a single batched operation. This would halve the number of Linear calls and enable contiguous tensor access patterns.

**Estimated gain**: 2–3× reduction in Linear forward time.

---

## Optimization 2: Fuse multi-pass edge traversal

**Current behavior**: The PA attention block makes 10 separate `index_select` passes over the 19,645-edge list (per forward call):
```
DotPA<1> → ExpPA<1> → DenomPA<1> → SoftPA<1> → MsgPA<1>   (5 passes for head 1)
DotPA<2> → ExpPA<2> → DenomPA<2> → SoftPA<2> → MsgPA<2>   (5 passes for head 2)
```
Each `Join.forward()` calls `emb.index_select(0, idx)` to gather node features onto edges.

**What PyG does**: A single pass over edges using fused scatter/gather: gather source node features, compute dot product and softmax, scatter-add to target nodes — all in one C++ kernel.

**Fix options**:
- **DSL-level**: Detect the attention pattern (dot→exp→sum→div→weighted_agg) and rewrite to a single fused op at compile time (e.g., using e-graph rewrites — see `docs/design/e-graphs-future-optimizer.md`).
- **Engine-level**: Add a `ScaledDotProductAttention` built-in op to the tensor term compiler that maps to `torch.nn.functional.scaled_dot_product_attention`.
- **Manual DSL**: Expose a `softmax_agg` built-in that fuses exp+sum+div+multiply+aggregate.

**Estimated gain**: 3–5× reduction in edge-traversal time.

---

## Optimization 3: Reduce Python dispatch overhead

**Current behavior**: RelNN evaluates ~20 graph nodes sequentially in Python for the 1L PA-path HGT, each being a `nn.Module.__call__`. With only 19,645 edges, GPU kernels finish in microseconds — Python dispatch overhead dominates.

**What PyG does**: One Python call dispatching to a single C++ kernel.

**Fix**: 
- **torch.compile / TorchScript**: After instantiate, `torch.compile` the forward function over the operator graph. This eliminates Python dispatch overhead by tracing the computation and fusing it into a single compiled kernel.
- **Operator fusion at engine level**: Detect chains of ops (e.g., Linear → ReLU) and fuse them into a single `nn.Sequential` before execution.

**Estimated gain**: 2–4× for small edge counts; less impactful for large graphs where GPU time dominates Python dispatch.

---

## Optimization 4: Use scatter instead of index_select + sum

**Current behavior**: RelNN's aggregation materializes intermediate tensors explicitly:
1. `index_select` gathers node embeddings to produce an `(E, d/h)` edge tensor.
2. A separate aggregation op sums edge tensors to target nodes.

This creates `(E, d/h)` intermediate tensors that may not fit in GPU cache for large graphs.

**What PyG does**: `scatter_add` (from `torch_scatter`) accumulates directly from compact source-node embeddings to target nodes without materializing the full `(E, d/h)` edge tensor.

**Fix**: Add `scatter_add` as a built-in aggregation in `era_operations.py::Aggregation`. When the aggregation pattern is `sum(z)` and the join is a simple source→target lookup, replace `index_select + sum` with `scatter_add`.

**Estimated gain**: Reduces memory bandwidth; most impactful for graphs with high average degree.

---

## Priority order

| Priority | Optimization | Estimated gain | Complexity |
|---|---|---|---|
| 1 | Fuse multi-pass edge traversal (#2) | 3–5× | Medium |
| 2 | Batch multi-head attention (#1) | 2–3× | Medium |
| 3 | torch.compile forward (#3) | 2–4× | Low (mostly transparent) |
| 4 | scatter_add aggregation (#4) | 1.5–2× | Medium |

Applying all four could plausibly bring RelNN to within 1.5–2× of PyG on PA-path, i.e., ~6–8s vs 11.4s — at which point the PA-path scope advantage (12× fewer edges) is the dominant factor.

---

## Encode column cache (implemented)

**Where:** `parent/tensor_term_compiler.py` — `_ColumnExtractModule` caches the result of reading + tensorizing a content column (or the text `pd.Series` passthrough), keyed by `id(DataFrame)` so epochs over a stable table avoid repeated pandas extraction.

**Remaining gap:** If a `DataFrame` is mutated in place without changing object identity, the cache could be stale — a hash-based invalidation (see `docs/design/encode-decode.md` “Next steps”) would harden this.

---

## Notes

- These optimizations are **not needed for correctness** and are not planned for the current paper submission.
- The paper frames the current gap honestly: RelNN is a general-purpose relational framework, not a GNN kernel. The 2× slowdown vs PyG is the cost of generality — the same DSL expresses GCN, HGT, DHN, and arbitrary relational architectures without hand-tuned CUDA kernels.
- The e-graph rewriting approach (#2, DSL-level) is the most principled long-term solution. See `docs/design/e-graphs-future-optimizer.md`.
