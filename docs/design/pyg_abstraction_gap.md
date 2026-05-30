# Why "users must drop to raw index code" — the PyG abstraction gap

This document explains the core claim that PyG (PyTorch Geometric) automates
index management for standard message-passing GNNs, but provides **no such
abstraction** for architectures like DHN or HyGNN — forcing users to write
raw index-manipulation code in plain PyTorch.

## What PyG automates (the happy path)

For a standard GCN, PyG's `MessagePassing` base class handles everything:

```python
class GCNConv(MessagePassing):
    def __init__(self, in_ch, out_ch):
        super().__init__(aggr='add')
        self.lin = Linear(in_ch, out_ch)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=self.lin(x))

    def message(self, x_j):
        return x_j
```

**Zero index code.** You define the message; `propagate()` internally does:

1. `x_j = x[edge_index[0]]` — gather source features by index
2. Calls your `message(x_j)`
3. `scatter_add(msg, edge_index[1], dim=0)` — scatter-aggregate at targets

You never see `scatter_add`, `unsqueeze`, `expand_as`, or manual index tensors.
PyG's `DataLoader` also handles **batching** — offsetting node indices when
multiple graphs are packed into a single batch.

## What breaks: the DHN case (multi-way patterns)

DHN aggregates over **cycle homomorphisms** (triangles, 4-cycles, ..., 8-cycles).
A triangle involves 3 nodes simultaneously. PyG's `MessagePassing` only handles
**pairwise edges** (2 nodes). There is no `TriangleConv` or `HomConv` in PyG.

### The official DHN implementation (raw PyTorch)

The DHN authors (Maehara & Hoang, NeurIPS 2024) built their own `HomConv` layer
on raw PyTorch. Here is their actual code (from `_external/dhn/dhn/layers.py`):

```python
class HomConv(torch.nn.Module):
    def forward(self, x, mapping_index):
        # mapping_index: (num_hom, hom_size) — pre-enumerated patterns

        product = 1
        for i, f in enumerate(self.f):
            product *= f(x[mapping_index[:, i]])          # <-- raw index gather

        output = torch.zeros(x.size(0), product.size(1)).to(x.device)  # <-- manual alloc
        output.scatter_add_(                                            # <-- raw scatter
            0,
            mapping_index[:, 0].unsqueeze(1).expand(-1, product.size(1)),  # <-- manual reshape
            product
        )
        return output
```

And their pattern enumeration (from `_external/dhn/dhn/graph_enumerations.py`):

```python
def cycle_mapping_index(nxg, length_bound=10):
    base_cycles = [*nx.simple_cycles(nxg, length_bound=length_bound)]
    index_dict = defaultdict(list)
    for c in base_cycles:
        index_dict[f'c{len(c)}'].append(c)
        index_dict[f'c{len(c)}'].append([*reversed(c)])
    index_dict['c2'] = list(nxg.edges())
    result = dict()
    for k, v in index_dict.items():
        result[k] = torch.tensor(
            np.vstack([np.roll(v, i, axis=1) for i in range(1, int(k[1:])+1)])
        ).long()
    return result
```

### Line-by-line: where the raw index code lives

| Line | What it does | PyG equivalent for GCN |
|------|-------------|----------------------|
| `x[mapping_index[:, i]]` | Gather features for the i-th node of each pattern instance | `propagate()` does this automatically |
| `torch.zeros(x.size(0), ...)` | Allocate output tensor | `propagate()` handles this |
| `.scatter_add_(0, ...)` | Aggregate pattern results back to nodes | `aggr='add'` — one keyword |
| `mapping_index[:, 0].unsqueeze(1).expand(-1, ...)` | Reshape index to match feature dims | Hidden inside `propagate()` |
| `nx.simple_cycles(nxg, ...)` | Enumerate all cycles | No equivalent — PyG has no pattern enumeration |
| `np.vstack([np.roll(...)])` | Generate rooted variants of each cycle | No equivalent |

Every one of these operations is something PyG automates for pairwise edges
but **does not automate** for multi-way patterns. The user must write them
from scratch.

### Batching is also manual

In PyG, `DataLoader` automatically increments `edge_index` per graph in a batch.
For DHN, the authors must manually track `batch_idx` and scatter across the
batch dimension themselves:

```python
# From _external/dhn/dhn/models.py — manual batch scatter
batch_agg = torch.zeros((batch_size, self.out_dim), dtype=h.dtype, device=h.device)
batch_agg.scatter_add_(0, batch_idx.unsqueeze(-1).expand(-1, self.out_dim), h)
```

## What RelNN looks like for the same thing

The same C3 (triangle) aggregation in RelNN (from `dhn_C2_8_templated.relnn`):

```relnn
C3_T0(graph_id, u, v, w; Mu_L3<'C3', 0>(ReLU()(Mu_L2<'C3', 0>(ReLU()(Mu_L1<'C3', 0>(z))))) * wh)
    :- Hom_C3(graph_id, u, v, w; wh), H0(graph_id, u; z) .

C3_T1(graph_id, u, v, w; Mu_L3<'C3', 1>(ReLU()(Mu_L2<'C3', 1>(ReLU()(Mu_L1<'C3', 1>(z))))))
    :- Hom_C3(graph_id, u, v, w; _), H0(graph_id, v; z) .

C3_T2(graph_id, u, v, w; Mu_L3<'C3', 2>(ReLU()(Mu_L2<'C3', 2>(ReLU()(Mu_L1<'C3', 2>(z))))))
    :- Hom_C3(graph_id, u, v, w; _), H0(graph_id, w; z) .

C3_Agg(graph_id, u; sum(z0 * z1 * z2))
    :- C3_T0(graph_id, u, v, w; z0),
       C3_T1(graph_id, u, v, w; z1),
       C3_T2(graph_id, u, v, w; z2) .
```

The join on `(graph_id, u, v, w)` **is** the pattern enumeration.
The `sum(...)` **is** the scatter aggregation.
The engine compiles all index management, gather, scatter, and batching automatically.

No `scatter_add_`, no `unsqueeze`, no `expand`, no `torch.zeros`, no `nx.simple_cycles`.

## The HyGNN case

PyG provides `HypergraphConv`, but it only implements basic HGNN (Feng et al. 2019):
`X' = D^{-1} H W B^{-1} H^T X Θ` — a fixed incidence-matrix convolution.

HyGNN (Saifuddin et al. 2022) has a different architecture: an **attention-based
hyperedge encoder** that computes attention over variable-sized node sets within
each hyperedge. This requires:

1. Custom attention over variable-size sets (not pairwise)
2. Separate hyperedge→node and node→hyperedge message passing
3. Manual index management for both directions

PyG's `HypergraphConv` cannot be configured to do this. A user would need to
write a new layer from scratch with raw index code.

### PyG's own hypergraph roadmap confirms the gap

- **Issue #7312** (May 2023): Roadmap for better hypergraph support. Closed
  Nov 2023 with only 2/5 items done (data container + DataLoader). The items
  "Add hypergraph GNNs" and "Add examples" were **never completed**.
- **Issue #8501** (Dec 2023): Extended roadmap for hypergraph support. Still
  **open** as of Jan 2025 with no new GNN layers added.

## Summary: the abstraction gap

| Capability | PyG provides it? | Without PyG, you need... |
|-----------|-----------------|------------------------|
| Pairwise message passing (GCN, GAT, GIN) | Yes (`MessagePassing`) | — |
| Multi-way pattern aggregation (DHN) | **No** | Raw index gather + scatter + pattern enumeration |
| Attention-based hyperedge encoding (HyGNN) | **No** (only basic HGNN) | Custom attention over variable-size sets |
| Batch index offsetting for standard graphs | Yes (`DataLoader`) | — |
| Batch handling for custom structures | **No** | Manual `scatter_add_` with `batch_idx` |

The claim is not that PyG **disallows** these architectures. It's that PyG
**provides no abstraction** for them — the user must write the same low-level
index code they would write in raw PyTorch, losing exactly the benefit that
PyG provides for standard GNNs.

In RelNN, all of these architectures — GCN, HGT, DHN, HyGNN — are expressed
at the same abstraction level: declarative queries with joins and aggregations.
The index management is always the engine's job, never the user's.
