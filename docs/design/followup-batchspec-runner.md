# Follow-up PR — BatchSpec runner: wiring `load_by_keys` through the engine

**Status:** designed, not implemented
**Depends on:** PR #47 (`feature/encode-decode-nn-modules`) — provides the
`RelationSource` ABC, `load_by_keys` on `DataFrameSource` and table-mode
`SqlSource`, and lazy materialisation in `Engine.__init__`.
**Goal:** make RelNN viable on relations that don't fit in memory by sampling
keys per batch instead of loading the full table.

---

## 1. What ships in PR #47 (already done)

- `RelationSource.load_by_keys(key_col, keys)` is defined on the ABC.
- `DataFrameSource.load_by_keys` implemented (pandas `isin` mask + tensor slice).
- `SqlSource.load_by_keys` implemented for `table=` mode via SQLAlchemy Core
  (`select(table).where(col.in_(keys))`); raises `NotImplementedError` for
  `query=` mode.
- `Engine.__init__` is lazy: `RelationSource` entries are kept as objects
  until first reference, then materialised once via `load_full()` and cached
  in-place in `self.db`.
- Each load (`load_full` / `load_by_keys`) bumps `data_version` so
  `_ColumnExtractModule`'s composite cache key invalidates correctly.

What's missing: **no engine code calls `load_by_keys`.** `_collect_data_sources`
only goes through `_materialise_if_source` → `load_full()`. So even if a user
constructs a `BatchSpec`, nothing consumes it.

## 2. What this follow-up PR delivers

### 2.1 `BatchSpec` dataclass

```python
# relann/data_sources.py (or a new relann/batch_spec.py)
from dataclasses import dataclass, field
from typing import Sequence, Literal

@dataclass(frozen=True)
class BatchSpec:
    anchor_relation: str               # e.g. "Papers"
    anchor_key_col: str                # primary key column on anchor_relation
    anchor_keys: Sequence              # the K keys in this minibatch
    hops: int = 1                      # how many join hops to expand
    sampler: Literal["node", "per_relation"] = "node"
    # Optional: cap fan-out per hop (per-edge sampling); None = no cap
    max_neighbours_per_hop: int | None = None
```

The two sampler modes:
- **`node`** (default): GNN-style. Anchor K nodes; for each join in the term
  graph, pull the FK values from the loaded anchor rows and `load_by_keys` the
  joined relation by those FKs. Closure walked `hops` times. Correct for rules
  with joins.
- **`per_relation`**: each relation is sampled independently from the anchor
  set. Cheaper to compute, but join-correctness is not preserved unless the
  user pre-aligns keys.

### 2.2 Term-graph planner

```python
# relann/batch_planner.py (new)
def plan_minibatch(
    engine: Engine,
    ground_sub_tg: nx.DiGraph,
    batch_spec: BatchSpec,
) -> dict[str, dict]:
    """
    Walk ``ground_sub_tg`` from the anchor relation outward and call
    ``source.load_by_keys`` per relation with the propagated key set.
    Returns ``{relation_name: er_dict}`` ready to feed into ``module.instantiate``.
    """
```

Algorithm sketch (node sampling):

1. Materialise the anchor: `engine.db[batch_spec.anchor_relation].load_by_keys(anchor_key_col, anchor_keys)`.
2. Topologically walk the term graph downstream from the anchor:
   - **Join (`,`)** with relation `R` on key `k`:
     - Collect `k`-values from rows already loaded for the join's other side.
     - Call `R.load_by_keys(k, those_values)`.
   - **Union (`|`)** of two relations:
     - Load both with the same key set.
   - **Aggregation / Transformation**: pass through (no extra load).
3. Stop after `hops` join expansions (deeper joins are not loaded; the rule
   becomes a "boundary node" that uses zero embeddings — same as PyG's
   `num_neighbors=0`).

### 2.3 Execution wire-through

Two integration points:

```python
# relann/era_operations.py — DataLoader.forward
def forward(self, sons=None, ctx: Optional[ExecutionContext] = None):
    payload = ctx.relations[self.name]   # already substituted by planner
    return _to_er_dict(payload)

# relann/engine.py — Engine.fit / Engine.predict
def fit(self, fit_stmt, *, batch_spec: BatchSpec | None = None):
    ...
    if batch_spec is None:
        data_sources = self._collect_data_sources(ground_sub_tg)
    else:
        from relann.batch_planner import plan_minibatch
        data_sources = plan_minibatch(self, ground_sub_tg, batch_spec)
    module.instantiate(data_sources)
    ...
```

`Session.fit(...)` and `Session.predict(...)` grow a `batch_spec=None` kwarg
that threads down. `batch_spec=None` keeps the current full-load behavior —
zero hot-path impact for users who don't opt in.

### 2.4 Cache invalidation

Each `load_by_keys` call already bumps `data_version`. `_ColumnExtractModule`'s
cache key includes `data_version`, so the composite key changes per batch and
the cache is automatically invalidated. **No additional code needed**, just a
test that asserts categorical codes / numeric tensors are correctly recomputed
between batches.

## 3. Test plan

| # | Test | Verifies |
|---|------|----------|
| 1 | `test_batch_spec_anchor_only_loads_anchor_rows` | `BatchSpec` with `hops=0` only loads anchor rows; other relations untouched. |
| 2 | `test_batch_spec_one_hop_join_propagates_fk_values` | Join on `(uid, edge_uid)` correctly loads `Edges` rows with `src in anchor_keys`. |
| 3 | `test_batch_spec_two_hop_node_sampling` | 2-hop GNN: anchor → 1-hop neighbours → 2-hop neighbours; depth bound respected. |
| 4 | `test_batch_spec_per_relation_sampler` | Independent sampling mode; assert correct shapes (no join-correctness assertion). |
| 5 | `test_batch_spec_query_mode_sql_source_raises` | `BatchSpec` against a `SqlSource(query=...)` source raises a clear error pre-flight, not mid-loop. |
| 6 | `test_batch_spec_categorical_codes_stable_across_batches` | Same label gets the same code across two `BatchSpec` runs — already covered by `data_version` + canonical vocab; just needs an explicit minibatching test. |
| 7 | `test_batch_spec_loss_finite_across_minibatches` | E2E: 5 minibatches, MSELoss; assert loss stays finite and decreases or plateaus. |
| 8 | `test_batch_spec_full_vs_minibatch_equivalence` | Run the same model with `batch_spec=None` and with a `BatchSpec` covering all keys; assert outputs are equal (within `torch.allclose` tolerance). |

## 4. Risks / things to watch

- **Join correctness in `per_relation` mode**: documented, tested for shape
  but not for join-correctness (because it isn't). Make the docstring loud
  about this.
- **Aggregation across batches**: rules like `Agg(dst; sum(z))` aggregate
  *within* a batch. If the same `dst` appears in two batches, the user gets
  two partial sums, not one. This is a known GNN-vs-full-batch semantic and
  matches PyG/DGL behavior. Document.
- **Fixed `column_vocabs` across batches**: `_ColumnExtractModule` already
  reads `er.column_vocabs`. The planner must propagate the *full* source's
  `column_vocabs` (not the slice's) so codes stay stable. `SqlSource` already
  caches `_cached_column_vocabs` from the first full load — the planner
  should call `load_full()` once for vocab building if `_cached_column_vocabs`
  is `None`, then use `load_by_keys` for actual rows. (Add a
  `RelationSource.prime_vocabs()` helper for clarity.)
- **`SqlSource(query=...)` is not minibatchable**: the planner must verify
  every relevant source supports `load_by_keys` before starting; raise
  pre-flight, not mid-loop.

## 4a. Known optimizer interaction (separate small follow-up)

After merging `origin/main` into this branch, four SQL-edge-table tests began
failing because the V1 e-graph optimizer (merged in main as PR #52/#53)
inserts a `Transformation` around aggregations on `DataLoader`s that have
**no embeddings**. This is a new pattern only enabled by `RelationSource` —
SQL pure-edge tables loaded via `SqlSource(..., table=...)` carry rows but
no embedding tensor. `Transformation.instantiate` rejects sons with neither
`embedding_shapes` nor encode leaves.

Workaround in this PR: the four affected tests set
`session.engine._enable_optimizations_for_predict = False`. The encode-only
fallback in `Transformation.instantiate` handles this case correctly; only
the optimizer's rewrite trips on it.

Affected tests (all in `nbs/tests/feature/test_data_sources_sql.py`):
- `test_sql_source_session_predict_with_argmax_decode`
- `test_sql_source_used_directly_as_edge_table`
- `test_sql_source_lazy_in_rule_path`
- `test_engine_mixed_relation_source_and_legacy_tuple`

Permanent fix (small, separate PR): the optimizer's lift/extract pass should
detect `DataLoader`s with empty `embedding_shapes` and skip the synthesised
`Transformation` wrap, leaving the original
`Aggregation(... DataLoader(edges) ...)` shape intact. Alternative: relax
`Transformation.instantiate` to treat a transformation containing only
`_InputSelector` content as a pure pass-through. The first option is
cleaner (the optimizer is the new constraint); the second risks masking
genuine "missing embeddings" errors elsewhere.

## 5. Out of scope for this follow-up

- Distributed sampling / cross-machine batching.
- Negative sampling for link prediction (a different kind of sampler).
- Adaptive batch sizes.
- The type-design refactors from the original CR (typed ER-dict,
  discriminated `EncodeItem`/`ContentDecode`, `_ColumnExtractModule` context
  object) — separate PR.

## 6. Estimated effort

- `BatchSpec` dataclass + tests: ~0.5 day
- Planner (node sampler, 1–2 hops): ~1 day
- Engine + DataLoader wiring: ~0.5 day
- E2E test on Cora-from-SQLite (already in PR #47): ~0.5 day
- Documentation + edge cases (per-relation sampler, vocab priming): ~1 day

**~3 person-days end-to-end.** Most of the risk is in the planner's join-edge
walk; the source layer and cache invalidation are already in place from PR #47.
