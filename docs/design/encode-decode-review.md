# Encode/Decode Design Review — PR #47

**Branch:** `feature/encode-decode-nn-modules`  
**Reviewer:** AI agent (Cursor)  
**Status:** awaiting author sign-off before Phase 2a/2b implementation begins

---

## 1. Correctness audit

### 1.1 Parse → AST path

Grammar additions in `parent/relann_grammar.lark`:

```
content_encode : "[" encode_item ("," encode_item)* "]"
content_decode : "[" CNAME ("(" arith_term* ")")? "]"
encode_item    : CNAME                                # bare column
               | tensor_term "(" CNAME ")"            # Encoder(col)
               | tensor_term "(" arith_terms ")" "(" CNAME ")"  # Encoder(hp)(col)
```

Transformer (`parent/parser.py`) builds `ContentEncode` / `ContentDecode` / `EncodeItem` Pydantic objects (added in `parent/pydantic_classes.py`). These attach to `LHSRelation.derived_content_attrs`. **No issues found here.**

### 1.2 Compilation path (RHS encode)

`TensorTermCompiler.compile_tensor_term` in `parent/tensor_term_compiler.py` recognises a `ContentEncode` node and builds:

- **bare `[col]`** → `_ColumnExtractModule(column_name)` — auto-tensorizes via `tensorize_column` in `parent/encode.py`.
- **`[Encoder(col)]`** → `_EncodeWrapper(_ColumnExtractModule, instantiated_encoder)`.
- **`[a, b, Enc(c)]`** → `_MultiEncodeModule([mod_a, mod_b, mod_c])` which cats on last dim.

`Transformation.instantiate` wires the leaf extractors via `_inject_encode_source(son)`, then runs a 1-row dummy forward to infer output shapes. `Transformation.forward` calls `_inject_encode_source` again before the real forward. **No logic errors found;** all three entry points (`instantiate`, `forward`) correctly call `_inject_encode_source`.

### 1.3 LHS decode path (`?pred`)

`Engine._apply_lhs_decode` (engine.py line 2355) collects `ContentDecode` attrs from `lhs.derived_content_attrs`, always uses `predictions.embeddings[0]`, and writes decoded numpy arrays back into a copy of `predictions.content`. Result is wrapped in a new `EmbeddedRelation`. Called from `Engine.predict` (line 2448) inside `torch.no_grad()` after `module.forward()`. **Functionally correct for `?pred` use.**

### 1.4 `tensorize_column` (`parent/encode.py`)

- bool → float32 (−1,1) with `view(-1,1)` ✓
- category → int64 codes from `series.cat.codes` ✓
- numeric → float32 with `view(-1,1)` ✓
- datetime/timedelta/complex → `EncodeTypeError` ✓
- text/object → `EncodeTypeError` ✓ (caller must use an encoder wrapper)

---

## 2. Scalability / design issues

### Issue A — `id(df)` cache key is minibatching-incompatible (HIGH)

**Location:** `_ColumnExtractModule.forward` in `parent/tensor_term_compiler.py`

```python
cache_key = id(df)
if self._cached_result is not None and self._cache_key == cache_key:
    raw = self._cached_result
```

`id(df)` is the Python object identity of the DataFrame. This works well for full-batch training (same DataFrame object every epoch = cache always hits). It breaks silently for minibatching in two ways:

1. Each mini-batch is a new DataFrame slice → `id` never matches → re-tensorizes every step, defeating the cache entirely.
2. Python can reuse `id` values for deallocated objects. A stale cache hit after GC would silently return the previous batch's data.

**Proposed fix (Phase 2a):** Replace the cache key with a tuple `(id(df), len(df), df.index[0] if len(df) > 0 else None)`. Add an explicit `invalidate()` method on `_ColumnExtractModule` so a future `BatchSpec` runner can call it per step. The cache still provides its epoch-level benefit for full-batch training.

### Issue B — Vocab ownership is per-extractor, not per-source (HIGH)

**Location:** `_ColumnExtractModule._vocab` in `parent/tensor_term_compiler.py`

The vocabulary mapping `{category_label → int}` is built on the first `tensorize_column` call and stored on the extractor module. Two problems:

1. **Cross-batch drift:** when a mini-batch arrives as a fresh `pd.Categorical` series with a different set of `.cat.categories`, `tensorize_column` receives the old `self._vocab` as the `vocab` parameter, but the actual encoding uses `series.cat.codes` — which reflects the series' *own* category ordering, not the stored vocab. If batch 1 trains `Embedding(3, dim)` with categories `{A→0, B→1, C→2}` and batch 2 presents `{A→0, B→1, D→2}`, the index for D is silently interpreted as the index for C by the embedding layer.
2. **Predict-time mismatch:** at predict time a new `Session` could rebuild a different vocab mapping than the training run, causing subtle label-shift bugs.

**Proposed fix (Phase 2b):** Move vocab ownership to the `RelationSource`. Each source builds its vocabulary once during `load_full()` and exposes a `vocab(column)` accessor. `_ColumnExtractModule` receives the vocab at construction time (injected alongside `_source_er`), not built lazily.

### Issue C — `_apply_lhs_decode` silently ignores multi-embedding output (MEDIUM)

**Location:** `Engine._apply_lhs_decode`, line 2368

```python
emb = predictions.embeddings[0]  # always uses first embedding, no guard
```

If a rule produces two embeddings (e.g., a join result), only the first is decoded. The second is silently ignored. If a rule produces zero embeddings, a `ValueError` is raised — good. But the zero-check message says "check that the rule produces an embedding tensor" without telling you how many it found.

**Proposed fix (Phase 2a):** Assert `len(predictions.embeddings) == 1` and give a clear error if more than one embedding is present.

### Issue D — `_apply_lhs_decode` can be called outside `?pred` (LOW)

The function is decorated `@patch` on `Engine` and has no guard against being invoked outside `predict()`. It does nothing if there are no `ContentDecode` attrs (early return), so in practice it's harmless. But an erroneous call mid-graph (future decode-as-an-operator feature) would write into a copy and not propagate correctly.

**Proposed fix (Phase 2a):** Add a `_is_predict_context: bool` parameter, default `True`. Raise `NotImplementedError("decode mid-graph not yet supported")` if `False`.

### Issue E — `EncodeTypeError` fires at forward time for text columns (LOW)

Text columns that lack an encoder are only caught during `Transformation.instantiate` via the dummy-run → non-tensor output path. This is a runtime error at graph instantiation (before training begins), so in practice the user sees it quickly. However, the dummy run consumes some compute, and the error message is assembled by scanning extractor leaves post-hoc.

**Proposed fix (Phase 2a):** In `Transformation.instantiate`, before the dummy run, walk `collect_column_extract_leaves` and check whether any bare-column leaf points to a text series with no wrapping encoder; raise `EncodeTypeError` eagerly with the column name. Eliminates the need to inspect `_last_was_text` flags.

### Issue F — `Engine.db` has no seam for deferred / SQL loading (HIGH, addressed in Phase 2b)

`Engine._collect_data_sources` always calls `_to_er_dict(self.db[rel_name])`, which requires `self.db[rel_name]` to already be a fully-materialized `(df, tensor)` pair. There is no abstraction layer between "how a relation's content is sourced" and "what the runtime sees". Adding SQL loading by adding more cases to `_to_er_dict` would further inflate its already-6-branch `isinstance` chain.

**Proposed fix (Phase 2b):** Introduce `parent/data_sources.py::RelationSource` ABC. `Engine.__init__` calls `source.load_full()` for each source and stores the resulting dict in `self.db`. `_to_er_dict` stays unchanged; everything downstream is transparent.

### Issue G — Cache invalidation at `instantiate` is one-sided (LOW)

`_reset_encode_caches_for_instantiate` clears all extract-leaf caches before the dummy run. That prevents the 1-row dummy result from polluting the real-data cache. But it means the very first real `forward` always re-tensorizes even if the data hasn't changed since a previous call to `instantiate`. For large tables this is an extra pass. **Not worth fixing now** — noted for when a `module.reset_and_reuse(new_data)` API is designed.

---

## 3. Proposed `RelationSource` abstraction

New file: `parent/data_sources.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd


class RelationSource(ABC):
    """Abstraction over how a named relation is loaded into the RelNN runtime."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def schema(self) -> List[str]:
        """Column names of the content table (not including embedding columns)."""
        ...

    @abstractmethod
    def load_full(self) -> Dict[str, Any]:
        """Return a fully-materialized ER dict: {content, content_schema, embedding_shapes, embeddings}."""
        ...

    def load_by_keys(self, key_col: str, keys) -> Dict[str, Any]:
        """
        Return only the rows whose `key_col` is in `keys`.
        Default: raise NotImplementedError — implement for minibatching support.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support key-based loading. "
            "Implement load_by_keys() to enable GNN-style node sampling."
        )
```

### 3.1 `DataFrameSource`

Wraps the existing `(df, tensor)` tuple. `load_full` delegates to `_to_er_dict`. Vocab built once and stored per column. Backward-compatible with `db={name: (df, tensor)}` (auto-wrapped in `Engine.__init__`).

### 3.2 `SqlSource`

```python
SqlSource(
    name: str,
    engine_or_url: str | sqlalchemy.Engine,
    table: str | None = None,
    query: str | None = None,    # overrides table if given
    primary_key: str | None = None,
    dtype_map: dict | None = None,  # {col: pd_dtype} post-load casts
)
```

`load_full` calls `pd.read_sql_table` / `pd.read_sql_query`. No embeddings at load time (embeddings are computed by the rules). `load_by_keys` calls `pd.read_sql_query(f"SELECT * FROM {table} WHERE {pk} IN ({keys})")` — reserved for Phase 3 (minibatching).

SQLAlchemy is used so the same code works with SQLite, Postgres, MySQL, DuckDB. Required dep: `sqlalchemy>=2.0`.

---

## 4. Minibatching seam design (reserved, not implemented in this PR)

The following is designed now and reserved in the interface. **No code ships for this in this PR.**

### 4.1 BatchSpec

```python
@dataclass
class BatchSpec:
    anchor_relation: str      # which relation is sampled (e.g. "Papers")
    anchor_key_col: str       # primary key column
    anchor_keys: list         # the K keys in this mini-batch
    hops: int = 1             # how many join hops to include
    sampler: str = "node"     # "node" (GNN-style closure) | "per_relation" (independent)
```

### 4.2 Execution-time change

`DataLoader.forward(ctx, batch_spec=None)` — if `batch_spec` is not None, call `ctx.sources[name].load_by_keys(...)` instead of using the pre-loaded dict. This is the only change to the hot path; when `batch_spec is None` the current full-load path is taken unchanged.

### 4.3 Planner (future)

A topological walk of the term graph propagates key sets along join edges:

- Anchor relation: `load_by_keys(pk, anchor_keys)`.
- Joined relation: collect FK values from the anchor batch → `load_by_keys(fk_col, fk_values)`.
- Union: union of both children's key sets.

Supports both sampler modes (node-sampling and per-relation random) as a pluggable `Sampler` protocol on `BatchSpec`.

---

## 5. Demo architecture — DBLP from SQLite

| File | Purpose |
|------|---------|
| `scripts/data_setup/dblp_from_sqlite/build_dblp_sqlite.py` | One-time script. Loads DBLP via PyG (same as `run_compare_dblp_original_hgt.py`), writes node/edge tables to `dblp.sqlite` with `df.to_sql(table, engine, if_exists="replace", index=False)`. |
| `examples/008_relnn_dblp_sqlite.py` | Opens `sqlite:///dblp.sqlite`, creates `SqlSource` for each table, builds a `Session(db=sources)`, runs the HGT RelNN program, prints accuracy. |

**Write-back** (`?pred` → SQL table) is described here for future reference but **not implemented** in this PR. It would be a post-`predict()` hook: `session.relation("Out").content.to_sql("Out", engine, if_exists="replace")`.

---

## 6. Summary of changes by phase

### Phase 2a — Hardening (no new modules)

| Fix | File(s) | Notebook |
|-----|---------|----------|
| A: Better cache key + `invalidate()` | `parent/tensor_term_compiler.py` | not notebook-backed |
| B: Vocab ownership deferred to Phase 2b | — | — |
| C: Multi-embedding guard in decode | `parent/engine.py` | `nbs/021_engine.ipynb` |
| D: Context guard in `_apply_lhs_decode` | `parent/engine.py` | `nbs/021_engine.ipynb` |
| E: Eager text-column check in `Transformation.instantiate` | `parent/era_operations.py` | `nbs/011_embedded_RA_operations.ipynb` |

### Phase 2b — SQL source + demo

| Deliverable | New/Modified |
|-------------|-------------|
| `parent/data_sources.py` | New |
| `parent/engine.py` — accept `RelationSource` in `__init__` | Modified |
| `parent/session.py` — no change needed (delegates to Engine) | — |
| `nbs/021_engine.ipynb` | Modified |
| `nbs/tests/feature/test_data_sources_dataframe.py` | New |
| `nbs/tests/feature/test_data_sources_sql.py` | New |
| `scripts/data_setup/dblp_from_sqlite/` | New |
| `docs/design/data-sources.md` | New |
| `requirements.txt` — add `sqlalchemy>=2.0` | Modified |

---

**Please review this document and approve (or request changes) before Phase 2a implementation begins.**
