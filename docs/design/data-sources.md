# RelationSource — pluggable data loading for RelNN

**Module:** `relann/data_sources.py`  
**Status:** shipped in `feature/encode-decode-nn-modules`

---

## Overview

`RelationSource` is a small ABC that decouples *how a relation's content is materialized* from *how the RelNN runtime processes it*.

Before this abstraction, `Engine.db` was always a flat dict of `{name: (pd.DataFrame, torch.Tensor)}` pairs. Adding a new source type (SQL, Parquet, Arrow, lazy remote) required teaching `_to_er_dict` yet another input format — already at 6 branches before this PR.

With `RelationSource`:

- `Engine.__init__` calls `source.load_full()` once per source at startup, producing the standard ER-dict that `_to_er_dict` and `DataLoader` already understand.
- The rest of the engine (`term_graph`, `relnn`, `era_operations`) is unchanged.
- The minibatching hook (`load_by_keys`) is reserved on the interface, so a future `BatchSpec` runner can plug in without touching Session/Engine.

---

## Interface

```python
from abc import ABC, abstractmethod

class RelationSource(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Logical relation name (must match the name used in RelNN rules)."""

    @abstractmethod
    def schema(self) -> list[str]:
        """Column names of the content table."""

    @abstractmethod
    def load_full(self) -> dict:
        """Return a fully-materialized ER-dict (see below). Called once at Engine.__init__ per source."""

    def load_by_keys(self, key_col: str, keys) -> dict:
        """Minibatching hook. Implemented on ``DataFrameSource`` and table-backed ``SqlSource``;
        default ABC implementation raises ``NotImplementedError``.
        See section 'Minibatching' below."""
        raise NotImplementedError(...)
```

### ER-dict shape (normalized)

Every `load_full()` / `load_by_keys()` result (and legacy payloads after `_to_er_dict`) includes:

| Key | Meaning |
|-----|---------|
| `content` | `pd.DataFrame` |
| `content_schema` | `list[str]` column names |
| `embedding_shapes` | `list[tuple]` per embedding tensor |
| `embeddings` | `list[torch.Tensor]` or `None` |
| `column_vocabs` | Optional `dict[str, dict[label, int]]` for **categorical** columns (stable codes for encode) |
| `data_version` | `int` incremented on each load from this source (cache invalidation for `_ColumnExtractModule`) |

---

## Provided implementations

### `DataFrameSource`

```python
from relann.data_sources import DataFrameSource

src = DataFrameSource("Users", df)                  # no embeddings
src = DataFrameSource("Users", df, embeddings=z)    # single tensor
src = DataFrameSource("Users", df, embeddings=[z1, z2])  # multi-tensor
```

- Wraps an existing `pd.DataFrame + tensor` pair.
- `load_full()` returns `{content: df, content_schema: df.columns, embedding_shapes: [...], embeddings: [...]}`
- This is what `Engine.__init__` creates internally when you pass the legacy `(df, tensor)` tuple — backward-compatible.

### `SqlSource`

```python
from relann.data_sources import SqlSource

# Table name = relation name
src = SqlSource("Papers", "sqlite:///dblp.sqlite", table="Papers")

# Custom SQL query
src = SqlSource("RecentPapers", engine, query="SELECT * FROM papers WHERE year >= 2020")

# With post-load dtype casting
src = SqlSource("Authors", url, table="authors", dtype_map={"area": "category"})
```

- Requires `sqlalchemy>=2.0` (`pip install sqlalchemy`).
- Works with any SQLAlchemy dialect: SQLite, PostgreSQL, MySQL, DuckDB, etc.
- `load_full()` calls `pd.read_sql_table` / `pd.read_sql_query` and returns no embeddings (they are computed by RelNN rules). Populates `column_vocabs` and bumps `data_version`.
- **`load_by_keys`** is implemented for **`table=`** sources (SQL `IN` filter). It is **not** supported for arbitrary **`query=`** sources (`NotImplementedError`). Uses the same categorical vocabs as the last full load when available.
- **`schema()`** for **`query=`** sources runs a **zero-row** read (`LIMIT 0` subquery) when supported by the dialect, so callers can introspect column names without loading all rows.

---

## Usage with Session

### Legacy dict (backward-compatible, no change required)

```python
session = Session(db={
    "Users": (users_df, users_tensor),
    "Edges": (edges_df, torch.ones(len(edges_df), 1)),
})
```

Legacy tuples are left as-is; `Engine._normalize_relation_payload` handles them via `_to_er_dict`.

### DataFrameSource

```python
from relann.data_sources import DataFrameSource

session = Session(db={
    "Users": DataFrameSource("Users", users_df, embeddings=users_tensor),
    "Edges": DataFrameSource("Edges", edges_df),
})
```

### SqlSource

```python
from relann.data_sources import SqlSource

session = Session(db={
    "Papers": SqlSource("Papers", "sqlite:///dblp.sqlite", table="Papers"),
    "Cites":  SqlSource("Cites",  "sqlite:///dblp.sqlite", table="Cites"),
})
```

### Mixed (SQL + DataFrame)

```python
session = Session(db={
    "Author": DataFrameSource("Author", author_df, embeddings=author_features),  # pre-built features
    "AuthorPaper": SqlSource("AuthorPaper", url, table="AuthorPaper"),           # edge table from SQL
    "AuthorLabels": DataFrameSource("AuthorLabels", labels_df, embeddings=labels_tensor),
})
```

---

## Writing a custom RelationSource

```python
from relann.data_sources import RelationSource
import pandas as pd

class ParquetSource(RelationSource):
    def __init__(self, name: str, path: str):
        self._name = name
        self._path = path
        self._data_version = 0

    @property
    def name(self):
        return self._name

    def schema(self):
        import pyarrow.parquet as pq
        return pq.read_schema(self._path).names

    def load_full(self):
        self._data_version += 1
        df = pd.read_parquet(self._path)
        from relann.encode import build_column_vocabs
        return {
            "content": df,
            "content_schema": list(df.columns),
            "embedding_shapes": [],
            "embeddings": None,
            "column_vocabs": build_column_vocabs(df) or None,
            "data_version": self._data_version,
        }

    # Optional: implement for minibatching
    def load_by_keys(self, key_col, keys):
        self._data_version += 1
        df = pd.read_parquet(self._path, filters=[(key_col, "in", list(keys))])
        from relann.encode import build_column_vocabs
        full = pd.read_parquet(self._path)
        return {
            "content": df,
            "content_schema": list(df.columns),
            "embedding_shapes": [],
            "embeddings": None,
            "column_vocabs": build_column_vocabs(full) or None,
            "data_version": self._data_version,
        }
```

---

## Minibatching protocol (`load_by_keys`)

`load_by_keys(key_col, keys)` is the hook for GNN-style node sampling. **`DataFrameSource`** and **table-backed `SqlSource`** implement it; **`query=`** `SqlSource` does not.

When a full `BatchSpec` runner lands, it will:

1. Sample K anchor keys from the anchor relation.
2. Walk the term graph topologically, calling `load_by_keys` on each DataLoader's source with the key set propagated via join edges.
3. Pass the resulting ER-dicts to `module.forward(batch_spec=...)`.

Two sampler modes are planned:

| Mode | Description |
|------|-------------|
| `node` (default) | GNN-style: anchor K nodes, pull the join closure. Correct for rules with joins. |
| `per_relation` | Independent random sample per relation. Simpler; incorrect for cross-relation joins. |

The sampler mode is specified via a `BatchSpec` dataclass (designed in `docs/design/encode-decode-review.md`, not yet implemented).

---

## Compatibility policy

### Legacy `(df, tensor)` dict entries

The `db={name: (df, tensor)}` shorthand is **preserved indefinitely** and does not require migration.

`Engine.__init__` materialises any `RelationSource` via `load_full()` at startup; legacy tuples are left untouched and normalised on-demand by `_to_er_dict` (the existing code path, unchanged).

Both styles can be mixed freely:

```python
session = Session(db={
    "Papers":  papers_dataframe_source,          # RelationSource
    "Labels":  (labels_df, labels_tensor),        # legacy tuple — still works
})
```

### Planned follow-up: `relann/datasets.py`

The 14+ `load_*_dataset()` functions in `relann/datasets.py` currently return `(df, tensor)` tuples internally. A follow-up commit will rewrite their internals to return `DataFrameSource` / `SqlSource` objects, making `RelationSource` the canonical API for all built-in datasets.

This is **non-breaking** — `to_dict()` and other public APIs remain the same. The refactor is deferred to a separate PR after `feature/encode-decode-nn-modules` lands on `main`.

---

## DBLP from SQLite demo

- Data-setup: `scripts/data_setup/dblp_from_sqlite/build_dblp_sqlite.py` — one-time dump of DBLP PyG tensors into `data/dblp_demo.sqlite`.
- Tutorial: `examples/008_relnn_dblp_sqlite.py` — trains a 1-layer HGT using `SqlSource` + `DataFrameSource`.

```bash
python scripts/data_setup/dblp_from_sqlite/build_dblp_sqlite.py
uv run poe nb examples/008_relnn_dblp_sqlite.py   # open in Jupyter
# or: uv run python examples/008_relnn_dblp_sqlite.py
```

## Cora from SQLite notebook

See `nbs/demos/007_relnn_sql_loading.ipynb` for a step-by-step tutorial using the Cora dataset:

- `scripts/data_setup/cora_from_sqlite/build_cora_sqlite.py` — one-time dump of Cora into `data/cora_demo.sqlite`.
- The notebook shows `SqlSource` for the edge table (`Citation`) and `DataFrameSource` for node features and labels (reconstructed from SQL numeric columns). The same 2-layer GCN model program from `001_relnn_hello_world.ipynb` is used unchanged.
