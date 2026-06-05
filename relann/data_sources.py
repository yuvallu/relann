"""
RelationSource — pluggable data-source abstraction for RelNN.

Provides a uniform interface between "where relation content comes from"
(DataFrames, SQL databases, future: Arrow, Parquet, …) and the RelNN
runtime which only cares about the normalised ER-dict shape:

    {"content": pd.DataFrame, "content_schema": [...], "embedding_shapes": [...],
     "embeddings": ..., "column_vocabs": optional, "data_version": int}

Usage::

    from relann.data_sources import DataFrameSource, SqlSource

    # Legacy equivalent (auto-wrapped by Engine.__init__ for backward-compat):
    src = DataFrameSource("Users", df, embeddings=z)

    # SQL — loads full table at Session init:
    src = SqlSource("Papers", "sqlite:///dblp.sqlite", table="Papers")

    # SQL — custom query:
    src = SqlSource("Abstracts", engine, query="SELECT paper_id, title FROM papers WHERE year > 2015")

    session = Session(db={"Users": src, "Papers": src2})

Minibatching seam
-----------------
``load_by_keys(key_col, keys)`` supports GNN-style node sampling for
``DataFrameSource`` and table-backed ``SqlSource`` (not for free-form
``query=`` SQL). Each load bumps ``data_version`` for cache invalidation;
``column_vocabs`` are built from the full relation for stable categorical codes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from relann.encode import build_column_vocabs

logger = logging.getLogger(__name__)

__all__ = ["RelationSource", "DataFrameSource", "SqlSource"]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class RelationSource(ABC):
    """
    Abstract base for all RelNN relation sources.

    A ``RelationSource`` knows how to produce the normalised ER-dict consumed
    by ``DataLoader`` / ``_to_er_dict``.  It does **not** know about the
    tensor-term compilation or the training loop; those remain in the engine.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Logical name of the relation (matches the name used in RelNN rules)."""
        ...

    @abstractmethod
    def schema(self) -> List[str]:
        """Column names of the content table (content attrs, not embedding)."""
        ...

    @abstractmethod
    def load_full(self) -> Dict[str, Any]:
        """
        Return the full relation as an ER-dict::

            {
                "content":          pd.DataFrame,
                "content_schema":   List[str],
                "embedding_shapes": List[tuple],
                "embeddings":       List[torch.Tensor] | None,
            }

        Called once during ``Engine.__init__`` (or lazily on first access).
        May be called again if the source is asked to reload (e.g. after a
        SQLite write-back).
        """
        ...

    def load_by_keys(self, key_col: str, keys: Sequence) -> Dict[str, Any]:
        """
        Return only the rows whose ``key_col`` is in ``keys``, in ER-dict form.

        This is the **minibatching hook**.  Default implementation raises
        ``NotImplementedError``; concrete sources implement it where supported.
        """
        raise NotImplementedError(
            f"{type(self).__name__}({self.name!r}) does not support key-based "
            "loading yet.  Implement load_by_keys() to enable mini-batching."
        )


# ---------------------------------------------------------------------------
# DataFrameSource
# ---------------------------------------------------------------------------


class DataFrameSource(RelationSource):
    """
    Wraps an in-memory ``(pd.DataFrame, embeddings)`` pair as a
    ``RelationSource``.

    This is the direct equivalent of the legacy ``db={name: (df, tensor)}``
    dict entry.  ``Engine.__init__`` auto-wraps legacy tuples so existing code
    continues to work without modification.

    Args:
        name:       Logical relation name (must match the name used in rules).
        content:    A ``pd.DataFrame`` holding the content (non-embedding)
                    columns.
        embeddings: A single ``torch.Tensor``, a list of tensors, or ``None``
                    when the relation has no pre-existing embeddings (they will
                    be produced by the rules).
    """

    def __init__(
        self,
        name: str,
        content: pd.DataFrame,
        embeddings: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ):
        self._name = name
        self._content = content
        if embeddings is None:
            self._embeddings: Optional[List[torch.Tensor]] = None
        elif isinstance(embeddings, torch.Tensor):
            self._embeddings = [embeddings]
        else:
            self._embeddings = list(embeddings)
        self._data_version = 0

    @property
    def name(self) -> str:
        return self._name

    def schema(self) -> List[str]:
        return list(self._content.columns)

    def _bump_version(self) -> int:
        self._data_version += 1
        return self._data_version

    def load_full(self) -> Dict[str, Any]:
        self._bump_version()
        embs = self._embeddings
        shapes = [e.shape for e in embs] if embs is not None else []
        vocabs = build_column_vocabs(self._content)
        return {
            "content": self._content,
            "content_schema": list(self._content.columns),
            "embedding_shapes": shapes,
            "embeddings": embs,
            "column_vocabs": vocabs or None,
            "data_version": self._data_version,
        }

    def load_by_keys(self, key_col: str, keys: Sequence) -> Dict[str, Any]:
        if key_col not in self._content.columns:
            raise KeyError(
                f"DataFrameSource({self._name!r}): key column {key_col!r} not in columns {list(self._content.columns)}"
            )
        keys_list = list(keys)
        self._bump_version()
        vocabs = build_column_vocabs(self._content)
        if not keys_list:
            empty = self._content.iloc[0:0].copy()
            embs = self._embeddings
            sliced = None
            if embs is not None:
                sliced = [e[0:0] for e in embs]
            shapes = [e.shape for e in sliced] if sliced else []
            return {
                "content": empty,
                "content_schema": list(empty.columns),
                "embedding_shapes": shapes,
                "embeddings": sliced,
                "column_vocabs": vocabs or None,
                "data_version": self._data_version,
            }
        mask = self._content[key_col].isin(keys_list)
        sub = self._content.loc[mask].reset_index(drop=True)
        idx = np.nonzero(mask.to_numpy())[0]
        embs = self._embeddings
        sliced = None
        if embs is not None:
            sliced = [e[idx] for e in embs]
        shapes = [e.shape for e in sliced] if sliced else []
        return {
            "content": sub,
            "content_schema": list(sub.columns),
            "embedding_shapes": shapes,
            "embeddings": sliced,
            "column_vocabs": vocabs or None,
            "data_version": self._data_version,
        }


# ---------------------------------------------------------------------------
# SqlSource
# ---------------------------------------------------------------------------


def _ensure_sqlalchemy() -> None:
    try:
        import sqlalchemy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SqlSource requires sqlalchemy>=2.0.  "
            "Install it with: pip install 'sqlalchemy>=2.0'"
        ) from exc


class SqlSource(RelationSource):
    """
    Loads a relation from a SQL database via SQLAlchemy.

    Supports any database supported by SQLAlchemy (SQLite, PostgreSQL, MySQL,
    DuckDB via ``duckdb-engine``, etc.).

    Args:
        name:           Logical relation name (must match the name used in rules).
        engine_or_url:  A SQLAlchemy ``Engine`` object or a connection string
                        such as ``"sqlite:///dblp.sqlite"`` or
                        ``"postgresql://user:pass@host/db"``.
        table:          Name of the table to load.  Required unless ``query``
                        is given.
        query:          Raw SQL query string (overrides ``table``).  The query
                        must return all columns needed by the RelNN rules.
        primary_key:    Name of the primary-key column.  Used by ``load_by_keys``
                        (not yet implemented) and informational only for now.
        dtype_map:      Optional ``{column_name: pandas_dtype}`` mapping applied
                        to the loaded DataFrame via ``df.astype()``.  Useful to
                        force categorical dtypes or cast numeric columns that SQL
                        returns as object.

    Example::

        SqlSource("Papers", "sqlite:///dblp.sqlite", table="Papers",
                  primary_key="paper_id")

        SqlSource("Cites", engine, query="SELECT src, dst FROM edges WHERE type='cites'")
    """

    def __init__(
        self,
        name: str,
        engine_or_url: Any,
        *,
        table: Optional[str] = None,
        query: Optional[str] = None,
        primary_key: Optional[str] = None,
        dtype_map: Optional[Dict[str, Any]] = None,
    ):
        if table is None and query is None:
            raise ValueError("SqlSource requires either 'table' or 'query'.")
        _ensure_sqlalchemy()

        self._name = name
        self._engine_or_url = engine_or_url
        self._table = table
        self._query = query
        self._primary_key = primary_key
        self._dtype_map = dtype_map or {}
        self._engine = None  # lazy init
        self._reflected_table: Any = None  # sqlalchemy.Table, lazy
        self._dtype_map_validated = False
        self._data_version = 0
        self._cached_column_vocabs: Optional[Dict[str, Dict[Any, int]]] = None

    @property
    def name(self) -> str:
        return self._name

    def _bump_version(self) -> int:
        self._data_version += 1
        return self._data_version

    def _get_engine(self):
        if self._engine is None:
            if isinstance(self._engine_or_url, str):
                from sqlalchemy import create_engine
                self._engine = create_engine(self._engine_or_url)
            else:
                self._engine = self._engine_or_url
        return self._engine

    def _reflect_table(self):
        """Reflect ``self._table`` once via SQLAlchemy Core. Returns ``sqlalchemy.Table``.

        Raises ``RuntimeError`` if this source was constructed with ``query=`` instead.
        Used by ``load_by_keys`` (and the empty-keys path) to build dialect-correct
        ``SELECT ... WHERE col IN (...)`` expressions instead of f-string SQL.
        """
        if self._table is None:
            raise RuntimeError(
                f"SqlSource({self._name!r}): _reflect_table called on a query-only source."
            )
        if self._reflected_table is None:
            from sqlalchemy import MetaData, Table
            md = MetaData()
            self._reflected_table = Table(
                self._table, md, autoload_with=self._get_engine()
            )
        return self._reflected_table

    def schema(self) -> List[str]:
        """Return column names for this source.

        Table mode uses ``sqlalchemy.inspect``. Query mode runs the query and reads
        ``Result.keys()`` without consuming any rows — dialect-portable, unlike the
        previous ``LIMIT 0`` subquery wrap which is not universally supported.
        """
        if self._table is not None:
            from sqlalchemy import inspect as sa_inspect
            insp = sa_inspect(self._get_engine())
            cols = insp.get_columns(self._table)
            return [c["name"] for c in cols]
        if self._query is None:
            raise RuntimeError("SqlSource has neither table nor query.")
        from sqlalchemy import text
        with self._get_engine().connect() as conn:
            result = conn.execute(text(self._query))
            try:
                return list(result.keys())
            finally:
                result.close()

    def _validate_dtype_map(self, columns: List[str]) -> None:
        """Fail fast if ``dtype_map`` keys reference unknown columns. Once per source."""
        if self._dtype_map_validated or not self._dtype_map:
            return
        unknown = [k for k in self._dtype_map if k not in columns]
        if unknown:
            raise ValueError(
                f"SqlSource({self._name!r}): dtype_map references unknown columns "
                f"{unknown}. Available columns: {columns}."
            )
        self._dtype_map_validated = True

    def _load_df(self) -> pd.DataFrame:
        engine = self._get_engine()
        if self._query is not None:
            from sqlalchemy import text
            with engine.connect() as conn:
                df = pd.read_sql_query(text(self._query), conn)
        else:
            with engine.connect() as conn:
                df = pd.read_sql_table(self._table, conn)
        self._validate_dtype_map(list(df.columns))
        if self._dtype_map:
            df = df.astype(self._dtype_map)
        return df

    def load_full(self) -> Dict[str, Any]:
        """Load the entire table / query result as an ER-dict (no embeddings)."""
        self._bump_version()
        df = self._load_df()
        vocabs = build_column_vocabs(df)
        self._cached_column_vocabs = vocabs or None
        logger.debug("SqlSource(%r): loaded %d rows, columns=%s", self._name, len(df), list(df.columns))
        return {
            "content": df,
            "content_schema": list(df.columns),
            "embedding_shapes": [],
            "embeddings": None,
            "column_vocabs": self._cached_column_vocabs,
            "data_version": self._data_version,
        }

    def load_by_keys(self, key_col: str, keys: Sequence) -> Dict[str, Any]:
        """
        Return only the rows matching the given keys (``table=`` sources only;
        not supported for arbitrary ``query=``).

        Uses SQLAlchemy Core (``select(table).where(col.in_(keys))``) so identifier
        quoting is dialect-correct (SQLite/Postgres/MySQL/MSSQL/Oracle) instead of
        the previous f-string with ANSI double quotes which broke MySQL.
        """
        if self._table is None:
            raise NotImplementedError(
                f"SqlSource({self._name!r}): load_by_keys is only implemented when "
                "loading from a concrete table (table=...), not a free-form query."
            )
        keys_list = list(keys)
        self._bump_version()
        engine = self._get_engine()

        if self._cached_column_vocabs is None:
            self._cached_column_vocabs = build_column_vocabs(self._load_df()) or None
        vocabs = self._cached_column_vocabs

        from sqlalchemy import select
        table = self._reflect_table()
        if key_col not in table.c:
            raise KeyError(
                f"SqlSource({self._name!r}): key column {key_col!r} not in table columns "
                f"{[c.name for c in table.c]}."
            )
        col = table.c[key_col]
        stmt = select(table).where(col.in_(keys_list))
        with engine.connect() as conn:
            sub = pd.read_sql_query(stmt, conn)
        self._validate_dtype_map(list(sub.columns))
        if self._dtype_map:
            sub = sub.astype(self._dtype_map)
        return {
            "content": sub,
            "content_schema": list(sub.columns),
            "embedding_shapes": [],
            "embeddings": None,
            "column_vocabs": vocabs,
            "data_version": self._data_version,
        }
