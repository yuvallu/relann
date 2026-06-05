# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %% [markdown]
# # Embedded RA Operations
#
# > A module for handling embedded RA operations in the DSL

# %%
import logging
from relann.term_graph import program_to_graph, create_simple_join_program
from relann.column_ref import ColumnRef
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Union
from relann.pydantic_classes import ComparisonExpression, ArithTerm, Var

logger = logging.getLogger(__name__)

# %%
# `torch_scatter` ships as compiled wheels matched to your exact torch + CUDA
# build, so a plain `pip install relann` can't pull it (see docs/install-gpu.md).
# Import it lazily: `import relann` then works without the PyG sparse stack, and
# the scatter ops only raise an install hint if a scatter aggregation is run.
def _require_torch_scatter():
    try:
        import torch_scatter
    except ImportError as exc:
        raise ImportError(
            "relann's scatter/aggregation operators need the PyG sparse stack "
            "(torch-scatter), which ships as prebuilt wheels matched to your "
            "torch + CUDA build and is NOT installed by `pip install relann`.\n"
            "Install it with:\n"
            "    pip install --no-build-isolation torch-scatter torch-sparse "
            "torch-cluster torch-geometric \\\n"
            "        -f https://data.pyg.org/whl/torch-2.6.0+cu124.html\n"
            "(swap cu124 for your CUDA tag, or +cpu for a CPU-only host). "
            "See https://github.com/yuvallu/relann/blob/main/docs/install-gpu.md"
        ) from exc
    return torch_scatter


def scatter_add(*args, **kwargs):
    """Lazy passthrough to torch_scatter.scatter_add (see _require_torch_scatter)."""
    return _require_torch_scatter().scatter_add(*args, **kwargs)


def scatter_mean(*args, **kwargs):
    """Lazy passthrough to torch_scatter.scatter_mean (see _require_torch_scatter)."""
    return _require_torch_scatter().scatter_mean(*args, **kwargs)


def scatter_softmax(*args, **kwargs):
    """Lazy passthrough to torch_scatter.scatter_softmax (see _require_torch_scatter)."""
    return _require_torch_scatter().scatter_softmax(*args, **kwargs)

# %%
# Import cuDF and pandas for GPU acceleration
try:
    import cudf
except ImportError:
    cudf = None
import pandas

# Utility function to check if object is a DataFrame (pandas or cuDF)
def _is_df(x: Any) -> bool:
    """Check if object is a pandas or cuDF DataFrame."""
    try:
        import pandas as pd
        if pd is not None and isinstance(x, pd.DataFrame):
            return True
    except (ImportError, AttributeError):
        pass
    
    try:
        if cudf is not None and isinstance(x, cudf.DataFrame):  # type: ignore
            return True
    except (ImportError, AttributeError, TypeError):
        pass
    
    return False


def _rename_columns_safe(df, old_cols: List[str], new_cols: List[str]):
    """Rename DataFrame columns positionally (old_cols -> new_cols)."""
    if not hasattr(df, "columns") or old_cols == new_cols:
        return df
    if len(old_cols) != len(new_cols):
        return df
    out = df.copy()
    out.columns = list(new_cols)
    return out


def _maybe_apply_output_schema(df, current_schema: List[str], output_schema: Optional[List[str]]):
    """Return (df, schema) with optional output_schema applied."""
    if output_schema is None:
        return df, current_schema
    if not hasattr(df, "columns"):
        return df, current_schema
    cols = list(getattr(df, "columns", []))
    if len(cols) != len(output_schema):
        return df, current_schema
    df2 = _rename_columns_safe(df, cols, list(output_schema))
    return df2, list(output_schema)

# %%
if __name__ == "__main__":

    import torch_geometric

# %%
if __name__ == "__main__":

    device = torch_geometric.device('auto')  # Note: make sure this line exists in the code that calls this file.
    print(f"Using device: {device}")

# %%
if __name__ == "__main__":
    # Test helper functions for CPU/GPU compatibility
    # These functions allow tests to work with both pandas (CPU) and cuDF (GPU)

    def get_df_class():
        """Returns the appropriate DataFrame class (cuDF if available, otherwise pandas)."""
        try:
            import cudf
            if cudf is not None:
                return cudf.DataFrame
        except (ImportError, AttributeError):
            pass
        
        try:
            import pandas as pd
            if pd is not None:
                return pd.DataFrame
        except (ImportError, AttributeError):
            pass
        
        raise RuntimeError("Neither cuDF nor pandas is available")

    def create_test_df(data):
        """Create a DataFrame using the appropriate backend (cuDF if available, otherwise pandas)."""
        df_class = get_df_class()
        return df_class(data)

    def test_merge(left, right, on=None, left_on=None, right_on=None, how='inner'):
        """Merge two DataFrames using the appropriate backend."""
        try:
            import cudf
            # Check if either DataFrame is cuDF
            if cudf is not None and ((hasattr(left, '__class__') and left.__class__.__module__.startswith('cudf')) or \
               (hasattr(right, '__class__') and right.__class__.__module__.startswith('cudf'))):
                if on is not None:
                    return cudf.merge(left, right, on=on, how=how)
                else:
                    return cudf.merge(left, right, left_on=left_on, right_on=right_on, how=how)
        except (ImportError, AttributeError):
            pass
        
        # Fallback to pandas merge
        if on is not None:
            return left.merge(right, on=on, how=how)
        else:
            return left.merge(right, left_on=left_on, right_on=right_on, how=how)

    def get_test_device():
        """Get the appropriate device for tests (CPU if CUDA not available)."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    # Update device to ensure CPU fallback if CUDA not available
    device = get_test_device()
    print(f"Using device: {device}")

# %%
def where_is(df):
    """Determine if DataFrame is cuDF (GPU) or pandas (CPU)."""
    try:
        import cudf
        if cudf is not None and isinstance(df, cudf.DataFrame):
            return "GPU (cuDF)"
    except (ImportError, AttributeError, TypeError):
        pass
    
    try:
        import pandas as pd
        if pd is not None and isinstance(df, pd.DataFrame):
            return "CPU (pandas)"
    except (ImportError, AttributeError, TypeError):
        pass
    
    return "unknown"

# %%
if __name__ == "__main__":

    try:
        gdf = cudf.DataFrame({'x': [1, 2, 3]})
        print(where_is(gdf))   # → GPU (cuDF)
    except Exception as e:
        print(f"Could not create cuDF DataFrame: {e}")

    try:
        pdf = pandas.DataFrame({'x': [1, 2, 3]})
        print(where_is(pdf))   # → CPU (pandas)
    except Exception as e:
        print(f"Could not create pandas DataFrame: {e}")

# %%
from relann.embedded_relation import EmbeddedRelation as EmbeddedRelation
from relann.embedded_relation import HAS_CUDF, HAS_PANDAS
import pandas as pd

from dataclasses import dataclass


@dataclass
class ExecutionContext:
    """Execution context passed through the DAG; holds relation data for DataLoader and future flags."""
    relations: Dict[str, Any]


def _merge_column_vocabs(
    *parts: Optional[Dict[str, Dict[Any, int]]],
) -> Optional[Dict[str, Dict[Any, int]]]:
    """Merge per-column categorical code maps; raise if the same column disagrees."""
    out: Dict[str, Dict[Any, int]] = {}
    for p in parts:
        if not p:
            continue
        for col, mapping in p.items():
            if col in out and out[col] != mapping:
                raise ValueError(
                    f"column_vocabs conflict for column {col!r}: incompatible categorical code maps "
                    "when combining relations."
                )
            out[col] = mapping
    return out or None


def _max_data_version(sons: List["EmbeddedRelation"]) -> int:
    if not sons:
        return 0
    return max(int(getattr(s, "data_version", 0) or 0) for s in sons)


def _project_column_vocabs(
    full: Optional[Dict[str, Dict[Any, int]]],
    keys: Optional[List[str]],
) -> Optional[Dict[str, Dict[Any, int]]]:
    """Restrict full relation vocabs to the subset of content columns present after project/agg."""
    if not full or not keys:
        return None
    out = {k: full[k] for k in keys if k in full}
    return out or None


def _with_er_dict_metadata(content: Any, d: Dict[str, Any]) -> Dict[str, Any]:
    """Fill ``column_vocabs`` / ``data_version`` on a normalised ER-dict when missing."""
    if d.get("column_vocabs") is None and _is_df(content):
        from relann.encode import build_column_vocabs

        cv = build_column_vocabs(content)
        if cv:
            d["column_vocabs"] = cv
    d.setdefault("data_version", 0)
    d["data_version"] = int(d["data_version"])
    return d


# Input normalization — converts multiple formats to a standard dict
def _to_er_dict(rel: Any) -> Dict[str, Any]:
    """
    Normalize a relation entry to:
      {content, content_schema, embedding_shapes, embeddings}
    Accepts dict/EmbeddedRelation/DataFrame/tuple(list) variants.
    """
    # dict
    if isinstance(rel, dict):
        content = rel.get("content", None)
        schema  = rel.get("content_schema", list(getattr(content, "columns", [])) if content is not None else [])
        embs    = rel.get("embeddings", None)
        shapes  = rel.get("embedding_shapes", [])
        if (not shapes) and embs is not None:
            embs_list = list(embs) if isinstance(embs, (list, tuple)) else [embs]
            shapes = [e.shape for e in embs_list]
            embs   = embs_list
        d: Dict[str, Any] = {
            "content": content,
            "content_schema": schema,
            "embedding_shapes": shapes or [],
            "embeddings": embs,
        }
        if "column_vocabs" in rel:
            d["column_vocabs"] = rel["column_vocabs"]
        if "data_version" in rel:
            d["data_version"] = int(rel["data_version"])
        return _with_er_dict_metadata(content, d)

    # EmbeddedRelation-like
    if hasattr(rel, "content") and hasattr(rel, "content_schema"):
        content = rel.content
        schema  = list(getattr(rel, "content_schema", list(getattr(content, "columns", []))))
        embs    = getattr(rel, "embeddings", None)
        shapes  = getattr(rel, "embedding_shapes", None)
        if (not shapes) and embs is not None:
            embs_list = list(embs) if isinstance(embs, (list, tuple)) else [embs]
            shapes = [e.shape for e in embs_list]
            embs   = embs_list
        d = {"content": content, "content_schema": schema, "embedding_shapes": shapes or [], "embeddings": embs}
        if getattr(rel, "column_vocabs", None) is not None:
            d["column_vocabs"] = rel.column_vocabs
        d["data_version"] = int(getattr(rel, "data_version", 0))
        return _with_er_dict_metadata(content, d)

    # bare DataFrame
    if _is_df(rel):
        return _with_er_dict_metadata(
            rel,
            {"content": rel, "content_schema": list(rel.columns), "embedding_shapes": [], "embeddings": None},
        )

    # tuple/list forms
    if isinstance(rel, (list, tuple)):
        items = list(rel)
        n = len(items)
        if n == 4:
            content, schema, shapes, embs = items
        elif n == 3:
            content, schema, third = items
            if isinstance(third, (list, tuple)) and len(third) > 0 and torch.is_tensor(third[0]):
                embs = list(third)
                shapes = [e.shape for e in embs]
            else:
                shapes = third
                embs = None
        elif n == 2:
            content, second = items
            if _is_df(content):
                # Check if second is a single tensor or a list/tuple of tensors
                if torch.is_tensor(second):
                    embs = [second]
                    schema = list(content.columns)
                    shapes = [second.shape]
                elif isinstance(second, (list, tuple)) and len(second) > 0 and torch.is_tensor(second[0]):
                    embs = list(second)
                    schema = list(content.columns)
                    shapes = [e.shape for e in embs]
                else:
                    # Treat second as schema (backward compatibility)
                    schema = list(second) if isinstance(second, (list, tuple)) else [second]
                    embs = None
                    shapes = []
            else:
                schema = list(second) if isinstance(second, (list, tuple)) else [second]
                embs = None
                shapes = []
        elif n == 1:
            content = items[0]
            schema  = list(getattr(content, "columns", [])) if _is_df(content) else []
            shapes, embs = [], None
        else:
            raise TypeError(f"Unsupported relation tuple length {n}")
        if isinstance(embs, tuple): embs = list(embs)
        return _with_er_dict_metadata(
            content,
            {"content": content, "content_schema": schema, "embedding_shapes": shapes or [], "embeddings": embs},
        )

    raise TypeError(f"Unsupported relation type: {type(rel)}")

# %% [markdown]
# ## Join Operation

# %%
class Join(nn.Module):
    """Join operation for embedded relations. Uses pre-computed merge_steps for efficient execution."""

    def __init__(self, output_schema: List[str], merge_steps: List[Dict], input_schemas: List[List[str]]):
        super().__init__()
        self.output_schema = output_schema
        self.merge_steps = merge_steps
        self.input_schemas = input_schemas
        self._cached_join = None  # (df_out, input_row_indices_cpu)

    @staticmethod
    def _is_cudf_df(df) -> bool:
        try:
            import cudf  # type: ignore
            return isinstance(df, cudf.DataFrame)
        except Exception:
            return False

    @staticmethod
    def _add_tmp_idx(df, name: str):
        n = len(df)
        if Join._is_cudf_df(df):
            import cupy as cp
            df[name] = cp.arange(n, dtype=cp.int64)
        else:
            import numpy as np
            df[name] = np.arange(n, dtype=np.int64)

    @staticmethod
    def _series_to_long_cpu(series):
        if Join._is_cudf_df(series):
            import cupy as cp
            arr = cp.asnumpy(series.values)
        else:
            arr = series.to_numpy()
        return torch.as_tensor(arr, dtype=torch.long, device="cpu")

    @staticmethod
    def _merge_on(left, right, on: List[str], suffixes=("_x", "_y")):
        if Join._is_cudf_df(left) or Join._is_cudf_df(right):
            import cudf  # type: ignore
            return cudf.merge(left, right, on=on, how="inner", suffixes=suffixes)  # type: ignore
        else:
            return left.merge(right, on=on, how="inner", suffixes=suffixes)

    @staticmethod
    def _merge_lr(left, right, left_on: List[str], right_on: List[str], suffixes=("", "_y")):
        if Join._is_cudf_df(left) or Join._is_cudf_df(right):
            import cudf  # type: ignore
            return cudf.merge(left, right, left_on=left_on, right_on=right_on, how="inner", suffixes=suffixes)  # type: ignore
        else:
            return left.merge(right, left_on=left_on, right_on=right_on, how="inner", suffixes=suffixes)

    @staticmethod
    def _resolve_column_ref(
        col_ref: ColumnRef,
        sons: List["EmbeddedRelation"],
        fallback_column_name: Optional[str] = None,
        override_schemas: Optional[List[List[str]]] = None,
    ) -> str:
        """
        Resolve a ColumnRef to an actual column name.
        When override_schemas is provided (e.g. Join.input_schemas), use it so merge keys
        match the normalized DataFrame columns.
        """
        input_idx, column_idx = col_ref.input_idx, col_ref.column_idx

        if input_idx < 0 or input_idx >= len(sons):
            raise ValueError(
                f"ColumnRef({input_idx}, {column_idx}): input index {input_idx} out of bounds "
                f"(Join has {len(sons)} inputs)"
            )

        if override_schemas and input_idx < len(override_schemas) and column_idx < len(override_schemas[input_idx]):
            return override_schemas[input_idx][column_idx]

        son = sons[input_idx]
        schema = getattr(son, 'content_schema', None)
        
        # Fallback to DataFrame columns if schema not available
        if not schema and hasattr(son, 'content') and son.content is not None:
            try:
                schema = list(son.content.columns) if len(son.content.columns) > 0 else None
            except (AttributeError, TypeError):
                pass
        
        # Last resort: use fallback name for first column only
        if not schema and fallback_column_name and column_idx == 0:
            schema = [fallback_column_name]
        
        if not schema:
            raise ValueError(
                f"ColumnRef({input_idx}, {column_idx}): Cannot resolve schema for input {input_idx}. "
                f"content_schema={getattr(son, 'content_schema', None)}, "
                f"content.columns={list(son.content.columns) if hasattr(son, 'content') and son.content is not None else None}"
            )
        
        if column_idx < 0 or column_idx >= len(schema):
            raise ValueError(
                f"ColumnRef({input_idx}, {column_idx}): column index {column_idx} out of bounds "
                f"(schema has {len(schema)} columns: {schema})"
            )
        
        return schema[column_idx]

    @staticmethod
    def _resolve_normalized_refs(
        normalized_refs: List[ColumnRef], 
        sons: List["EmbeddedRelation"],
        fallback_column_name: Optional[str] = None
    ) -> List[str]:
        """
        Resolve a list of ColumnRef objects to actual column names.
        
        Args:
            normalized_refs: List of ColumnRef objects
            sons: List of EmbeddedRelation objects containing DataFrames
            fallback_column_name: Optional column name to use as fallback when schema is empty
        
        Returns:
            List of column names as strings
        
        Raises:
            ValueError: If any ColumnRef is invalid (see _resolve_column_ref)
        """
        resolved_keys = []
        for col_ref in normalized_refs:
            column_name = Join._resolve_column_ref(col_ref, sons, fallback_column_name=fallback_column_name)
            resolved_keys.append(column_name)
        return resolved_keys

    @staticmethod
    def _prepare_dfs(sons: List["EmbeddedRelation"], input_schemas: List[List[str]]) -> List[Any]:
        """Copy each son.content, normalize to logical names by position, add __idx; return list of DataFrames."""
        dfs = []
        for i, son in enumerate(sons):
            dfc = son.content.copy()
            # Prefer Join's expected schema so DataFrames align with merge keys (logical names)
            schema = (input_schemas[i] if i < len(input_schemas) and input_schemas[i] else None) or getattr(son, "content_schema", None)
            if schema:
                n = min(len(schema), len(dfc.columns))
                if n > 0:
                    # Align by position: first n columns get logical names from schema (handles extra cols in data)
                    dfc = dfc.iloc[:, :n].copy() if n < len(dfc.columns) else dfc
                    dfc.columns = list(schema)[:n]
                elif len(dfc.columns) == 0 and len(schema) > 0:
                    # Son has no columns but join expects schema (e.g. transformation passed empty content).
                    nrows = len(dfc)
                    try:
                        key_vals = list(dfc.index) if nrows > 0 and hasattr(dfc, "index") else list(range(nrows))
                    except Exception:
                        key_vals = list(range(nrows))
                    import pandas as pd
                    use_cudf = Join._is_cudf_df(dfc) if nrows > 0 else False
                    if not use_cudf:
                        for other_son in sons:
                            if other_son.content is not None and len(other_son.content) > 0:
                                use_cudf = Join._is_cudf_df(other_son.content)
                                break
                    if nrows == 0:
                        if use_cudf:
                            import cudf as _cudf  # type: ignore
                            dfc = _cudf.DataFrame(columns=schema)  # type: ignore
                        else:
                            dfc = pd.DataFrame(columns=schema)
                    else:
                        data = {schema[0]: key_vals}
                        for c in schema[1:]:
                            data[c] = [None] * nrows
                        if use_cudf:
                            import cudf as _cudf  # type: ignore
                            dfc = _cudf.DataFrame(data)  # type: ignore
                        else:
                            dfc = pd.DataFrame(data)
                elif len(dfc.columns) == 0 and len(dfc) == 0:
                    pass  # empty df; handle below
            if len(dfc) == 0 and len(dfc.columns) == 0 and schema:
                import pandas as pd
                use_cudf = False
                for other_son in sons:
                    if other_son.content is not None and len(other_son.content) > 0:
                        use_cudf = Join._is_cudf_df(other_son.content)
                        break
                if use_cudf:
                    import cudf  # type: ignore
                    dfc = cudf.DataFrame(columns=schema)  # type: ignore
                else:
                    dfc = pd.DataFrame(columns=schema)
            Join._add_tmp_idx(dfc, f"__idx{i}")
            dfs.append(dfc)
        return dfs

    def _resolve_step_keys(self, merge_step: Dict, sons: List["EmbeddedRelation"], step: int) -> tuple:
        """Resolve left_on and right_on for one merge step (use input_schemas so keys match normalized dfs)."""
        key_name = (merge_step.get("key_names") or [None])[0]
        left_on = [
            Join._resolve_column_ref(ref, sons[: step + 1], fallback_column_name=key_name, override_schemas=self.input_schemas)
            for ref in merge_step["left_refs"]
        ]
        right_on = [
            Join._resolve_column_ref(ref, sons, fallback_column_name=key_name, override_schemas=self.input_schemas)
            for ref in merge_step["right_refs"]
        ]
        # Diagnostic logging for the multi-step Join column-tracking bug. Enable with:
        #   logging.getLogger('relann.era_operations').setLevel(logging.DEBUG)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_resolve_step_keys step=%d  left_on=%r  right_on=%r  "
                "left_refs=%r  right_refs=%r  input_schemas=%r",
                step, left_on, right_on,
                merge_step.get("left_refs"), merge_step.get("right_refs"),
                self.input_schemas,
            )
        return (left_on, right_on)

    @staticmethod
    def _do_one_merge(
        df_joined: Any, df_next: Any, left_on: List[str], right_on: List[str], step: int,
        future_keys: "frozenset[str]" = frozenset(),
    ) -> Any:
        """Coerce types, optionally rename right key cols to left names, then merge on left_on.

        Uses suffixes=("", f"_iter{step}") so the accumulating left side keeps its column
        names verbatim and each merged-in right side gets a unique, step-indexed suffix.
        This prevents the pandas default `_x`/`_y` from colliding in chained joins.

        ``future_keys`` is the union of ``left_on`` across all *later* merge_steps —
        i.e. the set of column names that some later step will read from
        ``df_joined`` (right keys resolve against the fresh ``dfs[step]`` and don't
        need lookahead). When ``left_on != right_on``, the right key column is
        dropped (to honor the user's chosen output naming) — UNLESS the dropped
        name appears in ``future_keys``. This is the HGT case where step 1 joins
        ``left=t, right=s`` (drops ``s``) and
        step 2 then wants ``left_on=['s']``. Empty default preserves the original
        behaviour for any caller that doesn't supply this lookahead.
        """
        if logger.isEnabledFor(logging.DEBUG):
            missing_left = [k for k in left_on if k not in df_joined.columns]
            missing_right = [k for k in right_on if k not in df_next.columns]
            logger.debug(
                "_do_one_merge step=%d  left_on=%r  right_on=%r  "
                "df_joined.columns=%r  df_next.columns=%r  "
                "missing_left=%r  missing_right=%r",
                step, left_on, right_on,
                list(df_joined.columns), list(df_next.columns),
                missing_left, missing_right,
            )
        df_next = df_next.copy()
        for kl, kr in zip(left_on, right_on):
            if kl in df_joined.columns and kr in df_next.columns:
                dl, dr = df_joined[kl].dtype, df_next[kr].dtype
                if str(dl) != str(dr):
                    df_next[kr] = df_next[kr].astype(dl)
        suffixes = ("", f"_iter{step}")
        if left_on == right_on:
            return Join._merge_on(df_joined, df_next, on=left_on, suffixes=suffixes)
        df_joined = Join._merge_lr(
            df_joined, df_next, left_on=left_on, right_on=right_on, suffixes=suffixes,
        )
        # Drop right key columns when they differ from left — check both the original
        # name and the iter-suffixed name (pandas only suffixes when there's a collision).
        # But: keep any column a later step still needs (see ``future_keys`` in the docstring).
        drop_cols = []
        for kl, kr in zip(left_on, right_on):
            if kl != kr and kr not in future_keys:
                suffixed = f"{kr}{suffixes[1]}"
                if suffixed in df_joined.columns:
                    drop_cols.append(suffixed)
                elif kr in df_joined.columns:
                    drop_cols.append(kr)
        if drop_cols:
            df_joined = df_joined.drop(columns=drop_cols)
        return df_joined

    @staticmethod
    def _extract_row_indices(df_joined: Any, n_sons: int) -> List[torch.Tensor]:
        """Extract __idx{i} columns from joined DataFrame as CPU long tensors."""
        out = []
        for i in range(n_sons):
            col = df_joined[f"__idx{i}"]
            if hasattr(col, "fillna"):
                col = col.fillna(-1)
            out.append(Join._series_to_long_cpu(col))
        return out

    @staticmethod
    def _apply_join_output_schema(df_joined: Any, output_schema: List[str]) -> Any:
        """Coalesce duplicate columns to the canonical names in ``output_schema``.

        Recognized suffixes: ``_x``/``_y``/``_left``/``_right`` (from legacy pandas
        defaults) and ``_iter{N}`` (the step-indexed scheme used by ``_do_one_merge``
        to avoid chained-merge collisions).
        Without recognizing ``_iter{N}`` here, those duplicate copies would leak into
        the Join output.

        Candidate precedence: if a ``want`` column is already present verbatim it is
        kept as-is (no rename), and any ``_x``/``_y``/``_left``/``_right``/``_iter\\d+``
        variants are dropped. If ``want`` is missing, the first surviving candidate
        becomes ``want`` — iteration order is the explicit list above, then any
        ``_iter\\d+`` matches in column order.
        """
        import re
        cols = list(df_joined.columns)
        iter_re = re.compile(r"_iter\d+$")
        rename_map = {}
        columns_to_drop = set()
        for want in output_schema:
            candidates = [want] if want in cols else []
            for suffix in ("_x", "_y", "_left", "_right"):
                cand = want + suffix
                if cand in cols:
                    candidates.append(cand)
            candidates.extend(c for c in cols if iter_re.search(c) and iter_re.sub("", c) == want)
            if candidates:
                have = candidates[0]
                if have != want:
                    rename_map[have] = want
                for dup in candidates[1:]:
                    columns_to_drop.add(dup)
        df_out = df_joined.copy()
        if rename_map:
            df_out = df_out.rename(columns=rename_map)
        if columns_to_drop:
            df_out = df_out.drop(columns=list(columns_to_drop))
        return df_out

    def instantiate(self, sons: List["EmbeddedRelation"], ctx: Optional["ExecutionContext"] = None) -> "EmbeddedRelation":
        if len(sons) < 2:
            raise ValueError(f"Join operation requires at least 2 input relations. Got {len(sons)}")
        if not self.merge_steps:
            raise ValueError("Join requires non-empty merge_steps.")
        for i, son in enumerate(sons):
            if son.content is None:
                raise ValueError(
                    f"Join: input relation {i} has no content. "
                    f"This usually means the upstream operator failed to produce a result."
                )

        dfs = self._prepare_dfs(sons, self.input_schemas)
        # Pre-resolve every step's (left_on, right_on) so each step can know which
        # column names later steps will still need — see `_do_one_merge`'s
        # ``future_keys`` parameter (the HGT case).
        step_keys = [
            self._resolve_step_keys(ms, sons, ms["step"]) for ms in self.merge_steps
        ]
        df_joined = dfs[0]
        for i, merge_step in enumerate(self.merge_steps):
            step = merge_step["step"]
            left_on, right_on = step_keys[i]
            # Lookahead: only later steps' `left_on` reads from `df_joined`.
            # `right_on` resolves against the fresh `dfs[step]` each iteration
            # and never needs lookahead protection in `df_joined`.
            future_keys: "frozenset[str]" = frozenset(
                k for lon, _ in step_keys[i + 1 :] for k in lon
            )
            df_joined = self._do_one_merge(
                df_joined, dfs[step], left_on, right_on, step, future_keys,
            )

        input_row_indices_cpu = self._extract_row_indices(df_joined, len(sons))
        df_joined = df_joined.drop(columns=[f"__idx{i}" for i in range(len(sons))])
        df_out = (
            self._apply_join_output_schema(df_joined, self.output_schema)
            if self.output_schema
            else df_joined.copy()
        )

        expected_shapes = []
        for son in sons:
            if son.embedding_shapes:
                for shp in son.embedding_shapes:
                    expected_shapes.append((len(df_out), *tuple(shp[1:])))

        self._cached_join = (df_out, input_row_indices_cpu)
        # Per-son cache: maps son_index -> (device, index_tensor_on_device).
        # Populated lazily on first forward() call and updated when embedding device changes.
        self._cached_join_indices_by_son: List[Optional[tuple]] = [None] * len(sons)
        vmerge = _merge_column_vocabs(*[getattr(s, "column_vocabs", None) for s in sons])
        return EmbeddedRelation(
            content_schema=list(df_out.columns),
            embedding_shapes=expected_shapes,
            content=df_out,
            embeddings=None,
            column_vocabs=vmerge,
            data_version=_max_data_version(sons),
        )

    def forward(self, sons: List["EmbeddedRelation"], ctx: Optional["ExecutionContext"] = None) -> "EmbeddedRelation":
        if self._cached_join is None:
            raise RuntimeError("No cached join found. Did you forget to call instantiate() before forward()?")

        df_out, input_row_indices_cpu = self._cached_join

        aligned_embeddings = []
        for i, son in enumerate(sons):
            if not son.embeddings:
                continue
            # Determine this son's embedding device from its first embedding.
            son_device = son.embeddings[0].device
            cached = self._cached_join_indices_by_son[i]
            if cached is None or cached[0] != son_device:
                # First call or device changed — move index to the son's device and cache.
                idx_on_dev = input_row_indices_cpu[i].to(
                    device=son_device, dtype=torch.long, non_blocking=True
                )
                self._cached_join_indices_by_son[i] = (son_device, idx_on_dev)
            else:
                idx_on_dev = cached[1]
            for emb in son.embeddings:
                aligned_embeddings.append(emb.index_select(0, idx_on_dev))

        vmerge = _merge_column_vocabs(*[getattr(s, "column_vocabs", None) for s in sons])
        return EmbeddedRelation(
            content_schema=list(df_out.columns),  # Match actual DataFrame columns
            embedding_shapes=[e.shape for e in aligned_embeddings],
            content=df_out,
            embeddings=aligned_embeddings,
            column_vocabs=vmerge,
            data_version=_max_data_version(sons),
        )

# %%
if __name__ == "__main__":
    # Test for Join operation

    # Create test data
    df1 = create_test_df({
        'id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'age': [25, 30, 35, 40]
    })

    df2 = create_test_df({
        'id': [1, 2, 3, 5],
        'city': ['NYC', 'LA', 'Chicago', 'Miami'],
        'salary': [50000, 60000, 70000, 80000]
    })

    # Create embeddings for each relation
    emb1 = torch.randn(4, 10, device=device)  # 4 rows, 10 features
    emb2 = torch.randn(4, 8, device=device)   # 4 rows, 8 features

    # Create EmbeddedRelation objects
    rel1 = EmbeddedRelation(
        content_schema=['id', 'name', 'age'],
        embedding_shapes=[(4, 10)],
        content=df1,
        embeddings=[emb1]
    )

    rel2 = EmbeddedRelation(
        content_schema=['id', 'city', 'salary'],
        embedding_shapes=[(4, 8)],
        content=df2,
        embeddings=[emb2]
    )

    # Create Join operation
    output_schema = ['id', 'name', 'age', 'city', 'salary']
    input_schemas = [['id', 'name', 'age'], ['id', 'city', 'salary']]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 0)],
        "right_refs": [ColumnRef(1, 0)],
        "key_names": ["id"]
    }]
    join_op = Join(output_schema=output_schema, merge_steps=merge_steps, input_schemas=input_schemas)

    # Test instantiate
    print("Testing instantiate...")
    joined_rel = join_op.instantiate([rel1, rel2])

    print(f"Joined DataFrame shape: {joined_rel.content.shape}")
    print(f"Expected embedding shapes: {joined_rel.embedding_shapes}")
    print(f"Joined DataFrame:\n{joined_rel.content}")

    # Verify join results
    expected_joined_df = test_merge(df1, df2, on='id', how='inner')
    assert joined_rel.content.equals(expected_joined_df), "Join DataFrame doesn't match expected result"
    assert len(joined_rel.embedding_shapes) == 2, "Should have 2 embedding shapes"
    assert joined_rel.embedding_shapes[0] == (3, 10), f"First embedding shape should be (3, 10), got {joined_rel.embedding_shapes[0]}"
    assert joined_rel.embedding_shapes[1] == (3, 8), f"Second embedding shape should be (3, 8), got {joined_rel.embedding_shapes[1]}"

    # Test forward
    print("\nTesting forward...")
    result = join_op.forward([rel1, rel2])
    print(f"Result embedding shapes: {result.embedding_shapes}")
    print(f"Result embeddings length: {len(result.embeddings)}")

    # Verify forward results
    assert len(result.embeddings) == 2, "Should have 2 aligned embeddings"
    assert result.embeddings[0].shape == (3, 10), f"First embedding should be (3, 10), got {result.embeddings[0].shape}"
    assert result.embeddings[1].shape == (3, 8), f"Second embedding should be (3, 8), got {result.embeddings[1].shape}"

    # Verify that embeddings are properly aligned (check first row)
    # The first row should have id=1, so embeddings should match the first row of original data
    torch.testing.assert_close(result.embeddings[0][0], emb1[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result.embeddings[1][0], emb2[0], atol=1e-6, rtol=1e-6)

    print("✅ Join operation tests passed successfully!")

    # Test error handling

    # Test with insufficient inputs
    input_schemas = [['id', 'name'], ['id', 'name']]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 0)],
        "right_refs": [ColumnRef(1, 0)],
        "key_names": ["id"]
    }]
    join_op = Join(output_schema=['id', 'name'], merge_steps=merge_steps, input_schemas=input_schemas)

    # Create a test relation for the error test
    df1 = create_test_df({'id': [1, 2], 'name': ['A', 'B']})
    rel1 = EmbeddedRelation(
        content_schema=['id', 'name'],
        embedding_shapes=[(2, 5)],
        content=df1,
        embeddings=[torch.randn(2, 5, device=device)]
    )

    try:
        join_op.instantiate([rel1])  # Only one input
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "requires at least 2 input relations" in str(e)

    # Test forward without instantiate
    df2 = create_test_df({'id': [1, 2], 'city': ['X', 'Y']})

    rel2 = EmbeddedRelation(
        content_schema=['id', 'city'],
        embedding_shapes=[(2, 5)],
        content=df2,
        embeddings=[torch.randn(2, 5)]
    )

    try:
        join_op.forward([rel1, rel2])
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "No cached join found" in str(e)

    print("✅ Error handling tests passed!")

# %% [markdown]
# ## Transformation Operation

# %%
# TODO: Do make sure that we actually split the Transformation node into a graph of nodes.
#       The input of this graph will be based on the z1 and z2 inputs, each node will be a 
#       single torch op, and there will be a emb_project before them. 
#       so Concat(z1,z2) will be mapped to emb_proj(z1) emb_proj(z2) -> concat -> ...

# TODO: When we calculate Loss and use `global aggregation` we split the global agg to happen on 
#       the embeddings in Transformation node and on the content in Aggregation node.

class Transformation(nn.Module):
    """
    Embedding Map - Transformation operation for embedded relations.

    Note: We assume that the output of the transformation is always a single tensor, not a list of tensors.
    """

    def __init__(self, transformation: nn.Module, output_schema: Optional[List[str]] = None):
        super().__init__()
        self.transformation = transformation
        self.output_schema: Optional[List[str]] = list(output_schema) if output_schema is not None else None
        self._output_shape_templates = None  # Cache for output embedding shapes (excluding first dim)
        self._original_predicted_shapes = None  # Cache for original predicted shapes (to detect scalars)
        self._encode_only_no_embedding_inputs = False

    def _module_device_dtype(self, son=None):
        # If it's an nn.Module, infer device/dtype from params/buffers
        if isinstance(self.transformation, nn.Module):
            for p in self.transformation.parameters(recurse=True):
                return p.device, p.dtype
            # Skip integer buffers (e.g. num_batches_tracked) — they don't
            # represent the module's computational dtype.
            buf_device = None
            for b in self.transformation.buffers(recurse=True):
                if buf_device is None:
                    buf_device = getattr(b, "device", torch.device("cpu"))
                b_dtype = getattr(b, "dtype", torch.float32)
                if b_dtype.is_floating_point:
                    return buf_device, b_dtype
            # Only integer buffers (or none at all) — fall back to input.
            # Pick the FIRST FLOAT embedding rather than just embeddings[0]:
            # multi-arg ops (e.g. CrossEntropyLoss(predictions, targets)) take
            # both Float predictions and Long targets, and the embedding order
            # depends on the parent join's input_order — which can be commuted
            # by R1. Selecting `embeddings[0]` blindly returns the WRONG
            # mod_dtype when position 0 happens to be a Long target tensor;
            # the downstream `if t.dtype.is_floating_point: t = t.to(mod_dtype)`
            # cast then converts every Float embedding to Long, breaking
            # log_softmax / matmul kernels with an "X not implemented for 'Long'"
            # runtime error.
            if son is not None and getattr(son, "embeddings", None):
                for e in son.embeddings:
                    if e.dtype.is_floating_point:
                        return (buf_device or e.device), e.dtype
                # No float embedding — pick first emb's device but default to
                # float32 (any further `if t.is_floating_point:` cast skips
                # non-float embeddings, so float32 is a safe sentinel).
                e0 = son.embeddings[0]
                return (buf_device or e0.device), torch.float32
            return (buf_device or torch.device("cpu")), torch.float32
        # If it's a function/callable, fall back to first FLOAT input
        # embedding (same rationale as the nn.Module branch above).
        if son is not None and getattr(son, "embeddings", None):
            for e in son.embeddings:
                if e.dtype.is_floating_point:
                    return e.device, e.dtype
            e0 = son.embeddings[0]
            return e0.device, torch.float32
        # Final fallback
        return torch.device("cpu"), torch.float32

    def _reset_encode_caches_for_instantiate(self) -> None:
        from relann.tensor_term_compiler import collect_column_extract_leaves
        for leaf in collect_column_extract_leaves(self.transformation):
            leaf.clear_cache()

    def _inject_encode_source(self, son: EmbeddedRelation) -> None:
        """Wire content DataFrame into ``_ColumnExtractModule`` leaves for RHS ``[...]`` encode."""
        from relann.tensor_term_compiler import collect_column_extract_leaves
        mod_device, mod_dtype = self._module_device_dtype(son=son)
        for leaf in collect_column_extract_leaves(self.transformation):
            leaf._source_er = son
            leaf._target_device = mod_device
            leaf._target_dtype = mod_dtype

    def _reject_bare_text_columns(self, son: EmbeddedRelation) -> None:
        """Raise if a bare ``[col]`` bracket targets a text/object column.

        Text columns must be wrapped in an explicit encoder, e.g.
        ``[HashBucketTextEncoder()(bio)]``. Only numeric/bool/categorical
        columns can be auto-tensorized by a bare ``[col]``.
        """
        from relann.encode import EncodeTypeError, is_text_dtype
        from relann.tensor_term_compiler import _EncodeWrapper, collect_column_extract_leaves

        if son.content is None:
            return
        wrapped_extractor_ids = {
            id(m.extractor)
            for m in self.transformation.modules()
            if isinstance(m, _EncodeWrapper)
        }
        for leaf in collect_column_extract_leaves(self.transformation):
            if id(leaf) in wrapped_extractor_ids:
                continue
            col = leaf.column_name
            if col in son.content.columns and is_text_dtype(son.content[col]):
                raise EncodeTypeError(
                    f"Text column {col!r} needs an explicit encoder in its bracket, "
                    f"e.g. [HashBucketTextEncoder()({col})]. A bare [{col}] supports "
                    f"only numeric, bool, or categorical columns."
                )

    def instantiate(self, sons: List[EmbeddedRelation], ctx: Optional["ExecutionContext"] = None) -> EmbeddedRelation:
        if sons is None:
            raise ValueError(f"Transformation operation requires exactly 1 input relation. Got None (sons is None)")
        if len(sons) != 1:
            raise ValueError(f"Transformation operation requires exactly 1 input relation. Got {len(sons)}")

        son = sons[0]

        # Apply logical output schema rename (aliasing) to the content DataFrame, if requested.
        df_out, schema_out = _maybe_apply_output_schema(son.content, son.content_schema, self.output_schema)

        # Compute and cache the output shape templates (excluding first dim) if not already done
        if self._output_shape_templates is None:
            from relann.tensor_term_compiler import collect_column_extract_leaves

            input_shapes = son.embedding_shapes
            if not input_shapes and son.embeddings:
                input_shapes = [e.shape for e in son.embeddings]
            encode_leaves = collect_column_extract_leaves(self.transformation)
            if not input_shapes:
                if encode_leaves and son.content is not None:
                    self._encode_only_no_embedding_inputs = True
                else:
                    raise RuntimeError("Transformation.instantiate: missing embedding_shapes and embeddings.")
            else:
                self._encode_only_no_embedding_inputs = False

            mod_device, mod_dtype = self._module_device_dtype(son=son)

            dummy_inputs = []
            if not self._encode_only_no_embedding_inputs:
                for shape in input_shapes:
                    # Replace first dim with 1 (1 row), keep rest
                    dummy_shape = (1,) + tuple(shape[1:])
                    dummy_inputs.append(torch.zeros(dummy_shape, dtype=mod_dtype, device=mod_device))

            self._reject_bare_text_columns(son)
            self._reset_encode_caches_for_instantiate()
            self._inject_encode_source(son)
            with torch.no_grad():
                dummy_output = self.transformation(*dummy_inputs)
                # We assume dummy_output is a tensor, not a list of tensors
                if isinstance(dummy_output, (list, tuple)):
                    if len(dummy_output) != 1:
                        raise RuntimeError("Transformation expects a single output tensor.")
                    dummy_output = dummy_output[0]
                if not torch.is_tensor(dummy_output):
                    raise RuntimeError("Transformation must return a torch.Tensor.")
                predicted_shapes = [dummy_output.shape]

            # Store only the shape templates (excluding first dim)
            # For scalar outputs (e.g., loss functions), the shape is () and shape[1:] is also ()
            # We need to detect this case and handle it specially
            self._output_shape_templates = [shape[1:] for shape in predicted_shapes]
            # Store the original predicted shapes to detect scalars
            self._original_predicted_shapes = predicted_shapes

        # Determine output embedding shapes
        # For scalar outputs (loss functions), use the original shape directly
        # For non-scalar outputs, replace the first dim with the current number of rows
        num_rows = df_out.shape[0]
        output_embedding_shapes = []
        for i, shape_rest in enumerate(self._output_shape_templates):
            original_shape = self._original_predicted_shapes[i]
            # Check if output is scalar (empty shape)
            # A scalar output from a loss function will have shape () even with batch input
            if len(original_shape) == 0:
                # Scalar output: use as-is (shape ())
                output_embedding_shapes.append(original_shape)
            else:
                # Non-scalar output: replace first dim with num_rows
                output_embedding_shapes.append((num_rows,) + tuple(shape_rest))

        return EmbeddedRelation(
            content_schema=schema_out,
            embedding_shapes=output_embedding_shapes,
            content=df_out,
            embeddings=None,  # No embeddings computed during instantiate
            column_vocabs=getattr(son, "column_vocabs", None),
            data_version=int(getattr(son, "data_version", 0)),
        )

    def forward(self, sons: List[EmbeddedRelation], ctx: Optional["ExecutionContext"] = None) -> EmbeddedRelation:
        if len(sons) != 1:
            raise ValueError(f"Transformation operation requires exactly 1 input relation. Got {len(sons)}")

        son = sons[0]
        if not son.embeddings:
            if not getattr(self, "_encode_only_no_embedding_inputs", False):
                raise RuntimeError("Transformation.forward: input relation has no embeddings.")

        # Apply logical output schema rename (aliasing) to the content DataFrame, if requested.
        df_out, schema_out = _maybe_apply_output_schema(son.content, son.content_schema, self.output_schema)

        # Apply transformation to embeddings (move to module device). Only cast floating
        # tensors to the module dtype so integer index columns (e.g. for nn.Embedding) stay long.
        mod_device, mod_dtype = self._module_device_dtype(son=son)
        inputs: List[torch.Tensor] = []
        if son.embeddings:
            for e in son.embeddings:
                t = e.to(device=mod_device)
                if t.dtype.is_floating_point:
                    t = t.to(dtype=mod_dtype)
                inputs.append(t)
        self._inject_encode_source(son)
        out = self.transformation(*inputs)
        if isinstance(out, (list, tuple)):
            if len(out) != 1:
                raise RuntimeError("Transformation expects a single output tensor.")
            out = out[0]
        if not torch.is_tensor(out):
            raise RuntimeError("Transformation must return a torch.Tensor.")

        transformed_embeddings = [out]

        n_content = df_out.shape[0]
        if out.dim() == 0:
            raise RuntimeError(
                "Transformation produced scalar output. Loss functions must produce "
                "per-row output (reduction='none'). Check that the module inherits "
                "from nn.modules.loss._Loss or returns per-row tensors."
            )
        if out.shape[0] != n_content:
            raise RuntimeError(
                f"Transformation produced {out.shape[0]} embedding rows but content "
                f"has {n_content} rows. Transformations must be per-row; loss "
                f"functions should use reduction='none' and let Aggregation reduce."
            )

        return EmbeddedRelation(
            content_schema=schema_out,
            embedding_shapes=[emb.shape for emb in transformed_embeddings],
            content=df_out,
            embeddings=transformed_embeddings,
            column_vocabs=getattr(son, "column_vocabs", None),
            data_version=int(getattr(son, "data_version", 0)),
        )

# %%
if __name__ == "__main__":
    # Test for Transformation operation

    # Create test data
    df = create_test_df({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35]
    })

    # Create embeddings for the relation
    emb = torch.randn(3, 5, device=device)  # 3 rows, 5 features

    # Create EmbeddedRelation object
    rel = EmbeddedRelation(
        content_schema=['id', 'name', 'age'],
        embedding_shapes=[(3, 5)],
        content=df,
        embeddings=[emb]
    )

    # Create a simple transformation (linear layer)
    transformation = nn.Linear(5, 3, device=device)  # Transform from 5 to 3 features

    # Create Transformation operation
    transform_op = Transformation(transformation=transformation)

    # Test instantiate
    print("Testing instantiate...")
    transformed_rel = transform_op.instantiate([rel])

    print(f"Original embedding shapes: {rel.embedding_shapes}")
    print(f"Transformed embedding shapes: {transformed_rel.embedding_shapes}")
    print(f"Content schema: {transformed_rel.content_schema}")

    # Verify instantiate results
    assert transformed_rel.content_schema == rel.content_schema, "Content schema should remain the same"
    # The embedding shape should be updated to reflect the transformation (e.g., (3, 3) for Linear(5, 3))
    expected_shape = (3, 3)
    assert transformed_rel.embedding_shapes[0] == expected_shape, f"Embedding shape should be {expected_shape} after transformation, got {transformed_rel.embedding_shapes[0]}"
    assert transformed_rel.embeddings is None, "No embeddings should be computed during instantiate"

    # Test forward
    print("\nTesting forward...")
    result = transform_op.forward([rel])

    print(f"Result embedding shapes: {result.embedding_shapes}")
    print(f"Result embeddings length: {len(result.embeddings)}")
    print(f"Transformed embedding shape: {result.embeddings[0].shape}")

    # Verify forward results
    assert len(result.embeddings) == 1, "Should have 1 transformed embedding"
    assert result.embeddings[0].shape == (3, 3), f"Transformed embedding should be (3, 3), got {result.embeddings[0].shape}"
    assert result.content_schema == rel.content_schema, "Content schema should remain the same"

    # Verify that transformation was actually applied
    # The transformed embedding should be different from the original
    # Note: We can't directly compare shapes since transformation changes dimensions
    # Instead, verify the transformation was applied by checking the output shape
    assert result.embeddings[0].shape == (3, 3), f"Transformed embedding should be (3, 3), got {result.embeddings[0].shape}"
    assert emb.shape == (3, 5), f"Original embedding should be (3, 5), got {emb.shape}"
    print("✅ Transformation operation tests passed successfully!")

    # Test error handling

    # Test with wrong number of inputs
    transformation = nn.Linear(5, 3, device=device)
    transform_op = Transformation(transformation=transformation)

    # Create test relations for the error tests
    df1 = create_test_df({'id': [1, 2], 'name': ['A', 'B']})
    rel1 = EmbeddedRelation(
        content_schema=['id', 'name'],
        embedding_shapes=[(2, 5)],
        content=df1,
        embeddings=[torch.randn(2, 5, device=device)]
    )

    df2 = create_test_df({'id': [3, 4], 'name': ['C', 'D']})
    rel2 = EmbeddedRelation(
        content_schema=['id', 'name'],
        embedding_shapes=[(2, 5)],
        content=df2,
        embeddings=[torch.randn(2, 5, device=device)]
    )

    # Test with no inputs
    try:
        transform_op.instantiate([])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "requires exactly 1 input relation" in str(e)

    # Test with too many inputs
    try:
        transform_op.instantiate([rel1, rel2])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "requires exactly 1 input relation" in str(e)

    # Test forward with wrong number of inputs
    try:
        transform_op.forward([rel1, rel2])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "requires exactly 1 input relation" in str(e)

    print("✅ Error handling tests passed!")

# %% [markdown]
# ## DataLoader Operation

# %% [markdown]
# ### Registry and Utilities

# %%
class DataLoader(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._cached_content = None
        self._cached_schema = None
        self._cached_embedding_shapes = None

    def instantiate(self, sons: list = None, ctx: Optional["ExecutionContext"] = None) -> EmbeddedRelation:
        if sons is not None and len(sons) > 0:
            raise ValueError("DataLoader is a leaf in the term graph and does not accept input relations.")
        if ctx is None:
            raise ValueError("DataLoader requires ExecutionContext (relations).")
        if self.name not in ctx.relations:
            raise KeyError(f"DataLoader Error: Relation '{self.name}' not found in context.relations.")
        rel = ctx.relations[self.name]
        # rel is expected to be a dict with 'content', 'content_schema', 'embedding_shapes', 'embeddings'
        content = rel['content']
        content_schema = rel['content_schema']
        embedding_shapes = rel.get('embedding_shapes', [])
        self._cached_content = content
        self._cached_schema = content_schema
        self._cached_embedding_shapes = embedding_shapes
        return EmbeddedRelation(
            content_schema=content_schema,
            embedding_shapes=embedding_shapes,
            content=content,
            embeddings=None,  # No embeddings in instantiate
            column_vocabs=rel.get("column_vocabs"),
            data_version=int(rel.get("data_version", 0)),
        )

    def forward(self, sons: list = None, ctx: Optional["ExecutionContext"] = None) -> EmbeddedRelation:
        if sons is not None and len(sons) > 0:
            raise ValueError("DataLoader operation does not accept input relations.")
        if ctx is None:
            raise ValueError("DataLoader requires ExecutionContext (relations).")
        if self.name not in ctx.relations:
            raise KeyError(f"Relation '{self.name}' not found in context.relations.")
        rel = ctx.relations[self.name]
        content = rel['content']
        content_schema = rel['content_schema']
        embedding_shapes = rel.get('embedding_shapes', [])
        embeddings = rel.get('embeddings', None)
        return EmbeddedRelation(
            content_schema=content_schema,
            embedding_shapes=embedding_shapes,
            content=content,
            embeddings=embeddings,
            column_vocabs=rel.get("column_vocabs"),
            data_version=int(rel.get("data_version", 0)),
        )

# %%
if __name__ == "__main__":
    # Test for DataLoader operation

    # Prepare test data
    df = create_test_df({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35]
    })
    emb = torch.randn(3, 5, device=device)
    rel_dict = {
        'content': df,
        'content_schema': ['id', 'name', 'age'],
        'embedding_shapes': [emb.shape],
        'embeddings': [emb]
    }
    ctx = ExecutionContext(relations={'test_rel': rel_dict})

    # Instantiate DataLoader
    loader = DataLoader('test_rel')

    # Test instantiate
    rel_inst = loader.instantiate(ctx=ctx)
    assert isinstance(rel_inst, EmbeddedRelation)
    assert rel_inst.content_schema == ['id', 'name', 'age']
    assert rel_inst.embedding_shapes == [emb.shape]
    assert rel_inst.content.equals(df)
    assert rel_inst.embeddings is None  # instantiate should not return embeddings

    # Test forward
    rel_fwd = loader.forward(ctx=ctx)
    assert isinstance(rel_fwd, EmbeddedRelation)
    assert rel_fwd.content_schema == ['id', 'name', 'age']
    assert rel_fwd.embedding_shapes == [emb.shape]
    assert rel_fwd.content.equals(df)
    assert rel_fwd.embeddings is not None
    assert len(rel_fwd.embeddings) == 1
    torch.testing.assert_close(rel_fwd.embeddings[0], emb, atol=1e-6, rtol=1e-6)

    # Test error: DataLoader as non-leaf
    try:
        loader.instantiate([rel_inst], ctx=ctx)
        assert False, "Should have raised ValueError for non-leaf instantiate"
    except ValueError as e:
        assert "does not accept input relations" in str(e)

    try:
        loader.forward([rel_inst], ctx=ctx)
        assert False, "Should have raised ValueError for non-leaf forward"
    except ValueError as e:
        assert "does not accept input relations" in str(e)

    # Test error: missing relation
    loader_missing = DataLoader('missing_rel')
    ctx_empty = ExecutionContext(relations={})
    try:
        loader_missing.instantiate(ctx=ctx_empty)
        assert False, "Should have raised KeyError for missing relation"
    except (KeyError, ValueError) as e:
        assert "not found in context.relations" in str(e) or "requires ExecutionContext" in str(e)

    try:
        loader_missing.forward(ctx=ctx_empty)
        assert False, "Should have raised KeyError for missing relation"
    except (KeyError, ValueError) as e:
        assert "not found in context.relations" in str(e) or "requires ExecutionContext" in str(e)

    print("✅ DataLoader operation tests passed successfully!")

# %% [markdown]
# ## Selection Operation

# %%
## Selection Operation

class Selection(nn.Module):
    """
    Selection operator: filters rows in EmbeddedRelation content and embeddings
    based on ComparisonExpression objects. Multiple filters are combined with AND logic.
    """

    def __init__(self, filter_expressions: List[ComparisonExpression]):
        super().__init__()
        self.filter_expressions = filter_expressions
        self._cached_filter_mask = None  # CPU tensor of row indices
        self._cached_filtered_content = None  # Store filtered DataFrame from instantiate

    def _evaluate_arith_term_for_df(
        self, 
        term: ArithTerm, 
        df, 
        schema: List[str]
    ) -> Any:
        """
        Evaluate an ArithTerm in the context of a DataFrame.
        
        Args:
            term: ArithTerm to evaluate
            df: DataFrame (pandas or cuDF)
            schema: List of column names in the DataFrame
            
        Returns:
            Series for column references, scalar for constants, Series for operations
        """
        from relann.arith_eval import evaluate_arith_term

        def resolve_var(var: Var) -> Any:
            """Resolve a Var to a DataFrame column Series."""
            if var.name not in schema:
                raise ValueError(
                    f"Selection: Column '{var.name}' not found in schema {schema}. "
                    f"Available columns: {list(df.columns) if hasattr(df, 'columns') else 'unknown'}"
                )
            return df[var.name]

        return evaluate_arith_term(term, resolve_var)

    def _evaluate_comparison(
        self, 
        expr: ComparisonExpression, 
        df, 
        schema: List[str]
    ) -> Any:
        """
        Evaluate a ComparisonExpression to a boolean Series.
        
        Args:
            expr: ComparisonExpression to evaluate
            df: DataFrame (pandas or cuDF)
            schema: List of column names in the DataFrame
            
        Returns:
            Boolean Series (pandas or cuDF)
        """
        lhs = self._evaluate_arith_term_for_df(expr.lhs, df, schema)
        rhs = self._evaluate_arith_term_for_df(expr.rhs, df, schema)
        
        # Apply comparison operator
        match expr.comp_op:
            case "==":
                return lhs == rhs
            case "!=":
                return lhs != rhs
            case ">":
                return lhs > rhs
            case ">=":
                return lhs >= rhs
            case "<":
                return lhs < rhs
            case "<=":
                return lhs <= rhs
            case _:
                raise NotImplementedError(f"Unsupported comparison operator '{expr.comp_op}'")

    def instantiate(self, sons: List[EmbeddedRelation], ctx: Optional["ExecutionContext"] = None) -> EmbeddedRelation:
        """
        Evaluate and cache filter indices; returns filtered EmbeddedRelation.
        """
        if len(sons) != 1:
            raise ValueError(f"Selection operation requires exactly 1 input relation. Got {len(sons)}")
        
        son = sons[0]
        df = son.content
        
        if df is None:
            raise ValueError("Selection: input relation has no content DataFrame")
        
        # Get schema from content_schema or DataFrame columns
        schema = son.content_schema
        if not schema and hasattr(df, 'columns'):
            schema = list(df.columns)
        if not schema:
            raise ValueError("Selection: cannot determine schema from input relation")
        
        # Evaluate all filter expressions and combine with AND logic
        if not self.filter_expressions:
            # No filters: pass through unchanged
            self._cached_filter_mask = torch.arange(len(df), dtype=torch.long, device="cpu")
            filtered_df = df
        else:
            # Evaluate each filter expression
            masks = []
            for expr in self.filter_expressions:
                mask = self._evaluate_comparison(expr, df, schema)
                masks.append(mask)
            
            # Combine all masks with AND logic
            if masks:
                combined_mask = masks[0]
                for mask in masks[1:]:
                    combined_mask = combined_mask & mask
            else:
                # No filters found at this point, which should not happen. Raise an error for clarity.
                raise RuntimeError("Selection: No filter expressions provided and mask logic reached unreachable state.")
            
            # Apply filter to DataFrame
            filtered_df = df[combined_mask].reset_index(drop=True)
            
            # Convert boolean mask to row indices (CPU tensor)
            # Get indices where mask is True
            if Join._is_cudf_df(combined_mask):
                import cupy as cp
                indices = cp.where(combined_mask.values)[0]
                self._cached_filter_mask = torch.as_tensor(cp.asnumpy(indices), dtype=torch.long, device="cpu")
            else:
                import numpy as np
                indices = np.where(combined_mask.values)[0]
                self._cached_filter_mask = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        
        # Cache the filtered content for use in forward
        self._cached_filtered_content = filtered_df

        # Update embedding shapes (number of rows may have changed)
        num_rows = len(filtered_df)
        updated_embedding_shapes = [
            (num_rows,) + tuple(shape[1:]) for shape in son.embedding_shapes
        ] if son.embedding_shapes else []
        
        return EmbeddedRelation(
            content_schema=schema,  # Schema unchanged (only rows filtered)
            embedding_shapes=updated_embedding_shapes,
            content=filtered_df,
            embeddings=None,  # No embeddings computed during instantiate
            column_vocabs=getattr(son, "column_vocabs", None),
            data_version=int(getattr(son, "data_version", 0)),
        )

    def forward(self, sons: List[EmbeddedRelation], ctx: Optional["ExecutionContext"] = None) -> EmbeddedRelation:
        """
        Use cached mask to filter embeddings and return cached filtered content.
        """
        if len(sons) != 1:
            raise ValueError(f"Selection operation requires exactly 1 input relation. Got {len(sons)}")
        
        if self._cached_filter_mask is None:
            raise RuntimeError(
                "No cached filter mask found. Did you forget to call instantiate() before forward()?"
            )
        if self._cached_filtered_content is None:
            raise RuntimeError(
                "No cached filtered content found. Did you forget to call instantiate() before forward()?"
            )
        
        son = sons[0]
        if not son.embeddings:
            raise RuntimeError("Selection.forward: input relation has no embeddings.")
        
        # Use cached indices to filter embeddings
        filtered_embeddings = []
        for emb in son.embeddings:
            # Move indices to the same device as the embedding
            idx_dev = self._cached_filter_mask.to(device=emb.device, dtype=torch.long, non_blocking=True)
            filtered_emb = emb.index_select(0, idx_dev)
            filtered_embeddings.append(filtered_emb)
        
        # Get the filtered content cached in instantiate
        df = self._cached_filtered_content
        schema = son.content_schema or (list(df.columns) if hasattr(df, 'columns') else [])
        
        return EmbeddedRelation(
            content_schema=schema,
            embedding_shapes=[emb.shape for emb in filtered_embeddings],
            content=df,
            embeddings=filtered_embeddings,
            column_vocabs=getattr(son, "column_vocabs", None),
            data_version=int(getattr(son, "data_version", 0)),
        )

# %%
if __name__ == "__main__":
    # Test for Selection operation


    # Create test data
    df = create_test_df({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 45],
        'salary': [50000, 60000, 70000, 80000, 90000]
    })

    emb = torch.randn(5, 10, device=device)  # 5 rows, 10 features
    rel = EmbeddedRelation(
        content_schema=['id', 'name', 'age', 'salary'],
        embedding_shapes=[(5, 10)],
        content=df,
        embeddings=[emb]
    )

    # Test 1: Simple filter - age > 30
    filter_expr1 = ComparisonExpression(
        lhs=ArithTerm(value=Var(name='age')),
        comp_op='>',
        rhs=ArithTerm(value=30)
    )
    sel_op1 = Selection(filter_expressions=[filter_expr1])

    # Test instantiate
    filtered_rel1 = sel_op1.instantiate([rel])
    assert filtered_rel1.content.shape[0] == 3, f"Expected 3 rows, got {filtered_rel1.content.shape[0]}"
    assert (filtered_rel1.content['age'] > 30).all(), "All filtered rows should have age > 30"
    assert filtered_rel1.embedding_shapes[0] == (3, 10), "Embedding shape should be (3, 10)"

    # Test forward
    result_rel1 = sel_op1.forward([rel])
    assert result_rel1.content.shape[0] == 3, "Forward: Expected 3 rows"
    assert result_rel1.embeddings[0].shape[0] == 3, "Forward: Embeddings should have 3 rows"

    # Test 2: Multiple filters (AND logic) - age > 25 AND salary < 80000
    filter_expr2a = ComparisonExpression(
        lhs=ArithTerm(value=Var(name='age')),
        comp_op='>',
        rhs=ArithTerm(value=25)
    )
    filter_expr2b = ComparisonExpression(
        lhs=ArithTerm(value=Var(name='salary')),
        comp_op='<',
        rhs=ArithTerm(value=80000)
    )
    sel_op2 = Selection(filter_expressions=[filter_expr2a, filter_expr2b])

    filtered_rel2 = sel_op2.instantiate([rel])
    assert filtered_rel2.content.shape[0] == 2, f"Expected 2 rows, got {filtered_rel2.content.shape[0]}"
    assert (filtered_rel2.content['age'] > 25).all(), "All rows should have age > 25"
    assert (filtered_rel2.content['salary'] < 80000).all(), "All rows should have salary < 80000"

    # Test 3: Arithmetic expression - age + 5 > 35
    filter_expr3 = ComparisonExpression(
        lhs=ArithTerm(op='+', sons=[
            ArithTerm(value=Var(name='age')),
            ArithTerm(value=5)
        ]),
        comp_op='>',
        rhs=ArithTerm(value=35)
    )
    sel_op3 = Selection(filter_expressions=[filter_expr3])
    filtered_rel3 = sel_op3.instantiate([rel])
    assert filtered_rel3.content.shape[0] == 3, f"Expected 3 rows (age+5 > 35), got {filtered_rel3.content.shape[0]}"

    # Test 4: Equality filter - name == "Alice"
    filter_expr4 = ComparisonExpression(
        lhs=ArithTerm(value=Var(name='name')),
        comp_op='==',
        rhs=ArithTerm(value='Alice')
    )
    sel_op4 = Selection(filter_expressions=[filter_expr4])
    filtered_rel4 = sel_op4.instantiate([rel])
    assert filtered_rel4.content.shape[0] == 1, "Expected 1 row for name == 'Alice'"
    assert str(filtered_rel4.content['name'].iloc[0]) == 'Alice', "Row should have name == 'Alice'"

    # Test 5: Empty result
    filter_expr5 = ComparisonExpression(
        lhs=ArithTerm(value=Var(name='age')),
        comp_op='>',
        rhs=ArithTerm(value=100)
    )
    sel_op5 = Selection(filter_expressions=[filter_expr5])
    filtered_rel5 = sel_op5.instantiate([rel])
    assert filtered_rel5.content.shape[0] == 0, "Expected 0 rows for age > 100"

    # Test 6: No filters (pass through)
    sel_op6 = Selection(filter_expressions=[])
    filtered_rel6 = sel_op6.instantiate([rel])
    assert filtered_rel6.content.shape[0] == 5, "No filters should pass through all rows"

    print("✅ Selection operation tests passed successfully!")

# %%
class Zero(nn.Module):
    """
    Zero operator: replaces embeddings with zeros.

    This is used when a relation is referenced with a constant embedding variable
    (e.g., A(a_id; 0)) but context.relations contains actual embeddings for A.
    DataLoader stays a pure loader; the transformation is explicit in the graph.
    """

    def __init__(self):
        super().__init__()

    def instantiate(self, sons: List[EmbeddedRelation], ctx: Optional["ExecutionContext"] = None) -> EmbeddedRelation:
        if len(sons) != 1:
            raise ValueError(f"Zero operation requires exactly 1 input relation. Got {len(sons)}")

        son = sons[0]
        df = son.content
        if df is None:
            raise ValueError("Zero: input relation has no content DataFrame")

        schema = son.content_schema
        if not schema and hasattr(df, "columns"):
            schema = list(df.columns)
        if not schema:
            raise ValueError("Zero: cannot determine schema from input relation")

        return EmbeddedRelation(
            content_schema=schema,
            embedding_shapes=son.embedding_shapes,
            content=df,
            embeddings=None,
            column_vocabs=getattr(son, "column_vocabs", None),
            data_version=int(getattr(son, "data_version", 0)),
        )

    def forward(self, sons: List[EmbeddedRelation], ctx: Optional["ExecutionContext"] = None) -> EmbeddedRelation:
        if len(sons) != 1:
            raise ValueError(f"Zero operation requires exactly 1 input relation. Got {len(sons)}")

        son = sons[0]
        df = son.content
        if df is None:
            raise ValueError("Zero.forward: input relation has no content DataFrame")

        schema = son.content_schema or (list(df.columns) if hasattr(df, "columns") else [])

        if son.embeddings:
            embedding_shapes = [e.shape for e in son.embeddings]
            device = son.embeddings[0].device
            dtype = son.embeddings[0].dtype
        elif son.embedding_shapes:
            import warnings
            warnings.warn(
                "Zero.forward: input has no embeddings, using default device 'cpu' and dtype 'float32' for zero embeddings."
            )
            embedding_shapes = son.embedding_shapes
            device = torch.device("cpu")
            dtype = torch.float32
        else:
            raise RuntimeError(
                "Zero.forward: cannot infer embedding shape. Input has no embeddings and no embedding_shapes."
            )

        zero_embeddings = [
            torch.zeros(shape, dtype=dtype, device=device) for shape in embedding_shapes
        ]

        return EmbeddedRelation(
            content_schema=schema,
            embedding_shapes=embedding_shapes,
            content=df,
            embeddings=zero_embeddings,
            column_vocabs=getattr(son, "column_vocabs", None),
            data_version=int(getattr(son, "data_version", 0)),
        )

# %% [markdown]
# ## OrderBy Operation

# %%
class OrderBy(nn.Module):
    """
    OrderBy operator: sorts rows in EmbeddedRelation content by the first column
    in ascending order and reorders embeddings accordingly.
    """
    
    def __init__(self):
        super().__init__()
        self._cached_sort_indices = None  # CPU tensor
        self._cached_sorted_content = None
    
    def instantiate(self, sons: List[EmbeddedRelation], ctx: Optional["ExecutionContext"] = None) -> EmbeddedRelation:
        """
        Sort by first column, cache indices and sorted DataFrame.
        """
        if len(sons) != 1:
            raise ValueError(f"OrderBy operation requires exactly 1 input relation. Got {len(sons)}")
        
        son = sons[0]
        df = son.content
        
        if df is None:
            raise ValueError("OrderBy: input relation has no content DataFrame")
        
        # Get schema from content_schema or DataFrame columns
        schema = son.content_schema
        if not schema and hasattr(df, 'columns'):
            schema = list(df.columns)
        # Handle empty DataFrame or no columns: nothing to sort, pass through
        if not schema or len(df) == 0 or len(schema) == 0:
            self._cached_sort_indices = torch.arange(len(df), dtype=torch.long, device="cpu")
            self._cached_sorted_content = df.copy()
            return EmbeddedRelation(
                content_schema=schema,
                embedding_shapes=son.embedding_shapes,
                content=df.copy(),
                embeddings=None,
                column_vocabs=getattr(son, "column_vocabs", None),
                data_version=int(getattr(son, "data_version", 0)),
            )
        
        # Get first column name
        first_col = schema[0]
        
        # Sort DataFrame by first column (ascending)
        sorted_df = df.sort_values(by=first_col, ascending=True).reset_index(drop=True)
        
        # Get sort indices
        # For pandas: use get_indexer to map original indices to sorted indices
        # For cuDF: use argsort directly
        if Join._is_cudf_df(df):
            import cupy as cp
            # Get argsort indices (indices that would sort the original column)
            col = df[first_col]
            sort_indices = col.argsort()
            self._cached_sort_indices = torch.as_tensor(cp.asnumpy(sort_indices), dtype=torch.long, device="cpu")
        else:
            import numpy as np
            # Get argsort indices
            sort_indices = np.argsort(df[first_col].values)
            self._cached_sort_indices = torch.as_tensor(sort_indices, dtype=torch.long, device="cpu")
        
        # Cache sorted DataFrame
        self._cached_sorted_content = sorted_df
        
        # Schema and embedding shapes unchanged (only row order changes)
        return EmbeddedRelation(
            content_schema=schema,
            embedding_shapes=son.embedding_shapes,
            content=sorted_df,
            embeddings=None,  # No embeddings computed during instantiate
            column_vocabs=getattr(son, "column_vocabs", None),
            data_version=int(getattr(son, "data_version", 0)),
        )
    
    def forward(self, sons: List[EmbeddedRelation], ctx: Optional["ExecutionContext"] = None) -> EmbeddedRelation:
        """
        Use cached indices to reorder embeddings.
        """
        if len(sons) != 1:
            raise ValueError(f"OrderBy operation requires exactly 1 input relation. Got {len(sons)}")
        
        if self._cached_sort_indices is None:
            raise RuntimeError(
                "No cached sort indices found. Did you forget to call instantiate() before forward()?"
            )
        if self._cached_sorted_content is None:
            raise RuntimeError(
                "No cached sorted content found. Did you forget to call instantiate() before forward()?"
            )
        
        son = sons[0]
        if not son.embeddings:
            raise RuntimeError("OrderBy.forward: input relation has no embeddings.")
        
        # Use cached indices to reorder embeddings
        reordered_embeddings = []
        for emb in son.embeddings:
            idx_dev = self._cached_sort_indices.to(device=emb.device, dtype=torch.long, non_blocking=True)
            if idx_dev.nelement() == 0:
                # Nothing to reorder; just pass through
                reordered_embeddings.append(emb)
            else:
                reordered_embeddings.append(emb.index_select(0, idx_dev))
        
        # Get the sorted content cached in instantiate
        df = self._cached_sorted_content
        schema = son.content_schema or (list(df.columns) if hasattr(df, 'columns') else [])
        
        return EmbeddedRelation(
            content_schema=schema,
            embedding_shapes=[emb.shape for emb in reordered_embeddings],
            content=df,
            embeddings=reordered_embeddings,
            column_vocabs=getattr(son, "column_vocabs", None),
            data_version=int(getattr(son, "data_version", 0)),
        )

# %%
if __name__ == "__main__":
    # Test for OrderBy operation

    # Test 1: Numeric column sorting
    df1 = create_test_df({
        'id': [3, 1, 2],
        'name': ['Charlie', 'Alice', 'Bob'],
        'age': [35, 25, 30]
    })
    emb1 = torch.randn(3, 5, device=device)
    rel1 = EmbeddedRelation(
        content_schema=['id', 'name', 'age'],
        embedding_shapes=[(3, 5)],
        content=df1,
        embeddings=[emb1]
    )

    orderby_op1 = OrderBy()
    sorted_rel1 = orderby_op1.instantiate([rel1])
    assert (sorted_rel1.content['id'].to_pandas().tolist() if hasattr(sorted_rel1.content['id'], 'to_pandas') else sorted_rel1.content['id'].tolist()) == [1, 2, 3], "Should be sorted by id: [1, 2, 3]"
    assert sorted_rel1.content_schema == ['id', 'name', 'age'], "Schema should be unchanged"

    # Test forward
    result1 = orderby_op1.forward([rel1])
    assert (result1.content['id'].to_pandas().tolist() if hasattr(result1.content['id'], 'to_pandas') else result1.content['id'].tolist()) == [1, 2, 3], "Forward: Should be sorted by id"
    assert result1.embeddings[0].shape == (3, 5), "Embeddings should be reordered"
    # Verify embeddings are correctly reordered (first row should be original row 1, etc.)
    torch.testing.assert_close(result1.embeddings[0][0], emb1[1], atol=1e-6, rtol=1e-6)  # id=1 was originally at index 1

    # Test 2: String column sorting
    df2 = create_test_df({
        'name': ['c', 'a', 'b'],
        'value': [3, 1, 2]
    })
    emb2 = torch.randn(3, 4, device=device)
    rel2 = EmbeddedRelation(
        content_schema=['name', 'value'],
        embedding_shapes=[(3, 4)],
        content=df2,
        embeddings=[emb2]
    )

    orderby_op2 = OrderBy()
    sorted_rel2 = orderby_op2.instantiate([rel2])
    assert (sorted_rel2.content['name'].to_pandas().tolist() if hasattr(sorted_rel2.content['name'], 'to_pandas') else sorted_rel2.content['name'].tolist()) == ['a', 'b', 'c'], "Should be sorted by name: ['a', 'b', 'c']"

    # Test forward
    result2 = orderby_op2.forward([rel2])
    assert (result2.content['name'].to_pandas().tolist() if hasattr(result2.content['name'], 'to_pandas') else result2.content['name'].tolist()) == ['a', 'b', 'c'], "Forward: Should be sorted by name"
    assert result2.embeddings[0].shape == (3, 4), "Embeddings should be reordered"

    # Test 3: Empty DataFrame
    df3 = create_test_df({
        'id': [],
        'name': []
    })
    rel3 = EmbeddedRelation(
        content_schema=['id', 'name'],
        embedding_shapes=[(0, 3)],
        content=df3,
        embeddings=[torch.empty(0, 3, device=device)]
    )

    orderby_op3 = OrderBy()
    sorted_rel3 = orderby_op3.instantiate([rel3])
    assert len(sorted_rel3.content) == 0, "Empty DataFrame should remain empty"
    assert sorted_rel3.content_schema == ['id', 'name'], "Schema should be preserved"

    # Test forward with empty
    result3 = orderby_op3.forward([rel3])
    assert len(result3.content) == 0, "Forward: Empty DataFrame should remain empty"
    assert result3.embeddings[0].shape == (0, 3), "Empty embeddings should be preserved"

    print("✅ OrderBy operation tests passed successfully!")

# %% [markdown]
# ## Project Operation

# %%
# Note: Projection is currently handled within the aggregation node.
# In the future, we could optimize this by replacing agg with a separate type="project" node.

# %%
def _default_scatter_mean(x: torch.Tensor, idx: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Fallback mean aggregation via index_add_ (no torch_scatter dependency)."""
    num_g = int(idx.max().item()) + 1
    d = x.size(1) if x.dim() > 1 else 1
    x_2d = x.view(-1, d) if x.dim() == 1 else x  # (N,) -> (N, 1); (N, D) unchanged
    idx_dev = idx.to(x.device)
    sums = torch.zeros((num_g, d), device=x.device, dtype=x.dtype)
    sums.index_add_(0, idx_dev, x_2d)
    cnts = torch.zeros((num_g, 1), device=x.device, dtype=x.dtype)
    cnts.index_add_(0, idx_dev, torch.ones((x_2d.size(0), 1), device=x.device, dtype=x.dtype))
    return sums / torch.clamp(cnts, min=torch.finfo(x.dtype).eps)


# Factory — maps string names to aggregation functions
def get_aggregation_function(name: str) -> Callable:
    if name is None:
        return None
    name = name.lower()
    if name in ("mean", "avg"):
        return _require_torch_scatter().scatter_mean
    if name in ("sum", "add"):
        return _require_torch_scatter().scatter_add
    if name == "max":
        _scatter_max = _require_torch_scatter().scatter_max

        def _scatter_max_values(x, idx, dim=0):
            out, _ = _scatter_max(x, idx, dim=dim)
            return out

        return _scatter_max_values
    raise ValueError(f"Unknown aggregation_name={name!r}")

class Aggregation(nn.Module):
    """
    Aggregation over rows using a group index derived from `group_by_refs` (ColumnRefs).
    - Resolves ColumnRefs to column names during instantiate() using the single input's schema.
    - If `group_by_refs` is None or empty: global aggregation (single group).
    """

    def __init__(
        self,
        group_by_refs: Optional[List[ColumnRef]] = None,
        aggregation_function: Optional[Callable] = None,
        aggregation_name: Optional[str] = None,
        output_schema: Optional[List[str]] = None,
    ):
        super().__init__()
        self.group_by_refs: List[ColumnRef] = list(group_by_refs) if group_by_refs else []
        self.aggregation_function = aggregation_function
        self.aggregation_name = aggregation_name
        self.output_schema: Optional[List[str]] = list(output_schema) if output_schema is not None else None
        self._cached_group_idx = None
        self._cached_num_groups = None
        self._cached_output_content = None
        self._cached_output_schema = None
        self._cached_column_vocabs: Optional[Dict[str, Dict[Any, int]]] = None

    def instantiate(self, sons: List["EmbeddedRelation"], ctx: Optional["ExecutionContext"] = None) -> "EmbeddedRelation":
        if len(sons) != 1:
            raise ValueError(f"Aggregation operation requires exactly 1 input relation. Got {len(sons)}")
        son = sons[0]
        df = son.content.copy()
        # Normalize to logical schema so resolved keys (logical names) exist as columns
        schema = getattr(son, "content_schema", None)
        if schema and len(schema) == len(df.columns):
            df.columns = list(schema)

        # Empty group_by_refs is valid: means global aggregation (one group, single output row).
        keys = Join._resolve_normalized_refs(self.group_by_refs, [son]) if self.group_by_refs else []
        old_keys = list(keys)

        if keys:
            # Factorize groups (works for pandas/cuDF); get codes on CPU
            grp_codes = df.groupby(keys, sort=False).ngroup()
            grp_np = grp_codes.to_numpy()
            self._cached_group_idx = torch.as_tensor(grp_np, dtype=torch.long, device="cpu")

            # One row per group, keep only grouping columns
            uniq = df[keys].drop_duplicates()
            # reset index consistently for pandas/cuDF
            if hasattr(uniq, "reset_index"):
                uniq = uniq.reset_index(drop=True)
            output_content = uniq
            output_schema = list(keys)

            # If a logical output schema is provided (e.g. aliasing target_id -> node_id),
            # rename the grouping columns to match it.
            if self.output_schema is not None and len(self.output_schema) == len(output_schema):
                output_content = _rename_columns_safe(output_content, output_schema, list(self.output_schema))
                output_schema = list(self.output_schema)

            output_len = len(uniq)
            self._cached_num_groups = int(output_len)
        else:
            # Global aggregation: single group — one row, no columns
            self._cached_group_idx = torch.zeros(len(df), dtype=torch.long, device="cpu")
            self._cached_num_groups = 1
            if _is_df(df) and df.__class__.__module__.startswith("cudf"):
                import cudf
                output_content = cudf.DataFrame([{}])
            else:
                import pandas as pd
                output_content = pd.DataFrame([{}])
            output_schema = []
            output_len = 1

        self._cached_output_content = output_content
        self._cached_output_schema = output_schema

        sub_v = _project_column_vocabs(getattr(son, "column_vocabs", None), old_keys if old_keys else None)
        if sub_v and self.output_schema is not None and old_keys and len(self.output_schema) == len(old_keys):
            sub_v = {
                new: sub_v[old]
                for old, new in zip(old_keys, self.output_schema)
                if old in sub_v
            }
        self._cached_column_vocabs = sub_v

        expected_embedding_shapes = []
        if son.embedding_shapes:
            for emb_shape in son.embedding_shapes:
                expected_embedding_shapes.append((output_len, *emb_shape[1:]))

        return EmbeddedRelation(
            content_schema=output_schema,
            embedding_shapes=expected_embedding_shapes,
            content=output_content,
            embeddings=None,  # computed in forward
            column_vocabs=self._cached_column_vocabs,
            data_version=int(getattr(son, "data_version", 0)),
        )

    def forward(self, sons: List["EmbeddedRelation"], ctx: Optional["ExecutionContext"] = None) -> "EmbeddedRelation":
        if len(sons) != 1:
            raise ValueError(f"Aggregation operation requires exactly 1 input relation. Got {len(sons)}")
        if self._cached_group_idx is None or self._cached_num_groups is None:
            raise RuntimeError("No cached group index found. Did you forget to call instantiate() before forward()?")

        # Resolve aggregation function if missing (default to mean)
        agg_fn = (
            self.aggregation_function
            or get_aggregation_function(self.aggregation_name)
            or _default_scatter_mean
        )

        son = sons[0]
        
        aggregated_embeddings = []
        group_idx_cpu = self._cached_group_idx
        for emb in (son.embeddings or []):
            if emb.dim() == 0:
                raise RuntimeError(
                    "Aggregation received scalar embedding from upstream Transformation. "
                    "Loss functions must use reduction='none' for per-row output."
                )
            idx = group_idx_cpu.to(device=emb.device, dtype=torch.long, non_blocking=True)
            out = agg_fn(emb, idx, dim=0)
            aggregated_embeddings.append(out)

        return EmbeddedRelation(
            content_schema=self._cached_output_schema,
            embedding_shapes=[e.shape for e in aggregated_embeddings],
            content=self._cached_output_content,
            embeddings=aggregated_embeddings,
            column_vocabs=self._cached_column_vocabs,
            data_version=int(getattr(son, "data_version", 0)),
        )

# %%
if __name__ == "__main__":
    # Test for Aggregation operation

    # Create test data
    df = create_test_df({
        'id': [1, 2, 1, 2, 3],
        'group': ['A', 'A', 'B', 'B', 'A'],
        'value': [10, 20, 30, 40, 50]
    })

    # Create embeddings for the relation
    emb = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0]
    ], device=device)  # shape (5, 2)

    rel = EmbeddedRelation(
        content_schema=['id', 'group', 'value'],
        embedding_shapes=[(5, 2)],
        content=df,
        embeddings=[emb]
    )

    # Use torch_scatter.scatter_add directly as the aggregation function
    # group_by_refs: ColumnRef(0, 1) = second column ("group") in schema ['id', 'group', 'value']
    agg_op = Aggregation(group_by_refs=[ColumnRef(0, 1)], aggregation_function=scatter_add, aggregation_name="sum")

    # Test instantiate
    print("Testing instantiate (group-by sum)...")
    agg_rel = agg_op.instantiate([rel])
    print(f"Aggregated DataFrame:\n{agg_rel.content}")
    print(f"Expected embedding shapes: {agg_rel.embedding_shapes}")

    # Should have as many rows as unique groups
    assert agg_rel.content.shape[0] == df['group'].nunique(), "Output rows should match number of groups"
    assert agg_rel.embedding_shapes[0][0] == df['group'].nunique(), "Embedding shape should match number of groups"

    # Test forward
    print("\nTesting forward (group-by sum)...")
    result = agg_op.forward([rel])
    print(f"Result embedding shapes: {result.embedding_shapes}")
    print(f"Result embeddings:\n{result.embeddings[0]}")

    # Manually compute expected sums for each group
    try:
        import cudf
        if cudf is not None and df.__class__.__module__.startswith('cudf'):
            # cuDF-specific handling
            with cudf.option_context("mode.pandas_compatible", False):
                uniques = df["group"].unique()        # GPU Series, no .values call
            group_map = {g: i for i, g in enumerate(sorted(uniques.to_arrow().to_pylist()))}
            expected = torch.zeros(len(group_map), emb.shape[1], device=device)
            # Convert to pandas for iteration since cuDF doesn't support iterrows
            df_pandas = df.to_pandas()
            for i, row in df_pandas.iterrows():
                idx = group_map[row['group']]
                expected[idx] += emb[i]
        else:
            # pandas handling
            uniques = df["group"].unique()
            group_map = {g: i for i, g in enumerate(sorted(uniques.tolist()))}
            expected = torch.zeros(len(group_map), emb.shape[1], device=device)
            for i, row in df.iterrows():
                idx = group_map[row['group']]
                expected[idx] += emb[i]
    except (ImportError, AttributeError):
        # Fallback to pandas handling
        uniques = df["group"].unique()
        group_map = {g: i for i, g in enumerate(sorted(uniques.tolist()))}
        expected = torch.zeros(len(group_map), emb.shape[1], device=device)
        for i, row in df.iterrows():
            idx = group_map[row['group']]
            expected[idx] += emb[i]
    torch.testing.assert_close(result.embeddings[0], expected, atol=1e-6, rtol=1e-6)

    # Test global aggregation (no group-by)
    agg_op_global = Aggregation(group_by_refs=[], aggregation_function=scatter_add, aggregation_name="sum")
    print("\nTesting instantiate (global sum)...")
    agg_rel_global = agg_op_global.instantiate([rel])
    print(f"Aggregated DataFrame (global):\n{agg_rel_global.content}")
    print(f"Expected embedding shapes (global): {agg_rel_global.embedding_shapes}")

    print("\nTesting forward (global sum)...")
    result_global = agg_op_global.forward([rel])
    # Since no group-by, expect a single row (sum of all)
    expected_global = emb.sum(dim=0, keepdim=True)
    if result_global.embeddings:
        torch.testing.assert_close(result_global.embeddings[0], expected_global, atol=1e-6, rtol=1e-6)

    print("✅ Aggregation operation tests passed successfully!\n")

    # Test error handling for Aggregation

    # Test with insufficient inputs
    agg_op = Aggregation(group_by_refs=[ColumnRef(0, 0)], aggregation_function=lambda x, idx, dim=0: x, aggregation_name="sum")
    df = create_test_df({'group': ['A', 'B']})
    rel = EmbeddedRelation(
        content_schema=['group'],
        embedding_shapes=[(2, 3)],
        content=df,
        embeddings=[torch.randn(2, 3, device=device)]
    )

    try:
        agg_op.instantiate([])
        assert False, "Should have raised ValueError for no input"
    except ValueError as e:
        assert "requires exactly 1 input relation" in str(e)

    # Test forward without instantiate
    try:
        agg_op.forward([rel])
        assert False, "Should have raised RuntimeError for missing cache"
    except RuntimeError as e:
        assert "No cached group index found" in str(e)

    print("✅ Aggregation error handling tests passed!\n")

# %%
class Project(nn.Module):
    """
    Project operator: selects a subset of columns from the input relation's content DataFrame.
    The embeddings are passed through unchanged.
    """

    def __init__(self, project_keys):
        super().__init__()
        self.project_keys = [project_keys] if isinstance(project_keys, str) else list(project_keys)
        self._cached_projected_df = None

    def instantiate(self, sons, ctx: Optional["ExecutionContext"] = None):
        if len(sons) != 1:
            raise ValueError("Project operation requires exactly 1 input relation")
        son = sons[0]
        # Project the DataFrame to the specified columns
        projected_df = son.content[self.project_keys].copy()
        self._cached_projected_df = projected_df
        # Return a new EmbeddedRelation with projected content, same embeddings
        pv = _project_column_vocabs(getattr(son, "column_vocabs", None), self.project_keys)
        return EmbeddedRelation(
            content_schema=self.project_keys,
            embedding_shapes=son.embedding_shapes,
            content=projected_df,
            embeddings=son.embeddings,
            column_vocabs=pv,
            data_version=int(getattr(son, "data_version", 0)),
        )

    def forward(self, sons, ctx: Optional["ExecutionContext"] = None):
        if len(sons) != 1:
            raise ValueError("Project operation requires exactly 1 input relation")
        son = sons[0]
        if self._cached_projected_df is None:
            raise RuntimeError("No cached projected DataFrame found. Did you forget to call instantiate()?")
        # Pass through the embeddings unchanged
        pv = _project_column_vocabs(getattr(son, "column_vocabs", None), self.project_keys)
        return EmbeddedRelation(
            content_schema=self.project_keys,
            embedding_shapes=son.embedding_shapes,
            content=self._cached_projected_df,
            embeddings=son.embeddings,
            column_vocabs=pv,
            data_version=int(getattr(son, "data_version", 0)),
        )


class Rename(nn.Module):
    """Renames columns of the single input EmbeddedRelation in-place.

    Embeddings pass through unchanged. `mapping` is a list of (old, new)
    pairs (positional pairing): the i-th old name in input.content is
    renamed to the i-th new name. Optimizer ingests this when templated
    DSL bodies rename body-atom column names (e.g. `Emb(citing; z)` over
    `Emb(pid;...)`).
    """

    def __init__(self, mapping, output_schema=None):
        super().__init__()
        self.mapping = [tuple(pair) for pair in (mapping or [])]
        self.output_schema = list(output_schema) if output_schema else [
            new for _, new in self.mapping
        ]
        self._cached_renamed_df = None

    def _do_rename(self, son):
        if son.content is None:
            raise RuntimeError("Rename: input relation has no content DataFrame")
        rename_dict = {old: new for old, new in self.mapping if old != new}
        df = son.content.rename(columns=rename_dict) if rename_dict else son.content.copy()
        return df

    def instantiate(self, sons, ctx: Optional["ExecutionContext"] = None):
        if len(sons) != 1:
            raise ValueError("Rename operation requires exactly 1 input relation")
        son = sons[0]
        df = self._do_rename(son)
        self._cached_renamed_df = df
        return EmbeddedRelation(
            content_schema=self.output_schema,
            embedding_shapes=son.embedding_shapes,
            content=df,
            embeddings=son.embeddings,
        )

    def forward(self, sons, ctx: Optional["ExecutionContext"] = None):
        if len(sons) != 1:
            raise ValueError("Rename operation requires exactly 1 input relation")
        son = sons[0]
        if self._cached_renamed_df is None:
            self._cached_renamed_df = self._do_rename(son)
        return EmbeddedRelation(
            content_schema=self.output_schema,
            embedding_shapes=son.embedding_shapes,
            content=self._cached_renamed_df,
            embeddings=son.embeddings,
        )

# %%
if __name__ == "__main__":
    # Test for Project operation
    # Create test data
    df = create_test_df({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['NYC', 'LA', 'Chicago']
    })
    emb = torch.randn(3, 5)
    rel = EmbeddedRelation(
        content_schema=['id', 'name', 'age', 'city'],
        embedding_shapes=[(3, 5)],
        content=df,
        embeddings=[emb]
    )

    # Project to a subset of columns
    project_keys = ['id', 'city']
    proj_op = Project(project_keys=project_keys)

    # Test instantiate
    projected_rel = proj_op.instantiate([rel])
    assert list(projected_rel.content.columns) == project_keys, "Projected columns do not match"
    assert projected_rel.content_schema == project_keys, "Projected schema does not match"
    assert projected_rel.embeddings == rel.embeddings, "Embeddings should be unchanged"
    assert projected_rel.embedding_shapes == rel.embedding_shapes, "Embedding shapes should be unchanged"
    assert projected_rel.content.shape[0] == df.shape[0], "Row count should be unchanged"

    # Test forward
    result_rel = proj_op.forward([rel])
    assert list(result_rel.content.columns) == project_keys, "Forward: Projected columns do not match"
    assert result_rel.embeddings == rel.embeddings, "Forward: Embeddings should be unchanged"
    assert result_rel.content.equals(projected_rel.content), "Forward: Projected DataFrame does not match instantiate"

    print("✅ Project operation tests passed successfully!")

# %% [markdown]
# ## Union

# %%
class Union(nn.Module):
    """
    Union operator: concatenates multiple relations.
    Duplicate handling is done by a subsequent Aggregation operation.
    """
    
    def __init__(self):
        super().__init__()
        self._cached_union_df = None
    
    def instantiate(self, sons: List[EmbeddedRelation], ctx: Optional["ExecutionContext"] = None) -> EmbeddedRelation:
        if len(sons) < 2:
            raise ValueError(f"Union operation requires at least 2 input relations. Got {len(sons)}")
        
        # Concatenate all DataFrames
        dfs = []
        for son in sons:
            if son.content is None:
                raise ValueError(f"Union: input relation has no content DataFrame")
            dfs.append(son.content.copy())
        
        # Use pandas or cuDF concat based on first DataFrame
        use_cudf = Join._is_cudf_df(dfs[0])
        if use_cudf:
            import cudf
            df_union = cudf.concat(dfs, ignore_index=True)
        else:
            import pandas as pd
            df_union = pd.concat(dfs, ignore_index=True)
        
        # Cache the concatenated DataFrame (including duplicates)
        self._cached_union_df = df_union
        
        # Compute output embedding shapes (sum of all input rows)
        total_rows = sum(len(df) for df in dfs)
        output_embedding_shapes = []
        if sons[0].embedding_shapes:
            for emb_shape in sons[0].embedding_shapes:
                output_embedding_shapes.append((total_rows, *emb_shape[1:]))
        
        vmerge = _merge_column_vocabs(*[getattr(s, "column_vocabs", None) for s in sons])
        return EmbeddedRelation(
            content_schema=list(df_union.columns),
            embedding_shapes=output_embedding_shapes,
            content=df_union,
            embeddings=None,
            column_vocabs=vmerge,
            data_version=_max_data_version(sons),
        )
    
    def forward(self, sons: List[EmbeddedRelation], ctx: Optional["ExecutionContext"] = None) -> EmbeddedRelation:
        if self._cached_union_df is None:
            raise RuntimeError("No cached union found. Did you forget to call instantiate() before forward()?")
        
        # Collect first embedding from each input
        all_embeddings = []
        for son in sons:
            if son.embeddings and len(son.embeddings) > 0:
                all_embeddings.append(son.embeddings[0])
        
        if not all_embeddings:
            raise RuntimeError("Union.forward: no embeddings found in input relations")

        # Concatenate embeddings along first dimension
        concat_emb = torch.cat(all_embeddings, dim=0)
        
        vmerge = _merge_column_vocabs(*[getattr(s, "column_vocabs", None) for s in sons])
        return EmbeddedRelation(
            content_schema=list(self._cached_union_df.columns),
            embedding_shapes=[concat_emb.shape],
            content=self._cached_union_df,
            embeddings=[concat_emb],
            column_vocabs=vmerge,
            data_version=_max_data_version(sons),
        )

# %%
if __name__ == "__main__":
    # Test for Union operation
    df1 = create_test_df({'id': [1, 2, 3], 'value': ['a', 'b', 'c']})
    df2 = create_test_df({'id': [2, 4], 'value': ['b', 'd']})  # id=2, value='b' is duplicate

    emb1 = torch.randn(3, 4, device=device)
    emb2 = torch.randn(2, 4, device=device)

    rel1 = EmbeddedRelation(
        content_schema=['id', 'value'],
        embedding_shapes=[(3, 4)],
        content=df1,
        embeddings=[emb1]
    )

    rel2 = EmbeddedRelation(
        content_schema=['id', 'value'],
        embedding_shapes=[(2, 4)],
        content=df2,
        embeddings=[emb2]
    )

    union_op = Union()
    union_rel = union_op.instantiate([rel1, rel2])

    # Should have 5 rows total (3 + 2, including duplicate) - Union just concatenates
    assert len(union_rel.content) == 5, f"Expected 5 rows (concatenated), got {len(union_rel.content)}"

    result = union_op.forward([rel1, rel2])
    assert result.embeddings[0].shape[0] == 5, "Output should have 5 concatenated embeddings"

    print("✅ Union operation tests passed successfully!")

# %%
from relann.embedded_relation import _format_embedding_cell, pretty_print_er as pretty_print_er
