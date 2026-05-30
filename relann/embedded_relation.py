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
# # Embedded Relation
#
# > The EmbeddedRelation data structure: content (DataFrame) + embeddings (tensors), with display utilities.

# %%
import logging
import torch
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# %%
try:
    import cudf  # type: ignore
    HAS_CUDF = True
except Exception:
    cudf = None  # type: ignore
    HAS_CUDF = False

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    pd = None  # type: ignore
    HAS_PANDAS = False

# %% [markdown]
# ## EmbeddedRelation

# %%
class EmbeddedRelation:
    """
    Represents an embedded relation with content (pandas or cuDF) and embeddings (list of torch.Tensor).
    - Does NOT auto-convert pandas\u2192cuDF (prevents unintended GPU backend usage in CPU tests).
    - Optionally accepts target_device/target_dtype to normalize embeddings.
    """
    def __init__(
        self,
        content_schema: List[str],
        embedding_shapes: List[torch.Size],
        content: Optional[Union["pd.DataFrame", "cudf.DataFrame"]] = None,
        embeddings: Optional[List[torch.Tensor]] = None,
        *,
        target_device: Optional[torch.device] = None,
        target_dtype: Optional[torch.dtype] = None,
        column_vocabs: Optional[Dict[str, Dict[Any, int]]] = None,
        data_version: int = 0,
    ):
        self.content_schema = content_schema
        self.embedding_shapes = embedding_shapes
        self.column_vocabs = column_vocabs
        self.data_version = int(data_version)

        if content is not None:
            if HAS_PANDAS and isinstance(content, pd.DataFrame):
                self.content = content
            elif HAS_CUDF and cudf is not None and isinstance(content, cudf.DataFrame):  # type: ignore
                self.content = content
            else:
                self.content = content
        else:
            self.content = None

        if embeddings is not None:
            if target_device is not None or target_dtype is not None:
                self.embeddings = [
                    e.to(device=target_device or e.device, dtype=target_dtype or e.dtype) for e in embeddings
                ]
            else:
                self.embeddings = embeddings
        else:
            self.embeddings = None

    def __len__(self):
        return len(self.content) if self.content is not None else 0

    _EMBEDDING_COLOR = "#007BA7"  # Cerulean

    def __repr__(self):
        rows = len(self)
        schema_str = ", ".join(self.content_schema) if self.content_schema else "no schema"
        emb_info = ""
        if self.embeddings:
            shapes = [str(tuple(e.shape)) for e in self.embeddings]
            emb_info = f", embeddings=[{', '.join(shapes)}]"
        elif self.embedding_shapes:
            shapes = [str(tuple(s)) for s in self.embedding_shapes]
            emb_info = f", embedding_shapes=[{', '.join(shapes)}]"
        return f"EmbeddedRelation({rows} rows, [{schema_str}]{emb_info})"

    def _repr_html_(self, max_rows: int = 12, max_embedding_display: int = 6):
        """Jupyter rich display: styled table with blue embedding column."""
        if self.content is None:
            parts = ["EmbeddedRelation(no content)"]
            if self.embeddings:
                parts.append(f"  embeddings: {[tuple(e.shape) for e in self.embeddings]}")
            return "<pre>" + "\n".join(parts) + "</pre>"

        styler_or_df = self.to_df(max_rows=max_rows, max_embedding_display=max_embedding_display, style=True)
        html = styler_or_df.to_html() if hasattr(styler_or_df, "to_html") else styler_or_df.to_html()
        num_rows = len(self.content)
        if num_rows > max_rows:
            html += f"<p><em>[{num_rows} rows total]</em></p>"
        return html

    def to_df(self, max_rows=None, max_embedding_display=6, style=True):
        """Return a DataFrame with content columns and an optional styled embedding column.

        Args:
            max_rows: Maximum rows to include (None = all rows).
            max_embedding_display: Max scalar values shown per embedding cell.
            style: If True and embeddings exist, return a Styler with the embedding
                   column in cerulean (#007BA7). If False, return a plain DataFrame.
        """
        import pandas as _pd
        if self.content is None:
            return _pd.DataFrame()

        df = self.content.head(max_rows).copy() if max_rows is not None else self.content.copy()

        if self.embeddings and len(self.embeddings) > 0:
            embs_cpu = [e.cpu() if e.is_cuda else e for e in self.embeddings]
            n = len(df)
            if len(embs_cpu) == 1:
                df["embedding"] = [
                    _format_embedding_cell(embs_cpu[0], i, max_embedding_display) for i in range(n)
                ]
            else:
                df["embedding"] = [
                    ", ".join(_format_embedding_cell(emb, i, max_embedding_display) for emb in embs_cpu)
                    for i in range(n)
                ]

        if not style or "embedding" not in df.columns:
            return df

        return df.style.set_properties(
            subset=["embedding"], **{"color": self._EMBEDDING_COLOR}
        ).set_table_styles(
            [{"selector": "th:last-child", "props": [("color", self._EMBEDDING_COLOR)]}],
            overwrite=False,
        )

# %% [markdown]
# ## Display Utilities

# %%
def _format_embedding_cell(emb_tensor, idx: int, max_vals: int = 6):
    """Format one row's embedding for display (compact, PyTorch-style)."""
    if emb_tensor.numel() == 0:
        return "[]"
    if len(emb_tensor.shape) == 0:
        return f"{emb_tensor.item():.4f}"
    if len(emb_tensor.shape) == 1:
        return f"{emb_tensor[idx].item():.4f}"
    if len(emb_tensor.shape) == 2:
        row = emb_tensor[idx]
        n = row.numel()
        if n <= max_vals:
            vals = [f"{x:.4g}" for x in row.tolist()]
            return "[" + ", ".join(vals) + "]"
        vals = [f"{x:.4g}" for x in row[:max_vals].tolist()]
        return "[" + ", ".join(vals) + ", ...]"
    return str(emb_tensor[idx].shape)

# %%
def pretty_print_er(er: EmbeddedRelation, max_rows: int = 12, max_embedding_display: int = 6, show_header: bool = False, display: bool = True):
    """Display an EmbeddedRelation. Thin wrapper kept for backward compatibility.

    In Jupyter, simply typing ``er`` now renders the same styled table via
    ``_repr_html_``.  This helper is still useful for ``display=False``
    (returns a Styler/DataFrame) or ``show_header=True``.
    """
    if not display:
        return er.to_df(max_rows=max_rows, max_embedding_display=max_embedding_display, style=True)

    if show_header:
        num_rows = len(er) if er.content is not None else 0
        n_cols = len(er.content.columns) if er.content is not None else 0
        caption = f"EmbeddedRelation: {num_rows} rows \u00d7 {n_cols} columns"
        if er.embeddings:
            shapes = ", ".join(str(e.shape) for e in er.embeddings)
            caption += f"  |  embeddings: {shapes}"
        print(caption)

    try:
        from IPython.display import display as ipy_display, HTML
        ipy_display(HTML(er._repr_html_(max_rows=max_rows, max_embedding_display=max_embedding_display)))
    except ImportError:
        print(repr(er))

# %% [markdown]
# ## Tests

# %%
if __name__ == "__main__":
    import pandas as pd, torch

    df = pd.DataFrame({"pid": [10, 20, 30], "name": ["Alice", "Bob", "Carol"]})
    emb = torch.randn(3, 8)
    er = EmbeddedRelation(content_schema=["pid", "name"], embedding_shapes=[emb.shape], content=df, embeddings=[emb])
    er

# %%
if __name__ == "__main__":
    import pandas as pd, torch

    # --- __repr__ tests ---
    er_empty = EmbeddedRelation(content_schema=[], embedding_shapes=[])
    assert "0 rows" in repr(er_empty)
    print("repr(empty):", repr(er_empty))

    df = pd.DataFrame({"pid": [1, 2, 3]})
    emb = torch.randn(3, 8)
    er = EmbeddedRelation(content_schema=["pid"], embedding_shapes=[emb.shape], content=df, embeddings=[emb])
    r = repr(er)
    assert "3 rows" in r
    assert "pid" in r
    assert "(3, 8)" in r
    print("repr(with data):", r)

    # --- _repr_html_ tests ---
    html = er._repr_html_()
    assert "#007BA7" in html, f"Expected #007BA7 in _repr_html_ output but not found"
    assert "embedding" in html.lower(), "Expected 'embedding' column header in HTML"
    print("_repr_html_ contains #007BA7 color: OK")

    # No-content case
    html_empty = er_empty._repr_html_()
    assert "no content" in html_empty
    print("_repr_html_(no content): OK")

    # Truncation indicator for large ER
    df_big = pd.DataFrame({"x": range(20)})
    emb_big = torch.randn(20, 4)
    er_big = EmbeddedRelation(content_schema=["x"], embedding_shapes=[emb_big.shape], content=df_big, embeddings=[emb_big])
    html_big = er_big._repr_html_(max_rows=5)
    assert "20 rows total" in html_big
    print("_repr_html_ truncation indicator: OK")

    print("__repr__ + _repr_html_ tests passed")

# %%
if __name__ == "__main__":
    import pandas as pd, torch

    # --- to_df tests ---
    # No content
    er_none = EmbeddedRelation(content_schema=[], embedding_shapes=[])
    assert len(er_none.to_df()) == 0
    print("to_df(no content):", er_none.to_df())

    # With content, no embeddings
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    er1 = EmbeddedRelation(content_schema=["a", "b"], embedding_shapes=[], content=df1)
    result1 = er1.to_df(style=False)
    assert list(result1.columns) == ["a", "b"]
    assert len(result1) == 2
    print("to_df(no embeddings):\n", result1)

    # With content + embeddings
    emb2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    er2 = EmbeddedRelation(content_schema=["a", "b"], embedding_shapes=[emb2.shape], content=df1, embeddings=[emb2])
    result2 = er2.to_df(style=False)
    assert "embedding" in result2.columns
    assert len(result2) == 2
    print("to_df(with embeddings, style=False):\n", result2)

    # Styled version returns Styler with #007BA7 color
    styled = er2.to_df(style=True)
    assert hasattr(styled, "to_html"), "Expected a Styler object"
    styled_html = styled.to_html()
    assert "#007BA7" in styled_html, f"Expected #007BA7 color in styled HTML"
    print("to_df(style=True) contains #007BA7 color: OK")

    # max_rows
    result3 = er2.to_df(max_rows=1, style=False)
    assert len(result3) == 1
    print("to_df(max_rows=1):\n", result3)

    # pretty_print_er backward compat (display=False returns Styler)
    pp_result = pretty_print_er(er2, display=False)
    assert hasattr(pp_result, "to_html"), "pretty_print_er(display=False) should return a Styler"
    print("pretty_print_er(display=False) returns Styler: OK")

    print("\nAll to_df + pretty_print_er tests passed")
