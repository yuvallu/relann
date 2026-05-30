"""Content column tensorization for RelNN encode brackets [...].

Encoders/decoders are standard ``nn.Module``s; this module only implements
auto-tensorization for bare ``[col]`` (numeric / bool / categorical).

Historical note: this feature was previously called "shuttling" in early design docs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "EncodeTypeError",
    "tensorize_column",
    "is_text_dtype",
    "TEXT_OBJECT_DTYPES",
    "build_column_vocabs",
]


class EncodeTypeError(TypeError):
    """Raised when a column cannot be auto-tensorized or encode rules are violated."""


# Pandas / numpy dtype strings treated as text/object (pass through as pd.Series to encoders)
TEXT_OBJECT_DTYPES = frozenset({"object", "string", "str"})


def is_text_dtype(series: pd.Series) -> bool:
    """True if this column should not be auto-tensorized (raw strings for user encoders).

    Recognises plain ``object``/``string``/``str`` dtypes plus pandas extension string
    dtypes (e.g. ``pd.StringDtype("pyarrow")`` whose repr is ``"string[pyarrow]"``) and
    arrow large-string dtypes. Categorical columns are NOT text — they go through the
    categorical branch in ``tensorize_column``.
    """
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return False
    dtype_str = str(dtype)
    if dtype_str in TEXT_OBJECT_DTYPES:
        return True
    if dtype_str.startswith("string") or dtype_str.startswith("large_string"):
        return True
    if isinstance(dtype, pd.StringDtype):
        return True
    return pd.api.types.is_string_dtype(series) and not pd.api.types.is_numeric_dtype(series)


def build_column_vocabs(content: pd.DataFrame) -> Dict[str, Dict[Any, int]]:
    """
    Build stable label→code maps for every categorical column in ``content``.

    Used by data sources and ``_to_er_dict`` so categorical codes stay consistent
    across loads and with ``EmbeddedRelation.column_vocabs`` (not on ``_ColumnExtractModule``).
    """
    out: Dict[str, Dict[Any, int]] = {}
    if content is None or not hasattr(content, "columns"):
        return out
    for col in content.columns:
        s = content[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            out[str(col)] = {c: i for i, c in enumerate(s.cat.categories)}
    return out


def tensorize_column(
    series: pd.Series,
    vocab: Optional[Dict[Any, int]] = None,
) -> Tuple[torch.Tensor, Dict[Any, int]]:
    """
    Convert a pandas Series to a torch tensor. Does not handle text/object dtypes
    (caller should pass those as pd.Series to a text encoder).

    Returns:
        (tensor, vocab) where vocab maps category labels to int64 indices for categorical
        columns (built incrementally). Numeric/bool ignore vocab for output but return
        the dict for caching.
    """
    if is_text_dtype(series):
        raise EncodeTypeError(
            f"tensorize_column cannot tensorize text/object column {series.name!r}. "
            "Use an encoder inside brackets, e.g. [MyEncoder(col)]."
        )

    name = series.name
    dtype_str = str(series.dtype)

    # Unsupported dtypes
    if dtype_str.startswith("datetime") or dtype_str.startswith("timedelta") or "complex" in dtype_str:
        raise EncodeTypeError(
            f"Cannot tensorize column {name!r} with dtype {dtype_str!r}. "
            "Supported types: int, float, bool, category. "
            "For text, use an encoder inside brackets. "
            "Otherwise convert the column before passing to RelNN."
        )

    if pd.api.types.is_bool_dtype(series):
        s = series.astype("float32")
        s = s.fillna(0.0)
        t = torch.from_numpy(s.to_numpy()).view(-1, 1)
        return t, vocab or {}

    if isinstance(series.dtype, pd.CategoricalDtype):
        if vocab is None or len(vocab) == 0:
            vocab = {c: i for i, c in enumerate(series.cat.categories)}
            codes = series.cat.codes.fillna(-1).astype(np.int64)
            t = torch.from_numpy(codes.to_numpy())
            return t, vocab

        # Re-encode against the caller-supplied vocab so codes are stable across
        # slices / batches whose own ``cat.categories`` order may differ. Refuse
        # if the series carries categories the vocab doesn't know about.
        unknown = [c for c in series.cat.categories if c not in vocab]
        if unknown:
            raise EncodeTypeError(
                f"Categorical column {name!r} has categories not in the supplied vocab: "
                f"{unknown[:5]}{'...' if len(unknown) > 5 else ''}. "
                "Source ``column_vocabs`` must cover every category that appears in any slice."
            )
        # ``pd.Categorical(..., categories=ordered)`` re-codes against the canonical
        # ordering and assigns -1 to any value missing from ``categories`` (filtered
        # above). ``recoded.codes`` is a numpy int array, so no fillna detour needed.
        recoded = pd.Categorical(series, categories=list(vocab.keys()))
        codes_np = np.asarray(recoded.codes, dtype=np.int64)
        t = torch.from_numpy(codes_np)
        return t, vocab

    # Numeric
    if pd.api.types.is_numeric_dtype(series):
        s = series.astype("float32")
        s = s.fillna(0.0)
        t = torch.from_numpy(s.to_numpy()).view(-1, 1)
        return t, vocab or {}

    # object with non-text: attempt numeric, else error
    try:
        s = pd.to_numeric(series, errors="raise")
    except (ValueError, TypeError):
        raise EncodeTypeError(
            f"Cannot tensorize column {name!r} with dtype {dtype_str!r}. "
            "Cast to a numeric or categorical dtype, or use a text encoder inside brackets."
        ) from None
    s = s.astype("float32").fillna(0.0)
    t = torch.from_numpy(s.to_numpy()).view(-1, 1)
    return t, vocab or {}
