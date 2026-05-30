"""Unit tests for ``parent.encode.tensorize_column``."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from relann.encode import EncodeTypeError, tensorize_column, is_text_dtype

def test_tensorize_numeric_int():
    s = pd.Series([1, 2, 3], dtype="int64", name="x")
    t, _ = tensorize_column(s)
    assert t.shape == (3, 1)
    assert t.dtype == torch.float32

def test_tensorize_numeric_float_nan():
    s = pd.Series([1.0, np.nan, 3.0], name="x")
    t, _ = tensorize_column(s)
    assert t[1, 0].item() == 0.0

def test_tensorize_boolean():
    s = pd.Series([True, False], name="b")
    t, _ = tensorize_column(s)
    assert t.shape == (2, 1)
    assert t[0, 0].item() == 1.0

def test_tensorize_categorical():
    s = pd.Series(pd.Categorical(["a", "b", "a"]))
    t, vocab = tensorize_column(s)
    assert t.shape == (3,)
    assert t.dtype == torch.int64

def test_tensorize_text_raises_in_tensorize():
    s = pd.Series(["hello", "world"], dtype="object")
    assert is_text_dtype(s)
    with pytest.raises(EncodeTypeError):
        tensorize_column(s)

def test_tensorize_timedelta_raises():
    s = pd.Series(pd.to_timedelta([1, 2], unit="s"))
    with pytest.raises(EncodeTypeError):
        tensorize_column(s)

def test_tensorize_empty():
    s = pd.Series([], dtype="float64")
    t, _ = tensorize_column(s)
    assert t.shape == (0, 1)

def test_is_text_dtype_recognises_pandas_string_extension():
    """``pd.StringDtype()`` (Python backend) should be detected as text."""
    s = pd.Series(["a", "b"], dtype=pd.StringDtype())
    assert is_text_dtype(s)

def test_is_text_dtype_recognises_pyarrow_string():
    """``pd.StringDtype("pyarrow")`` should be detected as text (regression for CR fix #4).

    Skipped if pyarrow is not installed.
    """
    pytest.importorskip("pyarrow")
    s = pd.Series(["a", "b"], dtype=pd.StringDtype("pyarrow"))
    assert is_text_dtype(s)

def test_tensorize_categorical_honours_supplied_vocab():
    """Caller-supplied vocab determines codes, not ``series.cat.categories`` order
    (regression for CR fix #1: cross-batch / cross-source categorical drift)."""
    # Source vocab in canonical order: a=0, b=1, c=2
    vocab = {"a": 0, "b": 1, "c": 2}
    # Slice has only {a, c} and pandas may reorder cat.categories
    s = pd.Series(pd.Categorical(["c", "a", "c"], categories=["c", "a"]))
    t, returned_vocab = tensorize_column(s, vocab=vocab)
    # If we used series.cat.codes, we'd get [0, 1, 0] (c=0, a=1 in slice order).
    # With the canonical vocab honoured, we must get [2, 0, 2].
    assert t.tolist() == [2, 0, 2]
    assert returned_vocab is vocab

def test_tensorize_categorical_raises_on_unknown_category():
    """If the series has a category not in the supplied vocab, raise EncodeTypeError."""
    vocab = {"a": 0, "b": 1}
    s = pd.Series(pd.Categorical(["a", "c"]))  # 'c' not in vocab
    with pytest.raises(EncodeTypeError, match="not in the supplied vocab"):
        tensorize_column(s, vocab=vocab)
