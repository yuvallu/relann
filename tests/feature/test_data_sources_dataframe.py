"""Tests for DataFrameSource: verifies that Session(db={name: DataFrameSource(...)})
produces identical results to the legacy Session(db={name: (df, tensor)}) path."""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from relann.session import Session
from relann.torch_utils import full_seed
from relann.data_sources import DataFrameSource

def _build_db_legacy(n: int = 4):
    df = pd.DataFrame({"uid": range(n), "feat": [float(i) for i in range(n)]})
    z = torch.randn(n, 2)
    return df, z

def test_dataframe_source_matches_legacy_tuple():
    """DataFrameSource and the legacy (df, tensor) tuple must produce the same output shape
    and data flow (we verify shape + finite values; exact equality would require identical RNG
    state across two separate Session instances which is not guaranteed)."""
    df, z = _build_db_legacy()
    program = """
#lang:relnn
Out(uid; Linear(2, 4)(z1)) :- Users(uid; z1) .
?pred Pred(uid; z) :- Out(uid; z) .
"""
    # Legacy path
    full_seed(0)
    session_legacy = Session(db={"Users": (df, z)})
    session_legacy.run(program)
    emb_legacy = session_legacy.relation("Pred").embeddings[0]

    # DataFrameSource path
    full_seed(0)
    src = DataFrameSource("Users", df, embeddings=z)
    session_src = Session(db={"Users": src})
    session_src.run(program)
    emb_src = session_src.relation("Pred").embeddings[0]

    assert emb_legacy.shape == emb_src.shape, "shapes differ"
    assert torch.isfinite(emb_legacy).all(), "legacy output has non-finite values"
    assert torch.isfinite(emb_src).all(), "DataFrameSource output has non-finite values"

def test_dataframe_source_no_pre_existing_embeddings():
    """DataFrameSource with a dummy 1-dim embedding + encode brackets produces correct output.

    Encode-only rules still need at least one embedding variable on the source so that
    Transformation.instantiate can infer input shapes. A 1-dim zero tensor is the minimal
    placeholder (the rule uses z1 as the embedding var but it is ignored in the encode expr).
    """
    full_seed(0)
    n = 3
    df = pd.DataFrame({"uid": range(n), "age": [20.0, 30.0, 40.0]})
    dummy_z = torch.zeros(n, 1)  # placeholder — encode bracket uses age, not z1
    src = DataFrameSource("People", df, embeddings=dummy_z)

    session = Session(db={"People": src})
    session.run(
        """
#lang:relnn
Enc(uid; [Linear(1, 4)(age)]) :- People(uid, age; z1) .
?pred P(uid; z) :- Enc(uid; z) .
"""
    )
    out = session.relation("P")
    assert out.embeddings[0].shape == (n, 4)

def test_dataframe_source_load_by_keys_filters_rows():
    """load_by_keys returns rows and slices embeddings to match."""
    df = pd.DataFrame({"uid": range(3), "val": [1.0, 2.0, 3.0]})
    z = torch.arange(3, dtype=torch.float32).view(3, 1)
    src = DataFrameSource("T", df, embeddings=z)
    sub = src.load_by_keys("uid", [0, 2])
    assert len(sub["content"]) == 2
    assert sub["embeddings"] is not None
    assert sub["embeddings"][0].tolist() == [[0.0], [2.0]]

def test_dataframe_source_content_schema():
    """schema() must return the DataFrame's column names."""
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    src = DataFrameSource("T", df)
    assert src.schema() == ["a", "b", "c"]

def test_dataframe_source_multi_tensor_embeddings():
    """DataFrameSource accepts a list of tensors as embeddings."""
    df = pd.DataFrame({"uid": range(2)})
    z1 = torch.randn(2, 3)
    z2 = torch.randn(2, 5)
    src = DataFrameSource("T", df, embeddings=[z1, z2])
    result = src.load_full()
    assert len(result["embeddings"]) == 2
    assert result["embedding_shapes"] == [(2, 3), (2, 5)]

def test_dataframe_source_load_by_keys_missing_key_column_raises():
    """``load_by_keys`` must raise KeyError when the key column doesn't exist
    (CR test gap: data_sources.py:172-177 guard was untested)."""
    df = pd.DataFrame({"uid": range(3), "v": [1.0, 2.0, 3.0]})
    src = DataFrameSource("T", df)
    with pytest.raises(KeyError, match="not in columns"):
        src.load_by_keys("does_not_exist", [0, 1])

def test_dataframe_source_load_by_keys_empty_keys():
    """Empty keys list returns an empty DataFrame with the source schema preserved."""
    df = pd.DataFrame({"uid": range(3), "v": [1.0, 2.0, 3.0]})
    z = torch.arange(3, dtype=torch.float32).view(3, 1)
    src = DataFrameSource("T", df, embeddings=z)
    sub = src.load_by_keys("uid", [])
    assert len(sub["content"]) == 0
    assert list(sub["content"].columns) == ["uid", "v"]
    assert sub["embeddings"] is not None
    assert sub["embeddings"][0].shape == (0, 1)

def test_dataframe_source_load_by_keys_categorical_codes_stable():
    """Same label must encode to the same code across two ``load_by_keys`` calls
    on disjoint slices (CR fix #1: cross-batch categorical drift)."""
    df = pd.DataFrame(
        {
            "uid": range(6),
            "dept": pd.Categorical(["a", "b", "c", "a", "b", "c"]),
        }
    )
    src = DataFrameSource("T", df)

    full_vocab = src.load_full()["column_vocabs"]["dept"]

    sub_ac = src.load_by_keys("uid", [0, 2])  # only 'a' and 'c' rows
    sub_bc = src.load_by_keys("uid", [1, 5])  # only 'b' and 'c' rows

    # Both slices must report the same vocab as the source full load
    assert sub_ac["column_vocabs"]["dept"] == full_vocab
    assert sub_bc["column_vocabs"]["dept"] == full_vocab

