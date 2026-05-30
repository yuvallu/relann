"""
Tests for Aggregation refactor.

- DerivedER exposes group-by names via group_by_column_names.
- Aggregation accepts group_by_refs and resolves them at instantiate().
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from relann.column_ref import ColumnRef
from relann.pydantic_classes import DerivedER, Var, EmbeddingExpression
from relann.era_operations import (
    Aggregation,
    EmbeddedRelation as RuntimeER,
    get_aggregation_function,
)

# --- DerivedER.group_by_column_names ---

def test_derived_er_group_by_column_names_single():
    """group_by_column_names: one derived attr maps to one name."""
    lhs = DerivedER(
        name="Out",
        derived_content_attrs=[Var(name="K")],
        embedding_expression=EmbeddingExpression(aggregation_fn="mean"),
    )
    names = lhs.group_by_column_names
    assert names == ["K"]

def test_derived_er_group_by_column_names_multiple():
    """group_by_column_names: preserves attr order."""
    lhs = DerivedER(
        name="Out",
        derived_content_attrs=[Var(name="K1"), Var(name="K2")],
        embedding_expression=EmbeddingExpression(aggregation_fn="sum"),
    )
    names = lhs.group_by_column_names
    assert names == ["K1", "K2"]

def test_derived_er_group_by_column_names_includes_non_var():
    """group_by_column_names stringifies non-Var attrs."""
    lhs = DerivedER(
        name="Out",
        derived_content_attrs=[Var(name="K"), "other", Var(name="A")],
        embedding_expression=EmbeddingExpression(aggregation_fn="mean"),
    )
    names = lhs.group_by_column_names
    assert names == ["K", "other", "A"]

def test_derived_er_group_by_column_names_keeps_missing_attr_name():
    """group_by_column_names does not depend on rhs/input attrs."""
    lhs = DerivedER(
        name="Out",
        derived_content_attrs=[Var(name="K"), Var(name="Missing")],
        embedding_expression=EmbeddingExpression(aggregation_fn="mean"),
    )
    names = lhs.group_by_column_names
    assert names == ["K", "Missing"]

def test_derived_er_group_by_column_names_empty():
    """group_by_column_names: empty derived_content_attrs -> empty list."""
    lhs = DerivedER(
        name="Out",
        derived_content_attrs=[],
        embedding_expression=EmbeddingExpression(aggregation_fn="mean"),
    )
    names = lhs.group_by_column_names
    assert names == []

# --- Aggregation constructor ---

def test_aggregation_accepts_group_by_refs_only():
    """Aggregation.__init__ accepts group_by_refs, aggregation_name, output_schema."""
    agg = Aggregation(
        group_by_refs=[ColumnRef(0, 0)],
        aggregation_name="mean",
        output_schema=["node_id"],
    )
    assert len(agg.group_by_refs) == 1
    assert agg.group_by_refs[0].column_idx == 0
    assert agg.aggregation_name == "mean"
    assert agg.output_schema == ["node_id"]

def test_aggregation_empty_group_by_refs_global():
    """Aggregation with group_by_refs=[] or None is global aggregation."""
    agg = Aggregation(group_by_refs=[], aggregation_name="mean", output_schema=None)
    assert agg.group_by_refs == []

# --- Aggregation.instantiate: group by one column ---

def test_aggregation_instantiate_one_key():
    """Aggregation with one group_by_ref produces one row per group."""
    df = pd.DataFrame({"K": [1, 1, 2], "V": [10, 20, 30]})
    son = RuntimeER(
        content_schema=["K", "V"],
        embedding_shapes=[],
        content=df,
        embeddings=None,
    )
    agg = Aggregation(
        group_by_refs=[ColumnRef(0, 0)],
        aggregation_name="mean",
        output_schema=["K"],
    )
    out = agg.instantiate([son])
    assert out.content is not None
    assert len(out.content) == 2
    assert list(out.content.columns) == ["K"]
    assert set(out.content["K"].tolist()) == {1, 2}

def test_aggregation_instantiate_two_keys():
    """Aggregation with two group_by_refs uses composite key."""
    df = pd.DataFrame({
        "K1": [1, 1, 2],
        "K2": [10, 20, 10],
        "V": [100, 200, 300],
    })
    son = RuntimeER(
        content_schema=["K1", "K2", "V"],
        embedding_shapes=[],
        content=df,
        embeddings=None,
    )
    agg = Aggregation(
        group_by_refs=[ColumnRef(0, 0), ColumnRef(0, 1)],
        aggregation_name="mean",
        output_schema=["K1", "K2"],
    )
    out = agg.instantiate([son])
    assert out.content is not None
    assert len(out.content) == 3
    assert list(out.content.columns) == ["K1", "K2"]

def test_aggregation_instantiate_global():
    """Aggregation with no group_by_refs is global (single group)."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [10, 20, 30]})
    son = RuntimeER(
        content_schema=["A", "B"],
        embedding_shapes=[torch.Size([3, 4])],
        content=df,
        embeddings=None,
    )
    agg = Aggregation(
        group_by_refs=[],
        aggregation_name="mean",
        output_schema=None,
    )
    out = agg.instantiate([son])
    assert out.content is not None
    assert len(out.content) == 1
    assert out.embedding_shapes == [torch.Size([1, 4])]

def test_aggregation_instantiate_output_schema_rename():
    """Aggregation renames grouping columns to output_schema when provided."""
    df = pd.DataFrame({"target_id": [1, 1, 2], "V": [10, 20, 30]})
    son = RuntimeER(
        content_schema=["target_id", "V"],
        embedding_shapes=[],
        content=df,
        embeddings=None,
    )
    agg = Aggregation(
        group_by_refs=[ColumnRef(0, 0)],
        aggregation_name="mean",
        output_schema=["node_id"],
    )
    out = agg.instantiate([son])
    assert list(out.content.columns) == ["node_id"]

def test_aggregation_physical_names_differ_from_logical():
    """Aggregation with group_by_refs: logical schema (a,b), physical columns (x,y); groups by col 0."""
    df = pd.DataFrame({"x": [1, 1, 2], "y": [10, 20, 30]})
    son = RuntimeER(
        content_schema=["a", "b"],
        embedding_shapes=[],
        content=df,
        embeddings=None,
    )
    agg = Aggregation(
        group_by_refs=[ColumnRef(0, 0)],
        aggregation_name="mean",
        output_schema=["a"],
    )
    out = agg.instantiate([son])
    assert out.content is not None
    assert list(out.content.columns) == ["a"]
    assert len(out.content) == 2

# --- Aggregation.forward ---

def test_aggregation_forward_mean():
    """Aggregation.forward aggregates embeddings by group index."""
    df = pd.DataFrame({"K": [1, 1, 2], "V": [10, 20, 30]})
    embs = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=torch.float32)
    son = RuntimeER(
        content_schema=["K", "V"],
        embedding_shapes=[torch.Size([3, 2])],
        content=df,
        embeddings=[embs],
    )
    agg = Aggregation(
        group_by_refs=[ColumnRef(0, 0)],
        aggregation_name="mean",
        aggregation_function=get_aggregation_function("mean"),
        output_schema=["K"],
    )
    agg.instantiate([son])
    out = agg.forward([son])
    assert out.embeddings is not None
    assert len(out.embeddings) == 1
    assert out.embeddings[0].shape == (2, 2)
    # Group 1: mean of [1,1] and [2,2] -> [1.5, 1.5]; group 2: [3,3]
    mean_k1 = out.embeddings[0][0].tolist()
    mean_k2 = out.embeddings[0][1].tolist()
    assert mean_k1 == [1.5, 1.5]
    assert mean_k2 == [3.0, 3.0]

def test_aggregation_forward_sum():
    """Aggregation.forward with sum aggregates correctly."""
    df = pd.DataFrame({"K": [1, 1, 2]})
    embs = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    son = RuntimeER(
        content_schema=["K"],
        embedding_shapes=[torch.Size([3, 1])],
        content=df,
        embeddings=[embs],
    )
    agg = Aggregation(
        group_by_refs=[ColumnRef(0, 0)],
        aggregation_name="sum",
        aggregation_function=get_aggregation_function("sum"),
        output_schema=["K"],
    )
    agg.instantiate([son])
    out = agg.forward([son])
    assert out.embeddings[0].shape == (2, 1)
    assert out.embeddings[0][0].item() == 3.0
    assert out.embeddings[0][1].item() == 3.0

def test_aggregation_forward_global():
    """Aggregation.forward with global agg produces single output row."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    embs = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    son = RuntimeER(
        content_schema=["A"],
        embedding_shapes=[torch.Size([3, 1])],
        content=df,
        embeddings=[embs],
    )
    agg = Aggregation(
        group_by_refs=[],
        aggregation_name="mean",
        aggregation_function=get_aggregation_function("mean"),
        output_schema=None,
    )
    agg.instantiate([son])
    out = agg.forward([son])
    assert out.embeddings[0].shape == (1, 1)
    assert out.embeddings[0].item() == 2.0

# --- Edge cases ---

def test_aggregation_empty_dataframe():
    """Aggregation with empty DataFrame returns empty content, correct schema."""
    df = pd.DataFrame({"K": [], "V": []})
    son = RuntimeER(
        content_schema=["K", "V"],
        embedding_shapes=[],
        content=df,
        embeddings=None,
    )
    agg = Aggregation(
        group_by_refs=[ColumnRef(0, 0)],
        aggregation_name="mean",
        output_schema=["K"],
    )
    out = agg.instantiate([son])
    assert out.content is not None
    assert len(out.content) == 0
    assert list(out.content.columns) == ["K"]

def test_aggregation_instantiate_requires_one_son():
    """Aggregation.instantiate with != 1 son raises."""
    agg = Aggregation(group_by_refs=[], aggregation_name="mean")
    son = RuntimeER(content_schema=["A"], embedding_shapes=[], content=pd.DataFrame({"A": [1]}), embeddings=None)
    try:
        agg.instantiate([])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "exactly 1 input" in str(e)
    try:
        agg.instantiate([son, son])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "exactly 1 input" in str(e)

def test_aggregation_forward_without_instantiate_raises():
    """Aggregation.forward without prior instantiate raises."""
    son = RuntimeER(
        content_schema=["K"],
        embedding_shapes=[torch.Size([1, 2])],
        content=pd.DataFrame({"K": [1]}),
        embeddings=[torch.randn(1, 2)],
    )
    agg = Aggregation(group_by_refs=[ColumnRef(0, 0)], aggregation_name="mean", output_schema=["K"])
    try:
        agg.forward([son])
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "instantiate" in str(e).lower() or "cached" in str(e).lower()

if __name__ == "__main__":
    test_derived_er_group_by_column_names_single()
    test_derived_er_group_by_column_names_multiple()
    test_derived_er_group_by_column_names_includes_non_var()
    test_derived_er_group_by_column_names_keeps_missing_attr_name()
    test_derived_er_group_by_column_names_empty()
    test_aggregation_accepts_group_by_refs_only()
    test_aggregation_empty_group_by_refs_global()
    test_aggregation_instantiate_one_key()
    test_aggregation_instantiate_two_keys()
    test_aggregation_instantiate_global()
    test_aggregation_instantiate_output_schema_rename()
    test_aggregation_physical_names_differ_from_logical()
    test_aggregation_forward_mean()
    test_aggregation_forward_sum()
    test_aggregation_forward_global()
    test_aggregation_empty_dataframe()
    test_aggregation_instantiate_requires_one_son()
    test_aggregation_forward_without_instantiate_raises()
    print("All aggregation refactor tests passed.")
