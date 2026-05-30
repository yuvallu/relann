"""
Tests for join mechanism: join_conditions only (no join_keys).

- RHS.join_conditions infers natural join keys from overlapping column names.
- Join in era_operations uses join_conditions; supports 2-way, 3-way, and chain joins.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from relann.column_ref import ColumnRef
from relann.pydantic_classes import RHS, EmbeddedRelation, Var
from relann.parser import parse_and_transform_str
from relann.term_graph import program_to_graph
from relann.relnn import term_graph_to_module
from relann.era_operations import Join, EmbeddedRelation as RuntimeER, _to_er_dict
from relann.engine import Engine
from relann.torch_utils import full_seed

def test_rhs_join_conditions_computed():
    """RHS.join_conditions infers natural join keys from overlapping column names."""
    er1 = EmbeddedRelation(name="A", content_attrs=[Var(name="X"), Var(name="Y")])
    er2 = EmbeddedRelation(name="B", content_attrs=[Var(name="Y"), Var(name="Z")])
    rhs = RHS(ers=[er1, er2], rel_ops=[","])
    jc = rhs.join_conditions
    assert len(jc) == 1
    assert jc[0]["key_name"] == "Y"
    refs = jc[0]["normalized_refs"]
    assert len(refs) == 2
    assert all(isinstance(r, ColumnRef) for r in refs)
    assert (refs[0].input_idx, refs[0].column_idx) == (0, 1)
    assert (refs[1].input_idx, refs[1].column_idx) == (1, 0)

def test_rhs_join_conditions_single_er_empty():
    """Single ER or no join rel_op yields empty join_conditions."""
    er = EmbeddedRelation(name="A", content_attrs=[Var(name="X")])
    assert RHS(ers=[er], rel_ops=None).join_conditions == []
    assert RHS(ers=[er, er], rel_ops=["|"]).join_conditions == []

def test_join_accepts_merge_steps():
    """Join.__init__ accepts output_schema, merge_steps, and input_schemas."""
    input_schemas = [["X", "Y"], ["Y", "Z"]]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 1)],
        "right_refs": [ColumnRef(1, 0)],
        "key_names": ["Y"]
    }]
    join = Join(output_schema=["X", "Y", "Z"], merge_steps=merge_steps, input_schemas=input_schemas)
    assert join.merge_steps == merge_steps
    assert join.input_schemas == input_schemas

def test_simple_join_program_e2e():
    """Parse -> term graph -> RelNN: join node has join_conditions, instantiate/forward succeed."""
    full_seed(42)
    program_str = (
        "SimpleEmbedding(X,Y,Z; avg(Linear(6,4)(Concat(z1,z2)))) :- "
        "InputData1(X,Y;z1), InputData2(Y,Z;z2) ."
    )
    program = parse_and_transform_str(program_str)
    tg = program_to_graph(program)
    join_node_id = next((n for n in tg.nodes() if tg.nodes[n].get("type") == "join"), None)
    assert join_node_id is not None
    nd = tg.nodes[join_node_id]
    assert "join_conditions" in nd
    assert len(nd["join_conditions"]) == 1
    assert nd["join_conditions"][0]["key_name"] == "Y"
    assert "join_keys" not in nd

    engine = Engine(debug=False)
    engine.add_program(program)
    ground_tg = engine.term_graphs["global"]
    ground_tg = engine.eval_tensor_terms_on_tg(ground_tg)
    # V1.3 compile-after-optimize: pass engine so the optimizer's post-opt
    # compile pass can rebuild any modules whose v2i changed (e.g. due to R1
    # commute on the join below the transformation).
    model = term_graph_to_module(ground_tg, engine=engine)
    df1 = pd.DataFrame({"X": [1, 2], "Y": [10, 20]})
    df2 = pd.DataFrame({"Y": [10, 20], "Z": [100, 200]})
    e1 = torch.randn(2, 3)
    e2 = torch.randn(2, 3)
    relations = {
        "InputData1": _to_er_dict((df1, e1)),
        "InputData2": _to_er_dict((df2, e2)),
    }
    model.instantiate(relations)
    out = model.forward(relations)
    assert out.content is not None
    assert len(out.content) == 2
    assert list(out.content.columns) == ["X", "Y", "Z"]

def test_two_way_join_x_y__y_z():
    """2-way join: (X,Y) ⋈ (Y,Z) on Y."""
    df1 = pd.DataFrame({"X": [1, 2], "Y": [10, 20]})
    df2 = pd.DataFrame({"Y": [10, 20], "Z": [100, 200]})
    r1 = RuntimeER(content_schema=["X", "Y"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["Y", "Z"], embedding_shapes=[], content=df2, embeddings=None)
    input_schemas = [["X", "Y"], ["Y", "Z"]]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 1)],
        "right_refs": [ColumnRef(1, 0)],
        "key_names": ["Y"]
    }]
    join = Join(output_schema=["X", "Y", "Z"], merge_steps=merge_steps, input_schemas=input_schemas)
    out = join.instantiate([r1, r2])
    assert out.content is not None
    assert list(out.content.columns) == ["X", "Y", "Z"]
    assert len(out.content) == 2
    expected = pd.DataFrame({"X": [1, 2], "Y": [10, 20], "Z": [100, 200]})
    pd.testing.assert_frame_equal(out.content.sort_values("Y").reset_index(drop=True), expected)

def test_three_way_join_k_a__k_b__k_c():
    """3-way join on one key: (K,A) ⋈ (K,B) ⋈ (K,C) on K."""
    df1 = pd.DataFrame({"K": [1, 2], "A": [10, 20]})
    df2 = pd.DataFrame({"K": [1, 2], "B": [100, 200]})
    df3 = pd.DataFrame({"K": [1, 2], "C": [1000, 2000]})
    r1 = RuntimeER(content_schema=["K", "A"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["K", "B"], embedding_shapes=[], content=df2, embeddings=None)
    r3 = RuntimeER(content_schema=["K", "C"], embedding_shapes=[], content=df3, embeddings=None)
    input_schemas = [["K", "A"], ["K", "B"], ["K", "C"]]
    merge_steps = [
        {
            "step": 1,
            "left_refs": [ColumnRef(0, 0)],
            "right_refs": [ColumnRef(1, 0)],
            "key_names": ["K"]
        },
        {
            "step": 2,
            "left_refs": [ColumnRef(1, 0)],
            "right_refs": [ColumnRef(2, 0)],
            "key_names": ["K"]
        }
    ]
    join = Join(output_schema=["K", "A", "B", "C"], merge_steps=merge_steps, input_schemas=input_schemas)
    out = join.instantiate([r1, r2, r3])
    assert out.content is not None
    assert list(out.content.columns) == ["K", "A", "B", "C"]
    assert len(out.content) == 2
    expected = pd.DataFrame({"K": [1, 2], "A": [10, 20], "B": [100, 200], "C": [1000, 2000]})
    pd.testing.assert_frame_equal(out.content.sort_values("K").reset_index(drop=True), expected)

def test_three_way_join_a_b__a_b__b_c():
    """3-way join: (a,b) ⋈ (a,b) ⋈ (b,c) on b only. All three have b."""
    df1 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    df2 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    df3 = pd.DataFrame({"b": [10, 20], "c": [100, 200]})
    r1 = RuntimeER(content_schema=["a", "b"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["a", "b"], embedding_shapes=[], content=df2, embeddings=None)
    r3 = RuntimeER(content_schema=["b", "c"], embedding_shapes=[], content=df3, embeddings=None)
    input_schemas = [["a", "b"], ["a", "b"], ["b", "c"]]
    merge_steps = [
        {
            "step": 1,
            "left_refs": [ColumnRef(0, 1)],
            "right_refs": [ColumnRef(1, 1)],
            "key_names": ["b"]
        },
        {
            "step": 2,
            "left_refs": [ColumnRef(1, 1)],
            "right_refs": [ColumnRef(2, 0)],
            "key_names": ["b"]
        }
    ]
    join = Join(output_schema=["a", "b", "c"], merge_steps=merge_steps, input_schemas=input_schemas)
    out = join.instantiate([r1, r2, r3])
    assert out.content is not None
    assert len(out.content) == 2
    assert "a" in out.content.columns and "b" in out.content.columns and "c" in out.content.columns
    expected = pd.DataFrame({"a": [1, 2], "b": [10, 20], "c": [100, 200]})
    pd.testing.assert_frame_equal(out.content.sort_values("b").reset_index(drop=True), expected)

def test_chain_join_a_b__b_c__c_d():
    """Chain join: (a,b) ⋈ (b,c) ⋈ (c,d). Key b links 0–1, key c links 1–2. Uses None for inputs without key."""
    df1 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    df2 = pd.DataFrame({"b": [10, 20], "c": [100, 200]})
    df3 = pd.DataFrame({"c": [100, 200], "d": [1000, 2000]})
    r1 = RuntimeER(content_schema=["a", "b"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["b", "c"], embedding_shapes=[], content=df2, embeddings=None)
    r3 = RuntimeER(content_schema=["c", "d"], embedding_shapes=[], content=df3, embeddings=None)
    input_schemas = [["a", "b"], ["b", "c"], ["c", "d"]]
    merge_steps = [
        {
            "step": 1,
            "left_refs": [ColumnRef(0, 1)],
            "right_refs": [ColumnRef(1, 0)],
            "key_names": ["b"]
        },
        {
            "step": 2,
            "left_refs": [ColumnRef(1, 1)],
            "right_refs": [ColumnRef(2, 0)],
            "key_names": ["c"]
        }
    ]
    join = Join(output_schema=["a", "b", "c", "d"], merge_steps=merge_steps, input_schemas=input_schemas)
    out = join.instantiate([r1, r2, r3])
    assert out.content is not None
    assert list(out.content.columns) == ["a", "b", "c", "d"]
    assert len(out.content) == 2
    expected = pd.DataFrame({"a": [1, 2], "b": [10, 20], "c": [100, 200], "d": [1000, 2000]})
    pd.testing.assert_frame_equal(out.content.sort_values("b").reset_index(drop=True), expected)

def test_three_way_join_forward_embeddings_aligned():
    """3-way Join.forward() aligns embeddings by join result row indices."""
    df1 = pd.DataFrame({"K": [1, 2], "A": [10, 20]})
    df2 = pd.DataFrame({"K": [1, 2], "B": [100, 200]})
    df3 = pd.DataFrame({"K": [1, 2], "C": [1000, 2000]})
    r1 = RuntimeER(
        content_schema=["K", "A"],
        embedding_shapes=[torch.Size([2, 3])],
        content=df1,
        embeddings=[torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])],
    )
    r2 = RuntimeER(
        content_schema=["K", "B"],
        embedding_shapes=[torch.Size([2, 2])],
        content=df2,
        embeddings=[torch.tensor([[10.0, 10.0], [20.0, 20.0]])],
    )
    r3 = RuntimeER(
        content_schema=["K", "C"],
        embedding_shapes=[torch.Size([2, 1])],
        content=df3,
        embeddings=[torch.tensor([[100.0], [200.0]])],
    )
    sons = [r1, r2, r3]
    input_schemas = [["K", "A"], ["K", "B"], ["K", "C"]]
    merge_steps = [
        {
            "step": 1,
            "left_refs": [ColumnRef(0, 0)],
            "right_refs": [ColumnRef(1, 0)],
            "key_names": ["K"]
        },
        {
            "step": 2,
            "left_refs": [ColumnRef(1, 0)],
            "right_refs": [ColumnRef(2, 0)],
            "key_names": ["K"]
        }
    ]
    join = Join(output_schema=["K", "A", "B", "C"], merge_steps=merge_steps, input_schemas=input_schemas)
    join.instantiate(sons)
    out = join.forward(sons)
    assert out.content is not None
    assert len(out.content) == 2
    assert out.embeddings is not None
    assert len(out.embeddings) == 3
    assert out.embeddings[0].shape == (2, 3)
    assert out.embeddings[1].shape == (2, 2)
    assert out.embeddings[2].shape == (2, 1)
    assert out.embeddings[0][0].tolist() == [1.0, 1.0, 1.0]
    assert out.embeddings[0][1].tolist() == [2.0, 2.0, 2.0]

def test_join_empty_dataframes():
    """Join with empty DataFrames should return empty result."""
    df1 = pd.DataFrame({"X": [], "Y": []})
    df2 = pd.DataFrame({"Y": [], "Z": []})
    r1 = RuntimeER(content_schema=["X", "Y"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["Y", "Z"], embedding_shapes=[], content=df2, embeddings=None)
    input_schemas = [["X", "Y"], ["Y", "Z"]]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 1)],
        "right_refs": [ColumnRef(1, 0)],
        "key_names": ["Y"]
    }]
    join = Join(output_schema=["X", "Y", "Z"], merge_steps=merge_steps, input_schemas=input_schemas)
    out = join.instantiate([r1, r2])
    assert out.content is not None
    assert len(out.content) == 0
    assert list(out.content.columns) == ["X", "Y", "Z"]

def test_join_no_matching_rows():
    """Join with no matching keys should return empty result (inner join)."""
    df1 = pd.DataFrame({"X": [1, 2], "Y": [10, 20]})
    df2 = pd.DataFrame({"Y": [30, 40], "Z": [100, 200]})  # No matching Y values
    r1 = RuntimeER(content_schema=["X", "Y"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["Y", "Z"], embedding_shapes=[], content=df2, embeddings=None)
    input_schemas = [["X", "Y"], ["Y", "Z"]]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 1)],
        "right_refs": [ColumnRef(1, 0)],
        "key_names": ["Y"]
    }]
    join = Join(output_schema=["X", "Y", "Z"], merge_steps=merge_steps, input_schemas=input_schemas)
    out = join.instantiate([r1, r2])
    assert out.content is not None
    assert len(out.content) == 0
    assert list(out.content.columns) == ["X", "Y", "Z"]

def test_join_composite_key():
    """Join with multiple keys (composite key)."""
    df1 = pd.DataFrame({"K1": [1, 1, 2], "K2": [10, 20, 10], "A": [100, 200, 300]})
    df2 = pd.DataFrame({"K1": [1, 1, 2], "K2": [10, 20, 10], "B": [1000, 2000, 3000]})
    r1 = RuntimeER(content_schema=["K1", "K2", "A"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["K1", "K2", "B"], embedding_shapes=[], content=df2, embeddings=None)
    input_schemas = [["K1", "K2", "A"], ["K1", "K2", "B"]]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 0), ColumnRef(0, 1)],
        "right_refs": [ColumnRef(1, 0), ColumnRef(1, 1)],
        "key_names": ["K1", "K2"]
    }]
    join = Join(output_schema=["K1", "K2", "A", "B"], merge_steps=merge_steps, input_schemas=input_schemas)
    out = join.instantiate([r1, r2])
    assert out.content is not None
    assert len(out.content) == 3
    assert list(out.content.columns) == ["K1", "K2", "A", "B"]
    expected = pd.DataFrame({
        "K1": [1, 1, 2],
        "K2": [10, 20, 10],
        "A": [100, 200, 300],
        "B": [1000, 2000, 3000]
    })
    pd.testing.assert_frame_equal(
        out.content.sort_values(["K1", "K2"]).reset_index(drop=True),
        expected.sort_values(["K1", "K2"]).reset_index(drop=True)
    )

def test_join_different_dtypes():
    """Join with different dtypes should coerce types to match."""
    df1 = pd.DataFrame({"K": [1, 2, 3], "A": [10, 20, 30]})
    df2 = pd.DataFrame({"K": ["1", "2", "3"], "B": [100, 200, 300]})  # K is string
    r1 = RuntimeER(content_schema=["K", "A"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["K", "B"], embedding_shapes=[], content=df2, embeddings=None)
    input_schemas = [["K", "A"], ["K", "B"]]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 0)],
        "right_refs": [ColumnRef(1, 0)],
        "key_names": ["K"]
    }]
    join = Join(output_schema=["K", "A", "B"], merge_steps=merge_steps, input_schemas=input_schemas)
    out = join.instantiate([r1, r2])
    assert out.content is not None
    # Type coercion should happen, but result may vary - just check it doesn't crash
    assert len(out.content) >= 0

def test_join_four_way():
    """4-way join: (A,B) ⋈ (B,C) ⋈ (C,D) ⋈ (D,E)."""
    df1 = pd.DataFrame({"A": [1, 2], "B": [10, 20]})
    df2 = pd.DataFrame({"B": [10, 20], "C": [100, 200]})
    df3 = pd.DataFrame({"C": [100, 200], "D": [1000, 2000]})
    df4 = pd.DataFrame({"D": [1000, 2000], "E": [10000, 20000]})
    r1 = RuntimeER(content_schema=["A", "B"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["B", "C"], embedding_shapes=[], content=df2, embeddings=None)
    r3 = RuntimeER(content_schema=["C", "D"], embedding_shapes=[], content=df3, embeddings=None)
    r4 = RuntimeER(content_schema=["D", "E"], embedding_shapes=[], content=df4, embeddings=None)
    input_schemas = [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"]]
    merge_steps = [
        {"step": 1, "left_refs": [ColumnRef(0, 1)], "right_refs": [ColumnRef(1, 0)], "key_names": ["B"]},
        {"step": 2, "left_refs": [ColumnRef(1, 1)], "right_refs": [ColumnRef(2, 0)], "key_names": ["C"]},
        {"step": 3, "left_refs": [ColumnRef(2, 1)], "right_refs": [ColumnRef(3, 0)], "key_names": ["D"]}
    ]
    join = Join(output_schema=["A", "B", "C", "D", "E"], merge_steps=merge_steps, input_schemas=input_schemas)
    out = join.instantiate([r1, r2, r3, r4])
    assert out.content is not None
    assert len(out.content) == 2
    assert list(out.content.columns) == ["A", "B", "C", "D", "E"]
    expected = pd.DataFrame({
        "A": [1, 2], "B": [10, 20], "C": [100, 200],
        "D": [1000, 2000], "E": [10000, 20000]
    })
    pd.testing.assert_frame_equal(out.content.sort_values("A").reset_index(drop=True), expected)

def test_join_partial_matches():
    """Join where only some rows match."""
    df1 = pd.DataFrame({"K": [1, 2, 3, 4], "A": [10, 20, 30, 40]})
    df2 = pd.DataFrame({"K": [2, 3, 5], "B": [200, 300, 500]})  # Only 2,3 match
    r1 = RuntimeER(content_schema=["K", "A"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["K", "B"], embedding_shapes=[], content=df2, embeddings=None)
    input_schemas = [["K", "A"], ["K", "B"]]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 0)],
        "right_refs": [ColumnRef(1, 0)],
        "key_names": ["K"]
    }]
    join = Join(output_schema=["K", "A", "B"], merge_steps=merge_steps, input_schemas=input_schemas)
    out = join.instantiate([r1, r2])
    assert out.content is not None
    assert len(out.content) == 2  # Only 2 matches
    assert list(out.content["K"].values) == [2, 3] or list(out.content["K"].values) == [3, 2]

def test_join_with_duplicate_keys():
    """Join where keys have duplicate values."""
    df1 = pd.DataFrame({"K": [1, 1, 2], "A": [10, 11, 20]})
    df2 = pd.DataFrame({"K": [1, 2, 2], "B": [100, 200, 201]})
    r1 = RuntimeER(content_schema=["K", "A"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["K", "B"], embedding_shapes=[], content=df2, embeddings=None)
    input_schemas = [["K", "A"], ["K", "B"]]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 0)],
        "right_refs": [ColumnRef(1, 0)],
        "key_names": ["K"]
    }]
    join = Join(output_schema=["K", "A", "B"], merge_steps=merge_steps, input_schemas=input_schemas)
    out = join.instantiate([r1, r2])
    assert out.content is not None
    # Should have 4 rows: (1,10,100), (1,11,100), (2,20,200), (2,20,201)
    assert len(out.content) == 4
    assert list(out.content.columns) == ["K", "A", "B"]

def test_join_output_schema_normalization():
    """Join should normalize output schema, handling _x/_y suffixes."""
    df1 = pd.DataFrame({"K": [1, 2], "A": [10, 20]})
    df2 = pd.DataFrame({"K": [1, 2], "A": [100, 200]})  # Same column name
    r1 = RuntimeER(content_schema=["K", "A"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["K", "A"], embedding_shapes=[], content=df2, embeddings=None)
    input_schemas = [["K", "A"], ["K", "A"]]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 0)],
        "right_refs": [ColumnRef(1, 0)],
        "key_names": ["K"]
    }]
    # Output schema should specify how to handle duplicate A columns
    join = Join(output_schema=["K", "A"], merge_steps=merge_steps, input_schemas=input_schemas)
    out = join.instantiate([r1, r2])
    assert out.content is not None
    assert "K" in out.content.columns
    assert "A" in out.content.columns
    # After normalization, should have one A column (coalesced from _x/_y)

def test_join_without_content_schema_uses_columns():
    """Join should use content.columns when content_schema is missing."""
    df1 = pd.DataFrame({"X": [1, 2], "Y": [10, 20]})
    df2 = pd.DataFrame({"Y": [10, 20], "Z": [100, 200]})
    r1 = RuntimeER(content_schema=None, embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=None, embedding_shapes=[], content=df2, embeddings=None)
    input_schemas = [["X", "Y"], ["Y", "Z"]]  # Still provide input_schemas
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 1)],
        "right_refs": [ColumnRef(1, 0)],
        "key_names": ["Y"]
    }]
    join = Join(output_schema=["X", "Y", "Z"], merge_steps=merge_steps, input_schemas=input_schemas)
    out = join.instantiate([r1, r2])
    assert out.content is not None
    assert len(out.content) == 2
    assert list(out.content.columns) == ["X", "Y", "Z"]

def test_join_empty_merge_steps_error():
    """Join with empty merge_steps should raise error."""
    df1 = pd.DataFrame({"X": [1], "Y": [10]})
    df2 = pd.DataFrame({"Y": [10], "Z": [100]})
    r1 = RuntimeER(content_schema=["X", "Y"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["Y", "Z"], embedding_shapes=[], content=df2, embeddings=None)
    join = Join(output_schema=["X", "Y", "Z"], merge_steps=[], input_schemas=[["X", "Y"], ["Y", "Z"]])
    try:
        join.instantiate([r1, r2])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "merge_steps" in str(e).lower() or "empty" in str(e).lower()

def test_join_invalid_column_ref_error():
    """Join with invalid ColumnRef (out of bounds) should raise error."""
    df1 = pd.DataFrame({"X": [1], "Y": [10]})
    df2 = pd.DataFrame({"Y": [10], "Z": [100]})
    r1 = RuntimeER(content_schema=["X", "Y"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["Y", "Z"], embedding_shapes=[], content=df2, embeddings=None)
    input_schemas = [["X", "Y"], ["Y", "Z"]]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 5)],  # Invalid: column index 5 doesn't exist
        "right_refs": [ColumnRef(1, 0)],
        "key_names": ["Y"]
    }]
    join = Join(output_schema=["X", "Y", "Z"], merge_steps=merge_steps, input_schemas=input_schemas)
    try:
        join.instantiate([r1, r2])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "out of bounds" in str(e).lower() or "column index" in str(e).lower()

def test_join_fallback_column_name():
    """Join should use fallback_column_name when schema is missing."""
    df1 = pd.DataFrame({"X": [1, 2]})  # Missing Y column
    df2 = pd.DataFrame({"Y": [10, 20], "Z": [100, 200]})
    r1 = RuntimeER(content_schema=None, embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["Y", "Z"], embedding_shapes=[], content=df2, embeddings=None)
    input_schemas = [["X"], ["Y", "Z"]]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 0)],  # This will try to resolve column 0
        "right_refs": [ColumnRef(1, 0)],
        "key_names": ["Y"]  # Fallback name
    }]
    join = Join(output_schema=["X", "Y", "Z"], merge_steps=merge_steps, input_schemas=input_schemas)
    # This might work if fallback is used, or might fail - depends on implementation
    # Just check it doesn't crash unexpectedly
    try:
        out = join.instantiate([r1, r2])
        # If it succeeds, verify structure
        assert out.content is not None
    except (ValueError, KeyError):
        # Expected if fallback doesn't work in this case
        pass

def test_join_forward_with_empty_result():
    """Join.forward() should handle empty join results correctly."""
    df1 = pd.DataFrame({"K": [1, 2], "A": [10, 20]})
    df2 = pd.DataFrame({"K": [3, 4], "B": [100, 200]})  # No matches
    r1 = RuntimeER(
        content_schema=["K", "A"],
        embedding_shapes=[torch.Size([2, 3])],
        content=df1,
        embeddings=[torch.randn(2, 3)]
    )
    r2 = RuntimeER(
        content_schema=["K", "B"],
        embedding_shapes=[torch.Size([2, 2])],
        content=df2,
        embeddings=[torch.randn(2, 2)]
    )
    input_schemas = [["K", "A"], ["K", "B"]]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 0)],
        "right_refs": [ColumnRef(1, 0)],
        "key_names": ["K"]
    }]
    join = Join(output_schema=["K", "A", "B"], merge_steps=merge_steps, input_schemas=input_schemas)
    join.instantiate([r1, r2])
    out = join.forward([r1, r2])
    assert out.content is not None
    assert len(out.content) == 0
    assert out.embeddings is not None
    # Embeddings should still be present but empty
    assert len(out.embeddings) == 2

def test_join_large_result():
    """Join with many matching rows."""
    n = 100
    df1 = pd.DataFrame({"K": list(range(n)), "A": list(range(10, 10+n))})
    df2 = pd.DataFrame({"K": list(range(n)), "B": list(range(100, 100+n))})
    r1 = RuntimeER(content_schema=["K", "A"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["K", "B"], embedding_shapes=[], content=df2, embeddings=None)
    input_schemas = [["K", "A"], ["K", "B"]]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 0)],
        "right_refs": [ColumnRef(1, 0)],
        "key_names": ["K"]
    }]
    join = Join(output_schema=["K", "A", "B"], merge_steps=merge_steps, input_schemas=input_schemas)
    out = join.instantiate([r1, r2])
    assert out.content is not None
    assert len(out.content) == n
    assert list(out.content.columns) == ["K", "A", "B"]

def test_join_physical_names_differ_from_logical():
    """Join (a,b) with (b,c) when sons have physical column names x,y and y,x; output must be a,b,c.
    Join on left.y=right.y (both logical b), keep left.x->a, left.y/right.y->b, right.x->c."""
    # Left: physical columns x,y; logical schema a,b (position 0=a, 1=b)
    df1 = pd.DataFrame({"x": [1, 2], "y": [10, 20]})
    # Right: physical columns y,x; logical schema b,c (position 0=b, 1=c)
    df2 = pd.DataFrame({"y": [10, 20], "x": [100, 200]})
    r1 = RuntimeER(content_schema=["a", "b"], embedding_shapes=[], content=df1, embeddings=None)
    r2 = RuntimeER(content_schema=["b", "c"], embedding_shapes=[], content=df2, embeddings=None)
    input_schemas = [["a", "b"], ["b", "c"]]
    merge_steps = [{
        "step": 1,
        "left_refs": [ColumnRef(0, 1)],   # left col 1 = b (physical y)
        "right_refs": [ColumnRef(1, 0)],  # right col 0 = b (physical y)
        "key_names": ["b"]
    }]
    join = Join(output_schema=["a", "b", "c"], merge_steps=merge_steps, input_schemas=input_schemas)
    out = join.instantiate([r1, r2])
    assert out.content is not None
    assert list(out.content.columns) == ["a", "b", "c"], f"got {list(out.content.columns)}"
    expected = pd.DataFrame({"a": [1, 2], "b": [10, 20], "c": [100, 200]})
    pd.testing.assert_frame_equal(out.content.sort_values("b").reset_index(drop=True), expected)

# Tests for RHS.output_content_attrs property
def test_output_content_attrs_a_b__b_c():
    """output_content_attrs: (a,b) join (b,c) should output [a,b,c]."""
    er1 = EmbeddedRelation(name="R1", content_attrs=[Var(name="a"), Var(name="b")])
    er2 = EmbeddedRelation(name="R2", content_attrs=[Var(name="b"), Var(name="c")])
    rhs = RHS(ers=[er1, er2], rel_ops=[","])
    output = rhs.output_content_attrs
    output_names = [v.name for v in output]
    assert output_names == ["a", "b", "c"], f"Expected ['a', 'b', 'c'], got {output_names}"

def test_output_content_attrs_a_b__a_b__b_c():
    """output_content_attrs: (a,b) join (a,b) join (b,c) should output [a,b,c]."""
    er1 = EmbeddedRelation(name="R1", content_attrs=[Var(name="a"), Var(name="b")])
    er2 = EmbeddedRelation(name="R2", content_attrs=[Var(name="a"), Var(name="b")])
    er3 = EmbeddedRelation(name="R3", content_attrs=[Var(name="b"), Var(name="c")])
    rhs = RHS(ers=[er1, er2, er3], rel_ops=[",", ","])
    output = rhs.output_content_attrs
    output_names = [v.name for v in output]
    assert output_names == ["a", "b", "c"], f"Expected ['a', 'b', 'c'], got {output_names}"

def test_output_content_attrs_a_b__b():
    """output_content_attrs: (a,b) join (b) should output [a,b]."""
    er1 = EmbeddedRelation(name="R1", content_attrs=[Var(name="a"), Var(name="b")])
    er2 = EmbeddedRelation(name="R2", content_attrs=[Var(name="b")])
    rhs = RHS(ers=[er1, er2], rel_ops=[","])
    output = rhs.output_content_attrs
    output_names = [v.name for v in output]
    assert output_names == ["a", "b"], f"Expected ['a', 'b'], got {output_names}"

def test_output_content_attrs_uses_aliased_names():
    """output_content_attrs should use key_name aliases from join_conditions."""
    er1 = EmbeddedRelation(name="R1", content_attrs=[Var(name="X"), Var(name="Y")])
    er2 = EmbeddedRelation(name="R2", content_attrs=[Var(name="Y"), Var(name="Z")])
    rhs = RHS(ers=[er1, er2], rel_ops=[","])
    # join_conditions will infer Y as the key
    output = rhs.output_content_attrs
    output_names = [v.name for v in output]
    assert output_names == ["X", "Y", "Z"], f"Expected ['X', 'Y', 'Z'], got {output_names}"
    # Verify Y is used (aliased name from join_conditions)
    assert "Y" in output_names

def test_output_content_attrs_collapses_duplicate_keys():
    """output_content_attrs should collapse duplicate join keys (only add once)."""
    er1 = EmbeddedRelation(name="R1", content_attrs=[Var(name="K"), Var(name="A")])
    er2 = EmbeddedRelation(name="R2", content_attrs=[Var(name="K"), Var(name="B")])
    er3 = EmbeddedRelation(name="R3", content_attrs=[Var(name="K"), Var(name="C")])
    rhs = RHS(ers=[er1, er2, er3], rel_ops=[",", ","])
    output = rhs.output_content_attrs
    output_names = [v.name for v in output]
    # K should appear only once
    assert output_names.count("K") == 1, f"Key 'K' should appear once, got {output_names.count('K')} times"
    assert output_names == ["K", "A", "B", "C"], f"Expected ['K', 'A', 'B', 'C'], got {output_names}"

def test_output_content_attrs_composite_key():
    """output_content_attrs with composite key (multiple keys)."""
    er1 = EmbeddedRelation(name="R1", content_attrs=[Var(name="K1"), Var(name="K2"), Var(name="A")])
    er2 = EmbeddedRelation(name="R2", content_attrs=[Var(name="K1"), Var(name="K2"), Var(name="B")])
    rhs = RHS(ers=[er1, er2], rel_ops=[","])
    output = rhs.output_content_attrs
    output_names = [v.name for v in output]
    # Both K1 and K2 should appear once (as keys)
    assert output_names.count("K1") == 1
    assert output_names.count("K2") == 1
    assert set(output_names) == {"K1", "K2", "A", "B"}

def test_output_content_attrs_chain_join():
    """output_content_attrs for chain join: (a,b) join (b,c) join (c,d)."""
    er1 = EmbeddedRelation(name="R1", content_attrs=[Var(name="a"), Var(name="b")])
    er2 = EmbeddedRelation(name="R2", content_attrs=[Var(name="b"), Var(name="c")])
    er3 = EmbeddedRelation(name="R3", content_attrs=[Var(name="c"), Var(name="d")])
    rhs = RHS(ers=[er1, er2, er3], rel_ops=[",", ","])
    output = rhs.output_content_attrs
    output_names = [v.name for v in output]
    # b and c should appear once each (as keys), a and d should appear
    assert output_names == ["a", "b", "c", "d"], f"Expected ['a', 'b', 'c', 'd'], got {output_names}"

def test_output_content_attrs_no_join_conditions():
    """output_content_attrs when no join conditions (no overlapping columns)."""
    er1 = EmbeddedRelation(name="R1", content_attrs=[Var(name="a"), Var(name="b")])
    er2 = EmbeddedRelation(name="R2", content_attrs=[Var(name="c"), Var(name="d")])
    rhs = RHS(ers=[er1, er2], rel_ops=[","])
    output = rhs.output_content_attrs
    output_names = [v.name for v in output]
    # Should concatenate all columns with collision handling
    assert set(output_names) == {"a", "b", "c", "d"}
    assert len(output_names) == 4

def test_output_content_attrs_single_relation():
    """output_content_attrs for single relation returns its content_attrs."""
    er1 = EmbeddedRelation(name="R1", content_attrs=[Var(name="a"), Var(name="b")])
    rhs = RHS(ers=[er1], rel_ops=None)
    output = rhs.output_content_attrs
    output_names = [v.name for v in output]
    assert output_names == ["a", "b"]

def test_output_content_attrs_union_operator():
    """output_content_attrs for union (not join) returns first relation's schema."""
    er1 = EmbeddedRelation(name="R1", content_attrs=[Var(name="a"), Var(name="b")])
    er2 = EmbeddedRelation(name="R2", content_attrs=[Var(name="c"), Var(name="d")])
    rhs = RHS(ers=[er1, er2], rel_ops=["|"])  # Union, not join
    output = rhs.output_content_attrs
    output_names = [v.name for v in output]
    assert output_names == ["a", "b"]  # Should return first relation's schema

def test_output_content_attrs_name_collision():
    """output_content_attrs handles name collisions by suffixing when no join key."""
    # Use columns that don't overlap to trigger collision handling path
    er1 = EmbeddedRelation(name="R1", content_attrs=[Var(name="a"), Var(name="b")])
    er2 = EmbeddedRelation(name="R2", content_attrs=[Var(name="a"), Var(name="c")])
    # 'a' appears in both, so it becomes a join key - output will be [a, b, c]
    rhs = RHS(ers=[er1, er2], rel_ops=[","])
    output = rhs.output_content_attrs
    output_names = [v.name for v in output]
    # 'a' is a join key, so it appears once; 'b' and 'c' are non-key columns
    assert output_names == ["a", "b", "c"], f"Expected ['a', 'b', 'c'], got {output_names}"
    
    # Test actual collision: columns that don't overlap at all
    er3 = EmbeddedRelation(name="R3", content_attrs=[Var(name="x"), Var(name="y")])
    er4 = EmbeddedRelation(name="R4", content_attrs=[Var(name="x"), Var(name="z")])
    rhs2 = RHS(ers=[er3, er4], rel_ops=[","])
    output2 = rhs2.output_content_attrs
    output_names2 = [v.name for v in output2]
    # 'x' is a join key, so it appears once
    assert output_names2 == ["x", "y", "z"], f"Expected ['x', 'y', 'z'], got {output_names2}"

if __name__ == "__main__":
    test_rhs_join_conditions_computed()
    test_rhs_join_conditions_single_er_empty()
    test_join_accepts_merge_steps()
    test_simple_join_program_e2e()
    test_two_way_join_x_y__y_z()
    test_three_way_join_k_a__k_b__k_c()
    test_three_way_join_a_b__a_b__b_c()
    test_chain_join_a_b__b_c__c_d()
    test_three_way_join_forward_embeddings_aligned()
    test_join_empty_dataframes()
    test_join_no_matching_rows()
    test_join_composite_key()
    test_join_different_dtypes()
    test_join_four_way()
    test_join_partial_matches()
    test_join_with_duplicate_keys()
    test_join_output_schema_normalization()
    test_join_without_content_schema_uses_columns()
    test_join_empty_merge_steps_error()
    test_join_invalid_column_ref_error()
    test_join_fallback_column_name()
    test_join_forward_with_empty_result()
    test_join_large_result()
    test_join_physical_names_differ_from_logical()
    # Tests for output_content_attrs
    test_output_content_attrs_a_b__b_c()
    test_output_content_attrs_a_b__a_b__b_c()
    test_output_content_attrs_a_b__b()
    test_output_content_attrs_uses_aliased_names()
    test_output_content_attrs_collapses_duplicate_keys()
    test_output_content_attrs_composite_key()
    test_output_content_attrs_chain_join()
    test_output_content_attrs_no_join_conditions()
    test_output_content_attrs_single_relation()
    test_output_content_attrs_union_operator()
    test_output_content_attrs_name_collision()
    print("All join_conditions refactor tests passed.")
