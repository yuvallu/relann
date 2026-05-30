"""Smoke tests: template parameters for TransformDef, FunctionDef, and Rule.

End-to-end tests that define templated DSL code, run predict, and verify output shapes.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from relann.session import Session
from relann.torch_utils import full_seed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _papers_db(n: int = 6, d: int = 8):
    df = pd.DataFrame({"pid": list(range(n))})
    z = torch.randn(n, d)
    return {"Papers": (df, z)}

def _gcn_db(n: int = 6, d: int = 8):
    papers_df = pd.DataFrame({"pid": list(range(n))})
    papers_z = torch.randn(n, d)
    citing = list(range(n))
    cited = [(i + 1) % n for i in range(n)]
    citation_df = pd.DataFrame({"citing": citing, "cited": cited})
    citation_w = torch.ones(len(citation_df), 1)
    return {
        "Papers": (papers_df, papers_z),
        "Citation": (citation_df, citation_w),
    }

def _msg_pass_db(n: int = 6, d: int = 4):
    """DB with generic node/edge tables for message-passing tests."""
    nodes_df = pd.DataFrame({"nid": list(range(n))})
    nodes_z = torch.randn(n, d)
    sources = list(range(n))
    targets = [(i + 1) % n for i in range(n)]
    edges_df = pd.DataFrame({"source": sources, "target": targets})
    edges_w = torch.ones(len(edges_df), 1)
    return {
        "Nodes": (nodes_df, nodes_z),
        "Edges": (edges_df, edges_w),
    }

# ---------------------------------------------------------------------------
# TransformDef template smoke tests
# ---------------------------------------------------------------------------

def test_templated_transformdef_predict_shape():
    """Define a template TransformDef, use with concrete args, predict correct shape."""
    full_seed(42)

    n, in_channels, out_channels = 6, 8, 3

    session = Session(db=_papers_db(n, in_channels))

    session.run(f"""
#lang:relnn
in_channels = {in_channels} .

Lin<d_out> = Linear(in_channels, d_out, False) .

Output(pid; Lin<{out_channels}>(z)) :- Papers(pid; z) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(pid; z) :- Output(pid; z) .
""")

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n, out_channels)

def test_templated_transformdef_two_params_predict_shape():
    """A TransformDef with two template params should materialize correctly."""
    full_seed(42)

    n, in_ch, out_ch = 6, 8, 5

    session = Session(db=_papers_db(n, in_ch))

    session.run(f"""
#lang:relnn
Lin<d_in, d_out> = Linear(d_in, d_out, False) .

Output(pid; Lin<{in_ch}, {out_ch}>(z)) :- Papers(pid; z) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(pid; z) :- Output(pid; z) .
""")

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n, out_ch)

# ---------------------------------------------------------------------------
# FunctionDef template smoke tests
# ---------------------------------------------------------------------------

def test_templated_functiondef_predict_shape():
    """Template function with scalar params should materialize and predict."""
    full_seed(42)

    n, in_channels, hidden_channels, out_channels = 6, 8, 4, 3

    session = Session(db=_gcn_db(n, in_channels))

    session.run(f"""
#lang:relnn
def GCN<in_ch, out_ch>(Papers, Citation):
    Emb(pid; Linear(in_ch, out_ch, False)(z)) :- Papers(pid; z) .
    Agg(cited; sum(z * w)) :- Emb(citing; z), Citation(citing, cited; w) .
enddef

Output(cited; z) :- GCN<{in_channels}, {out_channels}>(Papers, Citation)(cited; z) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(cited; z) :- Output(cited; z) .
""")

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n, out_channels)

def test_templated_functiondef_with_er_template_param_predict_shape():
    """A function with an ER template param should substitute the ER name in the body."""
    full_seed(42)

    n, d = 6, 4

    session = Session(db=_msg_pass_db(n, d))

    session.run("""
#lang:relnn
def MsgPass<EdgeER>(Nodes):
    Msg(target; sum(z * w)) :- Nodes(source; z), EdgeER(source, target; w) .
enddef

Output(target; z) :- MsgPass<Edges>(Nodes)(target; z) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(target; z) :- Output(target; z) .
""")

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n, d)

def test_templated_functiondef_mixed_scalar_and_er_params():
    """Template function with both scalar and ER template params."""
    full_seed(42)

    n, d, out_ch = 6, 4, 2

    session = Session(db=_msg_pass_db(n, d))

    session.run(f"""
#lang:relnn
def MsgPassEmb<EdgeER, out_d>(Nodes):
    Msg(target; sum(z * w)) :- Nodes(source; z), EdgeER(source, target; w) .
    Emb(target; Linear({d}, out_d, False)(z)) :- Msg(target; z) .
enddef

Output(target; z) :- MsgPassEmb<Edges, {out_ch}>(Nodes)(target; z) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(target; z) :- Output(target; z) .
""")

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n, out_ch)

# ---------------------------------------------------------------------------
# Rule template smoke tests
# ---------------------------------------------------------------------------

def test_templated_rule_predict_shape():
    """A templated rule should be materialized when referenced with concrete args."""
    full_seed(42)

    n, in_ch, out_ch = 6, 8, 3

    session = Session(db=_papers_db(n, in_ch))

    session.run(f"""
#lang:relnn
Layer<d_out>(pid; Linear({in_ch}, d_out, False)(z)) :- Papers(pid; z) .

Output(pid; z) :- Layer<{out_ch}>(pid; z) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(pid; z) :- Output(pid; z) .
""")

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n, out_ch)

# ---------------------------------------------------------------------------
# Weight sharing tests
# ---------------------------------------------------------------------------

def test_same_template_different_args_different_param_shapes():
    """Two instantiations of the same template with different args get different parameters."""
    full_seed(42)

    n, in_ch = 6, 8
    out_a, out_b = 3, 5

    session = Session(db=_papers_db(n, in_ch))

    session.run(f"""
#lang:relnn
Lin<d_out> = Linear({in_ch}, d_out, False) .

OutputA(pid; Lin<{out_a}>(z)) :- Papers(pid; z) .
OutputB(pid; Lin<{out_b}>(z)) :- Papers(pid; z) .
""")

    session.run("""
#lang:relnn
?pred PredA(pid; z) :- OutputA(pid; z) .
""")

    result_b = session.run("""
#lang:relnn
?pred PredB(pid; z) :- OutputB(pid; z) .
""")

    # Find parameters for each instantiation in the parameter store
    params = session.engine.parameter_store
    lin_a_params = {k: v for k, v in params.items() if str(out_a) in k or "OutputA" in k}
    lin_b_params = {k: v for k, v in params.items() if str(out_b) in k or "OutputB" in k}

    # They should have different shapes
    a_shapes = {tuple(v.shape) for v in lin_a_params.values()}
    b_shapes = {tuple(v.shape) for v in lin_b_params.values()}
    assert a_shapes != b_shapes, (
        f"Different template args should produce different param shapes: {a_shapes} vs {b_shapes}"
    )

def test_same_template_same_args_shared_weights():
    """Two uses of the same template with identical args should share weights."""
    full_seed(42)

    n, in_ch, out_ch = 6, 8, 3

    papers_a_df = pd.DataFrame({"pid": list(range(n))})
    papers_b_df = pd.DataFrame({"pid": list(range(n))})
    papers_a_z = torch.randn(n, in_ch)
    papers_b_z = torch.randn(n, in_ch)

    session = Session(db={
        "PapersA": (papers_a_df, papers_a_z),
        "PapersB": (papers_b_df, papers_b_z),
    })

    session.run(f"""
#lang:relnn
Lin<d_out> = Linear({in_ch}, d_out, False) .

OutA(pid; Lin<{out_ch}>(z)) :- PapersA(pid; z) .
OutB(pid; Lin<{out_ch}>(z)) :- PapersB(pid; z) .

Pair(pid; Concat(z1, z2)) :- OutA(pid; z1), OutB(pid; z2) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(pid; z) :- Pair(pid; z) .
""")

    assert result is not None
    emb = result.embeddings[0]
    assert tuple(emb.shape) == (n, out_ch * 2)

    # Both halves should use the same weight matrix.
    # Apply the same input to both branches: if weights are shared, outputs match.
    same_input = torch.randn(n, in_ch)
    # Since both Lin<out_ch> instantiations share weights, applying the same input
    # through both branches should produce identical results.
    # We verify by checking the parameter store has shared entries.
    params = session.engine.parameter_store
    lin_params = [v for k, v in params.items() if "Lin" in k or "weight" in k.lower()]
    # With weight sharing, there should be exactly 1 weight tensor for Lin<out_ch>
    # (not 2 separate ones)
    weight_tensors = [p for p in lin_params if p.shape == (out_ch, in_ch)]
    assert len(weight_tensors) >= 1, "Should have at least one Lin weight tensor"
    if len(weight_tensors) > 1:
        assert weight_tensors[0].data_ptr() == weight_tensors[1].data_ptr(), (
            "Same template args should share the underlying weight tensor"
        )

# ---------------------------------------------------------------------------
# Nested / recursive template tests
# ---------------------------------------------------------------------------

def test_nested_templates_function_using_templated_transformdef():
    """A templated FunctionDef whose body uses a templated TransformDef should work end-to-enddef"""
    full_seed(42)

    n, in_ch, hidden_ch, out_ch = 6, 8, 5, 3

    session = Session(db=_gcn_db(n, in_ch))

    session.run(f"""
#lang:relnn
Lin<d_in, d_out> = Linear(d_in, d_out, False) .

def GCN<h>(Papers, Citation):
    Emb(pid; Lin<{in_ch}, h>(z)) :- Papers(pid; z) .
    Agg(cited; sum(z * w)) :- Emb(citing; z), Citation(citing, cited; w) .
enddef

Output(cited; Lin<{hidden_ch}, {out_ch}>(z)) :- GCN<{hidden_ch}>(Papers, Citation)(cited; z) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(cited; z) :- Output(cited; z) .
""")

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n, out_ch)

# ---------------------------------------------------------------------------
# Edge-case tests (from code review)
# ---------------------------------------------------------------------------

def test_same_templated_rule_used_twice_in_program():
    """Using the same templated Rule with the same args in two outer rules
    should not produce duplicate term-graph nodes or errors."""
    full_seed(42)

    n, in_ch, out_ch = 6, 8, 3

    session = Session(db=_papers_db(n, in_ch))

    session.run(f"""
#lang:relnn
Layer<d_out>(pid; Linear({in_ch}, d_out, False)(z)) :- Papers(pid; z) .

OutputA(pid; z) :- Layer<{out_ch}>(pid; z) .
OutputB(pid; z) :- Layer<{out_ch}>(pid; z) .
""")

    result_a = session.run("""
#lang:relnn
?pred PredA(pid; z) :- OutputA(pid; z) .
""")
    result_b = session.run("""
#lang:relnn
?pred PredB(pid; z) :- OutputB(pid; z) .
""")

    assert tuple(result_a.embeddings[0].shape) == (n, out_ch)
    assert tuple(result_b.embeddings[0].shape) == (n, out_ch)

def test_template_with_expression_arg():
    """Template arg can be an arithmetic expression like d/2."""
    full_seed(42)

    n, d = 6, 8

    session = Session(db=_papers_db(n, d))

    session.run(f"""
#lang:relnn
d = {d} .
Lin<d_out> = Linear({d}, d_out, False) .

Output(pid; Lin<d / 2>(z)) :- Papers(pid; z) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(pid; z) :- Output(pid; z) .
""")

    assert result is not None
    assert tuple(result.embeddings[0].shape) == (n, d // 2)

def test_template_redefinition_overwrites():
    """Re-defining a templated TransformDef should overwrite the first definition."""
    full_seed(42)

    n, in_ch = 6, 8
    out_ch = 3

    session = Session(db=_papers_db(n, in_ch))

    session.run(f"""
#lang:relnn
Lin<d_out> = Linear(999, d_out, False) .
Lin<d_out> = Linear({in_ch}, d_out, False) .

Output(pid; Lin<{out_ch}>(z)) :- Papers(pid; z) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(pid; z) :- Output(pid; z) .
""")

    assert result is not None
    assert tuple(result.embeddings[0].shape) == (n, out_ch)

# ---------------------------------------------------------------------------
# Tag-only template params (param not in body, just differentiates cache key)
# ---------------------------------------------------------------------------

def test_tag_only_template_param_transformdef():
    """A TransformDef template param that never appears in the body should still
    produce separate cache entries (and thus separate weights) per tag value."""
    full_seed(42)

    n, d = 6, 4
    db_a = pd.DataFrame({"nid": list(range(n))})
    db_b = pd.DataFrame({"nid": list(range(n))})
    za = torch.randn(n, d)
    zb = torch.randn(n, d)
    session = Session(db={"TableA": (db_a, za), "TableB": (db_b, zb)})

    session.run(f"""
#lang:relnn
W<tag> = Linear({d}, {d}, False) .

OutA(nid; W<TypeA>(z)) :- TableA(nid; z) .
OutB(nid; W<TypeB>(z)) :- TableB(nid; z) .
Pair(nid; Concat(z1, z2)) :- OutA(nid; z1), OutB(nid; z2) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(nid; z) :- Pair(nid; z) .
""")

    assert result is not None
    emb = result.embeddings[0]
    assert tuple(emb.shape) == (n, d * 2)

    cache = session.engine._template_instance_cache
    assert "W<TypeA>" in cache, f"Expected W<TypeA> in cache, got {list(cache.keys())}"
    assert "W<TypeB>" in cache, f"Expected W<TypeB> in cache, got {list(cache.keys())}"

    params = session.engine.parameter_store
    w_a = [v for k, v in params.items() if "TypeA" in k]
    w_b = [v for k, v in params.items() if "TypeB" in k]
    assert len(w_a) >= 1 and len(w_b) >= 1, "Should have separate params per tag"
    assert w_a[0].data_ptr() != w_b[0].data_ptr(), "Tag-only params must NOT share weights"

# ---------------------------------------------------------------------------
# Real-architecture tests (Cora dataset)
# ---------------------------------------------------------------------------

def test_heterogeneous_gcn_2types():
    """Heterogeneous GCN with 2 node types, tag-only params for per-type projection,
    and ER template params for per-edge message passing."""
    full_seed(42)

    n_a, d_a = 8, 4
    n_b, d_b = 6, 8
    d_out = 4

    type_a_df = pd.DataFrame({"nid": list(range(n_a))})
    type_a_z = torch.randn(n_a, d_a)

    type_b_df = pd.DataFrame({"nid": list(range(n_b))})
    type_b_z = torch.randn(n_b, d_b)

    a_to_b_src = list(range(n_a))
    a_to_b_dst = [i % n_b for i in range(n_a)]
    a_b_df = pd.DataFrame({"source": a_to_b_src, "target": a_to_b_dst})
    a_b_w = torch.ones(len(a_b_df), 1)

    b_to_a_src = [i % n_b for i in range(n_a)]
    b_to_a_dst = list(range(n_a))
    b_a_df = pd.DataFrame({"source": b_to_a_src, "target": b_to_a_dst})
    b_a_w = torch.ones(len(b_a_df), 1)

    session = Session(db={
        "TypeA": (type_a_df, type_a_z),
        "TypeB": (type_b_df, type_b_z),
        "A_B": (a_b_df, a_b_w),
        "B_A": (b_a_df, b_a_w),
    })

    session.run(f"""
#lang:relnn
d = {d_out} .

Proj<tag, d_in> = Linear(d_in, d, False) .

def MsgPass<EdgeER>(SrcNodes):
    Out(target; sum(z * w)) :- SrcNodes(source; z), EdgeER(source, target; w) .
enddef

TypeA_Emb(nid; Proj<TypeA, {d_a}>(z)) :- TypeA(nid; z) .
TypeB_Emb(nid; Proj<TypeB, {d_b}>(z)) :- TypeB(nid; z) .

AtoB(target; z) :- MsgPass<A_B>(TypeA_Emb)(target; z) .
BtoA(target; z) :- MsgPass<B_A>(TypeB_Emb)(target; z) .

Output(nid; Concat(z1, z2)) :- BtoA(nid; z1), TypeA_Emb(nid; z2) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(nid; z) :- Output(nid; z) .
""")

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n_a, d_out * 2)

    cache = session.engine._template_instance_cache
    assert "Proj<TypeA,{}>".format(d_a) in cache
    assert "Proj<TypeB,{}>".format(d_b) in cache

def _skip_if_no_cora():
    """Return True (and print SKIP) when the Cora dataset is unavailable."""
    try:
        from relann.torch_utils import get_project_root
        path = get_project_root() / "data" / "Planetoid" / "Cora"
        if not path.exists():
            print("SKIP: Cora data not found at", path)
            return True
    except Exception as e:
        print(f"SKIP: cannot locate Cora data ({e})")
        return True
    return False

def test_templated_gcn_layer_cora():
    """2-layer GCN on Cora built from a GCNLayer<d_in, d_out> FunctionDef template."""
    if _skip_if_no_cora():
        return

    full_seed(42)
    from relann.datasets import load_cora_dataset, evaluate_node_classification

    data = load_cora_dataset()
    db = {k: data[k] for k in ("Papers", "Citation", "Labels")}
    session = Session(db=db)

    session.run("""
#lang:relnn
in_channels = 1433 .
hidden_channels = 16 .
out_channels = 7 .

def GCNLayer<d_in, d_out>(Nodes, Edges):
    Emb(pid; Linear(d_in, d_out, False)(z)) :- Nodes(pid; z) .
    Out(cited; sum(z * w)) :- Emb(citing; z), Edges(citing, cited; w) .
enddef

L1(cited; ReLU(z)) :- GCNLayer<in_channels, hidden_channels>(Papers, Citation)(cited; z) .
Output(cited; z) :- GCNLayer<hidden_channels, out_channels>(L1, Citation)(cited; z) .
""")

    session.run("""
#lang:relnn
?fit <epochs=100, lr=0.01, weight_decay=0.0005>
Loss(; CrossEntropyLoss()(z_pred, z)) :- Output(cited; z_pred), Labels(cited; z) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(cited; ArgMax()(z)) :- Output(cited; z) .
""")

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    n_nodes = 2708
    assert result.embeddings[0].shape[0] == n_nodes

    acc = evaluate_node_classification(data, result, return_value=True)
    assert acc > 0.65, f"Expected >65% test accuracy, got {acc:.1%}"

def test_templated_gcn_composed_cora():
    """GCN<d_in, d_hidden, d_out> composes two GCNLayer calls (template-of-template)."""
    if _skip_if_no_cora():
        return

    full_seed(42)
    from relann.datasets import load_cora_dataset, evaluate_node_classification

    data = load_cora_dataset()
    db = {k: data[k] for k in ("Papers", "Citation", "Labels")}
    session = Session(db=db)

    session.run("""
#lang:relnn
in_channels = 1433 .
hidden_channels = 16 .
out_channels = 7 .

def GCNLayer<d_in, d_out>(Nodes, Edges):
    Emb(pid; Linear(d_in, d_out, False)(z)) :- Nodes(pid; z) .
    Out(cited; sum(z * w)) :- Emb(citing; z), Edges(citing, cited; w) .
enddef

def GCN<d_in, d_hidden, d_out>(Nodes, Edges):
    L1(cited; ReLU(z)) :- GCNLayer<d_in, d_hidden>(Nodes, Edges)(cited; z) .
    Output(cited; z) :- GCNLayer<d_hidden, d_out>(L1, Edges)(cited; z) .
enddef

Output(cited; z) :- GCN<in_channels, hidden_channels, out_channels>(Papers, Citation)(cited; z) .
""")

    session.run("""
#lang:relnn
?fit <epochs=100, lr=0.01, weight_decay=0.0005>
Loss(; CrossEntropyLoss()(z_pred, z)) :- Output(cited; z_pred), Labels(cited; z) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(cited; ArgMax()(z)) :- Output(cited; z) .
""")

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    n_nodes = 2708
    assert result.embeddings[0].shape[0] == n_nodes

    acc = evaluate_node_classification(data, result, return_value=True)
    assert acc > 0.65, f"Expected >65% test accuracy, got {acc:.1%}"

def test_exp_and_edge_softmax():
    """Verify exp() works as a DSL op and EdgeSoftmax FunctionDef computes correct softmax."""
    edges_df = pd.DataFrame({"s": [0, 1, 0], "t": [1, 1, 0]})
    edges_z = torch.tensor([[1.0], [1.0], [1.0]])
    nodes_df = pd.DataFrame({"n": [0, 1]})
    nodes_z = torch.tensor([[1.0], [2.0]])

    db = {"Edges": (edges_df, edges_z), "Nodes": (nodes_df, nodes_z)}
    session = Session(db=db)

    result = session.run("""
#lang:relnn
Scores(s, t; z * 1.0) :- Nodes(s; z), Edges(s, t; w) .

def EdgeSoftmax(Scores):
    Exp(s, t; exp(z)) :- Scores(s, t; z) .
    Denom(t; sum(z)) :- Exp(s, t; z) .
    Out(s, t; z1 / z2) :- Exp(s, t; z1), Denom(t; z2) .
enddef

Soft(s, t; z) :- EdgeSoftmax(Scores)(s, t; z) .
?pred P(s, t; z) :- Soft(s, t; z) .
""")

    assert result is not None
    out = result.embeddings[0]
    df = result.content

    for _, row in df.iterrows():
        s_id, t_id = int(row["s"]), int(row["t"])
        idx = df.index[df.index == row.name][0]
        val = out[idx].item()
        if t_id == 0:
            assert abs(val - 1.0) < 1e-5, f"Edge ({s_id},{t_id}): expected 1.0, got {val}"
        elif t_id == 1:
            expected = torch.exp(torch.tensor(float(s_id + 1))) / (torch.exp(torch.tensor(1.0)) + torch.exp(torch.tensor(2.0)))
            assert abs(val - expected.item()) < 1e-4, f"Edge ({s_id},{t_id}): expected {expected.item():.4f}, got {val:.4f}"

# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_templated_transformdef_predict_shape()
    print("  PASS: test_templated_transformdef_predict_shape")

    test_templated_transformdef_two_params_predict_shape()
    print("  PASS: test_templated_transformdef_two_params_predict_shape")

    test_templated_functiondef_predict_shape()
    print("  PASS: test_templated_functiondef_predict_shape")

    test_templated_functiondef_with_er_template_param_predict_shape()
    print("  PASS: test_templated_functiondef_with_er_template_param_predict_shape")

    test_templated_functiondef_mixed_scalar_and_er_params()
    print("  PASS: test_templated_functiondef_mixed_scalar_and_er_params")

    test_templated_rule_predict_shape()
    print("  PASS: test_templated_rule_predict_shape")

    test_same_template_different_args_different_param_shapes()
    print("  PASS: test_same_template_different_args_different_param_shapes")

    test_same_template_same_args_shared_weights()
    print("  PASS: test_same_template_same_args_shared_weights")

    test_nested_templates_function_using_templated_transformdef()
    print("  PASS: test_nested_templates_function_using_templated_transformdef")

    test_same_templated_rule_used_twice_in_program()
    print("  PASS: test_same_templated_rule_used_twice_in_program")

    test_template_with_expression_arg()
    print("  PASS: test_template_with_expression_arg")

    test_template_redefinition_overwrites()
    print("  PASS: test_template_redefinition_overwrites")

    test_tag_only_template_param_transformdef()
    print("  PASS: test_tag_only_template_param_transformdef")

    test_heterogeneous_gcn_2types()
    print("  PASS: test_heterogeneous_gcn_2types")

    test_templated_gcn_layer_cora()
    print("  PASS: test_templated_gcn_layer_cora")

    test_templated_gcn_composed_cora()
    print("  PASS: test_templated_gcn_composed_cora")

    test_exp_and_edge_softmax()
    print("  PASS: test_exp_and_edge_softmax")

    print("\nAll smoke tests passed!")
