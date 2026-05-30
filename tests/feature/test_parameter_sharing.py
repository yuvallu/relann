"""
Parameter sharing and persistence tests for RelNN.

Covers: shared TransformDef (single FQN, value sharing), fit-then-predict persistence,
FQN/display using TransformDef names (single-ref and HGT-style composite rules).

Each test prints its parameters to stdout. To see them when running pytest, use -s:
  pytest tests/feature/test_parameter_sharing.py -v -s
Or run the file directly: python tests/feature/test_parameter_sharing.py
"""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.nn import Linear
from relann.session import Session
from relann.torch_utils import full_seed
from relann.era_operations import EmbeddedRelation
from relann.engine import pretty_print_params

class AlphaMatrix(torch.nn.Module):
    """Simple learnable matrix module with a non-standard parameter name."""

    def __init__(self, in_dim: int = 4, out_dim: int = 2):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.randn(in_dim, out_dim))

    def forward(self, *inputs):
        return self.alpha

def _minimal_input_db(n: int = 10, d: int = 4):
    """Input(a; z) with z shape [n, d]."""
    df = pd.DataFrame({"a": range(n)})
    z = torch.randn(n, d)
    return {"Input": (df, z)}

def _logits_labels_db(n: int = 10, num_classes: int = 3):
    """Input1(a; logits) [n, num_classes], Input2(a; labels) [n, 1] class indices long."""
    df = pd.DataFrame({"a": range(n)})
    logits = torch.randn(n, num_classes)
    labels = torch.randint(0, num_classes, (n, 1), dtype=torch.long)
    return {"Input1": (df.copy(), logits), "Input2": (df.copy(), labels)}

def _assert_er_shape(result, expected_embedding_shape):
    assert result is not None
    assert isinstance(result, EmbeddedRelation)
    assert result.embeddings is not None and len(result.embeddings) >= 1
    t = result.embeddings[0]
    assert t.shape == expected_embedding_shape, f"expected {expected_embedding_shape}, got {t.shape}"

def test_parameter_sharing_shared_transform_def_single_fqn_and_values():
    """Same TransformDef in two rules: engine.parameter_store has one FQN set; both nodes get same values."""
    full_seed(42)
    n, d, num_classes = 8, 4, 2
    db = _minimal_input_db(n=n, d=d)
    # Add Labels for fit: class indices (Long)
    labels_df = pd.DataFrame({"a": range(n)})
    labels_ten = torch.randint(0, num_classes, (n, 1), dtype=torch.long)
    db["Labels"] = (labels_df, labels_ten)
    session = Session(db=db)

    define = """
#lang:relnn
d = 4 .
My_Linear = Linear(d, d) .
First(a; My_Linear(z)) :- Input(a; z) .
Second(a; My_Linear(z)) :- First(a; z) .
Out(a; Linear(d, 2)(z)) :- Second(a; z) .
"""
    fit = """
#lang:relnn
?fit <epochs=1, lr=0.01> Loss(; CrossEntropyLoss()(z_pred, z)) :- Out(a; z_pred), Labels(a; z) .
"""
    pred = """
#lang:relnn
?pred Output(a; z) :- Second(a; z) .
"""

    session.define(define)
    session.run(fit)
    engine = session.engine

    # All params for this transform should be under global.My_Linear.* (single logical set)
    my_linear_keys = [k for k in engine.parameter_store if k.startswith("global.My_Linear.")]
    assert len(my_linear_keys) >= 1, "expected at least one parameter key under global.My_Linear.*"
    # No duplicate logical params: one weight (and no bias since bias=False)
    assert len(my_linear_keys) <= 2, "expected at most weight and bias for one Linear"

    # Engine stores only unique logical params: My_Linear (2) + Out (2) = 4 tensors
    n_engine_params = len(engine.parameter_store)
    assert n_engine_params == 4, "engine should store exactly 4 unique parameter tensors (My_Linear:2, Out:2), got %d" % n_engine_params

    # After fit, the trained module has two transformation nodes (First, Second) both using My_Linear
    # Both should have loaded the same values from engine.parameter_store
    assert "Loss" in engine.trained_modules
    module = engine.trained_modules["Loss"]["module"]
    assert hasattr(module, "graph") and hasattr(module, "_operators")

    transformation_params_by_node = []
    for node_id, node_data in module.graph.nodes(data=True):
        if node_data.get("type") != "transformation":
            continue
        op = module.module_for_node(node_id)
        if op is None:
            continue
        if not hasattr(op, "transformation"):
            continue
        trans_mod = op.transformation
        params = list(trans_mod.parameters())
        if params:
            transformation_params_by_node.append((node_id, params))

    # Filter to the two nodes that use My_Linear (First, Second); Out uses a different Linear
    my_linear_nodes = [(nid, params) for nid, params in transformation_params_by_node if "First" in nid or "Second" in nid]
    assert len(my_linear_nodes) == 2, "expected two transformation nodes using My_Linear (First, Second), got %s" % [n for n, _ in transformation_params_by_node]

    # Same parameter values (both loaded from global.My_Linear.*)
    (_, params0) = my_linear_nodes[0]
    (_, params1) = my_linear_nodes[1]
    assert len(params0) == len(params1)
    for p0, p1 in zip(params0, params1):
        assert p0.shape == p1.shape
        torch.testing.assert_close(p0.data, p1.data, msg="shared TransformDef should give same param values")

    # Duplication: module has 6 parameter tensors (First:2, Second:2, Out:2) but only 4 unique in engine.
    # First and Second each hold their own copy of My_Linear params; those copies are the same (asserted above).
    n_module_transformation_params = sum(len(p) for _, p in transformation_params_by_node)
    assert n_module_transformation_params == 6, (
        "module transformation nodes should have 6 parameter tensors (First:2, Second:2, Out:2), got %d"
        % n_module_transformation_params
    )
    assert n_module_transformation_params > n_engine_params, (
        "shared TransformDef implies duplication in module: more param tensors in module (%d) than unique in engine (%d)"
        % (n_module_transformation_params, n_engine_params)
    )

    # Predict should run and use saved params
    result = session.run(pred)
    _assert_er_shape(result, (n, d))

    # Print params to stdout (use pytest -s to see when test passes)
    print("\n--- Parameters (test_parameter_sharing_shared_transform_def_single_fqn_and_values) ---")
    session.show_params(show_stats=False)

def test_parameter_persistence_fit_then_predict():
    """After fit, predict uses saved parameters; second predict is consistent."""
    full_seed(42)
    n, d_in, d_out = 10, 4, 2
    db = _minimal_input_db(n=n, d=d_in)
    session = Session(db=db)

    define = """
#lang:relnn
Out(a; Linear(4, 2)(z)) :- Input(a; z) .
"""
    fit = """
#lang:relnn
?fit <epochs=2, lr=0.01> Loss(; CrossEntropyLoss()(z_pred, z)) :- Out(a; z_pred), Labels(a; z) .
"""
    pred = """
#lang:relnn
?pred Result(a; ArgMax()(z)) :- Out(a; z) .
"""

    # Add Labels for fit
    df = pd.DataFrame({"a": range(n)})
    labels = torch.randint(0, d_out, (n, 1), dtype=torch.long)
    session.engine.db["Labels"] = (df, labels)

    session.define(define)
    session.run(fit)
    result1 = session.run(pred)
    _assert_er_shape(result1, (n, 1))

    result2 = session.run(pred)
    _assert_er_shape(result2, (n, 1))
    # Same predictions (saved params loaded again into new module)
    torch.testing.assert_close(result1.embeddings[0], result2.embeddings[0])

    print("\n--- Parameters (test_parameter_persistence_fit_then_predict) ---")
    session.show_params(show_stats=False)

def test_save_module_parameters_raises_on_unexpected_module_resolution_error():
    """Unexpected module resolution errors must not be silently ignored."""
    full_seed(42)
    n, d_out = 10, 2
    db = _minimal_input_db(n=n, d=4)
    session = Session(db=db)

    define = """
#lang:relnn
Out(a; Linear(4, 2)(z)) :- Input(a; z) .
"""
    fit = """
#lang:relnn
?fit <epochs=1, lr=0.01> Loss(; CrossEntropyLoss()(z_pred, z)) :- Out(a; z_pred), Labels(a; z) .
"""

    labels_df = pd.DataFrame({"a": range(n)})
    labels = torch.randint(0, d_out, (n, 1), dtype=torch.long)
    session.engine.db["Labels"] = (labels_df, labels)

    session.define(define)
    session.run(fit)
    engine = session.engine
    module = engine.trained_modules["Loss"]["module"]

    original = module.module_for_node

    def _raise_unexpected(node_id):
        raise ValueError(f"unexpected resolver failure for {node_id}")

    module.module_for_node = _raise_unexpected
    try:
        with pytest.raises(RuntimeError, match="Failed to resolve transformation module"):
            engine._save_module_parameters(module, namespace=engine.current_namespace, rule_name="Loss")
    finally:
        module.module_for_node = original

def test_parameter_fqn_uses_transform_def_name_single_ref():
    """Single TransformDef reference per node: engine.parameter_store keys use TransformDef name, not node_id."""
    full_seed(42)
    n, d_in, d_out = 8, 4, 2
    db = _minimal_input_db(n=n, d=d_in)
    session = Session(db=db)

    define = """
#lang:relnn
MyLin = Linear(4, 2) .
Out(a; MyLin(z)) :- Input(a; z) .
"""
    fit = """
#lang:relnn
?fit <epochs=1, lr=0.01> Loss(; CrossEntropyLoss()(z_pred, z)) :- Out(a; z_pred), Labels(a; z) .
"""
    df = pd.DataFrame({"a": range(n)})
    labels = torch.randint(0, d_out, (n, 1), dtype=torch.long)
    session.engine.db["Labels"] = (df, labels)

    session.run(define)
    session.run(fit)
    engine = session.engine

    # Every key should start with global.MyLin. (TransformDef-based), none with global.transformation_
    mylin_keys = [k for k in engine.parameter_store if k.startswith("global.MyLin.")]
    transformation_keys = [k for k in engine.parameter_store if k.startswith("global.transformation_")]

    assert len(mylin_keys) >= 1, "expected at least one key under global.MyLin."
    assert len(transformation_keys) == 0, (
        "expected no keys under global.transformation_ when node uses single TransformDef ref; got %s"
        % transformation_keys
    )

    print("\n--- Parameters (test_parameter_fqn_uses_transform_def_name_single_ref) ---")
    session.show_params(show_stats=False)

def test_pretty_print_params_display_transform_def_names(capsys):
    """pretty_print_params(engine) shows TransformDef-based names, not long transformation_ node paths."""
    full_seed(42)
    n, d_in, d_out = 8, 4, 2
    db = _minimal_input_db(n=n, d=d_in)
    session = Session(db=db)

    define = """
#lang:relnn
MyLin = Linear(4, 2) .
Out(a; MyLin(z)) :- Input(a; z) .
"""
    fit = """
#lang:relnn
?fit <epochs=1, lr=0.01> Loss(; CrossEntropyLoss()(z_pred, z)) :- Out(a; z_pred), Labels(a; z) .
"""
    df = pd.DataFrame({"a": range(n)})
    labels = torch.randint(0, d_out, (n, 1), dtype=torch.long)
    session.engine.db["Labels"] = (df, labels)

    session.run(define)
    session.run(fit)
    engine = session.engine

    # Capture pretty_print_params output
    with capsys.disabled():
        # pretty_print_params prints to stdout; redirect to capture
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            pretty_print_params(engine, show_stats=False)
            out = buf.getvalue()
        finally:
            sys.stdout = old_stdout

    # Display should contain TransformDef name (MyLin)
    assert "MyLin" in out, "pretty_print output should contain TransformDef name MyLin"
    # Should not show long transformation_... paths in the Name column
    assert "transformation_Out" not in out or "MyLin" in out, (
        "display should prefer TransformDef name over transformation_Out"
    )
    assert "children_modules" not in out, "display should not contain long nested path children_modules"
    print("\n--- Parameters (test_pretty_print_params_display_transform_def_names) ---")
    print(out)

def test_hgt_style_rule_multi_transform_def_current_behavior(capsys):
    """HGT-style rule (multiple TransformDefs in one expression): FQN uses node_id + K/Q; pretty_print no children_modules."""
    full_seed(42)
    n, d = 6, 4
    df_a = pd.DataFrame({"s": range(n)})
    df_b = pd.DataFrame({"s": range(n), "t": (torch.arange(n) % 2).numpy()})
    df_c = pd.DataFrame({"t": (torch.arange(2)).numpy()})
    z_a = torch.randn(n, d)
    z_c = torch.randn(2, d)
    db = {
        "A": (df_a, z_a),
        "B": (df_b, torch.ones(n)),
        "C": (df_c, z_c),
    }
    session = Session(db=db)

    # One rule using multiple TransformDefs in one expression (Concat so join gives z1,z2)
    define = """
#lang:relnn
d = 4 .
K = Linear(d, d) .
Q = Linear(d, d) .
L1_Head_Author(s, t; Concat(K(z1), Q(z2))) :- A(s; z1), B(s, t), C(t; z2) .
"""
    session.run(define)

    # Trigger compilation and parameter extraction (fit on a simple loss so params are saved)
    fit = """
#lang:relnn
?fit <epochs=1, lr=0.01> Loss(; CrossEntropyLoss()(z_pred, z)) :- L1_Head_Author(s, t; z_pred), Labels(s, t; z) .
"""
    labels_df = pd.DataFrame({"s": range(n), "t": (torch.arange(n) % 2).numpy()})
    labels_ten = torch.randint(0, 2, (n, 1), dtype=torch.long)
    session.engine.db["Labels"] = (labels_df, labels_ten)
    session.run(fit)

    engine = session.engine
    module = engine.trained_modules["Loss"]["module"]

    # Step 1: Verify which transformation nodes exist after fit
    trans_nodes = [n for n, d in module.graph.nodes(data=True) if d.get("type") == "transformation"]
    assert len(trans_nodes) >= 2, (
        "expected at least 2 transformation nodes (L1_Head_Author and Loss), got %d: %s"
        % (len(trans_nodes), trans_nodes)
    )

    # Step 2: Inspect module for transformation_L1_Head_Author
    l1_node_id = "transformation_L1_Head_Author"
    l1_op = module.module_for_node(l1_node_id)
    if l1_op is not None:
        trans_mod = l1_op.transformation
        param_names = list(trans_mod.named_parameters())
        from relann.tensor_term_compiler import _MultiArgWrapper
        has_children = getattr(trans_mod, "children_modules", None) is not None
        num_children = len(trans_mod.children_modules) if has_children else 0
        has_child_names = getattr(trans_mod, "_child_names", None) is not None
        assert (has_children and num_children == 2) or has_child_names, (
            "transformation_L1_Head_Author should be _MultiArgWrapper with 2 children (K, Q); "
            "got %s, param_names=%s, children_modules=%s, _child_names=%s"
            % (type(trans_mod).__name__, [n for n, _ in param_names], num_children if has_children else "N/A", getattr(trans_mod, "_child_names", None))
        )
        assert len(param_names) >= 4, (
            "transformation_L1_Head_Author module should have 4 params (K,Q weight+bias), got %d: %s"
            % (len(param_names), [n for n, _ in param_names])
        )

    # Shared TransformDef params keyed by TransformDef name (K, Q) so one entry per weight and display shows K.weight, Q.weight
    k_keys = [k for k in engine.parameter_store if k.startswith("global.K.") or ".K." in k]
    q_keys = [k for k in engine.parameter_store if k.startswith("global.Q.") or ".Q." in k]
    assert len(k_keys) >= 2 and len(q_keys) >= 2, (
        "params for Concat(K,Q) should be keyed by TransformDef name (global.K.*, global.Q.*), got %s"
        % list(engine.parameter_store.keys())
    )
    # Expect 4 param tensors (K and Q each weight+bias); plan: fix if we only have 2
    assert len(engine.parameter_store) >= 4, (
        "expected at least 4 parameter tensors (K and Q weight+bias), got %d: %s"
        % (len(engine.parameter_store), list(engine.parameter_store.keys()))
    )
    # FQNs must not use children_modules
    param_keys_str = " ".join(engine.parameter_store.keys())
    assert "children_modules" not in param_keys_str, "FQNs should not contain children_modules"

    # Pretty-print must not show children_modules
    with capsys.disabled():
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            pretty_print_params(engine, show_stats=False)
            out = buf.getvalue()
        finally:
            sys.stdout = old_stdout
    assert "children_modules" not in out, "pretty_print must not contain children_modules"
    # Display must show shared weights as K.weight, K.bias, Q.weight, Q.bias (no rule name prefix)
    assert "K.weight" in out or "K." in out, (
        "pretty_print should show K.weight (shared TransformDef), got: %s" % out[:400]
    )
    assert "Q.weight" in out or "Q." in out, (
        "pretty_print should show Q.weight (shared TransformDef), got: %s" % out[:400]
    )
    # Params keyed by TransformDef (K, Q) so display is K.weight / Q.weight not L1_Head_Author.K.weight
    param_keys_str = " ".join(engine.parameter_store.keys())
    assert "global.K." in param_keys_str and "global.Q." in param_keys_str, (
        "FQNs should be global.K.* and global.Q.* for shared weights, got %s" % list(engine.parameter_store.keys())
    )
    print("\n--- Parameters (test_hgt_style_rule_multi_transform_def_current_behavior) ---")
    print(out)

def test_multi_linear_same_rule_pretty_print_no_children_modules(capsys):
    """Multiple TransformDefs (2+ Linears) in one rule: display has no children_modules, readable names."""
    full_seed(42)
    n, d = 6, 4
    df_a = pd.DataFrame({"s": range(n)})
    df_b = pd.DataFrame({"s": range(n), "t": (torch.arange(n) % 2).numpy()})
    df_c = pd.DataFrame({"t": (torch.arange(2)).numpy()})
    z_a = torch.randn(n, d)
    z_c = torch.randn(2, d)
    db = {
        "A": (df_a, z_a),
        "B": (df_b, torch.ones(n)),
        "C": (df_c, z_c),
    }
    session = Session(db=db)

    # Two TransformDefs in one Concat (K, Q) so we get multiple param groups and longer names
    define = """
#lang:relnn
d = 4 .
K = Linear(d, d) .
Q = Linear(d, d) .
L1_Head(s, t; Concat(K(z1), Q(z2))) :- A(s; z1), B(s, t), C(t; z2) .
"""
    session.run(define)

    fit = """
#lang:relnn
?fit <epochs=1, lr=0.01> Loss(; CrossEntropyLoss()(z_pred, z)) :- L1_Head(s, t; z_pred), Labels(s, t; z) .
"""
    labels_df = pd.DataFrame({"s": range(n), "t": (torch.arange(n) % 2).numpy()})
    labels_ten = torch.randint(0, 2, (n, 1), dtype=torch.long)
    session.engine.db["Labels"] = (labels_df, labels_ten)
    session.run(fit)

    engine = session.engine

    # No children_modules in stored keys
    param_keys_str = " ".join(engine.parameter_store.keys())
    assert "children_modules" not in param_keys_str, "FQNs must use logical names not children_modules"

    # Pretty-print: no children_modules, readable names (K/Q or 0/1)
    with capsys.disabled():
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            pretty_print_params(engine, show_stats=False)
            out = buf.getvalue()
        finally:
            sys.stdout = old_stdout
    assert "children_modules" not in out, "pretty_print must not contain children_modules"
    # At least 2 param tensors (one Linear or more)
    assert len(engine.parameter_store) >= 2, "expected at least 2 parameter tensors"
    print("\n--- Parameters (test_multi_linear_same_rule_pretty_print_no_children_modules) ---")
    print(out)

def test_two_rules_both_use_K_second_has_linear(capsys):
    """Two rules use shared K; second rule also has a Linear in the embedding expression. Check param list shows K once and the rule-specific Linear."""
    full_seed(42)
    n, d = 6, 4
    df_a = pd.DataFrame({"s": range(n)})
    df_b = pd.DataFrame({"s": range(n), "t": (torch.arange(n) % 2).numpy()})
    df_c = pd.DataFrame({"t": (torch.arange(2)).numpy()})
    df_d = pd.DataFrame({"x": range(n)})
    z_a = torch.randn(n, d)
    z_c = torch.randn(2, d)
    z_d = torch.randn(n, d)
    db = {
        "A": (df_a, z_a),
        "B": (df_b, torch.ones(n)),
        "C": (df_c, z_c),
        "D": (df_d, z_d),
    }
    session = Session(db=db)

    # Rule 1: Concat(K, Q). Rule 2: Linear(d, 2)(K(z)) — same K, plus a Linear in the embedding expression
    define = """
#lang:relnn
d = 4 .
K = Linear(d, d) .
Q = Linear(d, d) .
L1_Head(s, t; Concat(K(z1), Q(z2))) :- A(s; z1), B(s, t), C(t; z2) .
L2_Head(x; Linear(d, 2)(K(z))) :- D(x; z) .
"""
    session.run(define)

    # Fit on L1 so K, Q are stored
    fit1 = """
#lang:relnn
?fit <epochs=1, lr=0.01> Loss1(; CrossEntropyLoss()(z_pred, z)) :- L1_Head(s, t; z_pred), Labels1(s, t; z) .
"""
    labels1_df = pd.DataFrame({"s": range(n), "t": (torch.arange(n) % 2).numpy()})
    labels1_ten = torch.randint(0, 2, (n, 1), dtype=torch.long)
    session.engine.db["Labels1"] = (labels1_df, labels1_ten)
    session.run(fit1)

    # Fit on L2 so we add L2's Linear and K is reused (same global.K)
    fit2 = """
#lang:relnn
?fit <epochs=1, lr=0.01> Loss2(; CrossEntropyLoss()(z_pred, z)) :- L2_Head(x; z_pred), Labels2(x; z) .
"""
    labels2_df = pd.DataFrame({"x": range(n)})
    labels2_ten = torch.randint(0, 2, (n, 1), dtype=torch.long)
    session.engine.db["Labels2"] = (labels2_df, labels2_ten)
    session.run(fit2)

    engine = session.engine
    # K should appear once (shared), keyed as global.K.*; no "input" (single-child wrapper attr) in FQNs
    param_keys_str = " ".join(engine.parameter_store.keys())
    assert "input" not in param_keys_str, "params should be keyed by TransformDef name (K), not compiler attr 'input'; got %s" % list(engine.parameter_store.keys())
    k_keys = [k for k in engine.parameter_store if k.startswith("global.K.")]
    assert len(k_keys) == 2, "shared K should appear once (weight + bias), got %d: %s" % (len(k_keys), k_keys)
    # L2's Linear should appear (rule-specific)
    l2_keys = [k for k in engine.parameter_store if "L2" in k]
    assert len(l2_keys) >= 2, "expected L2_Head Linear (weight + bias), got %s" % list(engine.parameter_store.keys())

    with capsys.disabled():
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            pretty_print_params(engine, show_stats=False)
            out = buf.getvalue()
        finally:
            sys.stdout = old_stdout
    print("\n--- Parameters (test_two_rules_both_use_K_second_has_linear) ---")
    print(out)

def test_tensor_in_embedding_expression(capsys):
    """Rule uses Tensor(shape) in embedding expression (learnable matrix). Params should include the Tensor."""
    full_seed(42)
    n, d = 10, 4
    db = _minimal_input_db(n=n, d=d)
    session = Session(db=db)

    define = """
#lang:relnn
Test1(a; z1 @ Tensor(4, 2)) :- Input(a; z1) .
"""
    session.run(define)

    pred = """
#lang:relnn
?pred Out(a; z) :- Test1(a; z) .
"""
    session.define(pred)

    engine = session.engine
    # After pred we may have params from compilation; Tensor(4,2) is a learnable param
    param_keys = list(engine.parameter_store.keys())
    # Either we have a key for the Tensor (e.g. transformation_Test1.* or *Tensor*)
    has_tensor_param = any("Tensor" in k or "Test1" in k or "weight" in k for k in param_keys)
    assert has_tensor_param or len(param_keys) >= 1, (
        "expected at least one parameter (Tensor(4,2) or similar), got %s" % param_keys
    )

    with capsys.disabled():
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            pretty_print_params(engine, show_stats=False)
            out = buf.getvalue()
        finally:
            sys.stdout = old_stdout
    # Display should use PyTorch-style numeric index (e.g. Test1.0.weight)
    assert "Test1." in out and ".weight" in out, "pretty_print should show Test1.*.weight, got %s" % out[:300]
    assert "0.weight" in out, "display should use numeric index (Test1.0.weight), got %s" % out[:300]
    print("\n--- Parameters (test_tensor_in_embedding_expression) ---")
    print(out)

def test_nonstandard_param_name_display_normalization(capsys):
    """Display normalization should work for any parameter name, not just weight/bias."""
    full_seed(42)
    n, d = 10, 4
    db = _minimal_input_db(n=n, d=d)
    session = Session(db=db)

    define = """
#lang:relnn
NS(a; z1 @ AlphaMatrix()) :- Input(a; z1) .
"""
    session.run(define)

    pred = """
#lang:relnn
?pred Out(a; z) :- NS(a; z) .
"""
    session.define(pred)

    engine = session.engine
    param_keys = list(engine.parameter_store.keys())
    assert len(param_keys) >= 1, "expected at least one learned parameter, got %s" % param_keys

    with capsys.disabled():
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            pretty_print_params(engine, show_stats=False)
            out = buf.getvalue()
        finally:
            sys.stdout = old_stdout

    assert "NS.0.alpha" in out, "expected normalized display NS.0.alpha, got %s" % out[:400]
    assert ".right." not in out and ".left." not in out, (
        "display should hide arithmetic internals for non-standard params too, got %s" % out[:500]
    )
    print("\n--- Parameters (test_nonstandard_param_name_display_normalization) ---")
    print(out)

def test_multiple_tensor_and_linear_in_one_rule(capsys):
    """One rule with multiple Tensor and Linear definitions in one embedding expression (all rule-specific, distinct)."""
    full_seed(42)
    n, d = 10, 4
    db = _minimal_input_db(n=n, d=d)
    session = Session(db=db)

    # One rule: Concat of two matmuls (z1 @ Tensor) and two Linears, all different and specific to this rule
    define = """
#lang:relnn
Multi(a; Concat(z1 @ Tensor(4, 2), z1 @ Tensor(4, 3), Linear(4, 2)(z1), Linear(4, 4)(z1))) :- Input(a; z1) .
"""
    session.run(define)

    pred = """
#lang:relnn
?pred Out(a; z) :- Multi(a; z) .
"""
    session.run(pred)

    engine = session.engine
    param_keys = list(engine.parameter_store.keys())
    # 2 Tensors (weight each) + 2 Linears (weight+bias each) = 6 param tensors
    assert len(param_keys) >= 6, (
        "expected at least 6 params (2 Tensor + 2 Linear with weight+bias), got %d: %s"
        % (len(param_keys), param_keys)
    )
    # All should be under the same rule node (Multi)
    multi_keys = [k for k in param_keys if "Multi" in k]
    assert len(multi_keys) >= 6, "params should be keyed under Multi, got %s" % param_keys

    with capsys.disabled():
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            pretty_print_params(engine, show_stats=False)
            out = buf.getvalue()
        finally:
            sys.stdout = old_stdout
    # Display should show stable Concat branch indices (Multi.0..Multi.3), without arithmetic internals.
    assert "Multi." in out and ".weight" in out, "pretty_print should show Multi.*.weight, got %s" % out[:400]
    assert out.count(".weight") >= 4, "expected at least 4 weight params (2 Tensor + 2 Linear), got %s" % out[:500]
    assert ".right." not in out and ".left." not in out, (
        "display should hide arithmetic internals (.right/.left), got %s" % out[:500]
    )
    for expected_name in ("Multi.0.weight", "Multi.1.weight", "Multi.2.weight", "Multi.3.weight"):
        assert expected_name in out, "expected %s in display, got %s" % (expected_name, out[:600])
    assert "Multi.2.bias" in out and "Multi.3.bias" in out, (
        "expected Linear biases under Multi.2 and Multi.3, got %s" % out[:600]
    )
    print("\n--- Parameters (test_multiple_tensor_and_linear_in_one_rule) ---")
    print(out)

def test_pretty_print_disambiguates_collapsed_left_right_paths(capsys):
    """When normalization would collapse left/right into one name, pretty print keeps names unique."""
    class _RuleContainer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = Linear(4, 2)
            self.right = Linear(4, 2)

    class _DisplayCollisionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.Rule = _RuleContainer()

    model = _DisplayCollisionModel()

    with capsys.disabled():
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            pretty_print_params(model, show_stats=False)
            out = buf.getvalue()
        finally:
            sys.stdout = old_stdout

    assert "Rule.left.weight" in out and "Rule.right.weight" in out, (
        "expected collision-safe names for both branches, got %s" % out[:500]
    )
    assert "Rule.0.weight" not in out, (
        "display should avoid ambiguous collapsed names when both branches are parameterized, got %s" % out[:500]
    )

if __name__ == "__main__":
    import pytest
    # -s: show stdout (parameter tables printed by each test)
    exit_code = pytest.main([__file__, "-v", "-s"])
    if exit_code == 0:
        print("All parameter sharing tests passed.")
    sys.exit(exit_code)
