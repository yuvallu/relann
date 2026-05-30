"""
E2E tests: one rule per op on minimal Input relation(s).
Each test defines one rule, runs predict, and asserts result shape so we verify the op really runs.

Ops covered here: Linear, ReLU, transpose, view, sqrt, ArgMax, Concat, * (mul), + (add), @ (matmul),
CrossEntropyLoss (fit), and ArgMax+CrossEntropy fit+pred pipeline.

About @ (matmul): Use Tensor(shape) for a learnable matrix, e.g. ``z @ Tensor(4, 2)`` gives
(N, 4) @ (4, 2) -> (N, 2). Tensor(shape) is a built-in that compiles to a module with a single
nn.Parameter of the given shape.

Full fit/loss (CrossEntropyLoss, etc.) on Cora is exercised in test_e2e_cora_gcn.py.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from relann.session import Session
from relann.torch_utils import full_seed
from relann.era_operations import EmbeddedRelation

def _minimal_input_db(n: int = 10, d: int = 4):
    """Input(a; z) with z shape [n, d]."""
    df = pd.DataFrame({"a": range(n)})
    z = torch.randn(n, d)
    return {"Input": (df, z)}

def _two_input_db(n: int = 10, d1: int = 2, d2: int = 3):
    """Input1(a; z1) and Input2(a; z2) with same key column a; z1 [n, d1], z2 [n, d2]."""
    df = pd.DataFrame({"a": range(n)})
    z1 = torch.randn(n, d1)
    z2 = torch.randn(n, d2)
    return {"Input1": (df.copy(), z1), "Input2": (df.copy(), z2)}

def _assert_er_shape(result, expected_embedding_shape):
    assert result is not None
    assert isinstance(result, EmbeddedRelation)
    assert result.embeddings is not None and len(result.embeddings) >= 1
    t = result.embeddings[0]
    assert t.shape == expected_embedding_shape, f"expected {expected_embedding_shape}, got {t.shape}"

def test_e2e_one_rule_linear():
    full_seed(42)
    db = _minimal_input_db(n=10, d=4)
    session = Session(db=db)
    define = """
#lang:relnn
Test1(a; Linear(4, 2)(z1)) :- Input(a; z1) .
"""
    pred = """
#lang:relnn
?pred Out(a; ArgMax()(z)) :- Test1(a; z) .
"""
    session.run(define)
    result = session.run(pred)
    # ArgMax returns (N, 1) for row-first consistency: one scalar per row
    _assert_er_shape(result, (10, 1))

def test_e2e_one_rule_relu():
    full_seed(42)
    db = _minimal_input_db(n=10, d=4)
    session = Session(db=db)
    define = """
#lang:relnn
Test1(a; ReLU(z1)) :- Input(a; z1) .
"""
    pred = """
#lang:relnn
?pred Out(a; z) :- Test1(a; z) .
"""
    session.run(define)
    result = session.run(pred)
    _assert_er_shape(result, (10, 4))

def test_e2e_one_rule_gelu():
    full_seed(42)
    db = _minimal_input_db(n=10, d=4)
    session = Session(db=db)
    define = """
#lang:relnn
Test1(a; GELU(z1)) :- Input(a; z1) .
"""
    pred = """
#lang:relnn
?pred Out(a; z) :- Test1(a; z) .
"""
    session.run(define)
    result = session.run(pred)
    _assert_er_shape(result, (10, 4))

def test_e2e_one_rule_sigmoid():
    full_seed(42)
    db = _minimal_input_db(n=10, d=4)
    session = Session(db=db)
    define = """
#lang:relnn
Test1(a; Sigmoid(z1)) :- Input(a; z1) .
"""
    pred = """
#lang:relnn
?pred Out(a; z) :- Test1(a; z) .
"""
    session.run(define)
    result = session.run(pred)
    _assert_er_shape(result, (10, 4))

def test_e2e_one_rule_transpose():
    full_seed(42)
    db = _minimal_input_db(n=10, d=4)
    session = Session(db=db)
    define = """
#lang:relnn
Test1(a; transpose(z1)) :- Input(a; z1) .
"""
    pred = """
#lang:relnn
?pred Out(a; z) :- Test1(a; z) .
"""
    session.run(define)
    result = session.run(pred)
    # transpose (E,d) -> (E,d,1)
    _assert_er_shape(result, (10, 4, 1))

def test_e2e_one_rule_view():
    full_seed(42)
    db = _minimal_input_db(n=10, d=4)
    session = Session(db=db)
    define = """
#lang:relnn
Test1(a; view(2, 2)(z1)) :- Input(a; z1) .
"""
    pred = """
#lang:relnn
?pred Out(a; z) :- Test1(a; z) .
"""
    session.run(define)
    result = session.run(pred)
    _assert_er_shape(result, (10, 2, 2))

def test_e2e_one_rule_unsqueeze_function_style():
    full_seed(42)
    db = _minimal_input_db(n=10, d=4)
    session = Session(db=db)
    define = """
#lang:relnn
Test1(a; unsqueeze(z1, 1)) :- Input(a; z1) .
"""
    pred = """
#lang:relnn
?pred Out(a; z) :- Test1(a; z) .
"""
    session.run(define)
    result = session.run(pred)
    _assert_er_shape(result, (10, 1, 4))

def test_e2e_one_rule_sqrt():
    full_seed(42)
    db = _minimal_input_db(n=10, d=4)
    # Use positive values so sqrt is well-defined
    df = pd.DataFrame({"a": range(10)})
    z = torch.rand(10, 4)
    db = {"Input": (df, z)}
    session = Session(db=db)
    define = """
#lang:relnn
Test1(a; sqrt(z1)) :- Input(a; z1) .
"""
    pred = """
#lang:relnn
?pred Out(a; z) :- Test1(a; z) .
"""
    session.run(define)
    result = session.run(pred)
    _assert_er_shape(result, (10, 4))

def test_e2e_one_rule_concat():
    full_seed(42)
    db = _two_input_db(n=10, d1=2, d2=3)
    session = Session(db=db)
    define = """
#lang:relnn
Test1(a; Concat(z1, z2)) :- Input1(a; z1), Input2(a; z2) .
"""
    pred = """
#lang:relnn
?pred Out(a; z) :- Test1(a; z) .
"""
    session.run(define)
    result = session.run(pred)
    _assert_er_shape(result, (10, 5))

def test_e2e_one_rule_mul():
    full_seed(42)
    db = _two_input_db(n=10, d1=4, d2=4)
    session = Session(db=db)
    define = """
#lang:relnn
Test1(a; z1 * z2) :- Input1(a; z1), Input2(a; z2) .
"""
    pred = """
#lang:relnn
?pred Out(a; z) :- Test1(a; z) .
"""
    session.run(define)
    result = session.run(pred)
    _assert_er_shape(result, (10, 4))

def test_e2e_one_rule_add():
    full_seed(42)
    db = _two_input_db(n=10, d1=4, d2=4)
    session = Session(db=db)
    define = """
#lang:relnn
Test1(a; z1 + z2) :- Input1(a; z1), Input2(a; z2) .
"""
    pred = """
#lang:relnn
?pred Out(a; z) :- Test1(a; z) .
"""
    session.run(define)
    result = session.run(pred)
    _assert_er_shape(result, (10, 4))

# Note: Consider shortening tests like this: Test with define+pred in one go (pred-only); same as matmul but single run.
def test_e2e_one_rule_matmul():
    # Explicit @ in the DSL: z @ Tensor(shape) with Tensor a learnable parameter matrix.
    # Tensor(4, 2) has no args and compiles to _ParameterTensor(4, 2); forward returns weight (4, 2).
    # (10, 4) @ (4, 2) -> (10, 2).
    full_seed(42)
    db = _minimal_input_db(n=10, d=4)
    session = Session(db=db)
    define = """
#lang:relnn
Test1(a; z1 @ Tensor(4, 2)) :- Input(a; z1) .
"""
    pred = """
#lang:relnn
?pred Out(a; z) :- Test1(a; z) .
"""
    session.run(define)
    result = session.run(pred)
    _assert_er_shape(result, (10, 2))

def test_e2e_one_rule_argmax_only():
    full_seed(42)
    db = _minimal_input_db(n=10, d=4)
    session = Session(db=db)
    define = """
#lang:relnn
Test1(a; ArgMax()(z1)) :- Input(a; z1) .
"""
    pred = """
#lang:relnn
?pred Out(a; z) :- Test1(a; z) .
"""
    session.run(define)
    result = session.run(pred)
    _assert_er_shape(result, (10, 1))

def _logits_labels_db(n: int = 10, num_classes: int = 3):
    """Input1(a; logits) [n, num_classes], Input2(a; labels) [n, 1] class indices long."""
    df = pd.DataFrame({"a": range(n)})
    logits = torch.randn(n, num_classes)
    # Class indices in [0, num_classes-1]
    labels = torch.randint(0, num_classes, (n, 1), dtype=torch.long)
    return {"Input1": (df.copy(), logits), "Input2": (df.copy(), labels)}

def test_e2e_crossentropy():
    """E2E: Loss(; CrossEntropyLoss()(logits, labels)) with minimal Logits (Linear) and Labels from Input1/Input2.
    Uses a small Linear so the fit has parameters to optimize (engine requires non-empty params)."""
    full_seed(42)
    n, num_classes = 10, 3
    db = _logits_labels_db(n=n, num_classes=num_classes)
    session = Session(db=db)
    define = """
#lang:relnn
Logits(a; Linear(3, 3)(z1)) :- Input1(a; z1) .
Labels(a; z2) :- Input2(a; z2) .
"""
    fit = """
#lang:relnn
?fit <epochs=1, lr=0.01> Loss(; CrossEntropyLoss()(z_pred, z)) :- Logits(a; z_pred), Labels(a; z) .
"""
    session.run(define)
    session.run(fit)
    assert "Loss" in session.engine.trained_modules
    info = session.engine.trained_modules["Loss"]
    assert "loss_history" in info and len(info["loss_history"]) == 1
    loss_val = info["loss_history"][0]
    assert isinstance(loss_val, (int, float)), f"expected scalar loss, got {type(loss_val)}"
    assert torch.isfinite(torch.tensor(loss_val, dtype=torch.float64)).item(), f"loss must be finite, got {loss_val}"

def test_e2e_argmax_after_crossentropy_fit():
    """E2E: fit Loss with CrossEntropyLoss, then pred with ArgMax; asserts pred shape (N, 1).
    Logits use a small Linear so fit has parameters; ArgMax is applied to those logits."""
    full_seed(42)
    n, num_classes = 10, 3
    db = _logits_labels_db(n=n, num_classes=num_classes)
    session = Session(db=db)
    define = """
#lang:relnn
Logits(a; Linear(3, 3)(z1)) :- Input1(a; z1) .
Labels(a; z2) :- Input2(a; z2) .
"""
    fit = """
#lang:relnn
?fit <epochs=1, lr=0.01> Loss(; CrossEntropyLoss()(z_pred, z)) :- Logits(a; z_pred), Labels(a; z) .
"""
    pred = """
#lang:relnn
?pred Out(a; ArgMax()(z)) :- Logits(a; z) .
"""
    session.run(define)
    session.run(fit)
    result = session.run(pred)
    _assert_er_shape(result, (n, 1))

if __name__ == "__main__":
    test_e2e_one_rule_linear()
    test_e2e_one_rule_relu()
    test_e2e_one_rule_gelu()
    test_e2e_one_rule_sigmoid()
    test_e2e_one_rule_transpose()
    test_e2e_one_rule_view()
    test_e2e_one_rule_unsqueeze_function_style()
    test_e2e_one_rule_sqrt()
    test_e2e_one_rule_concat()
    test_e2e_one_rule_mul()
    test_e2e_one_rule_add()
    test_e2e_one_rule_matmul()
    test_e2e_one_rule_argmax_only()
    test_e2e_crossentropy()
    test_e2e_argmax_after_crossentropy_fit()
    print("All e2e single-op tests passed.")
