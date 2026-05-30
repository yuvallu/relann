"""
Unit tests for TensorTermCompiler: resolution and compile per op.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from relann.engine import Engine
from relann.tensor_term_compiler import resolve_op, TensorTermCompiler, ArgMax, Concat, Tensor
from relann.pydantic_classes import TensorTerm, TensorOp, Var, ArithTerm, TransformDef

# ---- Resolution ----

def test_resolve_from_run_globals():
    engine = Engine(db={})
    engine.set_run_globals({"MyLinear": nn.Linear})
    got = resolve_op("MyLinear", engine.get_run_globals())
    assert got is nn.Linear

def test_resolve_from_torch_nn():
    got_linear = resolve_op("Linear", {})
    assert got_linear is nn.Linear
    got_relu = resolve_op("ReLU", {})
    assert got_relu is nn.ReLU

def test_resolve_dsl_native():
    assert resolve_op("transpose", {}) is not None
    assert resolve_op("view", {}) is not None
    assert resolve_op("sqrt", {}) is not None
    # ArgMax/argmax resolve to built-in (tensor_term_compiler.ArgMax)
    assert resolve_op("argmax", {}) is ArgMax
    assert resolve_op("ArgMax", {}) is ArgMax

def test_resolve_not_found():
    assert resolve_op("NonExistentOp", {}) is None

# ---- Compile: leaves and arithmetic ----

def test_compile_linear():
    engine = Engine(db={})
    engine.set_run_globals({"Linear": nn.Linear})
    tterm = TensorTerm(
        op=TensorOp(op="Linear", hyper_params=[ArithTerm(value=4), ArithTerm(value=8)]),
        sons=[TensorTerm(value=Var(name="z"))],
    )
    m = engine.tensor_term_to_module(tterm, {"z": 0})
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(5, 4))
    assert out.shape == (5, 8)

def test_compile_relu():
    engine = Engine(db={})
    engine.set_run_globals({"ReLU": nn.ReLU})
    tterm = TensorTerm(op=TensorOp(op="ReLU"), sons=[TensorTerm(value=Var(name="z"))])
    m = engine.tensor_term_to_module(tterm, {"z": 0})
    m.eval()
    with torch.no_grad():
        x = torch.randn(3, 10)
        out = m(x)
    assert out.shape == x.shape
    assert (out >= 0).all()

def test_compile_concat():
    engine = Engine(db={})
    # Concat is a built-in (tensor_term_compiler); no need to set run_globals
    tterm = TensorTerm(
        op=TensorOp(op="Concat"),
        sons=[TensorTerm(value=Var(name="z1")), TensorTerm(value=Var(name="z2"))],
    )
    m = engine.tensor_term_to_module(tterm, {"z1": 0, "z2": 1})
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(5, 2), torch.randn(5, 3))
    assert out.shape == (5, 5)

def test_compile_view():
    engine = Engine(db={})
    tterm = TensorTerm(
        op=TensorOp(op="view", hyper_params=[ArithTerm(value=2), ArithTerm(value=3)]),
        sons=[TensorTerm(value=Var(name="z"))],
    )
    m = engine.tensor_term_to_module(tterm, {"z": 0})
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(4, 6))
    assert out.shape == (4, 2, 3)

def test_compile_sqrt():
    engine = Engine(db={})
    tterm = TensorTerm(op=TensorOp(op="sqrt"), sons=[TensorTerm(value=Var(name="z"))])
    m = engine.tensor_term_to_module(tterm, {"z": 0})
    m.eval()
    with torch.no_grad():
        x = torch.rand(5, 3) + 0.1
        out = m(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, torch.sqrt(x))

def test_compile_transpose():
    engine = Engine(db={})
    tterm = TensorTerm(op=TensorOp(op="transpose"), sons=[TensorTerm(value=Var(name="z1"))])
    m = engine.tensor_term_to_module(tterm, {"z1": 0})
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(100, 2))
    assert out.shape == (100, 2, 1)  # transpose (E,d) -> (E,d,1)

def test_compile_transpose_capital_t():
    engine = Engine(db={})
    tterm = TensorTerm(op=TensorOp(op="Transpose"), sons=[TensorTerm(value=Var(name="z1"))])
    m = engine.tensor_term_to_module(tterm, {"z1": 0})
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(50, 4))
    assert out.shape == (50, 4, 1)  # transpose (E,d) -> (E,d,1)

def test_compile_unsqueeze_function_style():
    engine = Engine(db={})
    tterm = TensorTerm(
        op=TensorOp(op="unsqueeze"),
        sons=[TensorTerm(value=Var(name="z1")), TensorTerm(value=1)],
    )
    m = engine.tensor_term_to_module(tterm, {"z1": 0})
    m.eval()
    with torch.no_grad():
        x = torch.randn(7, 4)
        out = m(x)
    assert out.shape == (7, 1, 4)
    torch.testing.assert_close(out, torch.unsqueeze(x, 1))

def test_compile_unsqueeze_negative_dim():
    engine = Engine(db={})
    tterm = TensorTerm(
        op=TensorOp(op="unsqueeze"),
        sons=[TensorTerm(value=Var(name="z1")), TensorTerm(value=-1)],
    )
    m = engine.tensor_term_to_module(tterm, {"z1": 0})
    m.eval()
    with torch.no_grad():
        x = torch.randn(7, 4)
        out = m(x)
    assert out.shape == (7, 4, 1)
    torch.testing.assert_close(out, torch.unsqueeze(x, -1))

def test_compile_callable_factory_from_run_globals_still_works():
    engine = Engine(db={})

    def MakeReLU():
        return nn.ReLU()

    engine.set_run_globals({"MakeReLU": MakeReLU})
    tterm = TensorTerm(op=TensorOp(op="MakeReLU"), sons=[TensorTerm(value=Var(name="z"))])
    m = engine.tensor_term_to_module(tterm, {"z": 0})
    m.eval()
    with torch.no_grad():
        x = torch.randn(5, 3)
        out = m(x)
    assert out.shape == x.shape
    assert (out >= 0).all()

def test_compile_varargs_callable_from_run_globals_uses_function_style():
    engine = Engine(db={})

    def FirstArg(*xs):
        return xs[0]

    engine.set_run_globals({"FirstArg": FirstArg})
    tterm = TensorTerm(op=TensorOp(op="FirstArg"), sons=[TensorTerm(value=Var(name="z"))])
    m = engine.tensor_term_to_module(tterm, {"z": 0})
    m.eval()
    with torch.no_grad():
        x = torch.randn(5, 3)
        out = m(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, x)

def test_compile_arithmetic():
    engine = Engine(db={})
    tterm = TensorTerm(
        op=TensorOp(op="*"),
        sons=[TensorTerm(value=Var(name="z1")), TensorTerm(value=Var(name="z2"))],
    )
    m = engine.tensor_term_to_module(tterm, {"z1": 0, "z2": 1})
    m.eval()
    with torch.no_grad():
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        out = m(a, b)
    assert out.shape == (2, 3)
    torch.testing.assert_close(out, a * b)

def test_compile_mseloss():
    engine = Engine(db={})
    engine.set_run_globals({"MSELoss": nn.MSELoss})
    tterm = TensorTerm(
        op=TensorOp(op="MSELoss"),
        sons=[TensorTerm(value=Var(name="z1")), TensorTerm(value=Var(name="z2"))],
    )
    m = engine.tensor_term_to_module(tterm, {"z1": 0, "z2": 1})
    m.eval()
    with torch.no_grad():
        pred = torch.randn(4, 2)
        target = torch.randn(4, 2)
        out = m(pred, target)
    # Per-row: reduction='none' returns per-element loss; Aggregation handles batch reduction.
    assert out.shape == (4, 2)
    assert out.dtype == torch.float32

def test_compile_crossentropy():
    engine = Engine(db={})
    engine.set_run_globals({"CrossEntropyLoss": nn.CrossEntropyLoss})
    tterm = TensorTerm(
        op=TensorOp(op="CrossEntropyLoss"),
        sons=[TensorTerm(value=Var(name="z1")), TensorTerm(value=Var(name="z2"))],
    )
    m = engine.tensor_term_to_module(tterm, {"z1": 0, "z2": 1})
    m.eval()
    with torch.no_grad():
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        out = m(logits, targets)
    # Per-row: reduction='none' returns per-sample loss (N,); Aggregation handles batch reduction.
    assert out.shape == (4,)
    assert out.dtype == torch.float32

def test_compile_argmax():
    engine = Engine(db={})
    # ArgMax/argmax resolved as built-in from tensor_term_compiler (same path as ReLU)
    tterm = TensorTerm(op=TensorOp(op="ArgMax"), sons=[TensorTerm(value=Var(name="z"))])
    m = engine.tensor_term_to_module(tterm, {"z": 0})
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(6, 5))
    assert out.shape == (6, 1)  # row-first: one scalar per row
    assert out.dtype == torch.int64

def test_compile_argmax_lowercase():
    """argmax()(z) and ArgMax()(z) both resolve to built-in ArgMax and compile via _SingleChildWrapper."""
    engine = Engine(db={})
    tterm = TensorTerm(op=TensorOp(op="argmax"), sons=[TensorTerm(value=Var(name="z"))])
    m = engine.tensor_term_to_module(tterm, {"z": 0})
    m.eval()
    x = torch.randn(6, 5)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (6, 1)
    assert out.dtype == torch.int64
    assert out.equal(torch.argmax(x, dim=1, keepdim=True))

def test_compile_tensor():
    engine = Engine(db={})
    tterm = TensorTerm(
        op=TensorOp(op="Tensor", hyper_params=[ArithTerm(value=4), ArithTerm(value=2)]),
        sons=[],
    )
    m = engine.tensor_term_to_module(tterm, {})
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(10, 4))  # inputs ignored; returns parameter (4, 2)
    assert out.shape == (4, 2)

def test_compile_linear_zero_children_raises():
    import pytest
    engine = Engine(db={})
    engine.set_run_globals({"Linear": nn.Linear})
    tterm = TensorTerm(
        op=TensorOp(op="Linear", hyper_params=[ArithTerm(value=2), ArithTerm(value=4), ArithTerm(value=False)]),
        sons=[],
    )
    with pytest.raises(NotImplementedError, match="Use Tensor\\(shape\\)"):
        engine.tensor_term_to_module(tterm, {})

def test_compile_transform_def():
    engine = Engine(db={})
    engine.set_run_globals({"Linear": nn.Linear})
    inner = TensorTerm(
        op=TensorOp(op="Linear", hyper_params=[ArithTerm(value=2), ArithTerm(value=4)]),
        sons=[TensorTerm(value=Var(name="z"))],
    )
    td = TransformDef(name="MyLin", tensor_term=inner)
    engine.symbol_table["global"]["MyLin"] = (TransformDef, td)
    tterm = TensorTerm(op=TensorOp(op="MyLin"), sons=[TensorTerm(value=Var(name="z"))])
    m = engine.tensor_term_to_module(tterm, {"z": 0})
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(3, 2))
    assert out.shape == (3, 4)

def main():
    test_resolve_from_run_globals()
    test_resolve_from_torch_nn()
    test_resolve_dsl_native()
    test_resolve_not_found()
    test_compile_linear()
    test_compile_relu()
    test_compile_concat()
    test_compile_view()
    test_compile_sqrt()
    test_compile_transpose()
    test_compile_transpose_capital_t()
    test_compile_unsqueeze_function_style()
    test_compile_unsqueeze_negative_dim()
    test_compile_callable_factory_from_run_globals_still_works()
    test_compile_varargs_callable_from_run_globals_uses_function_style()
    test_compile_arithmetic()
    test_compile_mseloss()
    test_compile_crossentropy()
    test_compile_argmax()
    test_compile_argmax_lowercase()
    test_compile_tensor()
    test_compile_linear_zero_children_raises()
    test_compile_transform_def()
    print("All tensor_term_compiler tests passed.")

# ---- replace_submodule (CR fix #2 regression) ----

def test_replace_submodule_in_module_list():
    """``replace_submodule`` must update slots inside ``nn.ModuleList``.

    Regression for CR fix #2: previously ``setattr(module_list, "0", x)`` no-ops
    on a ModuleList slot, leaving the original target live in forward(). Default
    text-encoder install on a leaf inside ``_MultiEncodeModule`` (which holds an
    ``nn.ModuleList``) silently failed.
    """
    from relann.tensor_term_compiler import replace_submodule

    target = nn.Linear(2, 2)
    other = nn.Linear(2, 2)
    replacement = nn.Linear(4, 4)
    parent = nn.Module()
    parent.lst = nn.ModuleList([other, target, other])

    assert replace_submodule(parent, target, replacement) is True
    assert parent.lst[1] is replacement
    # Slot index 0 / 2 untouched
    assert parent.lst[0] is other
    assert parent.lst[2] is other

def test_replace_submodule_returns_false_when_not_found():
    from relann.tensor_term_compiler import replace_submodule
    target = nn.Linear(2, 2)
    replacement = nn.Linear(4, 4)
    parent = nn.Module()
    parent.lst = nn.ModuleList([nn.Linear(2, 2)])
    assert replace_submodule(parent, target, replacement) is False

def test_replace_submodule_in_module_dict():
    """Same fix applies to ``nn.ModuleDict``."""
    from relann.tensor_term_compiler import replace_submodule
    target = nn.Linear(2, 2)
    replacement = nn.Linear(4, 4)
    parent = nn.Module()
    parent.dct = nn.ModuleDict({"a": nn.Linear(2, 2), "b": target})
    assert replace_submodule(parent, target, replacement) is True
    assert parent.dct["b"] is replacement

if __name__ == "__main__":
    main()
