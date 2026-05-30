"""
Tests for smart operations (parent.smart_ops) and their integration with
the tensor-term compiler and the Session E2E pipeline.

Covers: smart_matmul, smart_mul, smart_div, smart_add, smart_sub,
        smart_pow, smart_transpose, _align_dims_for_elementwise.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from relann.smart_ops import (
    _align_dims_for_elementwise,
    smart_matmul,
    smart_mul,
    smart_add,
    smart_sub,
    smart_div,
    smart_pow,
    smart_transpose,
)
from relann.engine import Engine
from relann.pydantic_classes import TensorTerm, TensorOp, Var, ArithTerm
from relann.session import Session
from relann.torch_utils import full_seed

# ========================================================================
# Unit tests: _align_dims_for_elementwise
# ========================================================================

def test_align_same_ndim_passthrough():
    a = torch.randn(5, 3)
    b = torch.randn(5, 3)
    a2, b2 = _align_dims_for_elementwise(a, b)
    assert a2.shape == (5, 3) and b2.shape == (5, 3)

def test_align_scalar_passthrough():
    a = torch.randn(5, 3, 4)
    b = torch.tensor(2.0)
    a2, b2 = _align_dims_for_elementwise(a, b)
    assert a2.shape == (5, 3, 4) and b2.shape == ()

def test_align_2d_vs_3d_trailing_unsqueeze():
    a = torch.randn(5, 1)
    b = torch.randn(5, 1, 1)
    a2, b2 = _align_dims_for_elementwise(a, b)
    assert a2.shape == (5, 1, 1) and b2.shape == (5, 1, 1)

def test_align_2d_vs_3d_feature():
    a = torch.randn(5, 4)
    b = torch.randn(5, 4, 8)
    a2, b2 = _align_dims_for_elementwise(a, b)
    assert a2.shape == (5, 4, 1) and b2.shape == (5, 4, 8)

def test_align_3d_vs_2d():
    a = torch.randn(5, 3, 4)
    b = torch.randn(5, 3)
    a2, b2 = _align_dims_for_elementwise(a, b)
    assert a2.shape == (5, 3, 4) and b2.shape == (5, 3, 1)

# ========================================================================
# Unit tests: smart_matmul
# ========================================================================

def test_smart_matmul_2d_2d():
    """Standard (E, d) @ (d, M) -> (E, M). No change from torch.matmul."""
    a = torch.randn(10, 4)
    b = torch.randn(4, 2)
    out = smart_matmul(a, b)
    assert out.shape == (10, 2)
    torch.testing.assert_close(out, torch.matmul(a, b))

def test_smart_matmul_2d_3d_hgt_attention():
    """(E, d) @ (E, d, 1) -> (E, 1, 1) via left unsqueeze. Core HGT case."""
    E, d = 7, 4
    a = torch.randn(E, d)
    b = torch.randn(E, d, 1)
    out = smart_matmul(a, b)
    assert out.shape == (E, 1, 1)
    expected = torch.matmul(a.unsqueeze(-2), b)
    torch.testing.assert_close(out, expected)

def test_smart_matmul_2d_3d_general():
    """(E, d) @ (E, d, M) -> (E, 1, M) via left unsqueeze."""
    E, d, M = 5, 3, 8
    a = torch.randn(E, d)
    b = torch.randn(E, d, M)
    out = smart_matmul(a, b)
    assert out.shape == (E, 1, M)

def test_smart_matmul_3d_2d():
    """(E, a, b) @ (E, b) -> unsqueeze right -> (E, a, 1)."""
    E, a, b = 5, 3, 4
    left = torch.randn(E, a, b)
    right = torch.randn(E, b)
    out = smart_matmul(left, right)
    assert out.shape == (E, a, 1)
    expected = torch.matmul(left, right.unsqueeze(-1))
    torch.testing.assert_close(out, expected)

def test_smart_matmul_3d_3d():
    """(E, a, b) @ (E, b, c) -> (E, a, c). Standard batched, unchanged."""
    E, a, b, c = 5, 3, 4, 2
    left = torch.randn(E, a, b)
    right = torch.randn(E, b, c)
    out = smart_matmul(left, right)
    assert out.shape == (E, a, c)
    torch.testing.assert_close(out, torch.matmul(left, right))

def test_smart_matmul_1d_2d():
    """1D @ 2D preserves torch.matmul native behavior."""
    a = torch.randn(4)
    b = torch.randn(4, 2)
    out = smart_matmul(a, b)
    assert out.shape == (2,)
    torch.testing.assert_close(out, torch.matmul(a, b))

def test_smart_matmul_2d_1d():
    """2D @ 1D preserves torch.matmul native behavior."""
    a = torch.randn(3, 4)
    b = torch.randn(4)
    out = smart_matmul(a, b)
    assert out.shape == (3,)
    torch.testing.assert_close(out, torch.matmul(a, b))

def test_smart_matmul_single_row_cora_dummy():
    """E=1, d=32: the Cora dummy-input scenario during instantiate."""
    a = torch.randn(1, 32)
    b = torch.randn(1, 32, 1)
    out = smart_matmul(a, b)
    assert out.shape == (1, 1, 1)

def test_smart_matmul_rejects_4d():
    """4-D operands are not valid in RelNN matmul context -- fail loud."""
    a = torch.randn(2, 3, 4, 5)
    b = torch.randn(2, 3, 5, 6)
    try:
        smart_matmul(a, b)
        assert False, "Expected ValueError for 4-D operands"
    except ValueError as e:
        assert "at most 3-D" in str(e)

    a2d = torch.randn(2, 5)
    try:
        smart_matmul(a2d, a)
        assert False, "Expected ValueError when one operand is 4-D"
    except ValueError as e:
        assert "at most 3-D" in str(e)

# ========================================================================
# Unit tests: smart_mul
# ========================================================================

def test_smart_mul_3d_2d_attention_mu():
    """(E, 1, 1) * (E, 1) -> (E, 1, 1). Core HGT attention * Mu pattern."""
    E = 7
    a = torch.randn(E, 1, 1)
    b = torch.randn(E, 1)
    out = smart_mul(a, b)
    assert out.shape == (E, 1, 1)

def test_smart_mul_native_wrong():
    """Verify that native PyTorch gives the WRONG result for this case."""
    a = torch.ones(5, 1, 1)
    b = torch.ones(5, 1)
    native = torch.mul(a, b)
    assert native.shape == (5, 5, 1), "expected native to give wrong (5,5,1)"
    smart = smart_mul(a, b)
    assert smart.shape == (5, 1, 1), "smart_mul should fix to (5,1,1)"

def test_smart_mul_2d_3d_trailing():
    """(E, d) * (E, d, M) -> (E, d, M) via trailing unsqueeze."""
    a = torch.randn(5, 3)
    b = torch.randn(5, 3, 4)
    out = smart_mul(a, b)
    assert out.shape == (5, 3, 4)

def test_smart_mul_same_ndim():
    """Same ndim: (E, d) * (E, d) -> (E, d). Native PyTorch, unchanged."""
    a = torch.randn(5, 3)
    b = torch.randn(5, 3)
    out = smart_mul(a, b)
    assert out.shape == (5, 3)
    torch.testing.assert_close(out, a * b)

def test_smart_mul_scalar():
    """(E, d) * scalar -> (E, d). Native PyTorch, unchanged."""
    a = torch.randn(5, 3)
    s = torch.tensor(2.0)
    out = smart_mul(a, s)
    assert out.shape == (5, 3)
    torch.testing.assert_close(out, a * 2.0)

# ========================================================================
# Unit tests: smart_div
# ========================================================================

def test_smart_div_3d_2d():
    """(E, 1, 1) / (E, 1) -> (E, 1, 1)."""
    a = torch.randn(7, 1, 1)
    b = torch.randn(7, 1).abs() + 0.1
    out = smart_div(a, b)
    assert out.shape == (7, 1, 1)

def test_smart_div_scalar():
    """(E, d) / scalar -> (E, d)."""
    a = torch.randn(5, 4)
    out = smart_div(a, torch.tensor(2.0))
    assert out.shape == (5, 4)

# ========================================================================
# Unit tests: smart_add / smart_sub
# ========================================================================

def test_smart_add_3d_2d():
    """(E, 1, 1) + (E, 1) -> (E, 1, 1)."""
    a = torch.randn(5, 1, 1)
    b = torch.randn(5, 1)
    out = smart_add(a, b)
    assert out.shape == (5, 1, 1)

def test_smart_sub_3d_2d():
    """(E, 1, 1) - (E, 1) -> (E, 1, 1)."""
    a = torch.randn(5, 1, 1)
    b = torch.randn(5, 1)
    out = smart_sub(a, b)
    assert out.shape == (5, 1, 1)

def test_smart_add_same_ndim():
    """Same ndim unchanged."""
    a = torch.randn(5, 4)
    b = torch.randn(5, 4)
    out = smart_add(a, b)
    torch.testing.assert_close(out, a + b)

# ========================================================================
# Unit tests: smart_pow
# ========================================================================

def test_smart_pow_3d_2d():
    """(E, 1, 1) ** (E, 1) -> (E, 1, 1)."""
    a = torch.rand(5, 1, 1) + 0.1
    b = torch.rand(5, 1) + 0.5
    out = smart_pow(a, b)
    assert out.shape == (5, 1, 1)

def test_smart_pow_scalar():
    """(E, d) ** 2 -> (E, d)."""
    a = torch.randn(5, 3)
    out = smart_pow(a, torch.tensor(2.0))
    assert out.shape == (5, 3)

# ========================================================================
# Unit tests: smart_transpose
# ========================================================================

def test_smart_transpose_2d():
    """(E, d) -> (E, d, 1): column-vector per row."""
    x = torch.randn(10, 4)
    out = smart_transpose(x)
    assert out.shape == (10, 4, 1)
    torch.testing.assert_close(out, x.unsqueeze(-1))

def test_smart_transpose_3d():
    """(E, a, b) -> (E, b, a): matrix transpose per row."""
    x = torch.randn(10, 3, 5)
    out = smart_transpose(x)
    assert out.shape == (10, 5, 3)
    torch.testing.assert_close(out, x.transpose(-1, -2))

def test_smart_transpose_1d():
    """1D: native transpose(-1, -2) on 1D would fail; unsqueeze not triggered."""
    x = torch.randn(4)
    out = smart_transpose(x)
    assert out.shape == (4,)

# ========================================================================
# Compiler-level tests
# ========================================================================

def test_compile_matmul_transpose_shape():
    """Compile a @ transpose(b) with 2D inputs and verify the output shape."""
    engine = Engine(db={})
    tterm = TensorTerm(
        op=TensorOp(op="@"),
        sons=[
            TensorTerm(value=Var(name="z1")),
            TensorTerm(
                op=TensorOp(op="transpose"),
                sons=[TensorTerm(value=Var(name="z2"))],
            ),
        ],
    )
    m = engine.tensor_term_to_module(tterm, {"z1": 0, "z2": 1})
    m.eval()
    E, d = 7, 4
    with torch.no_grad():
        out = m(torch.randn(E, d), torch.randn(E, d))
    assert out.shape == (E, 1, 1), f"expected ({E}, 1, 1), got {out.shape}"

def test_compile_transpose_2d_shape():
    """Compiler transpose on 2D produces (E, d, 1)."""
    engine = Engine(db={})
    tterm = TensorTerm(
        op=TensorOp(op="transpose"),
        sons=[TensorTerm(value=Var(name="z1"))],
    )
    m = engine.tensor_term_to_module(tterm, {"z1": 0})
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(100, 2))
    assert out.shape == (100, 2, 1)

def test_compile_transpose_3d_shape():
    """Compiler transpose on 3D produces (E, b, a)."""
    engine = Engine(db={})
    tterm = TensorTerm(
        op=TensorOp(op="transpose"),
        sons=[TensorTerm(value=Var(name="z1"))],
    )
    m = engine.tensor_term_to_module(tterm, {"z1": 0})
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(10, 3, 5))
    assert out.shape == (10, 5, 3)

def test_compile_mul_mismatched_ndims():
    """Compile a * b where a is 3D and b is 2D, verify smart broadcasting."""
    engine = Engine(db={})
    tterm_mul = TensorTerm(
        op=TensorOp(op="*"),
        sons=[
            TensorTerm(value=Var(name="z1")),
            TensorTerm(value=Var(name="z2")),
        ],
    )
    m = engine.tensor_term_to_module(tterm_mul, {"z1": 0, "z2": 1})
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(5, 1, 1), torch.randn(5, 1))
    assert out.shape == (5, 1, 1), f"expected (5, 1, 1), got {out.shape}"

def test_compile_hgt_attention_expression():
    """Compile K(z1) @ Tensor(d, d) @ transpose(Q(z2)) * c / sqrt(d).

    Full HGT attention score expression through the compiler.
    d = 4, c is a scalar 1.0.
    """
    engine = Engine(db={})
    engine.set_run_globals({"Linear": nn.Linear})
    d = 4

    z1_leaf = TensorTerm(value=Var(name="z1"))
    z2_leaf = TensorTerm(value=Var(name="z2"))

    k_z1 = TensorTerm(
        op=TensorOp(op="Linear", hyper_params=[ArithTerm(value=d), ArithTerm(value=d), ArithTerm(value=False)]),
        sons=[z1_leaf],
    )
    w_att = TensorTerm(
        op=TensorOp(op="Tensor", hyper_params=[ArithTerm(value=d), ArithTerm(value=d)]),
        sons=[],
    )
    kw = TensorTerm(op=TensorOp(op="@"), sons=[k_z1, w_att])
    transpose_q = TensorTerm(
        op=TensorOp(op="transpose"),
        sons=[
            TensorTerm(
                op=TensorOp(op="Linear", hyper_params=[ArithTerm(value=d), ArithTerm(value=d), ArithTerm(value=False)]),
                sons=[z2_leaf],
            ),
        ],
    )
    att_raw = TensorTerm(op=TensorOp(op="@"), sons=[kw, transpose_q])
    sqrt_d = TensorTerm(op=TensorOp(op="sqrt"), sons=[TensorTerm(value=d)])
    att_scaled = TensorTerm(op=TensorOp(op="/"), sons=[att_raw, sqrt_d])

    m = engine.tensor_term_to_module(att_scaled, {"z1": 0, "z2": 1})
    m.eval()
    E = 7
    with torch.no_grad():
        out = m(torch.randn(E, d), torch.randn(E, d))
    assert out.shape == (E, 1, 1), f"expected ({E}, 1, 1), got {out.shape}"

# ========================================================================
# E2E tests through Session
# ========================================================================

def _edge_db(n_src: int = 4, n_dst: int = 3, n_edges: int = 6, d: int = 4):
    """Create a minimal edge database: Src(s; z1), Dst(t; z2), Edge(s, e, t)."""
    src_df = pd.DataFrame({"s": range(n_src)})
    dst_df = pd.DataFrame({"t": range(n_dst)})
    edges = [(i % n_src, i % n_dst) for i in range(n_edges)]
    edge_df = pd.DataFrame({"s": [e[0] for e in edges], "t": [e[1] for e in edges]})
    return {
        "Src": (src_df, torch.randn(n_src, d)),
        "Dst": (dst_df, torch.randn(n_dst, d)),
        "Edge": (edge_df, torch.ones(n_edges, 1)),
    }

def test_e2e_matmul_tensor_still_works():
    """z1 @ Tensor(d, M) should still produce (N, M)."""
    full_seed(42)
    db = {"Input": (pd.DataFrame({"a": range(10)}), torch.randn(10, 4))}
    session = Session(db=db)
    session.run("""
#lang:relnn
Test(a; z1 @ Tensor(4, 2)) :- Input(a; z1) .
""")
    result = session.run("""
#lang:relnn
?pred Out(a; z) :- Test(a; z) .
""")
    assert result is not None
    assert result.embeddings[0].shape == (10, 2)

def test_e2e_transpose_output_shape():
    """transpose(z1) on (N, d) should produce (N, d, 1)."""
    full_seed(42)
    db = {"Input": (pd.DataFrame({"a": range(10)}), torch.randn(10, 4))}
    session = Session(db=db)
    session.run("""
#lang:relnn
Test(a; transpose(z1)) :- Input(a; z1) .
""")
    result = session.run("""
#lang:relnn
?pred Out(a; z) :- Test(a; z) .
""")
    assert result is not None
    assert result.embeddings[0].shape == (10, 4, 1)

def test_e2e_matmul_transpose_attention_score():
    """z1 @ Tensor(d, d) @ transpose(z2) should produce a scalar per edge row."""
    full_seed(42)
    d = 4
    n_src, n_dst, n_edges = 4, 3, 6
    db = _edge_db(n_src=n_src, n_dst=n_dst, n_edges=n_edges, d=d)
    session = Session(db=db)
    session.run(f"""
#lang:relnn
ATT(s, t; z1 @ Tensor({d}, {d}) @ transpose(z2)) :- Src(s; z1), Edge(s, t), Dst(t; z2) .
""")
    result = session.run("""
#lang:relnn
?pred Out(s, t; z) :- ATT(s, t; z) .
""")
    assert result is not None
    emb = result.embeddings[0]
    assert emb.shape[1:] == (1, 1), f"expected (*, 1, 1), got {emb.shape}"

def test_e2e_full_hgt_attention_chain():
    """K(z1) @ Tensor(d, d) @ transpose(Q(z2)) * Mu / sqrt(d).

    Full attention score expression through Session -- the original
    failing pattern that motivated smart ops.
    """
    full_seed(42)
    d = 4
    n_src, n_dst, n_edges = 4, 3, 6
    db = _edge_db(n_src=n_src, n_dst=n_dst, n_edges=n_edges, d=d)
    session = Session(db=db)
    session.run(f"""
#lang:relnn
K_Lin = Linear({d}, {d}, False) .
Q_Lin = Linear({d}, {d}, False) .
Mu = Linear(1, 1, False) .
ATT(s, t; K_Lin(z1) @ Tensor({d}, {d}) @ transpose(Q_Lin(z2)) * Mu(z3) / sqrt({d})) :- Src(s; z1), Edge(s, t; z3), Dst(t; z2) .
""")
    result = session.run("""
#lang:relnn
?pred Out(s, t; z) :- ATT(s, t; z) .
""")
    assert result is not None
    emb = result.embeddings[0]
    assert emb.shape[1:] == (1, 1), f"expected (*, 1, 1), got {emb.shape}"

# ========================================================================
# Main
# ========================================================================

if __name__ == "__main__":
    # _align_dims_for_elementwise
    test_align_same_ndim_passthrough()
    test_align_scalar_passthrough()
    test_align_2d_vs_3d_trailing_unsqueeze()
    test_align_2d_vs_3d_feature()
    test_align_3d_vs_2d()
    # smart_matmul
    test_smart_matmul_2d_2d()
    test_smart_matmul_2d_3d_hgt_attention()
    test_smart_matmul_2d_3d_general()
    test_smart_matmul_3d_2d()
    test_smart_matmul_3d_3d()
    test_smart_matmul_1d_2d()
    test_smart_matmul_2d_1d()
    test_smart_matmul_single_row_cora_dummy()
    test_smart_matmul_rejects_4d()
    # smart_mul
    test_smart_mul_3d_2d_attention_mu()
    test_smart_mul_native_wrong()
    test_smart_mul_2d_3d_trailing()
    test_smart_mul_same_ndim()
    test_smart_mul_scalar()
    # smart_div
    test_smart_div_3d_2d()
    test_smart_div_scalar()
    # smart_add / smart_sub
    test_smart_add_3d_2d()
    test_smart_sub_3d_2d()
    test_smart_add_same_ndim()
    # smart_pow
    test_smart_pow_3d_2d()
    test_smart_pow_scalar()
    # smart_transpose
    test_smart_transpose_2d()
    test_smart_transpose_3d()
    test_smart_transpose_1d()
    # Compiler-level
    test_compile_matmul_transpose_shape()
    test_compile_transpose_2d_shape()
    test_compile_transpose_3d_shape()
    test_compile_mul_mismatched_ndims()
    test_compile_hgt_attention_expression()
    # E2E Session
    test_e2e_matmul_tensor_still_works()
    test_e2e_transpose_output_shape()
    test_e2e_matmul_transpose_attention_score()
    test_e2e_full_hgt_attention_chain()
    print("All smart_ops tests passed.")
