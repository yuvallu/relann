"""Regression test for the ``Tensor(d/h, d/h)`` shape bug.

Background
----------

The DSL allows arithmetic in shape arguments::

    d = 16 .
    h = 4 .
    W = Tensor(d/h, d/h) .       # user intends shape (4, 4)

Pre-fix, Python's ``/`` returned ``4.0`` (true-division) for each dim, and
``_ParameterTensor.__init__``'s "last arg is float → fill value" heuristic
mis-read ``_ParameterTensor(4.0, 4.0)`` as "shape (4,) filled with 4.0".
The HGT ``transformation_L1`` "batch2 [1, 1] vs [1, 4]" matmul error was the
*downstream* symptom (``K @ W`` collapsed from ``(1, 4) @ (4, 4)`` to
``(1, 4) @ (4,) = (1,)``, killing the next matmul).

Fix (landed 2026-05-24): at the central
``TensorTermCompiler._eval_hyperparams`` chokepoint, distinguish DSL
**literal floats** from **computed** floats by inspecting the source
``ArithTerm``. Coerce only the computed integer-valued ones to int.
Literal floats (the ``1.0`` in pyHGT-faithful ``Tensor(1, 1.0)``
ones-init) are preserved so the fill-value heuristic still fires for
those callers.

These tests pin the post-fix contract. They are positive assertions —
flipping any of them to fail signals a regression in shape-arith
coercion (suspect ``_coerce_computed_int_float`` and its single caller in
``TensorTermCompiler._eval_hyperparams``).
"""
from __future__ import annotations

import pandas as pd
import torch

from relann.session import Session
from relann.tensor_term_compiler import _ParameterTensor, _coerce_computed_int_float
from relann.pydantic_classes import ArithTerm


# ---------------------------------------------------------------------------
# Layer 1: the helper itself (no Session, no DSL)
# ---------------------------------------------------------------------------

def test_helper_preserves_literal_floats():
    """``Tensor(1, 1.0)`` — pyHGT ones-init — must keep the ``1.0`` as float
    so the fill-value heuristic still fires. Helper sees source ArithTerm
    is a bare literal (``op=None, sons=None, value=float``) → preserve."""
    literal_one_pt_zero = ArithTerm(value=1.0)
    assert _coerce_computed_int_float(1.0, literal_one_pt_zero) == 1.0
    assert isinstance(_coerce_computed_int_float(1.0, literal_one_pt_zero), float)


def test_helper_coerces_computed_integer_floats():
    """``d/h`` arithmetic where both operands are int and the quotient is
    integer-valued: source ArithTerm has ``op='/'`` → coerce to int."""
    div_16_over_4 = ArithTerm(op="/", sons=[ArithTerm(value=16), ArithTerm(value=4)])
    out = _coerce_computed_int_float(4.0, div_16_over_4)
    assert out == 4
    assert isinstance(out, int)


def test_helper_leaves_non_integer_floats_alone():
    """``0.5`` is not integer-valued — never coerced, regardless of origin."""
    lit = ArithTerm(value=0.5)
    div = ArithTerm(op="/", sons=[ArithTerm(value=1), ArithTerm(value=2)])
    assert _coerce_computed_int_float(0.5, lit) == 0.5
    assert _coerce_computed_int_float(0.5, div) == 0.5


def test_helper_passthrough_for_ints():
    """Ints pass through unchanged."""
    assert _coerce_computed_int_float(4, None) == 4
    assert isinstance(_coerce_computed_int_float(4, None), int)


# ---------------------------------------------------------------------------
# Layer 2: ParameterTensor invariants (still no Session)
# ---------------------------------------------------------------------------

def test_parameter_tensor_int_dims_unchanged():
    """Literal int shape dims still give the expected shape with default fill."""
    assert tuple(_ParameterTensor(4, 4).weight.shape) == (4, 4)
    assert tuple(_ParameterTensor(16, 4).weight.shape) == (16, 4)


def test_parameter_tensor_explicit_non_integer_fill_unchanged():
    """``Tensor(4, 4, 0.5)`` — explicit non-integer fill value — keeps the
    (4, 4) shape and applies fill=0.5. Backward-compat for explicit fills."""
    p = _ParameterTensor(4, 4, 0.5)
    assert tuple(p.weight.shape) == (4, 4)
    assert torch.allclose(p.weight, torch.full((4, 4), 0.5))


# ---------------------------------------------------------------------------
# Layer 3: end-to-end through the engine — what the HGT scenario actually hits
# ---------------------------------------------------------------------------

def test_tensor_div_dims_via_dsl_gives_correct_shape():
    """``Tensor(d/h, d/h)`` with ``d=16, h=4`` produces shape (4, 4) when
    invoked through the engine — proves the chokepoint fix is wired into
    the DSL evaluation path."""
    session = Session(db={"Input": (pd.DataFrame({"i": [0, 1]}), torch.ones(2, 4))})
    session.run("""
    d = 16 .
    h = 4 .
    W = Tensor(d/h, d/h) .
    Probe(i; z @ W) :- Input(i; z) .
    ?pred Out(i; z) :- Probe(i; z) .
    """)
    out = session.engine.relation("Out")
    # z is (2, 4); W must be (4, 4) for z @ W to give (2, 4). If W is
    # mis-shaped to (4,) (the pre-fix bug), this collapses to (2,).
    assert tuple(out.embeddings[0].shape) == (2, 4), (
        f"Expected (2, 4); got {tuple(out.embeddings[0].shape)}. If shape is "
        f"(2,), `_coerce_computed_int_float` failed to coerce the d/h=4.0 "
        f"floats back to ints and `_ParameterTensor` mis-read them as a "
        f"fill-value call."
    )


def test_tensor_ones_init_pattern_still_works():
    """``Tensor(1, 1.0)`` — pyHGT ones-init used by dblp_from_sqlite and rgcn
    demos — still produces shape (1,) filled with 1.0. The literal-float
    preservation in ``_coerce_computed_int_float`` keeps this working."""
    session = Session(db={"Input": (pd.DataFrame({"i": [0, 1]}), torch.ones(2, 1))})
    session.run("""
    W = Tensor(1, 1.0) .
    Probe(i; z * W) :- Input(i; z) .
    ?pred Out(i; z) :- Probe(i; z) .
    """)
    out = session.engine.relation("Out")
    # If W somehow lost its fill=1.0 (e.g. naive "coerce all integer floats"),
    # z * W would be zeros and the test would still pass shape-wise but the
    # values would be wrong. Check both shape AND that the fill propagated.
    assert tuple(out.embeddings[0].shape) == (2, 1)
    assert torch.allclose(out.embeddings[0], torch.ones(2, 1))
