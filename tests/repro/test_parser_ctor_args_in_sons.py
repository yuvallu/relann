"""Pins the parser's representation of ``Linear(16, 32)`` ctor args.

- Paren form ``Linear(16, 32)`` → values land in ``tensor_term.sons``
  (NOT ``tensor_term.op.hyper_params``).
- Angle form ``K_Linear<l, S, i>`` → params land on
  ``TransformDef.template_params``.

Mirrors the live assertion in ``relann/parser.py:~2231``'s demo block.
If this test breaks, the parser representation has changed — revisit both
this file and that demo together. See
``docs/design/notebook-demos.md::Resolved: parser.py:2223``.
"""
from __future__ import annotations

from relann.engine import Engine
from relann.parser import RelnnTransformer, get_relnn_grammar_parser
from relann.pydantic_classes import TensorOp, TensorTerm, TransformDef


def _parse_transform_def(src: str) -> TransformDef:
    parser = get_relnn_grammar_parser(start="transform_def")
    tree = parser.parse(src)
    return RelnnTransformer(Engine()).transform(tree)


def test_paren_ctor_args_live_in_tensor_term_sons():
    """``Linear(16, 32)`` — round brackets — puts ``16`` and ``32`` in ``tensor_term.sons``.

    Current schema (verified 2026-05-24):
      result.tensor_term.op           = TensorOp(op='Linear', hyper_params=None, template_args=None)
      result.tensor_term.sons         = [TensorTerm(value=16), TensorTerm(value=32)]
      result.tensor_term.value        = None
    """
    result = _parse_transform_def("Lin = Linear(16, 32) .")

    assert isinstance(result, TransformDef)
    assert result.name == "Lin"
    assert result.template_params is None or result.template_params == []

    op = result.tensor_term.op
    assert isinstance(op, TensorOp)
    assert op.op == "Linear"
    # Ctor args live in `sons`, not in `op.hyper_params`.
    assert op.hyper_params is None, (
        f"expected hyper_params to be None for paren ctors; got {op.hyper_params}"
    )
    assert op.template_args is None, (
        f"expected template_args to be None for paren ctors; got {op.template_args}"
    )

    sons = result.tensor_term.sons
    assert sons is not None and len(sons) == 2
    assert all(isinstance(s, TensorTerm) for s in sons)
    assert [s.value for s in sons] == [16, 32], (
        f"expected ctor args [16, 32] in tensor_term.sons; got {[s.value for s in sons]}"
    )


def test_angle_template_params_live_on_transform_def():
    """``K_Linear<l, S, i> = 12 .`` — angle brackets — puts ``l, S, i`` on
    ``TransformDef.template_params``, NOT on ``tensor_term.op``.
    """
    result = _parse_transform_def("K_Linear<l, S, i> = 12 .")

    assert result.name == "K_Linear"
    assert result.template_params is not None
    assert [v.name for v in result.template_params] == ["l", "S", "i"]
    # The body is just the literal 12 — no op, no sons.
    assert result.tensor_term.op is None
    assert result.tensor_term.sons is None
    assert result.tensor_term.value == 12 or result.tensor_term.value == 12.0


def test_angle_with_int_template_args_for_K_Linear():
    """``K_Linear<16, 32> = 12 .`` — angle brackets with int literals. Same shape
    as the previous test (template_params is parsed as a list of Var-likes)."""
    result = _parse_transform_def("K_Linear<16, 32> = 12 .")

    assert result.name == "K_Linear"
    # The body is unchanged regardless of what's in the angle brackets.
    assert result.tensor_term.op is None
    assert result.tensor_term.sons is None
    assert result.tensor_term.value == 12 or result.tensor_term.value == 12.0
