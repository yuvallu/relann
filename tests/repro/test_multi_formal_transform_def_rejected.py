"""Engine-policy pin: a ``TransformDef`` body that resolves to MORE THAN ONE
free formal ``Var`` is rejected at substitution time.

Background — see the β-reduction code in ``relann/engine.py::_apply_call_argument`` /
``relann/pydantic_classes.py::collect_formal_vars``.

The pre-fix engine substituted a single hard-coded ``Var("inp")`` placeholder
and silently ignored any other free variables. The post-fix engine infers the
body's formals via ``collect_formal_vars`` and β-reduces them. The DSL only
supports single-formal transform bodies, so a body with two distinct free Vars
(e.g. ``Add(x, y)``) is a user error and is rejected LOUDLY rather than binding
only one formal and silently dropping the other.

These are unit-level pins (no Session / DSL): the malformed shape is built
directly, because the parser does not currently emit two-formal bodies and so
there is no DSL string that reaches this path today. If that ever changes, add
a DSL-level repro alongside these and update the design doc together.
"""
from __future__ import annotations

import pytest

from relann.engine import _apply_call_argument
from relann.pydantic_classes import TensorOp, TensorTerm, Var, collect_formal_vars


def _leaf(name: str) -> TensorTerm:
    """A bare ``Var`` leaf — the shape the parser emits for a free variable."""
    return TensorTerm(op=None, sons=None, value=Var(name=name))


def _two_formal_body() -> TensorTerm:
    """``Add(x, y)`` — two distinct free formal Vars in a single body."""
    return TensorTerm(
        op=TensorOp(op="Add", hyper_params=None),
        sons=[_leaf("x"), _leaf("y")],
        value=None,
    )


def test_collect_formal_vars_counts_distinct_free_vars():
    """Ordered, de-duplicated free Var names. One formal -> one name; two
    distinct formals -> both, in traversal order."""
    assert collect_formal_vars(_leaf("x")) == ["x"]
    assert collect_formal_vars(_two_formal_body()) == ["x", "y"]


def test_collect_formal_vars_filters_reserved_literals():
    """``True`` / ``False`` / ``None`` are encoded as ``Var`` by the parser but
    are literals, not formals — they must never count toward the formal set."""
    body = TensorTerm(
        op=TensorOp(op="Add", hyper_params=None),
        sons=[_leaf("x"), _leaf("True"), _leaf("None")],
        value=None,
    )
    assert collect_formal_vars(body) == ["x"]


def test_apply_call_argument_rejects_multi_formal_body():
    """The β-reduction entry point raises ``ValueError`` (not a silent
    single-binding) when the resolved body has more than one free formal.

    Pins the post-fix policy; pre-fix this path bound only ``Var("inp")`` and
    dropped the rest. The raise fires before ``engine`` / ``replace_fn`` /
    ``call_sons`` are touched, so passing ``None`` for them is safe here."""
    body = _two_formal_body()
    arg0 = _leaf("h")  # any "actual" argument; unused before the raise
    with pytest.raises(ValueError, match=r"multiple unresolved formals"):
        # engine / replace_fn / call_sons are unused on the multi-formal raise
        # path (the ValueError fires first), so None is safe here — silence the
        # type-checker rather than build a throwaway Engine just to be discarded.
        _apply_call_argument(body, arg0, None, None, [])  # type: ignore[arg-type]
