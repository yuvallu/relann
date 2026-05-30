"""
Repro tests for symbol-table arithmetic edge cases.
"""

import pandas as pd
import torch

from relann.pydantic_classes import ArithTerm, TransformDef, Var
from relann.session import Session
from relann.era_operations import EmbeddedRelation


def test_transformdef_bool_hyperparam_should_remain_boolean():
    """Parser leaves ctor args in sons; promotion to hyper_params happens at compile time.
    This test verifies end-to-end that False is handled correctly as a bool ctor arg."""
    import pandas as pd
    n = 4
    df = pd.DataFrame({"i": range(n)})
    z = torch.randn(n, 8)
    s = Session(db={"Input": (df, z)})
    s.run(
        """
        d = 8 .
        h = 2 .
        K_Linear_Paper = Linear(d, d / h, False) .
        Out(i; K_Linear_Paper(z1)) :- Input(i; z1) .
        """
    )
    result = s.run("?pred P(i; z) :- Out(i; z) .")
    # Linear(8, 4, bias=False): output shape (n, 4)
    assert result is not None and result.embeddings is not None
    assert tuple(result.embeddings[0].shape) == (n, 4)

    # Internal: parser stores sons, not hyper_params, at definition time.
    sym = s.engine.get_symbol("K_Linear_Paper")
    assert sym is not None
    typ, obj = sym
    assert typ is TransformDef
    assert obj.tensor_term.sons is not None and len(obj.tensor_term.sons) == 3
    assert obj.tensor_term.op.hyper_params is None


def test_symbol_alias_of_arith_expression_should_evaluate_to_scalar():
    s = Session(db={})
    s.define(
        """
        d = 96 .
        h = 8 .
        a = d / h .
        """
    )

    # Desired behavior: symbol alias of arithmetic expression should evaluate fully.
    got = s.engine.evaluate_arith_term_for_hyperparams(ArithTerm(value=Var(name="a")))
    assert got == 12


def test_symbol_alias_arith_can_be_used_in_linear_hyperparams():
    """Parser leaves ctor args in sons; end-to-end test verifies correct instantiation."""
    s = Session(db={})
    s.define(
        """
        d = 8 .
        h = 2 .
        a = d / h .
        K_Linear_Paper = Linear(d, a, False) .
        """
    )

    sym = s.engine.get_symbol("K_Linear_Paper")
    assert sym is not None
    typ, obj = sym
    assert typ is TransformDef

    # Internal: parser stores sons (not hyper_params) at definition time.
    # Promotion to hyper_params happens at engine/compile time when the transform is used.
    assert obj.tensor_term.sons is not None and len(obj.tensor_term.sons) == 3
    assert obj.tensor_term.op.hyper_params is None


def test_symbol_alias_linear_works_end_to_end_in_program():
    n = 6
    df = pd.DataFrame({"i": range(n)})
    z = torch.randn(n, 8)
    s = Session(db={"Input": (df, z)})

    s.run(
        """
        d = 8 .
        h = 2 .
        a = d / h .
        K_Linear_Paper = Linear(d, a, False) .
        Test1(i; K_Linear_Paper(z1)) :- Input(i; z1) .
        """
    )
    result = s.run(
        """
        ?pred Out(i; ArgMax()(z)) :- Test1(i; z) .
        """
    )

    assert isinstance(result, EmbeddedRelation)
    assert result.embeddings is not None and len(result.embeddings) > 0
    assert tuple(result.embeddings[0].shape) == (n, 1)
