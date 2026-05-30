"""Feature tests for template parameters in TransformDef, FunctionDef, and Rule.

Tests focus on symbol-table state, error handling, and internal invariants.
End-to-end predict shape tests are in smoke/test_template_smoke.py.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from relann.pydantic_classes import DerivedER, FunctionDef, Rule, TransformDef, Var
from relann.session import Session
from relann.torch_utils import full_seed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_papers_db(n: int = 6, d: int = 4):
    df = pd.DataFrame({"pid": list(range(n))})
    z = torch.randn(n, d)
    return {"Papers": (df, z)}

# ---------------------------------------------------------------------------
# TransformDef template tests
# ---------------------------------------------------------------------------

def test_templated_transformdef_stored_in_symbol_table():
    """A templated TransformDef should be stored in the symbol table with its template_params."""
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define("""
#lang:relnn
Lin<d> = Linear(d, d, False) .
""")

    assert "Lin" in s.engine.symbol_table["global"]
    typ, obj = s.engine.symbol_table["global"]["Lin"]
    assert typ is TransformDef
    assert obj.name == "Lin"
    assert obj.template_params is not None
    assert len(obj.template_params) == 1
    assert isinstance(obj.template_params[0], Var)
    assert obj.template_params[0].name == "d"

def test_templated_transformdef_not_added_to_tg():
    """A templated TransformDef should not produce any term graph nodes at definition time."""
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define("""
#lang:relnn
Lin<d> = Linear(d, d, False) .
""")

    tg = s.engine.term_graphs["global"]
    assert len(list(tg.nodes())) == 0

def test_templated_transformdef_multi_param_stored():
    """A TransformDef with multiple template params should store all of them."""
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define("""
#lang:relnn
Lin<d_in, d_out> = Linear(d_in, d_out, False) .
""")

    typ, obj = s.engine.symbol_table["global"]["Lin"]
    assert typ is TransformDef
    assert len(obj.template_params) == 2
    assert obj.template_params[0].name == "d_in"
    assert obj.template_params[1].name == "d_out"

def test_templated_transformdef_wrong_arg_count_raises():
    """Using a templated TransformDef with wrong number of template args should raise."""
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define("""
#lang:relnn
Lin<d> = Linear(d, d, False) .
""")

    with pytest.raises((ValueError, TypeError)):
        s.run("""
#lang:relnn
Output(pid; Lin<64, 32>(z)) :- Papers(pid; z) .
?pred Predictions(pid; z) :- Output(pid; z) .
""")

# ---------------------------------------------------------------------------
# FunctionDef template tests
# ---------------------------------------------------------------------------

def test_templated_functiondef_stored_in_symbol_table():
    """A templated FunctionDef should be stored in the global symbol table."""
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define("""
#lang:relnn
def Encoder<h, d>(Papers):
    Enc(pid; Linear(d, h, False)(z)) :- Papers(pid; z) .
enddef
""")

    assert "Encoder" in s.engine.symbol_table["global"]
    typ, obj = s.engine.symbol_table["global"]["Encoder"]
    assert typ is FunctionDef
    assert obj.name == "Encoder"
    assert obj.template_params is not None
    assert len(obj.template_params) == 2
    assert obj.template_params[0].name == "h"
    assert obj.template_params[1].name == "d"

def test_templated_functiondef_body_not_processed_until_called():
    """A templated FunctionDef body should NOT be processed at definition time."""
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define("""
#lang:relnn
def Encoder<h, d>(Papers):
    Enc(pid; Linear(d, h, False)(z)) :- Papers(pid; z) .
enddef
""")

    # Function is registered in global scope
    assert "Encoder" in s.engine.symbol_table["global"]

    # No function-scoped term graph was created (body not processed yet)
    assert "Encoder" not in s.engine.term_graphs

def test_templated_functiondef_wrong_arg_count_raises():
    """Calling a 2-param template function with 1 template arg should raise."""
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define("""
#lang:relnn
def Encoder<h, d>(Papers):
    Enc(pid; Linear(d, h, False)(z)) :- Papers(pid; z) .
enddef
""")

    with pytest.raises((ValueError, TypeError)):
        s.run("""
#lang:relnn
Output(pid; z) :- Encoder<16>(Papers)(pid; z) .
?pred Predictions(pid; z) :- Output(pid; z) .
""")

# ---------------------------------------------------------------------------
# Rule template tests
# ---------------------------------------------------------------------------

def test_templated_rule_stored_in_symbol_table():
    """A templated Rule should be stored in the symbol table (not added to tg)."""
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define("""
#lang:relnn
Layer<d_out>(pid; Linear(4, d_out, False)(z)) :- Papers(pid; z) .
""")

    assert "Layer" in s.engine.symbol_table["global"]
    typ, obj = s.engine.symbol_table["global"]["Layer"]
    assert typ is Rule
    assert obj.lhs.template_params is not None
    assert len(obj.lhs.template_params) == 1

def test_templated_rule_not_added_to_tg():
    """A templated Rule should not create any term graph nodes at definition time."""
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define("""
#lang:relnn
Layer<d_out>(pid; Linear(4, d_out, False)(z)) :- Papers(pid; z) .
""")

    tg = s.engine.term_graphs["global"]
    # No nodes should exist for the templated rule
    for _, data in tg.nodes(data=True):
        assert data.get("name") != "Layer", "Templated rule should not be in term graph"

def test_namespace_stack_unaffected_by_templated_function():
    """Defining a templated function should not alter the namespace stack."""
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define("""
#lang:relnn
def Encoder<h, d>(Papers):
    Enc(pid; Linear(d, h, False)(z)) :- Papers(pid; z) .
enddef
""")

    assert s.engine.current_namespace == "global"
    assert s.engine.namespace_stack == ["global"]

# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_templated_transformdef_stored_in_symbol_table()
    test_templated_transformdef_not_added_to_tg()
    test_templated_transformdef_multi_param_stored()
    test_templated_transformdef_wrong_arg_count_raises()
    test_templated_functiondef_stored_in_symbol_table()
    test_templated_functiondef_body_not_processed_until_called()
    test_templated_functiondef_wrong_arg_count_raises()
    test_templated_rule_stored_in_symbol_table()
    test_templated_rule_not_added_to_tg()
    test_namespace_stack_unaffected_by_templated_function()
    print("All feature tests passed!")
