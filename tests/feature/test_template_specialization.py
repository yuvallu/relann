"""Feature tests for template specialization (recursion base cases).

Tests cover:
- Specialization registration and dispatch
- Most-specific-match semantics
- Recursion depth limit / missing base case detection
- Multi-param specialization
- Same-type constraint enforcement
- End-to-end recursive rule / function with predict
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from relann.pydantic_classes import FunctionDef, Rule, TransformDef, Var
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

# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------

def test_specialization_registered_on_templated_rule():
    """Defining a templated rule should register it in _template_specializations."""
    full_seed(42)
    s = Session(db=_papers_db())
    s.define("""
#lang:relnn
Layer<d_out>(pid; Linear(8, d_out, False)(z)) :- Papers(pid; z) .
""")

    specs = s.engine._template_specializations
    assert "Layer" in specs
    assert len(specs["Layer"]) == 1
    pattern, typ, obj = specs["Layer"][0]
    assert pattern == [None]
    assert typ is Rule

def test_specialization_registered_on_templated_function():
    """Defining a templated function should register it in _template_specializations."""
    full_seed(42)
    s = Session(db=_papers_db())
    s.define("""
#lang:relnn
def Enc<d>(Papers):
    Out(pid; Linear(8, d, False)(z)) :- Papers(pid; z) .
enddef
""")

    specs = s.engine._template_specializations
    assert "Enc" in specs
    assert len(specs["Enc"]) == 1
    _, typ, _ = specs["Enc"][0]
    assert typ is FunctionDef

def test_specialization_registered_on_templated_transform():
    """Defining a templated TransformDef should register it in _template_specializations."""
    full_seed(42)
    s = Session(db=_papers_db())
    s.define("""
#lang:relnn
Lin<d> = Linear(d, d, False) .
""")

    specs = s.engine._template_specializations
    assert "Lin" in specs
    assert len(specs["Lin"]) == 1
    pattern, typ, _ = specs["Lin"][0]
    assert pattern == [None]
    assert typ is TransformDef

def test_multiple_specializations_registered():
    """Base case and general case should both be registered."""
    full_seed(42)
    s = Session(db=_gcn_db())
    s.define("""
#lang:relnn
H<0>(pid; Linear(8, 4, False)(z)) :- Papers(pid; z) .
H<L>(cited; sum(z * w)) :- H<L-1>(citing; z), Citation(citing, cited; w) .
""")

    specs = s.engine._template_specializations
    assert "H" in specs
    assert len(specs["H"]) == 2

    patterns = [p for p, _, _ in specs["H"]]
    assert [0] in patterns, "Base case pattern [0] should be registered"
    assert [None] in patterns, "General case pattern [None] should be registered"

# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------

def test_dispatch_base_case_selected():
    """Concrete args matching a base case should dispatch to it (not the general case)."""
    full_seed(42)
    s = Session(db=_gcn_db())
    s.define("""
#lang:relnn
H<0>(pid; Linear(8, 4, False)(z)) :- Papers(pid; z) .
H<L>(cited; sum(z * w)) :- H<L-1>(citing; z), Citation(citing, cited; w) .
""")

    typ, obj = s.engine._resolve_template_definition("H", [0])
    assert typ is Rule
    # Base case has concrete param 0, no Var
    tparams = obj.lhs.template_params
    assert len(tparams) == 1
    assert not isinstance(tparams[0], Var)

def test_dispatch_general_case_selected():
    """Non-matching concrete args should dispatch to the general (Var) case."""
    full_seed(42)
    s = Session(db=_gcn_db())
    s.define("""
#lang:relnn
H<0>(pid; Linear(8, 4, False)(z)) :- Papers(pid; z) .
H<L>(cited; sum(z * w)) :- H<L-1>(citing; z), Citation(citing, cited; w) .
""")

    typ, obj = s.engine._resolve_template_definition("H", [5])
    assert typ is Rule
    tparams = obj.lhs.template_params
    assert len(tparams) == 1
    assert isinstance(tparams[0], Var)

def test_dispatch_multi_param_most_specific():
    """With multi-param specialization, most-specific pattern wins.

    Note: ``H<Papers, 0>`` has template_params = [Var('Papers'), 0]
    because unquoted names are parsed as Var. So the pattern is [None, 0],
    not ['Papers', 0].  The ``0`` is the only concrete position.
    """
    full_seed(42)
    s = Session(db=_gcn_db())
    s.define("""
#lang:relnn
H<Papers, 0>(pid; Linear(8, 4, False)(z)) :- Papers(pid; z) .
H<T, L>(cited; sum(z * w)) :- H<T, L-1>(citing; z), Citation(citing, cited; w) .
""")

    specs = s.engine._template_specializations["H"]
    assert len(specs) == 2

    patterns = [p for p, _, _ in specs]
    assert [None, 0] in patterns, f"Expected [None, 0] pattern, got {patterns}"
    assert [None, None] in patterns, f"Expected [None, None] pattern, got {patterns}"

    # ('Papers', 0): pattern [None, 0] (specificity 1) beats [None, None] (specificity 0)
    typ, obj = s.engine._resolve_template_definition("H", ["Papers", 0])
    tparams = obj.lhs.template_params
    concrete_count = sum(1 for p in tparams if not isinstance(p, Var))
    assert concrete_count == 1  # just the 0

    # ('Papers', 3): only [None, None] matches (0 != 3)
    typ, obj = s.engine._resolve_template_definition("H", ["Papers", 3])
    tparams = obj.lhs.template_params
    var_count = sum(1 for p in tparams if isinstance(p, Var))
    assert var_count == 2

def test_dispatch_no_match_raises():
    """When no specialization matches the arity, raise ValueError."""
    full_seed(42)
    s = Session(db=_papers_db())
    s.define("""
#lang:relnn
H<L>(pid; Linear(8, 4, False)(z)) :- Papers(pid; z) .
""")

    with pytest.raises(ValueError, match="No specialization.*matches"):
        s.engine._resolve_template_definition("H", [1, 2])

def test_dispatch_definition_order_irrelevant():
    """Base case after general case should still dispatch correctly."""
    full_seed(42)
    s = Session(db=_gcn_db())
    # General case first, then base case
    s.define("""
#lang:relnn
H<L>(cited; sum(z * w)) :- H<L-1>(citing; z), Citation(citing, cited; w) .
H<0>(pid; Linear(8, 4, False)(z)) :- Papers(pid; z) .
""")

    # H<0> should still match base case
    typ, obj = s.engine._resolve_template_definition("H", [0])
    tparams = obj.lhs.template_params
    assert not isinstance(tparams[0], Var), "H<0> should dispatch to base case"

# ---------------------------------------------------------------------------
# Recursion safety tests
# ---------------------------------------------------------------------------

def test_missing_base_case_raises_recursion_error():
    """A recursive template without a base case should hit the depth limit."""
    full_seed(42)
    s = Session(db=_gcn_db())

    with pytest.raises(RecursionError, match="Missing base case"):
        s.run("""
#lang:relnn
H<L>(cited; sum(z * w)) :- H<L-1>(citing; z), Citation(citing, cited; w) .
Output(cited; z) :- H<3>(cited; z) .
""")

# ---------------------------------------------------------------------------
# Same-type constraint tests
# ---------------------------------------------------------------------------

def test_mixed_rule_and_function_specializations_allowed():
    """Defining H<0> as Rule and H<L> as FunctionDef should be allowed
    (needed for HGT: base case rules + recursive function defs)."""
    full_seed(42)
    s = Session(db=_gcn_db())
    s.define("""
#lang:relnn
H<0>(pid; Linear(8, 4, False)(z)) :- Papers(pid; z) .
def H<L>(Papers, Citation):
    Out(cited; sum(z * w)) :- Papers(citing; z), Citation(citing, cited; w) .
enddef
""")
    specs = s.engine._template_specializations["H"]
    types = {t.__name__ for _, t, _ in specs}
    assert "Rule" in types
    assert "FunctionDef" in types

# ---------------------------------------------------------------------------
# End-to-end recursive predict tests
# ---------------------------------------------------------------------------

def test_recursive_rule_1_layer_predict_shape():
    """H<0> = base, H<L> = recursive. H<1> should produce correct output shape."""
    full_seed(42)

    n, d_in, d_hidden = 6, 8, 4
    db = _gcn_db(n, d_in)
    session = Session(db=db)

    session.run(f"""
#lang:relnn
H<0>(pid; Linear({d_in}, {d_hidden}, False)(z)) :- Papers(pid; z) .
H<L>(cited; sum(z * w)) :- H<L-1>(citing; z), Citation(citing, cited; w) .

Output(cited; z) :- H<1>(cited; z) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(cited; z) :- Output(cited; z) .
""")

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n, d_hidden)

def test_recursive_rule_2_layers_predict_shape():
    """H<0> = base, H<L> = recursive. H<2> should produce correct output shape."""
    full_seed(42)

    n, d_in, d_hidden = 6, 8, 4
    db = _gcn_db(n, d_in)
    session = Session(db=db)

    session.run(f"""
#lang:relnn
H<0>(pid; Linear({d_in}, {d_hidden}, False)(z)) :- Papers(pid; z) .
H<L>(cited; sum(z * w)) :- H<L-1>(citing; z), Citation(citing, cited; w) .

Output(cited; z) :- H<2>(cited; z) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(cited; z) :- Output(cited; z) .
""")

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n, d_hidden)

def test_recursive_function_predict_shape():
    """Recursive FunctionDef with base case specialization."""
    full_seed(42)

    n, d_in, d_hidden = 6, 8, 4
    db = _gcn_db(n, d_in)
    session = Session(db=db)

    session.run(f"""
#lang:relnn
def GCN<0>(Papers, Citation):
    Out(pid; Linear({d_in}, {d_hidden}, False)(z)) :- Papers(pid; z) .
enddef

def GCN<L>(Papers, Citation):
    Prev(cited; z) :- GCN<L-1>(Papers, Citation)(cited; z) .
    Out(cited; sum(z * w)) :- Prev(citing; z), Citation(citing, cited; w) .
enddef

Output(cited; z) :- GCN<2>(Papers, Citation)(cited; z) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(cited; z) :- Output(cited; z) .
""")

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n, d_hidden)

# ---------------------------------------------------------------------------
# Backward compatibility: existing tests should still pass
# ---------------------------------------------------------------------------

def test_single_definition_still_works():
    """A single-definition template (no specialization) should work as before."""
    full_seed(42)

    n, d_in, d_out = 6, 8, 3
    session = Session(db=_papers_db(n, d_in))

    session.run(f"""
#lang:relnn
Lin<d_out> = Linear({d_in}, d_out, False) .
Output(pid; Lin<{d_out}>(z)) :- Papers(pid; z) .
""")

    result = session.run("""
#lang:relnn
?pred Predictions(pid; z) :- Output(pid; z) .
""")

    assert result is not None
    assert tuple(result.embeddings[0].shape) == (n, d_out)

# ---------------------------------------------------------------------------
# Quoted-string concrete param tests
# ---------------------------------------------------------------------------

def test_quoted_string_concrete_param_dispatch():
    """A definition with a quoted string literal (e.g. H<'Author', 0>) should
    register the string as a concrete pattern element and dispatch correctly."""
    full_seed(42)
    s = Session(db=_gcn_db())
    s.define("""
#lang:relnn
H<'Author', 0>(pid; Linear(8, 4, False)(z)) :- Papers(pid; z) .
H<T, L>(cited; sum(z * w)) :- H<T, L-1>(citing; z), Citation(citing, cited; w) .
""")

    specs = s.engine._template_specializations["H"]
    patterns = [p for p, _, _ in specs]
    # Quotes are stripped during pattern extraction for consistent dispatch.
    assert ["Author", 0] in patterns, f"Expected ['Author', 0] pattern, got {patterns}"
    assert [None, None] in patterns

    # ("'Author'", 0) matches base case (specificity 2) — quotes normalized.
    typ, obj = s.engine._resolve_template_definition("H", ["'Author'", 0])
    tparams = obj.lhs.template_params
    concrete_count = sum(1 for p in tparams if not isinstance(p, Var))
    assert concrete_count == 2

    # ("'Author'", 3) matches general case only
    typ, obj = s.engine._resolve_template_definition("H", ["'Author'", 3])
    tparams = obj.lhs.template_params
    assert all(isinstance(p, Var) for p in tparams)

    # Wrong arity raises
    with pytest.raises(ValueError, match="No specialization.*matches"):
        s.engine._resolve_template_definition("H", ["'Author'", 0, "extra"])

# ---------------------------------------------------------------------------
# Recursion depth counter reset tests
# ---------------------------------------------------------------------------

def test_depth_counter_zero_after_successful_materialization():
    """After a successful recursive materialization the depth counter should be 0."""
    full_seed(42)

    n, d_in, d_hidden = 6, 8, 4
    session = Session(db=_gcn_db(n, d_in))

    session.run(f"""
#lang:relnn
H<0>(pid; Linear({d_in}, {d_hidden}, False)(z)) :- Papers(pid; z) .
H<L>(cited; sum(z * w)) :- H<L-1>(citing; z), Citation(citing, cited; w) .

Output(cited; z) :- H<2>(cited; z) .
""")

    assert session.engine._template_materialization_depth == 0

def test_depth_counter_zero_after_recursion_error():
    """After a RecursionError the depth counter should be reset to 0."""
    full_seed(42)
    s = Session(db=_gcn_db())

    try:
        s.run("""
#lang:relnn
H<L>(cited; sum(z * w)) :- H<L-1>(citing; z), Citation(citing, cited; w) .
Output(cited; z) :- H<3>(cited; z) .
""")
    except RecursionError:
        pass

    assert s.engine._template_materialization_depth == 0

# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_specialization_registered_on_templated_rule()
    print("  PASS: test_specialization_registered_on_templated_rule")

    test_specialization_registered_on_templated_function()
    print("  PASS: test_specialization_registered_on_templated_function")

    test_specialization_registered_on_templated_transform()
    print("  PASS: test_specialization_registered_on_templated_transform")

    test_multiple_specializations_registered()
    print("  PASS: test_multiple_specializations_registered")

    test_dispatch_base_case_selected()
    print("  PASS: test_dispatch_base_case_selected")

    test_dispatch_general_case_selected()
    print("  PASS: test_dispatch_general_case_selected")

    test_dispatch_multi_param_most_specific()
    print("  PASS: test_dispatch_multi_param_most_specific")

    test_dispatch_no_match_raises()
    print("  PASS: test_dispatch_no_match_raises")

    test_dispatch_definition_order_irrelevant()
    print("  PASS: test_dispatch_definition_order_irrelevant")

    test_missing_base_case_raises_recursion_error()
    print("  PASS: test_missing_base_case_raises_recursion_error")

    test_mixed_rule_and_function_specializations_allowed()
    print("  PASS: test_mixed_rule_and_function_specializations_allowed")

    test_recursive_rule_1_layer_predict_shape()
    print("  PASS: test_recursive_rule_1_layer_predict_shape")

    test_recursive_rule_2_layers_predict_shape()
    print("  PASS: test_recursive_rule_2_layers_predict_shape")

    test_recursive_function_predict_shape()
    print("  PASS: test_recursive_function_predict_shape")

    test_single_definition_still_works()
    print("  PASS: test_single_definition_still_works")

    test_quoted_string_concrete_param_dispatch()
    print("  PASS: test_quoted_string_concrete_param_dispatch")

    test_depth_counter_zero_after_successful_materialization()
    print("  PASS: test_depth_counter_zero_after_successful_materialization")

    test_depth_counter_zero_after_recursion_error()
    print("  PASS: test_depth_counter_zero_after_recursion_error")

    print("\nAll template specialization tests passed!")
