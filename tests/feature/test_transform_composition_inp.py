"""
Tests for transform composition using the inp formal input placeholder.

This tests that composed transforms using the inp placeholder work correctly.
The two-paren form (e.g., Dropout(0.1)(Linear(4, 8)(inp))) is the primary use case.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from relann.session import Session
from relann.torch_utils import full_seed

def _simple_db(n: int = 10, d: int = 4):
    """Simple input-output database."""
    df = pd.DataFrame({"id": range(n)})
    x = torch.randn(n, d)
    return {"Input": (df.copy(), x)}

def test_transform_composition_two_paren():
    """Test that Linear(d_in, d_out)(inp) and Dropout(p)(inp) resolve correctly.
    
    This uses the explicit two-paren form: ctor(hypers)(inp).
    """
    full_seed(42)
    db = _simple_db(n=10, d=4)
    session = Session(db=db)
    
    # Define a composed transform: Linear then Dropout, explicit two-paren form
    session.run("""
L_Drop = Dropout(0.1)(Linear(4, 8)(inp)) .
Out(id; L_Drop(z)) :- Input(id; z) .
""")
    
    # Run prediction to verify the composition worked
    result = session.run("?pred Result(id; z) :- Out(id; z) .")
    
    # Verify output shape: should be (10, 8) since Linear(4, 8) outputs 8 dims
    assert result is not None
    assert result.embeddings[0].shape == (10, 8), (
        f"Expected (10, 8), got {result.embeddings[0].shape}"
    )

def test_transform_multiple_inp_occurrences():
    """Test that multiple inp occurrences in a transform are all replaced.
    
    For example, Concat(Linear(d, d')(inp), Linear(d, d'')(inp)) should
    replace both inp leaves with the actual argument.
    """
    full_seed(42)
    db = _simple_db(n=10, d=4)
    session = Session(db=db)
    
    # Define a transform with two parallel paths through inp
    session.run("""
Out(id; Concat(Linear(4, 8)(inp), Linear(4, 8)(inp))) :- Input(id; z) .
""")
    
    result = session.run("?pred Result(id; z) :- Out(id; z) .")
    assert result is not None
    # Concat should produce (10, 16) = (10, 8) + (10, 8)
    assert result.embeddings[0].shape == (10, 16), (
        f"Expected (10, 16), got {result.embeddings[0].shape}"
    )

def test_transform_nested_transforms_with_inp():
    """Test that nested transform definitions both using inp work correctly.
    
    When Mu2 body references Mu1(inp), and Mu2 is later called with Mu2(z),
    the inner inp in Mu1's body should be replaced by the outer z.
    """
    full_seed(42)
    db = _simple_db(n=10, d=4)
    session = Session(db=db)
    
    # Define nested transforms with explicit two-paren forms
    session.run("""
Mu1 = Linear(4, 8)(inp) .
Mu2 = Dropout(0.1)(Mu1(inp)) .
Out(id; Mu2(z)) :- Input(id; z) .
""")
    
    result = session.run("?pred Result(id; z) :- Out(id; z) .")
    assert result is not None
    assert result.embeddings[0].shape == (10, 8)

def test_transform_deep_nesting():
    """Test deeply nested transforms: Dropout(ReLU(Linear(inp)))."""
    full_seed(42)
    db = _simple_db(n=10, d=4)
    session = Session(db=db)
    
    session.run("""
Deep = Dropout(0.1)(ReLU()(Linear(4, 8)(inp))) .
Out(id; Deep(z)) :- Input(id; z) .
""")
    
    result = session.run("?pred Result(id; z) :- Out(id; z) .")
    assert result is not None
    assert result.embeddings[0].shape == (10, 8)

# ── runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_transform_composition_two_paren()
    test_transform_multiple_inp_occurrences()
    test_transform_nested_transforms_with_inp()
    test_transform_deep_nesting()
    print("All transform composition tests passed.")
