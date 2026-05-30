"""Smoke tests for session.relation() and session.params() inspection APIs."""

import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from relann.session import Session
from relann.torch_utils import full_seed
from relann.embedded_relation import EmbeddedRelation

def _make_session():
    full_seed(42)
    n, d_in = 6, 4
    df = pd.DataFrame({"a": range(n)})
    z = torch.randn(n, d_in)
    return Session(db={"Input": (df, z)}), n

def test_relation_before_run_raises():
    """session.relation() before any fit/predict should raise RuntimeError."""
    session, _ = _make_session()
    with pytest.raises(RuntimeError, match="No relation data available"):
        session.relation("Input")

def test_relation_after_predict():
    """session.relation() returns the intermediate ER after predict."""
    session, n = _make_session()

    session.run("""
#lang:relnn
Mid(a; Linear(4, 2)(z)) :- Input(a; z) .
""")
    session.run("""
#lang:relnn
?pred Out(a; Linear(2, 3)(z)) :- Mid(a; z) .
""")

    mid = session.relation("Mid")
    assert isinstance(mid, EmbeddedRelation)
    assert mid.embeddings is not None and len(mid.embeddings) == 1
    assert mid.embeddings[0].shape == (n, 2)

    out = session.relation("Out")
    assert isinstance(out, EmbeddedRelation)
    assert out.embeddings[0].shape == (n, 3)

def test_relation_nonexistent_raises():
    """session.relation() with unknown name raises KeyError listing available names."""
    session, _ = _make_session()

    session.run("""
#lang:relnn
Mid(a; Linear(4, 2)(z)) :- Input(a; z) .
""")
    session.run("""
#lang:relnn
?pred Out(a; Linear(2, 3)(z)) :- Mid(a; z) .
""")

    with pytest.raises(KeyError, match="Available relations"):
        session.relation("DoesNotExist")

def test_relation_after_fit():
    """session.relation() returns the intermediate ER after fit."""
    full_seed(42)
    n, d_in, d_out = 6, 4, 2
    df = pd.DataFrame({"a": range(n)})
    z = torch.randn(n, d_in)
    labels = torch.randint(0, d_out, (n,))
    session = Session(db={
        "Input": (df, z),
        "Labels": (df, labels),
    })

    session.run(f"""
#lang:relnn
Hidden(a; Linear({d_in}, {d_out})(z)) :- Input(a; z) .
?fit <epochs=3, lr=0.01> Loss(; CrossEntropyLoss()(z_pred, z)) :- Hidden(a; z_pred), Labels(a; z) .
""")

    hidden = session.relation("Hidden")
    assert isinstance(hidden, EmbeddedRelation)
    assert hidden.embeddings is not None and len(hidden.embeddings) == 1
    assert hidden.embeddings[0].shape == (n, d_out)

# ---------------------------------------------------------------------------
# session.params() tests
# ---------------------------------------------------------------------------

def test_params_before_run_raises():
    """session.params() before any fit/define should raise RuntimeError."""
    session, _ = _make_session()
    with pytest.raises(RuntimeError, match="No parameters available"):
        session.params()

def test_params_after_fit():
    """session.params() returns OrderedDict with expected keys after fit."""
    full_seed(42)
    n, d_in, d_out = 6, 4, 2
    df = pd.DataFrame({"a": range(n)})
    z = torch.randn(n, d_in)
    labels = torch.randint(0, d_out, (n,))
    session = Session(db={
        "Input": (df, z),
        "Labels": (df, labels),
    })

    session.run(f"""
#lang:relnn
Hidden(a; Linear({d_in}, {d_out})(z)) :- Input(a; z) .
?fit <epochs=3, lr=0.01> Loss(; CrossEntropyLoss()(z_pred, z)) :- Hidden(a; z_pred), Labels(a; z) .
""")

    p = session.params()
    assert isinstance(p, OrderedDict)
    assert len(p) > 0
    for name, param in p.items():
        assert isinstance(name, str)
        assert isinstance(param, nn.Parameter)
    weight_names = list(p.keys())
    assert any("weight" in n for n in weight_names)
    assert any("bias" in n for n in weight_names)

def test_params_after_predict():
    """session.params() returns OrderedDict after predict."""
    session, n = _make_session()

    session.run("""
#lang:relnn
Mid(a; Linear(4, 2)(z)) :- Input(a; z) .
""")
    session.run("""
#lang:relnn
?pred Out(a; Linear(2, 3)(z)) :- Mid(a; z) .
""")

    p = session.params()
    assert isinstance(p, OrderedDict)
    assert len(p) > 0
    for name, param in p.items():
        assert isinstance(name, str)
        assert isinstance(param, nn.Parameter)
