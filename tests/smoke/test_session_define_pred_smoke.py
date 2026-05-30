"""Deterministic smoke test for define + predict flow."""

import sys
from pathlib import Path

import pandas as pd
import torch
from relann.session import Session
from relann.torch_utils import full_seed

def test_session_define_pred_linear_shape():
    """Basic define/predict path should return expected embedding shape."""
    full_seed(42)
    n, d_in, d_out = 6, 4, 2
    df = pd.DataFrame({"a": range(n)})
    z = torch.randn(n, d_in)
    session = Session(db={"Input": (df, z)})

    define = """
#lang:relnn
Out(a; Linear(4, 2)(z)) :- Input(a; z) .
"""
    pred = """
#lang:relnn
?pred Result(a; z) :- Out(a; z) .
"""

    session.run(define)
    result = session.run(pred)

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n, d_out)

