"""Smoke test: define GCN as function and call it."""

import sys
from pathlib import Path

import pandas as pd
import torch
from relann.session import Session
from relann.torch_utils import full_seed

def test_gcn_function_def_and_call_predict_shape():
    """GCN function definition and call should run and return expected shape."""
    full_seed(42)

    n_nodes, in_channels, hidden_channels, out_channels = 6, 8, 4, 3

    papers_df = pd.DataFrame({"pid": list(range(n_nodes))})
    papers_z = torch.randn(n_nodes, in_channels)

    # Ring graph so every node appears at least once as `cited`.
    citing = list(range(n_nodes))
    cited = [(i + 1) % n_nodes for i in range(n_nodes)]
    citation_df = pd.DataFrame({"citing": citing, "cited": cited})
    citation_w = torch.ones(len(citation_df), 1)

    session = Session(db={"Papers": (papers_df, papers_z), "Citation": (citation_df, citation_w)})

    define_program = f"""
#lang:relnn
in_channels = {in_channels} .
hidden_channels = {hidden_channels} .
out_channels = {out_channels} .

def GCN(Papers, Citation):
    PapersEmb1(pid; Linear(in_channels, hidden_channels, False)(z)) :- Papers(pid; z) .
    PapersAgg1(cited; sum(z * w)) :- PapersEmb1(citing; z), Citation(citing, cited; w) .
    PapersAggNL_Layer1(cited; ReLU(z)) :- PapersAgg1(cited; z) .

    PapersEmb2(cited; Linear(hidden_channels, out_channels, False)(z)) :- PapersAggNL_Layer1(cited; z) .
    PapersAgg2(cited; sum(z * w)) :- PapersEmb2(citing; z), Citation(citing, cited; w) .
    Output(cited; ReLU(z)) :- PapersAgg2(cited; z) .
enddef

ModelOutput(cited; z) :- GCN(Papers, Citation)(cited; z) .
"""

    pred_program = """
#lang:relnn
?pred Predictions(cited; z) :- ModelOutput(cited; z) .
"""

    session.run(define_program)
    result = session.run(pred_program)

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n_nodes, out_channels)

def test_nested_function_defs_call_chain_predict_shape():
    """One function def calling another should compile and run."""
    full_seed(42)

    n_nodes, in_channels, hidden_channels, out_channels = 7, 6, 5, 2

    papers_df = pd.DataFrame({"pid": list(range(n_nodes))})
    papers_z = torch.randn(n_nodes, in_channels)
    session = Session(db={"Papers": (papers_df, papers_z)})

    define_program = f"""
#lang:relnn
in_channels = {in_channels} .
hidden_channels = {hidden_channels} .
out_channels = {out_channels} .

def Encoder(Papers):
    Enc(pid; Linear(in_channels, hidden_channels, False)(z)) :- Papers(pid; z) .
enddef

def Classifier(Papers):
    Hidden(pid; z) :- Encoder(Papers)(pid; z) .
    Logits(pid; Linear(hidden_channels, out_channels, False)(z)) :- Hidden(pid; z) .
enddef

ModelOutput(pid; z) :- Classifier(Papers)(pid; z) .
"""

    pred_program = """
#lang:relnn
?pred Predictions(pid; z) :- ModelOutput(pid; z) .
"""

    session.run(define_program)
    result = session.run(pred_program)

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n_nodes, out_channels)

def test_nested_function_defs_with_two_args_predict_shape():
    """Nested function defs with two ER args should materialize correctly."""
    full_seed(42)

    n_nodes, in_channels, hidden_channels, out_channels = 8, 6, 4, 3

    papers_df = pd.DataFrame({"pid": list(range(n_nodes))})
    papers_z = torch.randn(n_nodes, in_channels)
    citing = list(range(n_nodes))
    cited = [(i + 1) % n_nodes for i in range(n_nodes)]
    citation_df = pd.DataFrame({"citing": citing, "cited": cited})
    citation_w = torch.ones(len(citation_df), 1)

    session = Session(db={"Papers": (papers_df, papers_z), "Citation": (citation_df, citation_w)})

    define_program = f"""
#lang:relnn
in_channels = {in_channels} .
hidden_channels = {hidden_channels} .
out_channels = {out_channels} .

def BaseBlock(Papers, Citation):
    Emb(pid; Linear(in_channels, hidden_channels, False)(z)) :- Papers(pid; z) .
    Agg(cited; sum(z * w)) :- Emb(citing; z), Citation(citing, cited; w) .
enddef

def Head(Papers, Citation):
    Hidden(cited; z) :- BaseBlock(Papers, Citation)(cited; z) .
    Logits(cited; Linear(hidden_channels, out_channels, False)(z)) :- Hidden(cited; z) .
enddef

ModelOutput(cited; z) :- Head(Papers, Citation)(cited; z) .
"""

    pred_program = """
#lang:relnn
?pred Predictions(cited; z) :- ModelOutput(cited; z) .
"""

    session.run(define_program)
    result = session.run(pred_program)

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n_nodes, out_channels)

def test_function_def_called_twice_in_same_rule_concat_shape():
    """Calling the same function def twice in one rule should keep calls isolated."""
    full_seed(42)

    n_nodes, in_channels, hidden_channels = 9, 5, 3

    papers_a_df = pd.DataFrame({"pid": list(range(n_nodes))})
    papers_b_df = pd.DataFrame({"pid": list(range(n_nodes))})
    papers_a_z = torch.randn(n_nodes, in_channels)
    papers_b_z = torch.randn(n_nodes, in_channels)
    session = Session(db={"PapersA": (papers_a_df, papers_a_z), "PapersB": (papers_b_df, papers_b_z)})

    define_program = f"""
#lang:relnn
in_channels = {in_channels} .
hidden_channels = {hidden_channels} .

def Encoder(Papers):
    Enc(pid; Linear(in_channels, hidden_channels, False)(z)) :- Papers(pid; z) .
enddef

Pair(pid; Concat(z1, z2)) :- Encoder(PapersA)(pid; z1), Encoder(PapersB)(pid; z2) .
"""

    pred_program = """
#lang:relnn
?pred Predictions(pid; z) :- Pair(pid; z) .
"""

    session.run(define_program)
    result = session.run(pred_program)

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n_nodes, hidden_channels * 2)

def test_same_function_twice_with_different_inputs_keeps_inputs_separate():
    """Same function called twice with different ERs should preserve each input in its branch."""
    full_seed(42)

    n_nodes, d = 8, 4
    papers_a_df = pd.DataFrame({"pid": list(range(n_nodes))})
    papers_b_df = pd.DataFrame({"pid": list(range(n_nodes))})
    papers_a_z = torch.randn(n_nodes, d)
    papers_b_z = torch.randn(n_nodes, d)
    session = Session(db={"PapersA": (papers_a_df, papers_a_z), "PapersB": (papers_b_df, papers_b_z)})

    define_program = f"""
#lang:relnn
d = {d} .

def Echo(Papers):
    EchoOut(pid; z) :- Papers(pid; z) .
enddef

Pair(pid; Concat(z1, z2)) :- Echo(PapersA)(pid; z1), Echo(PapersB)(pid; z2) .
"""

    pred_program = """
#lang:relnn
?pred Predictions(pid; z) :- Pair(pid; z) .
"""

    session.run(define_program)
    result = session.run(pred_program)

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    emb = result.embeddings[0]
    assert tuple(emb.shape) == (n_nodes, d * 2)

    # Concat should keep PaperA in left half and PaperB in right half.
    torch.testing.assert_close(emb[:, :d], papers_a_z)
    torch.testing.assert_close(emb[:, d:], papers_b_z)

def test_global_transformdef_used_inside_function_body_predict_shape():
    """Global TransformDef should be usable inside a function body."""
    full_seed(42)

    n_nodes, in_channels, hidden_channels = 10, 4, 2

    papers_df = pd.DataFrame({"pid": list(range(n_nodes))})
    papers_z = torch.randn(n_nodes, in_channels)
    session = Session(db={"Papers": (papers_df, papers_z)})

    define_program = f"""
#lang:relnn
in_channels = {in_channels} .
hidden_channels = {hidden_channels} .

    GlobalLin = Linear(in_channels, hidden_channels, False) .

    def Block(Papers):
        Hidden(pid; GlobalLin(z)) :- Papers(pid; z) .
enddef

ModelOutput(pid; z) :- Block(Papers)(pid; z) .
"""

    pred_program = """
#lang:relnn
?pred Predictions(pid; z) :- ModelOutput(pid; z) .
"""

    session.run(define_program)
    result = session.run(pred_program)

    assert result is not None
    assert result.embeddings is not None and len(result.embeddings) == 1
    assert tuple(result.embeddings[0].shape) == (n_nodes, hidden_channels)

if __name__ == "__main__":
    test_gcn_function_def_and_call_predict_shape()
    test_nested_function_defs_call_chain_predict_shape()
    test_nested_function_defs_with_two_args_predict_shape()
    test_function_def_called_twice_in_same_rule_concat_shape()
    test_same_function_twice_with_different_inputs_keeps_inputs_separate()
    test_global_transformdef_used_inside_function_body_predict_shape()