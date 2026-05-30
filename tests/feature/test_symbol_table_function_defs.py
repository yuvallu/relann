"""Feature tests for symbol-table behavior around function definitions."""

import sys
from pathlib import Path

import pandas as pd
import torch
from relann.pydantic_classes import DerivedER, FunctionDef, TransformDef
from relann.session import Session
from relann.torch_utils import full_seed

def _minimal_papers_db(n: int = 6, d: int = 4):
    df = pd.DataFrame({"pid": list(range(n))})
    z = torch.randn(n, d)
    return {"Papers": (df, z)}

def test_symbol_table_registers_function_in_global():
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define(
        """
#lang:relnn
def Encoder(Papers):
    Enc(pid; ReLU(z)) :- Papers(pid; z) .
enddef
"""
    )

    assert "Encoder" in s.engine.symbol_table["global"]
    typ, obj = s.engine.symbol_table["global"]["Encoder"]
    assert typ is FunctionDef
    assert obj.name == "Encoder"

def test_symbol_table_function_namespace_has_body_symbols():
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define(
        """
#lang:relnn
d = 4 .
h = 2 .

def Encoder(Papers):
    LocalLin = Linear(d, h, False) .
    Hidden(pid; LocalLin(z)) :- Papers(pid; z) .
enddef
"""
    )

    fn_scope = s.engine.symbol_table["Encoder"]
    assert "LocalLin" in fn_scope
    assert "Hidden" in fn_scope
    assert fn_scope["LocalLin"][0] is TransformDef
    assert fn_scope["Hidden"][0] is DerivedER

def test_symbol_table_function_locals_do_not_leak_to_global():
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define(
        """
#lang:relnn
d = 4 .
h = 2 .

def Encoder(Papers):
    LocalLin = Linear(d, h, False) .
    Hidden(pid; LocalLin(z)) :- Papers(pid; z) .
enddef
"""
    )

    global_scope = s.engine.symbol_table["global"]
    assert "LocalLin" not in global_scope
    assert "Hidden" not in global_scope

def test_namespace_stack_returns_to_global_after_function_define():
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define(
        """
#lang:relnn
def Encoder(Papers):
    Enc(pid; ReLU(z)) :- Papers(pid; z) .
enddef
"""
    )

    assert s.engine.current_namespace == "global"
    assert s.engine.namespace_stack == ["global"]

def test_get_symbol_prefers_function_namespace_then_global():
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define(
        """
#lang:relnn
d = 4 .
h = 2 .
Shared = Linear(d, h, False) .

def Encoder(Papers):
    Shared = ReLU() .
    Enc(pid; Shared(z)) :- Papers(pid; z) .
enddef
"""
    )

    global_typ, global_obj = s.engine.get_symbol("Shared")
    local_typ, local_obj = s.engine.get_symbol("Shared", namespace="Encoder")

    assert global_typ is TransformDef
    assert local_typ is TransformDef
    assert global_obj.tensor_term.op.op == "Linear"
    assert local_obj.tensor_term.op.op == "ReLU"

def test_symbol_table_multiple_functions_registered_and_isolated():
    full_seed(42)
    s = Session(db=_minimal_papers_db())
    s.define(
        """
#lang:relnn
d = 4 .
h = 3 .
c = 2 .

def Encoder(Papers):
    Enc(pid; Linear(d, h, False)(z)) :- Papers(pid; z) .
enddef

def Classifier(Papers):
    Hidden(pid; z) :- Encoder(Papers)(pid; z) .
    Logits(pid; Linear(h, c, False)(z)) :- Hidden(pid; z) .
enddef
"""
    )

    global_scope = s.engine.symbol_table["global"]
    assert global_scope["Encoder"][0] is FunctionDef
    assert global_scope["Classifier"][0] is FunctionDef

    encoder_scope = s.engine.symbol_table["Encoder"]
    classifier_scope = s.engine.symbol_table["Classifier"]

    assert "Enc" in encoder_scope
    assert "Hidden" not in encoder_scope
    assert "Logits" not in encoder_scope

    assert "Hidden" in classifier_scope
    assert "Logits" in classifier_scope
    assert "Enc" not in classifier_scope

def test_symbol_table_when_same_function_called_twice_keeps_definition_scope_clean():
    full_seed(42)
    n, d = 6, 4
    papers_a_df = pd.DataFrame({"pid": list(range(n))})
    papers_b_df = pd.DataFrame({"pid": list(range(n))})
    papers_a_z = torch.randn(n, d)
    papers_b_z = torch.randn(n, d)
    s = Session(db={"PapersA": (papers_a_df, papers_a_z), "PapersB": (papers_b_df, papers_b_z)})
    s.define(
        """
#lang:relnn
d_in = 4 .
h = 3 .

def Echo(Papers):
    LocalLin = Linear(d_in, h, False) .
    EchoOut(pid; LocalLin(z)) :- Papers(pid; z) .
enddef

Pair(pid; Concat(z1, z2)) :- Echo(PapersA)(pid; z1), Echo(PapersB)(pid; z2) .
"""
    )

    global_scope = s.engine.symbol_table["global"]
    echo_scope = s.engine.symbol_table["Echo"]

    # Global stores only the function symbol and derived outputs, not per-call aliases.
    assert global_scope["Echo"][0] is FunctionDef
    assert global_scope["Pair"][0] is DerivedER
    assert "Echo_of_PapersA" not in global_scope
    assert "Echo_of_PapersB" not in global_scope
    assert not any(name.startswith("Echo_of_") for name in global_scope)

    # Function namespace remains definition-scoped (single LocalLin/EchoOut entries).
    assert echo_scope["LocalLin"][0] is TransformDef
    assert echo_scope["EchoOut"][0] is DerivedER
    assert not any(name.startswith("Echo_of_") for name in echo_scope)
