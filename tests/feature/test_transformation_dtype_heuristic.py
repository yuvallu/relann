"""Regression tests for `Transformation._module_device_dtype` (V1.4).

The heuristic picks the dtype to cast float embeddings to before invoking
the compiled module. Pre-V1.4 it blindly used `son.embeddings[0].dtype` as
the fallback when the module had no parameters. This silently broke
multi-arg ops (e.g. CrossEntropyLoss) when the optimizer commuted a join
and the FIRST embedding became a Long target tensor — every subsequent
Float embedding got cast to Long, and downstream kernels crashed with
`log_softmax_lastdim_kernel_impl not implemented for 'Long'`.

V1.4 fix: scan `son.embeddings` for the first FLOAT-dtype embedding;
default to `torch.float32` when none exists. Tests directly drive
`Transformation` with hand-constructed embeddings to pin the contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from relann.embedded_relation import EmbeddedRelation
from relann.era_operations import Transformation

class _NoParamModule(nn.Module):
    """A nn.Module without parameters (mimics nn.CrossEntropyLoss)."""

    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return args[0]  # not exercised; we only inspect dtype probing

def _make_son(content_df: pd.DataFrame, embeddings: list) -> EmbeddedRelation:
    return EmbeddedRelation(
        content_schema=list(content_df.columns),
        content=content_df,
        embedding_shapes=[e.shape for e in embeddings],
        embeddings=embeddings,
    )

def test_dtype_heuristic_prefers_float_over_long():
    """When `son.embeddings = [Long, Float]` (e.g. after R1 commute on a
    Hidden+Labels join), the heuristic must return Float, not Long. Otherwise
    the downstream `if t.dtype.is_floating_point: t = t.to(mod_dtype)` cast
    would convert all Float embeddings to Long, breaking matmul / log_softmax."""
    n = 4
    df = pd.DataFrame({"a": range(n)})
    long_emb = torch.zeros(n, dtype=torch.long)
    float_emb = torch.zeros(n, 3, dtype=torch.float32)
    son = _make_son(df, [long_emb, float_emb])

    op = Transformation(transformation=_NoParamModule())
    device, dtype = op._module_device_dtype(son=son)
    assert dtype == torch.float32, (
        f"expected float32 (first float emb's dtype), got {dtype}"
    )

def test_dtype_heuristic_prefers_float_when_first_is_float():
    """The natural pre-commute order [Float, Long] also resolves to Float."""
    n = 4
    df = pd.DataFrame({"a": range(n)})
    float_emb = torch.zeros(n, 3, dtype=torch.float32)
    long_emb = torch.zeros(n, dtype=torch.long)
    son = _make_son(df, [float_emb, long_emb])

    op = Transformation(transformation=_NoParamModule())
    device, dtype = op._module_device_dtype(son=son)
    assert dtype == torch.float32

def test_dtype_heuristic_all_long_falls_back_to_float32():
    """Pathological input: all embeddings are Long. Default to float32 — the
    `if t.dtype.is_floating_point` cast guard skips Long anyway, so float32
    is a safe sentinel that doesn't accidentally cast anything."""
    n = 4
    df = pd.DataFrame({"a": range(n)})
    e1 = torch.zeros(n, dtype=torch.long)
    e2 = torch.zeros(n, 2, dtype=torch.long)
    son = _make_son(df, [e1, e2])

    op = Transformation(transformation=_NoParamModule())
    device, dtype = op._module_device_dtype(son=son)
    assert dtype == torch.float32

def test_dtype_heuristic_picks_param_dtype_when_module_has_params():
    """When the module has parameters, prefer them — the parameter's dtype is
    authoritative (the matmul will use it). This matches V1.3 behavior."""
    n = 4
    df = pd.DataFrame({"a": range(n)})
    long_emb = torch.zeros(n, dtype=torch.long)
    son = _make_son(df, [long_emb])

    linear = nn.Linear(3, 3, dtype=torch.float64)  # use float64 to differ
    op = Transformation(transformation=linear)
    device, dtype = op._module_device_dtype(son=son)
    assert dtype == torch.float64
