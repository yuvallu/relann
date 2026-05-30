"""Pins a *separate*, pre-existing bug surfaced once the alias-substitution +
Join column-tracking fixes let the chain progress further.

This is **not** a Join engine bug. The Join is constructed with malformed
metadata by some earlier stage (parser / term-graph / bounded-set expansion):

    output_schema = ['t', 't_0', 't_1', 't_2']
    input_schemas = [['t', 't'], ['t', 't'], ['t', 't']]
    merge_steps   = [
        {'step': 1,
         'left_refs':  [ColumnRef(0, 0), ColumnRef(0, 0)],   # same ref twice
         'right_refs': [ColumnRef(1, 0), ColumnRef(1, 0)],
         'key_names':  ['s', 't']},
        {'step': 2, 'left_refs': [ColumnRef(1, 0), ColumnRef(1, 0)], ...},
    ]

Symptoms when you run ``tests/slow/run_hgt_template_cora.py``::

    RelNNNodeError: RelNN evaluation failed at node 'join_98'
                    (mode=instantiate, op=Join):
                    'DataFrame' object has no attribute 'dtype'

The Join is doing the right thing — it asks ``df_joined['t']`` to coerce dtypes
before merging; because the dataframe has TWO columns named ``'t'`` (the
malformed input_schemas literally said so), pandas returns a 2-column
DataFrame, and ``.dtype`` blows up. The real fix lives upstream in whatever
constructs the Join's input_schemas / merge_steps for this rule.

This test reproduces the symptom by hand-building a Join with the same
malformed metadata, no HGT scaffolding required. It runs in <1s.

When the upstream construction bug is fixed (likely in the parser or in
``EmbeddedRelation`` schema validation), this repro will start to either
succeed-with-different-shape or fail with a *clearer* error than the dtype
crash. Flip the assertion accordingly at that point.
"""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from relann.column_ref import ColumnRef
from relann.embedded_relation import EmbeddedRelation
from relann.era_operations import Join


def _make_er_with_dup_cols() -> EmbeddedRelation:
    """An ER whose *content* dataframe has the bad schema [t, t]."""
    df = pd.DataFrame({"t": [1, 2]})
    df["t_dup"] = [10, 20]
    df.columns = ["t", "t"]            # duplicate column names — what _prepare_dfs would produce
    return EmbeddedRelation(
        content_schema=["t", "t"],
        embedding_shapes=[torch.Size([2, 1])],
        content=df,
        embeddings=[torch.tensor([[1.0], [2.0]])],
    )


def test_join_with_duplicate_input_schemas_crashes_with_dataframe_no_dtype():
    """Reproduces the third Join-engine wall (post alias-substitution +
    Option A) in <1s, with no HGT scaffolding. The Join receives the
    malformed metadata directly and crashes the same way."""
    sons = [_make_er_with_dup_cols(), _make_er_with_dup_cols(), _make_er_with_dup_cols()]
    join = Join(
        output_schema=["t", "t_0", "t_1", "t_2"],
        merge_steps=[
            {"step": 1,
             "left_refs":  [ColumnRef(0, 0), ColumnRef(0, 0)],
             "right_refs": [ColumnRef(1, 0), ColumnRef(1, 0)],
             "key_names":  ["s", "t"]},
            {"step": 2,
             "left_refs":  [ColumnRef(1, 0), ColumnRef(1, 0)],
             "right_refs": [ColumnRef(2, 0), ColumnRef(2, 0)],
             "key_names":  ["s", "t"]},
        ],
        input_schemas=[["t", "t"], ["t", "t"], ["t", "t"]],
    )
    with pytest.raises(AttributeError, match=r"'DataFrame' object has no attribute 'dtype'"):
        join.instantiate(sons)
