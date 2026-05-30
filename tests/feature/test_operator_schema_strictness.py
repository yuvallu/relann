"""Schema-strictness invariant tests for the era operators.

For every operator that returns an ``EmbeddedRelation``, the output's
``content.columns`` must equal ``content_schema`` (as a set). A mismatch
means we either:
  - dropped a column the schema still names (downstream KeyError);
  - kept a column the schema doesn't name (silent column leak — bytes
    leaving the operator without being addressable by downstream rules).

The Join operator gets this contract for free via the new look-ahead
drop logic (see ``Join._do_one_merge``'s ``future_keys`` parameter and
the regression tests in ``tests/repro/test_join_chain_column_bug.py``).
Transformation, Aggregation, and Union should hold the same invariant
on their outputs; that's what this file pins.

When extending the engine with a new operator, add a matching test here.
"""
from __future__ import annotations

import pandas as pd
import torch

from relann.session import Session


def _content_set(result):
    return set(result.content.columns)


def _schema_set(result):
    return set(result.content_schema)


def test_transformation_schema_strictness():
    """Transformation is just a per-row transform of one input ER.
    Output columns must equal the LHS-declared content attrs (its schema)."""
    db = {"Input": (pd.DataFrame({"i": [0, 1]}), torch.ones(2, 4))}
    session = Session(db=db)
    result = session.run("""
    ?pred Out(i; Linear(4, 4)(z)) :- Input(i; z) .
    """)
    assert _content_set(result) == _schema_set(result), (
        f"Transformation leaked or dropped columns. "
        f"content.columns={sorted(_content_set(result))}, "
        f"content_schema={sorted(_schema_set(result))}"
    )


def test_join_schema_strictness():
    """Two-way Join: output schema is the union of the joined keys.
    Pins the Join cleanup we landed earlier in this PR."""
    db = {
        "A": (pd.DataFrame({"x": [1, 2], "y": [10, 20]}), torch.ones(2, 4)),
        "B": (pd.DataFrame({"x": [1, 2], "z": [100, 200]}), torch.ones(2, 4)),
    }
    session = Session(db=db)
    result = session.run("""
    ?pred Out(x, y, z; w) :- A(x, y; w1), B(x, z; w2) .
    """)
    assert _content_set(result) == _schema_set(result), (
        f"Join leaked or dropped columns. "
        f"content.columns={sorted(_content_set(result))}, "
        f"content_schema={sorted(_schema_set(result))}"
    )


def test_aggregation_schema_strictness():
    """Aggregation collapses on a key. Output schema = the key plus
    the aggregated embedding's logical name."""
    db = {"Edge": (pd.DataFrame({"src": [0, 0, 1, 1], "dst": [1, 2, 0, 2]}), torch.ones(4, 4))}
    session = Session(db=db)
    result = session.run("""
    ?pred OutDeg(src; sum(w)) :- Edge(src, dst; w) .
    """)
    assert _content_set(result) == _schema_set(result), (
        f"Aggregation leaked or dropped columns. "
        f"content.columns={sorted(_content_set(result))}, "
        f"content_schema={sorted(_schema_set(result))}"
    )


def test_union_schema_strictness():
    """Union concatenates two ERs of the same shape. Output schema is shared."""
    db = {
        "A": (pd.DataFrame({"i": [0, 1]}), torch.ones(2, 4)),
        "B": (pd.DataFrame({"i": [2, 3]}), torch.ones(2, 4)),
    }
    session = Session(db=db)
    result = session.run("""
    ?pred Out(i; z) :- A(i; z) | B(i; z) .
    """)
    assert _content_set(result) == _schema_set(result), (
        f"Union leaked or dropped columns. "
        f"content.columns={sorted(_content_set(result))}, "
        f"content_schema={sorted(_schema_set(result))}"
    )
