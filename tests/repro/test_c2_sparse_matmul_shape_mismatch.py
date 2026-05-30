"""Regression test for the ``transformation_C2_Sparse`` matmul shape mismatch.

This was a *symptom* of a deeper bug: the engine's `TransformDef` alias-body
substitution dropped nested alias calls. A body like
``Mu = L2(L1(x))``, when invoked as ``Mu(h_n)``, was being compiled as
``L2(h_n)`` — the inner ``L1`` was silently discarded. With ``L1 =
Linear(1, 20)`` and ``L2 = Linear(20, 10)``, the dummy probe inside
``Transformation.instantiate`` fed a shape-``(1, 1)`` synthetic input
directly to ``Linear(20, 10)`` → ``mat1 and mat2 shapes cannot be multiplied
(1x1 and 20x10)``.

Root cause: ``Join._do_one_merge`` had nothing to do with it; the suffix fix
was a separate concern. The actual fix is in ``engine.py``: replace the
hard-coded ``Var('inp')`` placeholder with proper β-reduction over inferred
formal parameters. Full design rationale in
``docs/design/transform-def-alias-substitution.md``; live code lives in
``relann/engine.py::_apply_call_argument`` / ``_inject_formal_param``.

These tests pass positively (no ``pytest.raises``) and document the post-fix
contract. If any starts failing, the regression is in the alias-substitution
path — check ``replace_all_vars_in_tg_using_symbol_table`` and
``collect_formal_vars``.
"""
from __future__ import annotations

import pandas as pd
import torch

from relann.session import Session


def _build_min_db() -> dict:
    """Two graphs, two nodes each, one directed edge each. Node embeddings
    are size-1 (matches the dhn_ghl_csl_c2_4 setup of ``d_in = 1``)."""
    node_df = pd.DataFrame({"graph_id": [0, 0, 1, 1], "n": [0, 1, 0, 1]})
    node_emb = torch.tensor([[0.5], [1.0], [2.0], [3.0]])
    edge_df = pd.DataFrame({"graph_id": [0, 1], "n": [0, 0], "v": [1, 1]})
    edge_emb = torch.ones(len(edge_df), 1)
    return {
        "Node": (node_df, node_emb),
        "Edge": (edge_df, edge_emb),
    }


def test_c2_sparse_minimal_completes_after_alias_fix():
    """The minimal DSL that previously raised the C2_Sparse matmul error now
    runs to completion. Output is one row per (graph_id, n) — 4 rows total —
    each with the 10-dim embedding produced by ``Mu_L3 = Linear(20, 10)``."""
    db = _build_min_db()
    session = Session(db=db)

    dsl = """
    Mu_L1<k, i> = Linear(1, 20) .
    Mu_L2<k, i> = Linear(20, 20) .
    Mu_L3<k, i> = Linear(20, 10) .
    Mu<k, i> = Mu_L3<k, i>(ReLU()(Mu_L2<k, i>(ReLU()(Mu_L1<k, i>(x))))) .

    H0(graph_id, n; z) :- Node(graph_id, n; z) .
    C2_Sparse(graph_id, n; sum(Mu<'C2', 0>(h_n) * Mu<'C2', 1>(h_v))) :- Edge(graph_id, n, v; we), H0(graph_id, n; h_n), H0(graph_id, v; h_v) .
    ?pred Out(graph_id, n; z) :- C2_Sparse(graph_id, n; z) .
    """

    result = session.run(dsl)

    assert result.embeddings is not None and len(result.embeddings) == 1
    emb = result.embeddings[0]
    # 4 input edges aggregated by (graph_id, n) onto the source node of each edge.
    # The exact row count comes from the join+aggregate; the 10-width is what
    # actually matters for proving the Mu chain was preserved.
    assert emb.shape[-1] == 10, (
        f"Mu_L3 = Linear(20, 10) should produce a 10-dim output; got shape {tuple(emb.shape)}"
    )


def test_alias_over_alias_composition_preserves_inner_calls():
    """Direct probe of the bug: ``Mu = L2(L1(x))`` called as ``Mu(h_n)``
    must compile the full chain, not collapse to ``L2(h_n)``."""
    db = _build_min_db()
    session = Session(db=db)

    dsl = """
    L1 = Linear(1, 20) .
    L2 = Linear(20, 10) .
    Mu = L2(L1(x)) .

    H0(graph_id, n; z) :- Node(graph_id, n; z) .
    C2(graph_id, n; sum(Mu(h_n))) :- Edge(graph_id, n, v; we), H0(graph_id, n; h_n) .
    ?pred Out(graph_id, n; z) :- C2(graph_id, n; z) .
    """

    result = session.run(dsl)
    emb = result.embeddings[0]
    assert emb.shape[-1] == 10, (
        f"Mu = L2(L1(x)) where L2 = Linear(20, 10) should produce a 10-dim "
        f"output; got shape {tuple(emb.shape)}. If this fails, "
        f"engine.replace_all_vars_in_tg_using_symbol_table is dropping the "
        f"inner L1 call."
    )
