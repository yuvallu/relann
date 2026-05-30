"""Regression test for the HGT ``transformation_L1`` shape failure.

Background — see ``docs/design/`` and the sibling
``test_parameter_tensor_div_shape_bug.py``. The "Expected size for first two
dimensions of batch2 tensor to be: [1, 1] but got: [1, 4]" matmul error that
originally surfaced from ``tests/scaffold/test_913_scaffold_hgt_first_order``
was a *symptom* of the ``Tensor(d/h, d/h)`` shape bug — ``K @ W`` collapsed
to ``(1, 4) @ (4,) = (1,)`` because ``W`` was mis-sized to 1-D.

Fix (landed 2026-05-24): ``TensorTermCompiler._eval_hyperparams`` now coerces
computed integer-valued floats to int before they reach module ctors.

This file pins the end-to-end HGT attention pattern as a regression test.
"""
from __future__ import annotations

import pandas as pd
import torch

from relann.session import Session


def test_hgt_attention_one_head_completes_end_to_end():
    """The minimal HGT attention rule (was the "transformation_L1 batch2"
    failure) now runs to completion. Asserts only on shape — the random-init
    values aren't what we're testing here, only that the chain composes."""
    db = {
        "Authors_Raw":  (pd.DataFrame({"s": [0, 1]}), torch.ones(2, 334)),
        "Papers_Raw":   (pd.DataFrame({"s": [0, 1]}), torch.ones(2, 4231)),
        "Author_Paper": (
            pd.DataFrame({"s": [0, 1], "e": [0, 1], "t": [0, 1]}),
            torch.ones(2, 1),
        ),
    }
    session = Session(db=db)

    dsl = """
    d=16 .
    h=4 .

    K_Linear_Author = Linear(d, d/h) .
    Q_Linear_Paper  = Linear(d, d/h) .
    W_ATT_Author_Paper = Tensor(d/h, d/h) .
    Mu_AAPP = Tensor(1, 1) .

    Authors(s; Linear(334, d, False)(z)) :- Authors_Raw(s; z) .
    Papers(s;  Linear(4231, d, False)(z)) :- Papers_Raw(s; z) .

    L1(s, e, t; K_Linear_Author(z1) @ W_ATT_Author_Paper @ transpose(Q_Linear_Paper(z2)) * Mu_AAPP / sqrt(d)) :-
        Authors(s; z1), Author_Paper(s, e, t), Papers(t; z2) .

    ?pred Out(s, e, t; z) :- L1(s, e, t; z) .
    """

    result = session.run(dsl)

    # Per-edge attention score. Two edges in the dummy DB, so 2 rows.
    # The shape of each score depends on the exact chain — we just need to
    # check the chain composed without dying. Shape should be 2D or 3D
    # rooted in the edge count.
    assert result.embeddings is not None and len(result.embeddings) == 1
    emb = result.embeddings[0]
    assert emb.shape[0] == 2, f"expected 2 edges; got shape {tuple(emb.shape)}"
