"""Engine policy: ``Name<T>(...)`` on the RHS requires ``Name`` to have been
defined with template params. This file pins that policy.

Background
----------

The HGT scaffold ``tests/scaffold/test_913_scaffold_hgt_first_order.py``
exposed this guardrail when its DSL contained::

    L1_Paper_AGG_MSG(t; …) :- L1_Paper_ATT(…), L1_Paper_MSG(…) .
    Paper_Layer1_OUT(t; …) :- L1_Paper_AGG_MSG<T>(t; z1), Papers(t; z2) .
                                                 ^^^

The DEFINITION has no template params. The USE site passes template args.
The engine correctly rejects this with::

    ValueError: 'L1_Paper_AGG_MSG' is not a templated definition

Without this guardrail, a typo of this shape would compile to *something*
silently — almost certainly producing wrong semantics in a subtle way.
Better to fail loud.

This test pins the engine's strict policy. If a future commit relaxes it
(e.g. "silently strip ``<…>`` if target isn't templated"), this test
catches the regression.

See also: ``tests/scaffold/test_913_scaffold_hgt_first_order.py`` — the
scaffold whose WIP DSL exhibits this typo (8 instances). Documented but
not auto-fixed; the scaffold has been WIP since 2026-02-18 (commit
``4d11d81 "… WIP"``) and predates the juplit refactor by months.
"""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from relann.session import Session


def test_template_args_on_plain_definition_raise():
    """Define a plain ``A`` (no template params), then reference it as
    ``A<T>`` — engine must raise a ValueError with a helpful message."""
    session = Session(db={"X": (pd.DataFrame({"i": [0, 1]}), torch.ones(2, 4))})
    session.run("A(i; z) :- X(i; z) .")  # plain definition — no <…>

    with pytest.raises(ValueError, match=r"is not a templated definition"):
        # Reference with template args — should raise.
        session.run("B(i; z) :- A<T>(i; z) .")


def test_template_args_on_templated_definition_succeed():
    """Counter-case: when the definition *is* templated, the same call shape
    works. Anchors the other half of the contract so the policy can't drift
    in the opposite direction."""
    session = Session(db={"X": (pd.DataFrame({"i": [0, 1]}), torch.ones(2, 4))})
    # `<tag>` is a tag-only template param (declared but doesn't appear in
    # the body) — gives a separate cache entry per tag value.
    session.run("A<tag>(i; z) :- X(i; z) .")
    session.run("B(i; z) :- A<TagOne>(i; z) .")
    result = session.run("?pred Out(i; z) :- B(i; z) .")

    assert tuple(result.embeddings[0].shape) == (2, 4)
