"""Regression pin: HGT 2-layer recursive template specialization drops
rows for any node type that has a single-element MetaRel bounded set.

Background:
    The generic templated HGT library in
    `tests/slow/run_compare_dblp_hgt_generic.py` defines

        def H<tt, L>():
            Agg(t; sum(z)) :- Union(Set(EdgeAgg<L, ts, pe, tt>(t; z)
                                         | MetaRel(ts, pe, tt))) .
            Updated(t; A_Lin<L, tt>(GELU(z))) :- Agg(t; z) .
            Out(t; Sigmoid(Skip<L, tt>) * z1 + (1 - Sigmoid(Skip<L, tt>)) * z2)
                :- Updated(t; z1), H<tt, L-1>(t; z2) .
        enddef

    For DBLP, four node types have these row counts at L=0:

        Author    : 4057    (1 incoming MetaRel: PaperAuthor)
        Paper     : 14328   (3 incoming MetaRel: AuthorPaper, TermPaper, ConferencePaper)
        Term      : 7723    (1 incoming: PaperTerm)
        Conference: 20      (1 incoming: ConferencePaper)

    At L=1 the recursive specialization works correctly: every node
    type's row count is preserved through the H<tt, 1> body. At L=2,
    however, single-source node types collapse:

        H<Paper     , 2> : 14328    [OK]
        H<Author    , 2> :     1    [BUG] (expected 4057)
        H<Term      , 2> :     5    [BUG] (expected 7723)
        H<Conference, 2> :     1    [BUG] (expected 20)

    With near-empty H<'Author', 2>, the loss is computed on ~1 row and
    the engine reports `Non-finite loss encountered at epoch 0: nan`.

    Pre-existing on `main` (verified at 5656608) and on
    `claude/v2-next-version`. NOT a juplit regression. See:
      docs/design/hgt-2l-recursive-template-row-loss.md
      docs/_archive/hgt-status-2026-05-25.md  (Cause C)

    The bug is in the engine's handling of single-element bounded-set
    expansion when the contained reference is itself another template
    specialization (i.e. recursion depth > 1). Multi-element bounded
    sets — e.g. Paper's 3-element MetaRel set at L=2 — are fine.

Why this is here, not next to the slow scripts:
    `tests/slow/run_hgt_generic_2l_accuracy.py` is a 5-seed,
    100-epoch performance sweep (~75s on CPU, longer on the runner
    for the DBLP slow path). This file is a one-shot xfail pin that
    runs in seconds: it loads the same database, runs the same
    DEFINE program, and asserts the row-count symptom without
    training. When the engine bug is fixed, this pin flips to
    pass-expected and signals the slow-script can be re-enabled.

Convention:
    `@pytest.mark.xfail(strict=True)` — the test currently FAILS
    (asserts the buggy 1-row state, marked as expected failure).
    When the engine bug is fixed, H<'Author', 2> will have 4057
    rows again and the assertion will pass, which `strict=True`
    turns into a TEST FAILURE so the fixer is forced to flip the
    pin from xfail to plain expect-pass. That's the desired ratchet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tests" / "slow"))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Engine bug: single-element Union(Set(...)) inside a recursive "
        "template specialization at depth > 1 collapses to one row. "
        "See docs/design/hgt-2l-recursive-template-row-loss.md."
    ),
)
def test_h_author_2_keeps_all_rows():
    """H<'Author', 2> should have 4057 rows (same as H<'Author', 0> and
    H<'Author', 1>). Currently the engine produces 1 row, which is the
    symptom of the recursive-template row-loss bug.

    The 1-layer pre-conditions are explicitly verified first so a future
    refactor that drops H<*, 1> row counts surfaces here rather than
    quietly altering this pin's semantics.
    """
    from relann.torch_utils import full_seed
    from relann.session import Session
    from run_compare_dblp_hgt_generic import (  # type: ignore[import]
        RELNN_GENERIC_DEFINE_2L, relnn_db,
    )

    full_seed(42)
    session = Session(db=relnn_db)
    session.run(RELNN_GENERIC_DEFINE_2L)
    # Materialize via a forward pass before querying intermediate ERs.
    session.run("""
#lang:relnn
?pred _Init(id; Classifier(z)) :- Output(id; z) .
""")

    def n_rows(layer: int) -> int:
        out = session.run(
            f"#lang:relnn\n?pred _H_Author_{layer}(id; z) :- H<'Author', {layer}>(id; z) ."
        )
        return 0 if out.content is None else len(out.content)

    # Sanity gate — the 1-layer path must be intact for this pin to be meaningful.
    assert n_rows(0) == 4057, "Test setup broken: H<'Author', 0> should expose 4057 rows"
    assert n_rows(1) == 4057, "H<'Author', 1> regressed (unexpected — was working pre-juplit)"

    # The real assertion this pin protects. Currently FAILS (engine produces 1).
    assert n_rows(2) == 4057, (
        f"H<'Author', 2> has wrong row count. "
        f"Engine row-loss bug at recursion depth > 1 — see "
        f"docs/design/hgt-2l-recursive-template-row-loss.md"
    )
