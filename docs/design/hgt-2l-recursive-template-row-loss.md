# HGT 2-layer recursive-template row-loss bug

> Long-standing engine bug. **Not** a refactor regression. Pre-dates
> juplit (verified on `main` at `5656608` and on `claude/v2-next-version`).
> Captured 2026-05-30 after the Softmax stability fix made the bug
> standalone-observable.

## TL;DR

The generic templated HGT library (`run_compare_dblp_hgt_generic.py`'s
`HGT_LIBRARY`) has a `def H<tt, L>(): ... :- ..., H<tt, L-1>(t; z2) .`
recursive specialization. At `L=1` everything works (`H<*, 1>` rows
match `H<*, 0>` rows). **At `L=2` only node types with multiple
incoming MetaRel edges keep their rows** — single-source node types
collapse:

```
H<Author    , 0>: 4057 rows   H<Author    , 1>: 4057   H<Author    , 2>:     1   ← BUG
H<Paper     , 0>: 14328       H<Paper     , 1>: 14328  H<Paper     , 2>: 14328   ← OK (3 sources)
H<Term      , 0>: 7723        H<Term      , 1>: 7723   H<Term      , 2>:     5   ← BUG
H<Conference, 0>: 20          H<Conference, 1>: 20     H<Conference, 2>:     1   ← BUG
```

With near-empty `H<'Author', 2>`, the loss is computed on ≤1 row and
the engine reports `Non-finite loss encountered at epoch 0: nan`.

## Setup that reproduces

```powershell
uv run python tests/slow/run_hgt_generic_2l_accuracy.py
```

Crashes at epoch 0 with `RuntimeError: Non-finite loss encountered at epoch 0: nan`.

The 1-layer counterpart (`run_compare_dblp_hgt_generic.py`) **passes**
including the param-count, weight-synced forward, direct-first-order
forward, and 2%-accuracy assertions. So the issue is specifically the
recursion path, not the templated infrastructure as a whole.

## What's happening

DBLP's `MetaRel` table has 6 entries:

| ts          | pe              | tt          |
|-------------|-----------------|-------------|
| Author      | AuthorPaper     | Paper       |
| Paper       | PaperAuthor     | Author      |
| Paper       | PaperTerm       | Term        |
| Term        | TermPaper       | Paper       |
| Paper       | PaperConference | Conference  |
| Conference  | ConferencePaper | Paper       |

For `tt='Paper'` the inner bounded set
`Set(EdgeAgg<L, ts, pe, Paper> | MetaRel(ts, pe, Paper))` has **3
elements** (rows from Author / Term / Conference). For
`tt ∈ {Author, Term, Conference}` it has **1 element** each.

At `L=1` the recursive base case `H<tt, 0>(t; z)` is a plain
ER-from-data-source, no template recursion. At `L=2` the body's
`H<tt, L-1>` resolves to `H<tt, 1>` — another template specialization
that has its own `Union(Set(... | MetaRel(ts, pe, tt)))`.

The observed pattern says **the single-element Union inside a recursive
template specialization at depth > 1 collapses to one row**, while the
3-element Union (Paper case) preserves all rows. Multi-source Unions
behave correctly; single-source ones don't.

Symmetric layer-1 calls work, so the failure mode is somewhere in the
engine's handling of *single-element* `Union(Set(... | filter))` when
the body itself is *another* template specialization. Bounded-set
expansion at the inner-template boundary is the prime suspect.

## Where in the engine to look (hypotheses, unverified)

1. **`Engine._expand_bounded_set`** (`relann/engine.py`) — single-element
   set expansion may take a different codepath than multi-element and
   misroute the filter-substitution under nested templates.
2. **`Engine._materialize_template_reference`** / `_materialize_function_call`
   — when a recursive template specialization is materialized for
   `H<'Author', 2>`, the inner `H<'Paper', 1>` call may carry a stale
   `ts/pe/tt` symbol binding that drops most rows.
3. **`Union` operator in `relann/era_operations.py`** — could be
   silently de-duplicating rows by some key that's identical across rows
   when the underlying single-element source is materialized inside a
   nested specialization context.

A focused investigator should monkey-patch
`EmbeddedRelation.__init__` to dump row counts at every node during
`engine.fit`'s instantiate phase and bisect the chain from the failing
`H<'Author', 2>` upwards. The earlier diagnostic that found duplicate
column names for the V2 optimizer (see
`docs/design/optimizer-homogeneous-hgt-bug.md`) gives a template for the
instrumentation.

## Workaround

Not in scope for this branch. Users who need a 2-layer HGT today should
use the non-templated reference at
`tests/slow/run_compare_dblp_original_hgt.py` (PA-only baseline) or
hand-roll the 2L body without the templated recursion. The
critical-path paper benchmarks (`run_hgt_table1.py`,
`run_match_hgt_accuracy.py`, `run_match_hgt_2l_accuracy.py`) **do not**
use the recursive templated path and remain green.

## Test pin

`tests/repro/test_hgt_2l_recursive_template_row_loss.py` (added in this
branch) pins the symptom: it constructs the minimal templated 2-layer
HGT setup, runs the forward pass, and asserts `H<'Author', 2>` has the
expected row count. The test is marked `@pytest.mark.xfail(strict=True)`
so it remains green today (proving the bug still exists) and will go
red the moment the engine bug is fixed, prompting the fixer to unmark
the xfail and re-run the full slow suite.

## Adjacent fix that DID land (Softmax stability)

Investigating this bug surfaced an unrelated **numerical-stability**
issue in `HGT_LIBRARY`'s `def Softmax(Scores)`: it used the naive
`exp(z) / sum(exp(z))` form instead of subtracting the per-target max
before `exp`. The naive form overflows once attention scores grow
during 2-layer stacking, contributing inf/inf NaN; it also made the
1-layer `run_compare_dblp_hgt_generic.py`'s `Forward match (vs first-
order)` assertion fail (`max_diff = 3.02e-01`). With max-subtraction:

```
def Softmax(Scores):
    Max(t; max(z))            :- Scores(s, t; z) .
    Stable(s, t; z1 - z2)     :- Scores(s, t; z1), Max(t; z2) .
    Exp(s, t; exp(z))         :- Stable(s, t; z) .
    Denom(t; sum(z))          :- Exp(s, t; z) .
    Out(s, t; z1 / z2)        :- Exp(s, t; z1), Denom(t; z2) .
enddef
```

the 1L `forward (vs first-order)` assertion now reports `max_diff = 2.98e-08`
(matches PyTorch within ~1e-8) and the 1L test passes the full
assertion list (param-count, both forward-match comparisons, accuracy
within 2%). The same pattern is already used in the non-templated
reference at `tests/slow/run_compare_dblp_original_hgt.py:535-537` and
in `examples/004_relnn_hygnn.py:76`.

The stable Softmax change is independent of the recursive-template
bug. It is committed separately so the architectural-stability fix is
reviewable on its own.

## Recommendation for the eventual engine fix

When fixing, also:

1. Add an inline `if __name__ == "__main__":` self-test in
   `relann/engine.py` near the bounded-set expansion code that
   constructs a minimal recursive template and asserts row counts at
   each layer.
2. Ensure `tests/repro/test_hgt_2l_recursive_template_row_loss.py`'s
   xfail flips to expected-pass.
3. Re-run `tests/slow/run_hgt_generic_2l_accuracy.py` (5-seed sweep)
   and confirm 2-layer accuracy is in the published HGT ballpark
   (~76% test acc on DBLP, comparable to the hand-rolled PyTorch 2L
   reference at `run_match_hgt_2l_accuracy.py`).
