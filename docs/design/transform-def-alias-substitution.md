# TransformDef alias substitution — β-reduction over inferred formals

> Companion to [join-chain-column-bug.md](join-chain-column-bug.md). This doc
> covers the *other* engine fix that landed in the same PR — the
> alias-over-alias TransformDef substitution rewrite. Read both if you're
> coming at the engine cold.

## The bug

DSL bodies like::

    Mu = L2(L1(x)) .

…used to compile as just ``L2(x)`` — the inner ``L1(x)`` call was silently
dropped. The downstream symptom in the failing dhn test was

    mat1 and mat2 shapes cannot be multiplied (1x1 and 20x10)

because ``L2 = Linear(20, 10)`` got a ``(1, 1)`` synthetic dummy input
directly, having skipped the ``(1, 1) → (1, 20)`` step that ``L1 = Linear(1, 20)``
was supposed to provide.

Root cause: ``replace_all_vars_in_tg_using_symbol_table`` substituted a
TransformDef call by looking up the body, then looked for a hardcoded
``Var("inp")`` placeholder to replace with the call argument. The grammar
comment at ``relann/relann_grammar.lark:74`` claimed this convention, but no
real ``.relnn`` file in the repo uses ``inp``. The actual DSL uses bare
identifiers like ``x``. The placeholder search missed them, and the engine
silently fell through to "the call's sons become the body's sons" — which
is correct only for bare unapplied ctors like ``K = Linear(1, 20)``, not for
nested compositions.

## The fix in one paragraph

A ``TransformDef`` is a λ-abstraction. Application is β-reduction. The fix
replaces the hardcoded ``"inp"`` placeholder with proper β-substitution over
*inferred* formal parameters: any ``Var`` leaf that survives in the *resolved*
body is a formal, and we substitute the call argument into it. The dispatch
is encapsulated in ``relann/engine.py::_apply_call_argument`` — see its
docstring for the case-split.

## Why infer formals from the resolved body instead of storing them at parse

This was the most subtle decision in the design and warrants its own
section.

The parser *could* compute formal_params at TransformDef construction time
and stash them on the data model. We tried this initially. It is wrong for
two reasons:

1. **Template materialization eliminates some Vars before application.**
   For a templated body like

       Mu<k, i> = Mu_L3<k, i>(ReLU()(Mu_L2<k, i>(ReLU()(Mu_L1<k, i>(x))))) .

   ``collect_formal_vars(body)`` at parse time returns ``['k', 'i', 'x']`` —
   all three are Var leaves. But ``k`` and ``i`` are *template* params, not
   runtime formals: a call like ``Mu<'C2', 0>(h_n)`` materializes them into
   literals (``'C2'`` and ``0``) before β-reduction. Only ``x`` survives as a
   runtime formal in the resolved body. Trusting the parse-time list would
   make us substitute the call argument into ``Var('k')`` — which by then
   doesn't exist.

2. **Symbol resolution eliminates other Vars too.** If a user writes
   ``Mu = L1(d)`` with ``d = 96 .`` as a global scalar TransformDef, the
   parser sees ``Var('d')`` in ``Mu``'s body and would mark it as a formal.
   But ``replace_tensor_term``'s scalar-resolution pass (engine.py:2264-2271)
   replaces ``Var('d')`` with the literal ``96`` before substitution. The
   parse-time list says ``['d']``; the resolved body has no Vars at all.

So: by the time ``_apply_call_argument`` runs, the resolved body's surviving
``Var`` leaves are exactly the runtime formals — no more, no less. Computing
fresh from the resolved body is more reliable than maintaining a separate
parse-time view that needs to be kept in sync with resolution. We removed
the dead-data ``formal_params`` field accordingly (see commit history).

## Why filter ``{True, False, None}``

The Lark grammar emits Python's reserved literal names as bare ``Var``
leaves (``Var('True')``, ``Var('False')``, ``Var('None')``). They're
literals from the parser's perspective and get converted to Python primitives
downstream by ``_arith_term_bool_var_to_primitive``. From the formal-inference
perspective they look like Vars but aren't formals. The filter at
``pydantic_classes.py::_RESERVED_VAR_LITERAL_NAMES`` short-circuits them so
they don't get mis-treated as formals.

## Multi-formal bodies

Today's DSL is unary — every ``TransformDef`` takes (at most) one runtime
formal. A body with two distinct ``Var`` leaves after resolution is almost
always a typo. ``_apply_call_argument`` raises ``ValueError`` rather than
silently leaving the extras unbound, which would otherwise produce confusing
downstream errors. When/if we extend the DSL to multi-arg TransformDefs, this
guard becomes a single-line change to loop over ``zip(formals, call_sons)``
and a grammar update for the call site.

## Known edge case

If a user writes ``d = 4.0 .`` (a scalar TransformDef whose body is a
*float literal*) and uses ``Tensor(d, d)``, the resolved substitution still
presents the value as a literal float-shaped ArithTerm and the
``_ParameterTensor`` fill-value heuristic returns (see
[join-chain-column-bug.md](join-chain-column-bug.md) and
``_coerce_computed_int_float``). Today nobody writes scalar TransformDefs
with float literals (verified by grep at fix-time); documented but
unhandled.

## Tests pinning this behaviour

- ``tests/repro/test_c2_sparse_matmul_shape_mismatch.py`` (2 tests) — the
  ``Mu = L2(L1(x))`` pattern that originally surfaced the bug.
- ``tests/repro/test_engine_rejects_template_args_on_plain_def.py`` (2 tests)
  — the related engine policy "``Name<T>`` requires ``Name`` to be defined
  with template params".
- ``tests/repro/test_hgt_minimal_shape_mismatch.py`` — end-to-end coverage
  through ``Session.run``.
