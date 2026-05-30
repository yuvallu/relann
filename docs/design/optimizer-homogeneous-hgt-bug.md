# Optimizer V2 — homogeneous-HGT ColRef bug

> **Audience: anyone re-introducing `optimizer/` or `optimizer_v2/` in PR #56 or later.**
> This bug is the reason the optimizer was deleted from `juplit` (PR #61).
> Without the fix described here, the same crash recurs the moment the
> optimizer is wired back in.

## TL;DR

`optimizer_v2/lift.py::OptimizerLift._emit_agg` mis-resolves ColRefs when a
templated **FunctionDef** is instantiated with an ER argument whose schema is
a strict subset of the body's expected schema (the **homogeneous-HGT** case —
source and target node sets coincide, e.g. Cora layer 2). The lift produces
an `Aggregation` node whose `group_by_refs` contains duplicates; the
downstream `Aggregation.instantiate` constructs an `EmbeddedRelation` with
`content_schema=['t', 't']`; the downstream `Join._do_one_merge` then does
`df_joined['t'].dtype` which (because pandas allows duplicate column names)
returns a `DataFrame`, not a `Series` — crash:

```
AttributeError: 'DataFrame' object has no attribute 'dtype'
```

Surfaces at `relann/era_operations.py:607` (the dtype access inside
`Join._do_one_merge`), but the originating bug is in V2's lift.

Confirmed cross-platform: reproduces on Linux+CUDA *and* Windows+CPU on
the juplit HEAD; `main` (at `53316a04`) **passes** the same script per the
2026-05-25 audit in `docs/_archive/hgt-status-2026-05-25.md`. So this is a
real regression that landed during the juplit refactor — not a pre-existing
issue main always had.

## How to reproduce (after re-introducing the optimizer)

```powershell
uv run python tests/slow/run_hgt_template_cora.py
```

The script runs Test 1 (single-layer templated HGT — Levels 1 + 2) and
Test 2 (two-layer templated HGT — Level 3, FunctionDef). Test 1 passes;
Test 2 crashes mid-instantiation at the dtype access.

The 2-layer program is at `tests/slow/run_hgt_template_cora.py:118–147`.
Key pattern: `L1(t; z) :- HGTLayer<1>(Papers_Emb, Citation)(t; z)` then
`Output(t; z) :- HGTLayer<2>(L1, Citation)(t; z)`. **The L1 ER has schema
`(t)` only — one column** — but the function body uses `InputNodes(s; z1)`
*and* `InputNodes(t; z2)`. That's the homogeneous case: source and target
of layer 2 are both Papers IDs, so the engine aliases `s → t`.

## Root-cause trace (from the failure backward)

Verified by instrumentation on the juplit HEAD at `6de90f4`.

1. **Engine graph is correct.** The `add_rule` path in
   `term_graph.py:486–491` adds an AGG node with the right
   `group_by_refs` (e.g. `[(0,0), (0,1)]` for a layer-2 head's
   `Head<head>(s, t; …)` LHS). The `_materialize_function_call` at
   `engine.py:1149` deep-copies it verbatim into the invocation graph,
   keeping the refs clean.

2. **V2 ingest is correct.** `optimizer_v2/ingest.py::_ingest_agg`
   (the `_ingest_agg` function — was at `ingest.py:642–680`)
   builds the e-graph `gb_vec` from the original refs:
   ```python
   group_by_refs = d.get("group_by_refs") or []
   group_by_cols = [parent_schema[ref.column_idx] for ref in group_by_refs]
   gb_vec = Vec(*group_by_cols)
   ```
   Output `gb_vec` faithfully reflects the original (clean) refs.

3. **V2 lift mis-resolves.** `optimizer_v2/lift.py::OptimizerLift._emit_agg`
   (was at `lift.py:332–393`) iterates `gb_vec.children` and looks up each
   surface name in the *child schema*:
   ```python
   child_schema = _term_schema_names(term.children[0], self.eg)
   for gb_term in gb_vec.children:
       col_name = self._colref_name(gb_term)
       if col_name in child_schema:
           idx = child_schema.index(col_name)
       elif self.name_aliases is not None:
           # alias-aware fallback
           target_canon = self.name_aliases.find(col_name)
           for i, c in enumerate(child_schema):
               if self.name_aliases.find(c) == target_canon:
                   idx = i; break
       ...
       group_by_refs.append(ColumnRef(0, idx))
       group_by_names.append(col_name)
   ```
   For layer 2's homogeneous case, the **alias-aware fallback** collapses
   both `s` and `t` to the *same* idx — the lift produces
   `group_by_refs=[(0,1), (0,1)]` and `output_schema=['t', 't']`.

4. **Aggregation propagates the dup.** `era_operations.py:2048–2063`
   (`Aggregation.instantiate`) reads `group_by_refs`, resolves them to
   column names via `Join._resolve_normalized_refs`, then sets
   `output_schema = list(keys)` verbatim — so the produced `EmbeddedRelation`
   has `content_schema=['t', 't']`.

5. **Join chokes.** Next Join's `_do_one_merge` does `df_joined['t'].dtype`,
   pandas returns a 2-column DataFrame (because the upstream ER had dup
   columns), `.dtype` doesn't exist on a DataFrame → AttributeError.

The full evidence chain is captured in the `juplit` session transcript
(in the local agent session logs).

## What would need to be true for the lift to be correct

Two valid framings, either fix is acceptable:

### Option R1 — V2 takes the fallback off-ramp

Make `_emit_agg` detect ambiguous alias-resolution (two distinct gb_terms
collapsing to the same `(col_name, idx)`) and raise `OptimizerSchemaMismatch`,
exactly the way other V2 emit paths already do for unrecognized inputs. The
engine's legacy-fallback wrapper in `relann/relnn.py::term_graph_to_module`
(was at lines 487–528 before deletion) already catches that exception and
re-runs the graph through the legacy optimizer. The legacy path handles
homogeneous-HGT correctly.

Cheapest, most surgical. Tells V2: "this is your incompleteness boundary,
take the off-ramp."

```python
# pseudocode for _emit_agg
seen_idx = set()
for gb_term in gb_vec.children:
    col_name = self._colref_name(gb_term)
    idx = ... # existing resolution
    if (col_name, idx) in seen_idx:
        raise OptimizerSchemaMismatch(
            f"V2 lift_agg: ambiguous alias resolution for {col_name!r} "
            f"(both gb_terms map to col_idx={idx} in {child_schema})"
        )
    seen_idx.add((col_name, idx))
    ...
```

### Option R2 — Aliases preserve distinct positions

Larger surgery in the alias system: ensure that when a FunctionDef body
parameter is bound to an ER whose schema has fewer columns than the body
uses, the alias map records position-aware bindings rather than a single
canonical-name collapse. Then `_emit_agg` could resolve `s → col 0 of L1`
and `t → col 0 of L1` as **distinct logical positions on the same physical
column** rather than identical refs.

Right call for V2 actually handling the homogeneous case. Bigger PR;
needs deeper understanding of `name_aliases` semantics.

## Regression test

`tests/slow/run_hgt_template_cora.py` Test 2 is the e2e canary. It was
added as a CI step in `9eba660` / `644c7bd` so future re-introductions
of the optimizer will fail this step on Linux+CPU (CI runs on Ubuntu).

Until PR #56 lands, the optimizer is fully removed from juplit; the
engine bypasses it entirely. See `docs/design/repo-structure.md` for the
post-removal layout.

## Recommended unit-level pin for the fix

When R1 (or R2) lands, add a focused unit test under `tests/optimizer_v2/`
(the dir will exist again then) that constructs the smallest e-graph with
the homogeneous-HGT shape and asserts:

- **R1 case**: `_emit_agg` raises `OptimizerSchemaMismatch`.
- **R2 case**: `_emit_agg` produces `group_by_refs` with distinct
  `(input_idx, column_idx)` pairs even when both gb_terms resolve to
  the same physical column.

Either of those plus the existing e2e (the cora HGT template script)
gives belt-and-braces coverage.

## See also

- `docs/_archive/hgt-status-2026-05-25.md` — 2026-05-25 audit on `main`
  confirming `run_hgt_template_cora.py` passed there.
- `docs/design/join-chain-column-bug.md` — Join column-tracking design
  notes (unaffected by this bug, but useful context for anyone reading
  the chain-join logic in `era_operations.py`).
- Git history: `git log -- relann/optimizer/ relann/optimizer_v2/` shows
  the V1/V2 implementations that were removed; `git show 644c7bd` is the
  removal commit; `git show 9eba660` is the prior default-disable commit.
