# Multi-step Join column-name bug

> **Status: FIXED 2026-05-23.** The DHN/HGT chained-merge `MergeError` is resolved by using `suffixes=("", f"_iter{step}")` in `Join._do_one_merge` (commit pending). Regression covered by `tests/repro/test_join_chain_column_bug.py`. The HGT `KeyError` variant (key dropped between steps) is *also* covered by the same suffix scheme — the right key column is now suffixed with `_iter{step}` instead of being silently dropped, so `_resolve_step_keys` for the next step still finds the column it expects.
>
> A separate, **downstream** bug (matmul shape mismatch in `transformation_C2_Sparse`, "mat1 and mat2 shapes cannot be multiplied (1x1 and 20x10)") still keeps `tests/dhn/test_csl_ghl_c2_4_forward_small_subset` red. That failure is past the Join layer and was previously masked by it. Tracked separately.
>
> Original diagnosis (2026-05-23) was done via the `logging.DEBUG`-gated instrumentation that lives in `relann/era_operations.py:Join._resolve_step_keys` and `_do_one_merge`. Surfaced by `tests/dhn/test_csl_ghl_c2_4_forward_small_subset` and `tests/slow/run_hgt_template_cora.py`.

## Symptom

`relann.relnn.RelNNNodeError: RelNN evaluation failed at node 'join_N' (mode=instantiate, op=Join): <err>`

with one of two underlying pandas errors depending on the rule:

1. **`pandas.errors.MergeError: Passing 'suffixes' which cause duplicate columns {'v_x', 'v_y'} is not allowed.`** — happens in the DHN test.
2. **`KeyError: 's'`** during `pandas.DataFrame.merge` — happens in the HGT-template-cora test.

## Root cause (same for both)

`Join.instantiate` runs through a list of `merge_steps`, calling `_resolve_step_keys` then `_do_one_merge` for each. `_resolve_step_keys` resolves `left_on` / `right_on` against `self.input_schemas` — the ORIGINAL schemas of each input relation. But `_do_one_merge` MUTATES `df_joined` along the way:

- pandas adds `_x` / `_y` suffixes to non-key columns that exist in both inputs.
- The function then drops the right-key column when it differs from the left key (lines 575–584 of `era_operations.py`).

The next step's `_resolve_step_keys` doesn't know about either mutation, so the keys it returns can refer to columns that **no longer exist** in `df_joined`, or that have **been suffixed**.

### DHN case (`v_x`, `v_y` collision)

Multi-way self-join on `(graph_id, n)` across 3+ relations, each carrying a `v` column. After step 2:

```
df_joined.columns = ['graph_id', 'n', 'v_x', '__idx0', 'v_y', '__idx1']
df_next.columns   = ['graph_id', 'n', 'v', '__idx2']
```

Step 3 calls `pandas.merge(df_joined, df_next, on=['graph_id', 'n', ...])`. pandas tries to apply default suffixes `_x`/`_y` to the non-key column `v`, but `v_x` and `v_y` **already exist** in df_joined → `MergeError`.

### HGT case (`s` missing)

Two-step join: step 1 joins `left('t')` with `right('s')`, step 2 joins `left('s')` with `right('t')`. After step 1, `df_joined` looks like:

```
df_joined.columns = ['t', '__idx0', '__idx1']   ← 's' has been dropped (line 577–584)
df_next.columns   = ['t', '__idx2']
left_on  = ['s']      ← _resolve_step_keys returned this from input_schemas
right_on = ['t']
missing_left = ['s']  ← but 's' isn't in df_joined!
```

pandas merge → `KeyError: 's'`.

## Diagnostic that confirms the analysis

Enable DEBUG-level logging on the `relann.era_operations` logger and run a failing test. The instrumentation in `Join._resolve_step_keys` and `Join._do_one_merge` emits one log record per step with `left_on`, `right_on`, the current `df_joined.columns`, and `missing_left` / `missing_right`.

```python
# In a notebook or test:
import logging
logging.getLogger('relann.era_operations').setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)  # so DEBUG records actually appear
```

Or via the project helper:

```python
from relann.log_utils import checkLogs
with checkLogs(name='relann.era_operations'):
    # ... call session.run / session.fit / etc. — diagnostic logs flow inside the block
    pass
```

The pattern is unmistakable — `missing_left` and/or `missing_right` becomes non-empty on the offending step, and you can see immediately whether the bug is in resolution (key not present) or accumulation (key was suffixed away by a prior merge).

## Fix (landed 2026-05-23)

The accepted approach is the **unique per-step suffix** option from the original two candidates, chosen for minimal surface area:

```python
# relann/era_operations.py — Join._do_one_merge
suffixes = ("", f"_iter{step}")
if left_on == right_on:
    return Join._merge_on(df_joined, df_next, on=left_on, suffixes=suffixes)
df_joined = Join._merge_lr(
    df_joined, df_next, left_on=left_on, right_on=right_on, suffixes=suffixes,
)
```

Why this works for both failure modes:

- **DHN (suffix collision)** — Left suffix is the empty string, so the accumulating side keeps every column name verbatim across all merges. The right side gets a unique `_iter{step}` suffix at each step, so two steps can never produce the same suffixed name. No `MergeError`.
- **HGT (dropped key)** — Pre-fix, when `left_on != right_on`, the right key was dropped after the merge. With per-step suffixes the right key is no longer overwritten by a "_y" version; the drop logic now targets the suffixed name explicitly when present, otherwise the original. Either way `_resolve_step_keys` for subsequent steps lands on a column that still exists.

`_apply_join_output_schema` was extended to recognize `_iter\d+$` as a duplicate suffix (in addition to `_x`/`_y`/`_left`/`_right`), so accumulated copies of an output column get coalesced down to the canonical name when the Join finishes.

### Reference inputs (used by the repro test)

```python
sons = [_make_son([10, 20]), _make_son([30, 40]), _make_son([50, 60]), _make_son([70, 80])]
merge_steps = [
    {"step": 1, "left_refs": [ColumnRef(0, 0)], "right_refs": [ColumnRef(1, 0)], "key_names": ["id"]},
    {"step": 2, "left_refs": [ColumnRef(1, 0)], "right_refs": [ColumnRef(2, 0)], "key_names": ["id"]},
    {"step": 3, "left_refs": [ColumnRef(2, 0)], "right_refs": [ColumnRef(3, 0)], "key_names": ["id"]},
]
# Pre-fix: pandas.errors.MergeError: duplicate columns {'v_x', 'v_y'}
# Post-fix: out columns include 'v', 'v_iter1', 'v_iter2', 'v_iter3'
```

### Alternative that was rejected

The first-cut design proposed tracking a full `rename_history: List[Dict[str, str]]` and remapping `left_on` / `right_on` on each subsequent step. That solves the same problem but adds a stateful object that has to stay in sync with every pandas mutation. The suffix approach above gets the same correctness from a one-line change to the merge call.

## Why this doesn't break more tests

The 145 optimizer tests + 263 feature tests don't exercise multi-way self-joins on shared embedding-column names. Most DSL rules join 2 relations and then either aggregate or project — single-step Joins work fine. The bug only surfaces in:

- Chained 3+ way joins on the same column name → DHN.
- Chained joins where right-key gets dropped, then later reference resurfaces → HGT.

## Files touched by the instrumentation

- `relann/era_operations.py` — added `if logger.isEnabledFor(logging.DEBUG):` branches in `_resolve_step_keys` and `_do_one_merge`. No new imports; no behavioural change at non-DEBUG levels.

The instrumentation is harmless at the default log level; remove it whenever the fix lands.
