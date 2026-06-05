"""Regression tests for the multi-step ``Join`` column-tracking bug.

The pre-fix engine called ``pandas.merge`` with default ``_x``/``_y`` suffixes
at every step of a multi-step Join. After 2 merges the accumulated columns
``v_x`` and ``v_y`` were both present, and the 3rd merge tried to apply the
same defaults again → ``pandas.errors.MergeError``.

Fix landed in ``Join._do_one_merge``: use ``suffixes=("", f"_iter{step}")``.
The empty left suffix preserves the accumulating side's column names verbatim,
and the step-indexed right suffix is unique by construction — no collision
is reachable, regardless of join depth.

These tests now PASS positively (no ``pytest.raises``) and document the post-fix
column-naming contract. If the column naming scheme changes again, update the
assertions here and the design doc together.

To watch the per-step trace, enable DEBUG on the era_operations logger before
running:

    import logging
    logging.getLogger("relann.era_operations").setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
"""
from __future__ import annotations

import pandas as pd
import torch

from relann.column_ref import ColumnRef
from relann.embedded_relation import EmbeddedRelation
from relann.era_operations import Join


def _make_son(v_values: list[int]) -> EmbeddedRelation:
    """One ``EmbeddedRelation`` with schema [id, v] — id is the join key,
    v is the non-key content column that used to collide across sons."""
    df = pd.DataFrame({"id": [1, 2], "v": v_values})
    return EmbeddedRelation(
        content_schema=["id", "v"],
        embedding_shapes=[torch.Size([2, 1])],
        content=df,
        embeddings=[torch.tensor([[1.0], [2.0]])],
    )


def _make_st_son(s_values: list[int], t_values: list[int]) -> EmbeddedRelation:
    """``EmbeddedRelation`` with schema [s, t] — for HGT-style alternating-key chains."""
    df = pd.DataFrame({"s": s_values, "t": t_values})
    return EmbeddedRelation(
        content_schema=["s", "t"],
        embedding_shapes=[torch.Size([len(df), 1])],
        content=df,
        embeddings=[torch.ones(len(df), 1)],
    )


def test_join_chain_four_sons_completes_with_iter_suffixes():
    """4-way join on ``id``, every son carries the non-key column ``v``.

    Pre-fix: pandas's default ``_x``/``_y`` suffixes collided after 2 merges
    and step 3 raised ``MergeError: Passing 'suffixes' which cause duplicate
    columns {'v_x', 'v_y'} is not allowed.``

    Post-fix: every step uses ``suffixes=("", f"_iter{step}")``. Left side keeps
    its column name verbatim (no suffix added to the accumulating side); right
    side gets a unique step-indexed suffix. Columns accumulate as:

        step 1: [id, v, __idx0, v_iter1, __idx1_iter1]
        step 2: [id, v, __idx0, v_iter1, __idx1_iter1, v_iter2, __idx2_iter2]
        step 3: [id, v, __idx0, v_iter1, __idx1_iter1, v_iter2, __idx2_iter2,
                 v_iter3, __idx3_iter3]
    """
    sons = [
        _make_son([10, 20]),
        _make_son([30, 40]),
        _make_son([50, 60]),
        _make_son([70, 80]),
    ]
    merge_steps = [
        {"step": 1, "left_refs": [ColumnRef(0, 0)], "right_refs": [ColumnRef(1, 0)], "key_names": ["id"]},
        {"step": 2, "left_refs": [ColumnRef(1, 0)], "right_refs": [ColumnRef(2, 0)], "key_names": ["id"]},
        {"step": 3, "left_refs": [ColumnRef(2, 0)], "right_refs": [ColumnRef(3, 0)], "key_names": ["id"]},
    ]
    input_schemas = [["id", "v"], ["id", "v"], ["id", "v"], ["id", "v"]]

    join = Join(output_schema=["id"], merge_steps=merge_steps, input_schemas=input_schemas)
    result = join.instantiate(sons)

    cols = set(result.content.columns)
    assert "id" in cols
    # The original (left-most) ``v`` is preserved with its bare name; each subsequent
    # merge contributes a uniquely-suffixed copy. The exact set of names is part of
    # the public contract that downstream ``_apply_join_output_schema`` relies on.
    assert {"v", "v_iter1", "v_iter2", "v_iter3"}.issubset(cols), (
        f"expected v + v_iter1/2/3 columns; got {sorted(cols)}"
    )


def test_join_chain_three_sons_completes_with_iter_suffixes():
    """Sanity check — same shape as the 4-way test but with 3 sons (2 merge steps).

    Pre-fix this DID work (the suffix collision needed 3+ merges). Post-fix the
    column names change from ``v_x, v_y, v`` to ``v, v_iter1, v_iter2`` — keep
    this test in lock-step with the design doc.
    """
    sons = [_make_son([10, 20]), _make_son([30, 40]), _make_son([50, 60])]
    merge_steps = [
        {"step": 1, "left_refs": [ColumnRef(0, 0)], "right_refs": [ColumnRef(1, 0)], "key_names": ["id"]},
        {"step": 2, "left_refs": [ColumnRef(1, 0)], "right_refs": [ColumnRef(2, 0)], "key_names": ["id"]},
    ]
    input_schemas = [["id", "v"], ["id", "v"], ["id", "v"]]

    join = Join(output_schema=["id"], merge_steps=merge_steps, input_schemas=input_schemas)
    result = join.instantiate(sons)

    cols = set(result.content.columns)
    assert "id" in cols
    assert {"v", "v_iter1", "v_iter2"}.issubset(cols), (
        f"expected v + v_iter1/2 columns; got {sorted(cols)}"
    )


def test_output_schema_coalesces_iter_suffixed_duplicates():
    """``_apply_join_output_schema`` regex ``_iter\\d+$`` coverage — if the
    regex anchor or digit class regresses (e.g. to ``_iter\\d+`` without ``$``,
    or ``_iter`` without ``\\d+``), this test catches it. With
    ``output_schema=["id", "v"]`` the post-merge df contains
    ``v, v_iter1, v_iter2`` — they must collapse down to a single ``v``."""
    sons = [_make_son([10, 20]), _make_son([30, 40]), _make_son([50, 60])]
    merge_steps = [
        {"step": 1, "left_refs": [ColumnRef(0, 0)], "right_refs": [ColumnRef(1, 0)], "key_names": ["id"]},
        {"step": 2, "left_refs": [ColumnRef(1, 0)], "right_refs": [ColumnRef(2, 0)], "key_names": ["id"]},
    ]
    input_schemas = [["id", "v"], ["id", "v"], ["id", "v"]]

    join = Join(output_schema=["id", "v"], merge_steps=merge_steps, input_schemas=input_schemas)
    result = join.instantiate(sons)

    cols = list(result.content.columns)
    # The three v-variants must coalesce into a single 'v' column. If any of
    # 'v_iter1' / 'v_iter2' survives, the regex regressed.
    assert cols.count("v") == 1, f"expected exactly one 'v' column; got {cols}"
    assert "v_iter1" not in cols, f"'_iter\\d+$' coalescing failed; got {cols}"
    assert "v_iter2" not in cols, f"'_iter\\d+$' coalescing failed; got {cols}"


def test_future_keys_keeps_dropped_right_key_for_later_step():
    """Direct ``Join``-API test for the HGT-case lookahead in ``_do_one_merge``.

    Without ``future_keys``, the engine would drop the right key column ``s``
    after step 1 (because ``left_on=['t'] != right_on=['s']``), and step 2's
    ``_resolve_step_keys`` would return ``left_on=['s']`` and crash on
    ``KeyError: 's'`` against ``df_joined``.

    With the lookahead, ``s`` is preserved through step 1 because step 2's
    ``left_on`` references it. Step 2 completes, and the final output
    contains both ``s`` and ``t`` (output schema captures them)."""
    # Sons: chain ER0(s, t) ⋈ ER1(s, t) ⋈ ER2(s, t) alternating keys.
    # Step 1: join ER0.t = ER1.s  (drops ER1's 's' WITHOUT lookahead)
    # Step 2: join ER1.s = ER2.t  (needs ER1's 's' — proves lookahead works)
    sons = [
        _make_st_son(s_values=[1, 2], t_values=[10, 20]),
        _make_st_son(s_values=[10, 20], t_values=[100, 200]),
        _make_st_son(s_values=[1000, 2000], t_values=[100, 200]),
    ]
    merge_steps = [
        {"step": 1, "left_refs": [ColumnRef(0, 1)], "right_refs": [ColumnRef(1, 0)], "key_names": ["t"]},
        {"step": 2, "left_refs": [ColumnRef(1, 0)], "right_refs": [ColumnRef(2, 1)], "key_names": ["s"]},
    ]
    input_schemas = [["s", "t"], ["s", "t"], ["s", "t"]]

    join = Join(output_schema=["s", "t"], merge_steps=merge_steps, input_schemas=input_schemas)
    # Must NOT raise KeyError: 's' on step 2.
    result = join.instantiate(sons)
    assert result.content is not None
    cols = set(result.content.columns)
    # Both keys present in the output.
    assert "s" in cols
    assert "t" in cols


def test_future_keys_does_not_over_preserve_right_keys():
    """Pin the S1 fix: ``future_keys`` only considers later steps' ``left_on``,
    not their ``right_on`` (which resolves against fresh ``dfs[step]`` and
    doesn't need lookahead protection in ``df_joined``).

    Pre-S1, this pattern leaked an extra ``'a'`` column into the final
    output because step 2's ``right_on=['a']`` (incorrectly) preserved
    step 1's right-key ``'a'`` through to the end.

    Post-S1, the output is exactly ``output_schema`` — no phantom columns.
    """
    # son0(x, y), son1(a, y), son2(a, e)
    # Step 1: join on son0.x = son1.a (so 'a' is the right key, normally dropped).
    # Step 2: join on son1.y = son2.a (left_on=['y'], right_on=['a']).
    # Pre-S1: step 1 would have kept 'a' because step 2's right_on=['a'] was in
    # future_keys. Post-S1: step 2's left_on=['y'] only -> 'a' correctly dropped.
    s0 = EmbeddedRelation(
        content_schema=["x", "y"],
        embedding_shapes=[torch.Size([2, 1])],
        content=pd.DataFrame({"x": [1, 2], "y": [10, 20]}),
        embeddings=[torch.ones(2, 1)],
    )
    s1 = EmbeddedRelation(
        content_schema=["a", "y"],
        embedding_shapes=[torch.Size([2, 1])],
        content=pd.DataFrame({"a": [1, 2], "y": [10, 20]}),
        embeddings=[torch.ones(2, 1)],
    )
    s2 = EmbeddedRelation(
        content_schema=["a", "e"],
        embedding_shapes=[torch.Size([2, 1])],
        content=pd.DataFrame({"a": [1, 2], "e": ["foo", "bar"]}),
        embeddings=[torch.ones(2, 1)],
    )
    join = Join(
        output_schema=["x", "y", "e"],
        merge_steps=[
            {"step": 1, "left_refs": [ColumnRef(0, 0)], "right_refs": [ColumnRef(1, 0)], "key_names": ["x"]},
            {"step": 2, "left_refs": [ColumnRef(1, 1)], "right_refs": [ColumnRef(2, 0)], "key_names": ["y"]},
        ],
        input_schemas=[["x", "y"], ["a", "y"], ["a", "e"]],
    )
    result = join.instantiate([s0, s1, s2])
    cols = set(result.content.columns)
    assert cols == {"x", "y", "e"}, (
        f"expected output exactly {{'x', 'y', 'e'}}; got {sorted(cols)}. "
        f"Phantom column means future_keys regressed to including later steps' right_on."
    )


def test_join_chain_does_not_regress_to_x_y_suffixes():
    """Negative assertion — make sure nobody re-introduces pandas default suffixes.

    If this fails, someone removed the ``suffixes=("", f"_iter{step}")`` argument
    in ``Join._do_one_merge``. The fix-doc explains why that matters for chains
    of length 3+.
    """
    sons = [_make_son([10, 20]), _make_son([30, 40]), _make_son([50, 60]), _make_son([70, 80])]
    merge_steps = [
        {"step": 1, "left_refs": [ColumnRef(0, 0)], "right_refs": [ColumnRef(1, 0)], "key_names": ["id"]},
        {"step": 2, "left_refs": [ColumnRef(1, 0)], "right_refs": [ColumnRef(2, 0)], "key_names": ["id"]},
        {"step": 3, "left_refs": [ColumnRef(2, 0)], "right_refs": [ColumnRef(3, 0)], "key_names": ["id"]},
    ]
    input_schemas = [["id", "v"], ["id", "v"], ["id", "v"], ["id", "v"]]

    join = Join(output_schema=["id"], merge_steps=merge_steps, input_schemas=input_schemas)
    result = join.instantiate(sons)

    cols = set(result.content.columns)
    assert "v_x" not in cols, f"pandas default _x suffix should not appear; got {sorted(cols)}"
    assert "v_y" not in cols, f"pandas default _y suffix should not appear; got {sorted(cols)}"


# ─────────────────────────────────────────────────────────────────────────────
# Run directly with `python tests/repro/test_join_chain_column_bug.py`.
# Useful when debugging in VS Code — set a breakpoint on `join.instantiate(sons)`
# below and step into `Join._do_one_merge` to watch the suffix scheme in action.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(name)s [%(levelname)s] %(message)s",
    )
    logging.getLogger("relann.era_operations").setLevel(logging.DEBUG)

    print("\n" + "=" * 70)
    print("  4-way join (the post-fix regression test)")
    print("=" * 70)
    sons = [_make_son([10, 20]), _make_son([30, 40]), _make_son([50, 60]), _make_son([70, 80])]
    merge_steps = [
        {"step": 1, "left_refs": [ColumnRef(0, 0)], "right_refs": [ColumnRef(1, 0)], "key_names": ["id"]},
        {"step": 2, "left_refs": [ColumnRef(1, 0)], "right_refs": [ColumnRef(2, 0)], "key_names": ["id"]},
        {"step": 3, "left_refs": [ColumnRef(2, 0)], "right_refs": [ColumnRef(3, 0)], "key_names": ["id"]},
    ]
    input_schemas = [["id", "v"], ["id", "v"], ["id", "v"], ["id", "v"]]
    join = Join(output_schema=["id"], merge_steps=merge_steps, input_schemas=input_schemas)

    try:
        result = join.instantiate(sons)
        print(f"\n  OK — output columns: {list(result.content.columns)}")
        print("  Fix is in place.\n")
    except pd.errors.MergeError as e:
        print(f"\n  REGRESSION — pandas.errors.MergeError: {e}")
        print("  The fix in Join._do_one_merge has been removed or reverted.\n")
    except Exception as e:
        print(f"\n  UNEXPECTED EXCEPTION: {type(e).__name__}: {e}\n")
