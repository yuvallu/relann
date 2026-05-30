"""Numerical-stability pin for the RelaNN softmax DSL pattern.

Background
----------
Naive softmax expressed as

    Exp(s, t; exp(z))   :- Scores(s, t; z) .
    Denom(t; sum(z))    :- Exp(s, t; z) .
    Out(s, t; z1 / z2)  :- Exp(s, t; z1), Denom(t; z2) .

overflows once any score is large (`exp(80)` > float32 max), producing
inf/inf = NaN. This bit `tests/slow/run_compare_dblp_hgt_generic.py` —
the 2-layer HGT was hitting NaN at epoch 0 and the 1-layer's "Forward
match (vs first-order)" assertion was failing at max_diff = 3.02e-01.

The stable form subtracts the per-group max before `exp`:

    Max(t; max(z))            :- Scores(s, t; z) .
    Stable(s, t; z1 - z2)     :- Scores(s, t; z1), Max(t; z2) .
    Exp(s, t; exp(z))         :- Stable(s, t; z) .
    Denom(t; sum(z))          :- Exp(s, t; z) .
    Out(s, t; z1 / z2)        :- Exp(s, t; z1), Denom(t; z2) .

This file pins both halves:

1. `test_naive_softmax_overflows_on_large_scores` confirms the failure
   mode is real (constructs scores with a 100-magnitude entry and shows
   the naive pattern returns non-finite values). Pure PyTorch; no DSL.

2. `test_stable_softmax_dsl_matches_torch_softmax` constructs the
   stable pattern AS A RELANN PROGRAM via Session, feeds it scores that
   would overflow the naive form (peak ~100), and verifies the engine
   computes the same values as `torch.softmax(scores, dim=0)` grouped
   by the target column. This is the building-block invariant that the
   HGT slow scripts rely on; if the DSL semantics ever drift, this
   pin fails fast (~1s) instead of waiting for the 60s+ slow runs.

If either pin fails after a future engine change, see also:
    tests/slow/run_compare_dblp_hgt_generic.py (the consumer)
    docs/design/hgt-2l-recursive-template-row-loss.md (sibling 2L bug)
"""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import pytest
import torch

from relann.session import Session


# Scores chosen to overflow naive exp32 (largest entry is ~100). Three
# distinct targets so the per-group max makes a difference.
_S = [
    (0, "a", 100.0),  # large -- overflows in naive form
    (1, "a", 99.5),
    (2, "a", 99.0),
    (0, "b", -1.0),
    (1, "b",  0.0),
    (2, "b",  1.0),
    (3, "c",  5.0),
    (4, "c",  6.0),
]


def _build_scores_db() -> Tuple[Dict[str, tuple], torch.Tensor, list[str]]:
    """Construct a Scores(s, t; z) relation as a (content_df, embedding_tensor)
    pair plus the bare tensor for torch.softmax verification."""
    src = [r[0] for r in _S]
    tgt = [r[1] for r in _S]
    z   = [r[2] for r in _S]
    df = pd.DataFrame({"s": src, "t": tgt})
    emb = torch.tensor(z, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    return ({"Scores": (df, emb)}, emb, tgt)


def test_naive_softmax_overflows_on_large_scores():
    """Pure-PyTorch check that the failure mode the DSL pattern guards
    against is real: ``exp(z) / sum(exp(z))`` produces NaN when any
    score is in the 100-magnitude range."""
    z = torch.tensor([100.0, 99.5, 99.0], dtype=torch.float32)
    naive_exp = torch.exp(z)            # inf, inf, inf in float32
    naive_denom = naive_exp.sum()       # inf
    naive_out = naive_exp / naive_denom # inf / inf = nan

    assert torch.isinf(naive_exp).any(), (
        "Setup broken: scores ~100 should overflow exp(float32)"
    )
    assert torch.isnan(naive_out).any(), (
        "Setup broken: inf/inf should produce NaN -- if this fails, the test "
        "scores chosen above no longer overflow on this platform; bump them up."
    )


def test_stable_softmax_dsl_matches_torch_softmax():
    """Build the stable Softmax DSL pattern, run it through the engine,
    and verify outputs equal ``torch.softmax`` grouped by target column.

    This is the canonical building-block test for the stable pattern.
    Slow HGT scripts compose this and would catch a regression too, but
    they take 60s+ and only run on opt-in; this test runs in <1s.
    """
    db, _, target_labels = _build_scores_db()
    session = Session(db=db)

    session.run("""
#lang:relnn
Max(t; max(z))            :- Scores(s, t; z) .
Stable(s, t; z1 - z2)     :- Scores(s, t; z1), Max(t; z2) .
Exp(s, t; exp(z))         :- Stable(s, t; z) .
Denom(t; sum(z))          :- Exp(s, t; z) .
Soft(s, t; z1 / z2)       :- Exp(s, t; z1), Denom(t; z2) .
""")

    pred = session.run("""
#lang:relnn
?pred Out(s, t; z) :- Soft(s, t; z) .
""")
    assert pred.content is not None and pred.embeddings is not None, (
        "Engine didn't materialize Soft(s, t; z) -- DSL or backend regression"
    )

    # Build the engine's output as a {(s_int, t_str): float} dict.
    out_df = pred.content
    out_z  = pred.embeddings[0].squeeze(1)
    engine_by_st = {
        (int(row.s), str(row.t)): float(out_z[i].item())
        for i, row in out_df.iterrows()
    }

    # Expected reference: per-target stable softmax via torch.
    # Same rows as _S; group by target, apply softmax over scores in
    # that group.
    by_target: dict[str, list[tuple[int, float]]] = {}
    for s, t, z in _S:
        by_target.setdefault(t, []).append((s, z))
    expected_by_st: dict[tuple[int, str], float] = {}
    for t, rows in by_target.items():
        s_idxs = [r[0] for r in rows]
        z_vals = torch.tensor([r[1] for r in rows], dtype=torch.float32)
        soft   = torch.softmax(z_vals, dim=0)
        for s_idx, val in zip(s_idxs, soft.tolist()):
            expected_by_st[(s_idx, t)] = val

    assert set(engine_by_st.keys()) == set(expected_by_st.keys()), (
        f"Engine emitted different (s, t) pairs than reference.\n"
        f"  engine:   {sorted(engine_by_st.keys())}\n"
        f"  expected: {sorted(expected_by_st.keys())}"
    )
    for key, want in expected_by_st.items():
        got = engine_by_st[key]
        assert abs(got - want) < 1e-6, (
            f"softmax mismatch at (s={key[0]}, t={key[1]!r}): "
            f"engine={got!r}, torch={want!r}, diff={got - want!r}"
        )

    # Final positive: every value is finite. Engine should NOT have any
    # NaN/inf — the whole point of the stable form.
    assert torch.isfinite(out_z).all(), (
        "Engine emitted non-finite softmax values despite max-subtraction. "
        "Stability pattern was bypassed somewhere."
    )


@pytest.mark.parametrize("group_count", [1, 2, 5])
def test_stable_softmax_sums_to_one_per_group(group_count: int):
    """Per-group softmax should sum to 1.0 within float-tolerance.
    Parametrized across a few group counts to cover the degenerate
    single-group case (Max(t; max(z)) on a single t) plus a few more."""
    # Build a Scores DB with group_count distinct targets, each with 3 sources.
    rows = []
    for tgt_idx in range(group_count):
        tgt_label = f"t{tgt_idx}"
        for src_idx in range(3):
            rows.append((src_idx, tgt_label, float(tgt_idx * 10 + src_idx)))
    df = pd.DataFrame({"s": [r[0] for r in rows], "t": [r[1] for r in rows]})
    emb = torch.tensor([r[2] for r in rows], dtype=torch.float32).unsqueeze(1)
    db = {"Scores": (df, emb)}

    session = Session(db=db)
    session.run("""
#lang:relnn
Max(t; max(z))            :- Scores(s, t; z) .
Stable(s, t; z1 - z2)     :- Scores(s, t; z1), Max(t; z2) .
Exp(s, t; exp(z))         :- Stable(s, t; z) .
Denom(t; sum(z))          :- Exp(s, t; z) .
Soft(s, t; z1 / z2)       :- Exp(s, t; z1), Denom(t; z2) .
""")
    pred = session.run("""
#lang:relnn
?pred Out(s, t; z) :- Soft(s, t; z) .
""")
    out_df = pred.content
    out_z  = pred.embeddings[0].squeeze(1)

    by_t_sum: dict[str, float] = {}
    for i, row in out_df.iterrows():
        by_t_sum[str(row.t)] = by_t_sum.get(str(row.t), 0.0) + float(out_z[i].item())

    assert len(by_t_sum) == group_count, (
        f"Expected {group_count} target groups, got {len(by_t_sum)}: {by_t_sum}"
    )
    for t, total in by_t_sum.items():
        assert abs(total - 1.0) < 1e-5, (
            f"softmax over target {t!r} sums to {total!r}, not 1.0"
        )
