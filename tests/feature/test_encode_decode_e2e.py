"""End-to-end Session tests for RHS encode brackets and LHS predict decode."""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from relann.encode import EncodeTypeError
from relann.relnn import RelNNNodeError
from relann.session import Session
from relann.torch_utils import full_seed
from relann.era_operations import EmbeddedRelation

def _users_db(n: int = 5):
    df = pd.DataFrame(
        {"uid": range(n), "age": [float(i + 1) for i in range(n)], "score": [0.1 * i for i in range(n)]}
    )
    z = torch.randn(n, 1)
    return {"Users": (df, z)}

def test_e2e_encode_linear_on_numeric_column():
    full_seed(0)
    session = Session(db=_users_db())
    session.run(
        """
#lang:relnn
Feat(uid; [Linear(1, 8)(age)]) :- Users(uid, age, score; z1) .
"""
    )
    session.run(
        """
#lang:relnn
?pred P(uid; z) :- Feat(uid; z) .
"""
    )
    out = session.relation("P")
    assert out.embeddings[0].shape == (5, 8)

def test_e2e_encode_multi_item_concat():
    full_seed(0)
    session = Session(db=_users_db())
    session.run(
        """
#lang:relnn
Feat(uid; [age, score]) :- Users(uid, age, score; z1) .
"""
    )
    session.run(
        """
#lang:relnn
?pred P(uid; z) :- Feat(uid; z) .
"""
    )
    out = session.relation("P")
    assert out.embeddings[0].shape == (5, 2)

def test_e2e_encode_embedding_and_linear():
    """Encode a categorical column with ``torch.nn.Embedding`` (resolved like ``Linear``).

    ``Embedding`` is PyTorch's lookup table: integer category index -> learned vector.
    Here ``Embedding(2, 4)`` means 2 rows in the table (categories ``a`` and ``b`` in
    ``dept``) and each row is a 4-dimensional vector. The ``dept`` column is
    ``pd.Categorical``; RelNN turns codes into indices and the module outputs shape
    ``(n_rows, 4)`` --- hence ``(3, 4)`` for three users.
    """
    full_seed(0)
    df = pd.DataFrame({"uid": [0, 1, 2], "dept": pd.Categorical(["a", "b", "a"])})
    z = torch.randn(3, 1)
    session = Session(db={"Users": (df, z)})
    session.run(
        """
#lang:relnn
Feat(uid; [Embedding(2, 4)(dept)]) :- Users(uid, dept; z1) .
"""
    )
    session.run(
        """
#lang:relnn
?pred P(uid; z) :- Feat(uid; z) .
"""
    )
    out = session.relation("P")
    assert out.embeddings[0].shape == (3, 4)

class MockTextEncoder(nn.Module):
    """Maps pd.Series of strings to random-feature vectors (learnable tail)."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(1, dim)

    def forward(self, x):
        if not hasattr(x, "tolist"):
            raise TypeError("expected pd.Series")
        # hash strings to a scalar feature then project
        vals = torch.tensor([float(hash(str(t)) % 1000) for t in x], dtype=torch.float32).view(-1, 1)
        return self.proj(vals)

def test_e2e_text_encoder_receives_series():
    full_seed(0)
    df = pd.DataFrame({"uid": [0, 1], "bio": ["aa", "bb"]})
    z = torch.randn(2, 1)
    session = Session(db={"Users": (df, z)})
    session.run(
        """
#lang:relnn
Feat(uid; [MockTextEncoder(8)(bio)]) :- Users(uid, bio; z1) .
"""
    )
    session.run(
        """
#lang:relnn
?pred P(uid; z) :- Feat(uid; z) .
"""
    )
    out = session.relation("P")
    assert out.embeddings[0].shape == (2, 8)

def test_bare_text_column_raises():
    """A bare ``[bio]`` on a text/object column requires an explicit encoder."""
    full_seed(0)
    df = pd.DataFrame({"uid": [0, 1], "bio": ["aa", "bb"]})
    z = torch.randn(2, 1)
    session = Session(db={"Users": (df, z)})
    with pytest.raises(RelNNNodeError) as excinfo:
        session.run(
            """
#lang:relnn
Feat(uid; [bio]) :- Users(uid, bio; z1) .
?pred P(uid; z) :- Feat(uid; z) .
"""
        )
    assert isinstance(excinfo.value.__cause__, EncodeTypeError)
    assert "explicit encoder" in str(excinfo.value.__cause__)

def test_explicit_hash_bucket_text_encoder():
    """An explicit ``HashBucketTextEncoder`` encodes a text column."""
    full_seed(0)
    df = pd.DataFrame({"uid": [0, 1], "bio": ["aa", "bb"]})
    z = torch.randn(2, 1)
    session = Session(db={"Users": (df, z)})
    session.run(
        """
#lang:relnn
Feat(uid; [HashBucketTextEncoder(64, 8)(bio)]) :- Users(uid, bio; z1) .
"""
    )
    session.run(
        """
#lang:relnn
?pred P(uid; z) :- Feat(uid; z) .
"""
    )
    out = session.relation("P")
    assert out.embeddings[0].shape == (2, 8)

def test_text_column_in_multi_bracket_raises():
    """A text column in a multi-item ``[bio, age]`` bracket requires an explicit encoder."""
    full_seed(0)
    df = pd.DataFrame({"uid": [0, 1], "bio": ["a", "b"], "age": [1.0, 2.0]})
    z = torch.randn(2, 1)
    session = Session(db={"Users": (df, z)})
    with pytest.raises(RelNNNodeError) as excinfo:
        session.run(
            """
#lang:relnn
Feat(uid; [bio, age]) :- Users(uid, bio, age; z1) .
?pred P(uid; z) :- Feat(uid; z) .
"""
        )
    assert isinstance(excinfo.value.__cause__, EncodeTypeError)
    assert "explicit encoder" in str(excinfo.value.__cause__)

def test_predict_decode_argmax():
    full_seed(0)
    db = {"Input": (pd.DataFrame({"a": range(4)}), torch.randn(4, 3))}
    session = Session(db=db)
    session.run(
        """
#lang:relnn
Logits(a; Linear(3, 2)(z1)) :- Input(a; z1) .
"""
    )
    pred = session.run(
        """
#lang:relnn
?pred Out([ArgMax(pred)], a; pred) :- Logits(a; pred) .
"""
    )
    assert isinstance(pred, EmbeddedRelation)
    assert pred.content is not None
    assert "pred" in pred.content.columns
    assert pred.embeddings[0].shape[0] == 4

def test_predict_decode_default_squeeze():
    full_seed(0)
    db = {"Input": (pd.DataFrame({"a": range(3)}), torch.randn(3, 1))}
    session = Session(db=db)
    session.run(
        """
#lang:relnn
S(a; Linear(1, 1)(z1)) :- Input(a; z1) .
"""
    )
    pred = session.run(
        """
#lang:relnn
?pred Out([x], a; x) :- S(a; x) .
"""
    )
    assert "x" in pred.content.columns

def test_predict_decode_bare_bracket_multiclass_raises():
    """A bare ``[col]`` decode on an (N, K) tensor (K>1) needs an explicit decoder."""
    from relann.pydantic_classes import ContentDecode, Var

    full_seed(0)
    db = {"Input": (pd.DataFrame({"a": range(4)}), torch.randn(4, 1))}
    session = Session(db=db)
    pred = session.run(
        """
#lang:relnn
?pred Out([x], a; x) :- Input(a; x) .
"""
    )
    # Swap in an (N, 3) embedding — a bare [x] cannot decode it into one column.
    multiclass = EmbeddedRelation(
        content_schema=list(pred.content_schema),
        embedding_shapes=[(4, 3)],
        content=pred.content,
        embeddings=[torch.randn(4, 3)],
    )

    class _FakeLHS:
        derived_content_attrs = [ContentDecode(column=Var(name="x"))]

    with pytest.raises(ValueError, match="Cannot decode"):
        session.engine._apply_lhs_decode(_FakeLHS(), multiclass)

def _regression_db(*, n: int, seed: int):
    """Users(uid, age; z1) with numeric age for RHS encode; Targets(uid; y)."""
    rng = torch.Generator().manual_seed(seed)
    df = pd.DataFrame({"uid": range(n), "age": [float(i + 1) for i in range(n)]})
    z = torch.randn(n, 1, generator=rng)
    y = torch.randn(n, 1, generator=rng)
    return {
        "Users": (df, z),
        "Targets": (pd.DataFrame({"uid": range(n)}), y),
    }

def test_e2e_fit_encode_then_linear_mse():
    """Fit: RHS encode [age] -> Linear, then MSELoss against Targets (trainable pipeline)."""
    full_seed(42)
    db = _regression_db(n=12, seed=0)
    session = Session(db=db)
    session.run(
        """
#lang:relnn
Feat(uid; Linear(1, 1)([age])) :- Users(uid, age; z1) .
"""
    )
    session.run(
        """
#lang:relnn
?fit <epochs=5, lr=0.05>
Loss(; MSELoss()(z_pred, z)) :- Feat(uid; z_pred), Targets(uid; z) .
"""
    )
    info = session.engine.trained_modules.get("Loss")
    assert info is not None and "loss_history" in info
    assert len(info["loss_history"]) == 5
    for v in info["loss_history"]:
        assert torch.isfinite(torch.tensor(v, dtype=torch.float64)).item()

def test_e2e_fit_swap_table_second_fit_encode_linear():
    """Train on one Users/Targets table, swap in new rows, fit again under a new loss rule name."""
    full_seed(42)
    session = Session(db=_regression_db(n=8, seed=1))
    session.run(
        """
#lang:relnn
Feat(uid; Linear(1, 1)([age])) :- Users(uid, age; z1) .
"""
    )
    session.run(
        """
#lang:relnn
?fit <epochs=3, lr=0.05>
Loss1(; MSELoss()(z_pred, z)) :- Feat(uid; z_pred), Targets(uid; z) .
"""
    )
    h1 = session.engine.trained_modules["Loss1"]["loss_history"]
    assert len(h1) == 3 and all(torch.isfinite(torch.tensor(v)) for v in h1)

    # Novel data: different row count and values (encode cache keys must not stale-read)
    session.engine.db = _regression_db(n=14, seed=2)

    session.run(
        """
#lang:relnn
?fit <epochs=2, lr=0.05>
Loss2(; MSELoss()(z_pred, z)) :- Feat(uid; z_pred), Targets(uid; z) .
"""
    )
    h2 = session.engine.trained_modules["Loss2"]["loss_history"]
    assert len(h2) == 2 and all(torch.isfinite(torch.tensor(v)) for v in h2)

def test_predict_decode_explicit_argmax_values_correct():
    """Explicit ``[ArgMax(col)]`` decoder also produces correct argmax values."""
    full_seed(0)
    n, k = 5, 4
    logits = torch.zeros(n, k)
    for i in range(n):
        logits[i, (i * 3) % k] = 5.0
    db = {"Input": (pd.DataFrame({"a": range(n)}), logits)}
    session = Session(db=db)
    session.run(
        """
#lang:relnn
Logits(a; z1) :- Input(a; z1) .
"""
    )
    pred = session.run(
        """
#lang:relnn
?pred Out([ArgMax(p)], a; p) :- Logits(a; p) .
"""
    )
    expected = [(i * 3) % k for i in range(n)]
    assert pred.content["p"].tolist() == expected

def test_apply_lhs_decode_multi_embedding_raises():
    """``_apply_lhs_decode`` must raise when the predict output has more than one embedding
    (CR test gap: explicit guard at engine.py:2401 was untested)."""
    from relann.pydantic_classes import ContentDecode, Var

    full_seed(0)
    db = {"Input": (pd.DataFrame({"a": range(2)}), torch.randn(2, 1))}
    session = Session(db=db)
    session.run(
        """
#lang:relnn
S(a; z1) :- Input(a; z1) .
"""
    )
    pred = session.run(
        """
#lang:relnn
?pred Out([x], a; x) :- S(a; x) .
"""
    )
    # Build an EmbeddedRelation copy with two embeddings
    bad = EmbeddedRelation(
        content_schema=list(pred.content_schema),
        embedding_shapes=[pred.embeddings[0].shape, pred.embeddings[0].shape],
        content=pred.content,
        embeddings=[pred.embeddings[0], pred.embeddings[0]],
    )

    class _FakeLHS:
        derived_content_attrs = [ContentDecode(column=Var(name="x"))]

    with pytest.raises(ValueError, match="exactly 1 embedding"):
        session.engine._apply_lhs_decode(_FakeLHS(), bad)

def test_apply_lhs_decode_mid_graph_raises():
    """``_apply_lhs_decode(_is_predict_context=False)`` must raise NotImplementedError
    (CR test gap: explicit guard at engine.py:2384 was untested)."""
    from relann.pydantic_classes import ContentDecode, Var

    full_seed(0)
    db = {"Input": (pd.DataFrame({"a": range(2)}), torch.randn(2, 1))}
    session = Session(db=db)
    session.run(
        """
#lang:relnn
S(a; z1) :- Input(a; z1) .
"""
    )
    pred = session.run(
        """
#lang:relnn
?pred Out([x], a; x) :- S(a; x) .
"""
    )

    # Synthesise a minimal LHS-like object with a decode attr — the function only
    # reads ``derived_content_attrs``.
    class _FakeLHS:
        derived_content_attrs = [ContentDecode(column=Var(name="x"))]

    with pytest.raises(NotImplementedError, match="intermediate term-graph"):
        session.engine._apply_lhs_decode(_FakeLHS(), pred, _is_predict_context=False)

def test_collect_data_sources_missing_relation_raises_keyerror():
    """A rule referring to a relation absent from ``db`` must raise a clear KeyError
    (CR test gap: engine.py:2026 guard was untested)."""
    full_seed(0)
    # Only Users in db; rule references Missing → must KeyError on _collect_data_sources
    db = {"Users": (pd.DataFrame({"a": range(2)}), torch.randn(2, 1))}
    session = Session(db=db)
    with pytest.raises((KeyError, RelNNNodeError)) as excinfo:
        session.run(
            """
#lang:relnn
?pred Out(a; z) :- Missing(a; z) .
"""
        )
    msg = str(excinfo.value)
    assert "Missing" in msg or "not found" in msg.lower()

def test_e2e_fit_encode_embedding_crossentropy():
    """Encode categorical with Embedding -> Linear -> CrossEntropyLoss (classification fit)."""
    full_seed(42)
    n, n_classes = 16, 3
    rng = torch.Generator().manual_seed(7)
    df = pd.DataFrame(
        {
            "uid": range(n),
            "cat": pd.Categorical([["a", "b", "c"][i % 3] for i in range(n)]),
        }
    )
    z = torch.randn(n, 1, generator=rng)
    labels = torch.randint(0, n_classes, (n, 1), generator=rng).long()
    session = Session(db={"Users": (df, z), "Labels": (pd.DataFrame({"uid": range(n)}), labels)})
    session.run(
        """
#lang:relnn
emb_dim = 5 .
n_classes = 3 .
Feat(uid; Linear(emb_dim, n_classes)([Embedding(3, emb_dim)(cat)])) :- Users(uid, cat; z1) .
"""
    )
    session.run(
        """
#lang:relnn
?fit <epochs=4, lr=0.05>
Loss(; CrossEntropyLoss()(z_pred, z)) :- Feat(uid; z_pred), Labels(uid; z) .
"""
    )
    info = session.engine.trained_modules["Loss"]
    assert len(info["loss_history"]) == 4
    for v in info["loss_history"]:
        assert torch.isfinite(torch.tensor(v, dtype=torch.float64)).item()
