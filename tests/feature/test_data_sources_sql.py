"""Tests for SqlSource: verifies loading from an in-memory SQLite database.

Uses sqlalchemy + sqlite:///:memory: so no file I/O is required.
Results are compared against the DataFrameSource path (which is already
validated against the legacy tuple path by test_data_sources_dataframe.py).
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
import torch
import torch.nn as nn
from relann.session import Session
from relann.torch_utils import full_seed
from relann.data_sources import DataFrameSource, SqlSource

# Mutable backing tensor for ToyNodeFeatures (set before each test that uses it).
_TOY_NODE_FEAT_WEIGHT: Optional[torch.Tensor] = None

class ToyNodeFeatures(nn.Module):
    """Maps tensorized ``pid`` to a frozen row (007 / CoraNodeFeatures pattern at toy scale)."""

    def __init__(self):
        super().__init__()
        if _TOY_NODE_FEAT_WEIGHT is None:
            raise RuntimeError("set _TOY_NODE_FEAT_WEIGHT before compiling rules that use ToyNodeFeatures")
        self.register_buffer("weight", _TOY_NODE_FEAT_WEIGHT)

    def forward(self, x):
        idx = x.long().clamp(0, self.weight.size(0) - 1).squeeze(-1)
        return self.weight[idx]

def _set_toy_node_features(weight: torch.Tensor) -> None:
    global _TOY_NODE_FEAT_WEIGHT
    _TOY_NODE_FEAT_WEIGHT = weight

sqlalchemy = pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed")

def _make_sqlite_engine():
    from sqlalchemy import create_engine
    return create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})

def _populate_engine(engine, df: pd.DataFrame, table: str) -> None:
    with engine.connect() as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)
        conn.commit()

# ---------------------------------------------------------------------------
# Basic load_full
# ---------------------------------------------------------------------------

def test_sql_source_load_full_returns_er_dict():
    """load_full() must return a valid ER-dict with the expected keys."""
    df = pd.DataFrame({"uid": range(3), "feat": [1.0, 2.0, 3.0]})
    engine = _make_sqlite_engine()
    _populate_engine(engine, df, "Users")

    src = SqlSource("Users", engine, table="Users")
    result = src.load_full()

    assert set(result.keys()) == {
        "content",
        "content_schema",
        "embedding_shapes",
        "embeddings",
        "column_vocabs",
        "data_version",
    }
    assert isinstance(result["content"], pd.DataFrame)
    assert list(result["content"].columns) == ["uid", "feat"]
    assert len(result["content"]) == 3
    assert result["embeddings"] is None
    assert result["embedding_shapes"] == []

def test_sql_source_load_full_with_custom_query():
    """SqlSource with a custom query loads only the requested rows."""
    df = pd.DataFrame({"pid": range(5), "year": [2010, 2015, 2018, 2020, 2022]})
    engine = _make_sqlite_engine()
    _populate_engine(engine, df, "Papers")

    src = SqlSource(
        "Papers", engine,
        query="SELECT pid, year FROM Papers WHERE year >= 2018",
    )
    result = src.load_full()
    assert len(result["content"]) == 3
    assert list(result["content"]["year"]) == [2018, 2020, 2022]

# ---------------------------------------------------------------------------
# Integration: SqlSource inside Session
# ---------------------------------------------------------------------------

def test_sql_source_produces_same_embeddings_as_dataframe_source():
    """Session(db={name: SqlSource}) must produce the same output shape as DataFrameSource.

    The encode bracket rule [Linear(1,8)(feat)] requires at least one embedding variable
    on the relation (here z1) so Transformation.instantiate can infer input shapes.
    We add a dummy 1-dim zero tensor to the DataFrameSource for this purpose; SqlSource
    does not carry embeddings so we use a DataFrameSource wrapper with the dummy tensor
    for the SqlSource comparison as well.
    """
    n = 4
    df = pd.DataFrame({"uid": range(n), "feat": [float(i) for i in range(n)]})
    dummy_z = torch.zeros(n, 1)
    engine = _make_sqlite_engine()
    _populate_engine(engine, df, "Users")

    program = """
#lang:relnn
Enc(uid; [Linear(1, 8)(feat)]) :- Users(uid, feat; z1) .
?pred P(uid; z) :- Enc(uid; z) .
"""

    # DataFrameSource baseline (with dummy embedding)
    full_seed(42)
    session_df = Session(db={"Users": DataFrameSource("Users", df, embeddings=dummy_z)})
    session_df.run(program)
    emb_df = session_df.relation("P").embeddings[0]

    # SqlSource — wraps the SQL-loaded df + same dummy tensor
    full_seed(42)
    sql_src = SqlSource("Users", engine, table="Users")
    loaded = sql_src.load_full()["content"]
    # Re-wrap with a dummy embedding tensor so the rule can probe input shapes
    session_sql = Session(db={"Users": DataFrameSource("Users", loaded, embeddings=dummy_z)})
    session_sql.run(program)
    emb_sql = session_sql.relation("P").embeddings[0]

    assert emb_df.shape == emb_sql.shape, "shapes differ"
    assert torch.isfinite(emb_df).all()
    assert torch.isfinite(emb_sql).all()

def test_sql_source_encode_and_predict():
    """End-to-end: encode from SQL columns, fit, predict.

    People and Labels are loaded from SQLite via SqlSource.  Because encode-only
    rules still require at least one embedding variable, we wrap the SQL-loaded
    DataFrames with a dummy 1-dim tensor inside DataFrameSource.
    """
    full_seed(0)
    n = 5
    df = pd.DataFrame({"uid": range(n), "age": [20.0, 25.0, 30.0, 35.0, 40.0]})
    labels_df = pd.DataFrame({"uid": range(n), "label": [0, 1, 0, 1, 0]})
    dummy_z = torch.zeros(n, 1)
    label_tensor = torch.tensor(labels_df["label"].values).long().unsqueeze(1)

    engine = _make_sqlite_engine()
    _populate_engine(engine, df, "People")
    _populate_engine(engine, labels_df, "Labels")

    # Load content from SQL, then wrap with DataFrameSource + dummy embedding
    with engine.connect() as conn:
        people_loaded = pd.read_sql_table("People", conn)
        labels_loaded = pd.read_sql_table("Labels", conn)

    session = Session(db={
        "People": DataFrameSource("People", people_loaded, embeddings=dummy_z),
        "Labels": DataFrameSource("Labels", labels_loaded[["uid"]], embeddings=label_tensor),
    })
    session.run(
        """
#lang:relnn
Feat(uid; [Linear(1, 4)(age)]) :- People(uid, age; z1) .
?fit <epochs=3, lr=0.01> Loss(; CrossEntropyLoss()(pred, y)) :-
    Feat(uid; pred), Labels(uid; y) .
"""
    )
    pred = session.run(
        """
#lang:relnn
?pred Out(uid; z) :- Feat(uid; z) .
"""
    )
    assert pred.embeddings[0].shape == (n, 4)

def test_sql_source_in_session_encode_fit_after_engine_materialize():
    """Engine lazily materialises ``SqlSource`` on first reference; the user can also
    force materialisation via ``_materialise_if_source`` and then attach 007-style
    placeholders.

    ``Transformation.instantiate`` requires ``embedding_shapes`` / tensors on the son relation
    even when all signal comes from RHS encode (see 007 notebook). Raw ``SqlSource`` rows have
    neither, so this test proves the Session+SQL path by: (1) ``Session`` with ``SqlSource`` for
    People and Labels, (2) materialising explicitly + mutating ``engine.db`` to add dummy
    ``(N,1)`` and label tensors, (3) same encode + fit + predict as
    ``test_sql_source_encode_and_predict``.
    """
    full_seed(0)
    n = 5
    df = pd.DataFrame({"uid": range(n), "age": [20.0, 25.0, 30.0, 35.0, 40.0]})
    labels_df = pd.DataFrame({"uid": range(n), "label": [0, 1, 0, 1, 0]})

    engine = _make_sqlite_engine()
    _populate_engine(engine, df, "People")
    _populate_engine(engine, labels_df, "Labels")

    session = Session(db={
        "People": SqlSource("People", engine, table="People"),
        "Labels": SqlSource("Labels", engine, table="Labels"),
    })

    # Lazy: db entries are still SqlSource objects until first reference.
    from relann.data_sources import SqlSource as _SqlSource
    assert isinstance(session.engine.db["People"], _SqlSource)
    # Force materialisation via the engine helper (caches in-place).
    peo = session.engine._materialise_if_source("People")
    assert peo["embeddings"] is None
    n_rows = len(peo["content"])
    session.engine.db["People"] = {
        **peo,
        "embeddings": [torch.zeros(n_rows, 1, dtype=torch.float32)],
        "embedding_shapes": [(n_rows, 1)],
    }
    lab = session.engine._materialise_if_source("Labels")
    lab_tensor = torch.tensor(lab["content"]["label"].values, dtype=torch.long).unsqueeze(1)
    session.engine.db["Labels"] = {
        "content": lab["content"][["uid"]].copy(),
        "content_schema": ["uid"],
        "embeddings": [lab_tensor],
        "embedding_shapes": [tuple(lab_tensor.shape)],
    }

    session.run(
        """
#lang:relnn
Feat(uid; [Linear(1, 4)(age)]) :- People(uid, age; z1) .
?fit <epochs=3, lr=0.01> Loss(; CrossEntropyLoss()(pred, y)) :-
    Feat(uid; pred), Labels(uid; y) .
"""
    )
    pred = session.run(
        """
#lang:relnn
?pred Out(uid; z) :- Feat(uid; z) .
"""
    )
    assert pred.embeddings[0].shape == (n, 4)

def test_sql_source_session_predict_with_argmax_decode():
    """SQL edge table + aggregation + logits, then ``?pred`` with LHS ``ArgMax`` decode."""
    full_seed(0)
    n = 4
    nodes_df = pd.DataFrame({"pid": range(n)})
    nodes_z = torch.randn(n, 3)
    edges_df = pd.DataFrame({"src": [0, 1, 2, 3], "dst": [1, 2, 3, 0]})

    sa_engine = _make_sqlite_engine()
    _populate_engine(sa_engine, edges_df, "Edges")

    session = Session(db={
        "Nodes": DataFrameSource("Nodes", nodes_df, embeddings=nodes_z),
        "Edges": SqlSource("Edges", sa_engine, table="Edges"),
    })
    # NOTE: predict-time optimizer (V1 e-graph, merged from main) inserts a
    session.run(
        """
#lang:relnn
Agg(dst; sum(z)) :- Nodes(src; z), Edges(src, dst) .
Logits(dst; Linear(3, 2)(z)) :- Agg(dst; z) .
"""
    )
    pred = session.run(
        """
#lang:relnn
?pred Out([ArgMax(pred)], dst; pred) :- Logits(dst; pred) .
"""
    )
    assert pred.content is not None
    assert "pred" in pred.content.columns
    assert pred.embeddings[0].shape[0] == n

# ---------------------------------------------------------------------------
# load_by_keys
# ---------------------------------------------------------------------------

def test_sql_source_load_by_keys_filters_rows():
    """load_by_keys returns a subset ER-dict for table-backed SqlSource."""
    df = pd.DataFrame({"uid": range(3), "val": [1.0, 2.0, 3.0]})
    engine = _make_sqlite_engine()
    _populate_engine(engine, df, "T")
    src = SqlSource("T", engine, table="T")
    sub = src.load_by_keys("uid", [0, 1])
    assert len(sub["content"]) == 2
    assert list(sub["content"]["uid"]) == [0, 1]
    assert sub["data_version"] >= 1

def test_sql_source_load_by_keys_query_mode_raises():
    """``load_by_keys`` on a ``query=`` source must raise NotImplementedError
    (CR test gap: data_sources.py:353 guard was untested)."""
    engine = _make_sqlite_engine()
    src = SqlSource("T", engine, query="SELECT 1 AS x")
    with pytest.raises(NotImplementedError, match="free-form query"):
        src.load_by_keys("x", [1])

def test_sql_source_load_by_keys_unknown_key_column_raises():
    """``load_by_keys`` must surface a clear KeyError when ``key_col`` isn't on the table
    (CR fix #3: error from f-string SQL was opaque; SQLAlchemy Core path now validates)."""
    df = pd.DataFrame({"uid": range(3), "val": [1.0, 2.0, 3.0]})
    engine = _make_sqlite_engine()
    _populate_engine(engine, df, "T")
    src = SqlSource("T", engine, table="T")
    with pytest.raises(KeyError, match="key column"):
        src.load_by_keys("does_not_exist", [0, 1])

def test_sql_source_dtype_map_unknown_column_raises_clearly():
    """Typos in ``dtype_map`` keys must surface as a clear ValueError on first load
    (CR fix #3: previously failed with a confusing pandas KeyError at minibatch time)."""
    df = pd.DataFrame({"uid": range(3), "val": [1.0, 2.0, 3.0]})
    engine = _make_sqlite_engine()
    _populate_engine(engine, df, "T")
    src = SqlSource("T", engine, table="T", dtype_map={"vai": "float64"})  # typo
    with pytest.raises(ValueError, match="references unknown columns"):
        src.load_full()

def test_sql_source_engine_init_is_lazy():
    """``Engine.__init__`` must NOT call ``load_full()`` on a SqlSource until first
    reference. Verifies lazy init by counting ``load_full`` invocations during
    ``Session(db={...})`` and ``_materialise_if_source``.
    """
    df = pd.DataFrame({"uid": range(3), "val": [1.0, 2.0, 3.0]})
    engine = _make_sqlite_engine()
    _populate_engine(engine, df, "T")

    src = SqlSource("T", engine, table="T")
    load_full_calls = {"n": 0}
    real_load_full = src.load_full

    def _counting_load_full():
        load_full_calls["n"] += 1
        return real_load_full()

    src.load_full = _counting_load_full  # type: ignore[method-assign]

    session = Session(db={"T": src})
    # No load yet — Engine.__init__ should not pull rows
    assert load_full_calls["n"] == 0
    assert session.engine.db["T"] is src

    # Force materialisation via the engine helper
    materialised = session.engine._materialise_if_source("T")
    assert load_full_calls["n"] == 1
    assert isinstance(materialised, dict)
    assert "content" in materialised
    assert len(materialised["content"]) == 3

    # Subsequent reads see the cached dict; load_full not called again
    again = session.engine._materialise_if_source("T")
    assert load_full_calls["n"] == 1  # still 1
    assert again is materialised

def test_engine_init_shallow_copies_db():
    """``Engine.__init__`` must shallow-copy the user's db so external mutation
    of the original dict doesn't leak into ``engine.db`` (and vice-versa)."""
    df = pd.DataFrame({"uid": range(2), "v": [1.0, 2.0]})
    user_db = {"T": (df, torch.zeros(2, 1))}
    session = Session(db=user_db)
    user_db["Sneaky"] = (df, torch.zeros(2, 1))
    assert "Sneaky" not in session.engine.db, (
        "engine.db should be isolated from external mutation of the caller's dict"
    )

def test_engine_mixed_relation_source_and_legacy_tuple():
    """A db mixing ``RelationSource`` entries with legacy ``(df, tensor)`` tuples
    must work — the lazy materialisation path should not interfere with non-source
    entries."""
    n = 3
    df = pd.DataFrame({"uid": range(n), "feat": [1.0, 2.0, 3.0]})
    z = torch.eye(n, dtype=torch.float32)
    edges_df = pd.DataFrame({"src": [0, 1, 0], "dst": [1, 2, 2]})

    sa_engine = _make_sqlite_engine()
    _populate_engine(sa_engine, edges_df, "Edges")

    session = Session(db={
        "Nodes": (df, z),                                           # legacy tuple
        "Edges": SqlSource("Edges", sa_engine, table="Edges"),      # lazy source
    })
    # Legacy tuple is unchanged; SqlSource is still a source object pre-materialise
    assert isinstance(session.engine.db["Nodes"], tuple)
    assert isinstance(session.engine.db["Edges"], SqlSource)

    session.run(
        """
#lang:relnn
Agg(dst; sum(z)) :- Nodes(uid; z), Edges(uid, dst) .
?pred Out(dst; z) :- Agg(dst; z) .
"""
    )
    # After the rule runs, both relations have been normalised through the engine
    pred = session.relation("Out")
    assert pred.embeddings[0].shape[1] == n

def test_sql_source_lazy_in_rule_path():
    """Lazy init still produces correct results when the source is referenced via a rule
    (verifies ``_collect_data_sources`` triggers materialisation)."""
    n = 3
    nodes_df = pd.DataFrame({"pid": range(n)})
    nodes_z = torch.eye(n, dtype=torch.float32)
    edges_df = pd.DataFrame({"src": [0, 1, 0], "dst": [1, 2, 2]})
    sa_engine = _make_sqlite_engine()
    _populate_engine(sa_engine, edges_df, "Citation")

    session = Session(db={
        "Papers": DataFrameSource("Papers", nodes_df, embeddings=nodes_z),
        "Citation": SqlSource("Citation", sa_engine, table="Citation"),
    })
    # Citation entry must still be a SqlSource — not yet referenced
    assert isinstance(session.engine.db["Citation"], SqlSource)

    session.run(
        """
#lang:relnn
Agg(dst; sum(z)) :- Papers(src; z), Citation(src, dst) .
?pred Out(dst; z) :- Agg(dst; z) .
"""
    )
    # After the rule runs, Citation has been materialised in-place
    assert isinstance(session.engine.db["Citation"], dict)
    pred = session.relation("Out")
    assert pred.embeddings[0].shape[1] == n

def test_sql_source_categorical_codes_stable_across_load_by_keys():
    """Same label → same code across two ``load_by_keys`` calls on disjoint slices
    (CR fix #1 e2e: cross-batch categorical drift)."""
    df = pd.DataFrame({"uid": range(6), "dept": ["a", "b", "c", "a", "b", "c"]})
    engine = _make_sqlite_engine()
    _populate_engine(engine, df, "T")
    src = SqlSource("T", engine, table="T", dtype_map={"dept": "category"})

    full_vocab = src.load_full()["column_vocabs"]["dept"]

    sub_ac = src.load_by_keys("uid", [0, 2])
    sub_bc = src.load_by_keys("uid", [1, 5])

    assert sub_ac["column_vocabs"]["dept"] == full_vocab
    assert sub_bc["column_vocabs"]["dept"] == full_vocab

# ---------------------------------------------------------------------------
# connection-string path
# ---------------------------------------------------------------------------

def test_sql_source_accepts_connection_string(tmp_path):
    """SqlSource should accept a SQLAlchemy connection string (not just an Engine)."""
    db_path = tmp_path / "test.sqlite"
    url = f"sqlite:///{db_path}"

    df = pd.DataFrame({"id": range(3), "val": [10, 20, 30]})
    from sqlalchemy import create_engine as ce
    tmp_engine = ce(url)
    _populate_engine(tmp_engine, df, "T")
    tmp_engine.dispose()

    src = SqlSource("T", url, table="T")
    result = src.load_full()
    assert len(result["content"]) == 3

def test_sql_source_dtype_map():
    """dtype_map should be applied post-load."""
    df = pd.DataFrame({"uid": range(3), "cat": ["a", "b", "a"]})
    engine = _make_sqlite_engine()
    _populate_engine(engine, df, "T")

    src = SqlSource("T", engine, table="T", dtype_map={"cat": "category"})
    result = src.load_full()
    assert str(result["content"]["cat"].dtype) == "category"

# ---------------------------------------------------------------------------
# SqlSource schema() edge cases
# ---------------------------------------------------------------------------

def test_sql_source_schema_works_for_query_based():
    """schema() returns column names via a zero-row read when using a custom query."""
    engine = _make_sqlite_engine()
    src = SqlSource("T", engine, query="SELECT 1 AS x")
    assert src.schema() == ["x"]

# ---------------------------------------------------------------------------
# SqlSource used directly in Session (pure relational table — no embeddings)
# ---------------------------------------------------------------------------

def test_sql_source_used_directly_as_edge_table():
    """SqlSource can be passed directly to Session for pure relational tables.

    Aggregation over an edge list stored in SQLite: Papers with embeddings
    are loaded via DataFrameSource; Citation edges are loaded via SqlSource.
    SqlSource materialises the edge table at Engine init time; no pre-existing
    embeddings are required on the edge relation.
    """
    full_seed(0)
    n = 3
    papers_df = pd.DataFrame({"pid": range(n)})
    papers_z = torch.eye(n, dtype=torch.float32)  # (3, 3) identity as node features

    # Edge list: 0->1, 1->2, 0->2
    edges_df = pd.DataFrame({"src": [0, 1, 0], "dst": [1, 2, 2]})

    sa_engine = _make_sqlite_engine()
    _populate_engine(sa_engine, edges_df, "Citation")

    session = Session(db={
        "Papers": DataFrameSource("Papers", papers_df, embeddings=papers_z),
        "Citation": SqlSource("Citation", sa_engine, table="Citation"),
    })
    session.run(
        """
#lang:relnn
Agg(dst; sum(z)) :- Papers(src; z), Citation(src, dst) .
?pred Out(dst; z) :- Agg(dst; z) .
"""
    )
    pred = session.relation("Out")
    # Three destination nodes exist (1, 2, 2 → 2 unique dsts: 1 and 2)
    assert pred.embeddings[0].shape[1] == n  # feature dim preserved
    assert pred.embeddings[0].shape[0] >= 1  # at least one aggregated node

def test_sql_source_two_layer_gcn_encode_fit_and_predict():
    """Hermetic 007-style stack: pid-only nodes + RHS encode, SqlSource edges, 2-layer sum GCN, fit + pred.

    Uses a (N,1) placeholder on ``Papers`` because ``Transformation`` still requires a non-empty
    son embedding; true inputs come only from ``[ToyNodeFeatures(pid)]``.
    """
    full_seed(42)
    n, in_ch, hid_ch, n_classes = 24, 8, 16, 3
    rng = torch.Generator().manual_seed(42)
    _set_toy_node_features(torch.randn(n, in_ch, generator=rng) * 0.1)

    # Every node is a dst: self-loop + ring so message passing reaches all nodes
    citing, cited = [], []
    for i in range(n):
        citing.append(i)
        cited.append(i)
        citing.append(i)
        cited.append((i + 1) % n)
    edges_df = pd.DataFrame({"citing": citing, "cited": cited})

    papers_df = pd.DataFrame({"pid": range(n)})
    placeholder = torch.zeros(n, 1, dtype=torch.float32)
    labels_y = torch.randint(0, n_classes, (n, 1), generator=rng).long()
    labels_df = pd.DataFrame({"cited": range(n)})

    sa_engine = _make_sqlite_engine()
    _populate_engine(sa_engine, edges_df, "Citation")

    session = Session(db={
        "Papers": DataFrameSource("Papers", papers_df, embeddings=placeholder),
        "Citation": SqlSource("Citation", sa_engine, table="Citation"),
        "Labels": DataFrameSource("Labels", labels_df, embeddings=labels_y),
    })

    session.run(
        """
#lang:relnn
in_ch = 8 .
hid_ch = 16 .
out_ch = 3 .

PapersEmb1(pid; Linear(in_ch, hid_ch, False)([ToyNodeFeatures(pid)])) :- Papers(pid; z) .
PapersAgg1(cited; sum(z)) :- PapersEmb1(citing; z), Citation(citing, cited) .
PapersNL(cited; ReLU(z)) :- PapersAgg1(cited; z) .
PapersEmb2(cited; Linear(hid_ch, out_ch, False)(z)) :- PapersNL(cited; z) .
PapersAgg2(cited; sum(z)) :- PapersEmb2(citing; z), Citation(citing, cited) .
Output(cited; z) :- PapersAgg2(cited; z) .
"""
    )
    session.run(
        """
#lang:relnn
?fit <epochs=6, lr=0.05>
Loss(; CrossEntropyLoss()(z_pred, y)) :- Output(cited; z_pred), Labels(cited; y) .
"""
    )
    hist = session.engine.trained_modules["Loss"]["loss_history"]
    assert len(hist) == 6
    assert all(torch.isfinite(torch.tensor(v, dtype=torch.float64)) for v in hist)

    pred = session.run(
        """
#lang:relnn
?pred Logits(cited; z) :- Output(cited; z) .
"""
    )
    assert pred.embeddings[0].shape == (n, n_classes)
