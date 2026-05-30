"""
Integration tests for inductive / changing-database scenarios.

These tests verify that:
1. After fit, the database can be updated and predict uses the new data.
2. Entities never seen during training can be predicted on (inductive).
3. Different row counts between fit and predict are handled correctly.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from relann.session import Session
from relann.torch_utils import full_seed
from relann.era_operations import EmbeddedRelation

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_db(n_nodes: int, n_edges: int, feature_dim: int, n_classes: int, seed: int = 0):
    """Build a small relational DB: Nodes(id; features), Edges(src, dst;), Labels(id; class)."""
    rng = torch.Generator().manual_seed(seed)
    node_ids = list(range(n_nodes))
    edge_src = torch.randint(0, n_nodes, (n_edges,), generator=rng).tolist()
    edge_dst = torch.randint(0, n_nodes, (n_edges,), generator=rng).tolist()

    nodes_df = pd.DataFrame({"nid": node_ids})
    nodes_emb = torch.randn(n_nodes, feature_dim, generator=rng)

    edges_df = pd.DataFrame({"src": edge_src, "dst": edge_dst})
    edges_emb = torch.ones(n_edges, 1)

    label_vals = torch.randint(0, n_classes, (n_nodes, 1), generator=rng).float()
    labels_df = pd.DataFrame({"nid": node_ids})

    return {
        "Nodes": (nodes_df, nodes_emb),
        "Edges": (edges_df, edges_emb),
        "Labels": (labels_df, label_vals),
    }

# ── 1. Predict on different data after fit ──────────────────────────────────

def test_predict_on_different_data():
    """Train on one dataset, swap the database, predict on a different dataset."""
    full_seed(42)
    train_db = _make_db(n_nodes=30, n_edges=60, feature_dim=8, n_classes=3, seed=0)
    session = Session(db=train_db)

    session.run("""
NodeEmb(nid; Linear(8, 4)(z)) :- Nodes(nid; z) .
Agg(dst; sum(z * w)) :- NodeEmb(src; z), Edges(src, dst; w) .
Output(dst; ReLU()(z)) :- Agg(dst; z) .
""")
    session.run("""
?fit <epochs=5, lr=0.01> Loss(; CrossEntropyLoss()(Linear(4, 3)(z_pred), z)) :- Output(nid; z_pred), Labels(nid; z) .
""")

    # Now swap the database with different data (different number of nodes/edges)
    test_db = _make_db(n_nodes=20, n_edges=40, feature_dim=8, n_classes=3, seed=99)
    session.engine.db = test_db

    result = session.run("""
?pred Predictions(dst; ArgMax()(Linear(4, 3)(z))) :- Output(dst; z) .
""")
    assert result is not None
    assert isinstance(result, EmbeddedRelation)
    assert result.embeddings[0].shape[0] > 0
    # The test data has 20 nodes -- predict should produce results for unique dst values
    n_pred = result.content.shape[0]
    assert n_pred > 0 and n_pred <= 20

# ── 2. Inductive: new entities at test time ─────────────────────────────────

def test_inductive_new_entities():
    """Train on entities 0..79; predict on entities 80..99 (never seen during training)."""
    full_seed(42)

    n_train, n_test, d = 80, 20, 6
    # Training data: entities 0..79
    train_nodes_df = pd.DataFrame({"nid": range(n_train)})
    train_nodes_emb = torch.randn(n_train, d)
    train_labels_df = pd.DataFrame({"nid": range(n_train)})
    train_labels_emb = torch.randint(0, 3, (n_train, 1)).float()

    train_db = {
        "Nodes": (train_nodes_df, train_nodes_emb),
        "Labels": (train_labels_df, train_labels_emb),
    }
    session = Session(db=train_db)

    session.run("""
NodeEmb(nid; Linear(6, 4)(z)) :- Nodes(nid; z) .
""")
    session.run("""
?fit <epochs=5, lr=0.01> Loss(; CrossEntropyLoss()(Linear(4, 3)(z_pred), z)) :- NodeEmb(nid; z_pred), Labels(nid; z) .
""")

    # Inductive: swap to entities 80..99 (never seen during training)
    test_nodes_df = pd.DataFrame({"nid": range(n_train, n_train + n_test)})
    test_nodes_emb = torch.randn(n_test, d)
    session.engine.db["Nodes"] = (test_nodes_df, test_nodes_emb)

    result = session.run("""
?pred Out(nid; ArgMax()(Linear(4, 3)(z))) :- NodeEmb(nid; z) .
""")
    assert result is not None
    assert result.embeddings[0].shape == (n_test, 1)
    assert result.content.shape[0] == n_test
    # Verify the node IDs are from the test set
    pred_ids = sorted(result.content.iloc[:, 0].tolist())
    assert pred_ids == list(range(n_train, n_train + n_test))

# ── 3. Different row counts between fit and predict ─────────────────────────

def test_fit_predict_different_row_counts():
    """Train on N=100 rows, predict on M=50 rows; verify content/embedding sync."""
    full_seed(42)

    def _simple_db(n, seed):
        rng = torch.Generator().manual_seed(seed)
        df = pd.DataFrame({"a": range(n)})
        x = torch.randn(n, 4, generator=rng)
        y = torch.randn(n, 1, generator=rng)
        return {"Features": (df.copy(), x), "Targets": (df.copy(), y)}

    session = Session(db=_simple_db(100, seed=0))
    session.run("""
Pred(a; Linear(4, 1)(z)) :- Features(a; z) .
""")
    session.run("""
?fit <epochs=3, lr=0.01> Loss(; MSELoss()(z_pred, z)) :- Pred(a; z_pred), Targets(a; z) .
""")

    # Switch to 50-row data for prediction
    session.engine.db = _simple_db(50, seed=1)
    result = session.run("""
?pred Out(a; z) :- Pred(a; z) .
""")
    assert result.content.shape[0] == 50
    assert result.embeddings[0].shape[0] == 50

# ── runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_predict_on_different_data()
    test_inductive_new_entities()
    test_fit_predict_different_row_counts()
    print("All inductive / data-change tests passed.")
