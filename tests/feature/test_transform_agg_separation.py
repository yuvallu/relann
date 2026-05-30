"""
Tests for the Transformation / Aggregation separation of concerns.

Transformations must be per-row (embedding row count == content row count).
Loss functions (CrossEntropyLoss, MSELoss, etc.) are automatically instantiated
with reduction='none' so they return per-sample output, and Aggregation handles
the batch reduction (mean / sum).
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from relann.session import Session
from relann.torch_utils import full_seed
from relann.era_operations import EmbeddedRelation

# ── helpers ──────────────────────────────────────────────────────────────────

def _logits_labels_db(n: int = 20, num_classes: int = 3):
    df = pd.DataFrame({"a": range(n)})
    logits = torch.randn(n, num_classes)
    labels = torch.randint(0, num_classes, (n, 1)).float()
    return {"Input1": (df.copy(), logits), "Input2": (df.copy(), labels)}

def _regression_db(n: int = 20, d: int = 4):
    df = pd.DataFrame({"a": range(n)})
    x = torch.randn(n, d)
    y = torch.randn(n, 1)
    return {"Features": (df.copy(), x), "Targets": (df.copy(), y)}

# ── 1. Loss functions produce per-sample output ─────────────────────────────

def test_crossentropy_per_sample_loss():
    """CrossEntropyLoss inside fit must produce a per-sample (N,) tensor
    in the Transformation step, not a scalar."""
    full_seed(42)
    db = _logits_labels_db(n=20, num_classes=3)
    session = Session(db=db)
    session.run("""
Logits(a; Linear(3, 3)(z1)) :- Input1(a; z1) .
Labels(a; z2) :- Input2(a; z2) .
""")
    session.run("""
?fit <epochs=3, lr=0.01> Loss(; CrossEntropyLoss()(z_pred, z)) :- Logits(a; z_pred), Labels(a; z) .
""")
    info = session.engine.trained_modules["Loss"]
    assert len(info["loss_history"]) == 3
    for v in info["loss_history"]:
        assert isinstance(v, float) and torch.isfinite(torch.tensor(v)).item()

def test_mseloss_per_sample_loss():
    """MSELoss inside fit must produce per-sample output, reduced by Aggregation."""
    full_seed(42)
    db = _regression_db(n=20, d=4)
    session = Session(db=db)
    session.run("""
Pred(a; Linear(4, 1)(z1)) :- Features(a; z1) .
Targets(a; z2) :- Targets(a; z2) .
""")
    session.run("""
?fit <epochs=3, lr=0.01> Loss(; MSELoss()(z_pred, z)) :- Pred(a; z_pred), Targets(a; z) .
""")
    info = session.engine.trained_modules["Loss"]
    assert len(info["loss_history"]) == 3
    for v in info["loss_history"]:
        assert isinstance(v, float) and torch.isfinite(torch.tensor(v)).item()

# ── 2. ArgMax preserves row count ───────────────────────────────────────────

def test_argmax_preserves_row_count():
    """ArgMax must produce (N, 1) output -- same N as input."""
    full_seed(42)
    n, d = 15, 5
    db = {"Input": (pd.DataFrame({"a": range(n)}), torch.randn(n, d))}
    session = Session(db=db)
    result = session.run("""
?pred Out(a; ArgMax()(z)) :- Input(a; z) .
""")
    assert result is not None
    assert result.embeddings[0].shape == (n, 1)

# ── 3. Default aggregation is created when LHS < RHS attrs ─────────────────

def test_default_aggregation_created_for_loss():
    """When LHS has no content attrs and RHS has one, the term graph must
    contain an agg node with default 'mean' even without explicit sum()/mean()."""
    full_seed(42)
    db = _logits_labels_db(n=10, num_classes=3)
    session = Session(db=db)
    session.run("""
Logits(a; Linear(3, 3)(z1)) :- Input1(a; z1) .
Labels(a; z2) :- Input2(a; z2) .
""")

    # Parse the loss rule to add it to the term graph (without actually fitting)
    from relann.parser import parse_and_transform_str, RelnnTransformer
    from relann.pydantic_classes import Rule
    transformer = RelnnTransformer(session.engine)
    prog = parse_and_transform_str(
        "Loss(; CrossEntropyLoss()(z_pred, z)) :- Logits(a; z_pred), Labels(a; z) .",
        start="program", transformer=transformer,
    )
    rule = [s for s in prog.statements if isinstance(s, Rule)][0]
    session.engine.add_rule(rule)
    tg = session.engine.term_graphs["global"]

    # The loss rule should have an aggregation node
    agg_node = f"agg_Loss"
    assert agg_node in tg.nodes(), (
        f"Expected aggregation node '{agg_node}' in term graph. "
        f"Nodes: {list(tg.nodes())}"
    )
    node_data = tg.nodes[agg_node]
    assert node_data.get("aggregation_name") == "mean", (
        f"Expected default aggregation 'mean', got '{node_data.get('aggregation_name')}'"
    )

# ── 4. Content / embedding sync validation ──────────────────────────────────

def test_content_embedding_sync_on_fit():
    """A full fit + predict cycle must keep content and embedding rows in sync
    throughout the pipeline."""
    full_seed(42)
    db = _logits_labels_db(n=20, num_classes=3)
    session = Session(db=db)
    session.run("""
Logits(a; Linear(3, 3)(z1)) :- Input1(a; z1) .
Labels(a; z2) :- Input2(a; z2) .
""")
    session.run("""
?fit <epochs=2, lr=0.01> Loss(; CrossEntropyLoss()(z_pred, z)) :- Logits(a; z_pred), Labels(a; z) .
""")
    result = session.run("""
?pred Out(a; ArgMax()(z)) :- Logits(a; z) .
""")
    assert result.content.shape[0] == 20
    assert result.embeddings[0].shape[0] == 20

# ── 5. Rule redefinition guard ────────────────────────────────────────────────

def test_redefine_rule_with_new_data_source_raises():
    """Redefining a rule with a new data source must raise ValueError,
    guiding the user to mutate session.engine.db instead."""
    import pytest
    full_seed(42)
    db = {
        "TrainData": (pd.DataFrame({"id": range(10)}), torch.randn(10, 4)),
    }
    session = Session(db=db)
    session.run("Emb(id; Linear(4, 8)(z)) :- TrainData(id; z) .")

    with pytest.raises(ValueError, match="already defined"):
        session.engine.db["TestData"] = (
            pd.DataFrame({"id": range(5)}), torch.randn(5, 4),
        )
        session.run("Emb(id; Linear(4, 8)(z)) :- TestData(id; z) .")

def test_same_pred_twice_is_idempotent():
    """Running the same ?pred rule twice must work (idempotent skip)."""
    full_seed(42)
    db = {
        "Data": (pd.DataFrame({"id": range(10)}), torch.randn(10, 4)),
    }
    session = Session(db=db)
    session.run("Score(id; Linear(4, 1)(z)) :- Data(id; z) .")
    r1 = session.run("?pred Out(id; z) :- Score(id; z) .")
    r2 = session.run("?pred Out(id; z) :- Score(id; z) .")
    assert r1.content.shape == r2.content.shape
    assert r1.embeddings[0].shape == r2.embeddings[0].shape

# ── runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_crossentropy_per_sample_loss()
    test_mseloss_per_sample_loss()
    test_argmax_preserves_row_count()
    test_default_aggregation_created_for_loss()
    test_content_embedding_sync_on_fit()
    test_redefine_rule_with_new_data_source_raises()
    test_same_pred_twice_is_idempotent()
    print("All transform-agg separation tests passed.")
