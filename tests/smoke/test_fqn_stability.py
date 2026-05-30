"""Smoke test: FQN stability for TransformDef inside different wrappers.

Verifies that a TransformDef (e.g. Classifier) gets the same parameter FQN
whether it appears inside CrossEntropyLoss (fit path) or ArgMax (predict path).
This is the exact scenario that broke HGT accuracy before the fix.

Run from repo root:
    python tests/smoke/test_fqn_stability.py
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from relann.session import Session
from relann.torch_utils import full_seed

def _simple_db(n=10, d=4, n_classes=3):
    """Minimal DB: nodes with features + labels."""
    nodes_df = pd.DataFrame({"nid": list(range(n))})
    nodes_z = torch.randn(n, d)
    labels_df = pd.DataFrame({"nid": list(range(n))})
    labels_z = torch.randint(0, n_classes, (n,)).float().unsqueeze(1)
    return {"Nodes": (nodes_df, nodes_z), "Labels": (labels_df, labels_z)}

def test_fqn_stable_classifier_in_cross_entropy():
    """Classifier TransformDef FQN must be the same during fit and predict."""
    full_seed(42)

    n, d, n_classes = 10, 4, 3
    db = _simple_db(n, d, n_classes)
    session = Session(db=db)

    session.run(f"""
#lang:relnn
Classifier = Linear({d}, {n_classes}) .
Output(nid; z) :- Nodes(nid; z) .
""")

    session.run(f"""
#lang:relnn
?fit <epochs=5, lr=0.01>
Loss(; CrossEntropyLoss()(Classifier(z_pred), z)) :- Output(target_id; z_pred), Labels(target_id; z) .
""")

    store = session.engine.parameter_store
    fit_keys = sorted(store.keys())
    print(f"  FQNs after fit: {fit_keys}")

    assert any("Classifier" in k for k in fit_keys), (
        f"No FQN contains 'Classifier' -- got {fit_keys}"
    )
    for k in fit_keys:
        if "Classifier" in k:
            assert "inner" not in k, f"FQN leaks compiler wrapper 'inner': {k}"
            assert "_module" not in k, f"FQN leaks compiler wrapper '_module': {k}"

    classifier_weight_fqn = [k for k in fit_keys if "Classifier" in k and "weight" in k]
    classifier_bias_fqn = [k for k in fit_keys if "Classifier" in k and "bias" in k]
    assert len(classifier_weight_fqn) == 1, f"Expected 1 Classifier weight FQN, got {classifier_weight_fqn}"
    assert len(classifier_bias_fqn) == 1, f"Expected 1 Classifier bias FQN, got {classifier_bias_fqn}"

    trained_weight = store[classifier_weight_fqn[0]].clone()
    trained_bias = store[classifier_bias_fqn[0]].clone()

    pred = session.run(f"""
#lang:relnn
?pred Cls(nid; ArgMax()(Classifier(z))) :- Output(nid; z) .
""")

    post_pred_keys = sorted(session.engine.parameter_store.keys())
    assert fit_keys == post_pred_keys, (
        f"FQNs changed between fit and predict:\n  fit:  {fit_keys}\n  pred: {post_pred_keys}"
    )

    post_weight = store[classifier_weight_fqn[0]]
    post_bias = store[classifier_bias_fqn[0]]
    assert torch.equal(trained_weight, post_weight), "Classifier weight changed after predict!"
    assert torch.equal(trained_bias, post_bias), "Classifier bias changed after predict!"

    assert pred is not None
    print(f"  Prediction shape: {tuple(pred.embeddings[0].shape)}")
    print("  PASS: FQN stability for Classifier inside CrossEntropyLoss")

def test_fqn_stable_multiple_transform_defs_in_loss():
    """Multiple TransformDefs inside one loss rule must each get stable FQNs."""
    full_seed(42)

    n, d, hidden, n_classes = 10, 4, 3, 2
    db = _simple_db(n, d, n_classes)
    session = Session(db=db)

    session.run(f"""
#lang:relnn
Proj = Linear({d}, {hidden}) .
Classifier = Linear({hidden}, {n_classes}) .
Output(nid; Proj(z)) :- Nodes(nid; z) .
""")

    session.run(f"""
#lang:relnn
?fit <epochs=5, lr=0.01>
Loss(; CrossEntropyLoss()(Classifier(z_pred), z)) :- Output(target_id; z_pred), Labels(target_id; z) .
""")

    store = session.engine.parameter_store
    fit_keys = sorted(store.keys())
    print(f"  FQNs after fit: {fit_keys}")

    proj_keys = [k for k in fit_keys if "Proj" in k]
    cls_keys = [k for k in fit_keys if "Classifier" in k]
    assert len(proj_keys) >= 2, f"Expected Proj weight+bias, got {proj_keys}"
    assert len(cls_keys) >= 2, f"Expected Classifier weight+bias, got {cls_keys}"

    for k in fit_keys:
        assert "inner" not in k, f"FQN leaks 'inner': {k}"
        assert "_module" not in k, f"FQN leaks '_module': {k}"

    pred = session.run(f"""
#lang:relnn
?pred Cls(nid; ArgMax()(Classifier(z))) :- Output(nid; z) .
""")

    post_keys = sorted(session.engine.parameter_store.keys())
    assert fit_keys == post_keys, (
        f"FQNs changed between fit/predict:\n  fit:  {fit_keys}\n  pred: {post_keys}"
    )

    assert pred is not None
    print(f"  Prediction shape: {tuple(pred.embeddings[0].shape)}")
    print("  PASS: FQN stability for multiple TransformDefs in loss")

if __name__ == "__main__":
    test_fqn_stable_classifier_in_cross_entropy()
    test_fqn_stable_multiple_transform_defs_in_loss()
    print("\nOK: All FQN stability tests passed.")
