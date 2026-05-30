"""
Smoke test for the RelBench rel-f1 pipeline.

Verifies the full flow: load dataset from RelBench, define multi-table
RelNN model, fit (few epochs), predict, evaluate with RelBench's evaluator.

Skips gracefully if relbench is not installed.
"""

import sys
from pathlib import Path

import numpy as np
import torch
pytest = __import__("pytest")
relbench = pytest.importorskip("relbench")

from relann.datasets import load_relbench_f1_dataset, _make_f1_labels
from relann.session import Session
from relann.torch_utils import full_seed
from relann.era_operations import EmbeddedRelation

def test_relbench_f1_smoke():
    full_seed(42)
    data = load_relbench_f1_dataset()

    assert "Drivers" in data.db
    assert "Constructors" in data.db
    assert "Races" in data.db
    assert "Results" in data.db
    assert "TrainLabels" in data.db

    info = data.dataset_info
    assert info["n_train"] > 0
    assert info["n_test"] > 0
    assert info["n_new_test_drivers"] > 0

    d_driver = info["driver_feature_dim"]
    d_cons = info["constructor_feature_dim"]
    d_race = info["race_feature_dim"]
    d_result = info["result_feature_dim"]

    session = Session(db=data.db)
    session.run(f"""
d_driver = {d_driver} .
d_cons   = {d_cons} .
d_race   = {d_race} .
d_result = {d_result} .
hidden   = 32 .

DriverEmb(driverId; ReLU()(Linear(d_driver, hidden)(z))) :- Drivers(driverId; z) .
ConsEmb(constructorId; ReLU()(Linear(d_cons, hidden)(z))) :- Constructors(constructorId; z) .
RaceEmb(raceId; ReLU()(Linear(d_race, hidden)(z))) :- Races(raceId; z) .
DriverHistory(driverId; mean(ReLU()(Linear(d_result + hidden * 2, hidden)(Concat(z_r, z_c, z_race))))) :- Results(resultId, driverId, raceId, constructorId; z_r), ConsEmb(constructorId; z_c), RaceEmb(raceId; z_race) .
Score(driverId; Linear(hidden * 2, 1)(Concat(z_d, z_h))) :- DriverEmb(driverId; z_d), DriverHistory(driverId; z_h) .
""")

    session.run("""
?fit <epochs=5, lr=0.01> Loss(; MSELoss()(z_score, z_label)) :- Score(driverId; z_score), TrainLabels(driverId; z_label) .
""")

    loss_info = session.engine.trained_modules["Loss"]
    assert len(loss_info["loss_history"]) == 5
    assert all(np.isfinite(v) for v in loss_info["loss_history"])

    result = session.run("""
?pred Predictions(driverId; z) :- Score(driverId; z) .
""")
    assert result is not None
    assert isinstance(result, EmbeddedRelation)
    assert result.content.shape[0] > 0
    assert result.embeddings[0].shape[0] == result.content.shape[0]

    # Evaluate MAE in RelNN (like hello-world accuracy check)
    test_labels = _make_f1_labels(data.test_table.df)
    session.engine.db["TestLabels"] = test_labels

    mae_result = session.run("""
?pred MAE(; mean(abs(z_pred - z_true))) :- Predictions(driverId; z_pred), TestLabels(driverId; z_true) .
""")
    assert mae_result is not None
    relnn_mae = mae_result.embeddings[0].item()
    assert np.isfinite(relnn_mae), f"RelNN MAE not finite: {relnn_mae}"
    assert relnn_mae < 15.0, f"RelNN MAE unexpectedly large: {relnn_mae}"
    print(f"RelNN MAE (per-driver): {relnn_mae:.4f}")

    # Also check RelBench official evaluation
    pred_df = result.content.copy()
    pred_df["pred"] = result.embeddings[0].detach().cpu().squeeze().numpy()
    pred_lookup = dict(zip(pred_df["driverId"], pred_df["pred"]))

    test_df = data.test_table.df
    train_mean = data.train_table.df["position"].mean()
    test_pred = np.array([pred_lookup.get(did, train_mean) for did in test_df["driverId"]])

    metrics = data.task.evaluate(test_pred, data.test_table)
    assert np.isfinite(metrics["mae"]), f"MAE not finite: {metrics['mae']}"
    assert metrics["mae"] < 10.0, f"MAE unexpectedly large: {metrics['mae']}"
    assert np.isfinite(metrics["r2"]), f"R² not finite: {metrics['r2']}"
    assert metrics["r2"] > -5.0, f"R² unexpectedly low: {metrics['r2']}"
    print(f"RelBench R²: {metrics['r2']:.4f}  MAE: {metrics['mae']:.4f}")

if __name__ == "__main__":
    test_relbench_f1_smoke()
    print("RelBench F1 smoke test passed.")
