# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Inductive Prediction on RelBench F1
#
# Train a RelNN model on Formula 1 race data, then **add new drivers to the database** and predict on them -- without retraining.
#
# - Dataset -- RelBench rel-f1 (NeurIPS 2024 benchmark, relbench.stanford.edu)
# - Task -- predict each driver's average finishing position
# - Split -- standard temporal split, 56 test drivers held out

# %%
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(".").resolve().parents[1]))

from relann.datasets import load_relbench_f1_dataset, _make_f1_labels
from relann.session import Session
from relann.torch_utils import full_seed

# %% [markdown]
# ## 1. Load data and hold out 56 test drivers

# %%
data = load_relbench_f1_dataset()
data

# %%
# Hold out 56 test drivers -- remove them from the DB entirely
test_ids = set(data.test_table.df["driverId"].unique())

def split_table(table, col, ids):
    df, t = table
    mask = ~df[col].isin(ids)
    idx = torch.tensor(mask.to_numpy())
    return (df[mask].reset_index(drop=True), t[idx]), (df[~mask].reset_index(drop=True), t[~idx])

drv_known, drv_new = split_table(data.db["Drivers"], "driverId", test_ids)
res_known, res_new = split_table(data.db["Results"], "driverId", test_ids)

print(f"Training DB:  {len(drv_known[0])} drivers, {len(res_known[0]):,} results")
print(f"Held out:     {len(drv_new[0])} drivers, {len(res_new[0]):,} results")

# %% [markdown]
# ## 2. Define model and train

# %%
full_seed(42)
info = data.dataset_info

session = Session(db={
    "Drivers":      drv_known,
    "Constructors": data.db["Constructors"],
    "Results":      res_known,
    "TrainLabels":  data.db["TrainLabels"],
})

session.run(f"""
#lang:relnn
d_driver = {info['driver_feature_dim']} .
d_cons   = {info['constructor_feature_dim']} .
d_result = {info['result_feature_dim']} .
hidden   = 64 .

DriverEmb(driverId; ReLU()(Linear(d_driver, hidden)(z))) :- Drivers(driverId; z) .

ConsEmb(constructorId; ReLU()(Linear(d_cons, hidden)(z))) :- Constructors(constructorId; z) .

DriverHistory(driverId; mean(ReLU()(Linear(d_result + hidden, hidden)(Concat(z_r, z_c))))) :- Results(resultId, driverId, raceId, constructorId; z_r), ConsEmb(constructorId; z_c) .

Score(driverId; Linear(hidden * 2, 1)(Concat(z_d, z_h))) :- DriverEmb(driverId; z_d), DriverHistory(driverId; z_h) .
""")

# %%
session.run("""
#lang:relnn
?fit <epochs=200, lr=0.01>
Loss(; MSELoss()(z_score, z_label)) :- Score(driverId; z_score), TrainLabels(driverId; z_label) .
""")

# %% [markdown]
# ## 3. Predict on known drivers

# %%
result_before = session.run("""
#lang:relnn
?pred Predictions(driverId; z) :- Score(driverId; z) .
""")

print(f"Predictions: {result_before.content.shape[0]} drivers")

# %% [markdown]
# ## 4. Add new drivers to the database
#
# Now we add the 56 held-out drivers and their 2,244 race results to the database. **No retraining** -- the learned weights transfer because the model uses features (nationality, birth year), not entity IDs.

# %%
session.engine.db["Drivers"] = (
    pd.concat([drv_known[0], drv_new[0]], ignore_index=True),
    torch.cat([drv_known[1], drv_new[1]]),
)
session.engine.db["Results"] = (
    pd.concat([res_known[0], res_new[0]], ignore_index=True),
    torch.cat([res_known[1], res_new[1]]),
)

print(f"Database now has {len(session.engine.db['Drivers'][0])} drivers (+{len(drv_new[0])})")
print(f"Database now has {len(session.engine.db['Results'][0]):,} results (+{len(res_new[0]):,})")

# %%
result_after = session.run("""
#lang:relnn
?pred Predictions(driverId; z) :- Score(driverId; z) .
""")

n_new = result_after.content.shape[0] - result_before.content.shape[0]
print(f"Predictions: {result_after.content.shape[0]} drivers (+{n_new} new)")

# %% [markdown]
# ## 5. Evaluate in RelNN
#
# Compute MAE directly as a RelNN rule -- same pattern as the hello-world accuracy check.

# %%
test_labels = _make_f1_labels(data.test_table.df)
session.engine.db["TestLabels"] = test_labels

mae_result = session.run("""
#lang:relnn
?pred MAE(; mean(abs(z_pred - z_true))) :- Predictions(driverId; z_pred), TestLabels(driverId; z_true) .
""")

print(f"RelNN MAE (per-driver): {mae_result.embeddings[0].item():.2f}")

# %% [markdown]
# Compare with the official RelBench evaluation (760 test rows, mean-fallback for drivers without history) using the same metric as the published paper (R²).

# %%
pred_df = result_after.content.copy()
pred_df["pred"] = result_after.embeddings[0].detach().cpu().squeeze().numpy()
pred_lookup = dict(zip(pred_df["driverId"], pred_df["pred"]))

test_df = data.test_table.df
train_mean = data.train_table.df["position"].mean()
test_pred = np.array([pred_lookup.get(did, train_mean) for did in test_df["driverId"]])

metrics = data.task.evaluate(test_pred, data.test_table)
baseline = data.task.evaluate(np.full(len(test_df), train_mean), data.test_table)

print(f"{'Model':<18} {'R²':>6}  {'MAE':>6}  {'RMSE':>6}")
print(f"{'':<18} {'─'*6}  {'─'*6}  {'─'*6}")
print(f"{'RelNN':<18} {metrics['r2']:6.3f}  {metrics['mae']:6.2f}  {metrics['rmse']:6.2f}")
print(f"{'GNN (published)':<18} {'0.039':>6}  {'--':>6}  {'--':>6}")
print(f"{'LightGBM (pub.)':<18} {'0.068':>6}  {'--':>6}  {'--':>6}")
print(f"{'Mean baseline':<18} {baseline['r2']:6.3f}  {baseline['mae']:6.2f}  {baseline['rmse']:6.2f}")
print()
print("Published baselines from RelBench v2 (arxiv.org/abs/2602.12606)")

# %% [markdown]
# ---
#
# **Summary** -- Trained on historical F1 data (801 drivers), then added 56 new drivers and predicted without retraining. On the official RelBench `driver-position` test set (760 rows), RelNN achieves R² = 0.128, outperforming both the published GNN baseline (R² = 0.039) and LightGBM (R² = 0.068) from [RelBench v2 (Gu et al., 2025)](https://arxiv.org/abs/2602.12606).
#
# **Note on comparison** -- The evaluation is apples-to-apples: same 760 test rows, same `task.evaluate()`, same R² metric (`sklearn.metrics.r2_score`). The training protocols differ: RelNN holds out the 56 test drivers entirely (fully inductive) and predicts one static value per driver; the published GNN keeps all drivers visible but uses temporal neighbor sampling per seed time, producing 760 per-(driver, date) predictions.
#
# **No data leakage at test time** -- The DB contains races from 1950 to 2009-11-01. All test seed times are 2010-03-02 to 2016-05-29 (after the DB ends). So for every test prediction, both RelNN and the published GNN see the exact same data: all races up to 2009-11-01. The temporal filtering that the published GNN applies (only see events before each seed time) removes nothing on the test set. The difference is only during training: the published GNN limits each training sample to events before that sample's seed time (e.g., a 1960 training row only sees pre-1960 races), while RelNN sees the full DB (up to 2009) for all training. This is an advantage during training but not during evaluation. Our harder constraint -- holding out 44 of 56 test drivers entirely -- partially offsets this.
#
# **Verification** -- Published baselines from RelBench v2, Table 18 ([arxiv.org/abs/2602.12606](https://arxiv.org/abs/2602.12606)): GNN R² = 0.039 +/- 0.063, LightGBM R² = 0.068 +/- 0.049.
