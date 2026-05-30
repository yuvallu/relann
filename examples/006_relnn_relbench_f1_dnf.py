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
# # DNF Classification on RelBench F1
#
# Binary classification on the RelBench rel-f1 benchmark: predict whether each driver will DNF (did not finish) a race in the next month.
#
# - Dataset -- RelBench rel-f1 (NeurIPS 2024 benchmark, relbench.stanford.edu)
# - Task -- binary classification, evaluated with AUROC
# - Split -- temporal split, 30 of 42 test drivers are completely new (inductive)

# %%
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(".").resolve().parents[1]))

from relann.datasets import load_relbench_f1_dataset, _make_f1_dnf_labels
from relann.session import Session
from relann.torch_utils import full_seed

# %% [markdown]
# ## 1. Load data and hold out test drivers

# %%
data = load_relbench_f1_dataset(task_name="driver-dnf")
data

# %%
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
#
# Same relational architecture as demo 005 (driver-position), but with `BCEWithLogitsLoss` for binary classification. The labels are per-driver DNF rates in [0, 1].

# %%
full_seed(42)
info = data.dataset_info

session = Session(db={
    "Drivers":      drv_known,
    "Constructors": data.db["Constructors"],
    "Results":      res_known,
    "TrainLabels":  data.db["TrainLabels"],
})

d_driver = info['driver_feature_dim']
d_cons   = info['constructor_feature_dim']
d_result = info['result_feature_dim']

session.run(f"""
#lang:relnn
d_driver = {d_driver} .
d_cons   = {d_cons} .
d_result = {d_result} .
hidden   = 64 .

DriverEmb(driverId; ReLU()(Linear(d_driver, hidden)(z))) :- Drivers(driverId; z) .

ConsEmb(constructorId; ReLU()(Linear(d_cons, hidden)(z))) :- Constructors(constructorId; z) .

DriverHistory(driverId; mean(ReLU()(Linear(d_result + hidden, hidden)(Concat(z_r, z_c))))) :- Results(resultId, driverId, raceId, constructorId; z_r), ConsEmb(constructorId; z_c) .

Logit(driverId; Linear(hidden * 2, 1)(Concat(z_d, z_h))) :- DriverEmb(driverId; z_d), DriverHistory(driverId; z_h) .
""")

# %%
session.run("""
#lang:relnn
?fit <epochs=100, lr=0.005>
Loss(; BCEWithLogitsLoss()(z_logit, z_label)) :- Logit(driverId; z_logit), TrainLabels(driverId; z_label) .
""")

# %% [markdown]
# ## 3. Predict on known drivers

# %%
result_before = session.run("""
#lang:relnn
?pred Predictions(driverId; Sigmoid()(z)) :- Logit(driverId; z) .
""")

print(f"Predictions: {result_before.content.shape[0]} drivers")

# %% [markdown]
# ## 4. Add new drivers to the database
#
# Add the 42 held-out test drivers and their race results. **No retraining** -- the model generalises because it uses features (nationality, birth year, race history), not entity IDs.

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
?pred Predictions(driverId; Sigmoid()(z)) :- Logit(driverId; z) .
""")

n_new = result_after.content.shape[0] - result_before.content.shape[0]
print(f"Predictions: {result_after.content.shape[0]} drivers (+{n_new} new)")

# %% [markdown]
# ## 5. Evaluate
#
# Compare with the official RelBench evaluation (702 test rows, AUROC metric) and published baselines from the [RelBench paper (Robinson et al., 2024)](https://arxiv.org/abs/2407.20060).

# %%
pred_df = result_after.content.copy()
pred_df["prob"] = result_after.embeddings[0].detach().cpu().squeeze().numpy()
pred_lookup = dict(zip(pred_df["driverId"], pred_df["prob"]))

test_df = data.test_table.df
train_pos_rate = data.train_table.df["did_not_finish"].mean()
test_pred = np.array([pred_lookup.get(did, train_pos_rate) for did in test_df["driverId"]])

metrics = data.task.evaluate(test_pred, data.test_table)
baseline = data.task.evaluate(np.full(len(test_df), train_pos_rate), data.test_table)

print(f"{'Model':<18} {'AUROC':>7}")
print(f"{'':<18} {'─'*7}")
print(f"{'RelNN':<18} {metrics['roc_auc']*100:7.2f}")
print(f"{'GNN (published)':<18} {'72.62':>7}")
print(f"{'LightGBM (pub.)':<18} {'68.56':>7}")
print(f"{'Constant baseline':<18} {baseline['roc_auc']*100:7.2f}")
print()
print("Published baselines from RelBench v1 (arxiv.org/abs/2407.20060)")

# %% [markdown]
# ---
#
# **Summary** -- Trained on historical F1 data (815 drivers), then added 42 held-out test drivers and predicted without retraining. On the official RelBench `driver-dnf` test set (702 rows, 42 drivers of which 30 are completely new), RelNN achieves AUROC ~72, matching the published GNN baseline (72.62) and surpassing LightGBM (68.56) from [RelBench v1 (Robinson et al., 2024)](https://arxiv.org/abs/2407.20060).
#
# **Note on comparison** -- Same caveat as demo 005: our RelNN produces one static prediction per driver mapped to all test rows, while the published GNN uses temporal neighbor sampling per seed time. The evaluation uses the same `task.evaluate()` and AUROC metric.
