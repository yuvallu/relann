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
# # HyGNN on Drug-Drug Interaction: Hypergraph Neural Network in RelNN
#
# We implement [HyGNN](https://arxiv.org/pdf/2206.12747) (Saifuddin et al., 2022) in RelNN for Drug-Drug Interaction (DDI) prediction.
#
# **Architecture**: A hypergraph where drugs are hyperedges and chemical substructures (k-mers from SMILES) are nodes. Two-level attention:
# 1. **Hyperedge → Node**: aggregate drug information to substructures via attention
# 2. **Node → Hyperedge**: aggregate substructure information back to drugs via attention
#
# The learned drug embeddings are then used to predict whether drug pairs interact (binary classification).
#
# **Key insight**: A hypergraph with incidence matrix H is a bipartite graph between substructures and drugs — so the two-level attention maps directly onto RelNN's join + aggregation.

# %% [markdown]
# ## Imports

# %%
from relann.session import Session
from relann.datasets import load_hygnn_dataset
from relann.torch_utils import full_seed
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    accuracy_score, precision_score, recall_score,
)

# %% [markdown]
# ## 1. Load data
#
# Load the precomputed DDI dataset from the [HyGNN reference repository](https://github.com/shouhengtuo/HyGNN-Drug-Drug-Interaction-Prediction-via-Hypergraph-Neural-Network). The loader downloads the hypergraph incidence tensor (k-mer substructures ↔ drugs) and the DDI edge list, then builds balanced positive/negative DDI pairs with a 90/10 train/test split.
#
# Use `source='DrugBank'` for the larger dataset (1706 drugs); `source='TWOSIDES'` for smaller (645 drugs).

# %%
data = load_hygnn_dataset(source='TWOSIDES', k=3, d=128, seed=42)
data

# %%
info = data.dataset_info
n_drugs = info['n_drugs']
print(f"n_drugs={n_drugs}, n_subs={info['n_subs']}, incidence_edges={info['n_incidence']}")

# %% [markdown]
# ## 2. Define HyGNN in RelNN
#
# ### Constants and Softmax

# %%
full_seed(42)
session = Session(db=data.db)

# %%
session.run(f"""
#lang:relnn
n_drugs = {n_drugs} .
d = 128 .
qd = 64 .

def Softmax(Scores):
    MaxPerT(t; max(z))        :- Scores(s, t; z) .
    Exp(s, t; exp(z - max_t)) :- Scores(s, t; z), MaxPerT(t; max_t) .
    Denom(t; sum(z))          :- Exp(s, t; z) .
    Out(s, t; z1 / z2)        :- Exp(s, t; z1), Denom(t; z2) .
enddef
""")

# %% [markdown]
# ### Initial Projection
#
# Drugs start as one-hot identity vectors (n_drugs-dimensional). Project to d=128 via a linear layer — functionally equivalent to a learned embedding table.
#
# Substructures start as all-ones (d=128); no projection needed.

# %%
session.run("""
#lang:relnn
DrugProj(drug_id; Linear(n_drugs, d)(z)) :- Drug(drug_id; z) .
""")

# %% [markdown]
# ### Encoder Level 1: Hyperedge → Node (drugs → substructures)
#
# **Paper Eqs. (4–6)**: For each (drug, substructure) pair in the incidence, compute a scalar attention score, softmax over drugs per substructure, then weighted-aggregate drug values to get substructure embeddings.
#
# $$e_j = \text{LeakyReLU}(W_2 q_j \cdot W_3 p_i) / \sqrt{d_q}$$
# $$Y_{ij} = \text{softmax}_j(e_j)$$
# $$p_i = \sum_{e_j \in E_i} Y_{ij} \, W_1 q_j$$

# %%
session.run("""
#lang:relnn
W1 = Linear(d, d) .
W2 = Linear(d, qd) .
W3 = Linear(d, qd) .

K_E(drug_id; W2(z)) :- DrugProj(drug_id; z) .
V_E(drug_id; W1(z)) :- DrugProj(drug_id; z) .
Q_E(sub_id; W3(z)) :- Substructure(sub_id; z) .

AttnE_raw(drug_id, sub_id; LeakyReLU()(view(1)(z_q @ transpose(z_k))) / sqrt(qd)) :- K_E(drug_id; z_k), Incidence(sub_id, drug_id; w), Q_E(sub_id; z_q) .
AttnE(drug_id, sub_id; z) :- Softmax(AttnE_raw)(drug_id, sub_id; z) .
SubEmb(sub_id; sum(z_att * z_v)) :- AttnE(drug_id, sub_id; z_att), V_E(drug_id; z_v) .
""")

# %% [markdown]
# ### Encoder Level 2: Node → Hyperedge (substructures → drugs)
#
# **Paper Eqs. (7–9)**: Same attention pattern, reversed direction. K and V come from substructures; Q comes from the initial drug projection.
#
# $$v_i = \text{LeakyReLU}(W_5 p_i \cdot W_6 q_j) / \sqrt{d_q}$$
# $$X_{ji} = \text{softmax}_i(v_i)$$
# $$q_j = \sum_{v_i \in e_j} X_{ji} \, W_4 p_i$$

# %%
session.run("""
#lang:relnn
W4 = Linear(d, d) .
W5 = Linear(d, qd) .
W6 = Linear(d, qd) .

K_V(sub_id; W5(z)) :- SubEmb(sub_id; z) .
V_V(sub_id; W4(z)) :- SubEmb(sub_id; z) .
Q_V(drug_id; W6(z)) :- DrugProj(drug_id; z) .

AttnV_raw(sub_id, drug_id; LeakyReLU()(view(1)(z_q @ transpose(z_k))) / sqrt(qd)) :- K_V(sub_id; z_k), Incidence(sub_id, drug_id; w), Q_V(drug_id; z_q) .
AttnV(sub_id, drug_id; z) :- Softmax(AttnV_raw)(sub_id, drug_id; z) .
DrugEmb_from_attn(drug_id; sum(z_att * z_v)) :- AttnV(sub_id, drug_id; z_att), V_V(sub_id; z_v) .
DrugEmb_pad(drug_id; zp - zp) :- DrugProj(drug_id; zp) .
DrugEmb_row(drug_id; z) :- DrugEmb_from_attn(drug_id; z1) | DrugEmb_pad(drug_id; z2) .
DrugEmb(drug_id; sum(z)) :- DrugEmb_row(drug_id; z) .
""")

# %% [markdown]
# ### Decoder: MLP
#
# **Paper Eq. (11)**: Concatenate drug pair embeddings, pass through 2-layer MLP.
#
# $$\gamma(q_x, q_y) = f_2(f_1(q_x \| q_y))$$
#
# We create `Drug1` and `Drug2` as column-aliased views of `DrugEmb` so that RelNN can join the same embedding table twice (once per drug in the pair).

# %%
session.run("""
#lang:relnn
Drug1(drug1; z) :- DrugEmb(drug1; z) .
Drug2(drug2; z) :- DrugEmb(drug2; z) .

MLP1 = Linear(2*d, d) .
MLP2 = Linear(d, 1) .

PairScore(drug1, drug2; MLP2(ReLU()(MLP1(Concat(z1, z2))))) :- Drug1(drug1; z1), TrainPairs(drug1, drug2; z_label), Drug2(drug2; z2) .
""")

# %% [markdown]
# ## 3. Fit
#
# Binary cross-entropy loss on the MLP decoder output vs. DDI labels.

# %%
session.run("""
#lang:relnn
?fit <epochs=500, lr=0.005>
Loss(; BCEWithLogitsLoss()(z_score, z_label)) :- PairScore(drug1, drug2; z_score), TrainPairs(drug1, drug2; z_label) .
""")

# %% [markdown]
# ## 4. Predict & Evaluate
#
# Score all test pairs using the trained encoder + MLP decoder, then compute F1, ROC-AUC, PR-AUC.
#
# For prediction we need to re-define PairScore to use TestPairs instead of TrainPairs.

# %%
session.run("""
#lang:relnn
TestScore(drug1, drug2; MLP2(ReLU()(MLP1(Concat(z1, z2))))) :- Drug1(drug1; z1), TestPairs(drug1, drug2; z_label), Drug2(drug2; z2) .
""")

# %%
pred_result = session.run("""
#lang:relnn
?pred Predictions(drug1, drug2; Sigmoid()(z)) :- TestScore(drug1, drug2; z) .
""")

# %%
# Accuracy: RelNN built-in (join Predictions with TestPairs on (drug1, drug2) aligns automatically)
acc_result = session.run("""
#lang:relnn
?pred Accuracy(; mean((round(z_pred) == z_label) * 1.0)) :- Predictions(drug1, drug2; z_pred), TestPairs(drug1, drug2; z_label) .
""")
acc = acc_result.embeddings[0].item()
print(f"Accuracy:  {acc:.4f}")

# Other metrics need aligned (pred, label) pairs; merge on (drug1, drug2) to fix RelNN join order
test_df = data.db["TestPairs"][0].copy()
test_df["label"] = data.test_labels.numpy().flatten()
pred_df = pred_result.content.copy()
pred_df["pred_score"] = pred_result.embeddings[0].detach().cpu().numpy().flatten()
merged = test_df.merge(pred_df, on=["drug1","drug2"], how="left")
pred_scores = merged["pred_score"].values
true_labels = merged["label"].values

pred_binary = (pred_scores >= 0.5).astype(int)
prec = precision_score(true_labels, pred_binary)
rec = recall_score(true_labels, pred_binary)
f1 = f1_score(true_labels, pred_binary)
roc_auc = roc_auc_score(true_labels, pred_scores)
pr_auc = average_precision_score(true_labels, pred_scores)

print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1:        {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

# %% [markdown]
# ## 5. Inspect model
#
# Show learned parameters and the term graph.

# %%
session.show_params(show_stats=False)
