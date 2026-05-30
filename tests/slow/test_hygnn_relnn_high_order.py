# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from juplit import test

# %% [markdown]
# # HyGNN High-Order: Multi-Layer Hypergraph Neural Network in RelNN
#
# We extend the [HyGNN](https://arxiv.org/pdf/2206.12747) (Saifuddin et al., 2022) demo to **multiple layers** using RelNN high-order constructs (template parameters, FunctionDefs).
#
# **Architecture**: Stack multiple HyGNN layers. Each layer has:
# 1. **Hyperedge → Node**: aggregate drug information to substructures via attention
# 2. **Node → Hyperedge**: aggregate substructure information back to drugs via attention
#
# Layer L takes `(DrugEmb_{L-1}, SubEmb_{L-1})` and produces `(DrugEmb_L, SubEmb_L)`. The final drug embeddings feed into the DDI pair classifier.

# %% [markdown]
# ## Imports

# %%
if test():
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
# Load the precomputed DDI dataset from the [HyGNN reference repository](https://github.com/shouhengtuo/HyGNN-Drug-Drug-Interaction-Prediction-via-Hypergraph-Neural-Network). Same as the single-layer demo.

# %%
if test():
    data = load_hygnn_dataset(source='DrugBank', k=9, d=128, seed=42)
    data

# %%
if test():
    info = data.dataset_info
    n_drugs = info['n_drugs']
    print(f"n_drugs={n_drugs}, n_subs={info['n_subs']}, incidence_edges={info['n_incidence']}")

# %% [markdown]
# ## 2. Constants and Session

# %%
if test():
    full_seed(42)
    session = Session(db=data.db)

    d = 128
    qd = 64
    num_layers = 1

# %% [markdown]
# ## 3. Softmax and Initial Projection

# %%
if test():
    session.run(f"""
    #lang:relnn
    n_drugs = {n_drugs} .
    d = {d} .
    qd = {qd} .

    def Softmax(Scores):
        MaxPerT(t; max(z))        :- Scores(s, t; z) .
        Exp(s, t; exp(z - max_t)) :- Scores(s, t; z), MaxPerT(t; max_t) .
        Denom(t; sum(z))          :- Exp(s, t; z) .
        Out(s, t; z1 / z2)        :- Exp(s, t; z1), Denom(t; z2) .
    enddef
    """)

# %%
if test():
    session.run("""
    #lang:relnn
    DrugProj(drug_id; Linear(n_drugs, d)(z)) :- Drug(drug_id; z) .
    """)

# %% [markdown]
# ## 4. Per-Layer Weight Templates, MLPs and HyGNN Base FunctionDefs
#
# Each layer L has its own weights (W1<L>..W6<L>).

# %%
if test():
    session.run(f"""
    #lang:relnn
    HyGNN_Subs<0>(sub_id; z) :- Substructure(sub_id; z) .
    HyGNN_Drugs<0>(drug_id; z) :- DrugProj(drug_id; z) .
    """)

# %%
if test():
    session.run("""
    #lang:relnn
    W1<L> = Linear(d, d) .
    W2<L> = Linear(d, qd) .
    W3<L> = Linear(d, qd) .
    W4<L> = Linear(d, d) .
    W5<L> = Linear(d, qd) .
    W6<L> = Linear(d, qd) .

    MLP1 = Linear(2*d, d) .
    MLP2 = Linear(d, 1) .
    """)

# %% [markdown]
# ## 5. Per-Layer Weight Templates and HyGNN FunctionDefs
#
# We define two FunctionDefs:
# - **HyGNN_Subs<L>**: Hyperedge→Node (drugs → substructures)
# - **HyGNN_Drugs<L>**: Node→Hyperedge (substructures → drugs)

# %%
if test():
    session.run("""
    #lang:relnn
    def HyGNN_Subs<L>():
        K_E(drug_id; W2<L>(z)) :- HyGNN_Drugs<L-1>(drug_id; z) .
        V_E(drug_id; W1<L>(z)) :- HyGNN_Drugs<L-1>(drug_id; z) .
        Q_E(sub_id; W3<L>(z)) :- HyGNN_Subs<L-1>(sub_id; z) .
        AttnE_raw(drug_id, sub_id; LeakyReLU()(view(1)(z_q @ transpose(z_k))) / sqrt(qd)) :- K_E(drug_id; z_k), Incidence(sub_id, drug_id; w), Q_E(sub_id; z_q) .
        AttnE(drug_id, sub_id; z) :- Softmax(AttnE_raw)(drug_id, sub_id; z) .
        Out(sub_id; sum(z_att * z_v)) :- AttnE(drug_id, sub_id; z_att), V_E(drug_id; z_v) .
    enddef
    """)

# %%
if test():
    session.run("""
    #lang:relnn
    def HyGNN_Drugs<L>():
        K_V(sub_id; W5<L>(z)) :- HyGNN_Subs<L>(sub_id; z) .
        V_V(sub_id; W4<L>(z)) :- HyGNN_Subs<L>(sub_id; z) .
        Q_V(drug_id; W6<L>(z)) :- HyGNN_Drugs<L-1>(drug_id; z) .
        AttnV_raw(sub_id, drug_id; LeakyReLU()(view(1)(z_q @ transpose(z_k))) / sqrt(qd)) :- K_V(sub_id; z_k), Incidence(sub_id, drug_id; w), Q_V(drug_id; z_q) .
        AttnV(sub_id, drug_id; z) :- Softmax(AttnV_raw)(sub_id, drug_id; z) .
        Out_attn(drug_id; sum(z_att * z_v)) :- AttnV(sub_id, drug_id; z_att), V_V(sub_id; z_v) .
        Out_pad(drug_id; zp - zp) :- DrugProj(drug_id; zp) .
        Out_row(drug_id; z) :- Out_attn(drug_id; z1) | Out_pad(drug_id; z2) .
        Out(drug_id; sum(z)) :- Out_row(drug_id; z) .
    enddef
    """)

# %% [markdown]
# ## 6. HyGNN FunctionDef: Apply MLP on Drug Pairs
#
# Same as single-layer demo: Drug1 and Drug2 are aliases of DrugEmb for the pair join.

# %%
if test():
    session.run("""
    #lang:relnn
    def HyGNN<L>(TargetPairs):
        DrugEmb(drug_id; z) :- HyGNN_Drugs<L>(drug_id; z) .
        Drug1(drug1; z) :- DrugEmb(drug1; z) .
        Drug2(drug2; z) :- DrugEmb(drug2; z) .
        Out(drug1, drug2; MLP2(ReLU()(MLP1(Concat(z1, z2))))) :- Drug1(drug1; z1), TargetPairs(drug1, drug2; z), Drug2(drug2; z2) .
    enddef
    """)

# %% [markdown]
# ## 7. Fit

# %%
if test():
    session.run(f"""
    #lang:relnn
    PairScore(drug1, drug2; z) :- HyGNN<{num_layers}>(TrainPairs)(drug1, drug2; z) .
    ?fit <epochs=200, lr=0.005>
    Loss(; BCEWithLogitsLoss()(z_score, z_label)) :- PairScore(drug1, drug2; z_score), TrainPairs(drug1, drug2; z_label) .
    """)

# %% [markdown]
# ## 8. Predict & Evaluate

# %%
if test():
    pred_result = session.run("""
    #lang:relnn
    TestScore(drug1, drug2; z) :- HyGNN<1>(TestPairs)(drug1, drug2; z) .
    ?pred Predictions(drug1, drug2; Sigmoid()(z)) :- TestScore(drug1, drug2; z) .
    """)

# %%
if test():
    acc_result = session.run("""
    #lang:relnn
    ?pred Accuracy(; mean((round(z_pred) == z_label) * 1.0)) :- Predictions(drug1, drug2; z_pred), TestPairs(drug1, drug2; z_label) .
    """)
    acc = acc_result.embeddings[0].item()
    print(f"Accuracy:  {acc:.4f}")
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
# ## 9. Inspect model

# %%
if test():
    session.show_params(show_stats=False)
