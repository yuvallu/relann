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
# **TL;DR — What is RelNN?**  
# RelNN is a framework for **relational neural networks**: you describe models as rules over tables where each tuple can carry an embedding, and each rule states how tuples join and how their embeddings combine (e.g. Linear then aggregate). The engine compiles these rules into a term graph and then into a PyTorch `nn.Module`. The execution uses Embedded Relational Algebra (join, aggregate, etc.). This notebook runs a small relational neural network on the Cora citation dataset.

# %% [markdown]
# # RelNN Hello World: Graph Convolutional Network on Cora
#
# This tutorial builds a 2-layer GCN on the [Cora citation dataset](https://graphsandnetworks.com/the-cora-dataset/): papers (nodes), citations (edges), bag-of-words features. Task: semi-supervised node classification (7 subject classes per paper).

# %% [markdown]
# ## Imports
#
# We start by importing the necessary modules. RelNN uses familiar PyTorch components like `Linear` and `ReLU`, making it easy to get started.

# %%
from relann.session import Session
from relann.datasets import load_cora_dataset

# %% [markdown]
# ## 1. Load data
#
# Load the [Cora dataset](https://graphsandnetworks.com/the-cora-dataset/) and build the relational db (Papers, Citation, Labels, TestLabels).

# %%
data = load_cora_dataset()
db = data.to_dict()

# %% [markdown]
# ### Database and task summary
#
# The dataset object’s repr shows the four tables and the task summary.  
# **7 class names:** Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory.

# %%
data

# %% [markdown]
# ## 2. Init Session
#
# Create a session with the database. All programs run via `session.run(...)`.

# %%
session = Session(db=db)

# %% [markdown]
# ## 3. Define the relnn model

# %% [markdown]
# ### Graph Convolutional Network (GCN) Equations
#
# The 2-layer GCN defined in RelNN can be written as:
#
# $$
# \begin{align*}
#   \text{PapersEmb}_1(\text{pid}) & = \text{Linear}_{1433 \to 16}(\text{Papers}(\text{pid})) \\
#   \text{PapersAgg}_1(\text{target\_id}) & = \sum_{(\text{citing}, \text{target\_id}) \in \text{Citation}} \text{PapersEmb}_1(\text{citing}) \cdot w_{\text{citing}, \text{target\_id}} \\
#   \text{PapersAggNL}_1(\text{target\_id}) & = \text{ReLU}(\text{PapersAgg}_1(\text{target\_id})) \\
#   \text{PapersEmb}_2(\text{target\_id}) & = \text{Linear}_{16 \to 7}(\text{PapersAggNL}_1(\text{target\_id})) \\
#   \text{PapersAgg}_2(\text{target\_id}) & = \sum_{(\text{citing}, \text{target\_id}) \in \text{Citation}} \text{PapersEmb}_2(\text{citing}) \cdot w_{\text{citing}, \text{target\_id}} \\
#   \text{Output}(\text{target\_id}) & = \text{PapersAgg}_2(\text{target\_id})
# \end{align*}
# $$

# %% [markdown]
# Here we define the GCN in RelNN language.  
# Then we register the Relational Neural Network with `session` (no training yet).

# %%
define_program = """
#lang:relnn
in_channels = 1433 .
hidden_channels = 16 .
out_channels = 7 .
lr = 0.01 .
epochs = 200 .
weight_decay = 0.0005 .

# Layer 1
PapersEmb1(pid; Linear(in_channels, hidden_channels, False)(z)) :- Papers(pid; z) .
PapersAgg1(cited; sum(z * w)) :- PapersEmb1(citing; z), Citation(citing, cited; w) .
PapersAggNL_Layer1(cited; ReLU(z)) :- PapersAgg1(cited; z) .

# Layer 2
PapersEmb2(cited; Linear(hidden_channels, out_channels, False)(z)) :- PapersAggNL_Layer1(cited; z) .
PapersAgg2(cited; sum(z * w)) :- PapersEmb2(citing; z), Citation(citing, cited; w) .
Output(cited; z) :- PapersAgg2(cited; z) .
"""

session.run(define_program)

# %% [markdown]
# ## 4. Train
#
# Run the fit statement: cross-entropy on labeled nodes for 200 epochs.

# %%
fit_program = """
#lang:relnn
?fit <epochs=epochs, lr=lr, weight_decay=weight_decay> 
Loss(; CrossEntropyLoss()(z_pred, z)) :- Output(cited; z_pred), Labels(cited; z) .
"""

session.run(fit_program)

# %% [markdown]
# ## 5. Predict
#
# Run the predict statement: predicted class (0–6) for every node.

# %%
pred_program = """
#lang:relnn
?pred Predictions(cited; ArgMax()(z)) :- Output(cited; z) .
"""

pred_result = session.run(pred_program)

# %%
print("Predictions:")
pred_result

# %% [markdown]
# ## 6. Evaluate
#
# Test accuracy in one RelNN rule: join `Predictions` with `TestLabels`, compare predicted vs. true class, average.

# %%
eval_program = """
#lang:relnn
?pred Accuracy(; mean((z_pred == z_label) * 1.0)) :- Predictions(cited; z_pred), TestLabels(cited; z_label) .
"""
acc = session.run(eval_program).embeddings[0].item()
print(f"Test Accuracy: {acc:.1%}")

# %% [markdown]
# ## 7. Parameters
#
# Learned parameters (weights).

# %%
session.show_params(show_stats=False)

# %% [markdown]
# ## 8. Term graph
#
# How the engine compiled the rules into a `nn.Module`.

# %%
session.show_term_graph(graph_attrs={'size': '10,30', 'nodesep': '1.5', 'ranksep': '1.0'})

# %% [markdown]
# ---
#
# ## Summary
#
# **Flow:** Load data → Session → Define (relnn rules) → Train (fit) → Predict → Evaluate (RelNN join + avg) → Parameters & term graph.
