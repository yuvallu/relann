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
# **Fast as PyG, simple as LaTeX.**  
# This demo runs the [Heterogeneous Graph Transformer (HGT)](https://arxiv.org/abs/2003.01332) on Cora in RelNN. Every equation from the paper appears as LaTeX right above its RelNN rule — so you can see the **one-to-one mapping** and why we mean *simple as LaTeX*. The same model runs as fast as PyTorch Geometric; we use the **generic** HGT library (recursion + bounded sets) so one DSL program works for both Cora and heterogeneous graphs like DBLP.

# %% [markdown]
# # HGT on Cora: Fast as PyG, Simple as LaTeX
#
# We implement HGT (Hu et al., WWW 2020) in RelNN on the Cora citation graph. Below, **each paper equation is shown in LaTeX, then the matching RelNN rule** — so the similarity is explicit. The model uses the **generic** HGT library (same code path as DBLP): recursion base case for layer 0, bounded sets over `MetaRel` and over heads. Cora has one node type (`Papers`) and one edge type (`Citation`); we load **MetaRel** so the generic DSL can expand over edge types at compile time.

# %% [markdown]
# ## Imports

# %%
from relann.session import Session
from relann.datasets import load_cora_dataset, evaluate_node_classification
from relann.torch_utils import full_seed

# %% [markdown]
# ## 1. Load data
#
# Cora: 2,708 papers, 13,264 citations, 7 classes. We use `data.db` so **MetaRel** is included (one row: Papers → Citation → Papers); the generic HGT library needs it for bounded-set expansion.

# %%
data = load_cora_dataset()
db = data.db  # Papers, Citation, Labels, TestLabels, MetaRel (needed for generic HGT)
data

# %% [markdown]
# ## 2. Init Session

# %%
full_seed(42)
session = Session(db=db)

# %% [markdown]
# ## 3. HGT: LaTeX → RelNN
#
# Below, each equation from the HGT paper (Hu et al., WWW 2020) is shown in **LaTeX**; the code cell under it defines that part of the model. Same generic library runs on Cora and DBLP; only the base case and MetaRel change.
#
# ---
#
# ### 3.0 Softmax (reusable)
#
# **Paper:** Normalize scores over source nodes for each target $t$:
# $$ \text{softmax}_t(x_{s,t}) = \frac{ \exp(x_{s,t}) }{ \sum_{s'} \exp(x_{s',t}) } $$

# %%
d = 16
h = 4
dh = d // h
in_features = 1433
n_classes = 7
epochs = 200
lr = 0.01
weight_decay = 0.0005

session.run(f"""
#lang:relnn
d = {d} .
h = {h} .
dh = {dh} .
epochs = {epochs} .
lr = {lr} .
weight_decay = {weight_decay} .

def Softmax(Scores):
    Exp(s, t; exp(z))   :- Scores(s, t; z) .
    Denom(t; sum(z))    :- Exp(s, t; z) .
    Out(s, t; z1 / z2)  :- Exp(s, t; z1), Denom(t; z2) .
enddef
""")

# %% [markdown]
# ### 3.1 Base case (layer 0)
#
# **Paper:** Initial node embedding for type $\tau$:
# $$ H^{(0)}_\tau(v) = \text{Linear}_\tau(x_v) $$
# (e.g. with ReLU; one such rule per node type.)

# %%
session.run(f"""
#lang:relnn
H<'Papers', 0>(s; ReLU(Linear({in_features}, d)(z))) :- Papers(s; z) .
""")

# %%
session.run("""
#lang:relnn
K<L, ts, i>  = Linear(d, dh) .
Q<L, tt, i>  = Linear(d, dh) .
M<L, ts, i>  = Linear(d, dh) .
W_ATT<L, pe, i>  = Linear(dh, dh, False) .
W_MSG<L, pe, i>  = Linear(dh, dh, False) .
Mu<L, pe, i>     = Tensor(1) .
A_Lin<L, tt>     = Linear(d, d) .
Skip<L, tt>      = Tensor(1) .
""")

# %% [markdown]
# ### 3.2 Key & Query (per head $i$)
#
# **Paper:** Type-specific key and query from previous-layer embeddings:
# $$ K^i(s) = \textbf{K-Linear}^i_{\tau(s)}\bigl( H^{(l-1)}[s] \bigr), \qquad Q^i(t) = \textbf{Q-Linear}^i_{\tau(t)}\bigl( H^{(l-1)}[t] \bigr) $$

# %% [markdown]
# ### 3.3 Attention (per head $i$)
#
# **Paper:** Relation-aware attention score, then softmax over source nodes $s \to t$:
# $$ \text{ATT-head}^i(s, e, t) = \text{softmax}_t\left( \frac{ \bigl( K^i(s) \, \textbf{W}^{ATT}_{\phi(e)} \, Q^i(t)^\top \bigr) \cdot \mu_{\langle \tau(s), \phi(e), \tau(t) \rangle} }{ \sqrt{d_h} } \right) $$

# %%
session.run("""
#lang:relnn
def ATT_Head<L, ts, pe, tt, i>():
    K(s; K<L, ts, i>(z))  :- H<ts, L-1>(s; z) .
    Q(t; Q<L, tt, i>(z))  :- H<tt, L-1>(t; z) .
    Dot(s, t; view(1)(z_q @ transpose(W_ATT<L, pe, i>(z_k))) * Mu<L, pe, i> / sqrt(dh) ) :- K(s; z_k), pe(s, t; w), Q(t; z_q) .
    Out(s, t; z) :- Softmax(Dot)(s, t; z) .
enddef
""")

# %% [markdown]
# ### 3.4 Message (per head $i$)
#
# **Paper:** Message from source, transformed by edge-type matrix:
# $$ \text{MSG-head}^i(s, e, t) = \textbf{M-Linear}^i_{\tau(s)}\bigl( H^{(l-1)}[s] \bigr) \, \textbf{W}^{MSG}_{\phi(e)} $$

# %%
session.run("""
#lang:relnn
def MSG_Head<L, ts, pe, tt, i>():
    M(s; M<L, ts, i>(z))  :- H<ts, L-1>(s; z) .
    Out(s, t; W_MSG<L, pe, i>(z_m)) :- M(s; z_m), pe(s, t; w) .
enddef
""")

# %% [markdown]
# ### 3.5 Weighted messages & aggregate (Eq. 5)
#
# **Paper:** Per-head attention-weighted message, concatenate heads, then sum over in-edges:
# $$ \tilde{m}(s, e, t) = \|_{i=1}^{h} \bigl( \text{ATT-head}^i(s, e, t) \cdot \text{MSG-head}^i(s, e, t) \bigr), \qquad \tilde{H}^{(l)}[t] = \sum_{\forall s \in N(t)} \tilde{m}(s, e, t) $$

# %%
session.run("""
#lang:relnn
def EdgeAgg<L, ts, pe, tt>():
    WMsg<i>(s, t; z_att * z_msg) :- ATT_Head<L, ts, pe, tt, i>(s, t; z_att), MSG_Head<L, ts, pe, tt, i>(s, t; z_msg) .
    AllHeads(s, t; Concat(*z)) :- Join(Set(WMsg<i>(s, t; z) | 1 <= i, i <= h)) .
    Out(t; sum(z)) :- AllHeads(s, t; z) .
enddef
""")

# %% [markdown]
# ### 3.6 Target aggregation over edge types + residual (Eq. 5–6)
#
# **Paper:** Aggregate over all meta-relations $\langle \tau(s), \phi(e), \tau(t) \rangle$ into target $t$, then linear + gated residual:
# $$ \text{Agg}(t) = \sum_{\langle \tau(s), \phi, \tau(t) \rangle} \tilde{m}(s, e, t), \qquad H^{(l)}[t] = \sigma(\text{skip}_{\tau(t)}) \cdot \textbf{A-Linear}_{\tau(t)}\bigl( \sigma(\tilde{H}^{(l)}[t]) \bigr) + \bigl( 1 - \sigma(\text{skip}_{\tau(t)}) \bigr) \cdot H^{(l-1)}[t] $$
# (Here $\sigma$ is the activation, e.g. GELU for the aggregate; $\sigma(\text{skip})$ is the learned gate.)

# %%
session.run(f"""
#lang:relnn
def H<tt, L>():
    Agg(t; sum(z)) :- Union(Set(EdgeAgg<L, ts, pe, tt>(t; z) | MetaRel(ts, pe, tt))) .
    Updated(t; A_Lin<L, tt>(GELU(z))) :- Agg(t; z) .
    Out(t; Sigmoid(Skip<L, tt>) * z1 + (1 - Sigmoid(Skip<L, tt>)) * z2 ) :- Updated(t; z1), H<tt, L-1>(t; z2) .
enddef

Classifier = Linear(d, {n_classes}) .
Output(target_id; Classifier(z)) :- H<'Papers', 1>(target_id; z) .
""")

# %% [markdown]
# ## 4. Train
#
# Cross-entropy loss on the 140 labeled training nodes, optimized with Adam for 200 epochs. `Output` produces 7-class logits (Classifier is applied inside the Output rule).

# %%
fit_program = """
#lang:relnn
?fit <epochs=epochs, lr=lr, weight_decay=weight_decay>
Loss(; CrossEntropyLoss()(z_pred, z)) :- Output(target_id; z_pred), Labels(target_id; z) .
"""

session.run(fit_program)

# %% [markdown]
# ## 5. Predict + Evaluate

# %%
pred_program = """
#lang:relnn
?pred Predictions(target_id; ArgMax()(z)) :- Output(target_id; z) .
"""

pred_result = session.run(pred_program)
evaluate_node_classification(data, pred_result)

# %% [markdown]
# ### Expected accuracy
#
# A single-layer HGT on Cora (one message-passing step) typically reaches **~60%** test accuracy with `d=16, h=4` and 200 epochs. The same generic DSL runs on DBLP and matches hand-rolled PyTorch (see `tests/slow/run_compare_dblp_hgt_generic.py`). For higher Cora accuracy, use a 2-layer GCN (demo 001, ~81%).

# %% [markdown]
# ## 6. Parameters (optional)
#
# The generic library yields templated parameters like `K_Lin<1,Papers,1>`, `W_ATT<1,Citation,2>`, etc. Run `session.show_params()` to inspect.

# %%
session.show_params(show_stats=False)

# %% [markdown]
# ## 7. Term graph
#
# The engine compiles the generic HGT rules into a PyTorch `nn.Module` (per-head and per–edge-type branching from the templated defs).

# %%
session.show_term_graph(graph_attrs={'size': '14,40', 'nodesep': '1.5', 'ranksep': '1.0'})

# %% [markdown]
# ---
#
# ## Summary
#
# **Flow:** Load data -> Session -> Define (HGT with templates) -> Train (fit) -> Predict -> Evaluate.
#
# Templates let you express per-head and per-layer parameterization declaratively. The engine handles weight instantiation, sharing, and compilation to a PyTorch module.
#
# For validation that these results match an equivalent PyTorch/PyG implementation, see the comparison scripts in `tests/slow/`.
