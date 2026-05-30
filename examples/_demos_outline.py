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
# # Demos outline
#
# - **001_relnn_hello_world**: GCN on Cora (Session, define, fit, predict).
# - **002_era_join_projection**: ERA under the hood: Join (index_select), Transformation, Projection (scatter). Users + Comments; instantiate + forward only.
# - **003_relnn_hgt**: Heterogeneous Graph Transformer on DBLP.
# - **004_relnn_hygnn**: HyGNN variant.
# - **005_relnn_relbench_f1**: RelBench F1 task.
# - **006_relnn_relbench_f1_dnf**: RelBench F1 with DNF rules.
# - **007_relnn_sql_loading**: Same 2-layer GCN as hello-world but loaded from SQLite. Demonstrates SqlSource (edge list), DataFrameSource (node features + labels), and the RelationSource abstraction.
