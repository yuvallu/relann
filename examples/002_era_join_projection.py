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
# # ERA Demo: Join, Transform, Project
#
# This demo shows how a **single RelNN rule** is compiled into a **term graph** and executed as three ERA steps:  
# &nbsp;&nbsp;&nbsp;&nbsp;**Join → Transform → Project**.

# %%
from relann.session import Session
from relann.datasets import load_era_join_demo_dataset, get_era_join_demo_db, show_era_join_demo_tables, build_and_run_era_demo_module, show_era_join_step, show_era_transform_step, show_era_project_step

# %% [markdown]
# ## Define a RelNN Rule
#
# Let's look at this rule: join **Users** and **Comments** on `user_id`, apply **Linear(Concat(z1,z2))**, then **sum** by user (project).

# %%
# Join on user_id → Linear(4,2)(Concat(z1,z2)) → sum by user
define_program = """
#lang:relnn
UCEmb(u; sum(Linear(4,2)(Concat(z1,z2)))) :- Users(u;z1), Comments(c, u;z2) .
"""

data = load_era_join_demo_dataset()
db = get_era_join_demo_db(data)
session = Session(db=db)
session.run(define_program)

print("Input tables:")
show_era_join_demo_tables(data)

# %% [markdown]
# ## Term Graph
#
# The rule is compiled into a term graph, which is composed of three main steps: **Join**, **Transform**, and **Project**.

# %%
session.show_term_graph(rule_name="UCEmb", graph_attrs={"size": "8,6", "nodesep": "1.2", "ranksep": "0.8"})

# %% [markdown]
# Build the RelNN module for the rule and run it so we can inspect the ER at each step.

# %%
module, cache, nodes = build_and_run_era_demo_module(session, "UCEmb")

# %% [markdown]
# ## Join
#
# **Input:** Users and Comments. **Output:** one row per (user, comment) pair, with aligned embeddings.  
# Join merges content on `user_id` and caches **join indices**: for each result row, which input row came from the left and which from the right. **Forward** uses `index_select(0, idx)` to gather embeddings by these indices.

# %%
show_era_join_step(module, cache, nodes["join_node"])

# %% [markdown]
# ## Transform
#
# **Input:** joined ER (5 rows, 2 embeddings each). **Output:** one embedding per row (Concat + Linear). No explanation needed.

# %%
show_era_transform_step(cache, nodes["join_node"], nodes["transform_node"])

# %% [markdown]
# ## Project
#
# **Input:** transformed ER (5 rows). **Output:** one row per user (3 rows), embedding = sum over that user's joined rows.  
# Project groups by `user_id` and caches a **group index**: each input row maps to an output row. **Forward** uses `torch_scatter.scatter_add(src, index, dim=0)` to reduce per group.

# %%
show_era_project_step(module, cache, nodes["transform_node"], nodes["agg_node"])

# %% [markdown]
# ## `scatter_add` Explained
#
# The `scatter_add` operation reduces rows based on a grouping index, summing items that share the same group. For each element in the source tensor (`src`), its value is *added* to the output at the position specified by its group index.
#
# ![scatter_add: index, input, and output—values with the same index are summed into the corresponding output position](../images/add.jpg)

# %% [markdown]
# ## Summary
#
# | Step      | Input → Output              | Implementation                          |
# |-----------|-----------------------------|-----------------------------------------|
# | **Join**  | Users, Comments → joined ER | Merge on key; `index_select(0, idx)`    |
# | **Transform** | joined ER → transformed ER | `nn.Module` (Concat + Linear)           |
# | **Project**   | transformed ER → agg ER | `scatter_add(src, group_idx, dim=0)`    |
