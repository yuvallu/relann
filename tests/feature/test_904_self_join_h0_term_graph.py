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
# # Self-join on `H0`: term graph inspection
#
# Three appearances of **`H0(graph_id, u; zu), H0(graph_id, v; zv), H0(graph_id, w; zw)`**
# on the RHS form a single **join** over the same derived relation three times
# (Datalog-style self-join).
#
# **Behind the scenes:** the compiler builds one pipeline for `H0`
# (visible as `agg_H0` in the term graph), then a **join** node whose
# `input_order` lists the **same graph-node id three times** —
# `relnn.Join` treats them as three separate inputs with independent row
# alignments.  
# The NetworkX `DiGraph` can hold **at most one directed edge** between any
# two nodes, so the visual representation shows only one predecessor, but
# **`input_order` is the authoritative source** for the engine's runtime
# arity.

# %%
if test():
    import sys
    from pathlib import Path

    repo_root = Path.cwd()
    if not (repo_root / "parent").is_dir():
        repo_root = (
            Path.cwd().parent
            if (Path.cwd().parent / "parent").is_dir()
            else Path.cwd()
        )
    sys.path.insert(0, str(repo_root))

    import pandas as pd
    import torch
    from relann.session import Session

# %%
if test():
    nodes = pd.DataFrame({"graph_id": [0, 0, 0], "n": [0, 1, 2]})
    db = {"Node": (nodes, torch.ones(len(nodes), 1))}

    dsl = """
    H0(graph_id, n; z) :- Node(graph_id, n; z) .

    # Self-join: H0 appears THREE times on the RHS — no aliases needed.
    TripleH0(graph_id, u, v, w; zu) :- H0(graph_id, u; zu), H0(graph_id, v; zv), H0(graph_id, w; zw) .

    ?pred Out(graph_id, u, v, w; z) :- TripleH0(graph_id, u, v, w; z) .
    """

    session = Session(db=db)
    session.define(dsl)

# %%
if test():
    tg = session.engine.term_graphs["global"]

    # 1. Logical name -> physical node id
    print("symbol_to_node (logical name -> physical node id):")
    for sym in ("Node", "H0", "TripleH0", "Out"):
        if sym in tg.symbol_to_node:
            print(f"  {sym!r:15s} -> {tg.symbol_to_node[sym]!r}")

    # 2. Full node list (acts like a relational query plan)
    print("\nAll term-graph nodes (query plan):")
    print(f"  {'node id':<35}  {'type':<14}  schema")
    print("  " + "-" * 70)
    for nid, data in tg.nodes(data=True):
        print(f"  {nid:<35}  {data.get('type', '?'):<14}  {data.get('output_schema', '')}")

    # 3. Join nodes: show both the DiGraph edge view and runtime input_order.
    # NetworkX DiGraph stores at most one edge per (src, dst) pair, so the visual
    # graph shows only ONE predecessor.  The engine uses input_order (not edge
    # count) to determine arity — that is where the self-join is visible.
    print("\nJoin nodes:")
    for nid, data in tg.nodes(data=True):
        if data.get("type") == "join":
            io = data.get("input_order", [])
            preds = list(tg.predecessors(nid))
            same = len(set(io)) == 1 and len(io) == 3
            print(f"  {nid}")
            print(f"    predecessors (DiGraph, deduplicated) : {preds}")
            print(f"    input_order  (runtime, full arity)   : {io}")
            print(f"    => same node listed 3x (self-join confirmed): {same}")

# %%
if test():
    er = session.relation("Out")
    print("Forward pass sanity check:")
    print(f"  rows: {len(er.content)}  (expected 3^3 = 27, all (u,v,w) triples)")
    print(er.content.head(9).to_string(index=False))
    print(f"  embedding shape: {er.embeddings[0].shape}")

# %%
if test():
    # Graphviz diagram of the TripleH0 subgraph.
    # show_term_graph() internally extracts ancestors; do NOT pass direction here.
    # If graphviz is not on PATH this returns None; input_order above is the ground truth.
    session.show_term_graph("TripleH0")
