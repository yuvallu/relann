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
# **TL;DR — SQL for relations, RHS encode for node features.**  
# This notebook runs the same 2-layer GCN as [hello-world](001_relnn_hello_world.ipynb), but **node keys** live in **SQLite** (`Papers` is **`pid` only**). BoW vectors are **not** stored as `f0…f1432` columns; they enter via **RHS encode** `[CoraNodeFeatures(pid)]`.  
# `DataFrameSource` still attaches a **tiny placeholder** `(N,1)` tensor so `Transformation` can run shape inference and `forward` (engine requirement); the **actual** 1433-dim inputs come only from the encoder. Labels use `DataFrameSource` with tensors from SQL as before.

# %% [markdown]
# # RelNN: Loading from SQL — GCN on Cora from SQLite
#
# This tutorial loads the [Cora citation dataset](https://graphsandnetworks.com/the-cora-dataset/) from a **SQLite database** and trains a 2-layer message-passing network for semi-supervised node classification (7 subject classes per paper).  
# It demonstrates `SqlSource` and `DataFrameSource`, plus **encode**: SQL holds **keys** (`pid`); **`CoraNodeFeatures`** maps each `pid` to the **same** Planetoid BoW row as hello-world (frozen buffer). A small **placeholder** embedding satisfies `Transformation`’s requirement for a non-empty input tensor; the **1433**-dim signal still comes only from `[CoraNodeFeatures(pid)]`.

# %% [markdown]
# ## Imports
#
# Hello-world imports, plus `RelationSource` helpers, SQLAlchemy, and `torch.nn` for the **CoraNodeFeatures** encoder used in RHS `[...]` brackets.

# %%
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sqlalchemy import create_engine

from relann.data_sources import DataFrameSource, SqlSource
from relann.datasets import load_cora_dataset
from relann.session import Session
from relann.torch_utils import get_project_root

# %%
# Frozen Planetoid BoW matrix for RHS encode: pid (from SQL) -> feature row.
# Loaded once; CoraNodeFeatures must be a class in this scope so session.run resolves it.
_cora_demo_feats = load_cora_dataset().db["Papers"][1].detach().cpu()


class CoraNodeFeatures(nn.Module):
    """Maps tensorized ``pid`` column (float32 from RelNN encode) to Cora ``x`` row (frozen)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", _cora_demo_feats)

    def forward(self, x):
        idx = x.long().clamp(0, self.weight.size(0) - 1).squeeze(-1)
        return self.weight[idx]

# %% [markdown]
# ## 1. Build the SQLite database
#
# The cell below produces `data/cora_demo.sqlite` with four tables:
#
# | Table | Columns | Notes |
# |---|---|---|
# | `Papers` | `pid` | Node id only — BoW features come from **`[CoraNodeFeatures(pid)]`** in rules, not from SQL |
# | `Citation` | `citing`, `target_id` | Raw edge list (no weights) |
# | `Labels` | `target_id`, `label` | Training labels as integer class index 0–6 |
# | `TestLabels` | `cited`, `label` | Test labels as integer class index 0–6 |
#
# If an **old** DB exists (e.g. `Papers` had `f0`…`f1432`), it is **deleted** and rebuilt so the schema matches this notebook. Otherwise, if the file already exists and `Papers` is `pid`-only, the build is a no-op.

# %%
import sqlite3

DB_PATH = get_project_root() / "data" / "cora_demo.sqlite"
DB_URL  = f"sqlite:///{DB_PATH}"


def _papers_table_is_pid_only(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    with sqlite3.connect(db_path) as c:
        if c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='Papers'"
        ).fetchone() is None:
            return False
        cols = [r[1] for r in c.execute("PRAGMA table_info(Papers)").fetchall()]
    return cols == ["pid"]


def build_cora_sqlite(db_path: Path) -> None:
    """Dump Cora to SQLite. Papers = pid only; drop DB if schema is outdated."""
    if db_path.exists() and not _papers_table_is_pid_only(db_path):
        print(f"Removing outdated DB (expected Papers(pid) only): {db_path}")
        db_path.unlink()
    if db_path.exists():
        print(f"Database already exists: {db_path}")
        return

    print("Loading Cora (downloads on first run)…")
    data = load_cora_dataset()
    raw = data.to_dict()

    papers_df, _papers_tensor = raw["Papers"]
    citation_df, _ = raw["Citation"]  # edge weights dropped intentionally
    labels_df, labels_t = raw["Labels"]
    test_df, test_t = raw["TestLabels"]

    # Papers: relational key only — features enter via [CoraNodeFeatures(pid)] in RelNN
    papers_out = papers_df.reset_index(drop=True)

    labels_out = labels_df.copy().reset_index(drop=True)
    labels_out["label"] = labels_t.squeeze(1).long().cpu().numpy()

    test_out = test_df.copy().reset_index(drop=True)
    test_out["label"] = test_t.squeeze(1).cpu().numpy().astype(int)

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        for name, df in [
            ("Papers", papers_out),
            ("Citation", citation_df.reset_index(drop=True)),
            ("Labels", labels_out),
            ("TestLabels", test_out),
        ]:
            df.to_sql(name, conn, if_exists="replace", index=False)
            conn.commit()
            print(f"  {name}: {len(df)} rows x {len(df.columns)} cols")
    engine.dispose()
    print(f"Done -> {db_path}")


build_cora_sqlite(DB_PATH)

# %% [markdown]
# ### Inspect the tables
#
# Let's peek at the schema and row counts.

# %%
import sqlite3

with sqlite3.connect(DB_PATH) as _c:
    for (tbl,) in _c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
        cols  = [r[1] for r in _c.execute(f"PRAGMA table_info([{tbl}])").fetchall()]
        count = _c.execute(f"SELECT COUNT(*) FROM [{tbl}]").fetchone()[0]
        short = cols[:5] + (["..."] if len(cols) > 5 else [])
        print(f"  {tbl:<12} | rows: {count:>6,} | cols: {short}")

# %% [markdown]
# ## 2. Wire up `RelationSource` instances
#
# **`SqlSource`** — `Citation` is a pure relational edge list (no tensor).
#
# **`DataFrameSource` for `Papers`** — read `pid` from SQL; attach a **placeholder** `zeros(N, 1)` tensor so `Transformation.instantiate` / `forward` see a non-empty embedding (required today). **BoW rows still come only from** **`[CoraNodeFeatures(pid)]`**, not from that placeholder.#
# **`DataFrameSource` for `Labels` / `TestLabels`** — same as before: key columns from SQL plus label tensors reconstructed from the `label` integer column.

# %%
sa_engine = create_engine(DB_URL)

# --- Papers: pid only in SQL; BoW via [CoraNodeFeatures(pid)]; placeholder z for Transformation API ---
with sa_engine.connect() as conn:
    papers_sql = pd.read_sql_table("Papers", conn)
_placeholder = torch.zeros(len(papers_sql), 1, dtype=torch.float32)
papers_src = DataFrameSource("Papers", papers_sql[["pid"]], embeddings=_placeholder)
print(
    f"Papers     -> DataFrameSource: {len(papers_sql)} nodes, pid-only SQL + "
    f"placeholder z {_placeholder.shape}; encode -> {_cora_demo_feats.shape[1]}-dim BoW"
)

# --- Citation: pure relational edge list -> SqlSource ---
citation_src = SqlSource("Citation", DB_URL, table="Citation")
print("Citation   -> SqlSource")

# --- Labels: read SQL, reconstruct integer label tensor ---
with sa_engine.connect() as conn:
    labels_sql = pd.read_sql_table("Labels", conn)
labels_tensor = torch.tensor(labels_sql["label"].values, dtype=torch.long).unsqueeze(1)
labels_src = DataFrameSource("Labels", labels_sql[["target_id"]], embeddings=labels_tensor)
print(f"Labels     -> DataFrameSource: {len(labels_sql)} training nodes")

# --- TestLabels: same pattern ---
with sa_engine.connect() as conn:
    test_sql = pd.read_sql_table("TestLabels", conn)
test_tensor = torch.tensor(test_sql["label"].values, dtype=torch.float32).unsqueeze(1)
test_src = DataFrameSource("TestLabels", test_sql[["cited"]], embeddings=test_tensor)
print(f"TestLabels -> DataFrameSource: {len(test_sql)} test nodes")

# %% [markdown]
# ## 3. Init Session
#
# Pass the `RelationSource` instances to `Session`. `Engine.__init__` calls `load_full()` on each source at startup.  
# Legacy `(df, tensor)` entries continue to work alongside `RelationSource` objects without any change.

# %%
session = Session(db={
    "Papers":     papers_src,
    "Citation":   citation_src,
    "Labels":     labels_src,
    "TestLabels": test_src,
})

# %% [markdown]
# ## 4. Define the model
#
# ### 2-layer message-passing network
#
# Same depth as hello-world: first linear maps **1433 → 16**, then **ReLU**, then **16 → 7**, with unweighted `sum` over neighbors (SQLite `Citation` has no edge-weight tensor).  
# **Difference:** node inputs are **`\\text{Linear}(\\text{CoraNodeFeatures}(\\text{pid}))`** — the BoW row comes from RHS **encode** `[CoraNodeFeatures(pid)]`. The relation still carries a **placeholder** `z` (shape `(N,1)`) for the transformation runtime; it is not the 1433-dim feature vector.
#
# $$
# \\begin{align*}
#   \\text{PapersEmb}_1(\\text{pid}) & = \\text{Linear}_{1433 \\to 16}(\\text{CoraNodeFeatures}(\\text{pid})) \\\\
#   \\text{PapersAgg}_1(\\text{cited}) & = \\sum_{(\\text{citing}, \\text{cited}) \\in \\text{Citation}} \\text{PapersEmb}_1(\\text{citing}) \\\\
#   \\text{PapersAggNL}_1(\\text{cited}) & = \\text{ReLU}(\\text{PapersAgg}_1(\\text{cited})) \\\\
#   \\text{PapersEmb}_2(\\text{cited}) & = \\text{Linear}_{16 \\to 7}(\\text{PapersAggNL}_1(\\text{cited})) \\\\
#   \\text{PapersAgg}_2(\\text{cited}) & = \\sum_{(\\text{citing}, \\text{cited}) \\in \\text{Citation}} \\text{PapersEmb}_2(\\text{citing}) \\\\
#   \\text{Output}(\\text{cited}) & = \\text{PapersAgg}_2(\\text{cited})
# \\end{align*}
# $$

# %%
define_program = """
#lang:relnn
in_channels     = 1433 .
hidden_channels = 16 .
out_channels    = 7 .
lr              = 0.01 .
epochs          = 200 .
weight_decay    = 0.0005 .

# Layer 1 — BoW from RHS encode; z is a (N,1) placeholder (see DataFrameSource cell)
PapersEmb1(pid; Linear(in_channels, hidden_channels, False)([CoraNodeFeatures(pid)])) :- Papers(pid; z) .
PapersAgg1(cited; sum(z))          :- PapersEmb1(citing; z), Citation(citing, cited) .
PapersAggNL_Layer1(cited; ReLU(z)) :- PapersAgg1(cited; z) .

# Layer 2
PapersEmb2(cited; Linear(hidden_channels, out_channels, False)(z)) :- PapersAggNL_Layer1(cited; z) .
PapersAgg2(cited; sum(z))          :- PapersEmb2(citing; z), Citation(citing, cited) .
Output(cited; z)                   :- PapersAgg2(cited; z) .
"""

session.run(define_program)

# %% [markdown]
# ## 5. Train
#
# Cross-entropy on labeled nodes for 200 epochs.

# %%
fit_program = """
#lang:relnn
?fit <epochs=epochs, lr=lr, weight_decay=weight_decay>
Loss(; CrossEntropyLoss()(z_pred, z)) :- Output(cited; z_pred), Labels(cited; z) .
"""

session.run(fit_program)

# %% [markdown]
# ## 6. Predict
#
# Predicted class (0–6) for every node via `ArgMax` on logits (same as hello-world).

# %%
pred_program = """
#lang:relnn
?pred Predictions(cited; ArgMax()(z)) :- Output(cited; z) .
"""

pred_result = session.run(pred_program)
print("Predictions:")
pred_result

# %% [markdown]
# ## 7. Evaluate accuracy
#
# Join `Predictions` with `TestLabels` (loaded from SQL) and compute the mean equality.

# %%
eval_program = """
#lang:relnn
?pred Accuracy(; mean((z_pred == z_label) * 1.0)) :-
    Predictions(cited; z_pred), TestLabels(cited; z_label) .
"""
acc = session.run(eval_program).embeddings[0].item()
print(f"Test Accuracy: {acc:.1%}")

# %% [markdown]
# ## 8. Parameters
#
# Learned parameters — identical to the hello-world model (same architecture, different data backend).

# %%
session.show_params(show_stats=False)

# %% [markdown]
# ## 9. Term graph
#
# Compiled from the same rules as the hello-world; the data backend is invisible after `load_full()` at startup.

# %%
session.show_term_graph()

# %% [markdown]
# ---
#
# ## Summary
#
# | Component | Role |
# |---|---|
# | `SqlSource` | Loads `Citation` directly from SQLite — zero code beyond the connection string. Swap to any SQLAlchemy dialect. |
# | `DataFrameSource` | Wraps `Papers`, `Labels`, `TestLabels` after tensor reconstruction from SQL numeric columns. |
# | `Engine.__init__` | Calls `source.load_full()` once per source at startup; the engine sees a plain ER-dict regardless of origin. |
# | RelNN program | **Unchanged** — two Linear layers, sum aggregation, CrossEntropyLoss. |
#
# **To use Postgres or MySQL** swap the connection string:  
# `SqlSource("Citation", "postgresql://user:pass@host/db", table="citation")` — no other changes needed.
#
# **Next steps:**
# - [`examples/008_relnn_dblp_sqlite.py`](008_relnn_dblp_sqlite.ipynb) — DBLP HGT showcase (larger graph, multiple node types) using the same SQL-loading pattern.
# - [`scripts/data_setup/cora_from_sqlite/build_cora_sqlite.py`](../scripts/data_setup/cora_from_sqlite/build_cora_sqlite.py) — standalone version of the `build_cora_sqlite()` helper above.
