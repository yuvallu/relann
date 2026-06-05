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
# **TL;DR — RelNN HGT on DBLP, loaded from SQLite.**
# Same SQL-loading idea as [examples/007](007_relnn_sql_loading.ipynb), but on the larger DBLP heterogeneous graph
# (Author / Paper / Term / Conference) instead of Cora, and running a 1-layer multi-head
# HGT-style attention chain instead of a plain GCN. Demonstrates how `SqlSource`
# and `DataFrameSource` cooperate at a non-trivial scale.

# %% [markdown]
# # RelNN: HGT on DBLP from SQLite
#
# This is the DBLP analogue of [the SQL-loading tutorial](007_relnn_sql_loading.ipynb).
# The DBLP dataset is loaded from a SQLite database (built by the data-setup script
# linked below); we then run a 1-layer Paper→Author HGT to classify each author's
# research area.
#
# **Pre-requisite — build the SQLite once:**
#
# ```bash
# python scripts/data_setup/dblp_from_sqlite/build_dblp_sqlite.py
# ```
#
# That dumps the DBLP relation tables into `data/dblp_demo.sqlite`. After running
# it once, this notebook can be re-run as many times as you like.

# %% [markdown]
# ## Loading strategy
#
# - **Node tables** (Author / Paper / Term / Conference) carry high-dimensional
#   feature vectors stored as `f0`…`fN` columns in SQLite. We rebuild them into a
#   tensor and wrap with `DataFrameSource(name, df, embeddings=tensor)`.
# - **Edge tables** and **MetaRel** are loaded as `SqlSource` (content only, no
#   embeddings) — the engine just needs them for joins.
# - **AuthorLabels** carries the supervision tensor.
# - **AuthorMeta** has the train/val/test split masks (for evaluation, not training).
#
# We don't try to encode raw `fi` columns in the RelNN rules directly (DBLP authors
# have 334-dimensional features — concatenating 334 `[fi]` brackets would be silly).
# The recommended pattern is: load feature vectors as tensors, project with a
# `Linear(d_in, hidden)(z)` rule.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine as sa_create_engine

from relann.data_sources import DataFrameSource, SqlSource
from relann.session import Session
from relann.torch_utils import full_seed, get_project_root

# %% [markdown]
# ## Config
#
# Tweak `EPOCHS` / `LR` for quick iteration. The defaults below mirror the
# 50-epoch comparison in `tests/slow/run_compare_dblp_original_hgt.py`.

# %%
DB_PATH      = get_project_root() / "data" / "dblp_demo.sqlite"
EPOCHS       = 50
LR           = 0.005
WEIGHT_DECAY = 0.001
SEED         = 42
HIDDEN       = 64
HEADS        = 2

# %%
if not DB_PATH.exists():
    raise FileNotFoundError(
        f"{DB_PATH} not found. Build it first:\n"
        "    python scripts/data_setup/dblp_from_sqlite/build_dblp_sqlite.py"
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device     : {DEVICE}")
print(f"Loading    : {DB_PATH}")

# %% [markdown]
# ## Load DBLP relations from SQLite

# %%
def _table_names(engine) -> set:
    from sqlalchemy import inspect as sa_inspect
    insp = sa_inspect(engine)
    return set(insp.get_table_names())


def load_sources_from_sqlite(db_path: Path):
    """Build ``{name: RelationSource}`` for all DBLP relation tables.

    Node tables carry pre-built feature tensors (reassembled from
    ``f0…fN`` columns). Edge / label tables are loaded as ``SqlSource``
    (content only, no embeddings).
    """
    url = f"sqlite:///{db_path}"
    engine = sa_create_engine(url)
    sources: dict[str, object] = {}

    node_tables = {
        "Author":      "author_id",
        "Paper":       "paper_id",
        "Term":        "term_id",
        "Conference":  "conference_id",
    }
    edge_tables = [
        "AuthorPaper", "PaperAuthor", "PaperTerm",
        "PaperConference", "TermPaper", "ConferencePaper", "MetaRel",
    ]

    # Node tables — reassemble tensor from f0…fN columns.
    for tbl, id_col in node_tables.items():
        with engine.connect() as conn:
            df = pd.read_sql_table(tbl, conn)
        feat_cols = sorted(
            [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
            key=lambda c: int(c[1:]),
        )
        if feat_cols:
            tensor = torch.tensor(df[feat_cols].values, dtype=torch.float32)
        else:
            tensor = torch.empty(len(df), 0)
        content_df = df[[id_col]].copy()
        sources[tbl] = DataFrameSource(tbl, content_df, embeddings=tensor)
        print(f"  {tbl:12s}: {len(content_df):6d} rows, feature dim={len(feat_cols)}")

    # AuthorLabels — supervision tensor.
    with engine.connect() as conn:
        labels_df = pd.read_sql_table("AuthorLabels", conn)
    if "label" in labels_df.columns:
        lbl_tensor = torch.tensor(labels_df["label"].values, dtype=torch.long).unsqueeze(1)
        sources["AuthorLabels"] = DataFrameSource(
            "AuthorLabels", labels_df[["author_id"]].copy(), embeddings=lbl_tensor,
        )
    else:
        sources["AuthorLabels"] = SqlSource("AuthorLabels", url, table="AuthorLabels")
    print(f"  AuthorLabels: {len(labels_df):6d} rows")

    # AuthorMeta — train/val/test split masks (for evaluation).
    if "AuthorMeta" in _table_names(engine):
        with engine.connect() as conn:
            meta_df = pd.read_sql_table("AuthorMeta", conn)
        sources["AuthorMeta"] = DataFrameSource("AuthorMeta", meta_df)
        print(f"  AuthorMeta  : {len(meta_df):6d} rows")

    # Edge tables — SqlSource only.
    for tbl in edge_tables:
        sources[tbl] = SqlSource(tbl, url, table=tbl)

    engine.dispose()
    return sources


sources = load_sources_from_sqlite(DB_PATH)

# Infer feature dimensions from the loaded DataFrameSource tensors.
author_dim = sources["Author"].load_full()["embedding_shapes"][0][1]
paper_dim  = sources["Paper"].load_full()["embedding_shapes"][0][1]
n_classes  = 4   # DBLP has 4 author research areas.
dh         = HIDDEN // HEADS

print(f"\nauthor_dim = {author_dim}, paper_dim = {paper_dim}, dh = {dh}")

# %% [markdown]
# ## RelNN program: 1-layer Paper → Author HGT
#
# This is the same structure as `tests/slow/run_compare_dblp_original_hgt.py`'s
# PA-path baseline, transcribed from PyTorch into RelNN DSL. Two heads,
# attention-weighted message passing from Paper to Author, residual-style
# skip update, final classifier.

# %%
RELNN_PROGRAM = f"""
#lang:relnn
hidden = {HIDDEN} .
dh = {dh} .

AuthorProj = Linear({author_dim}, hidden) .
PaperProj  = Linear({paper_dim},  hidden) .

AuthorEmb(author_id; ReLU(AuthorProj(z))) :- Author(author_id; z) .
PaperEmb(paper_id;   ReLU(PaperProj(z)))  :- Paper(paper_id;   z) .

K_paper<head> = Linear(hidden, dh) .
Q_author<head> = Linear(hidden, dh) .
V_paper<head>  = Linear(hidden, dh) .

PaperK<head>(paper_id;  K_paper<head>(z))  :- PaperEmb(paper_id;  z) .
AuthorQ<head>(author_id; Q_author<head>(z)) :- AuthorEmb(author_id; z) .
PaperV<head>(paper_id;  V_paper<head>(z))   :- PaperEmb(paper_id;  z) .

Krel_PA<head> = Linear(dh, dh, False) .
Vrel_PA<head> = Linear(dh, dh, False) .
Prel_PA<head> = Tensor(1, 1.0) .

DotPA<head>(paper_id, author_id;
    view(1)(view(1, dh)(z_q) @ transpose(Krel_PA<head>(z_k))) * Prel_PA<head> / sqrt(dh)) :-
    PaperK<head>(paper_id; z_k), PaperAuthor(paper_id, author_id; w), AuthorQ<head>(author_id; z_q) .

MaxPA<head>(author_id; max(z))         :- DotPA<head>(paper_id, author_id; z) .
StableDotPA<head>(paper_id, author_id; z1 - z2) :- DotPA<head>(paper_id, author_id; z1), MaxPA<head>(author_id; z2) .
ExpPA<head>(paper_id, author_id; exp(z)) :- StableDotPA<head>(paper_id, author_id; z) .
DenomPA<head>(author_id; sum(z))       :- ExpPA<head>(paper_id, author_id; z) .
SoftPA<head>(paper_id, author_id; z1 / z2) :- ExpPA<head>(paper_id, author_id; z1), DenomPA<head>(author_id; z2) .

MsgPA<head>(paper_id, author_id; Vrel_PA<head>(z_v) * z_att) :-
    PaperV<head>(paper_id; z_v), PaperAuthor(paper_id, author_id; w), SoftPA<head>(paper_id, author_id; z_att) .
MsgPACon(paper_id, author_id; Concat(z1, z2)) :- MsgPA<1>(paper_id, author_id; z1), MsgPA<2>(paper_id, author_id; z2) .

AggAuthor(author_id; sum(z)) :- MsgPACon(paper_id, author_id; z) .

OutLin_author = Linear(hidden, hidden) .
Skip_author   = Tensor(1, 1.0) .

AutLinOut(author_id; OutLin_author(GELU(z))) :- AggAuthor(author_id; z) .
AuthorOut(author_id; Sigmoid(Skip_author) * z1 + (1 - Sigmoid(Skip_author)) * z2) :-
    AutLinOut(author_id; z1), AuthorEmb(author_id; z2) .

Classifier = Linear(hidden, {n_classes}) .
Output(author_id; z) :- AuthorOut(author_id; z) .
"""

FIT_PROGRAM = f"""
#lang:relnn
?fit <epochs={EPOCHS}, lr={LR}, weight_decay={WEIGHT_DECAY}>
Loss(; CrossEntropyLoss()(Classifier(z_pred), z)) :-
    Output(author_id; z_pred), AuthorLabels(author_id; z) .
"""

PRED_PROGRAM = """
#lang:relnn
?pred AuthorPred(author_id; ArgMax()(Classifier(z))) :- Output(author_id; z) .
"""

# %% [markdown]
# ## Train

# %%
full_seed(SEED)
session = Session(db=sources, device=DEVICE)

print("Defining model…")
session.run(RELNN_PROGRAM)

print(f"Training: hidden={HIDDEN}, heads={HEADS}, epochs={EPOCHS}, lr={LR}")
session.run(FIT_PROGRAM)

# %% [markdown]
# ## Predict and evaluate

# %%
pred = session.run(PRED_PROGRAM)

pred_df = pred.content.copy()
pred_class = pred.embeddings[0].cpu().numpy().flatten().astype(int)
pred_df["_pred_class"] = pred_class

if "AuthorMeta" in sources:
    meta_df = sources["AuthorMeta"].load_full()["content"]
    merged = pred_df.merge(meta_df, left_on="author_id", right_on="node_id", how="left")
    for split_col, split_name in [("is_train", "train"), ("is_val", "val"), ("is_test", "test")]:
        mask = merged[split_col].fillna(False).astype(bool)
        if mask.sum() > 0:
            correct = int(np.sum(
                merged.loc[mask, "_pred_class"].values
                == merged.loc[mask, "label"].values
            ))
            acc = correct / int(mask.sum())
            print(f"  {split_name:5s} accuracy: {acc:.4f}")
        else:
            print(f"  {split_name} split not found.")
else:
    print(f"  Predicted {len(pred_df)} authors.")
    print("  (AuthorMeta not found — cannot compute accuracy splits.)")

print("\nDemo complete.")

# %% [markdown]
# ## Next steps
#
# - [`tests/slow/run_compare_dblp_hgt_generic.py`](../tests/slow/run_compare_dblp_hgt_generic.py) — the full rigorous comparison (param-count, weight-synced forward diff, training accuracy vs hand-rolled PyTorch) that produced the published 2026-03-30 DBLP benchmark.
# - [`scripts/data_setup/dblp_from_sqlite/build_dblp_sqlite.py`](../scripts/data_setup/dblp_from_sqlite/build_dblp_sqlite.py) — how the SQLite was built.
