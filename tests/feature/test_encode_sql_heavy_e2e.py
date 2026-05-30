"""Heavy SQL + RHS encode e2e tests (slow downloads or opt-in).

Tier 2: real Cora dumped to a temporary SQLite file (007-style schema), few epochs.
Tier 3: synthetic multi-table SQL (pub/citation-ish), opt-in via RELNN_E2E_SQL_GRAPH=1.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
import torch
import torch.nn as nn
from relann.data_sources import DataFrameSource, SqlSource
from relann.session import Session
from relann.torch_utils import full_seed

sqlalchemy = pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed")

# --- Tier 2: frozen Cora rows for temp-SQLite test ---------------------------------

_CORA_SQL_E2E_WEIGHT: Optional[torch.Tensor] = None

class CoraSqlE2EFeatures(nn.Module):
    """Same role as 007 ``CoraNodeFeatures``: pid -> frozen Planetoid ``x`` row."""

    def __init__(self):
        super().__init__()
        if _CORA_SQL_E2E_WEIGHT is None:
            raise RuntimeError("CoraSqlE2EFeatures: weight not set")
        self.register_buffer("weight", _CORA_SQL_E2E_WEIGHT)

    def forward(self, x):
        idx = x.long().clamp(0, self.weight.size(0) - 1).squeeze(-1)
        return self.weight[idx]

def _set_cora_sql_e2e_weight(w: torch.Tensor) -> None:
    global _CORA_SQL_E2E_WEIGHT
    _CORA_SQL_E2E_WEIGHT = w

# =============================================================================
# Tier 2 — slow: Cora -> temp SQLite, 007-style encode + 2-layer GCN + fit
# =============================================================================

@pytest.mark.slow
@pytest.mark.download
def test_slow_cora_from_sqlite_file_encode_gcn_fit(tmp_path):
    """Build pid-only Papers + Citation/Labels in a real sqlite file; encode + train briefly."""
    pytest.importorskip("torch_geometric", reason="Planetoid Cora requires torch_geometric")

    from sqlalchemy import create_engine

    from relann.datasets import load_cora_dataset

    full_seed(0)
    try:
        data = load_cora_dataset()
    except Exception as exc:  # noqa: BLE001 — surface skip reason
        pytest.skip(f"Could not load Cora (cache/network): {exc}")

    raw = data.to_dict()
    papers_df, papers_tensor = raw["Papers"]
    citation_df, _ = raw["Citation"]
    labels_df, labels_t = raw["Labels"]
    test_df, test_t = raw["TestLabels"]

    papers_out = papers_df.reset_index(drop=True)
    labels_out = labels_df.copy().reset_index(drop=True)
    labels_out["label"] = labels_t.squeeze(1).long().cpu().numpy()
    labels_out = labels_out.rename(columns={"target_id": "cited"})
    test_out = test_df.copy().reset_index(drop=True)
    test_out["label"] = test_t.squeeze(1).cpu().numpy().astype(int)

    # Match 007 / GCN rules: edge and label keys use ``cited`` (datasets.py uses ``target_id``).
    citation_out = citation_df.reset_index(drop=True).rename(columns={"target_id": "cited"})

    db_path = tmp_path / "cora_sql_e2e.sqlite"
    url = f"sqlite:///{db_path}"
    engine = create_engine(url, echo=False)
    with engine.connect() as conn:
        for name, df in [
            ("Papers", papers_out),
            ("Citation", citation_out),
            ("Labels", labels_out),
            ("TestLabels", test_out),
        ]:
            df.to_sql(name, conn, if_exists="replace", index=False)
            conn.commit()
    engine.dispose()

    _set_cora_sql_e2e_weight(papers_tensor.detach().cpu())

    from sqlalchemy import create_engine as ce2

    sa_engine = ce2(url)
    with sa_engine.connect() as conn:
        papers_sql = pd.read_sql_table("Papers", conn)
    n = len(papers_sql)
    placeholder = torch.zeros(n, 1, dtype=torch.float32)
    papers_src = DataFrameSource("Papers", papers_sql[["pid"]], embeddings=placeholder)
    citation_src = SqlSource("Citation", url, table="Citation")

    with sa_engine.connect() as conn:
        labels_sql = pd.read_sql_table("Labels", conn)
    labels_tensor = torch.tensor(labels_sql["label"].values, dtype=torch.long).unsqueeze(1)
    labels_src = DataFrameSource("Labels", labels_sql[["cited"]], embeddings=labels_tensor)

    session = Session(db={
        "Papers": papers_src,
        "Citation": citation_src,
        "Labels": labels_src,
    })

    session.run(
        """
#lang:relnn
in_channels     = 1433 .
hidden_channels = 16 .
out_channels    = 7 .

PapersEmb1(pid; Linear(in_channels, hidden_channels, False)([CoraSqlE2EFeatures(pid)])) :- Papers(pid; z) .
PapersAgg1(cited; sum(z)) :- PapersEmb1(citing; z), Citation(citing, cited) .
PapersNL(cited; ReLU(z)) :- PapersAgg1(cited; z) .
PapersEmb2(cited; Linear(hidden_channels, out_channels, False)(z)) :- PapersNL(cited; z) .
PapersAgg2(cited; sum(z)) :- PapersEmb2(citing; z), Citation(citing, cited) .
Output(cited; z) :- PapersAgg2(cited; z) .
"""
    )
    session.run(
        """
#lang:relnn
?fit <epochs=3, lr=0.01, weight_decay=0.0005>
Loss(; CrossEntropyLoss()(z_pred, y)) :- Output(cited; z_pred), Labels(cited; y) .
"""
    )
    hist = session.engine.trained_modules["Loss"]["loss_history"]
    assert len(hist) == 3
    assert all(torch.isfinite(torch.tensor(v, dtype=torch.float64)) for v in hist)

# =============================================================================
# Tier 3 — opt-in: synthetic SQL schema + Embedding(topic) + 1-hop agg + fit
# =============================================================================

@pytest.mark.feature
def test_optin_synthetic_pub_graph_sql_encode_fit():
    """Multi-table SQLite (paper + cite); encode integer ``topic`` via Embedding; opt-in only."""
    if os.environ.get("RELNN_E2E_SQL_GRAPH", "").strip() != "1":
        pytest.skip("Set RELNN_E2E_SQL_GRAPH=1 to run this integration test")

    from sqlalchemy import create_engine

    full_seed(7)
    n, n_cls = 32, 3
    rng = torch.Generator().manual_seed(7)

    papers = pd.DataFrame(
        {
            "pid": range(n),
            "topic": [i % 5 for i in range(n)],
        }
    )
    citing, cited = [], []
    for i in range(n):
        citing.append(i)
        cited.append(i)
        citing.append(i)
        cited.append((i + 1) % n)
    cites = pd.DataFrame({"citing": citing, "cited": cited})
    labels_y = torch.randint(0, n_cls, (n, 1), generator=rng).long()
    labels = pd.DataFrame({"cited": range(n)})

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    with engine.connect() as conn:
        papers.to_sql("Paper", conn, if_exists="replace", index=False)
        cites.to_sql("Cites", conn, if_exists="replace", index=False)
        conn.commit()

    ph = torch.zeros(n, 1, dtype=torch.float32)
    with engine.connect() as conn:
        paper_loaded = pd.read_sql_table("Paper", conn)
    # Numeric columns tensorize as float32; categorical uses int codes for Embedding.
    paper_loaded["topic"] = paper_loaded["topic"].astype("int64").astype("category")
    paper_src = DataFrameSource("Paper", paper_loaded, embeddings=ph)
    cites_src = SqlSource("Cites", engine, table="Cites")
    lab_src = DataFrameSource("Labels", labels, embeddings=labels_y)

    session = Session(db={
        "Paper": paper_src,
        "Cites": cites_src,
        "Labels": lab_src,
    })

    session.run(
        """
#lang:relnn
emb_dim = 6 .
hidden = 12 .
n_cls = 3 .

H(pid; Linear(emb_dim, hidden)([Embedding(5, emb_dim)(topic)])) :- Paper(pid, topic; z) .
Agg(cited; sum(h)) :- H(citing; h), Cites(citing, cited) .
Out(cited; ReLU(z)) :- Agg(cited; z) .
Logits(cited; Linear(hidden, n_cls, False)(z)) :- Out(cited; z) .
"""
    )
    session.run(
        """
#lang:relnn
?fit <epochs=4, lr=0.05>
Loss(; CrossEntropyLoss()(z_pred, y)) :- Logits(cited; z_pred), Labels(cited; y) .
"""
    )
    hist = session.engine.trained_modules["Loss"]["loss_history"]
    assert len(hist) == 4
    assert all(torch.isfinite(torch.tensor(v, dtype=torch.float64)) for v in hist)
