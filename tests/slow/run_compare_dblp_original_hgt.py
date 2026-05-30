"""Compare original paper HGT vs PyG HGTConv vs RelNN on DBLP.

This script provides explicit scope labels:
  - SCOPE=FULL_GRAPH_1L: Original pyHGT vs PyG (all 6 edge types)
  - SCOPE=FULL_GRAPH_2L: Original pyHGT vs PyG (all 6 edge types)
  - SCOPE=PA_PATH_1L: RelNN PA-path reference row (for context)

Timing uses synchronized GPU timers when CUDA is available.
"""
import sys
import math
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Add the cloned pyHGT to path
_pyhgt_path = Path(__file__).resolve().parents[2] / "_external" / "pyHGT" / "OAG"
sys.path.insert(0, str(_pyhgt_path))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv as PyGHGTConv, Linear as PyGLinear

from relann.torch_utils import full_seed, get_project_root
from relann.session import Session
from relann.datasets import load_dblp_dataset

DIVIDER = "=" * 70

# -- Load DBLP ---------------------------------------------------------------

dblp = load_dblp_dataset()
print(dblp)
print()

pyg_data = dblp.pyg_data
relnn_db = dblp.db
info = dblp.dataset_info

hidden = 64
num_heads = 2
n_classes = info["num_classes"]  # 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using {DEVICE}")

y_author = pyg_data["author"].y.to(DEVICE)
train_mask = pyg_data["author"].train_mask.to(DEVICE)
val_mask = pyg_data["author"].val_mask.to(DEVICE)
test_mask = pyg_data["author"].test_mask.to(DEVICE)

# Pre-load pyg_data to GPU once so forward() has zero CPU->GPU transfers.
pyg_data = pyg_data.to(DEVICE)

# -- Prepare flat representation for original pyHGT --------------------------

NODE_TYPE_ORDER = ["author", "paper", "term", "conference"]
node_type_to_id = {nt: i for i, nt in enumerate(NODE_TYPE_ORDER)}

node_features = {}
node_offsets = {}
offset = 0
for nt in NODE_TYPE_ORDER:
    x = pyg_data[nt].x  # already on DEVICE after pyg_data.to(DEVICE)
    node_features[nt] = x
    node_offsets[nt] = offset
    offset += x.size(0)

n_total_nodes = offset
max_feat_dim = max(x.size(1) for x in node_features.values())

node_inp_flat = torch.zeros(n_total_nodes, max_feat_dim, device=DEVICE)
node_type_flat = torch.zeros(n_total_nodes, dtype=torch.long, device=DEVICE)

for nt in NODE_TYPE_ORDER:
    x = node_features[nt]
    o = node_offsets[nt]
    n = x.size(0)
    node_inp_flat[o:o+n, :x.size(1)] = x.to(DEVICE)
    node_type_flat[o:o+n] = node_type_to_id[nt]

EDGE_TYPE_MAP = {}
all_edge_src = []
all_edge_dst = []
all_edge_types = []
edge_type_counter = 0

for edge_type_key, edge_index in pyg_data.edge_index_dict.items():
    src_type, rel, dst_type = edge_type_key
    EDGE_TYPE_MAP[edge_type_key] = edge_type_counter
    src_offset = node_offsets[src_type]
    dst_offset = node_offsets[dst_type]
    all_edge_src.append(edge_index[0] + src_offset)
    all_edge_dst.append(edge_index[1] + dst_offset)
    all_edge_types.append(torch.full((edge_index.size(1),), edge_type_counter, dtype=torch.long, device=DEVICE))
    edge_type_counter += 1

edge_index_flat = torch.stack([torch.cat(all_edge_src), torch.cat(all_edge_dst)]).to(DEVICE)
edge_type_flat = torch.cat(all_edge_types)
edge_time_flat = torch.zeros(edge_index_flat.size(1), dtype=torch.long, device=DEVICE)

n_edge_types = edge_type_counter
n_node_types = len(NODE_TYPE_ORDER)
author_offset = node_offsets["author"]
n_authors = node_features["author"].size(0)

print(f"Flat graph: {n_total_nodes} nodes, {edge_index_flat.size(1)} edges, "
      f"{n_node_types} node types, {n_edge_types} edge types, max_feat_dim={max_feat_dim}")

# =============================================================================
# 1. Original paper HGT (acbull/pyHGT)
# =============================================================================

from pyHGT.conv import HGTConv as OriginalHGTConv

class OriginalPaperHGT(nn.Module):
    """Wraps the original paper's HGTConv with per-type input projection + classifier.

    Follows the same pattern as pyHGT's GNN class but uses per-type input
    projections to handle heterogeneous feature dimensions naturally.
    """
    def __init__(self, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.adapt_ws = nn.ModuleDict()
        for nt in NODE_TYPE_ORDER:
            in_dim = node_features[nt].size(1)
            self.adapt_ws[nt] = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([
            OriginalHGTConv(
                in_dim=hidden,
                out_dim=hidden,
                num_types=n_node_types,
                num_relations=n_edge_types,
                n_heads=num_heads,
                dropout=0.2,
                use_norm=True,
                use_RTE=False,
            ) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self):
        res = torch.zeros(n_total_nodes, hidden, device=DEVICE)
        for nt in NODE_TYPE_ORDER:
            idx = (node_type_flat == node_type_to_id[nt])
            res[idx] = F.relu(self.adapt_ws[nt](node_features[nt]))
        out = res
        for conv in self.convs:
            out = conv(out, node_type_flat, edge_index_flat, edge_type_flat, edge_time_flat)
        author_out = out[author_offset:author_offset + n_authors]
        return self.classifier(author_out)

def _sync_cuda():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

def run_original_hgt(seed=42, epochs=100, lr=0.005, wd=0.001, num_layers=1):
    full_seed(seed)
    model = OriginalPaperHGT(num_layers=num_layers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n_params = sum(p.numel() for p in model.parameters())

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model()
        loss = F.cross_entropy(logits[train_mask], y_author[train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = model().argmax(dim=-1)
    accs = {}
    for split, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        accs[split] = (pred[mask] == y_author[mask]).float().mean().item()

    return model, n_params, losses, accs

# =============================================================================
# 2. PyG HGTConv (same as existing comparison)
# =============================================================================

class PyGHGT(nn.Module):
    def __init__(self, num_layers=1):
        super().__init__()
        self.lin_dict = nn.ModuleDict()
        for node_type in pyg_data.node_types:
            self.lin_dict[node_type] = PyGLinear(-1, hidden)
        self.convs = nn.ModuleList([PyGHGTConv(hidden, hidden, pyg_data.metadata(), num_heads) for _ in range(num_layers)])
        self.lin = PyGLinear(hidden, n_classes)

    def forward(self):
        x_dict = {nt: self.lin_dict[nt](pyg_data[nt].x).relu_() for nt in pyg_data.node_types}
        for conv in self.convs:
            x_dict = conv(x_dict, pyg_data.edge_index_dict)
        return self.lin(x_dict["author"])

def run_pyg_hgt(seed=42, epochs=100, lr=0.005, wd=0.001, num_layers=1):
    full_seed(seed)
    model = PyGHGT(num_layers=num_layers).to(DEVICE)
    with torch.no_grad():
        model()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n_params = sum(p.numel() for p in model.parameters())

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model()
        loss = F.cross_entropy(logits[train_mask], y_author[train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = model().argmax(dim=-1)
    accs = {}
    for split, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        accs[split] = (pred[mask] == y_author[mask]).float().mean().item()

    return model, n_params, losses, accs

# =============================================================================
# 3. RelNN HGT (full heterogeneous graph via template)
# =============================================================================

RELNN_DEFINE_DSL = f"""
#lang:relnn
hidden = {hidden} .
dh = {hidden // num_heads} .

AuthorProj = Linear({info['author_features']}, hidden) .
PaperProj = Linear({info['paper_features']}, hidden) .

AuthorEmb(author_id; ReLU(AuthorProj(z))) :- Author(author_id; z) .
PaperEmb(paper_id; ReLU(PaperProj(z))) :- Paper(paper_id; z) .

K_paper<head> = Linear(hidden, dh) .
Q_author<head> = Linear(hidden, dh) .
V_paper<head> = Linear(hidden, dh) .

PaperK<head>(paper_id; K_paper<head>(z)) :- PaperEmb(paper_id; z) .
AuthorQ<head>(author_id; Q_author<head>(z)) :- AuthorEmb(author_id; z) .
PaperV<head>(paper_id; V_paper<head>(z)) :- PaperEmb(paper_id; z) .

Krel_PA<head> = Linear(dh, dh, False) .
Vrel_PA<head> = Linear(dh, dh, False) .
Prel_PA<head> = Tensor(1) .

DotPA<head>(paper_id, author_id; view(1)(view(1, dh)(z_q) @ transpose(Krel_PA<head>(z_k))) * Prel_PA<head> / sqrt(dh)) :-
    PaperK<head>(paper_id; z_k), PaperAuthor(paper_id, author_id; w), AuthorQ<head>(author_id; z_q) .

ExpPA<head>(paper_id, author_id; exp(z)) :- DotPA<head>(paper_id, author_id; z) .
DenomPA<head>(author_id; sum(z)) :- ExpPA<head>(paper_id, author_id; z) .
SoftPA<head>(paper_id, author_id; z1 / z2) :- ExpPA<head>(paper_id, author_id; z1), DenomPA<head>(author_id; z2) .

MsgPA<head>(paper_id, author_id; Vrel_PA<head>(z_v) * z_att) :- PaperV<head>(paper_id; z_v), PaperAuthor(paper_id, author_id; w), SoftPA<head>(paper_id, author_id; z_att) .
MsgPACon(paper_id, author_id; Concat(z1, z2)) :- MsgPA<1>(paper_id, author_id; z1), MsgPA<2>(paper_id, author_id; z2) .

AggAuthor(author_id; sum(z)) :- MsgPACon(paper_id, author_id; z) .

OutLin_author = Linear(hidden, hidden) .
Skip_author = Tensor(1) .

AutLinOut(author_id; OutLin_author(GELU(z))) :- AggAuthor(author_id; z) .
AuthorOut(author_id; Sigmoid(Skip_author) * z1 + (1 - Sigmoid(Skip_author)) * z2) :- AutLinOut(author_id; z1), AuthorEmb(author_id; z2) .

Classifier = Linear(hidden, {n_classes}) .
Output(author_id; z) :- AuthorOut(author_id; z) .
"""

# Full 2-layer HGT: L1 updates Paper from {Author,Term,Conf} and Author from
# Paper; L2 uses updated Paper/Author from L1 for the final PA attention.
# This matches the full-graph PyG HGTConv computation for author classification.
RELNN_DEFINE_DSL_2L = f"""
#lang:relnn
hidden = {hidden} .
dh = {hidden // num_heads} .

# --- Initial projections (4 node types) ---
AuthorProj = Linear({info['author_features']}, hidden) .
PaperProj  = Linear({info['paper_features']},  hidden) .
TermProj   = Linear({info['term_features']},   hidden) .
ConfProj   = Linear({info['conference_features']}, hidden) .

AuthorEmb0(author_id;     ReLU(AuthorProj(z))) :- Author(author_id; z) .
PaperEmb0(paper_id;       ReLU(PaperProj(z)))  :- Paper(paper_id; z) .
TermEmb0(term_id;         ReLU(TermProj(z)))   :- Term(term_id; z) .
ConfEmb0(conference_id;   ReLU(ConfProj(z)))   :- Conference(conference_id; z) .

# === LAYER 1 ===
# --- L1 KQV parameters (per node type, per head) ---
K_paper_L1<head>  = Linear(hidden, dh) .
Q_paper_L1<head>  = Linear(hidden, dh) .
V_paper_L1<head>  = Linear(hidden, dh) .
K_author_L1<head> = Linear(hidden, dh) .
Q_author_L1<head> = Linear(hidden, dh) .
V_author_L1<head> = Linear(hidden, dh) .
K_term_L1<head>   = Linear(hidden, dh) .
V_term_L1<head>   = Linear(hidden, dh) .
K_conf_L1<head>   = Linear(hidden, dh) .
V_conf_L1<head>   = Linear(hidden, dh) .

PaperK_L1<head>(paper_id;       K_paper_L1<head>(z))  :- PaperEmb0(paper_id; z) .
PaperQ_L1<head>(paper_id;       Q_paper_L1<head>(z))  :- PaperEmb0(paper_id; z) .
PaperV_L1<head>(paper_id;       V_paper_L1<head>(z))  :- PaperEmb0(paper_id; z) .
AuthorK_L1<head>(author_id;     K_author_L1<head>(z)) :- AuthorEmb0(author_id; z) .
AuthorQ_L1<head>(author_id;     Q_author_L1<head>(z)) :- AuthorEmb0(author_id; z) .
AuthorV_L1<head>(author_id;     V_author_L1<head>(z)) :- AuthorEmb0(author_id; z) .
TermK_L1<head>(term_id;         K_term_L1<head>(z))   :- TermEmb0(term_id; z) .
TermV_L1<head>(term_id;         V_term_L1<head>(z))   :- TermEmb0(term_id; z) .
ConfK_L1<head>(conference_id;   K_conf_L1<head>(z))   :- ConfEmb0(conference_id; z) .
ConfV_L1<head>(conference_id;   V_conf_L1<head>(z))   :- ConfEmb0(conference_id; z) .

# --- L1 Edge 1: Paper->Author (PA) ---
Krel_PA_L1<head> = Linear(dh, dh, False) .
Vrel_PA_L1<head> = Linear(dh, dh, False) .
Prel_PA_L1<head> = Tensor(1) .

DotPA_L1<head>(paper_id, author_id; view(1)(view(1, dh)(z_q) @ transpose(Krel_PA_L1<head>(z_k))) * Prel_PA_L1<head> / sqrt(dh)) :-
    PaperK_L1<head>(paper_id; z_k), PaperAuthor(paper_id, author_id; w), AuthorQ_L1<head>(author_id; z_q) .
ExpPA_L1<head>(paper_id, author_id; exp(z))   :- DotPA_L1<head>(paper_id, author_id; z) .
DenomPA_L1<head>(author_id; sum(z))            :- ExpPA_L1<head>(paper_id, author_id; z) .
SoftPA_L1<head>(paper_id, author_id; z1 / z2) :- ExpPA_L1<head>(paper_id, author_id; z1), DenomPA_L1<head>(author_id; z2) .
MsgPA_L1<head>(paper_id, author_id; Vrel_PA_L1<head>(z_v) * z_att) :-
    PaperV_L1<head>(paper_id; z_v), PaperAuthor(paper_id, author_id; w), SoftPA_L1<head>(paper_id, author_id; z_att) .
MsgPACon_L1(paper_id, author_id; Concat(z1, z2)) :- MsgPA_L1<1>(paper_id, author_id; z1), MsgPA_L1<2>(paper_id, author_id; z2) .
AggAuthor_L1(author_id; sum(z)) :- MsgPACon_L1(paper_id, author_id; z) .

# --- L1 Edges 2-4: AP + TP + CP targeting Paper (global softmax across all types) ---
# PyG's HGTConv uses a single bipartite edge index combining all edge types,
# so softmax normalizes across ALL incoming edges (AP + TP + CP) per paper.

Krel_AP_L1<head> = Linear(dh, dh, False) .
Vrel_AP_L1<head> = Linear(dh, dh, False) .
Prel_AP_L1<head> = Tensor(1) .

DotAP_L1<head>(author_id, paper_id; view(1)(view(1, dh)(z_q) @ transpose(Krel_AP_L1<head>(z_k))) * Prel_AP_L1<head> / sqrt(dh)) :-
    AuthorK_L1<head>(author_id; z_k), AuthorPaper(author_id, paper_id; w), PaperQ_L1<head>(paper_id; z_q) .
ExpAP_L1<head>(author_id, paper_id; exp(z)) :- DotAP_L1<head>(author_id, paper_id; z) .

Krel_TP_L1<head> = Linear(dh, dh, False) .
Vrel_TP_L1<head> = Linear(dh, dh, False) .
Prel_TP_L1<head> = Tensor(1) .

DotTP_L1<head>(term_id, paper_id; view(1)(view(1, dh)(z_q) @ transpose(Krel_TP_L1<head>(z_k))) * Prel_TP_L1<head> / sqrt(dh)) :-
    TermK_L1<head>(term_id; z_k), TermPaper(term_id, paper_id; w), PaperQ_L1<head>(paper_id; z_q) .
ExpTP_L1<head>(term_id, paper_id; exp(z)) :- DotTP_L1<head>(term_id, paper_id; z) .

Krel_CP_L1<head> = Linear(dh, dh, False) .
Vrel_CP_L1<head> = Linear(dh, dh, False) .
Prel_CP_L1<head> = Tensor(1) .

DotCP_L1<head>(conference_id, paper_id; view(1)(view(1, dh)(z_q) @ transpose(Krel_CP_L1<head>(z_k))) * Prel_CP_L1<head> / sqrt(dh)) :-
    ConfK_L1<head>(conference_id; z_k), ConferencePaper(conference_id, paper_id; w), PaperQ_L1<head>(paper_id; z_q) .
ExpCP_L1<head>(conference_id, paper_id; exp(z)) :- DotCP_L1<head>(conference_id, paper_id; z) .

# --- Global denominator: sum of all exp values per paper across AP + TP + CP ---
DenomAP_partial_L1<head>(paper_id; sum(z)) :- ExpAP_L1<head>(author_id, paper_id; z) .
DenomTP_partial_L1<head>(paper_id; sum(z)) :- ExpTP_L1<head>(term_id, paper_id; z) .
DenomCP_partial_L1<head>(paper_id; sum(z)) :- ExpCP_L1<head>(conference_id, paper_id; z) .

# Left-join TP denominator: every paper gets TP contribution (zero if no term edges).
# DenomCP_partial_L1 covers all 14328 DBLP papers (every paper has exactly 1 conference),
# so it is the correct zero-scalar reference with matching shape (N, 1).
DenomTP_partial_L1_lj<head>(paper_id; sum(z)) :- DenomCP_partial_L1<head>(paper_id; 0) | DenomTP_partial_L1<head>(paper_id; z) .
DenomPaper_L1<head>(paper_id; z1 + z2 + z3) :-
    DenomAP_partial_L1<head>(paper_id; z1), DenomTP_partial_L1_lj<head>(paper_id; z2), DenomCP_partial_L1<head>(paper_id; z3) .

# --- Global softmax per edge type (each normalized by the global per-paper denominator) ---
SoftAP_L1<head>(author_id, paper_id; z1 / z2) :- ExpAP_L1<head>(author_id, paper_id; z1), DenomPaper_L1<head>(paper_id; z2) .
SoftTP_L1<head>(term_id, paper_id; z1 / z2)   :- ExpTP_L1<head>(term_id, paper_id; z1),   DenomPaper_L1<head>(paper_id; z2) .
SoftCP_L1<head>(conference_id, paper_id; z1 / z2) :- ExpCP_L1<head>(conference_id, paper_id; z1), DenomPaper_L1<head>(paper_id; z2) .

MsgAP_L1<head>(author_id, paper_id; Vrel_AP_L1<head>(z_v) * z_att) :-
    AuthorV_L1<head>(author_id; z_v), AuthorPaper(author_id, paper_id; w), SoftAP_L1<head>(author_id, paper_id; z_att) .
MsgTP_L1<head>(term_id, paper_id; Vrel_TP_L1<head>(z_v) * z_att) :-
    TermV_L1<head>(term_id; z_v), TermPaper(term_id, paper_id; w), SoftTP_L1<head>(term_id, paper_id; z_att) .
MsgCP_L1<head>(conference_id, paper_id; Vrel_CP_L1<head>(z_v) * z_att) :-
    ConfV_L1<head>(conference_id; z_v), ConferencePaper(conference_id, paper_id; w), SoftCP_L1<head>(conference_id, paper_id; z_att) .

MsgAPCon_L1(author_id, paper_id; Concat(z1, z2)) :- MsgAP_L1<1>(author_id, paper_id; z1), MsgAP_L1<2>(author_id, paper_id; z2) .
MsgTPCon_L1(term_id, paper_id; Concat(z1, z2))   :- MsgTP_L1<1>(term_id, paper_id; z1), MsgTP_L1<2>(term_id, paper_id; z2) .
MsgCPCon_L1(conference_id, paper_id; Concat(z1, z2)) :- MsgCP_L1<1>(conference_id, paper_id; z1), MsgCP_L1<2>(conference_id, paper_id; z2) .

AggPaperFromAP_L1(paper_id; sum(z)) :- MsgAPCon_L1(author_id, paper_id; z) .
AggPaperFromTP_L1(paper_id; sum(z)) :- MsgTPCon_L1(term_id, paper_id; z) .
AggPaperFromCP_L1(paper_id; sum(z)) :- MsgCPCon_L1(conference_id, paper_id; z) .

# Left-join AP and TP aggregations: ensure every paper appears even with zero contribution
AggPaperFromAP_L1_lj(paper_id; sum(z)) :- PaperEmb0(paper_id; 0) | AggPaperFromAP_L1(paper_id; z) .
AggPaperFromTP_L1_lj(paper_id; sum(z)) :- PaperEmb0(paper_id; 0) | AggPaperFromTP_L1(paper_id; z) .

# --- L1 Output: Author ---
OutLin_author_L1 = Linear(hidden, hidden) .
Skip_author_L1   = Tensor(1) .

AutLinOut_L1(author_id; OutLin_author_L1(GELU(z))) :- AggAuthor_L1(author_id; z) .
AuthorOut1(author_id; Sigmoid(Skip_author_L1) * z1 + (1 - Sigmoid(Skip_author_L1)) * z2) :-
    AutLinOut_L1(author_id; z1), AuthorEmb0(author_id; z2) .

# --- L1 Output: Paper (AP + TP + CP aggregations joined and summed) ---
OutLin_paper_L1 = Linear(hidden, hidden) .
Skip_paper_L1   = Tensor(1) .

AggPaper_L1(paper_id; z1 + z2 + z3) :-
    AggPaperFromAP_L1_lj(paper_id; z1), AggPaperFromTP_L1_lj(paper_id; z2), AggPaperFromCP_L1(paper_id; z3) .
PapLinOut_L1(paper_id; OutLin_paper_L1(GELU(z))) :- AggPaper_L1(paper_id; z) .
PaperOut1(paper_id; Sigmoid(Skip_paper_L1) * z1 + (1 - Sigmoid(Skip_paper_L1)) * z2) :-
    PapLinOut_L1(paper_id; z1), PaperEmb0(paper_id; z2) .

# === LAYER 2: Paper->Author only (using updated L1 embeddings) ===
K_paper_L2<head>  = Linear(hidden, dh) .
Q_author_L2<head> = Linear(hidden, dh) .
V_paper_L2<head>  = Linear(hidden, dh) .
Krel_PA_L2<head>  = Linear(dh, dh, False) .
Vrel_PA_L2<head>  = Linear(dh, dh, False) .
Prel_PA_L2<head>  = Tensor(1) .
OutLin_author_L2  = Linear(hidden, hidden) .
Skip_author_L2    = Tensor(1) .

PaperK_L2<head>(paper_id;   K_paper_L2<head>(z))  :- PaperOut1(paper_id; z) .
AuthorQ_L2<head>(author_id; Q_author_L2<head>(z)) :- AuthorOut1(author_id; z) .
PaperV_L2<head>(paper_id;   V_paper_L2<head>(z))  :- PaperOut1(paper_id; z) .

DotPA_L2<head>(paper_id, author_id; view(1)(view(1, dh)(z_q) @ transpose(Krel_PA_L2<head>(z_k))) * Prel_PA_L2<head> / sqrt(dh)) :-
    PaperK_L2<head>(paper_id; z_k), PaperAuthor(paper_id, author_id; w), AuthorQ_L2<head>(author_id; z_q) .
ExpPA_L2<head>(paper_id, author_id; exp(z))   :- DotPA_L2<head>(paper_id, author_id; z) .
DenomPA_L2<head>(author_id; sum(z))            :- ExpPA_L2<head>(paper_id, author_id; z) .
SoftPA_L2<head>(paper_id, author_id; z1 / z2) :- ExpPA_L2<head>(paper_id, author_id; z1), DenomPA_L2<head>(author_id; z2) .
MsgPA_L2<head>(paper_id, author_id; Vrel_PA_L2<head>(z_v) * z_att) :-
    PaperV_L2<head>(paper_id; z_v), PaperAuthor(paper_id, author_id; w), SoftPA_L2<head>(paper_id, author_id; z_att) .
MsgPACon_L2(paper_id, author_id; Concat(z1, z2)) :- MsgPA_L2<1>(paper_id, author_id; z1), MsgPA_L2<2>(paper_id, author_id; z2) .
AggAuthor_L2(author_id; sum(z)) :- MsgPACon_L2(paper_id, author_id; z) .
AutLinOut_L2(author_id; OutLin_author_L2(GELU(z))) :- AggAuthor_L2(author_id; z) .
AuthorOut2(author_id; Sigmoid(Skip_author_L2) * z1 + (1 - Sigmoid(Skip_author_L2)) * z2) :-
    AutLinOut_L2(author_id; z1), AuthorOut1(author_id; z2) .

Classifier = Linear(hidden, {n_classes}) .
Output(author_id; z) :- AuthorOut2(author_id; z) .
"""

def run_relnn_hgt(seed=42, epochs=100, lr=0.005, wd=0.001, num_layers=1):
    full_seed(seed)
    session = Session(db=relnn_db, device=DEVICE)
    if num_layers == 1:
        session.run(RELNN_DEFINE_DSL)
    elif num_layers == 2:
        session.run(RELNN_DEFINE_DSL_2L)
    else:
        raise ValueError(f"Unsupported RelNN PA-path layers={num_layers}; expected 1 or 2")

    full_seed(seed)
    session.run(f"""
#lang:relnn
?fit <epochs={epochs}, lr={lr}, weight_decay={wd}>
Loss(; CrossEntropyLoss()(Classifier(z_pred), z)) :- Output(author_id; z_pred), AuthorLabels(author_id; z) .
""")

    rn_params = sum(p.numel() for p in session.engine.parameter_store.values())

    pred = session.run("""
#lang:relnn
?pred AuthorPred(author_id; ArgMax()(Classifier(z))) :- Output(author_id; z) .
""")

    pred_df = pred.content.copy()
    pred_class = pred.embeddings[0].cpu().numpy().flatten().astype(int)
    pred_df["_pred_class"] = pred_class
    merge_col = "author_id" if "author_id" in pred_df.columns else pred_df.columns[0]
    merged = pred_df.merge(dblp.node_metadata, left_on=merge_col, right_on="node_id", how="left")

    accs = {}
    for split_col, split_name in [("is_train", "train"), ("is_val", "val"), ("is_test", "test")]:
        mask = merged[split_col].fillna(False).astype(bool)
        if mask.sum() > 0:
            correct = int(np.sum(merged.loc[mask, "_pred_class"].values == merged.loc[mask, "label"].values))
            accs[split_name] = correct / int(mask.sum())
        else:
            accs[split_name] = 0.0

    return session, rn_params, accs

# =============================================================================
# 4. RelNN pyHGT-faithful HGT (PA-path, Dropout + LayerNorm matching acbull/pyHGT)
# =============================================================================
#
# pyHGT HGTConv.update() does, per target node type:
#   1. GELU(agg)
#   2. a_linears[t](...)  (same as our OutLin_author)
#   3. Dropout(0.2)
#   4. alpha * trans_out + (1-alpha) * node_inp   (learnable skip gate)
#   5. LayerNorm(hidden)  (when use_norm=True, which is the default)
#
# The existing RELNN_DEFINE_DSL (PyG-matching) omits Dropout and LayerNorm.
# This DSL adds them to faithfully replicate pyHGT's architecture.

RELNN_PYHGT_DEFINE_DSL = f"""
#lang:relnn
hidden = {hidden} .
dh = {hidden // num_heads} .

AuthorProj = Linear({info['author_features']}, hidden) .
PaperProj = Linear({info['paper_features']}, hidden) .

AuthorEmb(author_id; ReLU(AuthorProj(z))) :- Author(author_id; z) .
PaperEmb(paper_id; ReLU(PaperProj(z))) :- Paper(paper_id; z) .

K_paper<head> = Linear(hidden, dh) .
Q_author<head> = Linear(hidden, dh) .
V_paper<head> = Linear(hidden, dh) .

PaperK<head>(paper_id; K_paper<head>(z)) :- PaperEmb(paper_id; z) .
AuthorQ<head>(author_id; Q_author<head>(z)) :- AuthorEmb(author_id; z) .
PaperV<head>(paper_id; V_paper<head>(z)) :- PaperEmb(paper_id; z) .

Krel_PA<head> = Linear(dh, dh, False) .
Vrel_PA<head> = Linear(dh, dh, False) .
Prel_PA<head> = Tensor(1, 1.0) .

DotPA<head>(paper_id, author_id; view(1)(view(1, dh)(z_q) @ transpose(Krel_PA<head>(z_k))) * Prel_PA<head> / sqrt(dh)) :-
    PaperK<head>(paper_id; z_k), PaperAuthor(paper_id, author_id; w), AuthorQ<head>(author_id; z_q) .

MaxPA<head>(author_id; max(z)) :- DotPA<head>(paper_id, author_id; z) .
StableDotPA<head>(paper_id, author_id; z1 - z2) :- DotPA<head>(paper_id, author_id; z1), MaxPA<head>(author_id; z2) .
ExpPA<head>(paper_id, author_id; exp(z)) :- StableDotPA<head>(paper_id, author_id; z) .
DenomPA<head>(author_id; sum(z)) :- ExpPA<head>(paper_id, author_id; z) .
SoftPA<head>(paper_id, author_id; z1 / z2) :- ExpPA<head>(paper_id, author_id; z1), DenomPA<head>(author_id; z2) .

MsgPA<head>(paper_id, author_id; Vrel_PA<head>(z_v) * z_att) :- PaperV<head>(paper_id; z_v), PaperAuthor(paper_id, author_id; w), SoftPA<head>(paper_id, author_id; z_att) .
MsgPACon(paper_id, author_id; Concat(z1, z2)) :- MsgPA<1>(paper_id, author_id; z1), MsgPA<2>(paper_id, author_id; z2) .

AggAuthor(author_id; sum(z)) :- MsgPACon(paper_id, author_id; z) .

OutLin_author = Linear(hidden, hidden) .
Skip_author = Tensor(1, 1.0) .
Drop_author = Dropout(0.2) .
Norm_author = LayerNorm(hidden) .

AutLinOut(author_id; Drop_author(OutLin_author(GELU(z)))) :- AggAuthor(author_id; z) .
AuthorOut(author_id; Norm_author(Sigmoid(Skip_author) * z1 + (1 - Sigmoid(Skip_author)) * z2)) :- AutLinOut(author_id; z1), AuthorEmb(author_id; z2) .

Classifier = Linear(hidden, {n_classes}) .
Output(author_id; z) :- AuthorOut(author_id; z) .
"""

def run_relnn_pyhgt_hgt(seed=42, epochs=100, lr=0.005, wd=0.001, num_layers=1):
    """RelNN implementation faithful to acbull/pyHGT (Dropout + LayerNorm in update step).

    Only num_layers=1 is supported; the pyHGT-faithful DSL targets the 1L case
    to match pyHGT's reported 79.5% test accuracy on DBLP.
    """
    if num_layers != 1:
        raise ValueError(f"RELNN_PYHGT_DEFINE_DSL only supports num_layers=1; got {num_layers}")
    full_seed(seed)
    session = Session(db=relnn_db, device=DEVICE)
    session.run(RELNN_PYHGT_DEFINE_DSL)

    full_seed(seed)
    session.run(f"""
#lang:relnn
?fit <epochs={epochs}, lr={lr}, weight_decay={wd}>
Loss(; CrossEntropyLoss()(Classifier(z_pred), z)) :- Output(author_id; z_pred), AuthorLabels(author_id; z) .
""")

    rn_params = sum(p.numel() for p in session.engine.parameter_store.values())

    pred = session.run("""
#lang:relnn
?pred AuthorPred(author_id; ArgMax()(Classifier(z))) :- Output(author_id; z) .
""")

    pred_df = pred.content.copy()
    pred_class = pred.embeddings[0].cpu().numpy().flatten().astype(int)
    pred_df["_pred_class"] = pred_class
    merge_col = "author_id" if "author_id" in pred_df.columns else pred_df.columns[0]
    merged = pred_df.merge(dblp.node_metadata, left_on=merge_col, right_on="node_id", how="left")

    accs = {}
    for split_col, split_name in [("is_train", "train"), ("is_val", "val"), ("is_test", "test")]:
        mask = merged[split_col].fillna(False).astype(bool)
        if mask.sum() > 0:
            correct = int(np.sum(merged.loc[mask, "_pred_class"].values == merged.loc[mask, "label"].values))
            accs[split_name] = correct / int(mask.sum())
        else:
            accs[split_name] = 0.0

    return session, rn_params, accs

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    rows = []

    # FULL graph 1-layer
    print()
    print(DIVIDER)
    print("SCOPE=FULL_GRAPH_1L")
    print(DIVIDER)
    _sync_cuda(); t0 = time.perf_counter()
    _, p_orig_1, _, a_orig_1 = run_original_hgt(epochs=100, num_layers=1)
    _sync_cuda(); t_orig_1 = time.perf_counter() - t0
    rows.append(("FULL_GRAPH_1L", "original_pyHGT", p_orig_1, a_orig_1, t_orig_1))
    _sync_cuda(); t0 = time.perf_counter()
    _, p_pyg_1, _, a_pyg_1 = run_pyg_hgt(epochs=100, num_layers=1)
    _sync_cuda(); t_pyg_1 = time.perf_counter() - t0
    rows.append(("FULL_GRAPH_1L", "pyg_hgtconv", p_pyg_1, a_pyg_1, t_pyg_1))

    # FULL graph 2-layer
    print()
    print(DIVIDER)
    print("SCOPE=FULL_GRAPH_2L")
    print(DIVIDER)
    _sync_cuda(); t0 = time.perf_counter()
    _, p_orig_2, _, a_orig_2 = run_original_hgt(epochs=100, num_layers=2)
    _sync_cuda(); t_orig_2 = time.perf_counter() - t0
    rows.append(("FULL_GRAPH_2L", "original_pyHGT", p_orig_2, a_orig_2, t_orig_2))
    _sync_cuda(); t0 = time.perf_counter()
    _, p_pyg_2, _, a_pyg_2 = run_pyg_hgt(epochs=100, num_layers=2)
    _sync_cuda(); t_pyg_2 = time.perf_counter() - t0
    rows.append(("FULL_GRAPH_2L", "pyg_hgtconv", p_pyg_2, a_pyg_2, t_pyg_2))

    # PA-path context rows
    print()
    print(DIVIDER)
    print("SCOPE=PA_PATH_1L (context row)")
    print(DIVIDER)
    _sync_cuda(); t0 = time.perf_counter()
    _, p_rn, a_rn = run_relnn_hgt(epochs=100, num_layers=1)
    _sync_cuda(); t_rn = time.perf_counter() - t0
    rows.append(("PA_PATH_1L", "relnn", p_rn, a_rn, t_rn))

    print()
    print(DIVIDER)
    print("SCOPE=PA_PATH_2L (context row)")
    print(DIVIDER)
    _sync_cuda(); t0 = time.perf_counter()
    _, p_rn2, a_rn2 = run_relnn_hgt(epochs=100, num_layers=2)
    _sync_cuda(); t_rn2 = time.perf_counter() - t0
    rows.append(("PA_PATH_2L", "relnn", p_rn2, a_rn2, t_rn2))

    # pyHGT-faithful RelNN (Dropout + LayerNorm, targets pyHGT's 79.5% accuracy)
    print()
    print(DIVIDER)
    print("SCOPE=PA_PATH_PYHGT_1L (pyHGT-faithful: Dropout + LayerNorm)")
    print(DIVIDER)
    _sync_cuda(); t0 = time.perf_counter()
    _, p_rn_pyhgt, a_rn_pyhgt = run_relnn_pyhgt_hgt(epochs=100, num_layers=1)
    _sync_cuda(); t_rn_pyhgt = time.perf_counter() - t0
    rows.append(("PA_PATH_PYHGT_1L", "relnn_pyhgt", p_rn_pyhgt, a_rn_pyhgt, t_rn_pyhgt))

    print()
    print(DIVIDER)
    print("SUMMARY")
    print(DIVIDER)
    print(f"{'Scope':<16} {'Implementation':<18} {'#Params':>10} {'Train':>8} {'Val':>8} {'Test':>8} {'Time':>9}")
    print("-" * 90)
    for scope, impl, params, accs, t in rows:
        print(f"{scope:<16} {impl:<18} {params:>10,} {accs['train']:>7.1%} {accs['val']:>7.1%} {accs['test']:>7.1%} {t:>8.1f}s")
    print()
    print("Done.")
