"""Compare PyG HGTConv vs hand-rolled PyTorch vs RelNN HGT on DBLP.

DBLP is a heterogeneous graph:
  Node types: author (4057, d=334), paper (14328, d=4231), term (7723, d=50), conference (20, d=1)
  Edge types: 6 directed relations
  Task: author classification (4 classes), 400 train / 400 val / 3257 test

Architecture (matches PyG HGTConv for the Paper->Author path):
  - Per-node-type linear projection to hidden dim + ReLU
  - Per-head K, Q, V projections from source/dest node embeddings
  - k_rel, v_rel (Linear(dh,dh,False)) per edge type per head
  - Dot-product attention with learnable p_rel scaling + sqrt(dh) normalization
  - EdgeSoftmax (compositional: exp -> sum -> divide)
  - Message = v_rel(V_src) * softmax_attention, aggregated by sum
  - Output = out_lin(GELU(agg)) with learnable skip gate to input embedding
  - Classifier on author nodes

Three comparison levels:
  1. PyG HGTConv reference -- train and report accuracy
  2. Hand-rolled PyTorch -- identical math to RelNN, same params
  3. Weight-synced forward comparison (hand-rolled <-> RelNN) proving equivalence

Run from repo root:
    python tests/slow/run_compare_dblp_hgt.py
"""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_pyhgt_path = Path(__file__).resolve().parents[2] / "_external" / "pyHGT" / "OAG"
sys.path.insert(0, str(_pyhgt_path))

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear as PyGLinear

from relann.torch_utils import full_seed, get_project_root
from relann.session import Session
from relann.datasets import load_dblp_dataset
from relann.comparison import SessionComparison, EvalResult
from pyHGT.conv import HGTConv as OriginalHGTConv

DIVIDER = "=" * 70

project_root = get_project_root()
dblp_path = project_root / "data" / "DBLP"
if not dblp_path.exists():
    print("SKIP: DBLP data not found at", dblp_path)
    sys.exit(0)

# -- Load DBLP ---------------------------------------------------------------

dblp = load_dblp_dataset()
print(dblp)
print()

pyg_data = dblp.pyg_data
relnn_db = dblp.db
info = dblp.dataset_info

hidden = 64
num_heads = 2
dh = hidden // num_heads  # 32
n_classes = info["num_classes"]  # 4

# Extract edge indices for Paper->Author (the only edge type affecting Author output with 1 layer)
pa_df = relnn_db["PaperAuthor"][0]
pa_edge_src = torch.tensor(pa_df["paper_id"].values, dtype=torch.long)
pa_edge_dst = torch.tensor(pa_df["author_id"].values, dtype=torch.long)

# Node features
x_author = pyg_data["author"].x  # (4057, 334)
x_paper = pyg_data["paper"].x    # (14328, 4231)
n_authors = x_author.size(0)
n_papers = x_paper.size(0)

# Labels and masks
y_author = pyg_data["author"].y
train_mask = pyg_data["author"].train_mask
val_mask = pyg_data["author"].val_mask
test_mask = pyg_data["author"].test_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using {DEVICE}")

# Device copies for torch baselines
x_author_dev = x_author.to(DEVICE)
x_paper_dev = x_paper.to(DEVICE)
pa_edge_src_dev = pa_edge_src.to(DEVICE)
pa_edge_dst_dev = pa_edge_dst.to(DEVICE)
y_author_dev = y_author.to(DEVICE)
train_mask_dev = train_mask.to(DEVICE)
val_mask_dev = val_mask.to(DEVICE)
test_mask_dev = test_mask.to(DEVICE)

# Flat heterogeneous representation (for pyHGT subset-mapped parity source)
NODE_TYPE_ORDER = ["author", "paper", "term", "conference"]
node_type_to_id = {nt: i for i, nt in enumerate(NODE_TYPE_ORDER)}
node_features = {nt: pyg_data[nt].x for nt in NODE_TYPE_ORDER}
node_offsets = {}
_offset = 0
for _nt in NODE_TYPE_ORDER:
    node_offsets[_nt] = _offset
    _offset += node_features[_nt].size(0)
n_total_nodes = _offset
max_feat_dim = max(x.size(1) for x in node_features.values())
node_inp_flat = torch.zeros(n_total_nodes, max_feat_dim)
node_type_flat = torch.zeros(n_total_nodes, dtype=torch.long)
for _nt in NODE_TYPE_ORDER:
    _x = node_features[_nt]
    _o = node_offsets[_nt]
    node_inp_flat[_o:_o + _x.size(0), :_x.size(1)] = _x
    node_type_flat[_o:_o + _x.size(0)] = node_type_to_id[_nt]
all_edge_src, all_edge_dst, all_edge_types = [], [], []
edge_type_counter = 0
for edge_type_key, edge_index in pyg_data.edge_index_dict.items():
    src_type, _, dst_type = edge_type_key
    src_offset = node_offsets[src_type]
    dst_offset = node_offsets[dst_type]
    all_edge_src.append(edge_index[0] + src_offset)
    all_edge_dst.append(edge_index[1] + dst_offset)
    all_edge_types.append(torch.full((edge_index.size(1),), edge_type_counter, dtype=torch.long))
    edge_type_counter += 1
edge_index_flat = torch.stack([torch.cat(all_edge_src), torch.cat(all_edge_dst)]).to(DEVICE)
edge_type_flat = torch.cat(all_edge_types).to(DEVICE)
edge_time_flat = torch.zeros(edge_index_flat.size(1), dtype=torch.long, device=DEVICE)
node_inp_flat = node_inp_flat.to(DEVICE)
node_type_flat = node_type_flat.to(DEVICE)
author_offset = node_offsets["author"]

# =============================================================================
# 1. PyG HGT reference
# =============================================================================

class PyGHGT(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()
        self.lin_dict = nn.ModuleDict()
        for node_type in pyg_data.node_types:
            self.lin_dict[node_type] = PyGLinear(-1, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels,
                           pyg_data.metadata(), num_heads)
            self.convs.append(conv)

        self.lin = PyGLinear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            nt: self.lin_dict[nt](x).relu_()
            for nt, x in x_dict.items()
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        return self.lin(x_dict["author"])

def run_pyg_hgt(seed=42, epochs=100, lr=0.005, wd=0.001):
    full_seed(seed)
    model = PyGHGT(hidden, n_classes, num_heads, 1).to(DEVICE)

    with torch.no_grad():
        model(
            {k: v.to(DEVICE) for k, v in pyg_data.x_dict.items()},
            {k: v.to(DEVICE) for k, v in pyg_data.edge_index_dict.items()},
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n_params = sum(p.numel() for p in model.parameters())

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(
            {k: v.to(DEVICE) for k, v in pyg_data.x_dict.items()},
            {k: v.to(DEVICE) for k, v in pyg_data.edge_index_dict.items()},
        )
        loss = F.cross_entropy(out[train_mask_dev], y_author_dev[train_mask_dev])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(
            {k: v.to(DEVICE) for k, v in pyg_data.x_dict.items()},
            {k: v.to(DEVICE) for k, v in pyg_data.edge_index_dict.items()},
        ).argmax(dim=-1)
    accs = {}
    for split, mask in [("train", train_mask_dev), ("val", val_mask_dev), ("test", test_mask_dev)]:
        accs[split] = (pred[mask] == y_author_dev[mask]).float().mean().item()

    return model, n_params, losses, accs

# =============================================================================
# 2. Hand-rolled PyTorch HGT (Paper->Author path only, matching RelNN DSL)
# =============================================================================

class HandRolledHGT(nn.Module):
    """Hand-rolled HGT matching PyG HGTConv math for the Paper->Author path.

    For 1-layer author classification, only the Paper->Author edge type
    affects the author output. This model implements exactly that path:
      K from Paper, Q from Author, V from Paper,
      k_rel/v_rel per head, dot-product attention with p_rel + sqrt(dh),
      scatter_softmax, message aggregation, out_lin(GELU) + skip.
    """

    def __init__(self):
        super().__init__()
        self.author_proj = nn.Linear(info["author_features"], hidden)
        self.paper_proj = nn.Linear(info["paper_features"], hidden)

        self.K_paper = nn.ModuleList([nn.Linear(hidden, dh) for _ in range(num_heads)])
        self.Q_author = nn.ModuleList([nn.Linear(hidden, dh) for _ in range(num_heads)])
        self.V_paper = nn.ModuleList([nn.Linear(hidden, dh) for _ in range(num_heads)])

        self.Krel_PA = nn.ModuleList([nn.Linear(dh, dh, bias=False) for _ in range(num_heads)])
        self.Vrel_PA = nn.ModuleList([nn.Linear(dh, dh, bias=False) for _ in range(num_heads)])
        self.Prel_PA = nn.ParameterList([nn.Parameter(torch.empty(1)) for _ in range(num_heads)])

        self.out_lin_author = nn.Linear(hidden, hidden)
        self.skip_author = nn.Parameter(torch.empty(1))

        self.classifier = nn.Linear(hidden, n_classes)
        self._reset_prel_skip()

    def _reset_prel_skip(self):
        for p in self.Prel_PA:
            nn.init.ones_(p)
        nn.init.ones_(self.skip_author)

    def forward(self, x_a, x_p, edge_src, edge_dst):
        """Forward pass.

        Args:
            x_a: Author features (N_a, 334)
            x_p: Paper features (N_p, 4231)
            edge_src: Paper node indices for PA edges (E,)
            edge_dst: Author node indices for PA edges (E,)

        Returns:
            author_emb: Author output embeddings (N_a, 64) -- before classifier
            logits: Classification logits (N_a, 4)
        """
        h_a = F.relu(self.author_proj(x_a))  # (N_a, 64)
        h_p = F.relu(self.paper_proj(x_p))   # (N_p, 64)

        msg_heads = []
        for head_idx in range(num_heads):
            k_src = self.K_paper[head_idx](h_p[edge_src])         # (E, dh)
            q_dst = self.Q_author[head_idx](h_a[edge_dst])        # (E, dh)
            v_src = self.V_paper[head_idx](h_p[edge_src])         # (E, dh)

            k_transformed = self.Krel_PA[head_idx](k_src)         # (E, dh)
            v_transformed = self.Vrel_PA[head_idx](v_src)         # (E, dh)

            alpha = (q_dst * k_transformed).sum(dim=-1)           # (E,)
            alpha = alpha * self.Prel_PA[head_idx]                # (E,)
            alpha = alpha / math.sqrt(dh)                         # (E,)

            # Softmax grouped by destination (author)
            alpha_max = torch.zeros(n_authors, device=alpha.device)
            alpha_max.scatter_reduce_(0, edge_dst, alpha, reduce="amax", include_self=True)
            alpha = alpha - alpha_max[edge_dst]
            alpha_exp = torch.exp(alpha)
            alpha_sum = torch.zeros(n_authors, device=alpha.device)
            alpha_sum.scatter_add_(0, edge_dst, alpha_exp)
            alpha_softmax = alpha_exp / alpha_sum[edge_dst].clamp(min=1e-12)

            msg = v_transformed * alpha_softmax.unsqueeze(-1)     # (E, dh)
            msg_heads.append(msg)

        msg_full = torch.cat(msg_heads, dim=-1)                   # (E, hidden)
        agg = torch.zeros(n_authors, hidden, device=msg_full.device)
        agg.scatter_add_(0, edge_dst.unsqueeze(-1).expand_as(msg_full), msg_full)

        out = self.out_lin_author(F.gelu(agg))
        skip = torch.sigmoid(self.skip_author)
        author_out = skip * out + (1 - skip) * h_a

        return author_out, self.classifier(author_out)

def run_hand_rolled(seed=42, epochs=100, lr=0.005, wd=0.001, model=None):
    if model is None:
        full_seed(seed)
        model = HandRolledHGT().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n_params = sum(p.numel() for p in model.parameters())

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        _, logits = model(x_author_dev, x_paper_dev, pa_edge_src_dev, pa_edge_dst_dev)
        loss = F.cross_entropy(logits[train_mask_dev], y_author_dev[train_mask_dev])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        _, logits = model(x_author_dev, x_paper_dev, pa_edge_src_dev, pa_edge_dst_dev)
        pred = logits.argmax(dim=-1)
    accs = {}
    for split, mask in [("train", train_mask_dev), ("val", val_mask_dev), ("test", test_mask_dev)]:
        accs[split] = (pred[mask] == y_author_dev[mask]).float().mean().item()

    return model, n_params, losses, accs

# =============================================================================
# 3. RelNN templated HGT (Paper->Author path, EdgeSoftmax, k_rel/v_rel, skip)
# =============================================================================

RELNN_DEFINE_DSL = f"""
#lang:relnn
hidden = {hidden} .
dh = {dh} .

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

RELNN_FIT_DSL = """
#lang:relnn
?fit <epochs={epochs}, lr={lr}, weight_decay={wd}>
Loss(; CrossEntropyLoss()(Classifier(z_pred), z)) :- Output(author_id; z_pred), AuthorLabels(author_id; z) .
"""

RELNN_PRED_DSL = """
#lang:relnn
?pred AuthorPred(author_id; ArgMax()(Classifier(z))) :- Output(author_id; z) .
"""

RELNN_FORWARD_DSL = """
#lang:relnn
?pred AuthorFwd(author_id; z) :- Output(author_id; z) .
"""

RELNN_LOGITS_DSL = """
#lang:relnn
?pred AuthorLogits(author_id; Classifier(z)) :- Output(author_id; z) .
"""

def _extract_fit_loss_history(session) -> list[float]:
    """Per-epoch training loss from the last ?fit (Engine stores loss_history)."""
    for _name, info in session.engine.trained_modules.items():
        hist = info.get("loss_history")
        if hist:
            return [float(x) for x in hist]
    return []

def run_relnn_hgt(seed=42, epochs=100, lr=0.005, wd=0.001, session=None):
    if session is None:
        full_seed(seed)
        session = Session(db=relnn_db)
        session.run(RELNN_DEFINE_DSL)
    full_seed(seed)
    session.run(RELNN_FIT_DSL.format(epochs=epochs, lr=lr, wd=wd))

    rn_params = sum(p.numel() for p in session.engine.parameter_store.values())
    rn_losses = _extract_fit_loss_history(session)

    pred = session.run(RELNN_PRED_DSL)
    return session, rn_params, pred, rn_losses

def export_dblp_hgt_artifacts(
    out_dir: Path,
    *,
    hr_losses: list[float],
    rn_losses: list[float],
    pyg_losses: list[float],
    metrics: dict,
) -> None:
    """Save loss curves (npz + json) and a PDF figure for the paper."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "dblp_hgt_losses.npz",
        hand_rolled=np.array(hr_losses, dtype=np.float64),
        relnn=np.array(rn_losses, dtype=np.float64),
        pyg=np.array(pyg_losses, dtype=np.float64),
    )
    with open(out_dir / "dblp_hgt_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Epoch-aligned CSV for pgfplots / reproducibility (no matplotlib required)
    max_e = max(len(hr_losses), len(rn_losses), len(pyg_losses))
    csv_path = out_dir / "dblp_hgt_loss.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,hand_rolled,relnn,pyg\n")
        for i in range(max_e):
            e = i + 1
            hr = hr_losses[i] if i < len(hr_losses) else ""
            rn = rn_losses[i] if i < len(rn_losses) else ""
            pg = pyg_losses[i] if i < len(pyg_losses) else ""
            f.write(f"{e},{hr},{rn},{pg}\n")
    print(f"[export] Wrote {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[export] matplotlib not installed; skipped PDF figure")
        return

    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    if hr_losses:
        ax.plot(range(1, len(hr_losses) + 1), hr_losses, label="Hand-rolled PyTorch", color="#1f77b4")
    if rn_losses:
        ax.plot(
            range(1, len(rn_losses) + 1),
            rn_losses,
            label="RelNN",
            color="#ff7f0e",
            linestyle="--",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="-", alpha=0.25)
    fig.tight_layout()
    pdf_path = out_dir / "exp_dblp_hgt_loss.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[export] Wrote {pdf_path} and metrics under {out_dir}")

def evaluate_dblp_relnn(pred, node_metadata):
    """Evaluate DBLP RelNN predictions against ground truth."""
    pred_df = pred.content.copy()
    pred_class = pred.embeddings[0].cpu().numpy().flatten().astype(int)
    pred_df["_pred_class"] = pred_class

    merge_col = "author_id" if "author_id" in pred_df.columns else pred_df.columns[0]
    merged = pred_df.merge(node_metadata, left_on=merge_col, right_on="node_id", how="left")

    accs = {}
    for split_col, split_name in [("is_train", "train"), ("is_val", "val"), ("is_test", "test")]:
        mask = merged[split_col].fillna(False).astype(bool)
        if mask.sum() > 0:
            correct = int(np.sum(merged.loc[mask, "_pred_class"].values == merged.loc[mask, "label"].values))
            accs[split_name] = correct / int(mask.sum())
        else:
            accs[split_name] = 0.0
    return accs

# =============================================================================
# 4. Weight mapping (Hand-rolled PyTorch -> RelNN)
# =============================================================================

def _build_weight_mapping(relnn_store):
    """Build {pytorch_param_name: relnn_fqn} for the PA-path HGT.

    The Krel_PA weights end up under the DotPA transformation's internal
    compiler path (deeply nested in the tensor term).  We discover them
    by scanning the parameter store for the matching shape.
    """
    m = {}
    m["author_proj.weight"] = "global.AuthorProj.weight"
    m["author_proj.bias"] = "global.AuthorProj.bias"
    m["paper_proj.weight"] = "global.PaperProj.weight"
    m["paper_proj.bias"] = "global.PaperProj.bias"

    for h in range(num_heads):
        rh = h + 1
        m[f"K_paper.{h}.weight"] = f"global.K_paper<{rh}>.weight"
        m[f"K_paper.{h}.bias"] = f"global.K_paper<{rh}>.bias"
        m[f"Q_author.{h}.weight"] = f"global.Q_author<{rh}>.weight"
        m[f"Q_author.{h}.bias"] = f"global.Q_author<{rh}>.bias"
        m[f"V_paper.{h}.weight"] = f"global.V_paper<{rh}>.weight"
        m[f"V_paper.{h}.bias"] = f"global.V_paper<{rh}>.bias"

        m[f"Vrel_PA.{h}.weight"] = f"global.Vrel_PA<{rh}>.weight"
        m[f"Prel_PA.{h}"] = f"global.Prel_PA<{rh}>.weight"

        # Krel_PA<head> is stored under the DotPA transformation's compiled path
        krel_key = _find_krel_fqn(relnn_store, rh)
        m[f"Krel_PA.{h}.weight"] = krel_key

    m["out_lin_author.weight"] = "global.OutLin_author.weight"
    m["out_lin_author.bias"] = "global.OutLin_author.bias"
    m["skip_author"] = "global.Skip_author.weight"
    return m

def _find_krel_fqn(store, head_idx):
    """Find the FQN for Krel_PA<head_idx> in the parameter store.

    The Krel_PA weight (32x32, no bias) is nested inside the DotPA tensor
    term compilation, so its FQN contains the DotPA transformation prefix.
    """
    prefix = f"global.transformation_DotPA<{head_idx}>."
    for k, v in store.items():
        if k.startswith(prefix) and v.shape == (dh, dh):
            return k
    raise KeyError(f"Cannot find Krel_PA<{head_idx}> weight in parameter store")

def _align_relnn_output(relnn_pred, ref_out):
    """Align RelNN output rows to dense [0..N-1] order matching PyTorch."""
    rn_out = relnn_pred.embeddings[0]
    rn_df = relnn_pred.content
    col = "author_id" if "author_id" in rn_df.columns else rn_df.columns[0]
    rn_ids = rn_df[col].values

    aligned = torch.zeros_like(ref_out)
    for pos, nid in enumerate(rn_ids):
        aligned[int(nid)] = rn_out[pos]
    return aligned

class OriginalPaperHGTSubset(nn.Module):
    """One-layer full-graph pyHGT wrapper used for subset-mapped parity."""
    def __init__(self):
        super().__init__()
        self.adapt_ws = nn.ModuleDict()
        for nt in NODE_TYPE_ORDER:
            self.adapt_ws[nt] = nn.Linear(node_features[nt].size(1), hidden)
        self.conv = OriginalHGTConv(
            in_dim=hidden,
            out_dim=hidden,
            num_types=len(NODE_TYPE_ORDER),
            num_relations=len(pyg_data.edge_index_dict),
            n_heads=num_heads,
            dropout=0.2,
            use_norm=True,
            use_RTE=False,
        )
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self):
        res = torch.zeros(n_total_nodes, hidden, device=DEVICE)
        for nt in NODE_TYPE_ORDER:
            idx = node_type_flat == node_type_to_id[nt]
            res[idx] = F.relu(self.adapt_ws[nt](node_features[nt].to(DEVICE)))
        out = self.conv(res, node_type_flat, edge_index_flat, edge_type_flat, edge_time_flat)
        author_out = out[author_offset:author_offset + n_authors]
        return self.classifier(author_out)

def _copy_(store, key, tensor):
    store[key].data.copy_(tensor.detach().to(store[key].device))

def _sync_pyg_to_relnn_subset(pyg_model, session):
    store = session.engine.parameter_store
    _copy_(store, "global.AuthorProj.weight", pyg_model.lin_dict["author"].weight)
    _copy_(store, "global.AuthorProj.bias", pyg_model.lin_dict["author"].bias)
    _copy_(store, "global.PaperProj.weight", pyg_model.lin_dict["paper"].weight)
    _copy_(store, "global.PaperProj.bias", pyg_model.lin_dict["paper"].bias)

    # PyG packs K/Q/V as [K(64), Q(64), V(64)] per node type
    w_paper = pyg_model.convs[0].kqv_lin.lins["paper"].weight
    b_paper = pyg_model.convs[0].kqv_lin.lins["paper"].bias
    w_author = pyg_model.convs[0].kqv_lin.lins["author"].weight
    b_author = pyg_model.convs[0].kqv_lin.lins["author"].bias
    k_paper_w, v_paper_w = w_paper[:hidden], w_paper[2 * hidden:3 * hidden]
    k_paper_b, v_paper_b = b_paper[:hidden], b_paper[2 * hidden:3 * hidden]
    q_author_w, q_author_b = w_author[hidden:2 * hidden], b_author[hidden:2 * hidden]

    pa_rel_idx = 1  # ('paper','to','author')
    for h in range(num_heads):
        rh = h + 1
        lo, hi = h * dh, (h + 1) * dh
        _copy_(store, f"global.K_paper<{rh}>.weight", k_paper_w[lo:hi])
        _copy_(store, f"global.K_paper<{rh}>.bias", k_paper_b[lo:hi])
        _copy_(store, f"global.Q_author<{rh}>.weight", q_author_w[lo:hi])
        _copy_(store, f"global.Q_author<{rh}>.bias", q_author_b[lo:hi])
        _copy_(store, f"global.V_paper<{rh}>.weight", v_paper_w[lo:hi])
        _copy_(store, f"global.V_paper<{rh}>.bias", v_paper_b[lo:hi])
        rel_head_idx = pa_rel_idx * num_heads + h
        _copy_(store, _find_krel_fqn(store, rh), pyg_model.convs[0].k_rel.weight[rel_head_idx])
        _copy_(store, f"global.Vrel_PA<{rh}>.weight", pyg_model.convs[0].v_rel.weight[rel_head_idx])
        _copy_(store, f"global.Prel_PA<{rh}>.weight", pyg_model.convs[0].p_rel["paper__to__author"][0, h].view(1))

    _copy_(store, "global.OutLin_author.weight", pyg_model.convs[0].out_lin.lins["author"].weight)
    _copy_(store, "global.OutLin_author.bias", pyg_model.convs[0].out_lin.lins["author"].bias)
    _copy_(store, "global.Skip_author.weight", pyg_model.convs[0].skip["author"].view(1))
    _copy_(store, "global.Classifier.weight", pyg_model.lin.weight)
    _copy_(store, "global.Classifier.bias", pyg_model.lin.bias)

def _sync_pyhgt_to_relnn_subset(pyhgt_model, session):
    store = session.engine.parameter_store
    _copy_(store, "global.AuthorProj.weight", pyhgt_model.adapt_ws["author"].weight)
    _copy_(store, "global.AuthorProj.bias", pyhgt_model.adapt_ws["author"].bias)
    _copy_(store, "global.PaperProj.weight", pyhgt_model.adapt_ws["paper"].weight)
    _copy_(store, "global.PaperProj.bias", pyhgt_model.adapt_ws["paper"].bias)

    author_type_idx, paper_type_idx = 0, 1
    pa_rel_idx = 1  # ('paper','to','author')
    for h in range(num_heads):
        rh = h + 1
        lo, hi = h * dh, (h + 1) * dh
        _copy_(store, f"global.K_paper<{rh}>.weight", pyhgt_model.conv.k_linears[paper_type_idx].weight[lo:hi])
        _copy_(store, f"global.K_paper<{rh}>.bias", pyhgt_model.conv.k_linears[paper_type_idx].bias[lo:hi])
        _copy_(store, f"global.Q_author<{rh}>.weight", pyhgt_model.conv.q_linears[author_type_idx].weight[lo:hi])
        _copy_(store, f"global.Q_author<{rh}>.bias", pyhgt_model.conv.q_linears[author_type_idx].bias[lo:hi])
        _copy_(store, f"global.V_paper<{rh}>.weight", pyhgt_model.conv.v_linears[paper_type_idx].weight[lo:hi])
        _copy_(store, f"global.V_paper<{rh}>.bias", pyhgt_model.conv.v_linears[paper_type_idx].bias[lo:hi])
        _copy_(store, _find_krel_fqn(store, rh), pyhgt_model.conv.relation_att[pa_rel_idx, h])
        _copy_(store, f"global.Vrel_PA<{rh}>.weight", pyhgt_model.conv.relation_msg[pa_rel_idx, h])
        _copy_(store, f"global.Prel_PA<{rh}>.weight", pyhgt_model.conv.relation_pri[pa_rel_idx, h].view(1))

    _copy_(store, "global.OutLin_author.weight", pyhgt_model.conv.a_linears[author_type_idx].weight)
    _copy_(store, "global.OutLin_author.bias", pyhgt_model.conv.a_linears[author_type_idx].bias)
    _copy_(store, "global.Skip_author.weight", pyhgt_model.conv.skip[author_type_idx].view(1))
    _copy_(store, "global.Classifier.weight", pyhgt_model.classifier.weight)
    _copy_(store, "global.Classifier.bias", pyhgt_model.classifier.bias)

def _max_diff(a, b):
    d = (a - b).abs()
    return float(d.max().item()), float(d.mean().item())

def run_subset_parity_from_pyg():
    print()
    print(DIVIDER)
    print("SCOPE=PA_PATH_1L  Parity source: PyG -> RelNN (subset-mapped)")
    print(DIVIDER)
    full_seed(123)
    src = PyGHGT(hidden, n_classes, num_heads, 1).to(DEVICE)
    src.eval()
    with torch.no_grad():
        src_logits = src(
            {k: v.to(DEVICE) for k, v in pyg_data.x_dict.items()},
            {k: v.to(DEVICE) for k, v in pyg_data.edge_index_dict.items()},
        ).detach().cpu()

    session = Session(db=relnn_db)
    session.run(RELNN_DEFINE_DSL)
    session.run(RELNN_LOGITS_DSL)
    _sync_pyg_to_relnn_subset(src, session)
    rn = session.run(RELNN_LOGITS_DSL)
    rn_logits = _align_relnn_output(rn, src_logits)
    max_d, mean_d = _max_diff(src_logits, rn_logits)
    print(f"  subset mapping copied, parity max_diff={max_d:.2e}, mean_diff={mean_d:.2e}")
    return max_d

def run_subset_parity_from_pyhgt():
    print()
    print(DIVIDER)
    print("SCOPE=PA_PATH_1L  Parity source: pyHGT -> RelNN (subset-mapped)")
    print(DIVIDER)
    full_seed(123)
    src = OriginalPaperHGTSubset().to(DEVICE)
    src.eval()
    with torch.no_grad():
        src_logits = src().detach().cpu()

    session = Session(db=relnn_db)
    session.run(RELNN_DEFINE_DSL)
    session.run(RELNN_LOGITS_DSL)
    _sync_pyhgt_to_relnn_subset(src, session)
    rn = session.run(RELNN_LOGITS_DSL)
    rn_logits = _align_relnn_output(rn, src_logits)
    max_d, mean_d = _max_diff(src_logits, rn_logits)
    print(f"  subset mapping copied, parity max_diff={max_d:.2e}, mean_diff={mean_d:.2e}")
    return max_d

# =============================================================================
# 5. Weight-synced forward comparison
# =============================================================================

def debug_forward_comparison():
    """Prove architectural equivalence via weight-synced forward pass."""
    print()
    print(DIVIDER)
    print("Weight-synced forward comparison (Hand-rolled <-> RelNN)")
    print(DIVIDER)

    full_seed(99)
    pt_model = HandRolledHGT()
    pt_model.eval()

    full_seed(99)
    session = Session(db=relnn_db)
    session.run(RELNN_DEFINE_DSL)

    # Force parameter compilation by running a forward pass
    session.run(RELNN_FORWARD_DSL)

    cmp = SessionComparison("DBLP-HGT-Forward", verbose=True)
    cmp.set_pyg_model(pt_model)
    cmp.set_relnn_session(session)
    cmp.print_params()

    mapping = _build_weight_mapping(session.engine.parameter_store)
    cmp.set_mapping(mapping)
    cmp.print_mapping()

    sync_result = cmp.sync_weights()
    print(f"  Synced {sync_result.n_synced}, skipped {sync_result.n_skipped}")
    if sync_result.warnings:
        for w in sync_result.warnings:
            print(f"  [WARN] {w}")

    result = cmp.compare_forward(
        pyg_fn=lambda m: m(x_author, x_paper, pa_edge_src, pa_edge_dst)[0],
        relnn_pred_dsl=RELNN_FORWARD_DSL,
        align_fn=_align_relnn_output,
        tolerance=1e-4,
    )

    print(f"  Max diff: {result.max_diff:.2e}")
    if result.passed:
        print("  [OK] Forward outputs match within tolerance")
    else:
        print("  [FAIL] Forward outputs differ beyond tolerance")

    return result

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="Compare PyG / hand-rolled / RelNN HGT on DBLP.")
    _ap.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="If set, write loss npz/json and exp_dblp_hgt_loss.pdf here (e.g. paper repo images/).",
    )
    _args = _ap.parse_args()
    if _args.export_dir is None:
        _env = os.environ.get("RELNN_PAPER_IMAGES")
        if _env:
            _args.export_dir = Path(_env)

    # -- PyG reference (different architecture, runs independently) --
    print(DIVIDER)
    print("1. PyG HGTConv on DBLP")
    print(DIVIDER)

    t0 = time.perf_counter()
    pyg_model, pyg_n_params, pyg_losses, pyg_accs = run_pyg_hgt(epochs=100)
    pyg_time = time.perf_counter() - t0
    print(f"  Params: {pyg_n_params:,}, Time: {pyg_time:.1f}s")
    for split, acc in pyg_accs.items():
        print(f"  {split}: {acc:.1%}")

    # -- Create hand-rolled + RelNN with synced weights, then train both --
    print()
    print(DIVIDER)
    print("2+3. Hand-rolled + RelNN HGT (PA-path, weight-synced init)")
    print(DIVIDER)

    full_seed(42)
    hr_model = HandRolledHGT().to(DEVICE)

    rn_session = Session(db=relnn_db)
    rn_session.run(RELNN_DEFINE_DSL)
    rn_session.run("""
#lang:relnn
?pred _Init(author_id; Classifier(z)) :- Output(author_id; z) .
""")

    mapping = _build_weight_mapping(rn_session.engine.parameter_store)
    mapping["classifier.weight"] = "global.Classifier.weight"
    mapping["classifier.bias"] = "global.Classifier.bias"
    cmp = SessionComparison("DBLP-HGT-Sync", verbose=True)
    cmp.set_pyg_model(hr_model)
    cmp.set_relnn_session(rn_session)
    cmp.set_mapping(mapping)
    sync_result = cmp.sync_weights()
    print(f"  Weight sync: HR -> RelNN ({sync_result.n_synced} synced)")

    print()
    print("  [Hand-rolled PyTorch]")
    full_seed(42)
    t0 = time.perf_counter()
    hr_model, hr_n_params, hr_losses, hr_accs = run_hand_rolled(
        epochs=100, model=hr_model)
    hr_time = time.perf_counter() - t0
    print(f"  Params: {hr_n_params:,}, Time: {hr_time:.1f}s")
    for split, acc in hr_accs.items():
        print(f"  {split}: {acc:.1%}")

    print()
    print("  [RelNN Templated (synced init)]")
    full_seed(42)
    t0 = time.perf_counter()
    rn_session, rn_n_params, rn_pred, rn_losses = run_relnn_hgt(
        epochs=100, session=rn_session)
    rn_time = time.perf_counter() - t0
    print(f"  Params: {rn_n_params:,}, Time: {rn_time:.1f}s")

    rn_accs = evaluate_dblp_relnn(rn_pred, dblp.node_metadata)
    for split, acc in rn_accs.items():
        print(f"  {split}: {acc:.1%}")

    # -- Param count check --
    print()
    print(DIVIDER)
    print("Param count comparison")
    print(DIVIDER)
    print(f"  PyG HGTConv:        {pyg_n_params:>10,}  (includes all 6 edge types)")
    print(f"  Hand-rolled (PA):   {hr_n_params:>10,}")
    print(f"  RelNN (PA):         {rn_n_params:>10,}")
    if hr_n_params == rn_n_params:
        print("  [OK] Hand-rolled and RelNN param counts match")
    else:
        print(f"  [WARN] Param count mismatch: HR={hr_n_params}, RN={rn_n_params}")

    # -- Weight-synced forward comparison --
    fwd_result = debug_forward_comparison()

    # -- Summary --
    print()
    print(DIVIDER)
    print("SUMMARY")
    print(DIVIDER)
    print(f"  PyG HGTConv   test acc: {pyg_accs['test']:.1%}  ({pyg_n_params:,} params)  Time: {pyg_time:.1f}s")
    print(f"  Hand-rolled   test acc: {hr_accs['test']:.1%}  ({hr_n_params:,} params)  Time: {hr_time:.1f}s")
    print(f"  RelNN         test acc: {rn_accs['test']:.1%}  ({rn_n_params:,} params)  Time: {rn_time:.1f}s")
    print(f"  HR-RelNN delta: {abs(hr_accs['test'] - rn_accs['test']):.1%}")
    print(f"  HR-RelNN time ratio: {rn_time/hr_time:.1f}x")
    print(f"  Forward match: {'[OK]' if fwd_result.passed else '[FAIL]'}  max_diff={fwd_result.max_diff:.2e}")
    print()
    print("NOTE: PyG has more params because it defines all 6 edge types.")
    print("The hand-rolled and RelNN models only implement Paper->Author,")
    print("which is the only path affecting Author output with 1 HGT layer.")
    print("Weight-synced init + forward comparison proves full equivalence.")

    # -- Additional subset-mapped parity checks from external baselines --
    pyg_subset_diff = run_subset_parity_from_pyg()
    pyhgt_subset_diff = run_subset_parity_from_pyhgt()
    print()
    print("Subset-mapped parity summary:")
    print(f"  PyG -> RelNN (PA-path):   max_diff={pyg_subset_diff:.2e}")
    print(f"  pyHGT -> RelNN (PA-path): max_diff={pyhgt_subset_diff:.2e}")

    assert fwd_result.passed, f"Weight-synced forward FAILED: max_diff={fwd_result.max_diff:.2e}"
    assert abs(hr_accs['test'] - rn_accs['test']) < 0.02, (
        f"HR-RelNN accuracy gap >2%: HR={hr_accs['test']:.1%}, RN={rn_accs['test']:.1%}")

    if _args.export_dir is not None:
        export_dblp_hgt_artifacts(
            _args.export_dir,
            hr_losses=hr_losses,
            rn_losses=rn_losses,
            pyg_losses=pyg_losses,
            metrics={
                "pyg_params": pyg_n_params,
                "hand_rolled_params": hr_n_params,
                "relnn_params": rn_n_params,
                "pyg_time_s": pyg_time,
                "hand_rolled_time_s": hr_time,
                "relnn_time_s": rn_time,
                "pyg_accs": pyg_accs,
                "hand_rolled_accs": hr_accs,
                "relnn_accs": rn_accs,
                "forward_max_diff": fwd_result.max_diff,
                "forward_passed": fwd_result.passed,
                "epochs": 100,
                "lr": 0.005,
                "weight_decay": 0.001,
                "hidden": hidden,
                "num_heads": num_heads,
                "seed_sync_train": 42,
            },
        )

    print()
    print("Done.")
