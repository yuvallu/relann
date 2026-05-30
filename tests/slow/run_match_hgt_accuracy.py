"""Verify that RelNN PA-path gives identical accuracy to PyG when weights are synced.

Theory (user's claim):
  In 1-layer HGT on DBLP, only Paper->Author edges contribute to author output.
  Therefore, a PA-path-only RelNN with the same weights should give IDENTICAL
  results to a full-graph PyG/pyHGT model, because:
    - Other edge types produce zero gradient for author outputs
    - Softmax denominator for author nodes covers only PA edges (the only incoming type)
    - There is no scope difference at inference for author nodes

This script:
  1. Initializes PyG with a fixed seed
  2. Syncs PyG weights -> hand-rolled PyTorch (PA-path), compares forward pass
  3. Syncs hand-rolled -> RelNN, compares forward pass  (already proven ~0 diff)
  4. Diagnoses step-by-step WHERE any residual diff comes from
  5. Trains all three from identical weights, checks final accuracy
  6. Repeats for 5 seeds to confirm reproducibility

Run from repo root:
    python tests/slow/run_match_hgt_accuracy.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv as PyGHGTConv, Linear as PyGLinear

from relann.torch_utils import full_seed
from relann.session import Session
from relann.datasets import load_dblp_dataset
from relann.comparison import SessionComparison

DIVIDER = "=" * 70

dblp = load_dblp_dataset()
pyg_data = dblp.pyg_data
relnn_db = dblp.db
info = dblp.dataset_info

hidden = 64
num_heads = 2
dh = hidden // num_heads  # 32
n_classes = info["num_classes"]

x_author = pyg_data["author"].x
x_paper = pyg_data["paper"].x
n_authors = x_author.size(0)
n_papers = x_paper.size(0)

y_author = pyg_data["author"].y
train_mask = pyg_data["author"].train_mask
val_mask = pyg_data["author"].val_mask
test_mask = pyg_data["author"].test_mask

pa_df = relnn_db["PaperAuthor"][0]
pa_edge_src = torch.tensor(pa_df["paper_id"].values, dtype=torch.long)
pa_edge_dst = torch.tensor(pa_df["author_id"].values, dtype=torch.long)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using {DEVICE}")

x_author_dev = x_author.to(DEVICE)
x_paper_dev = x_paper.to(DEVICE)
pa_edge_src_dev = pa_edge_src.to(DEVICE)
pa_edge_dst_dev = pa_edge_dst.to(DEVICE)
y_author_dev = y_author.to(DEVICE)
train_mask_dev = train_mask.to(DEVICE)
val_mask_dev = val_mask.to(DEVICE)
test_mask_dev = test_mask.to(DEVICE)

# ============================================================================
# Models
# ============================================================================

class PyGHGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_dict = nn.ModuleDict({
            nt: PyGLinear(-1, hidden) for nt in pyg_data.node_types
        })
        self.conv = PyGHGTConv(hidden, hidden, pyg_data.metadata(), num_heads)
        self.lin = PyGLinear(hidden, n_classes)

    def forward(self):
        x_dict = {nt: self.lin_dict[nt](pyg_data[nt].x.to(DEVICE)).relu_() for nt in pyg_data.node_types}
        edge_dict = {k: v.to(DEVICE) for k, v in pyg_data.edge_index_dict.items()}
        x_dict = self.conv(x_dict, edge_dict)
        return self.lin(x_dict["author"])

class HandRolledHGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.author_proj = nn.Linear(info["author_features"], hidden)
        self.paper_proj = nn.Linear(info["paper_features"], hidden)
        self.K_paper = nn.ModuleList([nn.Linear(hidden, dh) for _ in range(num_heads)])
        self.Q_author = nn.ModuleList([nn.Linear(hidden, dh) for _ in range(num_heads)])
        self.V_paper = nn.ModuleList([nn.Linear(hidden, dh) for _ in range(num_heads)])
        self.Krel_PA = nn.ModuleList([nn.Linear(dh, dh, bias=False) for _ in range(num_heads)])
        self.Vrel_PA = nn.ModuleList([nn.Linear(dh, dh, bias=False) for _ in range(num_heads)])
        self.Prel_PA = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_heads)])
        self.out_lin_author = nn.Linear(hidden, hidden)
        self.skip_author = nn.Parameter(torch.ones(1))
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x_a, x_p, edge_src, edge_dst):
        h_a = F.relu(self.author_proj(x_a))
        h_p = F.relu(self.paper_proj(x_p))

        msg_heads = []
        for i in range(num_heads):
            k_src = self.K_paper[i](h_p[edge_src])          # (E, dh)
            q_dst = self.Q_author[i](h_a[edge_dst])          # (E, dh)
            v_src = self.V_paper[i](h_p[edge_src])           # (E, dh)
            k_t = self.Krel_PA[i](k_src)                     # (E, dh)
            v_t = self.Vrel_PA[i](v_src)                     # (E, dh)
            alpha = (q_dst * k_t).sum(-1) * self.Prel_PA[i] / math.sqrt(dh)
            alpha_max = torch.zeros(n_authors, device=alpha.device)
            alpha_max.scatter_reduce_(0, edge_dst, alpha, reduce="amax", include_self=True)
            alpha = alpha - alpha_max[edge_dst]
            alpha_exp = torch.exp(alpha)
            alpha_sum = torch.zeros(n_authors, device=alpha.device)
            alpha_sum.scatter_add_(0, edge_dst, alpha_exp)
            alpha_soft = alpha_exp / alpha_sum[edge_dst].clamp(min=1e-12)
            msg_heads.append(v_t * alpha_soft.unsqueeze(-1))

        msg = torch.cat(msg_heads, dim=-1)
        agg = torch.zeros(n_authors, hidden, device=msg.device)
        agg.scatter_add_(0, edge_dst.unsqueeze(-1).expand_as(msg), msg)
        out = self.out_lin_author(F.gelu(agg))
        skip = torch.sigmoid(self.skip_author)
        author_out = skip * out + (1 - skip) * h_a
        return self.classifier(author_out)

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

RELNN_LOGITS_DSL = """
#lang:relnn
?pred AuthorLogits(author_id; Classifier(z)) :- Output(author_id; z) .
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

# ============================================================================
# Weight sync utilities
# ============================================================================

def _sync_cuda():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

def _c(dst, src):
    """Copy src tensor into dst in-place."""
    if isinstance(dst, torch.Tensor):
        dst.data.copy_(src.detach().to(dst.device))
    else:
        raise TypeError(type(dst))

def _inspect_pyg_conv(conv):
    """Print shapes of all weight tensors in a PyG HGTConv for diagnostic."""
    print("  [PyG conv internals]")
    for name, param in conv.named_parameters():
        print(f"    {name}: {tuple(param.shape)}")

def sync_pyg_to_handrolled(pyg_model: PyGHGT, hr_model: HandRolledHGT) -> None:
    """Sync all relevant PyG weights -> hand-rolled model."""
    _c(hr_model.author_proj.weight, pyg_model.lin_dict["author"].weight)
    _c(hr_model.author_proj.bias,   pyg_model.lin_dict["author"].bias)
    _c(hr_model.paper_proj.weight,  pyg_model.lin_dict["paper"].weight)
    _c(hr_model.paper_proj.bias,    pyg_model.lin_dict["paper"].bias)

    conv = pyg_model.conv
    # PyG kqv_lin packs [K | Q | V] all in one linear per node type
    w_paper = conv.kqv_lin.lins["paper"].weight  # (3*hidden, hidden)
    b_paper = conv.kqv_lin.lins["paper"].bias    # (3*hidden,)
    w_author = conv.kqv_lin.lins["author"].weight
    b_author = conv.kqv_lin.lins["author"].bias

    k_paper_w = w_paper[:hidden]               # K rows
    k_paper_b = b_paper[:hidden]
    q_author_w = w_author[hidden:2 * hidden]   # Q rows
    q_author_b = b_author[hidden:2 * hidden]
    v_paper_w = w_paper[2 * hidden:]           # V rows
    v_paper_b = b_paper[2 * hidden:]

    # PyG indexes krel/vrel as: head_idx * num_edge_types + edge_type_offset
    # (see _construct_src_node_feat: type_vec = arange(H).view(-1,1) * num_et + edge_type_offset)
    pa_edge_offset = conv.edge_types_map[("paper", "to", "author")]
    num_et = len(conv.edge_types)
    print(f"  [sync] PA edge type offset in conv.edge_types_map: {pa_edge_offset}")
    print(f"  [sync] num_edge_types: {num_et}")

    for h in range(num_heads):
        lo, hi = h * dh, (h + 1) * dh
        _c(hr_model.K_paper[h].weight, k_paper_w[lo:hi])
        _c(hr_model.K_paper[h].bias,   k_paper_b[lo:hi])
        _c(hr_model.Q_author[h].weight, q_author_w[lo:hi])
        _c(hr_model.Q_author[h].bias,   q_author_b[lo:hi])
        _c(hr_model.V_paper[h].weight,  v_paper_w[lo:hi])
        _c(hr_model.V_paper[h].bias,    v_paper_b[lo:hi])

        krel_idx = h * num_et + pa_edge_offset
        krel_w = conv.k_rel.weight[krel_idx]   # (dh, dh): PyG computes k @ krel_w
        vrel_w = conv.v_rel.weight[krel_idx]
        prel_v = _get_prel_value(conv, pa_edge_offset, h)

        print(f"  [sync] head={h+1} krel_idx={krel_idx} krel.shape={krel_w.shape} prel={prel_v.item():.4f}")
        # nn.Linear(bias=False) computes k @ W.T; to match PyG's k @ krel_w, store W = krel_w.T
        _c(hr_model.Krel_PA[h].weight, krel_w.T)
        _c(hr_model.Vrel_PA[h].weight, vrel_w.T)
        _c(hr_model.Prel_PA[h],        prel_v)

    _c(hr_model.out_lin_author.weight, conv.out_lin.lins["author"].weight)
    _c(hr_model.out_lin_author.bias,   conv.out_lin.lins["author"].bias)
    _c(hr_model.skip_author,           conv.skip["author"].view(1))
    _c(hr_model.classifier.weight, pyg_model.lin.weight)
    _c(hr_model.classifier.bias,   pyg_model.lin.bias)

def sync_handrolled_to_relnn(hr_model: HandRolledHGT, session: Session) -> None:
    """Sync hand-rolled weights -> RelNN parameter store."""
    store = session.engine.parameter_store

    def cp(key, tensor):
        store[key].data.copy_(tensor.detach().to(store[key].device))

    cp("global.AuthorProj.weight", hr_model.author_proj.weight)
    cp("global.AuthorProj.bias",   hr_model.author_proj.bias)
    cp("global.PaperProj.weight",  hr_model.paper_proj.weight)
    cp("global.PaperProj.bias",    hr_model.paper_proj.bias)

    for h in range(num_heads):
        rh = h + 1
        cp(f"global.K_paper<{rh}>.weight", hr_model.K_paper[h].weight)
        cp(f"global.K_paper<{rh}>.bias",   hr_model.K_paper[h].bias)
        cp(f"global.Q_author<{rh}>.weight", hr_model.Q_author[h].weight)
        cp(f"global.Q_author<{rh}>.bias",   hr_model.Q_author[h].bias)
        cp(f"global.V_paper<{rh}>.weight",  hr_model.V_paper[h].weight)
        cp(f"global.V_paper<{rh}>.bias",    hr_model.V_paper[h].bias)

        krel_key = _find_krel_fqn(store, rh)
        cp(krel_key,                         hr_model.Krel_PA[h].weight)
        cp(f"global.Vrel_PA<{rh}>.weight",  hr_model.Vrel_PA[h].weight)
        cp(f"global.Prel_PA<{rh}>.weight",  hr_model.Prel_PA[h])

    cp("global.OutLin_author.weight", hr_model.out_lin_author.weight)
    cp("global.OutLin_author.bias",   hr_model.out_lin_author.bias)
    cp("global.Skip_author.weight",   hr_model.skip_author.view(1))
    cp("global.Classifier.weight",    hr_model.classifier.weight)
    cp("global.Classifier.bias",      hr_model.classifier.bias)

def _get_pa_rel_idx(pyg_data_) -> int:
    """Index of ('paper', 'to', 'author') in the ordered edge type list."""
    for idx, et in enumerate(pyg_data_.edge_types):
        if et == ("paper", "to", "author"):
            return idx
    raise KeyError("PA edge type not found")

def _get_krel_weight(conv, rel_head_idx: int) -> torch.Tensor:
    """Extract krel weight for a given (rel, head) flat index."""
    w = conv.k_rel.weight if hasattr(conv.k_rel, "weight") else conv.k_rel
    if w.dim() == 2:
        # Linear weight: shape (num_relations*num_heads*dh, dh) – reshape to 3D
        n = w.size(0) // dh
        w3 = w.view(n, dh, dh)
        return w3[rel_head_idx]
    elif w.dim() == 3:
        return w[rel_head_idx]
    else:
        raise ValueError(f"Unexpected k_rel shape: {w.shape}")

def _get_vrel_weight(conv, rel_head_idx: int) -> torch.Tensor:
    w = conv.v_rel.weight if hasattr(conv.v_rel, "weight") else conv.v_rel
    if w.dim() == 2:
        n = w.size(0) // dh
        w3 = w.view(n, dh, dh)
        return w3[rel_head_idx]
    elif w.dim() == 3:
        return w[rel_head_idx]
    else:
        raise ValueError(f"Unexpected v_rel shape: {w.shape}")

def _get_prel_value(conv, pa_rel_idx: int, head: int) -> torch.Tensor:
    """Scalar p_rel for (PA edge type, head)."""
    key = "paper__to__author"
    p = conv.p_rel[key]
    if p.dim() == 1:
        return p[head:head+1]
    elif p.dim() == 2:
        return p[0, head:head+1]
    else:
        return p[pa_rel_idx, head:head+1]

def _find_krel_fqn(store, head_idx: int) -> str:
    prefix = f"global.transformation_DotPA<{head_idx}>."
    for k, v in store.items():
        if k.startswith(prefix) and v.shape == (dh, dh):
            return k
    raise KeyError(f"Cannot find Krel_PA<{head_idx}> in parameter store")

# ============================================================================
# Forward diagnostic
# ============================================================================

def _align_relnn(relnn_pred, ref_out: torch.Tensor) -> torch.Tensor:
    rn_out = relnn_pred.embeddings[0].cpu()
    rn_df = relnn_pred.content
    col = "author_id" if "author_id" in rn_df.columns else rn_df.columns[0]
    rn_ids = rn_df[col].values
    aligned = torch.zeros_like(ref_out)
    for pos, nid in enumerate(rn_ids):
        aligned[int(nid)] = rn_out[pos]
    return aligned

def _diff(a: torch.Tensor, b: torch.Tensor, label: str) -> float:
    d = (a.float() - b.float()).abs()
    m, mu = float(d.max()), float(d.mean())
    print(f"  [{label}] max={m:.2e}  mean={mu:.2e}")
    return m

def diagnose_pyg_vs_handrolled(pyg_model: PyGHGT, hr_model: HandRolledHGT) -> float:
    """Step-by-step forward comparison to find where PyG and hand-rolled diverge."""
    pyg_model.eval()
    hr_model.eval()

    with torch.no_grad():
        # -- PyG internals ---------------------------------------------------
        x_dict = {nt: pyg_model.lin_dict[nt](pyg_data[nt].x.to(DEVICE)).relu_()
                  for nt in pyg_data.node_types}
        edge_dict = {k: v.to(DEVICE) for k, v in pyg_data.edge_index_dict.items()}

        pyg_h_author = x_dict["author"]   # (N_a, hidden)
        pyg_h_paper  = x_dict["paper"]    # (N_p, hidden)

        conv = pyg_model.conv
        kqv_author = conv.kqv_lin.lins["author"](pyg_h_author)  # (N_a, 3*hidden)
        kqv_paper  = conv.kqv_lin.lins["paper"] (pyg_h_paper)   # (N_p, 3*hidden)

        pyg_K_paper  = kqv_paper [:, :hidden]        # (N_p, hidden)
        pyg_Q_author = kqv_author[:, hidden:2*hidden] # (N_a, hidden)
        pyg_V_paper  = kqv_paper [:, 2*hidden:]       # (N_p, hidden)

        pa_edge_index = pyg_data.edge_index_dict[("paper", "to", "author")].to(DEVICE)
        E_src = pa_edge_index[0]   # paper indices
        E_dst = pa_edge_index[1]   # author indices

        # -- Hand-rolled internals -------------------------------------------
        h_a = F.relu(hr_model.author_proj(x_author_dev))
        h_p = F.relu(hr_model.paper_proj(x_paper_dev))

        print()
        print("  -- Projection outputs --")
        _diff(pyg_h_author, h_a, "author emb")
        _diff(pyg_h_paper,  h_p, "paper emb")

        print("  -- K/Q/V on edges --")
        pa_edge_offset = conv.edge_types_map[("paper", "to", "author")]
        num_et = len(conv.edge_types)
        for i in range(num_heads):
            lo, hi = i * dh, (i + 1) * dh
            krel_idx = i * num_et + pa_edge_offset   # CORRECT: head * num_et + offset

            pyg_k = pyg_K_paper[E_src, lo:hi]   # (E, dh) from kqv slice
            hr_k  = hr_model.K_paper[i](h_p[E_src])
            _diff(pyg_k, hr_k, f"K head{i+1}")

            pyg_q = pyg_Q_author[E_dst, lo:hi]
            hr_q  = hr_model.Q_author[i](h_a[E_dst])
            _diff(pyg_q, hr_q, f"Q head{i+1}")

            pyg_v = pyg_V_paper[E_src, lo:hi]
            hr_v  = hr_model.V_paper[i](h_p[E_src])
            _diff(pyg_v, hr_v, f"V head{i+1}")

            # -- Krel transformation (correct index) -------------------------
            krel_w = conv.k_rel.weight[krel_idx]   # (dh, dh)
            vrel_w = conv.v_rel.weight[krel_idx]

            pyg_k_transformed = pyg_k @ krel_w          # PyG: k @ krel_w (no transpose)
            hr_k_transformed  = hr_model.Krel_PA[i](pyg_k)
            _diff(pyg_k_transformed, hr_k_transformed, f"K_transformed head{i+1}")

            pyg_v_transformed = pyg_v @ vrel_w
            hr_v_transformed  = hr_model.Vrel_PA[i](pyg_v)
            _diff(pyg_v_transformed, hr_v_transformed, f"V_transformed head{i+1}")

            # -- Attention score ---------------------------------------------
            prel = _get_prel_value(conv, pa_edge_offset, i).to(DEVICE)
            pyg_alpha = (pyg_q * pyg_k_transformed).sum(-1) * prel / math.sqrt(dh)
            hr_alpha  = (hr_q * hr_k_transformed).sum(-1) * hr_model.Prel_PA[i] / math.sqrt(dh)
            _diff(pyg_alpha, hr_alpha, f"alpha head{i+1}")

        # -- Full softmax + message + agg comparison (manual) ---------------
        print("  -- Softmax / message / agg --")
        pa_edge_offset = conv.edge_types_map[("paper", "to", "author")]
        num_et = len(conv.edge_types)
        msg_heads_pyg = []
        msg_heads_hr  = []
        for i in range(num_heads):
            lo, hi = i * dh, (i + 1) * dh
            krel_idx = i * num_et + pa_edge_offset   # CORRECT ordering
            krel_w = conv.k_rel.weight[krel_idx]
            vrel_w = conv.v_rel.weight[krel_idx]
            prel   = _get_prel_value(conv, pa_edge_offset, i).to(DEVICE)

            k_src = pyg_K_paper[E_src, lo:hi]
            q_dst = pyg_Q_author[E_dst, lo:hi]
            v_src = pyg_V_paper[E_src, lo:hi]

            k_t = k_src @ krel_w        # PyG convention
            v_t = v_src @ vrel_w
            alpha = (q_dst * k_t).sum(-1) * prel / math.sqrt(dh)

            # PyG-style stable softmax (edge_softmax)
            from torch_geometric.utils import softmax as pyg_softmax
            alpha_soft_pyg = pyg_softmax(alpha, E_dst, num_nodes=n_authors)
            msg_heads_pyg.append(v_t * alpha_soft_pyg.unsqueeze(-1))

            # hand-rolled softmax
            alpha_max = torch.zeros(n_authors, device=alpha.device)
            alpha_max.scatter_reduce_(0, E_dst, alpha, reduce="amax", include_self=True)
            alpha_exp = torch.exp(alpha - alpha_max[E_dst])
            alpha_sum = torch.zeros(n_authors, device=alpha.device)
            alpha_sum.scatter_add_(0, E_dst, alpha_exp)
            alpha_soft_hr = alpha_exp / alpha_sum[E_dst].clamp(min=1e-12)
            msg_heads_hr.append(v_t * alpha_soft_hr.unsqueeze(-1))

            _diff(alpha_soft_pyg, alpha_soft_hr, f"softmax head{i+1}")

        msg_pyg = torch.cat(msg_heads_pyg, dim=-1)   # (E, hidden)
        msg_hr  = torch.cat(msg_heads_hr,  dim=-1)
        _diff(msg_pyg, msg_hr, "messages")

        agg_pyg = torch.zeros(n_authors, hidden, device=DEVICE)
        agg_pyg.scatter_add_(0, E_dst.unsqueeze(-1).expand_as(msg_pyg), msg_pyg)
        agg_hr  = torch.zeros(n_authors, hidden, device=DEVICE)
        agg_hr.scatter_add_(0, E_dst.unsqueeze(-1).expand_as(msg_hr), msg_hr)
        _diff(agg_pyg, agg_hr, "aggregated")

        # out_lin
        out_pyg = conv.out_lin.lins["author"](F.gelu(agg_pyg))
        out_hr  = hr_model.out_lin_author(F.gelu(agg_hr))
        _diff(out_pyg, out_hr, "out_lin")

        # skip + author emb
        skip_v  = torch.sigmoid(conv.skip["author"])
        skip_hr = torch.sigmoid(hr_model.skip_author)
        _diff(skip_v.view(1), skip_hr.view(1), "skip value")

        def _fresh_x_dict():
            return {nt: pyg_model.lin_dict[nt](pyg_data[nt].x.to(DEVICE)).relu_()
                    for nt in pyg_data.node_types}

        # -- Hook into PyG conv to capture actual intermediate tensors --
        _hook_inputs = {}
        def _make_hook(name):
            def hook(module, inp, out):
                _hook_inputs[name] = (inp, out)
            return hook
        h1 = conv.out_lin.lins["author"].register_forward_hook(_make_hook("out_lin_author"))

        # Full graph (fresh x_dict each time to avoid in-place contamination)
        pyg_conv_out    = pyg_model.conv(_fresh_x_dict(), edge_dict)["author"]
        pyg_agg_full    = _hook_inputs["out_lin_author"][0][0].clone()

        # PA-only
        edge_dict_pa_only = {("paper", "to", "author"): edge_dict[("paper", "to", "author")]}
        pyg_conv_out_pa = pyg_model.conv(_fresh_x_dict(), edge_dict_pa_only)["author"]
        pyg_actual_out_lin_input = _hook_inputs["out_lin_author"][0][0].clone()
        h1.remove()

        _diff(pyg_conv_out, pyg_conv_out_pa, "full-graph vs PA-only conv output (author)")
        # Compare actual PyG agg-after-gelu with my manual agg-after-gelu
        _diff(pyg_actual_out_lin_input, F.gelu(agg_pyg), "PyG actual agg-gelu vs manual F.gelu(agg)")
        _diff(pyg_actual_out_lin_input, F.relu(agg_pyg), "PyG actual agg-gelu vs manual F.relu(agg)")
        _diff(pyg_actual_out_lin_input, agg_pyg, "PyG actual agg-gelu vs manual raw agg")
        _diff(pyg_agg_full, pyg_actual_out_lin_input, "full-graph agg vs PA-only agg (sanity)")
        print(f"    first few pyg_actual: {pyg_actual_out_lin_input[0, :5].tolist()}")
        print(f"    first few gelu(agg) : {F.gelu(agg_pyg)[0, :5].tolist()}")
        print(f"    first few relu(agg) : {F.relu(agg_pyg)[0, :5].tolist()}")
        print(f"    first few raw agg   : {agg_pyg[0, :5].tolist()}")

        # Manual: rebuild from scratch using PyG's KQV
        msg_heads_manual = []
        for i in range(num_heads):
            lo, hi = i * dh, (i + 1) * dh
            krel_idx_m = i * num_et + pa_edge_offset
            krel_w = conv.k_rel.weight[krel_idx_m]
            vrel_w = conv.v_rel.weight[krel_idx_m]
            prel   = _get_prel_value(conv, pa_edge_offset, i).to(DEVICE)
            k_src = pyg_K_paper[E_src, lo:hi]
            q_dst = pyg_Q_author[E_dst, lo:hi]
            v_src = pyg_V_paper[E_src, lo:hi]
            k_t = k_src @ krel_w
            v_t = v_src @ vrel_w
            alpha = (q_dst * k_t).sum(-1) * prel / math.sqrt(dh)
            from torch_geometric.utils import softmax as pyg_softmax
            alpha_s = pyg_softmax(alpha, E_dst, num_nodes=n_authors)
            msg_heads_manual.append(v_t * alpha_s.unsqueeze(-1))
        msg_m = torch.cat(msg_heads_manual, dim=-1)
        agg_m = torch.zeros(n_authors, hidden, device=DEVICE)
        agg_m.scatter_add_(0, E_dst.unsqueeze(-1).expand_as(msg_m), msg_m)
        out_m = conv.out_lin.lins["author"](F.gelu(agg_m))
        skip_v = torch.sigmoid(conv.skip["author"])
        author_emb_manual = skip_v * out_m + (1 - skip_v) * pyg_h_author
        _diff(pyg_conv_out, author_emb_manual, "PyG conv output vs manual recompute")

        # Full forward logit comparison
        pyg_logits = pyg_model()
        hr_logits  = hr_model(x_author_dev, x_paper_dev, E_src, E_dst)
        print("  -- Final logits --")
        max_diff = _diff(pyg_logits.cpu(), hr_logits.cpu(), "logits")

    return max_diff

# ============================================================================
# Training + evaluation
# ============================================================================

def _eval_hr(model: HandRolledHGT) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x_author_dev, x_paper_dev, pa_edge_src_dev, pa_edge_dst_dev)
        pred = logits.argmax(-1)
    accs = {}
    for split, mask in [("train", train_mask_dev), ("val", val_mask_dev), ("test", test_mask_dev)]:
        accs[split] = (pred[mask] == y_author_dev[mask]).float().mean().item()
    return accs

def _eval_pyg(model: PyGHGT) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model()
        pred = logits.argmax(-1)
    accs = {}
    for split, mask in [("train", train_mask_dev), ("val", val_mask_dev), ("test", test_mask_dev)]:
        accs[split] = (pred[mask] == y_author_dev[mask]).float().mean().item()
    return accs

def _eval_relnn(session: Session) -> dict[str, float]:
    pred_result = session.run(RELNN_PRED_DSL)
    pred_df = pred_result.content.copy()
    pred_class = pred_result.embeddings[0].cpu().numpy().flatten().astype(int)
    pred_df["_pred"] = pred_class
    col = "author_id" if "author_id" in pred_df.columns else pred_df.columns[0]
    merged = pred_df.merge(dblp.node_metadata, left_on=col, right_on="node_id", how="left")
    accs = {}
    for split_col, name in [("is_train", "train"), ("is_val", "val"), ("is_test", "test")]:
        mask = merged[split_col].fillna(False).astype(bool)
        accs[name] = (int(np.sum(merged.loc[mask, "_pred"].values == merged.loc[mask, "label"].values))
                      / max(1, int(mask.sum())))
    return accs

def train_one(model, forward_fn, epochs=100, lr=0.005, wd=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = forward_fn(model)
        loss = F.cross_entropy(logits[train_mask_dev], y_author_dev[train_mask_dev])
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}, Loss: {loss.item():.4f}")

def run_relnn_train(session: Session, epochs=100, lr=0.005, wd=0.001):
    session.run(RELNN_FIT_DSL.format(epochs=epochs, lr=lr, wd=wd))

# ============================================================================
# Single seed experiment
# ============================================================================

def run_seed(seed: int, epochs: int = 100) -> dict:
    print()
    print(DIVIDER)
    print(f"Seed {seed}")
    print(DIVIDER)

    # -- Init PyG -------------------------------------------------------
    full_seed(seed)
    pyg_model = PyGHGT().to(DEVICE)
    with torch.no_grad():
        pyg_model()   # materialize lazy linears

    # -- Init hand-rolled with same weights ----------------------------
    full_seed(seed)
    hr_model = HandRolledHGT().to(DEVICE)
    print("[sync] PyG -> hand-rolled")
    sync_pyg_to_handrolled(pyg_model, hr_model)

    # -- Diagnose forward diff PyG <-> hand-rolled --------------------
    max_diff_hr = diagnose_pyg_vs_handrolled(pyg_model, hr_model)
    print(f"  PyG <-> hand-rolled logit max_diff = {max_diff_hr:.2e}")

    # -- Init RelNN with same weights ---------------------------------
    session = Session(db=relnn_db)
    session.run(RELNN_DEFINE_DSL)
    session.run(RELNN_LOGITS_DSL)   # materialize params
    print("[sync] hand-rolled -> RelNN")
    sync_handrolled_to_relnn(hr_model, session)

    # Forward check RelNN vs hand-rolled
    with torch.no_grad():
        hr_fwd = hr_model(x_author_dev, x_paper_dev, pa_edge_src_dev, pa_edge_dst_dev).cpu()
    rn_logits_pred = session.run(RELNN_LOGITS_DSL)
    rn_fwd = _align_relnn(rn_logits_pred, hr_fwd)
    max_diff_rn = _diff(hr_fwd, rn_fwd, "hand-rolled <-> RelNN logits")

    # -- Train all three from synced init -----------------------------
    print()
    print("[train] PyG")
    full_seed(seed)
    _sync_cuda()
    t0 = time.perf_counter()
    train_one(pyg_model,
              lambda m: m(),
              epochs=epochs)
    _sync_cuda()
    pyg_time = time.perf_counter() - t0
    pyg_accs = _eval_pyg(pyg_model)

    print("[train] hand-rolled")
    full_seed(seed)
    _sync_cuda()
    t0 = time.perf_counter()
    train_one(hr_model,
              lambda m: m(x_author_dev, x_paper_dev, pa_edge_src_dev, pa_edge_dst_dev),
              epochs=epochs)
    _sync_cuda()
    hr_time = time.perf_counter() - t0
    hr_accs = _eval_hr(hr_model)

    print("[train] RelNN")
    full_seed(seed)
    _sync_cuda()
    t0 = time.perf_counter()
    run_relnn_train(session, epochs=epochs)
    _sync_cuda()
    rn_time = time.perf_counter() - t0
    rn_accs = _eval_relnn(session)

    print()
    print("  [results]")
    print(f"  PyG       test={pyg_accs['test']:.1%}  time={pyg_time:.1f}s")
    print(f"  HandRoled test={hr_accs['test']:.1%}  time={hr_time:.1f}s")
    print(f"  RelNN     test={rn_accs['test']:.1%}  time={rn_time:.1f}s")
    print(f"  PyG<->HR  diff={abs(pyg_accs['test'] - hr_accs['test']):.1%}")
    print(f"  HR<->RN   diff={abs(hr_accs['test'] - rn_accs['test']):.1%}")
    print(f"  fwd_diff_pyg_hr={max_diff_hr:.2e}  fwd_diff_hr_rn={max_diff_rn:.2e}")

    return {
        "seed": seed,
        "fwd_diff_pyg_hr": max_diff_hr,
        "fwd_diff_hr_rn": max_diff_rn,
        "pyg": pyg_accs,
        "hr": hr_accs,
        "relnn": rn_accs,
        "pyg_time": pyg_time,
        "hr_time": hr_time,
        "rn_time": rn_time,
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    # Print PyG conv structure once for diagnostics
    full_seed(0)
    _tmp = PyGHGT().to(DEVICE)
    with torch.no_grad():
        _tmp()
    print("[PyG conv parameter shapes]")
    _inspect_pyg_conv(_tmp.conv)
    del _tmp

    seeds = [args.seed_start + i for i in range(args.runs)]
    all_results = []
    for seed in seeds:
        r = run_seed(seed, epochs=args.epochs)
        all_results.append(r)

    print()
    print(DIVIDER)
    print("SUMMARY (all seeds)")
    print(DIVIDER)
    print(f"{'Seed':>6}  {'PyG test':>10}  {'HR test':>10}  {'RN test':>10}  {'PyG-HR':>8}  {'HR-RN':>8}  {'fwd_pyg_hr':>12}  {'fwd_hr_rn':>12}")
    for r in all_results:
        print(f"  {r['seed']:4d}  {r['pyg']['test']:9.1%}  {r['hr']['test']:9.1%}  {r['relnn']['test']:9.1%}"
              f"  {abs(r['pyg']['test']-r['hr']['test']):7.1%}  {abs(r['hr']['test']-r['relnn']['test']):7.1%}"
              f"  {r['fwd_diff_pyg_hr']:12.2e}  {r['fwd_diff_hr_rn']:12.2e}")

    pyg_tests  = [r["pyg"]["test"] for r in all_results]
    hr_tests   = [r["hr"]["test"] for r in all_results]
    rn_tests   = [r["relnn"]["test"] for r in all_results]
    print()
    print(f"  PyG  mean={np.mean(pyg_tests):.1%}  std={np.std(pyg_tests, ddof=1):.1%}")
    print(f"  HR   mean={np.mean(hr_tests):.1%}  std={np.std(hr_tests, ddof=1):.1%}")
    print(f"  RN   mean={np.mean(rn_tests):.1%}  std={np.std(rn_tests, ddof=1):.1%}")
