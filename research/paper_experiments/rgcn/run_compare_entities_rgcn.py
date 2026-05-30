"""R-GCN on AIFB / MUTAG: torch-rgcn + PyG + RelNN template DSL.

This script compares:
1) torch-rgcn (Schlichtkrull reproduction code path),
2) PyG FastRGCNConv (``examples/rgcn.py`` protocol), and
3) RelNN template DSL using bounded-set expansion over ``MetaRel``.

For AIFB, RelNN uses per-relation ``NodeLookup<pe>`` embedding tables for layer 1
(mathematically equivalent to ``W_r @ one_hot(s)`` from the original featureless
R-GCN — same parameter count — but expressed as a gather, avoiding the 8285×8285
identity materialization). The embeddings are Linear-init for parity with the
prior ``Linear(num_nodes, hidden)`` parameterization.
For MUTAG, RelNN uses a shared ``NodeLookup`` plus basis decomposition in layers 1
and 2 to avoid a multi-GB one-hot matrix while adding per-relation expressiveness.

Run from repo root:
  python research/paper_experiments/rgcn/run_compare_entities_rgcn.py --datasets AIFB MUTAG --runs 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.nn import FastRGCNConv

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from relann.session import Session
from relann.torch_utils import full_seed, get_project_root

# torch-rgcn baseline (original paper reproduction code path).
_TORCH_RGCN_ROOT = Path(__file__).resolve().parents[3] / "_external" / "torch-rgcn"
if _TORCH_RGCN_ROOT.exists():
    sys.path.insert(0, str(_TORCH_RGCN_ROOT))
    from torch_rgcn.models import NodeClassifier  # type: ignore
else:
    NodeClassifier = None


class NodeLookup(torch.nn.Module):
    """Memory-safe node feature lookup for large entity sets."""

    def __init__(self, num_nodes: int, out_dim: int, linear_init: bool = False):
        super().__init__()
        self.emb = torch.nn.Embedding(num_nodes, out_dim)
        if linear_init:
            bound = 1.0 / (num_nodes ** 0.5)
            torch.nn.init.uniform_(self.emb.weight, -bound, bound)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        idx = z[..., 0].long()
        return self.emb(idx)


def _edge_norm_mean(edge_index: torch.Tensor, edge_type: torch.Tensor, num_nodes: int, num_rels: int) -> torch.Tensor:
    """Per-edge normalization 1/|N_r(t)| for mean aggregation (matches PyG RGCNConv)."""
    src, dst = edge_index[0], edge_index[1]
    device = edge_index.device
    dtype = torch.float32
    norms = torch.empty(edge_type.size(0), device=device, dtype=dtype)
    et = edge_type.long()
    for r in range(num_rels):
        m = et == r
        if not m.any():
            continue
        dst_r = dst[m]
        deg = torch.zeros(num_nodes, device=device, dtype=dtype)
        deg.scatter_add_(0, dst_r, torch.ones_like(dst_r, dtype=dtype))
        inv = 1.0 / deg[dst_r].clamp(min=1.0)
        norms[m] = inv
    return norms.unsqueeze(-1)


def _pad_edges_per_relation(s_r: torch.Tensor, d_r: torch.Tensor, w_r: torch.Tensor, n: int):
    """Ensure every target node exists in each E_r (zero-weight rows)."""
    has_in = torch.zeros(n, dtype=torch.bool, device=d_r.device if d_r.numel() else s_r.device)
    if d_r.numel() > 0:
        has_in[d_r.unique()] = True
    es, ed, ew = [], [], []
    for t in range(n):
        if not has_in[t]:
            es.append(0)
            ed.append(t)
            ew.append(0.0)
    if not es:
        return s_r, d_r, w_r
    pad_s = torch.tensor(es, dtype=s_r.dtype, device=s_r.device)
    pad_d = torch.tensor(ed, dtype=d_r.dtype, device=d_r.device)
    pad_w = torch.tensor(ew, dtype=w_r.dtype, device=w_r.device).unsqueeze(1)
    return torch.cat([s_r, pad_s]), torch.cat([d_r, pad_d]), torch.cat([w_r, pad_w])


def _augment_edges_with_inverse_and_self_loop(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    num_rels: int,
) -> tuple:
    """Augment with inverse relations and a self-loop relation, matching torch-rgcn convention.

    For each forward edge (s, t, r), adds an inverse edge (t, s, r + num_rels). Adds a self-loop
    edge (n, n, 2*num_rels) for every node n. Returns ``(aug_edge_index, aug_edge_type,
    num_effective_rels)`` where ``num_effective_rels = 2 * num_rels + 1`` (forward + inverse
    + self-loop).
    """
    inv_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
    inv_edge_type = edge_type + num_rels
    self_nodes = torch.arange(num_nodes, dtype=edge_index.dtype, device=edge_index.device)
    self_edge_index = torch.stack([self_nodes, self_nodes], dim=0)
    self_edge_type = torch.full((num_nodes,), 2 * num_rels, dtype=edge_type.dtype, device=edge_type.device)
    aug_edge_index = torch.cat([edge_index, inv_edge_index, self_edge_index], dim=1)
    aug_edge_type = torch.cat([edge_type, inv_edge_type, self_edge_type])
    return aug_edge_index, aug_edge_type, 2 * num_rels + 1


def _augmented_data_view(data, num_nodes: int, num_rels: int):
    """Return a SimpleNamespace with augmented edge_index/edge_type and original train/test attrs."""
    from types import SimpleNamespace
    aug_ei, aug_et, aug_nr = _augment_edges_with_inverse_and_self_loop(
        data.edge_index, data.edge_type.long(), num_nodes, num_rels
    )
    return SimpleNamespace(
        edge_index=aug_ei,
        edge_type=aug_et,
        train_idx=data.train_idx,
        train_y=data.train_y,
        test_idx=data.test_idx,
        test_y=data.test_y,
        num_nodes=num_nodes,
    ), aug_nr


def _build_db(data, num_nodes: int, num_rels: int, norms: torch.Tensor, feature_mode: str) -> dict:
    """Build per-relation edge tables + MetaRel guard relation + node features.

    feature_mode:
      - "one_hot": db["Nodes"] = identity matrix (num_nodes × num_nodes float32).
      - "shared_lookup": db["NodeIds"] for a single shared NodeLookup (MUTAG path).
      - "per_relation_lookup": db["NodeIds"] for per-relation NodeLookup<pe> tables (AIFB fast path).
    """
    pd = __import__("pandas")
    db: dict = {}
    src, dst = data.edge_index[0], data.edge_index[1]
    et = data.edge_type.long()

    for r in range(num_rels):
        m = et == r
        if not m.any():
            s_r = torch.tensor([], dtype=torch.long)
            d_r = torch.tensor([], dtype=torch.long)
            w_r = torch.empty(0, 1, dtype=torch.float32)
        else:
            s_r, d_r = src[m], dst[m]
            w_r = norms[m]
        s_r, d_r, w_r = _pad_edges_per_relation(s_r, d_r, w_r, num_nodes)
        df = pd.DataFrame({"s": s_r.cpu().numpy(), "t": d_r.cpu().numpy()})
        db[f"E{r}"] = (df, w_r.cpu())

    # Guard relation for bounded-set expansion: MetaRel(ts, pe, tt)
    meta = pd.DataFrame(
        {
            "ts": ["Node"] * num_rels,
            "pe": [f"E{r}" for r in range(num_rels)],
            "tt": ["Node"] * num_rels,
        }
    )
    db["MetaRel"] = (meta, torch.empty((len(meta), 0), dtype=torch.float32))

    # Node features: one-hot identity for "one_hot"; long indices for both lookup modes.
    n_df = pd.DataFrame({"n": np.arange(num_nodes, dtype=np.int64)})
    if feature_mode == "one_hot":
        db["Nodes"] = (n_df, torch.eye(num_nodes, dtype=torch.float32))
    elif feature_mode in ("shared_lookup", "per_relation_lookup"):
        node_ids = torch.arange(num_nodes, dtype=torch.long).view(-1, 1)
        db["NodeIds"] = (n_df, node_ids)
    else:
        raise ValueError(f"unknown feature_mode: {feature_mode!r}")

    train_df = pd.DataFrame({"n": data.train_idx.cpu().numpy()})
    train_y = data.train_y.long().view(-1, 1)
    db["TrainLabels"] = (train_df, train_y)
    return db


def _generate_relnn_rgcn_dsl(
    d0: int,
    hidden: int,
    num_classes: int,
    feature_mode: str,
    num_nodes: int,
    use_basis_l1: bool,
    use_basis_l2: bool,
    num_bases: int,
) -> str:
    """Template R-GCN with bounded-set expansion over MetaRel.

    feature_mode:
      - "one_hot": H0 reads from Nodes (identity matrix); per-relation Linear(num_nodes, hidden).
      - "shared_lookup": single shared NodeLookup; per-relation Linear(hidden, hidden) on top.
      - "per_relation_lookup": per-relation NodeLookup<pe>(num_nodes, hidden) — featureless R-GCN
        equivalent to "one_hot" but expressed as embedding lookup (no identity matmul).
    """
    if feature_mode not in ("one_hot", "shared_lookup", "per_relation_lookup"):
        raise ValueError(f"unknown feature_mode: {feature_mode!r}")
    common = f"""
#lang:relnn
d0 = {d0} .
hidden = {hidden} .
n_cls = {num_classes} .
num_nodes = {num_nodes} .
num_bases = {num_bases} .
"""
    if feature_mode == "shared_lookup":
        base = """
NodeInit = NodeLookup(num_nodes, hidden) .
H0(n; NodeInit(z)) :- NodeIds(n; z) .
"""
        d1 = "hidden"
    elif feature_mode == "per_relation_lookup":
        # H0 carries node ids; per-relation NodeLookup is applied inside the message rule.
        base = """
H0(n; z) :- NodeIds(n; z) .
"""
        d1 = "hidden"
    else:  # one_hot
        base = """
H0(n; z) :- Nodes(n; z) .
"""
        d1 = "d0"

    if use_basis_l1:
        layer1 = f"""
# Basis-decomposed relation weights for layer 1 (MUTAG-scale graphs).
V1<b> = Linear(hidden, hidden, False) .
A1<pe, b> = Tensor(1) .
Root1 = Linear({d1}, hidden, False) .
Root2 = Linear(hidden, n_cls, False) .

def Msg1Basis<pe, b>():
    Out(s, t; A1<pe, b> * V1<b>(z) * w) :- H0(s; z), pe(s, t; w) .
enddef

def RelAgg1<ts, pe, tt>():
    Basis1(t; sum(z)) :- Union(Set(Msg1Basis<pe, b>(s, t; z) | 1 <= b, b <= num_bases)) .
    Out(t; z) :- Basis1(t; z) .
enddef

H1Agg(n; sum(z)) :- Union(Set(RelAgg1<ts, pe, tt>(n; z) | MetaRel(ts, pe, tt))) .
H1(n; ReLU(z_msg + Root1(z_self))) :- H1Agg(n; z_msg), H0(n; z_self) .
"""
    elif feature_mode == "per_relation_lookup":
        # Per-relation embedding tables replace per-relation Linear(num_nodes, hidden).
        # Mathematically equivalent to "one_hot" featureless R-GCN — same param count
        # (90 × num_nodes × hidden), Linear-style init via the third (True) ctor arg.
        layer1 = """
NodeEmb1<pe> = NodeLookup(num_nodes, hidden, True) .
Root1Emb = NodeLookup(num_nodes, hidden, True) .
Root2 = Linear(hidden, n_cls, False) .

def RelAgg1<ts, pe, tt>():
    Msg(s, t; NodeEmb1<pe>(z) * w) :- H0(s; z), pe(s, t; w) .
    Out(t; sum(z)) :- Msg(s, t; z) .
enddef

H1Agg(n; sum(z)) :- Union(Set(RelAgg1<ts, pe, tt>(n; z) | MetaRel(ts, pe, tt))) .
H1(n; ReLU(z_msg + Root1Emb(z_self))) :- H1Agg(n; z_msg), H0(n; z_self) .
"""
    else:
        layer1 = f"""
W1<pe> = Linear({d1}, hidden, False) .
Root1 = Linear({d1}, hidden, False) .
Root2 = Linear(hidden, n_cls, False) .

def RelAgg1<ts, pe, tt>():
    Msg(s, t; W1<pe>(z) * w) :- H0(s; z), pe(s, t; w) .
    Out(t; sum(z)) :- Msg(s, t; z) .
enddef

H1Agg(n; sum(z)) :- Union(Set(RelAgg1<ts, pe, tt>(n; z) | MetaRel(ts, pe, tt))) .
H1(n; ReLU(z_msg + Root1(z_self))) :- H1Agg(n; z_msg), H0(n; z_self) .
"""

    if use_basis_l2:
        l2 = """
# Basis-decomposed relation weights for layer 2.
V2<b> = Linear(hidden, n_cls, False) .
A2<pe, b> = Tensor(1) .

def Msg2Basis<pe, b>():
    Out(s, t; A2<pe, b> * V2<b>(z) * w) :- H1(s; z), pe(s, t; w) .
enddef

def RelAgg2<ts, pe, tt>():
    Basis(t; sum(z)) :- Union(Set(Msg2Basis<pe, b>(s, t; z) | 1 <= b, b <= num_bases)) .
    Out(t; z) :- Basis(t; z) .
enddef
"""
    else:
        l2 = """
W2<pe> = Linear(hidden, n_cls, False) .
def RelAgg2<ts, pe, tt>():
    Msg(s, t; W2<pe>(z) * w) :- H1(s; z), pe(s, t; w) .
    Out(t; sum(z)) :- Msg(s, t; z) .
enddef
"""

    return (
        common
        + base
        + layer1
        + l2
        + """
H2Agg(n; sum(z)) :- Union(Set(RelAgg2<ts, pe, tt>(n; z) | MetaRel(ts, pe, tt))) .
Logits(n; z_msg + Root2(z_self)) :- H2Agg(n; z_msg), H1(n; z_self) .
"""
    )


def _generate_arch_a_dsl(num_nodes: int, hidden: int, n_cls: int) -> str:
    """Arch A: full per-relation NodeLookup, layer biases, no separate root.

    Matches torch-rgcn's parameterization: with augmented MetaRel (forward + inverse +
    self-loop = ``2*num_rels + 1`` effective relations), the self-loop relation IS the root
    self-transform, so no separate ``Root1Emb`` is needed. Layer biases match torch-rgcn's
    per-layer bias vector. Total params: ``(2*num_rels + 1) * num_nodes * hidden + hidden +
    (2*num_rels + 1) * hidden * n_cls + n_cls``.
    """
    return f"""
#lang:relnn
hidden = {hidden} .
n_cls = {n_cls} .
num_nodes = {num_nodes} .

H0(n; z) :- NodeIds(n; z) .

NodeEmb1<pe> = NodeLookup(num_nodes, hidden, True) .
Bias1 = Tensor(1, hidden) .

W2<pe> = Linear(hidden, n_cls, False) .
Bias2 = Tensor(1, n_cls) .

def RelAgg1<ts, pe, tt>():
    Msg(s, t; NodeEmb1<pe>(z) * w) :- H0(s; z), pe(s, t; w) .
    Out(t; sum(z)) :- Msg(s, t; z) .
enddef

H1Agg(n; sum(z)) :- Union(Set(RelAgg1<ts, pe, tt>(n; z) | MetaRel(ts, pe, tt))) .
H1(n; ReLU(z_msg + Bias1)) :- H1Agg(n; z_msg) .

def RelAgg2<ts, pe, tt>():
    Msg(s, t; W2<pe>(z) * w) :- H1(s; z), pe(s, t; w) .
    Out(t; sum(z)) :- Msg(s, t; z) .
enddef

H2Agg(n; sum(z)) :- Union(Set(RelAgg2<ts, pe, tt>(n; z) | MetaRel(ts, pe, tt))) .
Logits(n; z_msg + Bias2) :- H2Agg(n; z_msg) .
"""


def _generate_arch_b_dsl(num_nodes: int, hidden: int, n_cls: int, num_bases: int) -> str:
    """Arch B: basis-decomposed per-relation NodeLookup, separate root, layer biases.

    Matches PyG ``FastRGCNConv`` parameterization: ``num_bases`` shared basis embedding
    tables ``V1<b>`` (NodeLookups for layer 1; Linears for layer 2 since L2 input is
    hidden-dim), per-relation coefficients ``A<pe, b>``, separate root self-transform
    (``Root1Emb`` / ``Root2``), and a layer-wide bias. Total params for layer 1:
    ``num_bases * num_nodes * hidden + num_rels * num_bases + num_nodes * hidden + hidden``.
    """
    return f"""
#lang:relnn
hidden = {hidden} .
n_cls = {n_cls} .
num_nodes = {num_nodes} .
num_bases = {num_bases} .

H0(n; z) :- NodeIds(n; z) .

V1<b> = NodeLookup(num_nodes, hidden, True) .
A1<pe, b> = Tensor(1, 1.0) .
Root1Emb = NodeLookup(num_nodes, hidden, True) .
Bias1 = Tensor(1, hidden) .

V2<b> = Linear(hidden, n_cls, False) .
A2<pe, b> = Tensor(1, 1.0) .
Root2 = Linear(hidden, n_cls, False) .
Bias2 = Tensor(1, n_cls) .

def Msg1Basis<pe, b>():
    Out(s, t; A1<pe, b> * V1<b>(z) * w) :- H0(s; z), pe(s, t; w) .
enddef

def RelAgg1<ts, pe, tt>():
    Basis1(t; sum(z)) :- Union(Set(Msg1Basis<pe, b>(s, t; z) | 1 <= b, b <= num_bases)) .
    Out(t; z) :- Basis1(t; z) .
enddef

H1Agg(n; sum(z)) :- Union(Set(RelAgg1<ts, pe, tt>(n; z) | MetaRel(ts, pe, tt))) .
H1(n; ReLU(z_msg + Root1Emb(z_self) + Bias1)) :- H1Agg(n; z_msg), H0(n; z_self) .

def Msg2Basis<pe, b>():
    Out(s, t; A2<pe, b> * V2<b>(z) * w) :- H1(s; z), pe(s, t; w) .
enddef

def RelAgg2<ts, pe, tt>():
    Basis2(t; sum(z)) :- Union(Set(Msg2Basis<pe, b>(s, t; z) | 1 <= b, b <= num_bases)) .
    Out(t; z) :- Basis2(t; z) .
enddef

H2Agg(n; sum(z)) :- Union(Set(RelAgg2<ts, pe, tt>(n; z) | MetaRel(ts, pe, tt))) .
Logits(n; z_msg + Root2(z_self) + Bias2) :- H2Agg(n; z_msg), H1(n; z_self) .
"""


def _count_dsl_loc(dsl: str) -> int:
    loc = 0
    for line in dsl.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        loc += 1
    return loc


class PyGNet(torch.nn.Module):
    def __init__(self, num_nodes: int, num_rels: int, num_classes: int, num_bases: int = 30):
        super().__init__()
        self.conv1 = FastRGCNConv(num_nodes, 16, num_rels, num_bases=num_bases)
        self.conv2 = FastRGCNConv(16, num_classes, num_rels, num_bases=num_bases)

    def forward(self, edge_index, edge_type):
        x = self.conv1(None, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


def _train_pyg(model, data, device, epochs: int = 50, lr: float = 0.01, wd: float = 5e-4):
    model = model.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)
    train_idx = data.train_idx.to(device)
    test_idx = data.test_idx.to(device)
    train_y = data.train_y.to(device)
    test_y = data.test_y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(edge_index, edge_type)
        loss = F.nll_loss(out[train_idx], train_y)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        pred = model(edge_index, edge_type).argmax(dim=-1)
        test_acc = (pred[test_idx] == test_y).float().mean().item()
    n_params = sum(p.numel() for p in model.parameters())
    return test_acc, elapsed, int(n_params)


def _train_torch_rgcn(data, device: torch.device, hidden: int, num_bases: int | None, epochs: int, lr: float, wd: float):
    """Run torch-rgcn NodeClassifier directly on PyG Entities tensors."""
    if NodeClassifier is None:
        raise RuntimeError("torch-rgcn is not available at _external/torch-rgcn")

    triples = torch.stack([data.edge_index[0], data.edge_type.long(), data.edge_index[1]], dim=1).cpu().numpy()
    num_nodes = int(data.num_nodes)
    num_rels = int(data.edge_type.max().item()) + 1
    num_classes = int(data.train_y.max().item()) + 1
    decomp = {"type": "basis", "num_bases": num_bases} if num_bases is not None else None

    model = NodeClassifier(
        triples=triples,
        nnodes=num_nodes,
        nrel=num_rels,
        nfeat=None,
        nhid=hidden,
        nlayers=2,
        nclass=num_classes,
        decomposition=decomp,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    train_idx = data.train_idx.to(device)
    test_idx = data.test_idx.to(device)
    train_y = data.train_y.to(device)
    test_y = data.test_y.to(device)

    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model()
        loss = F.cross_entropy(logits[train_idx], train_y)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        pred = model().argmax(dim=-1)
        test_acc = (pred[test_idx] == test_y).float().mean().item()
    n_params = sum(p.numel() for p in model.parameters())
    return float(test_acc), float(elapsed), int(n_params)


def _train_relnn(db: dict, dsl: str, data, device: torch.device, epochs: int = 50, lr: float = 0.01, wd: float = 5e-4):
    train_idx = data.train_idx
    test_idx = data.test_idx
    test_y = data.test_y
    session = Session(db=db, device=device)
    session.run(dsl)
    fit = f"""
#lang:relnn
?fit <epochs={epochs}, lr={lr}, weight_decay={wd}>
Loss(; CrossEntropyLoss()(z_pred, z_true)) :- Logits(n; z_pred), TrainLabels(n; z_true) .
"""
    t0 = time.perf_counter()
    session.run(fit)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    pred = session.run("""#lang:relnn\n?pred P(n; ArgMax()(z)) :- Logits(n; z) .""")
    rn_df = pred.content
    rn_cls = pred.embeddings[0].view(-1).long().cpu()
    n_to_pred = dict(zip(rn_df["n"].values, rn_cls.numpy()))
    correct = sum(1 for i, tid in enumerate(test_idx.tolist()) if int(n_to_pred.get(tid, -1)) == int(test_y[i].item()))
    test_acc = correct / len(test_idx)
    n_params = sum(p.numel() for p in session.engine.parameter_store.values())
    return float(test_acc), float(elapsed), int(n_params)


def run_one_dataset(
    name: str, root: Path, seed: int, device: torch.device,
    epochs: int, lr: float, wd: float, *, include_basis_arch: bool = False,
) -> dict:
    full_seed(seed)
    path = str(root / "data" / "Entities")
    ds = Entities(path, name)
    data = ds[0]
    num_nodes = data.num_nodes
    num_rels = ds.num_relations
    num_classes = int(data.train_y.max().item()) + 1
    hidden = 16

    pyg_model = PyGNet(num_nodes, num_rels, num_classes, num_bases=30)
    pyg_acc, pyg_time, pyg_params = _train_pyg(pyg_model, data, device, epochs=epochs, lr=lr, wd=wd)
    torch_rgcn_bases = 30 if name.upper() == "MUTAG" else None
    torch_rgcn_acc, torch_rgcn_time, torch_rgcn_params = _train_torch_rgcn(
        data=data,
        device=device,
        hidden=hidden,
        num_bases=torch_rgcn_bases,
        epochs=epochs,
        lr=lr,
        wd=wd if name.upper() == "MUTAG" else 0.0,
    )

    if name.upper() == "AIFB":
        # AIFB: report Arch A (RelNN-full) by default. Optionally include Arch B (RelNN-basis)
        # when ``include_basis_arch=True`` — see notes in the PR for why basis-decomp R-GCN is
        # currently impractical in pure DSL (needs random init in ``Tensor()``; tracked separately).

        # ---- Arch A: torch-rgcn match (full per-relation, augmented w/ inverse + self-loop, layer biases) ----
        aug_data, num_eff_rels = _augmented_data_view(data, num_nodes, num_rels)
        aug_norms = _edge_norm_mean(aug_data.edge_index, aug_data.edge_type, num_nodes, num_eff_rels)
        db_a = _build_db(aug_data, num_nodes, num_eff_rels, aug_norms.to(aug_data.edge_index.device), feature_mode="per_relation_lookup")
        dsl_a = _generate_arch_a_dsl(num_nodes=num_nodes, hidden=hidden, n_cls=num_classes)
        dsl_a_loc = _count_dsl_loc(dsl_a)
        full_seed(seed)
        relnn_a_acc, relnn_a_time, relnn_a_params = _train_relnn(db_a, dsl_a, data, device, epochs=epochs, lr=lr, wd=wd)

        out = {
            "dataset": name,
            "seed": seed,
            "num_nodes": num_nodes,
            "num_relations": num_rels,
            "num_classes": num_classes,
            "torch_rgcn": {"test_acc": torch_rgcn_acc, "time_s": torch_rgcn_time, "params": torch_rgcn_params},
            "pyg": {"test_acc": pyg_acc, "time_s": pyg_time, "params": pyg_params},
            "relnn_full": {
                "test_acc": relnn_a_acc, "time_s": relnn_a_time, "params": relnn_a_params,
                "dsl_loc": dsl_a_loc, "match": "torch-rgcn",
            },
        }

        if include_basis_arch:
            # ---- Arch B: PyG match (basis decomp, separate root, layer biases) ----
            # WARNING: with current ``Tensor()`` init (constant fill only), the basis coefficients
            # cannot be properly randomized → model overfits training set without generalizing.
            # Also slow: 90 × 30 = 2700 template specializations. See PR notes.
            norms = _edge_norm_mean(data.edge_index, data.edge_type, num_nodes, num_rels)
            db_b = _build_db(data, num_nodes, num_rels, norms.to(data.edge_index.device), feature_mode="per_relation_lookup")
            dsl_b = _generate_arch_b_dsl(num_nodes=num_nodes, hidden=hidden, n_cls=num_classes, num_bases=30)
            dsl_b_loc = _count_dsl_loc(dsl_b)
            full_seed(seed)
            relnn_b_acc, relnn_b_time, relnn_b_params = _train_relnn(db_b, dsl_b, data, device, epochs=epochs, lr=lr, wd=wd)
            out["relnn_basis"] = {
                "test_acc": relnn_b_acc, "time_s": relnn_b_time, "params": relnn_b_params,
                "dsl_loc": dsl_b_loc, "match": "PyG FastRGCN",
            }

        return out

    # ---- MUTAG path: existing shared_lookup + basis decomp ----
    norms = _edge_norm_mean(data.edge_index, data.edge_type, num_nodes, num_rels)
    feature_mode = "shared_lookup"
    use_basis_l1 = True
    use_basis_l2 = True
    relnn_num_bases = 30
    db = _build_db(data, num_nodes, num_rels, norms.to(data.edge_index.device), feature_mode=feature_mode)
    dsl = _generate_relnn_rgcn_dsl(
        d0=hidden, hidden=hidden, num_classes=num_classes,
        feature_mode=feature_mode, num_nodes=num_nodes,
        use_basis_l1=use_basis_l1, use_basis_l2=use_basis_l2, num_bases=relnn_num_bases,
    )
    dsl_loc = _count_dsl_loc(dsl)
    full_seed(seed)
    relnn_acc, relnn_time, relnn_params = _train_relnn(db, dsl, data, device, epochs=epochs, lr=lr, wd=wd)

    return {
        "dataset": name,
        "seed": seed,
        "num_nodes": num_nodes,
        "num_relations": num_rels,
        "num_classes": num_classes,
        "dsl_loc": dsl_loc,
        "torch_rgcn": {"test_acc": torch_rgcn_acc, "time_s": torch_rgcn_time, "params": torch_rgcn_params},
        "pyg": {"test_acc": pyg_acc, "time_s": pyg_time, "params": pyg_params},
        "relnn": {"test_acc": relnn_acc, "time_s": relnn_time, "params": relnn_params},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["AIFB", "MUTAG"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--num-threads", type=int, default=0, help="Set torch thread count (>0 to enable).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("research/paper_experiments/rgcn/results/rgcn_entities_results.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Markdown summary path. Defaults to --out-json with .json replaced by _summary.md.",
    )
    parser.add_argument(
        "--include-basis-arch",
        action="store_true",
        help="Also run RelNN-basis (Arch B) for AIFB. Off by default — basis decomp via DSL "
             "templates is slow (~32 min/seed) and currently doesn't generalize because "
             "Tensor() doesn't support random init for basis coefficients.",
    )
    args = parser.parse_args()
    if args.out_md is None:
        args.out_md = args.out_json.with_name(args.out_json.stem + "_summary.md")

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    device = torch.device("cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu"))
    project_root = get_project_root()
    results = []
    for name in args.datasets:
        run_list = []
        for i in range(args.runs):
            seed = args.seed_start + i
            print(f"[{name}] seed={seed} ...")
            row = run_one_dataset(
                name, project_root, seed, device,
                epochs=args.epochs, lr=args.lr, wd=args.weight_decay,
                include_basis_arch=args.include_basis_arch,
            )
            run_list.append(row)
            if name.upper() == "AIFB":
                msg = (
                    f"  PyG test_acc={row['pyg']['test_acc']:.4f} time={row['pyg']['time_s']:.2f}s | "
                    f"RelNN-full test_acc={row['relnn_full']['test_acc']:.4f} time={row['relnn_full']['time_s']:.2f}s "
                    f"params={row['relnn_full']['params']}"
                )
                if "relnn_basis" in row:
                    msg += (
                        f" | RelNN-basis test_acc={row['relnn_basis']['test_acc']:.4f} "
                        f"time={row['relnn_basis']['time_s']:.2f}s "
                        f"params={row['relnn_basis']['params']}"
                    )
                print(msg)
            else:
                print(
                    f"  PyG test_acc={row['pyg']['test_acc']:.4f} time={row['pyg']['time_s']:.2f}s | "
                    f"RelNN test_acc={row['relnn']['test_acc']:.4f} time={row['relnn']['time_s']:.2f}s"
                )
        results.append({"dataset": name, "runs": run_list})

    def _mean_std(xs):
        a = np.array(xs, dtype=np.float64)
        return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0

    summary_lines = [
        "# R-GCN Entities (AIFB / MUTAG)",
        "",
        f"- Timestamp (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Device: {device}",
        f"- Runs per dataset: {args.runs}",
        "",
        "| Dataset | Impl | Test Acc (mean ± std) | Time (s) (mean ± std) | Params | DSL LOC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    payload = {"metadata": {"device": str(device), "runs": args.runs, "seed_start": args.seed_start}, "results": results}

    def _emit_row(label: str, accs, times, params: int, loc):
        m_acc = _mean_std(accs)
        m_t = _mean_std(times)
        loc_str = loc if loc is not None else "—"
        summary_lines.append(
            f"| {name} | {label} | {100*m_acc[0]:.2f}% ± {100*m_acc[1]:.2f}% | {m_t[0]:.2f} ± {m_t[1]:.2f} | {params} | {loc_str} |"
        )

    for block in results:
        name = block["dataset"]
        torch_rgcn_runs = [r["torch_rgcn"] for r in block["runs"]]
        pyg_runs = [r["pyg"] for r in block["runs"]]
        _emit_row(
            "torch-rgcn",
            [r["test_acc"] for r in torch_rgcn_runs],
            [r["time_s"] for r in torch_rgcn_runs],
            torch_rgcn_runs[0]["params"],
            None,
        )
        _emit_row(
            "PyG FastRGCN",
            [r["test_acc"] for r in pyg_runs],
            [r["time_s"] for r in pyg_runs],
            pyg_runs[0]["params"],
            None,
        )
        # AIFB has at least one RelNN entry (full = torch-rgcn match); optionally a basis variant.
        if name.upper() == "AIFB":
            relnn_full_runs = [r["relnn_full"] for r in block["runs"]]
            _emit_row(
                "RelNN (full, ↔ torch-rgcn)",
                [r["test_acc"] for r in relnn_full_runs],
                [r["time_s"] for r in relnn_full_runs],
                relnn_full_runs[0]["params"],
                relnn_full_runs[0]["dsl_loc"],
            )
            if all("relnn_basis" in r for r in block["runs"]):
                relnn_basis_runs = [r["relnn_basis"] for r in block["runs"]]
                _emit_row(
                    "RelNN (basis, ↔ PyG)",
                    [r["test_acc"] for r in relnn_basis_runs],
                    [r["time_s"] for r in relnn_basis_runs],
                    relnn_basis_runs[0]["params"],
                    relnn_basis_runs[0]["dsl_loc"],
                )
        else:
            relnn_runs = [r["relnn"] for r in block["runs"]]
            _emit_row(
                "RelNN",
                [r["test_acc"] for r in relnn_runs],
                [r["time_s"] for r in relnn_runs],
                relnn_runs[0]["params"],
                block["runs"][0].get("dsl_loc"),
            )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
