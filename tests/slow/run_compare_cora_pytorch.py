"""3-way comparison: Templated RelNN vs Pure PyTorch (+ PyG GCNConv) on Cora.

Uses the SessionComparison harness from relann.comparison for structured
RelNN-vs-PyG comparison at three complexity levels:

  Level 1 -- GCN (no attention, no heads): all three models start from
             identical weights (Kaiming init synced from hand-rolled to
             PyG and RelNN). No final ReLU on output layer (standard GCN).
  Level 2 -- HGT single-layer (3 heads, attention): expect ~53%
  Level 3 -- HGT two-layer: weight-sync forward comparison proves
             architectural equivalence even when init ordering differs.

Run from repo root:
    python tests/slow/run_compare_cora_pytorch.py
"""
import sys
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from relann.torch_utils import full_seed, get_project_root
from relann.session import Session
from relann.datasets import load_cora_dataset, evaluate_node_classification
from relann.comparison import SessionComparison

project_root = get_project_root()
cora_path = project_root / "data" / "Planetoid" / "Cora"
if not cora_path.exists():
    print("SKIP: Cora data not found at", cora_path)
    sys.exit(0)

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, degree

# -- Load Cora ---------------------------------------------------------------

dataset = Planetoid(str(cora_path.parent), "Cora", transform=T.NormalizeFeatures())
pyg_data = dataset[0]

edge_index_sl, _ = add_self_loops(pyg_data.edge_index)
src_all, dst_all = edge_index_sl
deg = degree(src_all, pyg_data.x.size(0), dtype=pyg_data.x.dtype)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
norm_weights = (deg_inv_sqrt[src_all] * deg_inv_sqrt[dst_all]).unsqueeze(-1)

raw_edge_index = pyg_data.edge_index

x = pyg_data.x
y = pyg_data.y
train_mask = pyg_data.train_mask
test_mask = pyg_data.test_mask
N = x.size(0)
in_features = 1433
n_classes = 7

relnn_data = load_cora_dataset()
relnn_db = {k: relnn_data[k] for k in ("Papers", "Citation", "Labels")}

DIVIDER = "=" * 70

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def pyg_scatter_add(z, edge_src, edge_dst, w, n_nodes):
    msg = z[edge_src] * w
    out = torch.zeros(n_nodes, z.size(1), device=z.device)
    out.index_add_(0, edge_dst, msg)
    return out

# =============================================================================
# Level 1: GCN (2-layer, no attention)
# =============================================================================

class PyTorchGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(in_features, 16, bias=False)
        self.lin2 = nn.Linear(16, n_classes, bias=False)

    def forward(self, x_feat, edge_src, edge_dst, w):
        z = self.lin1(x_feat)
        z = pyg_scatter_add(z, edge_src, edge_dst, w, N)
        z = F.relu(z)
        z = self.lin2(z)
        z = pyg_scatter_add(z, edge_src, edge_dst, w, N)
        return z

class PyGGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(in_features, 16, bias=False, add_self_loops=True)
        self.conv2 = GCNConv(16, n_classes, bias=False, add_self_loops=True)

    def forward(self, x_feat, edge_index):
        z = self.conv1(x_feat, edge_index)
        z = F.relu(z)
        z = self.conv2(z, edge_index)
        return z

def _train_eval(model, forward_fn, epochs=100, lr=0.01, wd=0.0005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = forward_fn(model)
        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}, Loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        logits = forward_fn(model)
        preds = logits.argmax(dim=-1)
        test_acc = (preds[test_mask] == y[test_mask]).float().mean().item()
    return count_params(model), losses, test_acc

GCN_DSL = """
#lang:relnn
in_channels = {in_channels} .
hidden_channels = 16 .
out_channels = {out_channels} .

def GCNLayer<d_in, d_out>(Nodes, Edges):
    Emb(pid; Linear(d_in, d_out, False)(z)) :- Nodes(pid; z) .
    Out(cited; sum(z * w)) :- Emb(citing; z), Edges(citing, cited; w) .
enddef

def GCN<d_in, d_hidden, d_out>(Nodes, Edges):
    L1(cited; ReLU(z)) :- GCNLayer<d_in, d_hidden>(Nodes, Edges)(cited; z) .
    Output(cited; z) :- GCNLayer<d_hidden, d_out>(L1, Edges)(cited; z) .
enddef

Output(cited; z) :- GCN<in_channels, hidden_channels, out_channels>(Papers, Citation)(cited; z) .
"""

def level1_gcn():
    print(DIVIDER)
    print("LEVEL 1: GCN (2-layer, no attention)")
    print(DIVIDER)

    # -- Create all models with identical initial weights ----------------------
    full_seed(42)
    hr_model = PyTorchGCN()

    pyg_model = PyGGCN()
    with torch.no_grad():
        pyg_model.conv1.lin.weight.copy_(hr_model.lin1.weight)
        pyg_model.conv2.lin.weight.copy_(hr_model.lin2.weight)

    session = Session(db=relnn_db)
    session.run(GCN_DSL.format(in_channels=in_features, out_channels=n_classes))
    session.run("""
#lang:relnn
?pred _Init(cited; z) :- Output(cited; z) .
""")
    store = session.engine.parameter_store
    lin1_key = next(k for k, v in store.items() if v.shape == (16, in_features))
    lin2_key = next(k for k, v in store.items() if v.shape == (n_classes, 16))
    with torch.no_grad():
        store[lin1_key].data.copy_(hr_model.lin1.weight.data)
        store[lin2_key].data.copy_(hr_model.lin2.weight.data)
    print(f"  Weight sync: HR -> PyG, HR -> RelNN (identical Kaiming init)")

    # -- Train hand-rolled -----------------------------------------------------
    print("  [Hand-rolled PyTorch]")
    full_seed(42)
    t0 = time.perf_counter()
    hr_params, hr_losses, hr_acc = _train_eval(
        hr_model, lambda m: m(x, src_all, dst_all, norm_weights))
    hr_time = time.perf_counter() - t0
    print(f"    Params: {hr_params}, Test acc: {hr_acc:.1%}, Time: {hr_time:.1f}s")

    # -- Train PyG (synced init) -----------------------------------------------
    print("  [PyG GCNConv (synced init)]")
    full_seed(42)
    t0 = time.perf_counter()
    pyg_params, pyg_losses, pyg_acc = _train_eval(
        pyg_model, lambda m: m(x, raw_edge_index))
    pyg_time = time.perf_counter() - t0
    print(f"    Params: {pyg_params}, Test acc: {pyg_acc:.1%}, Time: {pyg_time:.1f}s")

    # -- Train RelNN (synced init) ---------------------------------------------
    print("  [RelNN Templated (synced init)]")
    full_seed(42)
    t0 = time.perf_counter()
    session.run("""
#lang:relnn
?fit <epochs=100, lr=0.01, weight_decay=0.0005>
Loss(; CrossEntropyLoss()(z_pred, z)) :- Output(cited; z_pred), Labels(cited; z) .
""")
    rn_time = time.perf_counter() - t0

    cmp = SessionComparison("GCN-Level1", verbose=True)
    cmp.set_pyg_model(hr_model)
    cmp.set_relnn_session(session)
    cmp.assert_param_count()

    rn_params = sum(p.numel() for p in session.engine.parameter_store.values())
    pred = session.run("""
#lang:relnn
?pred Predictions(cited; ArgMax()(z)) :- Output(cited; z) .
""")
    rn_acc = evaluate_node_classification(relnn_data, pred, return_value=True)
    print(f"    Params: {rn_params}, Test acc: {rn_acc:.1%}, Time: {rn_time:.1f}s")

    print(f"  Param count -- HR: {hr_params}, PyG: {pyg_params}, RelNN: {rn_params}")
    print(f"  Accuracy -- HR: {hr_acc:.1%}, PyG: {pyg_acc:.1%}, RelNN: {rn_acc:.1%}")
    print(f"  Training time -- HR: {hr_time:.1f}s, PyG: {pyg_time:.1f}s, RelNN: {rn_time:.1f}s")

    assert hr_params == rn_params, f"Param count mismatch: HR={hr_params}, RelNN={rn_params}"
    assert pyg_params == rn_params, f"Param count mismatch: PyG={pyg_params}, RelNN={rn_params}"
    assert hr_acc > 0.75, f"Hand-rolled GCN expected >75%, got {hr_acc:.1%}"
    assert pyg_acc > 0.75, f"PyG GCN expected >75%, got {pyg_acc:.1%}"
    assert rn_acc > 0.75, f"RelNN GCN expected >75%, got {rn_acc:.1%}"
    assert abs(hr_acc - pyg_acc) < 0.02, (
        f"HR-PyG accuracy gap >2% despite synced init: HR={hr_acc:.1%}, PyG={pyg_acc:.1%}")
    print("  PASS")
    return hr_acc, pyg_acc, rn_acc, hr_time, pyg_time, rn_time

def debug_gcn_forward():
    """Weight-synced forward comparison for 2-layer GCN (hand-rolled <-> RelNN)."""
    print()
    print(DIVIDER)
    print("GCN DEBUG: weight-synced forward-pass comparison")
    print(DIVIDER)

    full_seed(99)
    pt_model = PyTorchGCN()
    pt_model.eval()

    full_seed(99)
    session = Session(db=relnn_db)
    session.run(GCN_DSL.format(in_channels=in_features, out_channels=n_classes))

    session.run("""
#lang:relnn
?pred InitPred(cited; z) :- Output(cited; z) .
""")

    store = session.engine.parameter_store
    print(f"  RelNN params ({len(store)}):")
    for k in sorted(store.keys()):
        print(f"    {k}: {store[k].shape}")

    # Discover weight FQNs by shape
    lin1_key = None
    lin2_key = None
    for k, v in store.items():
        if v.shape == (16, in_features):
            lin1_key = k
        elif v.shape == (n_classes, 16):
            lin2_key = k
    assert lin1_key, f"lin1 weight ({16}, {in_features}) not found in store"
    assert lin2_key, f"lin2 weight ({n_classes}, 16) not found in store"

    mapping = {
        "lin1.weight": lin1_key,
        "lin2.weight": lin2_key,
    }
    cmp = SessionComparison("GCN-Forward", verbose=True)
    cmp.set_pyg_model(pt_model)
    cmp.set_relnn_session(session)
    cmp.print_params()
    cmp.set_mapping(mapping)
    cmp.print_mapping()
    cmp.sync_weights()

    def _align(relnn_pred, ref_out):
        rn_out = relnn_pred.embeddings[0]
        rn_df = relnn_pred.content
        col = rn_df.columns[0]
        rn_ids = rn_df[col].values
        aligned = torch.zeros_like(ref_out)
        for pos, nid in enumerate(rn_ids):
            aligned[int(nid)] = rn_out[pos]
        return aligned

    result = cmp.compare_forward(
        pyg_fn=lambda m: m(x, src_all, dst_all, norm_weights),
        relnn_pred_dsl="""
#lang:relnn
?pred PredOut(cited; z) :- Output(cited; z) .
""",
        align_fn=_align,
        tolerance=1e-4,
    )

    assert result.passed, f"GCN weight-synced forward FAILED: max_diff={result.max_diff:.2e}"
    print(f"  [{'OK' if result.passed else 'FAIL'}] GCN forward: max_diff={result.max_diff:.2e}")
    return result.passed

# =============================================================================
# Level 2: HGT single-layer (3 heads, element-wise attention)
# =============================================================================

d_hgt = 6
h_hgt = 3

class PyTorchHGT1(nn.Module):
    def __init__(self):
        super().__init__()
        dh = d_hgt // h_hgt
        self.papers_emb = nn.Linear(in_features, d_hgt, bias=False)
        self.K = nn.ModuleList([nn.Linear(d_hgt, dh) for _ in range(h_hgt)])
        self.Q = nn.ModuleList([nn.Linear(d_hgt, dh) for _ in range(h_hgt)])
        self.M = nn.ModuleList([nn.Linear(d_hgt, dh) for _ in range(h_hgt)])
        self.Mu = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(h_hgt)])
        self.A_LIN = nn.Linear(d_hgt, d_hgt)
        self.classifier = nn.Linear(d_hgt, n_classes)

    def forward(self, x_feat, edge_src, edge_dst):
        z = self.papers_emb(x_feat)
        att_heads = []
        for i in range(h_hgt):
            att_heads.append(self.K[i](z[edge_src]) * self.Q[i](z[edge_dst]) * self.Mu[i])
        att_con = torch.cat(att_heads, dim=-1)
        msg_heads = [self.M[i](z[edge_src]) for i in range(h_hgt)]
        msg = torch.cat(msg_heads, dim=-1)
        weighted_msg = msg * att_con
        agg = torch.zeros(N, d_hgt, device=x_feat.device)
        agg.index_add_(0, edge_dst, weighted_msg)
        out = self.A_LIN(F.relu(agg)) + z
        return out, self.classifier(out)

def _build_pt_to_relnn_map_hgt1():
    mapping = {}
    mapping["papers_emb.weight"] = "global.transformation_Papers_Emb._module.weight"
    for head in range(h_hgt):
        rh = head + 1
        for td in ("K", "Q", "M"):
            mapping[f"{td}.{head}.weight"] = f"global.{td}<{rh}>.weight"
            mapping[f"{td}.{head}.bias"] = f"global.{td}<{rh}>.bias"
        mapping[f"Mu.{head}"] = f"global.Mu<{rh}>.weight"
    mapping["A_LIN.weight"] = "global.A_LIN.weight"
    mapping["A_LIN.bias"] = "global.A_LIN.bias"
    mapping["classifier.weight"] = "global.Classifier.weight"
    mapping["classifier.bias"] = "global.Classifier.bias"
    return mapping

def level2_hgt1():
    print()
    print(DIVIDER)
    print(f"LEVEL 2: HGT single-layer (d={d_hgt}, h={h_hgt})")
    print(DIVIDER)

    # -- Create models with identical initial weights --------------------------
    full_seed(42)
    pt_model = PyTorchHGT1()

    session = Session(db=relnn_db)
    session.run(f"""
#lang:relnn
d = {d_hgt} .
h = {h_hgt} .
K<i> = Linear(d, d/h) .
Q<i> = Linear(d, d/h) .
M<i> = Linear(d, d/h) .
Mu<i> = Tensor(1) .
A_LIN = Linear(d, d) .
Classifier = Linear(d, {n_classes}) .
Papers_Emb(s; Linear({in_features}, d, False)(z)) :- Papers(s;z) .
ATT_Head<i>(s,t; K<i>(z1) * Q<i>(z2) * Mu<i>) :-
    Papers_Emb(s;z1), Citation(s, t; w), Papers_Emb(t;z2) .
ATT_Con(s,t; Concat(z1, z2, z3)) :- ATT_Head<1>(s,t;z1), ATT_Head<2>(s,t;z2), ATT_Head<3>(s,t;z3) .
MSG_Src(s,t; z) :- Papers_Emb(s;z), Citation(s, t; w) .
MSG_Head<i>(s,t; M<i>(z)) :- MSG_Src(s,t;z) .
MSG(s,t; Concat(z1, z2, z3)) :- MSG_Head<1>(s,t;z1), MSG_Head<2>(s,t;z2), MSG_Head<3>(s,t;z3) .
AGG_MSG(t; sum(z2 * z1)) :- MSG(s,t; z2), ATT_Con(s,t; z1) .
Output(t; A_LIN(ReLU(z1)) + z2) :- AGG_MSG(t; z1), Papers_Emb(t; z2) .
""")
    session.run(f"""
#lang:relnn
?pred _Init(t; Classifier(z)) :- Output(t; z) .
""")

    mapping = _build_pt_to_relnn_map_hgt1()
    cmp = SessionComparison("HGT1-Level2", verbose=True)
    cmp.set_pyg_model(pt_model)
    cmp.set_relnn_session(session)
    cmp.set_mapping(mapping)
    cmp.sync_weights()
    print(f"  Weight sync: PyTorch -> RelNN (identical init)")

    # -- Train PyTorch ---------------------------------------------------------
    print("  [PyTorch (synced init)]")
    full_seed(42)
    t0 = time.perf_counter()
    pt_params, pt_losses, pt_acc = _train_eval(
        pt_model, lambda m: m(x, src_all, dst_all)[1])
    pt_time = time.perf_counter() - t0
    print(f"    Params: {pt_params}, Test acc: {pt_acc:.1%}, Time: {pt_time:.1f}s")

    # -- Train RelNN (synced init) ---------------------------------------------
    print("  [RelNN Templated (synced init)]")
    full_seed(42)
    t0 = time.perf_counter()
    session.run(f"""
#lang:relnn
?fit <epochs=100, lr=0.01, weight_decay=0.0005>
Loss(; CrossEntropyLoss()(Classifier(z_pred), z)) :- Output(target_id; z_pred), Labels(target_id; z) .
""")
    rn_time = time.perf_counter() - t0

    cmp.assert_param_count()
    rn_params = sum(p.numel() for p in session.engine.parameter_store.values())
    pred = session.run("""
#lang:relnn
?pred Cls(t; ArgMax()(Classifier(z))) :- Output(t; z) .
""")
    rn_acc = evaluate_node_classification(relnn_data, pred, return_value=True)
    print(f"    Params: {rn_params}, Test acc: {rn_acc:.1%}, Time: {rn_time:.1f}s")

    print(f"  Accuracy -- PyTorch: {pt_acc:.1%}, RelNN: {rn_acc:.1%}, Delta: {abs(pt_acc - rn_acc):.1%}")
    print(f"  Training time -- PyTorch: {pt_time:.1f}s, RelNN: {rn_time:.1f}s, Ratio: {rn_time/pt_time:.1f}x")
    assert abs(pt_acc - rn_acc) < 0.02, f"Accuracy gap >2%: PyTorch={pt_acc:.1%}, RelNN={rn_acc:.1%}"
    print("  PASS")
    return pt_acc, rn_acc, pt_time, rn_time

# =============================================================================
# Level 3: HGT two-layer
# =============================================================================

class PyTorchHGT2(nn.Module):
    def __init__(self):
        super().__init__()
        dh = d_hgt // h_hgt
        self.papers_emb = nn.Linear(in_features, d_hgt, bias=False)
        self.K = nn.ModuleList([nn.ModuleList([nn.Linear(d_hgt, dh) for _ in range(h_hgt)]) for _ in range(2)])
        self.Q = nn.ModuleList([nn.ModuleList([nn.Linear(d_hgt, dh) for _ in range(h_hgt)]) for _ in range(2)])
        self.M = nn.ModuleList([nn.ModuleList([nn.Linear(d_hgt, dh) for _ in range(h_hgt)]) for _ in range(2)])
        self.Mu = nn.ParameterList([nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(h_hgt)]) for _ in range(2)])
        self.A_LIN = nn.ModuleList([nn.Linear(d_hgt, d_hgt) for _ in range(2)])
        self.classifier = nn.Linear(d_hgt, n_classes)

    def _hgt_layer(self, z, edge_src, edge_dst, layer_idx):
        att_heads = []
        for i in range(h_hgt):
            att_heads.append(self.K[layer_idx][i](z[edge_src]) * self.Q[layer_idx][i](z[edge_dst]) * self.Mu[layer_idx][i])
        att_con = torch.cat(att_heads, dim=-1)
        head_agg = torch.zeros(N, d_hgt, device=z.device)
        head_agg.index_add_(0, edge_dst, att_con)
        msg_heads = [self.M[layer_idx][i](z[edge_src]) for i in range(h_hgt)]
        msg = torch.cat(msg_heads, dim=-1)
        agg = torch.zeros(N, d_hgt, device=z.device)
        agg.index_add_(0, edge_dst, msg)
        out = self.A_LIN[layer_idx](F.relu(agg + head_agg)) + z
        return out

    def forward(self, x_feat, edge_src, edge_dst):
        z = self.papers_emb(x_feat)
        z = self._hgt_layer(z, edge_src, edge_dst, 0)
        z = self._hgt_layer(z, edge_src, edge_dst, 1)
        return z, self.classifier(z)

HGT2_DSL = f"""
#lang:relnn
d = {d_hgt} .
h = {h_hgt} .

K<layer, head> = Linear(d, d/h) .
Q<layer, head> = Linear(d, d/h) .
M<layer, head> = Linear(d, d/h) .
Mu<layer, head> = Tensor(1) .
A_LIN<layer> = Linear(d, d) .
Classifier = Linear(d, {n_classes}) .

Papers_Emb(s; Linear({in_features}, d, False)(z)) :- Papers(s;z) .

def HGTLayer<layer>(InputNodes, Edges):
    Head<head>(s,t; K<layer,head>(z1) * Q<layer,head>(z2) * Mu<layer,head>) :- InputNodes(s;z1), Edges(s, t; w), InputNodes(t;z2) .
    Head_Con(s,t; Concat(z1, z2, z3)) :- Head<1>(s,t;z1), Head<2>(s,t;z2), Head<3>(s,t;z3) .
    Head_Agg(t; sum(z)) :- Head_Con(s,t; z) .

    MSG_Src(s,t; z) :- InputNodes(s;z), Edges(s, t; w) .
    MSG_Head<head>(s,t; M<layer,head>(z)) :- MSG_Src(s,t;z) .
    MSG(s,t; Concat(z1, z2, z3)) :- MSG_Head<1>(s,t;z1), MSG_Head<2>(s,t;z2), MSG_Head<3>(s,t;z3) .
    AGG_MSG(t; sum(z)) :- MSG(s,t; z) .

    Out(t; A_LIN<layer>(ReLU(z1 + z3)) + z2) :- AGG_MSG(t; z1), InputNodes(t; z2), Head_Agg(t; z3) .
enddef

L1(t; z) :- HGTLayer<1>(Papers_Emb, Citation)(t; z) .
Output(t; z) :- HGTLayer<2>(L1, Citation)(t; z) .
"""

def level3_hgt2():
    print()
    print(DIVIDER)
    print(f"LEVEL 3: HGT two-layer (d={d_hgt}, h={h_hgt})")
    print(DIVIDER)

    # -- Create models with identical initial weights --------------------------
    full_seed(42)
    pt_model = PyTorchHGT2()

    session = Session(db=relnn_db)
    session.run(HGT2_DSL)
    session.run(f"""
#lang:relnn
?pred _Init(t; Classifier(z)) :- Output(t; z) .
""")

    mapping = _build_pt_to_relnn_map_hgt2()
    mapping["classifier.weight"] = "global.Classifier.weight"
    mapping["classifier.bias"] = "global.Classifier.bias"
    cmp = SessionComparison("HGT2-Level3", verbose=True)
    cmp.set_pyg_model(pt_model)
    cmp.set_relnn_session(session)
    cmp.set_mapping(mapping)
    cmp.sync_weights()
    print(f"  Weight sync: PyTorch -> RelNN (identical init)")

    # -- Train PyTorch ---------------------------------------------------------
    print("  [PyTorch (synced init)]")
    full_seed(42)
    t0 = time.perf_counter()
    pt_params, pt_losses, pt_acc = _train_eval(
        pt_model, lambda m: m(x, src_all, dst_all)[1])
    pt_time = time.perf_counter() - t0
    print(f"    Params: {pt_params}, Test acc: {pt_acc:.1%}, Time: {pt_time:.1f}s")

    # -- Train RelNN (synced init) ---------------------------------------------
    print("  [RelNN Templated (synced init)]")
    full_seed(42)
    t0 = time.perf_counter()
    session.run(f"""
#lang:relnn
?fit <epochs=100, lr=0.01, weight_decay=0.0005>
Loss(; CrossEntropyLoss()(Classifier(z_pred), z)) :- Output(target_id; z_pred), Labels(target_id; z) .
""")
    rn_time = time.perf_counter() - t0

    cmp.assert_param_count()
    rn_params = sum(p.numel() for p in session.engine.parameter_store.values())
    pred = session.run("""
#lang:relnn
?pred Cls(t; ArgMax()(Classifier(z))) :- Output(t; z) .
""")
    rn_acc = evaluate_node_classification(relnn_data, pred, return_value=True)
    print(f"    Params: {rn_params}, Test acc: {rn_acc:.1%}, Time: {rn_time:.1f}s")

    print(f"  Accuracy -- PyTorch: {pt_acc:.1%}, RelNN: {rn_acc:.1%}, Delta: {abs(pt_acc - rn_acc):.1%}")
    print(f"  Training time -- PyTorch: {pt_time:.1f}s, RelNN: {rn_time:.1f}s, Ratio: {rn_time/pt_time:.1f}x")
    assert abs(pt_acc - rn_acc) < 0.02, (
        f"Accuracy gap >2% despite synced init: PyTorch={pt_acc:.1%}, RelNN={rn_acc:.1%}")
    print("  PASS")
    return pt_acc, rn_acc, pt_time, rn_time

# =============================================================================
# Level 3 debug: weight-synced forward-pass comparison
# =============================================================================

def _build_pt_to_relnn_map_hgt2():
    """Build mapping from PyTorch HGT2 param names to RelNN FQN keys."""
    mapping = {}
    mapping["papers_emb.weight"] = "global.transformation_Papers_Emb._module.weight"
    for layer in range(2):
        for head in range(h_hgt):
            rl, rh = layer + 1, head + 1
            for td in ("K", "Q", "M"):
                mapping[f"{td}.{layer}.{head}.weight"] = f"global.{td}<{rl},{rh}>.weight"
                mapping[f"{td}.{layer}.{head}.bias"] = f"global.{td}<{rl},{rh}>.bias"
            mapping[f"Mu.{layer}.{head}"] = f"global.Mu<{rl},{rh}>.weight"
        mapping[f"A_LIN.{layer}.weight"] = f"global.A_LIN<{layer + 1}>.weight"
        mapping[f"A_LIN.{layer}.bias"] = f"global.A_LIN<{layer + 1}>.bias"
    return mapping

def _align_relnn_output(relnn_pred, pyg_out):
    """Align RelNN output rows to dense [0..N-1] order matching PyTorch."""
    rn_out = relnn_pred.embeddings[0]
    rn_df = relnn_pred.content
    if rn_df is not None and "target_id" in rn_df.columns:
        rn_ids = rn_df["target_id"].values
    elif rn_df is not None and len(rn_df.columns) >= 1:
        rn_ids = rn_df.iloc[:, 0].values
    else:
        rn_ids = list(range(len(rn_out)))

    rn_aligned = torch.zeros_like(pyg_out)
    for pos, nid in enumerate(rn_ids):
        rn_aligned[int(nid)] = rn_out[pos]
    return rn_aligned

def debug_hgt2_forward():
    """Weight-synced forward comparison using SessionComparison."""
    print()
    print(DIVIDER)
    print("LEVEL 3 DEBUG: weight-synced forward-pass comparison")
    print(DIVIDER)

    full_seed(99)
    pt_model = PyTorchHGT2()
    pt_model.eval()

    full_seed(99)
    session = Session(db=relnn_db)
    session.run(HGT2_DSL)

    # Force parameter compilation
    session.run("""
#lang:relnn
?pred InitPred(t; z) :- Output(t; z) .
""")

    cmp = SessionComparison("HGT2-Forward", verbose=True)
    cmp.set_pyg_model(pt_model)
    cmp.set_relnn_session(session)
    cmp.print_params()

    mapping = _build_pt_to_relnn_map_hgt2()
    cmp.set_mapping(mapping)
    cmp.print_mapping()
    cmp.sync_weights()

    result = cmp.compare_forward(
        pyg_fn=lambda m: m(x, src_all, dst_all)[0],
        relnn_pred_dsl="""
#lang:relnn
?pred PredOut(t; z) :- Output(t; z) .
""",
        align_fn=_align_relnn_output,
        tolerance=1e-5,
    )

    assert result.passed, f"HGT 2-layer weight-synced forward FAILED: max_diff={result.max_diff:.2e}"
    return result.passed

# =============================================================================
# Run all levels
# =============================================================================

if __name__ == "__main__":
    results = {}
    timings = {}

    hr_acc, pyg_acc, rn_acc, hr_t, pyg_t, rn_t = level1_gcn()
    results["gcn_hr"] = (hr_acc, rn_acc)
    results["gcn_pyg"] = (pyg_acc, rn_acc)
    timings["gcn"] = {"hr": hr_t, "pyg": pyg_t, "relnn": rn_t}

    gcn_fwd_ok = debug_gcn_forward()

    pt_acc2, rn_acc2, pt_t2, rn_t2 = level2_hgt1()
    results["hgt1"] = (pt_acc2, rn_acc2)
    timings["hgt1"] = {"ref": pt_t2, "relnn": rn_t2}

    pt_acc3, rn_acc3, pt_t3, rn_t3 = level3_hgt2()
    results["hgt2_train"] = (pt_acc3, rn_acc3)
    timings["hgt2"] = {"ref": pt_t3, "relnn": rn_t3}

    fwd_ok = debug_hgt2_forward()

    print()
    print(DIVIDER)
    print("SUMMARY")
    print(DIVIDER)
    for name, (pt, rn) in results.items():
        status = "OK" if abs(pt - rn) < 0.05 else "DIFF"
        print(f"  {name:12s}: Ref={pt:.1%}  RelNN={rn:.1%}  Delta={abs(pt-rn):.1%}  [{status}]")
    gcn_fwd_status = "OK" if gcn_fwd_ok else "FAIL"
    print(f"  {'gcn_fwd':12s}: weight-synced forward comparison  [{gcn_fwd_status}]")
    fwd_status = "OK" if fwd_ok else "FAIL"
    print(f"  {'hgt2_fwd':12s}: weight-synced forward comparison  [{fwd_status}]")

    print()
    print("  TRAINING TIME (100 epochs)")
    print(f"  {'GCN':12s}: HR={timings['gcn']['hr']:.1f}s  PyG={timings['gcn']['pyg']:.1f}s  RelNN={timings['gcn']['relnn']:.1f}s  (RelNN/HR={timings['gcn']['relnn']/timings['gcn']['hr']:.1f}x)")
    print(f"  {'HGT-1L':12s}: Ref={timings['hgt1']['ref']:.1f}s  RelNN={timings['hgt1']['relnn']:.1f}s  (RelNN/Ref={timings['hgt1']['relnn']/timings['hgt1']['ref']:.1f}x)")
    print(f"  {'HGT-2L':12s}: Ref={timings['hgt2']['ref']:.1f}s  RelNN={timings['hgt2']['relnn']:.1f}s  (RelNN/Ref={timings['hgt2']['relnn']/timings['hgt2']['ref']:.1f}x)")

    assert gcn_fwd_ok, "GCN weight-synced forward comparison FAILED"
    assert fwd_ok, "HGT 2-layer weight-synced forward comparison FAILED"

    print()
    print("All comparison levels completed. GCN and HGT architectures verified.")
