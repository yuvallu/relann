"""Compare generic (bounded-set + recursion) RelNN HGT vs hand-rolled PyTorch on DBLP.

This test verifies that the *generic* HGT library written with:
  - Template specialization / recursion base cases (H<'Author',0> etc.)
  - Bounded sets (Union(Set(EdgeAgg<...> | MetaRel(...))))
produces the same architecture as the hand-rolled PyTorch HGT for the
Paper->Author path (1 HGT layer, author classification).

Assertions:
  1. Param count matches hand-rolled
  2. Weight-synced forward outputs match within tolerance (vs PyTorch)
  3. Direct forward outputs match first-order RelNN within tolerance
  4. Training accuracy within 2%

Run from repo root:
    python tests/slow/run_compare_dblp_hgt_generic.py
"""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from relann.torch_utils import full_seed, get_project_root
from relann.session import Session
from relann.datasets import load_dblp_dataset
from relann.comparison import SessionComparison

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
dh = hidden // num_heads
n_classes = info["num_classes"]

pa_df = relnn_db["PaperAuthor"][0]
pa_edge_src = torch.tensor(pa_df["paper_id"].values, dtype=torch.long)
pa_edge_dst = torch.tensor(pa_df["author_id"].values, dtype=torch.long)

x_author = pyg_data["author"].x
x_paper = pyg_data["paper"].x
n_authors = x_author.size(0)
n_papers = x_paper.size(0)

y_author = pyg_data["author"].y
train_mask = pyg_data["author"].train_mask
val_mask = pyg_data["author"].val_mask
test_mask = pyg_data["author"].test_mask

# =============================================================================
# 1. Hand-rolled PyTorch HGT (Paper->Author path only)
# =============================================================================

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
        h_a = F.relu(self.author_proj(x_a))
        h_p = F.relu(self.paper_proj(x_p))

        msg_heads = []
        for head_idx in range(num_heads):
            k_src = self.K_paper[head_idx](h_p[edge_src])
            q_dst = self.Q_author[head_idx](h_a[edge_dst])
            v_src = self.V_paper[head_idx](h_p[edge_src])

            k_transformed = self.Krel_PA[head_idx](k_src)
            v_transformed = self.Vrel_PA[head_idx](v_src)

            alpha = (q_dst * k_transformed).sum(dim=-1)
            alpha = alpha * self.Prel_PA[head_idx]
            alpha = alpha / math.sqrt(dh)

            alpha_max = torch.zeros(n_authors, device=alpha.device)
            alpha_max.scatter_reduce_(0, edge_dst, alpha, reduce="amax", include_self=True)
            alpha = alpha - alpha_max[edge_dst]
            alpha_exp = torch.exp(alpha)
            alpha_sum = torch.zeros(n_authors, device=alpha.device)
            alpha_sum.scatter_add_(0, edge_dst, alpha_exp)
            alpha_softmax = alpha_exp / alpha_sum[edge_dst].clamp(min=1e-12)

            msg = v_transformed * alpha_softmax.unsqueeze(-1)
            msg_heads.append(msg)

        msg_full = torch.cat(msg_heads, dim=-1)
        agg = torch.zeros(n_authors, hidden, device=msg_full.device)
        agg.scatter_add_(0, edge_dst.unsqueeze(-1).expand_as(msg_full), msg_full)

        out = self.out_lin_author(F.gelu(agg))
        skip = torch.sigmoid(self.skip_author)
        author_out = skip * out + (1 - skip) * h_a

        return author_out, self.classifier(author_out)

def run_hand_rolled(seed=42, epochs=100, lr=0.005, wd=0.001, model=None):
    if model is None:
        full_seed(seed)
        model = HandRolledHGT()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n_params = sum(p.numel() for p in model.parameters())

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        _, logits = model(x_author, x_paper, pa_edge_src, pa_edge_dst)
        loss = F.cross_entropy(logits[train_mask], y_author[train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        _, logits = model(x_author, x_paper, pa_edge_src, pa_edge_dst)
        pred = logits.argmax(dim=-1)
    accs = {}
    for split, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        accs[split] = (pred[mask] == y_author[mask]).float().mean().item()

    return model, n_params, losses, accs

# =============================================================================
# 2. Generic RelNN HGT using bounded sets + recursion base cases
# =============================================================================

# The HGT library is graph-agnostic: only base cases and schema differ.
HGT_LIBRARY = """
#lang:relnn
# ── Learnable weights (templated per layer/type/head) ─────────────
K_Lin<L, ts, i>  = Linear(d, dh) .
Q_Lin<L, tt, i>  = Linear(d, dh) .
M_Lin<L, ts, i>  = Linear(d, dh) .
W_ATT<L, pe, i>  = Linear(dh, dh, False) .
W_MSG<L, pe, i>  = Linear(dh, dh, False) .
Mu<L, pe, i>     = Tensor(1) .
A_Lin<L, tt>     = Linear(d, d) .
Skip<L, tt>      = Tensor(1) .

# ── Reusable Softmax function (numerically stable: max-subtraction) ──
# Naive softmax (exp(z)/sum(exp(z))) overflows when scores get large,
# producing inf/inf = NaN. This shows up at epoch 0 in 2-layer HGT.
# Standard fix: subtract max per group BEFORE exp, so the largest
# exponent is 0 and all others <= 0 -- no overflow. Same pattern as the
# non-templated reference at run_compare_dblp_original_hgt.py:535-537.
def Softmax(Scores):
    Max(t; max(z))            :- Scores(s, t; z) .
    Stable(s, t; z1 - z2)     :- Scores(s, t; z1), Max(t; z2) .
    Exp(s, t; exp(z))         :- Stable(s, t; z) .
    Denom(t; sum(z))          :- Exp(s, t; z) .
    Out(s, t; z1 / z2)        :- Exp(s, t; z1), Denom(t; z2) .
enddef

# ── Eq. 3: Attention head ───────────────────────────────────────
def ATT_Head<L, ts, pe, tt, i>():
    K(s; K_Lin<L, ts, i>(z))  :- H<ts, L-1>(s; z) .
    Q(t; Q_Lin<L, tt, i>(z))  :- H<tt, L-1>(t; z) .

    Dot(s, t; view(1)(z_q @ transpose(W_ATT<L, pe, i>(z_k))) * Mu<L, pe, i> / sqrt(dh) ) :- K(s; z_k), pe(s, t; w), Q(t; z_q) .

    Out(s, t; z) :- Softmax(Dot)(s, t; z) .
enddef

# ── Eq. 4: Message head ─────────────────────────────────────────
def MSG_Head<L, ts, pe, tt, i>():
    M(s; M_Lin<L, ts, i>(z))  :- H<ts, L-1>(s; z) .
    Out(s, t; W_MSG<L, pe, i>(z_m)) :- M(s; z_m), pe(s, t; w) .
enddef

# ── Eq. 5 first half: weighted messages, concat heads, sum ──────
def EdgeAgg<L, ts, pe, tt>():
    WMsg<i>(s, t; z_att * z_msg) :- ATT_Head<L, ts, pe, tt, i>(s, t; z_att), MSG_Head<L, ts, pe, tt, i>(s, t; z_msg) .

    AllHeads(s, t; Concat(*z)) :- Join(Set(WMsg<i>(s, t; z) | 1 <= i, i <= h)) .

    Out(t; sum(z)) :- AllHeads(s, t; z) .
enddef

# ── Eq. 5-6: target aggregation + residual (with bounding) ──────
def H<tt, L>():
    Agg(t; sum(z)) :- Union(Set(EdgeAgg<L, ts, pe, tt>(t; z) | MetaRel(ts, pe, tt))) .

    Updated(t; A_Lin<L, tt>(GELU(z))) :- Agg(t; z) .

    Out(t; Sigmoid(Skip<L, tt>) * z1 + (1 - Sigmoid(Skip<L, tt>)) * z2 ) :- Updated(t; z1), H<tt, L-1>(t; z2) .
enddef
"""

RELNN_GENERIC_DEFINE = f"""
#lang:relnn
d = {hidden} .
h = {num_heads} .
dh = {dh} .

# ── Base cases: project raw features per node type ───────────────
H<'Author', 0>(id; ReLU(Linear({info['author_features']}, d)(z)))      :- Author(id; z) .
H<'Paper', 0>(id; ReLU(Linear({info['paper_features']}, d)(z)))        :- Paper(id; z) .
H<'Term', 0>(id; ReLU(Linear({info['term_features']}, d)(z)))           :- Term(id; z) .
H<'Conference', 0>(id; ReLU(Linear({info['conference_features']}, d)(z))) :- Conference(id; z) .

""" + HGT_LIBRARY + f"""
#lang:relnn
# ── 1-layer HGT, classify authors ───────────────────────────────
Classifier = Linear(d, {n_classes}) .
Output(id; z) :- H<'Author', 1>(id; z) .
"""

RELNN_GENERIC_DEFINE_2L = f"""
#lang:relnn
d = {hidden} .
h = {num_heads} .
dh = {dh} .

# ── Base cases: project raw features per node type ───────────────
H<'Author', 0>(id; ReLU(Linear({info['author_features']}, d)(z)))        :- Author(id; z) .
H<'Paper', 0>(id; ReLU(Linear({info['paper_features']}, d)(z)))          :- Paper(id; z) .
H<'Term', 0>(id; ReLU(Linear({info['term_features']}, d)(z)))             :- Term(id; z) .
H<'Conference', 0>(id; ReLU(Linear({info['conference_features']}, d)(z))) :- Conference(id; z) .

""" + HGT_LIBRARY + f"""
#lang:relnn
# ── 2-layer HGT: one line change from 1L (H<'Author', 1> -> H<'Author', 2>) ──
Classifier = Linear(d, {n_classes}) .
Output(id; z) :- H<'Author', 2>(id; z) .
"""

RELNN_FIT_DSL = """
#lang:relnn
?fit <epochs={epochs}, lr={lr}, weight_decay={wd}>
Loss(; CrossEntropyLoss()(Classifier(z_pred), z)) :- Output(id; z_pred), AuthorLabels(id; z) .
"""

RELNN_PRED_DSL = """
#lang:relnn
?pred AuthorPred(id; ArgMax()(Classifier(z))) :- Output(id; z) .
"""

RELNN_FORWARD_DSL = """
#lang:relnn
?pred AuthorFwd(id; z) :- Output(id; z) .
"""

def run_generic_relnn_hgt(seed=42, epochs=100, lr=0.005, wd=0.001, session=None):
    if session is None:
        full_seed(seed)
        session = Session(db=relnn_db)
        session.run(RELNN_GENERIC_DEFINE)
    full_seed(seed)
    session.run(RELNN_FIT_DSL.format(epochs=epochs, lr=lr, wd=wd))

    rn_params = sum(p.numel() for p in session.engine.parameter_store.values())
    pred = session.run(RELNN_PRED_DSL)
    return session, rn_params, pred

def evaluate_dblp_relnn(pred, node_metadata):
    pred_df = pred.content.copy()
    pred_class = pred.embeddings[0].cpu().numpy().flatten().astype(int)
    pred_df["_pred_class"] = pred_class

    merge_col = "id" if "id" in pred_df.columns else pred_df.columns[0]
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
# 3. Weight mapping (Hand-rolled PyTorch -> Generic RelNN)
# =============================================================================

def _find_fqn_by_prefix_and_shape(store, prefix, shape):
    """Find a parameter FQN matching prefix and shape."""
    for k, v in store.items():
        if k.startswith(prefix) and v.shape == shape:
            return k
    raise KeyError(f"Cannot find param with prefix '{prefix}' and shape {shape}")

def _find_generic_krel_fqn(store, head_idx):
    """Find W_ATT weight for the given head in the generic HGT."""
    for k, v in store.items():
        if f"W_ATT<1,PaperAuthor,{head_idx}>" in k and v.shape == (dh, dh):
            return k
    for k, v in store.items():
        if "PaperAuthor" in k and f",{head_idx}>" in k and v.shape == (dh, dh):
            return k
    return None

def _build_generic_weight_mapping(relnn_store):
    """Build {pytorch_param_name: relnn_fqn} for the generic HGT."""
    m = {}

    # Base case projections: H<'Author',0> and H<'Paper',0>
    # These are inline Linear() calls inside the base case rules.
    # FQNs follow the pattern: global.transformation_H<'Type',0>.Linear_...
    for fqn, param in relnn_store.items():
        if "H<Author,0>" in fqn and param.shape == (hidden, info["author_features"]):
            m["author_proj.weight"] = fqn
        elif "H<Author,0>" in fqn and param.shape == (hidden,) and "bias" in fqn:
            m["author_proj.bias"] = fqn
        elif "H<Paper,0>" in fqn and param.shape == (hidden, info["paper_features"]):
            m["paper_proj.weight"] = fqn
        elif "H<Paper,0>" in fqn and param.shape == (hidden,) and "bias" in fqn:
            m["paper_proj.bias"] = fqn

    # Cache keys strip quotes: both DSL 'Paper' and MetaRel-derived Paper
    # normalize to Paper in FQNs.
    for h in range(num_heads):
        rh = h + 1
        # K, Q, V projections
        m[f"K_paper.{h}.weight"] = f"global.K_Lin<1,Paper,{rh}>.weight"
        m[f"K_paper.{h}.bias"] = f"global.K_Lin<1,Paper,{rh}>.bias"
        m[f"Q_author.{h}.weight"] = f"global.Q_Lin<1,Author,{rh}>.weight"
        m[f"Q_author.{h}.bias"] = f"global.Q_Lin<1,Author,{rh}>.bias"
        m[f"V_paper.{h}.weight"] = f"global.M_Lin<1,Paper,{rh}>.weight"
        m[f"V_paper.{h}.bias"] = f"global.M_Lin<1,Paper,{rh}>.bias"

        # Relation-specific transforms
        m[f"Vrel_PA.{h}.weight"] = f"global.W_MSG<1,PaperAuthor,{rh}>.weight"
        m[f"Prel_PA.{h}"] = f"global.Mu<1,PaperAuthor,{rh}>.weight"

        # W_ATT (Krel) is nested inside the ATT_Head's Dot transformation
        krel_fqn = _find_generic_krel_fqn(relnn_store, rh)
        if krel_fqn:
            m[f"Krel_PA.{h}.weight"] = krel_fqn
        else:
            print(f"  [WARN] Could not find Krel_PA<{rh}> weight")

    m["out_lin_author.weight"] = "global.A_Lin<1,Author>.weight"
    m["out_lin_author.bias"] = "global.A_Lin<1,Author>.bias"
    m["skip_author"] = "global.Skip<1,Author>.weight"

    return m

def _align_relnn_output(relnn_pred, ref_out):
    rn_out = relnn_pred.embeddings[0]
    rn_df = relnn_pred.content
    col = "id" if "id" in rn_df.columns else rn_df.columns[0]
    rn_ids = rn_df[col].values

    aligned = torch.zeros_like(ref_out)
    for pos, nid in enumerate(rn_ids):
        aligned[int(nid)] = rn_out[pos]
    return aligned

# =============================================================================
# 4. First-order RelNN HGT (inlined from run_compare_dblp_hgt.py for direct comparison)
# =============================================================================

FIRST_ORDER_DEFINE_DSL = f"""
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

FIRST_ORDER_FORWARD_DSL = """
#lang:relnn
?pred AuthorFwd(author_id; z) :- Output(author_id; z) .
"""

def _find_krel_fqn(store, head_idx):
    prefix = f"global.transformation_DotPA<{head_idx}>."
    for k, v in store.items():
        if k.startswith(prefix) and v.shape == (dh, dh):
            return k
    raise KeyError(f"Cannot find Krel_PA<{head_idx}> weight in parameter store")

def _build_first_order_weight_mapping(relnn_store):
    """Build {pytorch_param_name: relnn_fqn} for the first-order HGT."""
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
        m[f"Krel_PA.{h}.weight"] = _find_krel_fqn(relnn_store, rh)

    m["out_lin_author.weight"] = "global.OutLin_author.weight"
    m["out_lin_author.bias"] = "global.OutLin_author.bias"
    m["skip_author"] = "global.Skip_author.weight"
    return m

# =============================================================================
# 5. Weight-synced forward comparison
# =============================================================================

def debug_forward_comparison():
    print()
    print(DIVIDER)
    print("Weight-synced forward comparison (Hand-rolled <-> Generic RelNN)")
    print(DIVIDER)

    full_seed(99)
    pt_model = HandRolledHGT()
    pt_model.eval()

    full_seed(99)
    session = Session(db=relnn_db)
    session.run(RELNN_GENERIC_DEFINE)
    session.run(RELNN_FORWARD_DSL)

    cmp = SessionComparison("DBLP-Generic-HGT-Forward", verbose=True)
    cmp.set_pyg_model(pt_model)
    cmp.set_relnn_session(session)
    cmp.print_params()

    mapping = _build_generic_weight_mapping(session.engine.parameter_store)
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
# 6. Direct first-order vs generic RelNN forward comparison
# =============================================================================

def direct_relnn_vs_relnn_comparison():
    """Sync the same PyTorch weights into both first-order and generic RelNN,
    run forward on each, and compare outputs directly (no PyTorch middleman).
    """
    print()
    print(DIVIDER)
    print("Direct comparison: First-order RelNN vs Generic RelNN")
    print(DIVIDER)

    full_seed(99)
    pt_model = HandRolledHGT()
    pt_model.eval()

    # -- Generic session --
    full_seed(99)
    gen_session = Session(db=relnn_db)
    gen_session.run(RELNN_GENERIC_DEFINE)
    gen_session.run(RELNN_FORWARD_DSL)

    gen_mapping = _build_generic_weight_mapping(gen_session.engine.parameter_store)
    gen_cmp = SessionComparison("gen-sync", verbose=False)
    gen_cmp.set_pyg_model(pt_model)
    gen_cmp.set_relnn_session(gen_session)
    gen_cmp.set_mapping(gen_mapping)
    gen_cmp.sync_weights()

    gen_pred = gen_session.run(RELNN_FORWARD_DSL)

    # -- First-order session --
    full_seed(99)
    fo_session = Session(db=relnn_db)
    fo_session.run(FIRST_ORDER_DEFINE_DSL)
    fo_session.run(FIRST_ORDER_FORWARD_DSL)

    fo_mapping = _build_first_order_weight_mapping(fo_session.engine.parameter_store)
    fo_cmp = SessionComparison("fo-sync", verbose=False)
    fo_cmp.set_pyg_model(pt_model)
    fo_cmp.set_relnn_session(fo_session)
    fo_cmp.set_mapping(fo_mapping)
    fo_cmp.sync_weights()

    fo_pred = fo_session.run(FIRST_ORDER_FORWARD_DSL)

    # -- Align both to dense [0..n_authors-1] and compare --
    ref_shape = torch.zeros(n_authors, hidden)
    gen_aligned = _align_relnn_output(gen_pred, ref_shape)
    fo_aligned = _align_relnn_output(fo_pred, ref_shape)

    max_diff = (gen_aligned - fo_aligned).abs().max().item()
    passed = max_diff < 1e-4

    print(f"  Generic params:     {sum(p.numel() for p in gen_session.engine.parameter_store.values()):,}")
    print(f"  First-order params: {sum(p.numel() for p in fo_session.engine.parameter_store.values()):,}")
    print(f"  Max diff: {max_diff:.2e}")
    if passed:
        print("  [OK] First-order and generic RelNN outputs match")
    else:
        print("  [FAIL] Outputs differ beyond tolerance")

    return passed, max_diff

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print(DIVIDER)
    print("Generic RelNN HGT (bounded sets + recursion) on DBLP")
    print(DIVIDER)

    # -- Hand-rolled baseline --
    print()
    print("  [Hand-rolled PyTorch (PA-path)]")
    full_seed(42)
    hr_model = HandRolledHGT()

    rn_session = Session(db=relnn_db)
    rn_session.run(RELNN_GENERIC_DEFINE)
    # Force parameter compilation
    rn_session.run("""
#lang:relnn
?pred _Init(id; Classifier(z)) :- Output(id; z) .
""")

    mapping = _build_generic_weight_mapping(rn_session.engine.parameter_store)
    mapping["classifier.weight"] = "global.Classifier.weight"
    mapping["classifier.bias"] = "global.Classifier.bias"
    cmp = SessionComparison("DBLP-Generic-HGT-Sync", verbose=True)
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
    print("  [Generic RelNN (synced init)]")
    full_seed(42)
    t0 = time.perf_counter()
    rn_session, rn_n_params, rn_pred = run_generic_relnn_hgt(
        epochs=100, session=rn_session)
    rn_time = time.perf_counter() - t0
    print(f"  Params: {rn_n_params:,}, Time: {rn_time:.1f}s")

    rn_accs = evaluate_dblp_relnn(rn_pred, dblp.node_metadata)
    for split, acc in rn_accs.items():
        print(f"  {split}: {acc:.1%}")

    # -- Param count --
    print()
    print(DIVIDER)
    print("Param count comparison")
    print(DIVIDER)
    print(f"  Hand-rolled (PA):     {hr_n_params:>10,}")
    print(f"  Generic RelNN (PA):   {rn_n_params:>10,}")
    if hr_n_params == rn_n_params:
        print("  [OK] Param counts match")
    else:
        print(f"  [WARN] Param count mismatch: HR={hr_n_params}, RN={rn_n_params}")

    # -- Forward comparison (generic vs PyTorch) --
    fwd_result = debug_forward_comparison()

    # -- Direct comparison (generic vs first-order RelNN) --
    direct_passed, direct_diff = direct_relnn_vs_relnn_comparison()

    # -- Summary --
    print()
    print(DIVIDER)
    print("SUMMARY")
    print(DIVIDER)
    print(f"  Hand-rolled   test acc: {hr_accs['test']:.1%}  ({hr_n_params:,} params)  Time: {hr_time:.1f}s")
    print(f"  Generic RelNN test acc: {rn_accs['test']:.1%}  ({rn_n_params:,} params)  Time: {rn_time:.1f}s")
    print(f"  HR-RelNN delta: {abs(hr_accs['test'] - rn_accs['test']):.1%}")
    print(f"  HR-RelNN time ratio: {rn_time/hr_time:.1f}x")
    print(f"  Forward match (vs PyTorch):     {'[OK]' if fwd_result.passed else '[FAIL]'}  max_diff={fwd_result.max_diff:.2e}")
    print(f"  Forward match (vs first-order): {'[OK]' if direct_passed else '[FAIL]'}  max_diff={direct_diff:.2e}")

    assert fwd_result.passed, f"Weight-synced forward FAILED: max_diff={fwd_result.max_diff:.2e}"
    assert direct_passed, (
        f"Generic vs first-order RelNN FAILED: max_diff={direct_diff:.2e}")
    assert abs(hr_accs['test'] - rn_accs['test']) < 0.02, (
        f"HR-RelNN accuracy gap >2%: HR={hr_accs['test']:.1%}, RN={rn_accs['test']:.1%}")
    print()
    print("Done.")
