"""Verify 2-layer full-graph RelNN HGT gives identical accuracy to PyG when weights are synced.

Layer 1 updates Paper from {Author, Term, Conference} and Author from Paper.
Layer 2 uses those updated embeddings for the final Paper->Author attention.

This script:
  1. Initialises PyG 2L HGTConv with a fixed seed
  2. Creates the 2L full-path RelNN session (RELNN_DEFINE_DSL_2L from run_compare_dblp_original_hgt.py)
  3. Syncs all PyG weights -> RelNN
  4. Compares forward outputs  (expect max diff < 1e-5)
  5. Trains both from identical synced init, checks final accuracy per seed
  6. Repeats for 5 seeds

Run from repo root:
    python tests/slow/run_match_hgt_2l_accuracy.py
"""
from __future__ import annotations

import json
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

# Import the 2L DSL string from the main comparison script
from run_compare_dblp_original_hgt import (
    RELNN_DEFINE_DSL_2L,
    relnn_db,
    dblp,
    info,
    hidden,
    num_heads,
    n_classes,
    y_author,
    train_mask,
    val_mask,
    test_mask,
    pyg_data,
)

DIVIDER = "=" * 70
dh = hidden // num_heads

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using {DEVICE}")

y_author_dev   = y_author.to(DEVICE)
train_mask_dev = train_mask.to(DEVICE)
val_mask_dev   = val_mask.to(DEVICE)
test_mask_dev  = test_mask.to(DEVICE)
pyg_data_dev   = pyg_data.to(DEVICE)

# ============================================================================
# PyG 2L model
# ============================================================================

class PyGHGT2L(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_dict = nn.ModuleDict({
            nt: PyGLinear(-1, hidden) for nt in pyg_data.node_types
        })
        self.convs = nn.ModuleList([
            PyGHGTConv(hidden, hidden, pyg_data.metadata(), num_heads)
            for _ in range(2)
        ])
        self.lin = PyGLinear(hidden, n_classes)

    def forward(self):
        x_dict = {nt: self.lin_dict[nt](pyg_data_dev[nt].x).relu_()
                  for nt in pyg_data.node_types}
        for conv in self.convs:
            x_dict = conv(x_dict, pyg_data_dev.edge_index_dict)
        return self.lin(x_dict["author"])

def _sync_cuda():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

def _c(dst: torch.Tensor, src: torch.Tensor):
    dst.data.copy_(src.detach().to(dst.device))

# ============================================================================
# Diagnostic helpers
# ============================================================================

def print_pyg_conv_shapes(conv, label=""):
    print(f"  [{label} PyG conv parameters]")
    for name, param in conv.named_parameters():
        print(f"    {name}: {tuple(param.shape)}")

def print_relnn_param_keys(store, label=""):
    print(f"  [{label} RelNN parameter store]")
    for key, val in sorted(store.items()):
        print(f"    {key}: {tuple(val.shape)}")

def _get_prel_value(conv, edge_key_str: str, head: int) -> torch.Tensor:
    """Extract p_rel scalar for a given edge type string and head index."""
    p = conv.p_rel[edge_key_str]
    if p.dim() == 1:
        return p[head:head+1]
    elif p.dim() == 2:
        return p[0, head:head+1]
    else:
        return p[head:head+1]

# ============================================================================
# Weight sync: PyG 2L -> RelNN
# ============================================================================

def _kv_slice(w: torch.Tensor, b: torch.Tensor, part: str, head: int):
    """Extract K, Q, or V slice for a given head from a packed (3*hidden, hidden) weight."""
    offsets = {"K": 0, "Q": hidden, "V": 2 * hidden}
    off = offsets[part]
    lo, hi = head * dh, (head + 1) * dh
    w_slice = w[off + lo : off + hi]  # (dh, hidden)
    b_slice = b[off + lo : off + hi]  # (dh,)
    return w_slice, b_slice

def _sync_conv_to_relnn(
    conv,
    rn_store: dict,
    layer_label: str,
    node_types_to_sync: list,
    edge_types_to_sync: list,
):
    """Sync one HGTConv layer's weights into RelNN parameter store.

    layer_label: e.g. "L1" or "L2" (without leading underscore).
    node_types_to_sync: [(pyg_nt, relnn_prefix, parts)] where parts ⊆ ["K","Q","V"].
    edge_types_to_sync: [(pyg_et_key, relnn_et_name, p_rel_str)] for Krel/Vrel/Prel.
    """
    edge_types_map = conv.edge_types_map
    num_et = len(conv.edge_types)

    def cp(key, tensor):
        if key not in rn_store:
            raise KeyError(f"RelNN key not found: {key!r}")
        _c(rn_store[key], tensor)

    # KQV
    for nt, relnn_prefix, parts in node_types_to_sync:
        w = conv.kqv_lin.lins[nt].weight  # (3*hidden, hidden)
        b = conv.kqv_lin.lins[nt].bias    # (3*hidden,)
        for part in parts:
            for h in range(num_heads):
                rh = h + 1
                w_sl, b_sl = _kv_slice(w, b, part, h)
                cp(f"global.{part}_{relnn_prefix}_{layer_label}<{rh}>.weight", w_sl)
                cp(f"global.{part}_{relnn_prefix}_{layer_label}<{rh}>.bias",   b_sl)

    # Krel / Vrel / Prel per edge type
    for et_key, relnn_et_name, p_rel_str in edge_types_to_sync:
        et_offset = edge_types_map[et_key]
        for h in range(num_heads):
            rh = h + 1
            idx    = h * num_et + et_offset
            krel_w = conv.k_rel.weight[idx]        # (dh, dh): PyG computes k @ krel_w
            vrel_w = conv.v_rel.weight[idx]
            prel_v = _get_prel_value(conv, p_rel_str, h)

            # Krel is scoped under the Dot rule's transformation context.
            # Key pattern: global.transformation_Dot{et}_{layer}<{h}>.*.weight
            krel_prefix = f"global.transformation_Dot{relnn_et_name}_{layer_label}<{rh}>."
            krel_key = next(
                (k for k, v in rn_store.items()
                 if k.startswith(krel_prefix) and v.shape == (dh, dh)),
                None,
            )
            if krel_key is None:
                raise KeyError(
                    f"Krel for {relnn_et_name} {layer_label} head {rh} not found.\n"
                    f"Prefix: {krel_prefix!r}\nStore keys: {sorted(rn_store)}"
                )
            # PyG: k @ krel_w; RelNN Linear(bias=False): k @ W.T  =>  W = krel_w.T
            _c(rn_store[krel_key], krel_w.T)
            cp(f"global.Vrel_{relnn_et_name}_{layer_label}<{rh}>.weight", vrel_w.T)
            cp(f"global.Prel_{relnn_et_name}_{layer_label}<{rh}>.weight", prel_v)

    # out_lin and skip for target node types
    for nt, relnn_out_name in [("author", "author"), ("paper", "paper")]:
        out_w_key = f"global.OutLin_{relnn_out_name}_{layer_label}.weight"
        out_b_key = f"global.OutLin_{relnn_out_name}_{layer_label}.bias"
        skip_key  = f"global.Skip_{relnn_out_name}_{layer_label}.weight"
        if out_w_key not in rn_store:
            continue
        cp(out_w_key, conv.out_lin.lins[nt].weight)
        cp(out_b_key, conv.out_lin.lins[nt].bias)
        cp(skip_key,  conv.skip[nt].view(1))

# Edge type constants
_PA = (("paper", "to", "author"), "PA", "paper__to__author")
_AP = (("author", "to", "paper"), "AP", "author__to__paper")
_TP = (("term",   "to", "paper"), "TP", "term__to__paper")
_CP = (("conference", "to", "paper"), "CP", "conference__to__paper")

def sync_pyg_2l_to_relnn(pyg_model: PyGHGT2L, session: Session) -> None:
    """Sync all PyG 2L weights into the RelNN parameter store."""
    conv0 = pyg_model.convs[0]
    conv1 = pyg_model.convs[1]
    rn_store = session.engine.parameter_store

    # Input projections (all 4 node types)
    for nt, relnn_name in [
        ("author", "Author"), ("paper", "Paper"),
        ("term", "Term"), ("conference", "Conf"),
    ]:
        _c(rn_store[f"global.{relnn_name}Proj.weight"], pyg_model.lin_dict[nt].weight)
        _c(rn_store[f"global.{relnn_name}Proj.bias"],   pyg_model.lin_dict[nt].bias)

    # Layer 1: all 4 edge types; Author and Paper updated
    _sync_conv_to_relnn(
        conv0, rn_store, "L1",
        node_types_to_sync=[
            ("paper",       "paper",  ["K", "Q", "V"]),
            ("author",      "author", ["K", "Q", "V"]),
            ("term",        "term",   ["K", "V"]),
            ("conference",  "conf",   ["K", "V"]),
        ],
        edge_types_to_sync=[_PA, _AP, _TP, _CP],
    )

    # Layer 2: PA edge only; Author updated
    _sync_conv_to_relnn(
        conv1, rn_store, "L2",
        node_types_to_sync=[
            ("paper",  "paper",  ["K", "V"]),
            ("author", "author", ["Q"]),
        ],
        edge_types_to_sync=[_PA],
    )

    # Classifier
    _c(rn_store["global.Classifier.weight"], pyg_model.lin.weight)
    _c(rn_store["global.Classifier.bias"],   pyg_model.lin.bias)

# ============================================================================
# Evaluation helpers
# ============================================================================

RELNN_LOGITS_DSL = """
#lang:relnn
?pred AuthorLogits(author_id; Classifier(z)) :- Output(author_id; z) .
"""

RELNN_PRED_DSL = """
#lang:relnn
?pred AuthorPred(author_id; ArgMax()(Classifier(z))) :- Output(author_id; z) .
"""

RELNN_FIT_DSL = """
#lang:relnn
?fit <epochs={epochs}, lr={lr}, weight_decay={wd}>
Loss(; CrossEntropyLoss()(Classifier(z_pred), z)) :- Output(author_id; z_pred), AuthorLabels(author_id; z) .
"""

def _eval_pyg(model: PyGHGT2L) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model().argmax(-1)
    accs = {}
    for split, mask in [("train", train_mask_dev), ("val", val_mask_dev), ("test", test_mask_dev)]:
        accs[split] = (pred[mask] == y_author_dev[mask]).float().mean().item()
    return accs

def _eval_relnn(session: Session) -> dict:
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

def _align_relnn_logits(pred_result, ref_n_authors: int) -> torch.Tensor:
    rn_out = pred_result.embeddings[0].cpu()
    rn_df  = pred_result.content
    col    = "author_id" if "author_id" in rn_df.columns else rn_df.columns[0]
    rn_ids = rn_df[col].values
    aligned = torch.zeros(ref_n_authors, rn_out.shape[-1])
    for pos, nid in enumerate(rn_ids):
        aligned[int(nid)] = rn_out[pos]
    return aligned

def _diff(a: torch.Tensor, b: torch.Tensor, label: str) -> float:
    d = (a.float() - b.float()).abs()
    m, mu = float(d.max()), float(d.mean())
    print(f"  [{label}] max={m:.2e}  mean={mu:.2e}")
    return m

def _align_relnn_table(pred_result, n_rows: int, dim: int, id_col: str) -> torch.Tensor:
    """Align a RelNN table output (possibly sparse) into a dense (n_rows, dim) tensor."""
    out = torch.zeros(n_rows, dim)
    emb = pred_result.embeddings[0].cpu()
    ids = pred_result.content[id_col].values
    for pos, nid in enumerate(ids):
        out[int(nid)] = emb[pos]
    return out

def diagnose_layers(pyg_model: PyGHGT2L, session: Session):
    """Step-by-step comparison of PyG and RelNN intermediate outputs."""
    print("\n[diagnose] Comparing PyG and RelNN intermediate outputs after weight sync")

    n_authors = pyg_data["author"].x.size(0)
    n_papers  = pyg_data["paper"].x.size(0)

    pyg_model.eval()

    # ---- capture PyG L1 outputs via hooks ----
    _captured = {}
    def _hook_after(name):
        def _h(module, inp, out):
            _captured[name] = out
        return _h

    h1 = pyg_model.convs[0].register_forward_hook(_hook_after("conv0_out"))
    h2 = pyg_model.convs[1].register_forward_hook(_hook_after("conv1_out"))

    with torch.no_grad():
        pyg_logits = pyg_model()

    h1.remove()
    h2.remove()

    pyg_paper_l1  = _captured["conv0_out"]["paper"].cpu()   # (N_paper, hidden)
    pyg_author_l1 = _captured["conv0_out"]["author"].cpu()  # (N_author, hidden)
    pyg_author_l2 = _captured["conv1_out"]["author"].cpu()  # (N_author, hidden) = L2 output

    # ---- query RelNN for intermediate tables ----
    _PAPER_OUT1_DSL = """
#lang:relnn
?pred _PaperOut1(paper_id; z) :- PaperOut1(paper_id; z) .
"""
    _AUTHOR_OUT1_DSL = """
#lang:relnn
?pred _AuthorOut1(author_id; z) :- AuthorOut1(author_id; z) .
"""
    _AUTHOR_OUT2_DSL = """
#lang:relnn
?pred _AuthorOut2(author_id; z) :- AuthorOut2(author_id; z) .
"""

    rn_paper_l1_pred  = session.run(_PAPER_OUT1_DSL)
    rn_paper_l1  = _align_relnn_table(rn_paper_l1_pred,  n_papers,  hidden, "paper_id")

    rn_author_l1_pred = session.run(_AUTHOR_OUT1_DSL)
    rn_author_l1 = _align_relnn_table(rn_author_l1_pred, n_authors, hidden, "author_id")

    rn_author_l2_pred = session.run(_AUTHOR_OUT2_DSL)
    rn_author_l2 = _align_relnn_table(rn_author_l2_pred, n_authors, hidden, "author_id")

    print()
    _diff(pyg_paper_l1,  rn_paper_l1,  "L1 Paper  output (PyG vs RelNN)")
    _diff(pyg_author_l1, rn_author_l1, "L1 Author output (PyG vs RelNN)")
    _diff(pyg_author_l2, rn_author_l2, "L2 Author output (PyG vs RelNN)")

    # check coverage: how many papers appear in RelNN's PaperOut1?
    paper_count_rn = rn_paper_l1_pred.content.shape[0]
    print(f"\n  RelNN PaperOut1 rows: {paper_count_rn} / {n_papers} total papers")
    if paper_count_rn < n_papers:
        print(f"  WARNING: {n_papers - paper_count_rn} papers missing from PaperOut1!")

    return {
        "l1_paper": (pyg_paper_l1, rn_paper_l1),
        "l1_author": (pyg_author_l1, rn_author_l1),
        "l2_author": (pyg_author_l2, rn_author_l2),
    }

def train_one(model: nn.Module, forward_fn, epochs=100, lr=0.005, wd=0.001):
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

# ============================================================================
# Single-seed experiment
# ============================================================================

def run_seed(seed: int, epochs: int = 100) -> dict:
    print()
    print(DIVIDER)
    print(f"Seed {seed}")
    print(DIVIDER)
    n_authors = pyg_data["author"].x.size(0)

    # -- Initialise PyG 2L -------------------------------------------------
    full_seed(seed)
    pyg_model = PyGHGT2L().to(DEVICE)
    with torch.no_grad():
        pyg_model()   # materialise lazy linears

    # -- Initialise RelNN 2L with same DSL ---------------------------------
    session = Session(db=relnn_db)
    session.run(RELNN_DEFINE_DSL_2L)

    # Materialise parameters via a forward pass
    with torch.no_grad():
        session.run(RELNN_LOGITS_DSL)

    # -- Diagnostic (first seed only) -------------------------------------
    if seed == 42:
        conv0 = pyg_model.convs[0]
        print()
        print("[PyG conv0 parameter shapes]")
        print_pyg_conv_shapes(conv0, "L1")
        print()
        print("[RelNN parameter store keys]")
        print_relnn_param_keys(session.engine.parameter_store, "2L")

    # -- Sync weights PyG -> RelNN ----------------------------------------
    print(f"\n[sync] PyG -> RelNN (seed {seed})")
    sync_pyg_2l_to_relnn(pyg_model, session)

    # -- Forward parity check ---------------------------------------------
    print(f"\n[forward check] seed {seed}")
    pyg_model.eval()
    with torch.no_grad():
        pyg_logits = pyg_model().cpu()

    rn_logits_pred = session.run(RELNN_LOGITS_DSL)
    rn_logits = _align_relnn_logits(rn_logits_pred, n_authors)
    max_diff = _diff(pyg_logits, rn_logits, "PyG <-> RelNN logits")

    # -- Layer-by-layer diagnostic (always run for debugging) -------------
    diagnose_layers(pyg_model, session)

    # -- Train from synced init -------------------------------------------
    print(f"\n[train PyG] seed {seed}")
    full_seed(seed)
    _sync_cuda()
    t0 = time.perf_counter()
    train_one(pyg_model, lambda m: m(), epochs=epochs)
    _sync_cuda()
    pyg_time = time.perf_counter() - t0
    pyg_accs = _eval_pyg(pyg_model)

    print(f"\n[train RelNN] seed {seed}")
    full_seed(seed)
    _sync_cuda()
    t0 = time.perf_counter()
    session.run(RELNN_FIT_DSL.format(epochs=epochs, lr=0.005, wd=0.001))
    _sync_cuda()
    rn_time = time.perf_counter() - t0
    rn_accs = _eval_relnn(session)

    print()
    print("  [results]")
    print(f"  PyG   test={pyg_accs['test']:.1%}  time={pyg_time:.1f}s")
    print(f"  RelNN test={rn_accs['test']:.1%}  time={rn_time:.1f}s")
    print(f"  fwd_diff={max_diff:.2e}  acc_diff={abs(pyg_accs['test'] - rn_accs['test']):.1%}")

    return {
        "seed": seed,
        "fwd_diff": max_diff,
        "pyg":   pyg_accs,
        "relnn": rn_accs,
        "pyg_time":  pyg_time,
        "relnn_time": rn_time,
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs",       type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--epochs",     type=int, default=100)
    ap.add_argument("--diag-only",  action="store_true",
                    help="Print param keys and exit (for debugging sync mapping)")
    args = ap.parse_args()

    if args.diag_only:
        print("[diagnostic mode]")
        full_seed(42)
        _pyg_tmp = PyGHGT2L().to(DEVICE)
        with torch.no_grad():
            _pyg_tmp()
        print("[PyG conv0]")
        print_pyg_conv_shapes(_pyg_tmp.convs[0], "L1")
        print("[PyG conv1]")
        print_pyg_conv_shapes(_pyg_tmp.convs[1], "L2")

        _sess_tmp = Session(db=relnn_db)
        _sess_tmp.run(RELNN_DEFINE_DSL_2L)
        with torch.no_grad():
            _sess_tmp.run(RELNN_LOGITS_DSL)
        print("[RelNN store]")
        print_relnn_param_keys(_sess_tmp.engine.parameter_store, "2L full")
        sys.exit(0)

    seeds = [args.seed_start + i for i in range(args.runs)]
    all_results = []
    for seed in seeds:
        r = run_seed(seed, epochs=args.epochs)
        all_results.append(r)

    print()
    print(DIVIDER)
    print("SUMMARY (all seeds, 2L full-graph)")
    print(DIVIDER)
    print(f"{'Seed':>6}  {'PyG test':>10}  {'RelNN test':>10}  {'diff':>8}  {'fwd_diff':>12}")
    for r in all_results:
        print(f"  {r['seed']:4d}  {r['pyg']['test']:9.1%}  {r['relnn']['test']:9.1%}"
              f"  {abs(r['pyg']['test'] - r['relnn']['test']):7.1%}  {r['fwd_diff']:12.2e}")

    pyg_tests  = [r["pyg"]["test"]   for r in all_results]
    rn_tests   = [r["relnn"]["test"] for r in all_results]
    fwd_diffs  = [r["fwd_diff"]      for r in all_results]
    print()
    print(f"  PyG   mean={np.mean(pyg_tests):.1%}  std={np.std(pyg_tests, ddof=1):.1%}")
    print(f"  RelNN mean={np.mean(rn_tests):.1%}  std={np.std(rn_tests, ddof=1):.1%}")
    print(f"  fwd_diff  max={max(fwd_diffs):.2e}  mean={np.mean(fwd_diffs):.2e}")

    out_path = (
        Path(__file__).resolve().parents[2]
        / "research" / "paper_experiments"
        / "hgt"
        / "results"
        / "hgt_dblp_2l_parity_results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    parity_passed = max(fwd_diffs) < 1e-4
    print(f"\nForward parity: {'[PASS]' if parity_passed else '[FAIL]'} (threshold 1e-4)")
    acc_diffs = [abs(r["pyg"]["test"] - r["relnn"]["test"]) for r in all_results]
    acc_passed = all(d < 0.005 for d in acc_diffs)
    print(f"Accuracy parity: {'[PASS]' if acc_passed else '[FAIL]'} (threshold 0.5%)")
