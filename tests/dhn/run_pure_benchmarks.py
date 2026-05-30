"""
Pure-RelNN DHN benchmark suite.

Reproduces Table 1 from the DHN paper (NeurIPS 2024) using pure-RelNN:
homomorphisms computed via cyclic/clique self-joins on the Edge relation,
with NO Python preprocessing of subgraph patterns.

For configurations where the join-based approach exceeds a 2-minute timeout,
falls back to the pre-computed approach and notes it in the results.

Usage:
    python run_pure_benchmarks.py                  # full benchmark
    python run_pure_benchmarks.py --quick          # smoke test (CSL C2:4)
    python run_pure_benchmarks.py --dataset CSL    # single dataset
"""

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))

from relann.session import Session
from relann.torch_utils import full_seed

import pandas as pd

from dhn_utils import (
    DHNConfig,
    _load_dataset,
    build_dhn_db,
    build_pure_dhn_db,
    build_walk_count_db,
    generate_count_dhn_program,
    generate_dhn_program,
    generate_pure_dhn_program,
    precompute_hom_counts,
    precompute_walk_counts,
)

# ── Paper Table 1 configurations ────────────────────────────────────────────

PAPER_CONFIGS = {
    "C2:4":              [["C2", "C3", "C4"]],
    "C2:5":              [["C2", "C3", "C4", "C5"]],
    "C2:8":              [["C2", "C3", "C4", "C5", "C6", "C7", "C8"]],
    "C2:10":             [["C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]],
    "C2K3:5":            [["C2", "K3", "K4", "K5"]],
    "C2:4, C2":          [["C2", "C3", "C4"], ["C2"]],
    "C2:5, C2":          [["C2", "C3", "C4", "C5"], ["C2"]],
    "C2:5, C2:5":        [["C2", "C3", "C4", "C5"], ["C2", "C3", "C4", "C5"]],
    "C5:10, C2":         [["C5", "C6", "C7", "C8", "C9", "C10"], ["C2"]],
    "C2K3:5, C2K3:5":    [["C2", "K3", "K4", "K5"], ["C2", "K3", "K4", "K5"]],
}

PAPER_ACCURACY = {
    ("CSL",      "C2:4"):           100.0,
    ("CSL",      "C2:5"):           100.0,
    ("CSL",      "C2:10"):          100.0,
    ("CSL",      "C2K3:5"):         100.0,
    ("CSL",      "C2:4, C2"):       100.0,
    ("CSL",      "C2:5, C2"):       100.0,
    ("CSL",      "C2:5, C2:5"):     100.0,
    ("CSL",      "C5:10, C2"):      100.0,
    ("CSL",      "C2K3:5, C2K3:5"): 100.0,
    ("EXP",      "C2:4"):            50.0,
    ("EXP",      "C2:5"):            81.0,
    ("EXP",      "C2:10"):           98.0,
    ("EXP",      "C2K3:5"):          50.0,
    ("EXP",      "C2:4, C2"):        50.0,
    ("EXP",      "C2:5, C2"):        99.0,
    ("EXP",      "C5:10, C2"):      100.0,
    ("EXP",      "C2K3:5, C2K3:5"): 100.0,
    ("SR25",     "C2:4"):             0.0,
    ("SR25",     "C2:5"):             0.0,
    ("SR25",     "C2:10"):            0.0,
    ("SR25",     "C2K3:5"):          53.0,
    ("SR25",     "C2:4, C2"):         0.0,
    ("SR25",     "C2:5, C2"):         0.0,
    ("SR25",     "C2K3:5, C2K3:5"): 100.0,
    ("ENZYMES",  "C2:4"):            64.3,
    ("ENZYMES",  "C2:5"):            63.7,
    ("ENZYMES",  "C2:10"):           58.0,
    ("ENZYMES",  "C2K3:5"):          63.3,
    ("PROTEINS", "C2:4"):            76.5,
    ("PROTEINS", "C2:5"):            77.0,
    ("PROTEINS", "C2:10"):           78.5,
    ("PROTEINS", "C2K3:5"):          76.0,
}

TIMEOUT_SECONDS = 120  # 2-minute timeout for pure-RelNN

@dataclass
class BenchmarkResult:
    dataset: str
    config: str
    paper_acc: Optional[float]
    relnn_acc: float
    approach: str  # "pure-relnn" or "pre-computed"
    time_s: float
    notes: str = ""

def _train_and_predict(
    db: dict,
    define_prog: str,
    fit_prog: str,
    pred_prog: str,
    labels: torch.Tensor,
    device: Optional[str] = None,
) -> Tuple[float, float]:
    """Train and predict, return (accuracy%, wall_time_seconds)."""
    t0 = time.time()
    session = Session(db=db, device=device)
    session.run(define_prog)
    session.run(fit_prog)
    result = session.run(pred_prog)
    elapsed = time.time() - t0
    preds = result.embeddings[0].view(-1).long()
    acc = 100.0 * (preds.cpu() == labels.cpu()).sum().item() / len(labels)
    return acc, elapsed

def _kfold_walk_count_cv(
    graphs, labels, node_features,
    patterns: List[str],
    d_in_override: Optional[int] = None,
    num_classes: int = 10,
    readout: str = "sum",
    lr: float = 0.001,
    epochs: int = 500,
    n_folds: int = 10,
    seed: int = 42,
    verbose: bool = False,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    dropout: float = 0.0,
    use_injective_counts: bool = False,
    device: Optional[str] = None,
) -> Tuple[float, float, float]:
    """Proper stratified k-fold CV using count-based node features.

    For each fold: trains on train graphs only (via GraphLabel), predicts on
    ALL graphs, evaluates accuracy on the held-out test graphs.

    If use_injective_counts is True, uses injective homomorphism counts
    (simple cycles, cliques) instead of non-injective walk counts. Slower
    but closer to the official DHN (gear/dhn) representation.

    Returns (mean_test_acc, std_test_acc, total_time).
    """
    from sklearn.model_selection import StratifiedKFold, LeaveOneOut

    # Pre-compute count features for ALL graphs once
    all_graph_ids = list(range(len(graphs)))
    if use_injective_counts:
        walk_node_df, walk_feats = precompute_hom_counts(graphs, patterns, all_graph_ids)
    else:
        walk_node_df, walk_feats = precompute_walk_counts(graphs, patterns, all_graph_ids)

    # Optionally append original node features
    if node_features is not None:
        combined_feats = torch.cat([walk_feats, node_features], dim=1)
    else:
        combined_feats = walk_feats

    d_in = d_in_override if d_in_override else combined_feats.shape[1]
    define, fit, pred = generate_count_dhn_program(
        d_in=d_in,
        num_classes=num_classes,
        readout=readout,
        lr=lr,
        epochs=epochs,
        optimizer=optimizer,
        weight_decay=weight_decay,
        dropout=dropout,
    )

    # Use LOO for datasets where stratified k-fold can't work (e.g. SR25: 15 classes, 1 per class)
    min_class_size = min(int((labels == c).sum()) for c in labels.unique())
    if min_class_size < n_folds:
        if verbose:
            print(f"    Using Leave-One-Out CV (min class size={min_class_size} < {n_folds})")
        splitter = LeaveOneOut()
        split_iter = splitter.split(np.zeros(len(labels)))
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(labels)), labels.numpy())

    fold_accs = []
    total_time = 0.0

    for fold_num, (train_idx, test_idx) in enumerate(split_iter):
        full_seed(seed + fold_num)

        # DB: Node has ALL graphs, GraphLabel has only TRAIN graph_ids
        train_label_df = pd.DataFrame({"graph_id": train_idx.tolist()})
        train_label_emb = labels[train_idx].view(-1, 1).float()

        db = {
            "Node": (walk_node_df, combined_feats),
            "GraphLabel": (train_label_df, train_label_emb),
        }

        try:
            t0 = time.time()
            session = Session(db=db, device=device)
            session.run(define)
            session.run(fit)
            result = session.run(pred)
            elapsed = time.time() - t0
            total_time += elapsed

            all_preds = result.embeddings[0].view(-1).long()
            test_preds = all_preds[test_idx]
            test_labels = labels[test_idx]
            test_acc = 100.0 * (test_preds.cpu() == test_labels.cpu()).sum().item() / len(test_labels)
            fold_accs.append(test_acc)

            if verbose:
                print(f"    Fold {fold_num+1}/{n_folds}: "
                      f"test_acc={test_acc:.1f}% ({elapsed:.1f}s)")
        except Exception as e:
            if verbose:
                print(f"    Fold {fold_num+1}/{n_folds}: FAILED ({e})")
            fold_accs.append(0.0)

    return float(np.mean(fold_accs)), float(np.std(fold_accs)), total_time

def _kfold_full_ghl_cv(
    graphs, labels, node_features,
    patterns_per_layer: List[List[str]],
    num_classes: int = 10,
    d_k: int = 5,
    d_hidden: int = 5,
    d_in: int = 21,
    readout: str = "sum",
    lr: float = 0.001,
    epochs: int = 500,
    n_folds: int = 10,
    seed: int = 42,
    verbose: bool = False,
    mu_n_layers: int = 2,
    mu_dropout: float = 0.05,
    pattern_combine: str = "concat",
    device: Optional[str] = None,
) -> Tuple[float, float, float]:
    """10-fold stratified CV using the full GHL architecture (pre-computed Hom tables).

    Matches the official DHN architecture: per-position transforms, element-wise
    products, scatter-add aggregation, with Concat pattern combination and
    2-layer Mu MLPs with Dropout.
    """
    from sklearn.model_selection import StratifiedKFold

    all_pats = sorted(set(p for layer in patterns_per_layer for p in layer))

    config = DHNConfig(
        patterns_per_layer=patterns_per_layer,
        d_in=d_in, d_k=d_k, d_hidden=d_hidden, num_classes=num_classes,
        readout=readout, lr=lr, epochs=epochs,
        mu_n_layers=mu_n_layers, mu_dropout=mu_dropout,
        pattern_combine=pattern_combine,
    )
    define, fit, pred = generate_dhn_program(config)

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []
    total_time = 0.0

    # Pre-compute Hom tables once (expensive)
    all_graph_ids = list(range(len(graphs)))
    t_precomp = time.time()
    base_db = build_dhn_db(
        graphs, all_pats,
        node_features=node_features,
        labels=None,
        graph_ids=all_graph_ids,
    )
    t_precomp = time.time() - t_precomp
    if verbose:
        print(f"    Hom table pre-computation: {t_precomp:.1f}s")

    for fold_num, (train_idx, test_idx) in enumerate(splitter.split(
        np.zeros(len(labels)), labels.numpy()
    )):
        full_seed(seed + fold_num)
        train_labels = labels[train_idx]

        try:
            t0 = time.time()

            db = dict(base_db)
            train_label_df = pd.DataFrame({"graph_id": train_idx.tolist()})
            train_label_emb = train_labels.view(-1, 1).float()
            db["GraphLabel"] = (train_label_df, train_label_emb)

            session = Session(db=db, device=device)
            session.run(define)
            session.run(fit)
            result = session.run(pred)
            elapsed = time.time() - t0
            total_time += elapsed

            all_preds = result.embeddings[0].view(-1).long()
            test_preds = all_preds[test_idx]
            test_labels = labels[test_idx]
            test_acc = 100.0 * (test_preds.cpu() == test_labels.cpu()).sum().item() / len(test_labels)
            fold_accs.append(test_acc)

            if verbose:
                print(f"    Fold {fold_num+1}/{n_folds}: "
                      f"test_acc={test_acc:.1f}% ({elapsed:.1f}s)")
        except Exception as e:
            if verbose:
                print(f"    Fold {fold_num+1}/{n_folds}: FAILED ({e})")
                import traceback as _tb
                _tb.print_exc()
            fold_accs.append(0.0)

    total_time += t_precomp
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), total_time

def run_single(
    dataset_name: str,
    config_name: str,
    patterns_per_layer: List[List[str]],
    epochs: int = 500,
    lr: float = 0.001,
    d_hidden: int = 20,
    d_k: int = 10,
    seed: int = 42,
    data_root: str = "./data",
    timeout: int = TIMEOUT_SECONDS,
    verbose: bool = True,
    eval_mode: str = "auto",
    use_injective_counts: bool = False,
    paper_align: bool = True,
    full_ghl: bool = False,
    device: Optional[str] = None,
    max_graphs: Optional[int] = None,
    forbid_walk_count_fallback: bool = False,
) -> BenchmarkResult:
    """Run a single benchmark.

    eval_mode:
      "auto"  - train acc for synthetic (CSL/EXP/SR25), 10-fold CV for real
      "train" - train accuracy only (all data)
      "cv"    - 10-fold CV test accuracy only
      "both"  - report both train and CV test accuracy

    use_injective_counts: if True, use injective homomorphism counts instead of
      walk counts for CV (ENZYMES/PROTEINS). Slower but closer to official DHN.
    paper_align: if True (default), force paper-aligned training (AdamW, dropout,
      100 epochs) for ENZYMES/PROTEINS. Set False to use caller's epochs/optimizer.
    full_ghl: if True, use full GHL architecture (pre-computed Hom tables, 2-layer,
      per-position transforms) for CV instead of count-MLP.
    device: Optional torch device string for Session (e.g. "cuda", "cpu").
    max_graphs: If set, use only the first N graphs (and matching labels / features).
    forbid_walk_count_fallback: If True, do not fall back to walk-count MLP when pure RelNN fails.
    """
    paper_acc = PAPER_ACCURACY.get((dataset_name, config_name))

    if eval_mode == "auto":
        if dataset_name.upper() in ("ENZYMES", "PROTEINS"):
            eval_mode = "cv"
        else:
            eval_mode = "train"

    graphs, labels, nc, nf, node_features = _load_dataset(
        dataset_name, root=data_root
    )

    subset_note = ""
    if max_graphs is not None and max_graphs > 0 and len(graphs) > max_graphs:
        graphs = graphs[:max_graphs]
        labels = labels[:max_graphs]
        if node_features is not None:
            raise ValueError(
                "max_graphs with per-node feature tensor is not supported; "
                "use datasets where _load_dataset returns node_features=None."
            )
        subset_note = f"subset_first_{max_graphs}_graphs"

    if verbose:
        print(f"\n{'='*70}")
        print(f"  {dataset_name} | DHN-({config_name}) | eval={eval_mode}")
        if subset_note:
            print(f"  {subset_note}")
        if device:
            print(f"  Session device: {device}")
        print(f"  Layers: {len(patterns_per_layer)}, "
              f"Patterns: {[p for layer in patterns_per_layer for p in layer]}")
        print(f"{'='*70}")

    all_pats = sorted(set(p for layer in patterns_per_layer for p in layer))

    config = DHNConfig(
        patterns_per_layer=patterns_per_layer,
        d_in=nf, d_k=d_k, d_hidden=d_hidden, num_classes=nc,
        readout="sum", lr=lr, epochs=epochs,
    )

    # --- Train accuracy (expressivity) ---
    train_acc = None
    train_time = 0.0
    approach = "walk-counts"
    notes = ""

    if eval_mode in ("train", "both"):
        try:
            full_seed(seed)
            db = build_pure_dhn_db(
                graphs, labels=labels, node_features=node_features
            )
            define, fit, pred = generate_pure_dhn_program(config)
            train_acc, train_time = _train_and_predict(
                db, define, fit, pred, labels, device=device
            )
            approach = "pure-relnn"
            if verbose:
                print(f"  Pure-RelNN train: {train_acc:.1f}% in {train_time:.1f}s")

            if train_time > timeout:
                raise TimeoutError(f"pure took {train_time:.0f}s")

        except Exception as e:
            if forbid_walk_count_fallback:
                if verbose:
                    print(f"  Pure-RelNN failed (--forbid-walk-count-fallback): {e}")
                raise
            if verbose:
                print(f"  Pure-RelNN exceeded limit: {e}")
                print(f"  Falling back to walk-counts for train acc...")
            full_seed(seed)
            approach = "walk-counts"
            db = build_walk_count_db(
                graphs, all_pats, labels=labels,
                node_features=node_features,
            )
            d_in = 1 + len(all_pats)
            if node_features is not None:
                d_in += node_features.shape[1]
            define, fit, pred = generate_count_dhn_program(
                d_in=d_in, num_classes=nc, readout="sum",
                lr=lr, epochs=epochs,
            )
            train_acc, train_time = _train_and_predict(
                db, define, fit, pred, labels, device=device
            )
            notes = "fallback"
            if verbose:
                print(f"  Walk-counts train: {train_acc:.1f}% in {train_time:.1f}s")

    # --- 10-fold CV test accuracy ---
    cv_acc, cv_std, cv_time = None, None, 0.0

    if eval_mode in ("cv", "both"):
        full_seed(seed)

        if full_ghl:
            if verbose:
                print(f"  Running 10-fold CV (full GHL, 2-layer)...")
            approach = "full-ghl"
            # Use 2-layer DHN matching official config
            ghl_layers = patterns_per_layer if len(patterns_per_layer) >= 2 else [all_pats, all_pats]
            cv_acc, cv_std, cv_time = _kfold_full_ghl_cv(
                graphs, labels, node_features,
                patterns_per_layer=ghl_layers,
                num_classes=nc, d_k=d_k, d_hidden=d_hidden, d_in=nf,
                readout="sum", lr=lr, epochs=epochs,
                n_folds=10, seed=seed, verbose=verbose,
                mu_n_layers=2, mu_dropout=0.05,
                pattern_combine="sum",
                device=device,
            )
        else:
            if verbose:
                count_type = "injective counts" if use_injective_counts else "walk-counts"
                print(f"  Running 10-fold CV ({count_type})...")
            use_paper_training = paper_align and dataset_name.upper() in ("ENZYMES", "PROTEINS")
            cv_epochs = 100 if use_paper_training else epochs
            cv_acc, cv_std, cv_time = _kfold_walk_count_cv(
                graphs, labels, node_features,
                patterns=all_pats,
                num_classes=nc, readout="sum",
                lr=lr, epochs=cv_epochs,
                n_folds=10, seed=seed, verbose=verbose,
                optimizer="adamw" if use_paper_training else "adam",
                weight_decay=0.01 if use_paper_training else 0.0,
                dropout=0.05 if use_paper_training else 0.0,
                use_injective_counts=use_injective_counts,
                device=device,
            )

        if verbose:
            print(f"  10-fold CV: {cv_acc:.1f} +/- {cv_std:.1f}% "
                  f"in {cv_time:.1f}s total")

    # Choose which accuracy to report as primary
    if full_ghl:
        approach = "full-ghl"
    if eval_mode == "cv":
        acc = cv_acc
        elapsed = cv_time
        mode_note = "full-ghl" if full_ghl else ("injective" if use_injective_counts else "")
        notes = f"10-fold CV; std={cv_std:.1f}" + (f"; {mode_note}" if mode_note else "") + (f"; {notes}" if notes else "")
    elif eval_mode == "both":
        acc = cv_acc
        elapsed = train_time + cv_time
        notes = (f"train={train_acc:.1f}%; 10-fold CV; std={cv_std:.1f}"
                 + (f"; {notes}" if notes else ""))
    else:
        acc = train_acc
        elapsed = train_time
        notes = f"train acc" + (f"; {notes}" if notes else "")

    if subset_note and subset_note not in (notes or ""):
        notes = f"{notes}; {subset_note}" if notes else subset_note

    result = BenchmarkResult(
        dataset=dataset_name,
        config=config_name,
        paper_acc=paper_acc,
        relnn_acc=acc,
        approach=approach,
        time_s=elapsed,
        notes=notes,
    )
    return result

# ── Benchmark suite ──────────────────────────────────────────────────────────

DEFAULT_DATASETS = ["CSL", "EXP", "SR25", "ENZYMES", "PROTEINS"]

# Configs to run per dataset (subset of PAPER_CONFIGS for practical runtime)
DATASET_CONFIGS = {
    "CSL":      ["C2:4", "C2:5", "C2:10", "C2K3:5", "C2:4, C2", "C2:5, C2"],
    "EXP":      ["C2:4", "C2:5", "C2:5, C2"],
    "SR25":     ["C2:4", "C2K3:5"],
    "ENZYMES":  ["C2:4", "C2:5"],
    "PROTEINS": ["C2:4", "C2:5"],
}

def run_all(
    datasets: Optional[List[str]] = None,
    configs: Optional[Dict[str, List[List[str]]]] = None,
    epochs: int = 500,
    quick: bool = False,
    data_root: str = "./data",
    eval_mode: str = "auto",
    use_injective_counts: bool = False,
    paper_align: bool = True,
    full_ghl: bool = False,
    device: Optional[str] = None,
    max_graphs: Optional[int] = None,
    forbid_walk_count_fallback: bool = False,
) -> List[BenchmarkResult]:
    """Run the full benchmark suite."""
    if quick:
        datasets = ["CSL"]
        configs = {"C2:4": PAPER_CONFIGS["C2:4"]}
        epochs = 50

    if datasets is None:
        datasets = DEFAULT_DATASETS

    results: List[BenchmarkResult] = []

    for ds in datasets:
        ds_configs = configs or {
            c: PAPER_CONFIGS[c] for c in DATASET_CONFIGS.get(ds, ["C2:4"])
        }
        for cname, ppl in ds_configs.items():
            try:
                r = run_single(
                    dataset_name=ds,
                    config_name=cname,
                    patterns_per_layer=ppl,
                    epochs=epochs,
                    data_root=data_root,
                    eval_mode=eval_mode,
                    use_injective_counts=use_injective_counts,
                    paper_align=paper_align,
                    full_ghl=full_ghl,
                    device=device,
                    max_graphs=max_graphs,
                    forbid_walk_count_fallback=forbid_walk_count_fallback,
                )
                results.append(r)
            except Exception as e:
                print(f"\n  FAILED: {ds}/{cname}: {e}")
                traceback.print_exc()
                results.append(BenchmarkResult(
                    dataset=ds, config=cname,
                    paper_acc=PAPER_ACCURACY.get((ds, cname)),
                    relnn_acc=-1, approach="error",
                    time_s=0, notes=str(e),
                ))

    print_results_table(results)
    return results

def print_results_table(results: List[BenchmarkResult]):
    """Print a markdown-formatted results table."""
    print(f"\n{'='*90}")
    print("  PURE-RELNN DHN BENCHMARK RESULTS")
    print(f"{'='*90}\n")

    print("| Dataset | Config | Paper Acc | RelNN Acc | Approach | Time (s) | Notes |")
    print("|---------|--------|-----------|-----------|----------|----------|-------|")

    for r in results:
        paper = f"{r.paper_acc:.1f}%" if r.paper_acc is not None else "--"
        relnn = f"{r.relnn_acc:.1f}%" if r.relnn_acc >= 0 else "ERROR"
        print(f"| {r.dataset:<7} | {r.config:<14} | {paper:>9} | {relnn:>9} "
              f"| {r.approach:<8} | {r.time_s:>8.1f} | {r.notes} |")

    print()

# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pure-RelNN DHN benchmarks (Table 1 reproduction)"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (CSL, C2:4, 50 epochs)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset (CSL, EXP, SR25, ENZYMES, PROTEINS)")
    parser.add_argument("--config", type=str, default=None,
                        help="Single config (e.g. 'C2:4', 'C2K3:5')")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--eval-mode", type=str, default="auto",
                        choices=["auto", "train", "cv", "both"],
                        help="Evaluation mode: auto (train for synthetic, CV "
                             "for real), train, cv (10-fold), or both")
    parser.add_argument("--cv", action="store_true",
                        help="Shorthand for --eval-mode cv (10-fold CV for all)")
    parser.add_argument("--injective-counts", action="store_true",
                        help="Use injective homomorphism counts for CV (slower, closer to official DHN)")
    parser.add_argument("--no-paper-align", action="store_true",
                        help="Don't force paper-aligned training for ENZYMES/PROTEINS (use caller's epochs/optimizer)")
    parser.add_argument("--full-ghl", action="store_true",
                        help="Use full GHL architecture (pre-computed Hom tables) for CV instead of count-MLP")
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device for Session (e.g. cuda, cpu). Default: engine default (CPU).",
    )
    parser.add_argument(
        "--max-graphs", type=int, default=None,
        help="Use only the first N graphs (CSL/EXP without concatenated node_features).",
    )
    parser.add_argument(
        "--forbid-walk-count-fallback",
        action="store_true",
        help="When pure RelNN train path fails, re-raise instead of using walk-count MLP.",
    )
    args = parser.parse_args()

    eval_mode = "cv" if args.cv else args.eval_mode

    ds_list = [args.dataset.upper()] if args.dataset else None
    cfg_dict = None
    if args.config:
        if args.config not in PAPER_CONFIGS:
            print(f"Unknown config: {args.config}")
            print(f"Available: {list(PAPER_CONFIGS.keys())}")
            sys.exit(1)
        cfg_dict = {args.config: PAPER_CONFIGS[args.config]}

    run_all(
        datasets=ds_list,
        configs=cfg_dict,
        epochs=args.epochs,
        eval_mode=eval_mode,
        quick=args.quick,
        data_root=args.data_root,
        use_injective_counts=args.injective_counts,
        paper_align=not args.no_paper_align,
        full_ghl=args.full_ghl,
        device=args.device,
        max_graphs=args.max_graphs,
        forbid_walk_count_fallback=args.forbid_walk_count_fallback,
    )
