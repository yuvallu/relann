"""Compare RelNN first-order HyGNN vs reference PyTorch HyGNN on DDI dataset.

Trains both implementations on the same loaded data (load_hygnn_dataset),
evaluates on TestPairs, and compares metrics (accuracy, F1, ROC-AUC, PR-AUC)
and parameter counts.

By default runs the full grid (8 runs): sources TWOSIDES and DrugBank;
decoders MLP and dot; substructures k-mer (k=9) and ESPF (min_support from
``--espf-support``, default 5 — must match values the HyGNN repo ships).

Supports MLP decoder (paper Eq. 11) and dot-product decoder. Dot training uses
``Sigmoid``(inner product) + ``BCELoss`` on probabilities; MLP keeps ``BCEWithLogitsLoss``
on logits (same as the PyTorch reference for each mode).

Before training, PyTorch and RelNN initial weights are aligned using
``parent.comparison.SessionComparison.sync_weights`` (pyg_to_relnn), so divergent
RNG order between the two stacks does not skew the comparison.

Run from repo root:
    python research/paper_experiments/hygnn/run_compare_hygnn.py
    python research/paper_experiments/hygnn/run_compare_hygnn.py --decoder mlp
    python research/paper_experiments/hygnn/run_compare_hygnn.py --espf-support 10
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Literal

import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relann.comparison import SessionComparison
from relann.datasets import HYGNN_ESPF_SUPPORT_VALUES, load_hygnn_dataset
from relann.session import Session
from relann.torch_utils import full_seed

DIVIDER = "=" * 70

# Sibling module (ensure this dir is on path for pytest / odd invocations).
_SLOW_DIR = Path(__file__).resolve().parent
if str(_SLOW_DIR) not in sys.path:
    sys.path.insert(0, str(_SLOW_DIR))
from run_hygnn_pytorch_ref import HyGNN, compute_loss, hygnn_encoder_param_count

DecoderName = Literal["mlp", "dot"]

# Full comparison grid (default ``main`` runs all combinations).
_HY_GNN_SOURCES: tuple[str, ...] = ("TWOSIDES", "DrugBank")

# Single HyGNN layer in reference (`num_layers=1`); RelNN DSL matches `layers.0.*`.
_HY_GNN_REF_LAYER_IDX = 0


def _hygnn_pyg_to_relnn_mapping(decoder: DecoderName) -> dict[str, str]:
    """Map PyTorch ``HyGNN.named_parameters()`` keys to RelNN ``parameter_store`` FQNs."""

    def add_linear(m: dict[str, str], pyg_prefix: str, relnn_symbol: str) -> None:
        m[f"{pyg_prefix}.weight"] = f"global.{relnn_symbol}.weight"
        m[f"{pyg_prefix}.bias"] = f"global.{relnn_symbol}.bias"

    prefix = f"layers.{_HY_GNN_REF_LAYER_IDX}"
    out: dict[str, str] = {}
    # Inline ``Linear`` on ``DrugProj`` rule is stored under transformation_DrugProj._module.*
    out[f"{prefix}.drug_proj.weight"] = "global.transformation_DrugProj._module.weight"
    out[f"{prefix}.drug_proj.bias"] = "global.transformation_DrugProj._module.bias"
    for i in range(1, 7):
        add_linear(out, f"{prefix}.w{i}", f"W{i}")
    if decoder == "mlp":
        add_linear(out, "mlp_predictor.w1", "MLP1")
        add_linear(out, "mlp_predictor.w2", "MLP2")
    return out


def _expected_hygnn_sync_tensor_count(decoder: DecoderName) -> int:
    """Number of ``nn.Linear`` parameter tensors synced (weight+bias each)."""
    n_encoder_linears = 7  # drug_proj + w1..w6
    n_decoder_linears = 2 if decoder == "mlp" else 0
    return 2 * (n_encoder_linears + n_decoder_linears)


def _hygnn_feature_grid(espf_support: int) -> tuple[tuple[str, dict[str, Any]], ...]:
    if espf_support not in HYGNN_ESPF_SUPPORT_VALUES:
        raise ValueError(
            f"espf_support must be one of {sorted(HYGNN_ESPF_SUPPORT_VALUES)}, got {espf_support}"
        )
    return (
        ("k-mer (k=9)", {"substructure_method": "kmer", "k": 9}),
        (
            f"ESPF (support={espf_support})",
            {"substructure_method": "espf", "espf_support": espf_support},
        ),
    )


def run_reference_hygnn(
    data,
    seed: int = 42,
    epochs: int = 500,
    verbose: bool = True,
    decoder: DecoderName = "mlp",
    *,
    model: HyGNN | None = None,
    apply_seed: bool = True,
):
    """Train and evaluate reference HyGNN.

    If ``model`` is provided, trains that module in place (used after weight sync).
    When ``apply_seed`` is False, skips ``full_seed`` so the caller controls RNG.

    Returns (metrics_dict, total_param_count, wall_time_s, encoder_param_count).
    """
    if apply_seed:
        full_seed(seed)
    info = data.dataset_info
    n_drugs = info["n_drugs"]
    n_subs = info["n_subs"]

    _, drug_tensor = data.db["Drug"]
    _, sub_tensor = data.db["Substructure"]
    inc_df, _ = data.db["Incidence"]
    train_df, train_labels = data.db["TrainPairs"]
    test_df, test_labels_tensor = data.db["TestPairs"]

    sub_ids = torch.tensor(inc_df["sub_id"].values, dtype=torch.long)
    drug_ids = torch.tensor(inc_df["drug_id"].values, dtype=torch.long)

    efeat = drug_tensor
    vfeat = sub_tensor

    train_d1 = train_df["drug1"].values
    train_d2 = train_df["drug2"].values
    train_labels_np = train_labels.numpy().flatten()
    train_pos_mask = train_labels_np == 1.0
    train_neg_mask = ~train_pos_mask
    train_pos_src = torch.tensor(train_d1[train_pos_mask], dtype=torch.long)
    train_pos_dst = torch.tensor(train_d2[train_pos_mask], dtype=torch.long)
    train_neg_src = torch.tensor(train_d1[train_neg_mask], dtype=torch.long)
    train_neg_dst = torch.tensor(train_d2[train_neg_mask], dtype=torch.long)

    test_src = torch.tensor(test_df["drug1"].values, dtype=torch.long)
    test_dst = torch.tensor(test_df["drug2"].values, dtype=torch.long)
    test_labels = test_labels_tensor.numpy().flatten()

    if model is None:
        model = HyGNN(
            input_dim=n_drugs,
            query_dim=64,
            vertex_dim=128,
            edge_dim=128,
            dropout=0.5,
            num_layers=1,
            decoder=decoder,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    ref_params = sum(p.numel() for p in model.parameters())
    ref_enc_params = hygnn_encoder_param_count(model)

    t0 = time.perf_counter()
    for e in range(epochs):
        model.train()
        feat_v, feat_e = model(
            sub_ids, drug_ids, vfeat, efeat,
            n_subs=n_subs, n_drugs=n_drugs,
        )
        h = feat_e
        pos_score = model.decode(train_pos_src, train_pos_dst, h)
        neg_score = model.decode(train_neg_src, train_neg_dst, h)
        loss = compute_loss(pos_score, neg_score, decoder=decoder)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose and e % (epochs / 10) == 0 or e == epochs - 1:
            print(f"    Epoch {e}, loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        _, feat_e = model(
            sub_ids, drug_ids, vfeat, efeat,
            n_subs=n_subs, n_drugs=n_drugs,
        )
        h = feat_e
        raw_logits = model.decode(test_src, test_dst, h)
        pred_probs = torch.sigmoid(raw_logits).numpy()

    pred_binary = (pred_probs >= 0.5).astype(int)
    wall_time = time.perf_counter() - t0

    metrics = {
        "accuracy": accuracy_score(test_labels, pred_binary),
        "precision": precision_score(test_labels, pred_binary, zero_division=0),
        "recall": recall_score(test_labels, pred_binary, zero_division=0),
        "f1": f1_score(test_labels, pred_binary, zero_division=0),
        "roc_auc": roc_auc_score(test_labels, pred_probs),
        "pr_auc": average_precision_score(test_labels, pred_probs),
    }
    return metrics, ref_params, wall_time, ref_enc_params


# =============================================================================
# RelNN first-order HyGNN
# =============================================================================

RELNN_DEFINE_CORE = """
#lang:relnn
n_drugs = {n_drugs} .
d = 128 .
qd = 64 .

def Softmax(Scores):
    MaxPerT(t; max(z))        :- Scores(s, t; z) .
    Exp(s, t; exp(z - max_t)) :- Scores(s, t; z), MaxPerT(t; max_t) .
    Denom(t; sum(z))          :- Exp(s, t; z) .
    Out(s, t; z1 / z2)        :- Exp(s, t; z1), Denom(t; z2) .
enddef

DrugProj(drug_id; Linear(n_drugs, d)(z)) :- Drug(drug_id; z) .

W1 = Linear(d, d) .
W2 = Linear(d, qd) .
W3 = Linear(d, qd) .

K_E(drug_id; W2(z)) :- DrugProj(drug_id; z) .
V_E(drug_id; W1(z)) :- DrugProj(drug_id; z) .
Q_E(sub_id; W3(z)) :- Substructure(sub_id; z) .

AttnE_raw(drug_id, sub_id; LeakyReLU()(view(1)(z_q @ transpose(z_k))) / sqrt(qd)) :- K_E(drug_id; z_k), Incidence(sub_id, drug_id; w), Q_E(sub_id; z_q) .
AttnE(drug_id, sub_id; z) :- Softmax(AttnE_raw)(drug_id, sub_id; z) .
SubEmb(sub_id; sum(z_att * z_v)) :- AttnE(drug_id, sub_id; z_att), V_E(drug_id; z_v) .

W4 = Linear(d, d) .
W5 = Linear(d, qd) .
W6 = Linear(d, qd) .

K_V(sub_id; W5(z)) :- SubEmb(sub_id; z) .
V_V(sub_id; W4(z)) :- SubEmb(sub_id; z) .
Q_V(drug_id; W6(z)) :- DrugProj(drug_id; z) .

AttnV_raw(sub_id, drug_id; LeakyReLU()(view(1)(z_q @ transpose(z_k))) / sqrt(qd)) :- K_V(sub_id; z_k), Incidence(sub_id, drug_id; w), Q_V(drug_id; z_q) .
AttnV(sub_id, drug_id; z) :- Softmax(AttnV_raw)(sub_id, drug_id; z) .
DrugEmb_from_attn(drug_id; sum(z_att * z_v)) :- AttnV(sub_id, drug_id; z_att), V_V(sub_id; z_v) .
DrugEmb_pad(drug_id; zp - zp) :- DrugProj(drug_id; zp) .
DrugEmb_row(drug_id; z) :- DrugEmb_from_attn(drug_id; z1) | DrugEmb_pad(drug_id; z2) .
DrugEmb(drug_id; sum(z)) :- DrugEmb_row(drug_id; z) .

Drug1(drug1; z) :- DrugEmb(drug1; z) .
Drug2(drug2; z) :- DrugEmb(drug2; z) .
"""

RELNN_PAIRSCORE_MLP = """
MLP1 = Linear(2*d, d) .
MLP2 = Linear(d, 1) .

PairScore(drug1, drug2; MLP2(ReLU()(MLP1(Concat(z1, z2))))) :- Drug1(drug1; z1), TrainPairs(drug1, drug2; z_label), Drug2(drug2; z2) .
"""

RELNN_PAIRSCORE_DOT = """
PairScore(drug1, drug2; view(1)(z1 @ transpose(z2))) :- Drug1(drug1; z1), TrainPairs(drug1, drug2; z_label), Drug2(drug2; z2) .
"""

# Dot: inner product → sigmoid → BCE on probabilities (mirrors PyTorch ``binary_cross_entropy(sigmoid(dot), y)``).
RELNN_PAIRPROB_DOT = """
PairProb(drug1, drug2; Sigmoid()(z)) :- PairScore(drug1, drug2; z), TrainPairs(drug1, drug2; z_label) .
"""

RELNN_FIT_DSL_MLP = """
#lang:relnn
?fit <epochs={epochs}, lr=0.005>
Loss(; BCEWithLogitsLoss()(z_score, z_label)) :- PairScore(drug1, drug2; z_score), TrainPairs(drug1, drug2; z_label) .
"""

RELNN_FIT_DSL_DOT = """
#lang:relnn
?fit <epochs={epochs}, lr=0.005>
Loss(; BCELoss()(z_p, z_label)) :- PairProb(drug1, drug2; z_p), TrainPairs(drug1, drug2; z_label) .
"""

RELNN_TESTSCORE_MLP = """
#lang:relnn
TestScore(drug1, drug2; MLP2(ReLU()(MLP1(Concat(z1, z2))))) :- Drug1(drug1; z1), TestPairs(drug1, drug2; z_label), Drug2(drug2; z2) .
"""

RELNN_TESTSCORE_DOT = """
#lang:relnn
TestScore(drug1, drug2; view(1)(z1 @ transpose(z2))) :- Drug1(drug1; z1), TestPairs(drug1, drug2; z_label), Drug2(drug2; z2) .
"""

RELNN_PRED_DSL = """
#lang:relnn
?pred Predictions(drug1, drug2; Sigmoid()(z)) :- TestScore(drug1, drug2; z) .
"""

# Parameters are registered only after a forward/predict materializes modules (define alone leaves
# ``parameter_store`` empty). Same pattern as DBLP compare scripts' ``_Init`` predict.
RELNN_MATERIALIZE_PRED = """
#lang:relnn
?pred _HyGNNInitMaterialize(drug1, drug2; z) :- PairScore(drug1, drug2; z), TrainPairs(drug1, drug2; z_label) .
"""


def relnn_encoder_param_count(session: Session) -> int:
    """RelNN params excluding MLP1/MLP2 (decoder); dot mode has no MLP keys so equals total."""
    return sum(
        v.numel()
        for k, v in session.engine.parameter_store.items()
        if "MLP1" not in k and "MLP2" not in k
    )


def run_relnn_hygnn(
    data,
    seed: int = 42,
    epochs: int = 500,
    verbose: bool = True,
    decoder: DecoderName = "mlp",
    *,
    session: Session | None = None,
    apply_seed: bool = True,
):
    """Train and evaluate RelNN first-order HyGNN.

    If ``session`` is provided, uses it in place (must already have define rules
    run, e.g. after weight sync). When ``apply_seed`` is False, skips ``full_seed``.

    Returns (metrics_dict, total_param_count, wall_time_s, encoder_param_count).
    """
    if apply_seed:
        full_seed(seed)
    info = data.dataset_info
    n_drugs = info["n_drugs"]

    core = RELNN_DEFINE_CORE.format(n_drugs=n_drugs)
    if decoder == "mlp":
        tail = RELNN_PAIRSCORE_MLP
        fit_dsl = RELNN_FIT_DSL_MLP
    else:
        tail = RELNN_PAIRSCORE_DOT + RELNN_PAIRPROB_DOT
        fit_dsl = RELNN_FIT_DSL_DOT
    if session is None:
        session = Session(db=data.db)
        session.run(core + tail)
        session.run(RELNN_MATERIALIZE_PRED)

    t0 = time.perf_counter()
    session.run(fit_dsl.format(epochs=epochs))
    wall_time = time.perf_counter() - t0

    rn_params = sum(p.numel() for p in session.engine.parameter_store.values())
    rn_enc_params = relnn_encoder_param_count(session)

    test_dsl = RELNN_TESTSCORE_MLP if decoder == "mlp" else RELNN_TESTSCORE_DOT
    session.run(test_dsl)
    pred_result = session.run(RELNN_PRED_DSL)

    test_df = data.db["TestPairs"][0].copy()
    test_labels = data.test_labels.cpu().numpy().flatten()
    pred_df = pred_result.content.copy()
    pred_df["pred_score"] = pred_result.embeddings[0].detach().cpu().numpy().flatten()
    merged = test_df.merge(pred_df, on=["drug1", "drug2"], how="left")
    pred_scores = merged["pred_score"].values
    pred_binary = (pred_scores >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(test_labels, pred_binary),
        "precision": precision_score(test_labels, pred_binary, zero_division=0),
        "recall": recall_score(test_labels, pred_binary, zero_division=0),
        "f1": f1_score(test_labels, pred_binary, zero_division=0),
        "roc_auc": roc_auc_score(test_labels, pred_scores),
        "pr_auc": average_precision_score(test_labels, pred_scores),
    }
    return metrics, rn_params, wall_time, rn_enc_params


# =============================================================================
# Main
# =============================================================================


def _run_compare_for_decoder(
    data,
    decoder: DecoderName,
    *,
    combo_tag: str | None = None,
    compare_seed: int = 42,
    verbose: bool = True,
) -> None:
    dec_label = decoder.upper()
    print(DIVIDER)
    if combo_tag:
        print(combo_tag)
        print(DIVIDER)
    print(f"Decoder: {dec_label}")
    print(DIVIDER)

    print(DIVIDER)
    print("0. Build PyTorch + RelNN, SessionComparison weight sync (pyg_to_relnn)")
    print(DIVIDER)
    full_seed(compare_seed)
    info = data.dataset_info
    n_drugs = info["n_drugs"]
    ref_model = HyGNN(
        input_dim=n_drugs,
        query_dim=64,
        vertex_dim=128,
        edge_dim=128,
        dropout=0.5,
        num_layers=1,
        decoder=decoder,
    )
    core = RELNN_DEFINE_CORE.format(n_drugs=n_drugs)
    if decoder == "mlp":
        relnn_tail = RELNN_PAIRSCORE_MLP
    else:
        relnn_tail = RELNN_PAIRSCORE_DOT + RELNN_PAIRPROB_DOT
    rn_session = Session(db=data.db)
    rn_session.run(core + relnn_tail)
    rn_session.run(RELNN_MATERIALIZE_PRED)

    cmp = SessionComparison("HyGNN-Sync", verbose=verbose)
    cmp.set_pyg_model(ref_model)
    cmp.set_relnn_session(rn_session)
    cmp.set_mapping(_hygnn_pyg_to_relnn_mapping(decoder))
    if verbose:
        cmp.print_mapping()
    sync_result = cmp.sync_weights("pyg_to_relnn")
    expected_n = _expected_hygnn_sync_tensor_count(decoder)
    assert sync_result.n_skipped == 0, (
        f"Weight sync skipped tensors: {sync_result.warnings}"
    )
    assert sync_result.n_synced == expected_n, (
        f"Expected {expected_n} tensors synced, got {sync_result.n_synced}"
    )
    if verbose:
        print(f"  Weight sync OK: {sync_result.n_synced} tensors (pyg_to_relnn)")

    print(DIVIDER)
    print(f"1. Reference HyGNN (PyTorch + torch_scatter), decoder={decoder}")
    print(DIVIDER)
    ref_metrics, ref_params, ref_time, ref_enc = run_reference_hygnn(
        data,
        verbose=verbose,
        decoder=decoder,
        model=ref_model,
        apply_seed=False,
    )
    print(f"  Params (total): {ref_params:,}  (encoder): {ref_enc:,}  Time: {ref_time:.1f}s")
    print(
        f"  Accuracy: {ref_metrics['accuracy']:.4f}  F1: {ref_metrics['f1']:.4f}  "
        f"ROC-AUC: {ref_metrics['roc_auc']:.4f}  PR-AUC: {ref_metrics['pr_auc']:.4f}"
    )

    print()
    print(DIVIDER)
    print(f"2. RelNN first-order HyGNN, decoder={decoder}")
    print(DIVIDER)
    rn_metrics, rn_params, rn_time, rn_enc = run_relnn_hygnn(
        data,
        verbose=verbose,
        decoder=decoder,
        session=rn_session,
        apply_seed=False,
    )
    print(f"  Params (total): {rn_params:,}  (encoder): {rn_enc:,}  Time: {rn_time:.1f}s")
    print(
        f"  Accuracy: {rn_metrics['accuracy']:.4f}  F1: {rn_metrics['f1']:.4f}  "
        f"ROC-AUC: {rn_metrics['roc_auc']:.4f}  PR-AUC: {rn_metrics['pr_auc']:.4f}"
    )

    print()
    print(DIVIDER)
    print("Param count comparison")
    print(DIVIDER)
    print(f"  Reference total: {ref_params:>10,}  encoder: {ref_enc:>10,}")
    print(f"  RelNN total:     {rn_params:>10,}  encoder: {rn_enc:>10,}")
    enc_ok = ref_enc == rn_enc
    total_ok = ref_params == rn_params
    print(
        f"  Encoder [{'OK' if enc_ok else 'MISMATCH'}]  "
        f"Total [{'OK' if total_ok else 'MISMATCH'}] (total match expected for MLP only)"
    )

    print()
    print(DIVIDER)
    summary_title = f"{combo_tag} | {dec_label}" if combo_tag else dec_label
    print(f"SUMMARY ({summary_title})")
    print(DIVIDER)
    acc_delta = abs(ref_metrics["accuracy"] - rn_metrics["accuracy"])
    pr_delta = abs(ref_metrics["pr_auc"] - rn_metrics["pr_auc"])
    print(
        f"  Reference   test acc: {ref_metrics['accuracy']:.1%}  "
        f"ROC-AUC: {ref_metrics['roc_auc']:.4f}  PR-AUC: {ref_metrics['pr_auc']:.4f}  "
        f"({ref_params:,} params)  Time: {ref_time:.1f}s"
    )
    print(
        f"  RelNN       test acc: {rn_metrics['accuracy']:.1%}  "
        f"ROC-AUC: {rn_metrics['roc_auc']:.4f}  PR-AUC: {rn_metrics['pr_auc']:.4f}  "
        f"({rn_params:,} params)  Time: {rn_time:.1f}s"
    )
    roc_delta = abs(ref_metrics["roc_auc"] - rn_metrics["roc_auc"])
    print(
        f"  Delta acc: {acc_delta:.1%}  |  Delta ROC-AUC: {roc_delta:.4f}  "
        f"|  Delta PR-AUC: {pr_delta:.4f}"
    )
    print()

    assert ref_enc == rn_enc, (
        f"Encoder param mismatch ({decoder}): Reference={ref_enc}, RelNN={rn_enc}"
    )
    if decoder == "mlp":
        assert ref_params == rn_params, (
            f"Param count mismatch (MLP): Reference={ref_params}, RelNN={rn_params}"
        )
    assert acc_delta < 0.05, (
        f"Accuracy gap >5% ({decoder}): Reference={ref_metrics['accuracy']:.1%}, "
        f"RelNN={rn_metrics['accuracy']:.1%}"
    )


if __name__ == "__main__":
    # Avoid UnicodeEncodeError on Windows consoles (dataset repr uses box-drawing chars).
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    p = argparse.ArgumentParser(description="Compare PyTorch vs RelNN HyGNN on DDI data.")
    p.add_argument(
        "--decoder",
        choices=("all", "mlp", "dot"),
        default="all",
        help="Which decoder(s) to compare within each (source, feature) combo (default: all).",
    )
    p.add_argument(
        "--source",
        choices=("all", "TWOSIDES", "DrugBank"),
        default="all",
        help="Limit to one data source or run all (default: all).",
    )
    p.add_argument(
        "--substructure",
        choices=("all", "kmer", "espf"),
        default="all",
        help="Limit to k-mer (k=9) or ESPF, or all (default: all).",
    )
    p.add_argument(
        "--espf-support",
        type=int,
        default=5,
        choices=sorted(HYGNN_ESPF_SUPPORT_VALUES),
        metavar="N",
        help=(
            "ESPF min_support for the ESPF branch of the grid (default: 5). "
            f"Allowed: {', '.join(str(x) for x in sorted(HYGNN_ESPF_SUPPORT_VALUES))}. "
            "Passed through to load_hygnn_dataset."
        ),
    )
    args = p.parse_args()

    print(DIVIDER)
    print("HyGNN Comparison: Reference (PyTorch) vs RelNN first-order")
    print(DIVIDER)

    if args.decoder == "all":
        decoders: list[DecoderName] = ["mlp", "dot"]
    elif args.decoder == "mlp":
        decoders = ["mlp"]
    else:
        decoders = ["dot"]

    feature_grid = _hygnn_feature_grid(args.espf_support)

    sources = list(_HY_GNN_SOURCES) if args.source == "all" else [args.source]
    if args.substructure == "all":
        feature_rows = list(feature_grid)
    elif args.substructure == "kmer":
        feature_rows = [r for r in feature_grid if r[1]["substructure_method"] == "kmer"]
    else:
        feature_rows = [r for r in feature_grid if r[1]["substructure_method"] == "espf"]

    combos = list(product(sources, feature_rows, decoders))
    print(
        f"Running {len(combos)} comparison(s): "
        f"{len(sources)} source(s) × {len(feature_rows)} substructure(s) × {len(decoders)} decoder(s)"
    )
    print()

    for source, (feat_label, feat_kw) in product(sources, feature_rows):
        data = load_hygnn_dataset(
            source=source,
            d=128,
            seed=42,
            **feat_kw,
        )
        print(DIVIDER)
        print(f"DATA: {source}  |  {feat_label}")
        print(DIVIDER)
        print(data)
        print()

        combo_tag = f"{source} | {feat_label}"
        for dec in decoders:
            _run_compare_for_decoder(data, dec, combo_tag=combo_tag)
            print()

    print("Done.")
