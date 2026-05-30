"""Run reference HyGNN (PyTorch + torch_scatter, no DGL) on parent's HyGNN data.

Uses load_hygnn_dataset for data. Implements the paper's HyGNN with pure PyTorch
and torch_scatter for message passing. Supports stacking multiple HyGNN layers
(per-layer weights, matching RelNN high-order HyGNN). Prints metrics for comparison
with RelNN.

Default is **one** layer (same as the original script). Deeper stacks need a smaller
learning rate; ``--num-layers 2`` with ``lr=0.005`` often diverges (loss spikes, then
~0.693 BCE / chance-level metrics).

Usage:
  python research/paper_experiments/hygnn/run_hygnn_pytorch_ref.py
  python research/paper_experiments/hygnn/run_hygnn_pytorch_ref.py --decoder dot
  python research/paper_experiments/hygnn/run_hygnn_pytorch_ref.py --num-layers 2 --lr 0.001
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Add project root for parent imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relann.datasets import load_hygnn_dataset
from relann.torch_utils import full_seed


class HyGNNLayer(nn.Module):
    """One HyGNN block: hyperedge↔node attention + scatter (Saifuddin et al.)."""

    def __init__(
        self,
        drug_in_dim: int,
        query_dim: int,
        vertex_dim: int,
        edge_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.dropout = dropout
        self.drug_proj = nn.Linear(drug_in_dim, vertex_dim)
        self.w1 = nn.Linear(edge_dim, vertex_dim)
        self.w2 = nn.Linear(edge_dim, query_dim)
        self.w3 = nn.Linear(vertex_dim, query_dim)
        self.w4 = nn.Linear(vertex_dim, edge_dim)
        self.w5 = nn.Linear(vertex_dim, query_dim)
        self.w6 = nn.Linear(edge_dim, query_dim)

    def forward(
        self,
        sub_ids: torch.Tensor,
        drug_ids: torch.Tensor,
        vfeat: torch.Tensor,
        efeat: torch.Tensor,
        last_layer: bool,
        n_subs: int,
        n_drugs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat_e = self.drug_proj(efeat)
        feat_v = vfeat

        # Hyperedge -> node (drug -> substructure)
        k_drug = self.w2(feat_e)
        v_drug = self.w1(feat_e)
        q_sub = self.w3(feat_v)
        scores = F.leaky_relu(
            (k_drug[drug_ids] * q_sub[sub_ids]).sum(-1)
        ) / math.sqrt(self.query_dim)
        attn = scatter_softmax(scores, sub_ids, dim=0)
        feat_v = scatter_add(
            attn.unsqueeze(-1) * v_drug[drug_ids],
            sub_ids,
            dim=0,
            dim_size=n_subs,
        )

        # Node -> hyperedge (substructure -> drug)
        k_sub = self.w5(feat_v)
        v_sub = self.w4(feat_v)
        q_drug = self.w6(feat_e)
        scores = F.leaky_relu(
            (k_sub[sub_ids] * q_drug[drug_ids]).sum(-1)
        ) / math.sqrt(self.query_dim)
        attn = scatter_softmax(scores, drug_ids, dim=0)
        feat_e = scatter_add(
            attn.unsqueeze(-1) * v_sub[sub_ids],
            drug_ids,
            dim=0,
            dim_size=n_drugs,
        )

        if not last_layer:
            feat_v = F.dropout(feat_v, self.dropout, training=self.training)
        return feat_v, feat_e


def hygnn_encoder_param_count(model: "HyGNN") -> int:
    """Learnable HyGNN layer params only (excludes MLP decoder when present)."""
    return sum(p.numel() for p in model.layers.parameters())


class HyGNN(nn.Module):
    """HyGNN encoder stack + MLP or dot-product DDI decoder (official HyGNN notebook)."""

    class MLPPredictor(nn.Module):
        """concat(h_src, h_dst) -> 2-layer MLP -> score (same as RelNN MLP1/MLP2)."""

        def __init__(self, h_feats: int):
            super().__init__()
            self.w1 = nn.Linear(h_feats * 2, h_feats)
            self.w2 = nn.Linear(h_feats, 1)

        def forward(self, src: torch.Tensor, dst: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            x = torch.cat([h[src], h[dst]], dim=1)
            return self.w2(F.relu(self.w1(x))).squeeze(-1)

    def __init__(
        self,
        input_dim: int,
        query_dim: int,
        vertex_dim: int,
        edge_dim: int,
        dropout: float,
        num_layers: int = 1,
        decoder: Literal["mlp", "dot"] = "mlp",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if decoder not in ("mlp", "dot"):
            raise ValueError("decoder must be 'mlp' or 'dot'")
        self.decoder_mode: Literal["mlp", "dot"] = decoder
        layers: list[HyGNNLayer] = []
        drug_in = input_dim
        for _ in range(num_layers):
            layers.append(
                HyGNNLayer(drug_in, query_dim, vertex_dim, edge_dim, dropout)
            )
            drug_in = edge_dim
        self.layers = nn.ModuleList(layers)
        self.mlp_predictor: HyGNN.MLPPredictor | None
        if decoder == "mlp":
            self.mlp_predictor = HyGNN.MLPPredictor(edge_dim)
        else:
            self.mlp_predictor = None

    def forward(
        self,
        sub_ids: torch.Tensor,
        drug_ids: torch.Tensor,
        vfeat: torch.Tensor,
        efeat: torch.Tensor,
        n_subs: int,
        n_drugs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for i, layer in enumerate(self.layers):
            last = i == len(self.layers) - 1
            vfeat, efeat = layer(
                sub_ids,
                drug_ids,
                vfeat,
                efeat,
                last_layer=last,
                n_subs=n_subs,
                n_drugs=n_drugs,
            )
        return vfeat, efeat

    def decode(
        self, src: torch.Tensor, dst: torch.Tensor, drug_emb: torch.Tensor
    ) -> torch.Tensor:
        """DDI logits from drug embeddings ``drug_emb`` (n_drugs, edge_dim).

        ``dot`` matches DGL ``u_dot_v`` in the reference HyGNN_DDI notebook (inner product).
        """
        if self.mlp_predictor is not None:
            return self.mlp_predictor(src, dst, drug_emb)
        return (drug_emb[src] * drug_emb[dst]).sum(dim=-1)


def compute_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    *,
    decoder: Literal["mlp", "dot"] = "mlp",
) -> torch.Tensor:
    """BCE training loss on positive vs negative pair scores.

    MLP decoder outputs logits → ``binary_cross_entropy_with_logits``.
    Dot decoder uses inner-product scores → apply **sigmoid** then ``binary_cross_entropy``
    (probability form), matching the RelNN dot program (``BCELoss`` on ``Sigmoid``(dot)).
    """
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    if decoder == "dot":
        return F.binary_cross_entropy(torch.sigmoid(scores), labels)
    return F.binary_cross_entropy_with_logits(scores, labels)


def main() -> None:
    p = argparse.ArgumentParser(description="Reference HyGNN (multi-layer) on TWOSIDES data.")
    p.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help=(
            "Stacked HyGNN layers (default 1 = original single-pass behavior). "
            "For 2+ layers use a lower --lr (e.g. 0.001) to avoid divergence."
        ),
    )
    p.add_argument("--epochs", type=int, default=500, help="Training epochs.")
    p.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Adam learning rate (default 0.005; reduce for multi-layer).",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument(
        "--decoder",
        type=str,
        choices=("mlp", "dot"),
        default="mlp",
        help="DDI decoder: MLP (paper Eq. 11) or dot product (official notebook DotPredictor).",
    )
    args = p.parse_args()

    full_seed(args.seed)

    # Load data (same as RelNN notebook)
    data = load_hygnn_dataset(source="TWOSIDES", k=3, d=128, seed=args.seed)
    info = data.dataset_info
    n_drugs = info["n_drugs"]
    n_subs = info["n_subs"]

    # Extract from DB
    _, drug_tensor = data.db["Drug"]
    _, sub_tensor = data.db["Substructure"]
    inc_df, _ = data.db["Incidence"]
    train_df, train_labels = data.db["TrainPairs"]
    test_df, test_labels_tensor = data.db["TestPairs"]

    # Incidence: (sub_id, drug_id) per edge
    sub_ids = torch.tensor(inc_df["sub_id"].values, dtype=torch.long)
    drug_ids = torch.tensor(inc_df["drug_id"].values, dtype=torch.long)

    # Features: drug identity (n_drugs, n_drugs), substructure ones (n_subs, 128)
    efeat = drug_tensor  # (n_drugs, n_drugs)
    vfeat = sub_tensor   # (n_subs, 128)

    # Train pairs: split pos/neg for BCE
    train_d1 = train_df["drug1"].values
    train_d2 = train_df["drug2"].values
    train_labels_np = train_labels.numpy().flatten()
    train_pos_mask = train_labels_np == 1.0
    train_neg_mask = ~train_pos_mask
    train_pos_src = torch.tensor(train_d1[train_pos_mask], dtype=torch.long)
    train_pos_dst = torch.tensor(train_d2[train_pos_mask], dtype=torch.long)
    train_neg_src = torch.tensor(train_d1[train_neg_mask], dtype=torch.long)
    train_neg_dst = torch.tensor(train_d2[train_neg_mask], dtype=torch.long)

    # Test pairs
    test_src = torch.tensor(test_df["drug1"].values, dtype=torch.long)
    test_dst = torch.tensor(test_df["drug2"].values, dtype=torch.long)
    test_labels = test_labels_tensor.numpy().flatten()

    # Model (qd=64, vertex=128, edge=128, dropout=0.5; drug input_dim = n_drugs)
    model = HyGNN(
        input_dim=n_drugs,
        query_dim=64,
        vertex_dim=128,
        edge_dim=128,
        dropout=0.5,
        num_layers=args.num_layers,
        decoder=args.decoder,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    for e in range(args.epochs):
        model.train()
        feat_v, feat_e = model(
            sub_ids, drug_ids, vfeat, efeat,
            n_subs=n_subs, n_drugs=n_drugs,
        )
        h = feat_e  # drug embeddings
        pos_score = model.decode(train_pos_src, train_pos_dst, h)
        neg_score = model.decode(train_neg_src, train_neg_dst, h)
        loss = compute_loss(pos_score, neg_score, decoder=args.decoder)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e % (args.epochs / 10) == 0 or e == args.epochs - 1:
            print(f"Epoch {e}, loss: {loss.item():.4f}")

    # Eval
    model.eval()
    with torch.no_grad():
        _, feat_e = model(
            sub_ids, drug_ids, vfeat, efeat,
            n_subs=n_subs, n_drugs=n_drugs,
        )
        h = feat_e
        raw_logits = model.decode(test_src, test_dst, h)

        # Convert logits to probabilities (0.0 to 1.0)
        pred_probs = torch.sigmoid(raw_logits).numpy()

    # Now a 0.5 threshold mathematically represents 50% confidence
    pred_binary = (pred_probs >= 0.5).astype(int)

    acc = accuracy_score(test_labels, pred_binary)
    prec = precision_score(test_labels, pred_binary, zero_division=0)
    rec = recall_score(test_labels, pred_binary, zero_division=0)
    f1 = f1_score(test_labels, pred_binary, zero_division=0)

    # Standard practice is to feed probabilities to AUC functions, not logits
    roc_auc = roc_auc_score(test_labels, pred_probs)
    pr_auc = average_precision_score(test_labels, pred_probs)

    print(
        f"\n--- Reference HyGNN (PyTorch + torch_scatter), "
        f"num_layers={args.num_layers}, lr={args.lr}, decoder={args.decoder} ---"
    )
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

if __name__ == "__main__":
    main()
