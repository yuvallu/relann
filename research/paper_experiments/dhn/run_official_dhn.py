r"""
Run official DHN (gear/dhn, pinned commit) on CSL or EXP with C2:4 or C2:10 (single layer, cycles C2..Ck).

**Canonical:** C2:4 (default --max-cycle=4) matches RelNN GHL canonical configuration for paper.

**Training protocol** matches RelNN ``run_dhn_ghl.py`` with train accuracy on all graphs (single fit).

500 epochs, full-batch AdamW (lr=1e-3, weight_decay=0), no scheduler, no early stopping.

Preprocessing (``cycle_mapping_index``) is timed separately. The **paper table** uses wall-clock of
define+train+predict on full batch (here: model init + 500 epochs + train-acc eval).

**Device:** Default CPU (matches RelNN timing); use ``--device cuda`` for GPU.

Usage (repo root)::

  python scripts/setup_external_dhn.py
  python research/paper_experiments/dhn/run_official_dhn.py --dataset CSL --data_root ./data
  python research/paper_experiments/dhn/run_official_dhn.py --dataset CSL --max-cycle 10
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# Repo root (parent of nbs/)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "nbs" / "tests" / "dhn"))

from dhn_external import DHN_PINNED_COMMIT, DHN_REPO_URL, ensure_external_dhn_on_path  # noqa: E402

ensure_external_dhn_on_path()

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from dhn.graph_enumerations import cycle_mapping_index  # noqa: E402
from dhn.datasets import HomDataLoader  # noqa: E402
from dhn.models import DHN  # noqa: E402
from dhn.utils import get_act_module, get_criterion, get_optimizer  # noqa: E402

from dhn_utils import _download_exp_pkl, _pyg_data_to_nx  # noqa: E402


def _build_hom_data_list(
    graphs: list,
    labels: torch.Tensor,
    xs: List[torch.Tensor],
    cycle_length_bound: int = 10,
) -> List[Data]:
    """Build ``mapping_index_dict`` with cycles only (C2:10 — no clique enumeration)."""
    _tmp_data = []
    _all_patterns = set()
    for g, x, y in zip(graphs, xs, labels):
        mappings = cycle_mapping_index(g, length_bound=cycle_length_bound)
        _all_patterns.update(mappings.keys())
        yt = y.view(1).long() if y.dim() == 0 else y.view(-1).long()
        _tmp_data.append((x, yt, mappings))

    out: List[Data] = []
    for x, y, mappings in _tmp_data:
        for k in _all_patterns:
            if k not in mappings:
                mappings[k] = None
        out.append(Data(x=x, y=y, mapping_index_dict=mappings))
    return out


def _load_csl(data_root: str) -> Tuple[list, torch.Tensor, List[torch.Tensor], int, int]:
    from torch_geometric.datasets import GNNBenchmarkDataset

    ds = GNNBenchmarkDataset(root=data_root, name="CSL")
    graphs = [to_networkx(d, to_undirected=True) for d in ds]
    nf = ds.num_node_features
    if nf is None or nf == 0:
        nf = 1
    xs = []
    for d in ds:
        n = d.num_nodes if hasattr(d, "num_nodes") else int(d.edge_index.max().item()) + 1
        if getattr(d, "x", None) is not None:
            xs.append(d.x.float())
        else:
            xs.append(torch.ones((n, nf), dtype=torch.float32))
    labels = torch.tensor([d.y.item() for d in ds], dtype=torch.long)
    return graphs, labels, xs, ds.num_classes, nf


def _load_exp(data_root: str) -> Tuple[list, torch.Tensor, List[torch.Tensor], int, int]:
    pkl_path = _download_exp_pkl(data_root)
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    graphs = [_pyg_data_to_nx(d) for d in raw]
    xs = []
    labels_list = []
    for d in raw:
        dd = d.__dict__
        x_old = dd.get("x")
        if x_old is None:
            raise RuntimeError("EXP graph missing node features in pickle.")
        xs.append(x_old.float())
        y_old = dd.get("y")
        labels_list.append(int(y_old.item()) if y_old.numel() == 1 else int(y_old[0].item()))
    labels = torch.tensor(labels_list, dtype=torch.long)
    nf = int(xs[0].shape[1])
    return graphs, labels, xs, 2, nf


def layers_config_c2_through(num_features: int, max_cycle: int, d_k: int = 5) -> dict:
    """Single-layer DHN with cycle patterns C2..C{max_cycle} (e.g. max_cycle=10 -> C2:10)."""
    if max_cycle < 2 or max_cycle > 20:
        raise ValueError(f"max_cycle must be in [2, 20], got {max_cycle}")
    return {f"c{k}": [num_features, d_k, k] for k in range(2, max_cycle + 1)}


def layers_config_c2_10(num_features: int, d_k: int = 5) -> dict:
    return layers_config_c2_through(num_features, max_cycle=10, d_k=d_k)


def _resolve_device(name: str) -> torch.device:
    n = (name or "cpu").lower()
    if n in ("cpu",):
        return torch.device("cpu")
    if n in ("cuda", "gpu", "cuda:0"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if n == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_official_dhn_c2_10_train(
    dataset_name: str,
    data_root: str,
    epochs: int = 500,
    scale_to_epochs: int | None = None,
    seed: int = 42,
    d_k: int = 5,
    dropout_p: float = 0.05,
    device: str = "cpu",
    verbose: bool = True,
    max_cycle: int = 10,
    max_graphs: int | None = None,
) -> dict:
    """One full-dataset training run; train accuracy at end (matches RelNN synthetic timing).

    max_cycle: largest cycle pattern (4 -> C2:4, 10 -> C2:10).
    max_graphs: if set, use only the first N graphs (same order as loader).
    """
    dev = _resolve_device(device)
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    name_u = dataset_name.upper()
    if name_u == "CSL":
        graphs, labels, xs, num_classes, num_features = _load_csl(data_root)
    elif name_u == "EXP":
        graphs, labels, xs, num_classes, num_features = _load_exp(data_root)
    else:
        raise ValueError("dataset must be CSL or EXP")

    if max_graphs is not None and max_graphs > 0 and len(graphs) > max_graphs:
        graphs = graphs[:max_graphs]
        labels = labels[:max_graphs]
        xs = xs[:max_graphs]
        if verbose:
            print(f"Using first {max_graphs} graphs only (subset).")

    if verbose:
        print(f"Dataset {name_u}: {len(graphs)} graphs, num_classes={num_classes}, num_features={num_features}")
        print(f"Device: {dev}, official DHN commit: {DHN_PINNED_COMMIT}")
        print("Protocol: train on ALL graphs (same as run_pure_benchmarks eval_mode=auto for CSL/EXP)")

    t_pre0 = time.perf_counter()
    data_list = _build_hom_data_list(graphs, labels, xs, cycle_length_bound=max_cycle)
    preprocess_s = time.perf_counter() - t_pre0
    if verbose:
        print(f"Preprocess (hom mappings): {preprocess_s:.2f}s")

    loader = HomDataLoader(data_list, batch_size=len(data_list), shuffle=True)
    lc = [layers_config_c2_through(num_features, max_cycle=max_cycle, d_k=d_k)]
    agg = [num_classes]

    t_train0 = time.perf_counter()
    model = DHN(
        out_dim=num_classes,
        layers_config=lc,
        act_module=get_act_module("ReLU"),
        agg=agg,
        inplace=False,
        p=dropout_p,
    ).to(dev)
    criterion = get_criterion("CrossEntropyLoss")(reduction="mean")
    optimizer = get_optimizer("AdamW")(
        model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0
    )

    for _epoch in range(epochs):
        model.train()
        for gdata in loader:
            gdata = gdata.to(dev)
            optimizer.zero_grad()
            out = model(gdata)
            loss = criterion(out, gdata.y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        for gdata in loader:
            gdata = gdata.to(dev)
            pred = model(gdata).argmax(dim=1)
            train_acc = 100.0 * (pred == gdata.y).float().mean().item()

    train_eval_s = time.perf_counter() - t_train0
    scale_factor = None
    train_eval_scaled_s = None
    if scale_to_epochs is not None and scale_to_epochs > 0 and epochs > 0:
        scale_factor = float(scale_to_epochs) / float(epochs)
        train_eval_scaled_s = train_eval_s * scale_factor
        if verbose:
            print(
                f"Scaled train+eval to {scale_to_epochs} epochs (×{scale_factor:.4f}): "
                f"{train_eval_scaled_s:.2f}s"
            )
    if verbose:
        print(f"Train+eval wall time ({epochs} ep): {train_eval_s:.2f}s  |  train accuracy: {train_acc:.2f}%")

    cfg_label = f"C2:{max_cycle}"
    out = {
        "dataset": name_u,
        "config": cfg_label,
        "max_cycle": max_cycle,
        "eval_protocol": "train_all_graphs_single_run_matches_run_pure_benchmarks_auto",
        "epochs": epochs,
        "seed": seed,
        "d_k": d_k,
        "dropout_p": dropout_p,
        "preprocess_wall_s": preprocess_s,
        "train_eval_wall_s": train_eval_s,
        "train_accuracy_pct": train_acc,
        "official_dhn_repo": DHN_REPO_URL,
        "official_dhn_commit": DHN_PINNED_COMMIT,
        "device": str(dev),
        "note": (
            f"cycle_mapping_index only (C2..C{max_cycle}); clique patterns omitted."
        ),
    }
    if scale_to_epochs is not None:
        out["scale_to_epochs"] = scale_to_epochs
        out["train_eval_wall_s_scaled"] = train_eval_scaled_s
        out["scaling_note"] = (
            f"train_eval_wall_s × ({scale_to_epochs}/{epochs}) for table comparison with 500-epoch RelNN runs"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Official DHN CSL/EXP C2:10 timing (train-all, RelNN-aligned)")
    parser.add_argument("--dataset", type=str, required=True, choices=["CSL", "EXP"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_k", type=int, default=5)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu (default, matches RelNN table timing), cuda, or auto",
    )
    parser.add_argument(
        "--scale-to-epochs",
        type=int,
        default=None,
        metavar="N",
        help="If set, multiply train_eval_wall_s by N/epochs (e.g. run 50 ep, scale to 500)",
    )
    parser.add_argument("--json_out", type=str, default=None, help="Write results JSON to this path")
    parser.add_argument(
        "--max-cycle",
        type=int,
        default=4,
        metavar="K",
        help="Largest cycle pattern C2..CK (default 4 = C2:4 canonical; use 10 for C2:10).",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        metavar="N",
        help="Train on only the first N graphs (subset for smoke / memory).",
    )
    args = parser.parse_args()

    try:
        import dhn  # noqa: F401
    except ImportError:
        print("Could not import official `dhn` package.")
        print(f"Expected clone at: {_REPO_ROOT / '_external' / 'dhn'}")
        print("Run: python scripts/setup_external_dhn.py")
        sys.exit(1)

    results = run_official_dhn_c2_10_train(
        args.dataset,
        data_root=args.data_root,
        epochs=args.epochs,
        scale_to_epochs=args.scale_to_epochs,
        seed=args.seed,
        d_k=args.d_k,
        device=args.device,
        verbose=True,
        max_cycle=args.max_cycle,
        max_graphs=args.max_graphs,
    )
    print("\n=== Summary ===")
    print(json.dumps(results, indent=2))
    if args.json_out:
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
