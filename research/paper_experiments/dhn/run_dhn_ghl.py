r"""
Train GHL (Graph Homomorphism Layer) on CSL or EXP datasets.

Supports C2:4 (edge-join) and C2:10 (simple-cycles precompute) configurations.

**C2:4 (default):** ``build_dhn_db_edge`` with inline Edge-join enumeration and
``dhn_ghl_csl_c2_4.relnn``.

**C2:10:** ``build_dhn_db(..., simple_cycles=True)`` with precomputed simple-cycles
and ``dhn_ghl_csl_c2_10.relnn``.

Usage (repo root)::

  python research/paper_experiments/dhn/run_dhn_ghl.py --dataset CSL --config c2_4 --epochs 500
  python research/paper_experiments/dhn/run_dhn_ghl.py --dataset EXP --config c2_10 --epochs 500
  python research/paper_experiments/dhn/run_dhn_ghl.py --dataset CSL --config c2_4 --max-graphs 10 --json-out results/test.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "nbs" / "tests" / "dhn"))

from dhn_utils import build_dhn_db, build_dhn_db_edge, load_csl_dataset, load_exp_dataset  # noqa: E402
from relann.session import Session  # noqa: E402
from relann.torch_utils import full_seed  # noqa: E402


_PAPER_EXPS_DIR = Path(__file__).resolve().parent
_RELNN_C2_4 = _PAPER_EXPS_DIR / "dhn_ghl_csl_c2_4.relnn"
_RELNN_C2_10 = _PAPER_EXPS_DIR / "dhn_ghl_csl_c2_10.relnn"


def split_relnn_file(path: Path) -> tuple[str, str, str]:
    """Split .relnn file into [define, fit, pred] blocks by #lang:relnn markers."""
    text = path.read_text(encoding="utf-8")
    parts = text.split("#lang:relnn")
    blocks: list[str] = []
    for chunk in parts[1:]:
        chunk = chunk.strip()
        if chunk:
            blocks.append("#lang:relnn\n" + chunk)
    if len(blocks) != 3:
        raise ValueError(
            f"Expected 3 #lang:relnn blocks in {path}, found {len(blocks)}"
        )
    return blocks[0], blocks[1], blocks[2]


def _fit_prog_for_epochs(base_fit: str, epochs: int) -> str:
    """Adapt fit program to requested epoch count."""
    if epochs != 500:
        return base_fit.replace("<epochs=500,", f"<epochs={epochs},")
    return base_fit


def run_ghl_train_pred(
    *,
    config: str,
    dataset: str,
    graphs: list,
    labels: torch.Tensor,
    graph_ids: list[int],
    epochs: int,
    seed: int,
    device: str | None,
) -> dict:
    """Single define → fit → pred pass; reproducible for given seed."""
    if config == "c2_4":
        relnn_path = _RELNN_C2_4
        patterns = ["C2", "C3", "C4"]
        use_edge = True
    elif config == "c2_10":
        relnn_path = _RELNN_C2_10
        patterns = ["C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]
        use_edge = False
    else:
        raise ValueError(f"Unknown config: {config}")

    define_prog, fit_prog, pred_prog = split_relnn_file(relnn_path)
    fit_prog = _fit_prog_for_epochs(fit_prog, epochs)

    if use_edge:
        db = build_dhn_db_edge(graphs, labels=labels, graph_ids=graph_ids)
    else:
        db = build_dhn_db(
            graphs, patterns, labels=labels, graph_ids=graph_ids, simple_cycles=True
        )

    full_seed(seed)
    t0 = time.perf_counter()
    session = Session(db=db, device=device)
    session.run(define_prog)
    session.run(fit_prog)
    result = session.run(pred_prog)
    elapsed = time.perf_counter() - t0

    preds = result.embeddings[0].view(-1).long().cpu()
    labels_cpu = labels.cpu().view(-1)
    acc = 100.0 * (preds == labels_cpu).sum().item() / len(labels_cpu)

    return {
        "config": config,
        "dataset": dataset,
        "num_graphs": len(graphs),
        "train_acc_pct": acc,
        "wall_s": elapsed,
        "relnn": str(relnn_path.name),
        "seed": seed,
        "epochs": epochs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=str,
        default="CSL",
        choices=["CSL", "EXP"],
        help="Dataset to use",
    )
    p.add_argument(
        "--config",
        type=str,
        default="c2_4",
        choices=["c2_4", "c2_10"],
        help="Config: c2_4 (edge) or c2_10 (simple-cycles)",
    )
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="e.g. cuda or cpu")
    p.add_argument("--max-graphs", type=int, default=None, help="First N graphs only")
    p.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Write results to JSON file",
    )
    args = p.parse_args()

    # Load dataset
    if args.dataset == "CSL":
        graphs, labels, _, _ = load_csl_dataset(root=args.data_root)
    else:
        graphs, labels, _, _ = load_exp_dataset(root=args.data_root)

    if args.max_graphs is not None and args.max_graphs > 0:
        graphs = graphs[: args.max_graphs]
        labels = labels[: args.max_graphs]
    gids = list(range(len(graphs)))

    # Run
    result = run_ghl_train_pred(
        config=args.config,
        dataset=args.dataset,
        graphs=graphs,
        labels=labels,
        graph_ids=gids,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )

    # Print summary
    print(
        f"GHL {args.dataset} {args.config.upper()} | graphs={result['num_graphs']} | "
        f"acc={result['train_acc_pct']:.2f}% | wall_s={result['wall_s']:.2f}"
    )
    print(f"  relnn: {result['relnn']}")
    print(f"  device: {args.device or 'default'}")

    # Optionally write JSON
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  results: {out_path}")


if __name__ == "__main__":
    main()
