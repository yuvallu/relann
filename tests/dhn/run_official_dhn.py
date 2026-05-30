r"""
Run the official DHN implementation (github.com/gear/dhn) for ENZYMES and PROTEINS
with 10-fold stratified CV and report mean +/- std test accuracy.

Usage (from repo root):
  python scripts/setup_external_dhn.py
  python tests/dhn/run_official_dhn.py --data_root ./data

Requires: torch, torch_geometric, sklearn, tqdm, pyyaml, networkx.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Put this script's directory on sys.path so sibling dhn_external.py is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dhn_external import (
    DHN_PINNED_COMMIT,
    DHN_REPO_URL,
    ensure_external_dhn_on_path,
    external_dhn_root,
)

ensure_external_dhn_on_path()


def _check_dhn():
    try:
        import torch
        from dhn.models import DHN
        from dhn.datasets import HomDataset, HomDataLoader, hom_collate
        from dhn.utils import get_act_module, get_criterion, get_optimizer, get_lr_scheduler
        return True, None
    except ImportError as e:
        return False, e


def run_official_dhn(
    dataset_name: str,
    data_root: str = "./data",
    seed: int = 0,
    epochs: int = 100,
    batch_size: int = 32,
    early_stopping_patience: int = 5,
    n_folds: int = 10,
    verbose: bool = True,
):
    """Run 10-fold CV with official DHN; return (mean_acc, std_acc)."""
    import torch
    from sklearn.model_selection import StratifiedKFold
    from tqdm import tqdm

    from dhn.models import DHN
    from dhn.datasets import HomDataset, HomDataLoader, hom_collate
    from dhn.utils import get_act_module, get_criterion, get_optimizer, get_lr_scheduler

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # HomDataset expects root_path = data root; it loads TUDataset(root=root_path, name=dataset_name)
    root_path = str(Path(data_root).resolve())

    # Load dataset (creates .pt cache on first run)
    dataset = HomDataset(name=dataset_name, root_path=root_path)
    num_classes = dataset.num_classes
    num_features = dataset.num_features

    # Config matching gear/dhn default.yaml: 2 layers C2,C3,C4; (in_dim, out_dim, pattern_size).
    # Layer 1: 3 patterns * 5 = 15; layer 2: 15; concat per layer then next layer, so final dim = 15.
    layers_config = [
        {"c2": [num_features, 5, 2], "c3": [num_features, 5, 3], "c4": [num_features, 5, 4]},
        {"c2": [15, 5, 2], "c3": [15, 5, 3], "c4": [15, 5, 4]},
    ]
    agg = [num_classes]  # Linear(15, num_classes) for graph-level logits

    labels = np.array([dataset[i].y.item() for i in range(len(dataset))])
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        train_dataset = [dataset[i] for i in train_idx]
        test_dataset = [dataset[i] for i in test_idx]
        train_loader = HomDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = HomDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = DHN(
            out_dim=num_classes,
            layers_config=layers_config,
            act_module=get_act_module("ReLU"),
            agg=agg,
            inplace=False,
            p=0.05,
        ).to(device)
        criterion = get_criterion("CrossEntropyLoss")(reduction="mean")
        optimizer = get_optimizer("AdamW")(
            model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01
        )
        scheduler = get_lr_scheduler("ExponentialLR")(optimizer, gamma=0.9)

        best_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            for gdata in train_loader:
                gdata = gdata.to(device)
                optimizer.zero_grad()
                out = model(gdata)
                loss = criterion(out, gdata.y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for gdata in test_loader:
                    gdata = gdata.to(device)
                    pred = model(gdata).argmax(1)
                    correct += (pred == gdata.y).sum().item()
                    total += gdata.y.size(0)
            acc = correct / total if total else 0.0
            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"  Fold {fold+1}: early stop at epoch {epoch+1}, best test acc={best_acc*100:.1f}%")
                break
        else:
            if verbose:
                print(f"  Fold {fold+1}: best test acc={best_acc*100:.1f}%")
        fold_accs.append(best_acc * 100.0)

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    return mean_acc, std_acc


def main():
    parser = argparse.ArgumentParser(description="Run official DHN 10-fold CV on ENZYMES and PROTEINS")
    parser.add_argument("--data_root", type=str, default="./data", help="Root for TUDataset (ENZYMES, PROTEINS)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dataset", type=str, default=None, choices=["ENZYMES", "PROTEINS", None],
                        help="Single dataset or both if not set")
    args = parser.parse_args()

    ok, err = _check_dhn()
    if not ok:
        dhn_root = external_dhn_root()
        print("Official DHN package not found.")
        print(f"Expected checkout: {dhn_root}")
        print("Bootstrap command:")
        print(
            "  python "
            "scripts/setup_external_dhn.py"
        )
        print(f"Repo: {DHN_REPO_URL}")
        print(f"Pinned commit: {DHN_PINNED_COMMIT}")
        print(f"ImportError: {err}")
        sys.exit(1)

    datasets = [args.dataset] if args.dataset else ["ENZYMES", "PROTEINS"]
    print("Official DHN (gear/dhn) 10-fold CV")
    print("Config: 100 epochs, early_stopping=5, AdamW lr=0.001 weight_decay=0.01, ExponentialLR gamma=0.9")
    for name in datasets:
        print(f"\n{name}:")
        try:
            mean_acc, std_acc = run_official_dhn(
                dataset_name=name,
                data_root=args.data_root,
                seed=args.seed,
                epochs=args.epochs,
                verbose=True,
            )
            print(f"  Result: {mean_acc:.1f} ± {std_acc:.1f}%")
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
