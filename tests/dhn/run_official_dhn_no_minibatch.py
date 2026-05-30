r"""
Run the official DHN with and without minibatching on ENZYMES and PROTEINS.

Compares:
  1. Official config: batch_size=32, ExponentialLR gamma=0.9, early_stopping=5
  2. Full-batch: batch_size=full, same LR scheduler + early stopping
  3. Full-batch + no scheduler: batch_size=full, constant LR, no early stopping, 500 epochs

Usage (from repo root):
  python scripts/setup_external_dhn.py
  python tests/dhn/run_official_dhn_no_minibatch.py --data_root ./data
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

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def run_official_dhn_configurable(
    dataset_name: str,
    data_root: str = "./data",
    seed: int = 0,
    epochs: int = 100,
    batch_size: int = 32,
    early_stopping_patience: int = 5,
    use_scheduler: bool = True,
    scheduler_gamma: float = 0.9,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    n_folds: int = 10,
    verbose: bool = True,
):
    """Run 10-fold CV with official DHN; return (mean_acc, std_acc, fold_accs)."""
    import torch
    from sklearn.model_selection import StratifiedKFold

    from dhn.models import DHN
    from dhn.datasets import HomDataset, HomDataLoader
    from dhn.utils import get_act_module, get_criterion, get_optimizer, get_lr_scheduler

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    root_path = str(Path(data_root).resolve())

    dataset = HomDataset(name=dataset_name, root_path=root_path)
    num_classes = dataset.num_classes
    num_features = dataset.num_features

    layers_config = [
        {"c2": [num_features, 5, 2], "c3": [num_features, 5, 3], "c4": [num_features, 5, 4]},
        {"c2": [15, 5, 2], "c3": [15, 5, 3], "c4": [15, 5, 4]},
    ]
    agg = [num_classes]

    labels = np.array([dataset[i].y.item() for i in range(len(dataset))])
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        train_dataset = [dataset[i] for i in train_idx]
        test_dataset = [dataset[i] for i in test_idx]

        effective_bs = batch_size if batch_size > 0 else len(train_dataset)
        train_loader = HomDataLoader(train_dataset, batch_size=effective_bs, shuffle=True)
        test_loader = HomDataLoader(test_dataset, batch_size=max(batch_size, len(test_dataset)), shuffle=False)

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
            model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay
        )
        scheduler = None
        if use_scheduler:
            scheduler = get_lr_scheduler("ExponentialLR")(optimizer, gamma=scheduler_gamma)

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
            if scheduler is not None:
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
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"    Fold {fold+1}: early stop at epoch {epoch+1}, best={best_acc*100:.1f}%")
                break
        else:
            if verbose:
                print(f"    Fold {fold+1}: epoch {epochs}, best={best_acc*100:.1f}%")
        fold_accs.append(best_acc * 100.0)

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    return mean_acc, std_acc, fold_accs


CONFIGS = {
    "official (bs=32, sched, es=5)": dict(
        batch_size=32, epochs=100, early_stopping_patience=5,
        use_scheduler=True, scheduler_gamma=0.9,
        lr=0.001, weight_decay=0.01,
    ),
    "full-batch + sched + es=5": dict(
        batch_size=0, epochs=100, early_stopping_patience=5,
        use_scheduler=True, scheduler_gamma=0.9,
        lr=0.001, weight_decay=0.01,
    ),
    "full-batch, no sched, no es, 500ep": dict(
        batch_size=0, epochs=500, early_stopping_patience=0,
        use_scheduler=False,
        lr=0.001, weight_decay=0.01,
    ),
    "full-batch, Adam, no sched, 500ep": dict(
        batch_size=0, epochs=500, early_stopping_patience=0,
        use_scheduler=False,
        lr=0.001, weight_decay=0.0,
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, default=None, choices=["ENZYMES", "PROTEINS"])
    parser.add_argument("--config", type=str, default=None, choices=list(CONFIGS.keys()),
                        help="Run single config; default runs all")
    args = parser.parse_args()

    try:
        from dhn.models import DHN
    except ImportError as e:
        print("Official DHN source not found under _external/dhn.")
        print(f"Expected checkout: {external_dhn_root()}")
        print("Bootstrap command:")
        print(
            "  python "
            "scripts/setup_external_dhn.py"
        )
        print(f"Repo: {DHN_REPO_URL}")
        print(f"Pinned commit: {DHN_PINNED_COMMIT}")
        print(f"  Error: {e}")
        sys.exit(1)

    datasets = [args.dataset] if args.dataset else ["ENZYMES", "PROTEINS"]
    configs = {args.config: CONFIGS[args.config]} if args.config else CONFIGS

    print("=" * 80)
    print("DHN Official Implementation: Minibatch Ablation Study")
    print("=" * 80)

    all_results = {}
    for ds_name in datasets:
        print(f"\n{'='*40}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*40}")
        all_results[ds_name] = {}

        for cfg_name, cfg in configs.items():
            print(f"\n  Config: {cfg_name}")
            print(f"  {cfg}")
            try:
                mean_acc, std_acc, fold_accs = run_official_dhn_configurable(
                    dataset_name=ds_name,
                    data_root=args.data_root,
                    seed=args.seed,
                    verbose=True,
                    **cfg,
                )
                all_results[ds_name][cfg_name] = (mean_acc, std_acc)
                print(f"  >> {ds_name} [{cfg_name}]: {mean_acc:.1f} ± {std_acc:.1f}%")
            except Exception as e:
                print(f"  >> FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_results[ds_name][cfg_name] = None

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for ds_name, results in all_results.items():
        print(f"\n{ds_name}:")
        for cfg_name, result in results.items():
            if result is not None:
                mean_acc, std_acc = result
                print(f"  {cfg_name:<45} {mean_acc:5.1f} ± {std_acc:.1f}%")
            else:
                print(f"  {cfg_name:<45} FAILED")

    print("\nRelNN comparison (from BENCHMARK_RESULTS.md):")
    print("  ENZYMES count-MLP 500ep Adam:  51.7 ± 6.2%")
    print("  PROTEINS count-MLP 500ep Adam: 71.2 ± 3.8%")
    print("  Paper reported:  ENZYMES 64.3 ± 5.5%, PROTEINS 76.5 ± 3.0%")


if __name__ == "__main__":
    main()
