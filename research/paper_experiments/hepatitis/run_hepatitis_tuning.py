"""Staged hyperparameter search for CTU Hepatitis binary classification.

Stage A: coarse grid (short epochs, single seed).
Stage B: confirm best config with more epochs and multiple seeds.

Run from repo root:
  python research/paper_experiments/hepatitis/run_hepatitis_tuning.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _load_train_fn():
    path = Path(__file__).resolve().parent / "run_hepatitis_multirun.py"
    spec = importlib.util.spec_from_file_location("hepatitis_multirun", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._train_and_eval_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a-epochs", type=int, default=30)
    parser.add_argument("--confirm-epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("research/paper_experiments/hepatitis/results/hepatitis_tuning.local.json"),
    )
    args = parser.parse_args()

    train_one = _load_train_fn()
    device = torch.device("cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu"))

    hidden_grid = [16, 32, 64]
    lr_grid = [0.003, 0.01, 0.02]
    wd_grid = [0.0, 1e-4]

    best_score = None
    best_meta = None
    stage_a = []

    for h in hidden_grid:
        for lr in lr_grid:
            for wd in wd_grid:
                row = train_one(
                    seed=42,
                    hidden=h,
                    epochs=args.stage_a_epochs,
                    lr=lr,
                    weight_decay=wd,
                    device=device,
                    cache_dir=args.cache_dir,
                )
                row["stage"] = "A"
                stage_a.append(row)
                score = float(row["metrics"].get("val_roc_auc", 0.0))
                print(f"  h={h} lr={lr} wd={wd}: val_auc={score:.4f}")
                if best_score is None or score > best_score:
                    best_score = score
                    best_meta = (h, lr, wd)

    print(f"\nBest stage-A: hidden={best_meta[0]}, lr={best_meta[1]}, wd={best_meta[2]} (val_auc={best_score:.4f})")

    confirm = []
    for i in range(args.seeds):
        row = train_one(
            seed=100 + i,
            hidden=best_meta[0],
            epochs=args.confirm_epochs,
            lr=best_meta[1],
            weight_decay=best_meta[2],
            device=device,
            cache_dir=args.cache_dir,
        )
        row["stage"] = "confirm"
        confirm.append(row)
        print(f"  confirm seed={100+i}: val={row['metrics']['val_roc_auc']:.4f} test={row['metrics']['test_roc_auc']:.4f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "best_stage_a": {"hidden": best_meta[0], "lr": best_meta[1], "weight_decay": best_meta[2], "val_auc": best_score},
        "stage_a_runs": stage_a,
        "confirm_runs": confirm,
    }
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
