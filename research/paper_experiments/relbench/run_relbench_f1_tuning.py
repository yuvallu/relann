"""Staged hyperparameter search for RelBench rel-f1 (driver-position, driver-dnf).

Stage A: coarse grid over ``hidden`` and ``lr`` (short epochs).
Stage B: confirm best config with ``--confirm-epochs`` and ``--seeds`` runs.

Requires: ``pip install relbench``

Run from repo root:
  python research/paper_experiments/relbench/run_relbench_f1_tuning.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

def _load_train_fn():
    path = Path(__file__).resolve().parent / "run_relbench_f1_multirun.py"
    spec = importlib.util.spec_from_file_location("relbench_multirun", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._train_and_eval_one


def main():
    __import__("pytest").importorskip("relbench")

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["driver-position", "driver-dnf"])
    parser.add_argument("--stage-a-epochs", type=int, default=15)
    parser.add_argument("--confirm-epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--extra-tables",
        action="store_true",
        help="Same as run_relbench_f1_multirun: merge qualifying+standings into Results features.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("research/paper_experiments/relbench/results/relbench_f1_tuning.json"),
    )
    args = parser.parse_args()

    train_one = _load_train_fn()
    device = torch.device("cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu"))
    hidden_grid = [16, 32, 64]
    lr_grid = [0.003, 0.005, 0.01, 0.02]
    wd_grid = [0.0, 1e-4, 5e-4]

    best_by_task: dict = {}
    stage_a: list = []

    for task_name in args.tasks:
        best_score = None
        best_meta = None
        for h in hidden_grid:
            for lr in lr_grid:
                for wd in wd_grid:
                    row = train_one(
                        task_name,
                        seed=42,
                        hidden=h,
                        epochs=args.stage_a_epochs,
                        lr=lr,
                        weight_decay=wd,
                        device=device,
                        extra_tables=args.extra_tables,
                    )
                    row["stage"] = "A"
                    stage_a.append(row)
                    m = row["metrics"]
                    if "mae" in m:
                        score = -float(m["mae"])
                    else:
                        score = float(m.get("roc_auc", m.get("auroc", 0.0)))
                    if best_score is None or score > best_score:
                        best_score = score
                        best_meta = (h, lr, wd, row)
        assert best_meta is not None
        best_by_task[task_name] = {
            "hidden": best_meta[0],
            "lr": best_meta[1],
            "weight_decay": best_meta[2],
            "metrics": best_meta[3]["metrics"],
        }

    confirm: list = []
    seed0 = 100
    for task_name in args.tasks:
        cfg = best_by_task[task_name]
        for i in range(args.seeds):
            row = train_one(
                task_name,
                seed=seed0 + i,
                hidden=cfg["hidden"],
                epochs=args.confirm_epochs,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                device=device,
                extra_tables=args.extra_tables,
            )
            row["stage"] = "confirm"
            confirm.append(row)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    out = {"best_stage_a": best_by_task, "stage_a_runs": stage_a, "confirm_runs": confirm}
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
