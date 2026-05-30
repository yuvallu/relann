"""RelBench rel-f1: multi-seed runs for driver-position and driver-dnf.

Uses the same RelNN architecture as ``tests/feature/test_relbench_f1_smoke.py``;
evaluates with RelBench ``task.evaluate`` on test predictions.

Requires: ``pip install relbench``

Run from repo root (per-task tuned hyperparameters by default; CPU-safe):
  python research/paper_experiments/relbench/run_relbench_f1_multirun.py --runs 5 --epochs 200 --cpu-only

Use one global (hidden, lr, wd) for every task:
  python research/paper_experiments/relbench/run_relbench_f1_multirun.py --uniform-hparams --hidden 64 --lr 0.01 --weight-decay 5e-4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from relann.datasets import load_relbench_f1_dataset
from relann.session import Session
from relann.torch_utils import full_seed

# From ``relbench_f1_tuning.json`` stage-A best + 200-epoch confirm (no extra tables).
# driver-top3 hparams are initial defaults (same as driver-dnf) pending tuning.
TUNED_TASK_HPARAMS = {
    "driver-position": {"hidden": 32, "lr": 0.01, "weight_decay": 1e-4},
    "driver-dnf": {"hidden": 16, "lr": 0.003, "weight_decay": 0.0},
    "driver-top3": {"hidden": 64, "lr": 0.005, "weight_decay": 5e-4},
}

# From ``relbench_f1_tuning_extra.json`` stage-A best (qualifying + standings features).
TUNED_TASK_HPARAMS_EXTRA = {
    "driver-position": {"hidden": 64, "lr": 0.02, "weight_decay": 5e-4},
    "driver-dnf": {"hidden": 32, "lr": 0.003, "weight_decay": 0.0},
}


def _build_dsl(d_driver: int, d_cons: int, d_race: int, d_result: int, hidden: int) -> str:
    return f"""
#lang:relnn
d_driver = {d_driver} .
d_cons   = {d_cons} .
d_race   = {d_race} .
d_result = {d_result} .
hidden   = {hidden} .

DriverEmb(driverId; ReLU()(Linear(d_driver, hidden)(z))) :- Drivers(driverId; z) .
ConsEmb(constructorId; ReLU()(Linear(d_cons, hidden)(z))) :- Constructors(constructorId; z) .
RaceEmb(raceId; ReLU()(Linear(d_race, hidden)(z))) :- Races(raceId; z) .
DriverHistory(driverId; mean(ReLU()(Linear(d_result + hidden * 2, hidden)(Concat(z_r, z_c, z_race))))) :- Results(resultId, driverId, raceId, constructorId; z_r), ConsEmb(constructorId; z_c), RaceEmb(raceId; z_race) .
Score(driverId; Linear(hidden * 2, 1)(Concat(z_d, z_h))) :- DriverEmb(driverId; z_d), DriverHistory(driverId; z_h) .
"""


def _train_and_eval_one(
    task_name: str,
    seed: int,
    hidden: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    extra_tables: bool,
) -> dict:
    full_seed(seed)
    data = load_relbench_f1_dataset(task_name=task_name, extra_tables=extra_tables)
    info = data.dataset_info
    d_driver = info["driver_feature_dim"]
    d_cons = info["constructor_feature_dim"]
    d_race = info["race_feature_dim"]
    d_result = info["result_feature_dim"]

    session = Session(db=data.db, device=device)
    dsl = _build_dsl(d_driver, d_cons, d_race, d_result, hidden)
    session.run(dsl)
    if task_name in ("driver-dnf", "driver-top3"):
        loss_rule = "Loss(; BCEWithLogitsLoss()(z_score, z_label)) :- Score(driverId; z_score), TrainLabels(driverId; z_label) ."
    else:
        loss_rule = "Loss(; MSELoss()(z_score, z_label)) :- Score(driverId; z_score), TrainLabels(driverId; z_label) ."
    session.run(
        f"""
#lang:relnn
?fit <epochs={epochs}, lr={lr}, weight_decay={weight_decay}>
{loss_rule}
"""
    )
    t0 = time.perf_counter()
    pred = session.run(
        """
#lang:relnn
?pred Predictions(driverId; z) :- Score(driverId; z) .
"""
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    pred_df = pred.content.copy()
    pred_df["pred"] = pred.embeddings[0].detach().cpu().squeeze().numpy()
    pred_lookup = dict(zip(pred_df["driverId"], pred_df["pred"]))

    test_df = data.test_table.df
    if task_name == "driver-dnf":
        train_mean = float(data.train_table.df["did_not_finish"].mean())
    elif task_name == "driver-top3":
        train_mean = float(data.train_table.df["qualifying"].mean())
    else:
        train_mean = float(data.train_table.df["position"].mean())
    test_ids = test_df["driverId"].values
    test_pred = np.array([pred_lookup.get(int(did), train_mean) for did in test_ids])

    metrics = data.task.evaluate(test_pred, data.test_table)
    n_params = sum(p.numel() for p in session.engine.parameter_store.values())

    return {
        "task_name": task_name,
        "seed": seed,
        "hidden": hidden,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "extra_relbench_tables": extra_tables,
        "metrics": {k: float(v) if isinstance(v, (float, int, np.floating)) else v for k, v in metrics.items()},
        "time_s": float(elapsed),
        "params": int(n_params),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["driver-position", "driver-dnf", "driver-top3"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--uniform-hparams",
        action="store_true",
        help="Use the same --hidden/--lr/--weight-decay for every task (default: tuned per task).",
    )
    parser.add_argument(
        "--extra-tables",
        action="store_true",
        help="Append qualifying + standings features to each Results row (see parent.datasets.load_relbench_f1_dataset).",
    )
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("research/paper_experiments/relbench/results/relbench_f1_multirun.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Markdown summary path. Defaults to --out-json with .json replaced by _summary.md.",
    )
    args = parser.parse_args()
    if args.out_md is None:
        args.out_md = args.out_json.with_name(args.out_json.stem + "_summary.md")

    pytest = __import__("pytest")
    pytest.importorskip("relbench")

    device = torch.device("cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu"))
    results = []
    for task_name in args.tasks:
        if args.uniform_hparams:
            h, lr, wd = args.hidden, args.lr, args.weight_decay
        else:
            table = TUNED_TASK_HPARAMS_EXTRA if args.extra_tables else TUNED_TASK_HPARAMS
            if task_name not in table:
                raise ValueError(
                    f"No tuned defaults for task {task_name!r}; pass --uniform-hparams or extend the appropriate TUNED_TASK_HPARAMS dict."
                )
            cfg = table[task_name]
            h, lr, wd = cfg["hidden"], cfg["lr"], cfg["weight_decay"]
        for i in range(args.runs):
            seed = args.seed_start + i
            print(f"[{task_name}] seed={seed} hidden={h} lr={lr} wd={wd} extra_tables={args.extra_tables} ...")
            row = _train_and_eval_one(
                task_name, seed, h, args.epochs, lr, wd, device, extra_tables=args.extra_tables
            )
            results.append(row)
            print(f"  metrics={row['metrics']} time={row['time_s']:.2f}s params={row['params']}")

    def _mean_std(keys, task):
        rows = [r for r in results if r["task_name"] == task]
        out = {}
        for k in keys:
            xs = [r["metrics"][k] for r in rows if k in r["metrics"]]
            if not xs:
                continue
            a = np.array(xs, dtype=np.float64)
            out[k] = (float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0)
        return out

    lines = [
        "# RelBench rel-f1 multi-run",
        "",
        f"- UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Device: {device}",
        f"- uniform_hparams={args.uniform_hparams}, extra_tables={args.extra_tables}",
        f"- epochs={args.epochs}",
        "",
    ]
    # Select correct hparam table for provenance logging (mirrors selection in main loop)
    _hparam_table = TUNED_TASK_HPARAMS_EXTRA if args.extra_tables else TUNED_TASK_HPARAMS
    if args.uniform_hparams:
        lines.append(f"- (all tasks) hidden={args.hidden}, lr={args.lr}, wd={args.weight_decay}")
    else:
        for t in args.tasks:
            c = _hparam_table[t]
            lines.append(f"- {t}: hidden={c['hidden']}, lr={c['lr']}, wd={c['weight_decay']}")
    lines.append("")

    for task in args.tasks:
        task_rows = [r for r in results if r["task_name"] == task]
        keys = list(task_rows[0]["metrics"].keys()) if task_rows else []
        ms = _mean_std(keys, task) if task_rows else {}
        lines.append(f"## {task}")
        for k, (m, s) in ms.items():
            lines.append(f"- {k}: {m:.6f} ± {s:.6f}")
        lines.append("")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "device": str(device),
        "runs": args.runs,
        "epochs": args.epochs,
        "uniform_hparams": args.uniform_hparams,
        "extra_relbench_tables": args.extra_tables,
    }
    if args.uniform_hparams:
        meta.update(
            {"hidden": args.hidden, "lr": args.lr, "weight_decay": args.weight_decay}
        )
    else:
        meta["tuned_per_task"] = {t: _hparam_table[t] for t in args.tasks}
    payload = {
        "metadata": meta,
        "runs": results,
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
