"""CTU Hepatitis binary classification (HBV vs HCV): multi-seed runs.

Uses the RelNN multi-table architecture over 3 relational tables from the
CTU Hepatitis_std dataset (Patients, Biopsies, Labs).

Baseline from ReDeLEx Table 1 (arXiv:2506.22199):
  LightGBM test AUC:          0.626
  GraphSAGE (ResNet) test AUC: 1.000
  DBFormer test AUC:           0.996

Run from repo root:
  python research/paper_experiments/hepatitis/run_hepatitis_multirun.py --runs 5 --epochs 200 --cpu-only
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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from relann.datasets import load_ctu_hepatitis_dataset
from relann.session import Session
from relann.torch_utils import full_seed


# DSL for CTU Hepatitis binary classification (B vs C).
#   d_patient=2  (sex binary, age normalized)
#   d_biopsy=2   (fibros, activity normalized)
#   d_lab=10     (10 normalized lab measurements)
#   3 tables + 3 aggregation rules + 1 scoring rule = 9 meaningful lines
HEPATITIS_DSL = """
#lang:relnn
d_patient = 2 .
d_biopsy  = 2 .
d_lab     = 10 .
hidden    = {hidden} .

PatientEmb(m_id; ReLU()(Linear(d_patient, hidden)(z))) :- Patients(m_id; z) .
BiopsyEmb(m_id; mean(ReLU()(Linear(d_biopsy, hidden)(z)))) :- Biopsies(biopsy_id, m_id; z) .
LabEmb(m_id; mean(ReLU()(Linear(d_lab, hidden)(z)))) :- Labs(lab_id, m_id; z) .
Score(m_id; Linear(hidden * 3, 1)(Concat(z_p, z_b, z_l))) :-
    PatientEmb(m_id; z_p), BiopsyEmb(m_id; z_b), LabEmb(m_id; z_l) .
"""

# Tuned hparams from run_hepatitis_tuning.py (stage-A 30 ep, confirm 100 ep).
# Best: hidden=64, lr=0.02, wd=0.0 (val AUC: 0.931 at stage-A, ~0.905 at 100-ep confirm).
TUNED_HPARAMS = {"hidden": 64, "lr": 0.02, "weight_decay": 0.0}


def _build_dsl(hidden: int) -> str:
    return HEPATITIS_DSL.format(hidden=hidden)


def _train_and_eval_one(
    seed: int,
    hidden: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    cache_dir=None,
) -> dict:
    full_seed(seed)
    data = load_ctu_hepatitis_dataset(cache_dir=cache_dir, seed=42)

    session = Session(db=data.db, device=device)
    dsl = _build_dsl(hidden)
    session.run(dsl)
    session.run(f"""
#lang:relnn
?fit <epochs={epochs}, lr={lr}, weight_decay={weight_decay}>
Loss(; BCEWithLogitsLoss()(z_score, z_label)) :- Score(m_id; z_score), TrainLabels(m_id; z_label) .
""")

    t0 = time.perf_counter()
    pred = session.run("""
#lang:relnn
?pred Predictions(m_id; z) :- Score(m_id; z) .
""")
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    pred_df = pred.content.copy()
    pred_df["score"] = pred.embeddings[0].detach().cpu().squeeze().numpy()
    score_lookup = dict(zip(pred_df["m_id"].astype(int), pred_df["score"].astype(float)))

    # Fallback: predict train mean for unseen m_ids
    train_mean_score = float(data.db["TrainLabels"][1].mean().item())

    def _predict(m_ids):
        return np.array([score_lookup.get(int(mid), train_mean_score) for mid in m_ids])

    val_scores = _predict(data._raw["val_m_ids"])
    test_scores = _predict(data._raw["test_m_ids"])

    val_auc = float(roc_auc_score(data.val_labels, val_scores))
    test_auc = float(roc_auc_score(data.test_labels, test_scores))

    n_params = sum(p.numel() for p in session.engine.parameter_store.values())

    return {
        "seed": seed,
        "hidden": hidden,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "metrics": {"val_roc_auc": val_auc, "test_roc_auc": test_auc},
        "time_s": float(elapsed),
        "params": int(n_params),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=None,
                        help="Hidden dim. Defaults to tuned value.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--uniform-hparams", action="store_true",
                        help="Use --hidden/--lr/--weight-decay for all runs.")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Directory for cached CTU Hepatitis parquet files.")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("research/paper_experiments/hepatitis/results/hepatitis_multirun.json"),
    )
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()
    if args.out_md is None:
        args.out_md = args.out_json.with_name(args.out_json.stem + "_summary.md")

    if args.uniform_hparams:
        h = args.hidden or TUNED_HPARAMS["hidden"]
        lr = args.lr or TUNED_HPARAMS["lr"]
        wd = args.weight_decay if args.weight_decay is not None else TUNED_HPARAMS["weight_decay"]
    else:
        h = TUNED_HPARAMS["hidden"]
        lr = TUNED_HPARAMS["lr"]
        wd = TUNED_HPARAMS["weight_decay"]

    device = torch.device("cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu"))

    results = []
    for i in range(args.runs):
        seed = args.seed_start + i
        print(f"[hepatitis] seed={seed} hidden={h} lr={lr} wd={wd} ...")
        row = _train_and_eval_one(seed, h, args.epochs, lr, wd, device, args.cache_dir)
        results.append(row)
        print(f"  val_auc={row['metrics']['val_roc_auc']:.4f}  "
              f"test_auc={row['metrics']['test_roc_auc']:.4f}  "
              f"time={row['time_s']:.2f}s  params={row['params']}")

    val_aucs = [r["metrics"]["val_roc_auc"] for r in results]
    test_aucs = [r["metrics"]["test_roc_auc"] for r in results]
    val_mean, val_std = float(np.mean(val_aucs)), float(np.std(val_aucs, ddof=1) if len(val_aucs) > 1 else 0.0)
    test_mean, test_std = float(np.mean(test_aucs)), float(np.std(test_aucs, ddof=1) if len(test_aucs) > 1 else 0.0)

    lines = [
        "# CTU Hepatitis multi-run (canonical paper results)",
        "",
        f"- UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Device: {device}",
        f"- epochs={args.epochs}, runs={args.runs}, seed_start={args.seed_start}",
        f"- hidden={h}, lr={lr}, wd={wd}",
        "",
        "## Results",
        "",
        f"- val_roc_auc: {val_mean:.6f} ± {val_std:.6f}",
        f"- test_roc_auc: {test_mean:.6f} ± {test_std:.6f}",
        "",
        "## Paper table values",
        "",
        "| Task | Metric | RelNN | ReDeLEx GNN baseline |",
        "|---|---|---|---|",
        f"| Hepatitis (HBV vs HCV) | AUC ROC ↑ | **{test_mean:.3f} ± {test_std:.3f}** | 1.000 (ResNet SAGE) |",
        "",
        "Baselines from ReDeLEx Table 1 (arXiv:2506.22199):",
        "  LightGBM test AUC: 0.626",
        "  GraphSAGE (Linear/ResNet) test AUC: 0.997 / 1.000",
        "  DBFormer test AUC: 0.996",
    ]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "device": str(device),
            "runs": args.runs,
            "epochs": args.epochs,
            "hidden": h,
            "lr": lr,
            "weight_decay": wd,
        },
        "summary": {
            "val_roc_auc": f"{val_mean:.6f} ± {val_std:.6f}",
            "test_roc_auc": f"{test_mean:.6f} ± {test_std:.6f}",
        },
        "runs": results,
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
