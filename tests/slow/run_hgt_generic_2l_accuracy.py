"""Verify generic-library 2L HGT accuracy over 5 random seeds.

Uses RELNN_GENERIC_DEFINE_2L from run_compare_dblp_hgt_generic.py — a single
H<'Author', 2> output line (vs H<'Author', 1> in 1L) driving the same 33-line
library.  Demonstrates that 2-layer HGT costs one extra line in the high-order
DSL.

Run from repo root:
    python tests/slow/run_hgt_generic_2l_accuracy.py
"""
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.nn.functional as F

from relann.torch_utils import full_seed
from relann.session import Session

from run_compare_dblp_hgt_generic import (
    RELNN_GENERIC_DEFINE_2L,
    RELNN_FIT_DSL,
    RELNN_PRED_DSL,
    relnn_db,
    dblp,
    info,
    hidden,
    num_heads,
    n_classes,
    evaluate_dblp_relnn,
)

DIVIDER = "=" * 70
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using {DEVICE}")

def run_seed(seed: int, epochs: int = 100) -> dict:
    print()
    print(DIVIDER)
    print(f"Seed {seed}")
    print(DIVIDER)

    full_seed(seed)
    # Generic library uses CPU (same as existing run_compare_dblp_hgt_generic.py)
    session = Session(db=relnn_db)
    session.run(RELNN_GENERIC_DEFINE_2L)

    # Materialise parameters via a forward pass before training
    session.run("""
#lang:relnn
?pred _Init(id; Classifier(z)) :- Output(id; z) .
""")

    n_params = sum(p.numel() for p in session.engine.parameter_store.values())
    print(f"  Parameters: {n_params:,}")

    full_seed(seed)
    t0 = time.perf_counter()
    session.run(RELNN_FIT_DSL.format(epochs=epochs, lr=0.005, wd=0.001))
    elapsed = time.perf_counter() - t0

    pred = session.run(RELNN_PRED_DSL)
    accs = evaluate_dblp_relnn(pred, dblp.node_metadata)

    print(f"  Train: {accs.get('train', '?'):.1%}  Val: {accs.get('val', '?'):.1%}  "
          f"Test: {accs.get('test', '?'):.1%}  Time: {elapsed:.1f}s")

    return {"seed": seed, "n_params": n_params, "time": elapsed, **accs}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    results = []
    for i in range(args.runs):
        seed = args.seed_start + i
        r = run_seed(seed, epochs=args.epochs)
        results.append(r)

    test_accs = [r["test"] for r in results]
    train_accs = [r["train"] for r in results]
    times = [r["time"] for r in results]

    print()
    print(DIVIDER)
    print("SUMMARY — Generic library 2L HGT (random init)")
    print(DIVIDER)
    print(f"  Test:  {np.mean(test_accs):.1%} ± {np.std(test_accs):.1%}")
    print(f"  Train: {np.mean(train_accs):.1%} ± {np.std(train_accs):.1%}")
    print(f"  Time:  {np.mean(times):.1f} ± {np.std(times):.1f}s")
    print()
    per_seed = "  " + "  ".join(f"[{r['seed']}] {r['test']:.1%}" for r in results)
    print(per_seed)

    out_path = (
        Path(__file__).resolve().parents[2]
        / "research" / "paper_experiments"
        / "hgt"
        / "results"
        / "hgt_generic_2l_results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results,
                   "summary": {"test_mean": float(np.mean(test_accs)),
                               "test_std":  float(np.std(test_accs)),
                               "time_mean": float(np.mean(times)),
                               "time_std":  float(np.std(times))}}, f, indent=2)
    print(f"\n  Results saved to {out_path}")
