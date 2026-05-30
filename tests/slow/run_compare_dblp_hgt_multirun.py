"""Run 5-seed HGT benchmark and report mean/std.

This script is the canonical HGT timing/accuracy aggregator for paper tables.
It reuses the existing implementations from run_compare_dblp_original_hgt.py:
  - FULL_GRAPH_1L: original pyHGT, PyG HGTConv
  - FULL_GRAPH_2L: original pyHGT, PyG HGTConv
  - PA_PATH_1L: RelNN
  - PA_PATH_2L: RelNN

All timings use synchronized CUDA timers when running on GPU.

Usage:
  python -u ^
      tests/slow/run_compare_dblp_hgt_multirun.py --runs 5
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_compare_dblp_original_hgt import (  # pylint: disable=import-error
    DEVICE,
    _sync_cuda,
    run_original_hgt,
    run_pyg_hgt,
    run_relnn_hgt,
    run_relnn_pyhgt_hgt,
)


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _run_timed(fn, *, seed: int, num_layers: int, epochs: int):
    _sync_cuda()
    t0 = time.perf_counter()
    out = fn(seed=seed, epochs=epochs, num_layers=num_layers)
    if len(out) == 4:
        _, n_params, _, accs = out
    elif len(out) == 3:
        _, n_params, accs = out
    else:
        raise RuntimeError(f"Unexpected return arity from runner: {len(out)}")
    _sync_cuda()
    elapsed = time.perf_counter() - t0
    return int(n_params), accs, float(elapsed)


def _write_outputs(out_json: Path, out_md: Path, payload: dict) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = payload["summary_rows"]
    lines = [
        "# HGT DBLP 5-Run Summary",
        "",
        f"- Timestamp (UTC): {payload['metadata']['timestamp_utc']}",
        f"- Device: {payload['metadata']['device']}",
        f"- Runs: {payload['metadata']['runs']}",
        f"- Seeds: {payload['metadata']['seeds']}",
        f"- Epochs per run: {payload['metadata']['epochs']}",
        "",
        "## Aggregated Results (mean +- std)",
        "",
        "| Scope | Implementation | #Params | Train | Val | Test | Time (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for r in rows:
        lines.append(
            f"| {r['scope']} | {r['implementation']} | {r['params']} | "
            f"{r['train_mean']:.1%} +- {r['train_std']:.1%} | "
            f"{r['val_mean']:.1%} +- {r['val_std']:.1%} | "
            f"{r['test_mean']:.1%} +- {r['test_std']:.1%} | "
            f"{r['time_mean_s']:.1f} +- {r['time_std_s']:.1f} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- All timed blocks are measured with `torch.cuda.synchronize()` before and after timing when CUDA is available.",
        "- Full-graph and PA-path rows are different compute scopes; compare runtimes within scope.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="HGT DBLP multi-run benchmark (mean/std).")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs/seeds (default: 5).")
    parser.add_argument("--seed-start", type=int, default=42, help="Starting seed (default: 42).")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per run (default: 100).")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("research/paper_experiments/hgt/results/hgt_dblp_5run_results.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("research/paper_experiments/hgt/results/hgt_dblp_5run_summary.md"),
        help="Output Markdown summary path",
    )
    args = parser.parse_args()

    seeds = [args.seed_start + i for i in range(args.runs)]
    print(f"[config] device={DEVICE} runs={args.runs} seeds={seeds} epochs={args.epochs}")

    configs = [
        ("FULL_GRAPH_1L", "original_pyHGT", run_original_hgt, 1),
        ("FULL_GRAPH_1L", "pyg_hgtconv", run_pyg_hgt, 1),
        ("FULL_GRAPH_1L", "relnn", run_relnn_hgt, 1),
        ("FULL_GRAPH_2L", "original_pyHGT", run_original_hgt, 2),
        ("FULL_GRAPH_2L", "pyg_hgtconv", run_pyg_hgt, 2),
        ("FULL_GRAPH_2L", "relnn", run_relnn_hgt, 2),
        # pyHGT-faithful RelNN: Dropout + LayerNorm to match acbull/pyHGT architecture
        ("PA_PATH_PYHGT_1L", "relnn_pyhgt", run_relnn_pyhgt_hgt, 1),
    ]

    raw_runs: dict[str, list[dict]] = {}

    for run_idx, seed in enumerate(seeds, start=1):
        print()
        print("=" * 80)
        print(f"[run {run_idx}/{args.runs}] seed={seed}")
        print("=" * 80)
        for scope, impl, fn, num_layers in configs:
            key = f"{scope}::{impl}"
            print(f"[{scope}] {impl} (layers={num_layers})")
            n_params, accs, elapsed = _run_timed(
                fn,
                seed=seed,
                num_layers=num_layers,
                epochs=args.epochs,
            )
            raw_runs.setdefault(key, []).append(
                {
                    "seed": seed,
                    "params": n_params,
                    "train": float(accs["train"]),
                    "val": float(accs["val"]),
                    "test": float(accs["test"]),
                    "time_s": elapsed,
                }
            )
            print(
                f"  params={n_params:,} "
                f"train={accs['train']:.1%} val={accs['val']:.1%} test={accs['test']:.1%} "
                f"time={elapsed:.1f}s"
            )

    summary_rows = []
    for scope, impl, _fn, _layers in configs:
        key = f"{scope}::{impl}"
        entries = raw_runs[key]
        params_set = sorted({e["params"] for e in entries})
        if len(params_set) != 1:
            raise RuntimeError(f"Parameter count changed across runs for {key}: {params_set}")
        train_mean, train_std = _mean_std([e["train"] for e in entries])
        val_mean, val_std = _mean_std([e["val"] for e in entries])
        test_mean, test_std = _mean_std([e["test"] for e in entries])
        time_mean, time_std = _mean_std([e["time_s"] for e in entries])
        summary_rows.append(
            {
                "scope": scope,
                "implementation": impl,
                "params": params_set[0],
                "train_mean": train_mean,
                "train_std": train_std,
                "val_mean": val_mean,
                "val_std": val_std,
                "test_mean": test_mean,
                "test_std": test_std,
                "time_mean_s": time_mean,
                "time_std_s": time_std,
            }
        )

    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device": str(DEVICE),
            "runs": args.runs,
            "seeds": seeds,
            "epochs": args.epochs,
        },
        "raw_runs": raw_runs,
        "summary_rows": summary_rows,
    }
    _write_outputs(args.out_json, args.out_md, payload)

    print()
    print("[done] wrote:")
    print(f"  - {args.out_json}")
    print(f"  - {args.out_md}")


if __name__ == "__main__":
    main()
