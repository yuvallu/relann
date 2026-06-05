# Scaffold Bucket

End-to-end "scaffold" tests — they exercise the full RelaNN pipeline (parse → term-graph → compile → train → predict) against PyTorch / PyG reference implementations to verify the framework matches established baselines.

The term "scaffold" comes from `relann/scaffold.py`, the test-infrastructure module that lets you set up a side-by-side comparison between a RelaNN program and a hand-written PyTorch model with identical weights.

## Tests in this bucket

- `test_902_gcn_relnn.py` — 2-layer GCN on Cora. Compares RelaNN vs hand-rolled PyTorch with weight syncing.
- `test_912_scaffold_gcn_cora.py` — GCN via the `scaffold` infrastructure, deeper parity checks.

(`test_913_scaffold_hgt_first_order.py` — deleted in PR #61 as 3-month-old WIP; was never functional.)

Each test trains a real model — expect ~tens-of-seconds per file rather than seconds.

## Running

Both files run cleanly in a single pytest invocation now:

```bash
uv run pytest tests/scaffold/ -v                                # ~22s combined
# or per-file (still works, gives CI cleaner step-by-step timing):
uv run pytest tests/scaffold/test_902_gcn_relnn.py -v           # ~15s
uv run pytest tests/scaffold/test_912_scaffold_gcn_cora.py -v   # ~40s
```

CI keeps the per-file invocation pattern for readability (each step
gets its own row in GitHub Actions output). The combined-run wedge that
used to require this is fixed (see `conftest.py` in this directory),
but split steps still make the timing easier to scan.

### History

Pre-fix, `pytest tests/scaffold/` would hang indefinitely during
collection of `test_912_scaffold_gcn_cora.py` because that file's
inline `if test():` demo calls `matplotlib.pyplot.show()` — under an
interactive backend, `show()` blocks. The fix is a
`conftest.py` in this directory that forces matplotlib's `Agg` backend
before any scaffold test module loads.
