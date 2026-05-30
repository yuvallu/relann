# DHN Bucket

Tests for **Deep Homomorphism Networks** in RelaNN — the model class from Maehara & Saito (NeurIPS 2024) that uses subgraph-homomorphism counts as the inductive bias for graph-level prediction.

## Layout

- `test_dhn_basic.py` — unit tests for homomorphism enumeration, DHN RelaNN program generation, forward pass on toy data, graph-level aggregation.
- `test_dhn_edge_hom_parity.py` — verifies that homomorphism precomputation matches the edge-join tuple sets for cycle motifs C2–C4.
- `test_dhn_ghl_csl_c2_4.py` — smoke test for the canonical GHL RelaNN program on the CSL dataset with edge-join semantics.
- `dhn_utils.py` — shared helpers: dataset loading, homomorphism enumeration for cycles/cliques, Paulus SR(25) graph validation.
- `dhn_external.py` — hooks for running the official DHN reference implementation alongside RelaNN. Requires `_external/dhn/` checkout (see `scripts/setup_external_dhn.py`).
- `*.relnn` — DSL programs for the various DHN configurations (`dhn_pure_C2_C4`, `dhn_full_C2_8`, etc.). These are the actual RelaNN code under test.
- `run_pure_benchmarks.py`, `run_official_dhn*.py` — scenario runners (not pytest-style tests) used to reproduce paper benchmarks.
- `data/` — CSL, ENZYMES, PROTEINS datasets (gitignored).

## Running

```bash
uv run pytest tests/dhn -v            # ~25-60s depending on configs
uv run python scripts/run_tests.py dhn
```

The `run_official_dhn*.py` files need `_external/dhn/` cloned at the right commit — run `uv run python scripts/setup_external_dhn.py` first.
