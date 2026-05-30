# HGT slow-script status, 2026-05-25 (snapshot)

> Captured during the juplit migration's verification pass. This is a
> point-in-time snapshot of which HGT slow scripts pass/fail on `main`,
> not an evergreen reference. Refresh after any future engine/parser
> work that touches the HGT code paths.

## Summary

8 HGT-related scripts under `tests/slow/run_*hgt*.py` (formerly
`nbs/tests/slow/`). None are part of pytest by design. Status on `main`
(commit `53316a0`) per an agent's run on a machine with `_external/pyHGT`
cloned, conda env unchanged, `CUDA_VISIBLE_DEVICES=1`:

| # | Script                                | Result  | Time | Notes |
|---|---------------------------------------|---------|------|-------|
| 1 | `run_compare_dblp_hgt.py`             | PASS    | 30s  | needs pyHGT clone |
| 2 | `run_compare_dblp_hgt_generic.py`     | PASS    | 21s  | self-contained |
| 3 | `run_compare_dblp_hgt_multirun.py`    | TIMEOUT | 300s | needs ≥10-15 min; likely PASS |
| 4 | `run_compare_dblp_original_hgt.py`    | PASS    | 123s | Table 1 driver |
| 5 | `run_hgt_generic_2l_accuracy.py`      | FAIL    | 75s  | long-standing NaN-loss bug, NOT a regression |
| 6 | `run_hgt_template_cora.py`            | PASS    | 12s  | mean acc 0.7942 across seeds 42..46 |
| 7 | `run_match_hgt_2l_accuracy.py`        | PASS    | 215s | needs pyHGT clone |
| 8 | `run_match_hgt_accuracy.py`           | PASS    | 108s | passes either way |

## Three root causes for the failures

### Cause A — missing local clone of `pyHGT` (4 scripts)

Affected: `run_compare_dblp_hgt`, `run_compare_dblp_hgt_multirun`,
`run_compare_dblp_original_hgt`, `run_match_hgt_2l_accuracy`.

Error: `ModuleNotFoundError: No module named 'pyHGT'`.

The scripts do:
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_external" / "pyHGT"))
from pyHGT.conv import HGTConv as OriginalHGTConv
```

`pyHGT` is not on PyPI and not tracked in git. Setup step is
documented in `CHEATSHEET.md`:

```bash
git clone https://github.com/acbull/pyHGT.git _external/pyHGT
```

Once cloned, all 4 affected scripts pass.

### Cause B — handled by the juplit refactor

`engine.py:tensor_term_to_module` on `main` was a big hand-rolled function
with a hardcoded list of ops that explicitly raised `NotImplementedError`
for `Tensor`. The juplit branch's `relann/engine.py:695-696` already
delegates to `TensorTermCompiler`, which handles `Tensor` (and every other
torch.nn op) via name resolution. So **the bug that broke
`nbs/tests/_913_scaffold_hgt_first_order.ipynb` on main does not exist on
juplit** — though the scaffold itself was moved to `_draft_*` for other
reasons (see `tests/scaffold/_draft_913_scaffold_hgt_first_order.py`).

### Cause C — NaN loss in `run_hgt_generic_2l_accuracy.py` (1 script)

Error: `RuntimeError: Non-finite loss encountered at epoch 0: nan`.

Pre-existing on both `main` and `claude/v2-next-version` (verified by the
agent). NOT a juplit regression. The 1-layer counterpart
(`run_compare_dblp_hgt_generic.py`) passes; the 2-layer variant adds a
second `H<'Author', 2>` output and somewhere in the extra layer the forward
pass produces NaN — most likely in the attention softmax (overflow / div
by zero) or an uninitialized parameter.

**This is not on the paper's critical path.** Table 1's HGT 2L row uses
`run_hgt_table1.py` which calls `run_compare_dblp_original_hgt.py`'s
PA-only baselines — completely different code path. Document as known
issue; fix in a separate focused session when convenient.

## Why CI never caught any of this

Three structural reasons, all pre-existing:

1. `_external/pyHGT` is not tracked and not in any CI setup script.
   Failures fall into a `ModuleNotFoundError` rather than a clean skip.
2. The slow `run_*.py` scripts are NOT pytest tests (no `def test_*`).
   pytest's default discovery ignores them entirely.
3. The pre-juplit notebook 913 was `.ipynb`, not exercised by pytest
   unless `nbval` is configured (it isn't).

The juplit migration didn't change any of this. A future PR could:
- Promote slow scripts to `@pytest.mark.slow`-tagged pytest tests with
  `pytest.importorskip("pyHGT")` at the top.
- Add a CI job that runs `poe slow-hgt` (added in this branch) on a
  longer timeout.

## Source / verification

- Agent run captured on 2026-05-25 on a Linux box with CUDA + conda
  `parent` env.
- Reference logs were in `/tmp/hgt_test_runs/*.log` on the agent's machine
  (not preserved).
- This document is the authoritative summary of that run; if you need
  to reproduce, the `poe slow-hgt` task added in this branch is the
  one-command way (after cloning `_external/pyHGT`).
