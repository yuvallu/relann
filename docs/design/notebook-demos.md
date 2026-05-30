# Inline notebook demos in `relann/*.py` — known issues and workflow

## Background

Each top-level file in `relann/` (`engine.py`, `parser.py`, `relnn.py`, etc.) is a juplit-paired notebook — meaning the `.py` and `.ipynb` are two views of the same content. When you open `relann/engine.ipynb` in Jupyter you see the same code as in `relann/engine.py`, organised as cells.

Many of these source files contain **inline demo cells** wrapped in `if __name__ == "__main__":`. They were originally `#| hide`, `#| eval: false`, or plain non-`#| export` cells under nbdev — meant to demonstrate the code that was just defined in the cell above, runnable interactively in Jupyter but **not** executed during automated testing.

## Current status

After the post-migration audit work, **`relann/engine.py` and `relann/relnn.py` run cleanly as scripts end-to-end** (exit code 0). All inline demos work in linear order — no remaining quarantined demos.

Previously quarantined (all now fixed):
- ~~`engine.py:303` (add_rule demo)~~ — runs cleanly once the Unicode encoding issue in `print(...)` was resolved by `relann/__init__.py` reconfiguring stdout to UTF-8 on Windows. The assumed ordering-bug never actually manifested for this demo's specific input.
- ~~`engine.py:811` (tensor_term_to_module demo)~~ — fixed by adding `var_to_input_index={'z1': 0, 'z2': 1}` to the call. Without it the compiler treated `z1`/`z2` as external symbols and failed; with it the demo runs in linear order.
- ~~`parser.py:~2231` (`Lin = Linear(16, 32)` ctor assertion)~~ — **RESOLVED 2026-05-24.** The original assertion checked `tensor_term.op.hyper_params == [16, 32]`. The parser actually stores the args in `tensor_term.sons` (interpretation #2 below, "intentional API drift"). The assertion was un-quarantined and rewritten to read from `sons`. Contract pinned by `tests/repro/test_parser_ctor_args_in_sons.py`.

## Why some inline demos break under "Run All Cells"

Several source modules — especially `engine.py` — use the `fastcore` `@patch` decorator to add methods onto the `Engine` class throughout the file. A typical layout is:

```
cell N:    @patch  def add_rule(self: Engine, rule): ...     # line ~220
cell N+1:  # DEMO: engine = Engine(); engine.add_rule(...)   # line ~290
…
cell M:    @patch  def _resolve_external_symbol(...): ...    # line ~2640
```

When you "Run All Cells" in Jupyter, cells fire top-to-bottom. By the time the demo at line ~290 runs, `add_rule` is patched, but `_resolve_external_symbol` is **not** — that cell is 2 000 lines below. If `add_rule`'s call path eventually reaches `_resolve_external_symbol`, you get:

```
AttributeError: 'Engine' object has no attribute '_resolve_external_symbol'
```

In practice this happens **only if a Var needs to be resolved as an external symbol** — primitives like `Linear(3, 4)` (ints), and Vars bound via `var_to_input_index`, short-circuit before reaching that code path. The current inline demos all use one of those forms, so they work in linear order.

A demo that uses bare `Var('foo')` without `var_to_input_index` AND without a real symbol-table entry would still hit the ordering bug. If you write a new such demo, either:
- Provide `var_to_input_index`, or
- Move the demo to the end of the file (after `@patch _resolve_external_symbol`), or
- Quarantine with `if False:` + a comment.

## Was this broken before the juplit migration?

Yes, in the same situations. The cells were marked `#| hide` or `#| eval: false` in nbdev, which **only meant they were skipped during `nbdev_test`**. Under interactive Jupyter use the cells still fired, and they hit the same ordering / stale-assertion bugs. The previous workflow tolerated this because:

1. `nbdev_test` skipped them automatically — CI never saw the failures.
2. Interactive users ran cells selectively, skipping ones that didn't work, or fed them the right context manually.
3. There was no "everyone runs `Run All`" expectation.

The juplit migration didn't introduce these bugs — it just made them visible because:

- `nbdev_test` no longer exists; pytest is the runner and doesn't execute inline demos by design.
- `Run All` is a more common workflow in VS Code + the JupyterLab kernel picker.
- We converted `if test():` to `if __name__ == "__main__":` so the cells fire any time the file is executed as a script (e.g., `python relann/engine.py`).

## Resolved: `parser.py:2223` (now ~line 2231)

> **Resolved 2026-05-24.** Interpretation #2 (intentional API drift) confirmed.
> Assertion now reads from `tensor_term.sons` and is un-quarantined.
> Contract pinned by [tests/repro/test_parser_ctor_args_in_sons.py](../../tests/repro/test_parser_ctor_args_in_sons.py).

Original framing (kept for context):

Test parses `Lin = Linear(16, 32) .` and the parser stores values in `sons`, not `op.hyper_params`:

```
tensor_term = TensorTerm(
    op   = TensorOp(op='Linear', hyper_params=None, template_args=None),
    sons = [TensorTerm(value=16), TensorTerm(value=32)]
)
```

The two interpretations at the time:

1. **Parser regression**: per the grammar comment, single-paren `Linear(16, 32)` in a `transform_def` should disambiguate as a ctor with `hyper_params=[16, 32]`. The disambiguation isn't running.
2. **Intentional API drift**: the project standardised on storing values in `sons` for both `Linear(16,32)` and `Linear(16,32)(z)`, and the assertion was never updated. ← **This was the actual answer.** The angle-bracket form (`K_Linear<l, S, i>`) is what populates `template_params` on the `TransformDef`; the paren form populates `sons` on the `tensor_term`. Both predate the migration.

## How to use the demos today

**In Jupyter / VS Code**:

- "Run All" works for `engine.py`, `relnn.py`, and most others.
- For `parser.py` specifically — all demo cells run; the previously-quarantined cell at ~2231 is now un-quarantined (asserts `tensor_term.sons` rather than `op.hyper_params`).
- For more complex demos you write yourself — use `var_to_input_index` when constructing `TensorTerm`s with `Var(...)` so the compiler doesn't fall through to external-symbol resolution.

**From a shell**:

- `uv run python relann/engine.py` and `uv run python relann/parser.py` both exit 0.
- Tests (`uv run pytest`) do NOT execute these demos — they run only the explicit `tests/` files.

**Useful demos to actually run**:

- All cells in `examples/*.py` are runnable end-to-end. Use those (`001_relnn_hello_world.py` etc.) for working code samples.
- Source-module inline demos are documentation, not validated examples.
