# PR #18 Code Review Summary: feature/torch_op_refactor → main

## PR Summary

**Title:** feat: torch op refactor – run scope + compiler, no whitelist

**Goal:** Replace the hardcoded DSL-to-torch op mapping with a single compiler module that resolves op names from **run scope** (caller’s globals), **built-ins** (ArgMax, Concat), **torch.nn**, and **torch**, with no whitelist.

### Intended changes (from PR body and plan)

| Area | Intended | Status before CR |
|------|----------|-------------------|
| **`relann/tensor_term_compiler.py`** (new) | `resolve_op(name, globals)` order: DSL-native → run scope → built-ins → torch.nn → torch. `TensorTermCompiler(engine).compile(tterm, var_to_input_index)` with primitives and DSL unary ops (view, sqrt, transpose). | ✅ Present |
| **Engine** | `_run_globals`, `set_run_globals()` / `get_run_globals()`; `tensor_term_to_module` and `eval_tensor_terms_on_tg` delegate to compiler; `evaluate_arith_term_for_hyperparams`. | ❌ Missing wiring |
| **Session** | `define()` calls `engine.set_run_globals(caller_globals)` before parsing. | ❌ Missing |
| **Parser** | `module_class` uses same resolution as compiler (e.g. `resolve_op`) instead of whitelist / `_function_or_nn_module_exists`. | ❌ Still used old API |
| **ArgMax/Concat** | Re-exported from compiler; demos work without explicit import. | ✅ In compiler; Engine still has Concat/ArgMax for backward compat |
| **Tests** | `test_tensor_term_compiler.py`, `test_e2e_cora_gcn.py`, `test_transpose_unary_op.py`. | ✅ Present (tests expect `engine.set_run_globals` / `get_run_globals`) |
| **Docs** | `docs/torch_op_refactor_plan.md`, `docs/transformation_era_operation.md`. | ✅ Plan present |

---

## Code review findings and fixes

### 1. Engine: run_globals and compiler delegation (fixed)

- **Finding:** Engine did not define `_run_globals`, `set_run_globals`, or `get_run_globals`. Tests and compiler assume these exist.
- **Fix:** In `Engine.__init__` added `self._run_globals: Optional[Dict[str, Any]] = None`. Added `set_run_globals(globals_dict)` and `get_run_globals()` returning `{}` when unset. Added `evaluate_arith_term_for_hyperparams(term)` delegating to `_evaluate_arith_term(term)` for the compiler.

### 2. Engine: tensor_term_to_module still contained full legacy implementation (fixed)

- **Finding:** `tensor_term_to_module` still had the long whitelist and if/elif implementation instead of delegating to the compiler.
- **Fix:** Replaced the body with delegation to `TensorTermCompiler(self).compile(tterm, var_to_input_index=var_to_input_index)`. Removed the duplicate/unused `ops_to_modules` from `eval_tensor_terms_on_tg`. Removed unused `_InputSelector` and `_ConstantValue` from engine (they live in the compiler).

### 3. Session: run_globals not set before define (fixed)

- **Finding:** `Session.define()` did not set run-scope globals, so op resolution would not see the caller’s `Linear`, `ReLU`, etc.
- **Fix:** At the start of `define()`, added `self.engine.set_run_globals(inspect.currentframe().f_back.f_globals)` before creating the transformer and parsing.

### 4. Parser: module_class still used old resolution (fixed)

- **Finding:** `module_class` used `engine._function_or_nn_module_exists(class_name)` instead of the same resolution path as the compiler.
- **Fix:** `module_class` now uses `resolve_op(class_name, self.engine.get_run_globals())` from `tensor_term_compiler`. Error message updated to direct users to import ops in the scope that calls `session.run()`.

### 5. Rules and style

- **nbdev:** Code under `parent/` is synced from `nbs/`; any edits in `parent/` should be reflected in the corresponding notebook so `nbdev_prepare` does not overwrite. Engine/Session/Parser edits should be synced to `nbs/021_engine.ipynb`, `nbs/006_session.ipynb`, `nbs/003_parser.ipynb` when exporting.
- **Row-first layout (HGT/RelNN):** No change to embedding layout; compiler and wrappers preserve (num_rows, *feature_dims).
- **No patches for shape bugs:** Fixes are in the refactor wiring, not in engine/op workarounds.

---

## Summary of code changes made (this CR pass)

1. **`relann/engine.py`**
   - Added `_run_globals`, `set_run_globals()`, `get_run_globals()`, `evaluate_arith_term_for_hyperparams()`.
   - Replaced `tensor_term_to_module` implementation with delegation to `TensorTermCompiler(self).compile(...)`.
   - Removed unused `ops_to_modules` from `eval_tensor_terms_on_tg`.
   - Removed unused `_InputSelector` and `_ConstantValue` (compiler owns these).

2. **`relann/session.py`**
   - In `define()`, call `self.engine.set_run_globals(inspect.currentframe().f_back.f_globals)` before parsing.

3. **`relann/parser.py`**
   - In `module_class`, use `resolve_op(class_name, self.engine.get_run_globals())` and align error message with run-scope resolution.

---

## Recommendation

- **Merge:** After syncing the same logic to the corresponding nbdev notebooks (021_engine, 006_session, 003_parser) and re-running the feature tests (e.g. `test_tensor_term_compiler.py`, `test_e2e_cora_gcn.py`, `test_transpose_unary_op.py`, `test_e2e_single_ops.py`), the PR is in good shape to merge.
- **Follow-up:** Consider removing the duplicate `Concat`/`ArgMax` class definitions from the engine and re-exporting them from `tensor_term_compiler` in `engine.py` for a single source of truth (with a deprecation period if needed for `from relann.engine import ArgMax`).
