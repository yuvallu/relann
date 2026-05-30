# Torch Op Refactor Plan: TensorTerm → nn.Module (No Whitelist, No Patches)

This plan describes how to replace the current hardcoded DSL-to-torch mapping with a **shorter, maintainable implementation** that resolves op names from **torch** and the **Python run scope**, **without** a whitelist and **without** using `@patch`.

---

## Goals

1. **No whitelist**: Resolve `nn.Module` classes and callables from `torch.nn`, `torch`, and the **caller’s globals** (run scope) instead of a fixed `ops_to_modules` dict.
2. **Shorter, readable, elegant**: One small compiler module; generic “resolve → instantiate → wrap” path instead of long if/elif per op.
3. **No patches**: New code uses normal classes and methods. Existing `Engine` methods that are refactored should become regular instance methods (no `@patch` from fastcore).
4. **Testability**: Unit tests per op (and per resolution path), plus 1–2 e2e tests that must keep working (e.g. Cora GCN demo).

---

## Current State (Summary)

- **engine.py**: `tensor_term_to_module` (and helpers like `_eval_arith_term`, `_InputSelector`, `_ConstantValue`, `Concat`, `ArgMax`) plus `eval_tensor_terms_on_tg` use a hardcoded `ops_to_modules` and a long if/elif chain for Linear, Concat, ReLU, MSELoss, CrossEntropyLoss, ArgMax.
- **Parser**: `module_class` returns the string name and uses `engine._function_or_nn_module_exists(name)` to validate (builtins, frame globals, torch.nn, user nn.Modules). So “run scope” is intended to be the caller’s globals, but `getframe(1)` may not be the user frame when parsing.
- **Session**: `define(code)` creates `RelnnTransformer(self.engine)` and calls `parse_and_transform_str(...)` then `self.engine.add_program(program)`. No explicit passing of run-scope globals to the engine.

---

## Architecture After Refactor

### 1. Run-scope globals

- **Session.define(code)** (or **Session.run(code)**): Before calling `engine.add_program(program)`, capture the **caller’s globals** (e.g. `inspect.currentframe().f_back.f_globals` or an explicit `run_globals` argument) and pass them to the engine.
- **Engine**: Holds an optional **run_globals** (e.g. `self._run_globals: Optional[dict] = None`). Provide:
  - `set_run_globals(globals: dict) -> None`
  - `get_run_globals() -> dict` (return `{}` or a safe default if not set)
- **Parser**: When validating `module_class`, use the same resolution logic as the compiler (see below) so that “available in run scope or torch” is defined in one place. Option A: Parser calls `engine.resolve_op(name)` which uses `engine.get_run_globals()` + torch; Option B: Parser receives run_globals from Session and passes them to the transformer. Prefer **Option A** so that resolution is centralized in the engine/compiler.

### 2. New module: `parent/tensor_term_compiler.py` (or `parent/torch_op_compiler.py`)

A single module that owns “TensorTerm → nn.Module” and **does not use @patch**.

**Classes / functions:**

- **resolve_op(name: str, globals: dict) -> Optional[type | callable]**  
  Resolve a string name to a class or callable. Lookup order:
  1. **globals** (run scope): if `name in globals` and the value is a type (e.g. `nn.Module` subclass) or a callable (e.g. function), return it.
  2. **torch.nn**: `getattr(nn, name, None)` if it’s an `nn.Module` class or callable.
  3. **torch**: `getattr(torch, name, None)` for functions (e.g. `torch.relu`).
  4. Optional: **builtins** for things like `True`/`False` if they appear as op names (unlikely for tensor ops).

  Return `None` if not found. Normalize name (e.g. allow `"Linear"` and `"linear"` by trying both or only the given casing depending on policy).

- **TensorTermCompiler**  
  - `__init__(self, engine: Engine)`  
    Stores a reference to the engine (for symbol table, `replace_all_vars`, and run_globals).
  - `compile(self, tterm: TensorTerm, var_to_input_index: Optional[Dict[str, int]] = None) -> nn.Module`  
    Converts one TensorTerm tree to one `nn.Module`. This is the main entry point (replaces `engine.tensor_term_to_module`).

**Internal design of `compile`:**

- **Leaves** (op is None): Same as today: Var → `_InputSelector(index)`; constant → `_ConstantValue(value)`. Keep `_InputSelector` and `_ConstantValue` in this module (or a small `_primitives` submodule).
- **Arithmetic / binary ops** (`*`, `+`, `-`, `/`, `@`, `**`, `==`): Same idea as today: build two child modules, wrap in a small nn.Module that calls `torch.mul`, `torch.add`, etc. No whitelist change here.
- **Unary ops** (`sqrt`, `transpose`, `view`): Either:
  - Resolve by name: `resolve_op("sqrt", globals)` → `torch.sqrt`; then a single generic “unary call” wrapper; or
  - Keep a tiny internal map for these if they are not in `torch` with the same name (e.g. `transpose` might be `lambda x: x.transpose(-1, -2)`). Prefer **resolve from torch** where possible; otherwise a minimal map only for DSL-specific behavior (e.g. “transpose means swap last two dims”).
- **Named op (e.g. Linear, ReLU, user MyModule)**:
  1. If the name is a **TransformDef** in the engine’s symbol table: recursively compile that TransformDef’s tensor_term (with same `var_to_input_index`), wrap in a **_TransformDefWrapper** (one child module that feeds the inner module). Same as today.
  2. Else: **resolve_op(name, engine.get_run_globals())** → get a class or callable.
  3. **Instantiate**:
     - If it’s already an `nn.Module` instance (e.g. passed in globals), use it (and optionally load saved params).
     - Else if it’s a class or callable: evaluate **hyper_params** to Python values (using existing ArithTerm evaluation logic, which needs symbol table for Vars). Then use **inspect.signature** to get constructor parameters and call with positional or keyword args (same idea as RelNN’s `_instantiate_callable_from_hyper`). No op-specific branching: one code path for “call ctor(*eval(hp))”.
  4. **Wrap for children**: The resulting module expects one or more tensor inputs. If the TensorTerm has **sons**:
     - Build child modules recursively.
     - **Single child**: forward = `module(child(*inputs))`.
     - **Multiple children**: Either `module(torch.cat([c(*inputs) for c in children], dim=1))` for “one tensor in” modules (e.g. Linear), or `module(*[c(*inputs) for c in children])` for “multiple args” (e.g. CrossEntropyLoss(pred, target)). We can infer from **inspect.signature(module.forward)** (or `module.__call__`): if the first parameter after `self` is `*args` or there are 2+ input params, pass one tensor per child; else concat children and pass one tensor. Document this convention clearly.

**Expected outcome per op type:**

- **nn.Linear(in_f, out_f, bias=...)**  
  One tensor in, one tensor out; shape (N, in_f) → (N, out_f). Children (if any) are concatenated on dim=1 then passed to Linear.
- **nn.ReLU()**  
  One tensor in, one tensor out; same shape. Same child/concat rule.
- **Concat** (or `torch.cat` with dim=1)  
  Multiple tensors in, one tensor out. One child per input; forward = `torch.cat([c(*inputs) for c in children], dim=1)`.
- **MSELoss**, **CrossEntropyLoss**  
  Two tensor inputs (predictions, targets). Two children; forward = `loss(child0(*inputs), child1(*inputs))`. CrossEntropy: convert one-hot to indices and ensure dtypes as today.
- **ArgMax**  
  One tensor in, one tensor out (class indices). Same as today.
- **view**, **sqrt**, **transpose**  
  One tensor in; behavior as today (view with hyper_params for shape; transpose = swap last two dims).

All of the above can be implemented with a **generic** “resolve → instantiate → wrap” path plus a **small** set of conventions (or one optional “adapter” dict for ops that need special input wiring, e.g. CrossEntropy’s one-hot→index). The plan is to **minimize** op-specific branches.

### 3. Engine changes (no new patches)

- **Remove** the current `tensor_term_to_module` implementation from `engine.py` (or leave a thin wrapper for backward compatibility that delegates to the compiler).
- **eval_tensor_terms_on_tg**: For each transformation node, call `compiler = TensorTermCompiler(self)` (or use a cached compiler), then `torch_module = compiler.compile(tterm, var_to_input_index)`, and set `tg.nodes[node]['torch_transformation'] = torch_module`. No duplicate `ops_to_modules` in `eval_tensor_terms_on_tg`.
- **Run globals**: Before `add_program`, the Session (or the entry point that calls `add_program`) must call `engine.set_run_globals(caller_globals)`. So in **Session.define(code)** we do something like:
  - `caller_globals = inspect.currentframe().f_back.f_globals`
  - `self.engine.set_run_globals(caller_globals)`
  - then `parse_and_transform_str(...)` and `self.engine.add_program(program)`.
- **Parser module_class**: When validating the name, call something that uses the same resolution (e.g. `engine.resolve_op(name)` or a function in `tensor_term_compiler` that takes `(name, engine.get_run_globals())`). So “valid at parse time” = “resolvable at compile time”.
- **No @patch in new code**: `TensorTermCompiler` is a normal class. If we refactor existing `Engine` methods (e.g. `eval_tensor_terms_on_tg`, `tensor_term_to_module`), we change them to normal instance methods and remove the `@patch` decorator for those methods. Prefer doing this only for the methods that are part of this refactor (tensor-term compilation and eval_tensor_terms_on_tg) to limit scope.

### 4. Where to put helpers

- **_InputSelector**, **_ConstantValue**, **Concat**, **ArgMax**: Today they live in `engine.py`. Move them to **tensor_term_compiler.py** (or a `parent/compiler_primitives.py`) so the compiler is self-contained. If other parts of the codebase use `Concat` or `ArgMax`, export them from the compiler module (or keep a re-export in engine for backward compatibility).
- **_eval_arith_term**: Logic that evaluates ArithTerms (including Var resolution from symbol table) should live in the **Engine** (it uses symbol table) or be a pure function that takes `(arith_term, get_symbol)`. The compiler can call `engine._evaluate_arith_term(term)` or a small helper that the engine provides. Prefer keeping symbol resolution inside Engine and having the compiler call into it.

---

## Tests

### 5.1 Unit tests: resolution

- **test_resolve_from_run_scope**: Set `run_globals = {"MyLinear": nn.Linear}`, resolve `"MyLinear"` → get `nn.Linear`.
- **test_resolve_from_torch_nn**: With empty (or no) run_globals, resolve `"Linear"`, `"ReLU"` → get `nn.Linear`, `nn.ReLU`.
- **test_resolve_from_torch**: Resolve `"relu"` (if we support torch.relu) or a function from torch.
- **test_resolve_not_found**: Resolve `"NonExistentOp"` → None or clear error.

### 5.2 Unit tests: compile (per op)

Each test builds a TensorTerm, calls `compiler.compile(tterm, var_to_input_index)`, runs the module on dummy tensors, and checks shape and (where useful) value.

- **Linear**: TensorTerm `Linear(4, 8)(z)` with `z` → input 0. Forward with (N, 4) → output (N, 8).
- **ReLU**: TensorTerm `ReLU(z)`. Forward with (N, D) → same shape, non-negative.
- **Concat**: TensorTerm `Concat(z1, z2)` with z1, z2 → 0, 1. Forward with (N, 2), (N, 3) → (N, 5).
- **view**: TensorTerm `view(2, 3)(z)`. Forward with (N, 6) → (N, 2, 3).
- **sqrt**: TensorTerm `sqrt(z)`. Forward with (N, D) positive → correct values.
- **transpose**: TensorTerm `transpose(z)`. Forward with (E, d) → (E, d, 1) or last-two-dims swapped per current spec.
- **Arithmetic**: TensorTerm `z1 * z2`, `z1 + z2`, etc. Forward with two tensors → correct shape and (optionally) value.
- **MSELoss**: TensorTerm `MSELoss()(pred, target)`. Two children. Forward → scalar loss.
- **CrossEntropyLoss**: TensorTerm `CrossEntropyLoss()(logits, targets)`. One-hot or indices; output scalar.
- **ArgMax**: TensorTerm `ArgMax()(z)`. Forward (N, C) → (N) long.
- **TransformDef**: Register a TransformDef whose tensor_term is e.g. `Linear(2, 4)(z)`. TensorTerm that references that TransformDef by name, with one son. Compile → module that applies Linear(2, 4) to the son’s output.

Reuse the style of existing tests (e.g. `nbs/tests/_feature_tests/test_transpose_unary_op.py`): construct TensorTerm and TensorOp by hand, call the compiler, run forward, assert shapes and optionally values.

### 5.3 E2E tests

- **E2E Cora GCN (001 demo)**: Run the same program as in `nbs/demos/001_relnn_hello_world.ipynb` (define program with Linear/ReLU, fit, predict). Assert:
  - No errors during define/fit/predict.
  - Training loss decreases (e.g. first epoch loss > last epoch loss).
  - Test accuracy is in a reasonable range (e.g. > 0.75) so that the model is actually learning.
  This can be a pytest in `nbs/tests/` or `tests/` that loads the dataset, builds the session, runs the same strings as the notebook, and checks loss and accuracy.
- **E2E parameters sharing (optional)**: If we have a test that uses a shared TransformDef (e.g. `test_parameters_sharing.py` or a notebook), run it and assert that parameters are shared (e.g. same param object for two uses of the same TransformDef) and that training still runs.

### 5.4 Where to put tests

- **Unit tests**: New file(s) under `nbs/tests/_feature_tests/` or a new `tests/` at repo root, e.g.:
  - `test_tensor_term_compiler.py` (resolution + compile for each op)
  - Or split: `test_torch_op_resolution.py`, `test_tensor_term_compile_ops.py`
- **E2E**: e.g. `test_e2e_cora_gcn.py` that mirrors the 001 notebook; optionally `test_e2e_parameters_sharing.py`.

Run all new tests in CI; ensure existing tests (e.g. `test_transpose_unary_op.py`, any other feature tests) still pass after the refactor.

---

## Implementation Order

1. **Run globals**: Add `_run_globals` to Engine, `set_run_globals` / `get_run_globals`. In Session.define, set run_globals from caller frame before add_program.
2. **New module**: Create `tensor_term_compiler.py` with `resolve_op`, `TensorTermCompiler`, and move/copy `_InputSelector`, `_ConstantValue`, Concat, ArgMax (and any other shared helpers). Implement `compile` for leaves, arithmetic, unary (sqrt, transpose, view), and one “named op” path: resolve_op → instantiate from hyper_params (using engine for ArithTerm eval) → generic wrapper (one child = one input; multiple children = concat or multi-arg by signature). Handle TransformDef in symbol table.
3. **Wire Engine**: In `eval_tensor_terms_on_tg`, use `TensorTermCompiler(self).compile(...)` instead of `self.tensor_term_to_module(...)`. Remove (or thin out) the old `tensor_term_to_module` and the `ops_to_modules` + long if/elif from engine.py. Update parser’s `module_class` to use the same resolution (e.g. `resolve_op`) so parse-time validation matches compile-time.
4. **Adapter for special ops**: If CrossEntropyLoss (one-hot→index, dtype) or similar need special handling, add a **small** adapter layer (e.g. a dict of op name → optional input adapter function) rather than re-introducing a big if/elif. Prefer keeping the generic path and only adapting where strictly necessary.
5. **Tests**: Add unit tests for resolution and for each op; add e2e Cora GCN test. Run full test suite and fix regressions.
6. **Cleanup**: Remove duplicate code from engine (old if/elif, unused helpers if moved). Ensure docs (e.g. `docs/transformation_era_operation.md`) are updated to describe “resolve from run scope and torch” instead of the whitelist.

---

## Success Criteria

- No hardcoded whitelist of op names in the compiler; resolution is via run scope + torch.nn + torch.
- TensorTerm → nn.Module logic lives in one module (`tensor_term_compiler.py`), is shorter and more readable than the current long function.
- No new uses of `@patch` in this feature; refactored Engine methods are normal methods.
- All new unit tests pass; e2e Cora GCN (001 demo) passes; existing feature tests (e.g. transpose, parameters sharing) still pass.
- Documentation and this plan are the single source of truth for “how we map a string to torch” and “expected outcome per op”.
