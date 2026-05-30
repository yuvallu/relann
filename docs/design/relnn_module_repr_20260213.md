# RelNN Module Repr: Torch-Style Visualization

This document describes the design, implementation, and usage of the readable `nn.Module` representation for RelNN and the transformation submodules produced by the tensor-term compiler. The goal is to make `print(module)` show a clear, torch-idiomatic tree instead of opaque `ModuleList((0): _InputSelector(), ...)`.

---

## 1. Motivation and problem

After running `session.run(define)` and then building the predict module (e.g. via `build_and_run_era_demo_module(session, "Out")`), the RelNN module is a nested `nn.Module` with:

- An `ops` `ModuleDict` (DataLoader, Transformation, Aggregation nodes).
- Each Transformation wraps a compiled tensor-term tree (e.g. `_UnaryOp` → `_InputSelector`, or `_SingleChildWrapper` around Linear/ReLU).

**Before** the changes, printing the module looked like:

```
RelNN(
  (ops): ModuleDict(
    (Input): DataLoader()
    (transformation_Test1): Transformation(
      (transformation): _UnaryOp(
        (children_modules): ModuleList(
          (0): _InputSelector()
        )
      )
    )
    ...
  )
)
```

Issues:

- `_InputSelector()` and `_UnaryOp()` gave no hint which input (e.g. `z1`) or which op (e.g. `transpose`) they represented.
- `(children_modules): ModuleList((0): ...)` was verbose and generic.
- The top-level `RelNN()` had no one-line summary of the pipeline.

---

## 2. PyTorch conventions used

PyTorch’s `nn.Module` representation is built from:

1. **`extra_repr()`**  
   Override this to add a one-line description after the class name. It is the standard way to show custom attributes (e.g. `Linear(in_features=2, out_features=2, bias=True)`). See [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.extra_repr).

2. **Named child modules**  
   Submodules are shown as `(name): ModuleType(...)`. The **name** is the attribute name (e.g. `self.conv1` → `(conv1): Conv2d(...)`). Using a single named attribute (e.g. `self.arg`) instead of `ModuleList([child])` yields one clear line like `(arg): _InputSelector("z1")` instead of `(children_modules): ModuleList((0): _InputSelector())`.

So the approach is:

- Add **`extra_repr()`** on every custom module to show semantic info (variable name, op name, shape, etc.).
- Where there is **exactly one child**, register it as a **named attribute** (e.g. `self.arg`, `self.input`) so the default repr prints a short, readable line.

---

## 3. Design decisions

| Decision | Rationale |
|----------|-----------|
| Use `extra_repr()` instead of overriding `__repr__` | Matches PyTorch; integrates with existing module tree printing. |
| Pass variable name into `_InputSelector` | Compiler already has `var_name` (e.g. `z1`) in `_compile_leaf`; storing it allows `extra_repr()` to show `"z1"`. |
| Single-child modules use named attr (e.g. `self.arg`) | Reduces `ModuleList((0): ...)` noise and makes the tree easier to scan. |
| Two-child modules use `self.left` / `self.right` | Same idea: clear names and a compact repr. |
| RelNN pipeline summary in `extra_repr()` | First line becomes e.g. `RelNN(Input -> Test1 -> Out)` so the data flow is obvious. |
| Keep `ModuleList` for multi-child (Concat, MultiArg) | More than two children; naming each would be arbitrary; `ModuleList` is acceptable there. |

---

## 4. Implementation details

All changes are in **`relann/tensor_term_compiler.py`** (transformation modules and compiler) and **`relann/relnn.py`** (RelNN top-level repr). The notebook **`nbs/012_relnn.ipynb`** was updated to match `relnn.py` (nbdev source of truth).

### 4.1 `_InputSelector` (tensor_term_compiler.py)

- **`__init__(self, index: int, name: Optional[str] = None)`**  
  Added optional `name`; store as `self.name`.
- **`extra_repr(self) -> str`**  
  Return `'"z1"'` when `self.name` is set, else `'index={self.index}'`.
- **Compiler** (`_compile_leaf`):  
  When creating `_InputSelector`, pass `name=var_name` whenever the variable name is known (from `var_to_input_index`, or from the `z1`/`z2` parsing). Fallbacks still use `_InputSelector(0)` with no name.

### 4.2 `_UnaryOp` (tensor_term_compiler.py)

- **`__init__`**: Replaced `self.children_modules = nn.ModuleList(children)` with **`self.arg = children[0]`** (single child only in practice).
- **`forward`**: Use **`self.arg(*inputs)`** instead of `self.children_modules[0](*inputs)`.
- **`extra_repr(self) -> str`**: Return `op='transpose'` (or `sqrt`/`view`), and for `view` append `, shape=(2, 2)` when `self.shape` is set.

### 4.3 `_SingleChildWrapper` (tensor_term_compiler.py)

- **`__init__(self, module, children, op_name: str = "")`**: Added **`op_name`**; store as `self._op_name`. Register single child as **`self.input = children[0]`** (no ModuleList).
- **`forward`**: Use **`self.input(*inputs)`** and then `self._module(x)`.
- **`extra_repr(self) -> str`**: Return `op='ReLU'` (or whatever `op_name`) when non-empty.
- **Compiler** (`_wrap_module`): Call **`_SingleChildWrapper(module, child_modules, op_name=op_name)`** so the wrapper knows the op name.

### 4.4 `_NoChildWrapper` and `_WeightOnlyWrapper` (tensor_term_compiler.py)

- **`extra_repr(self) -> str`**: Return **`module={type(self._module).__name__}`** (e.g. `module=Linear`, `module=ReLU`).

### 4.5 `_EqualityWrapper` and `_ArithmeticWrapper` (tensor_term_compiler.py)

- **`__init__`**: Replaced `ModuleList(children)` with **`self.left = children[0]`**, **`self.right = children[1]`**.
- **`forward`**: Use **`self.left(*inputs)`**, **`self.right(*inputs)`**.
- **`_ArithmeticWrapper`**: Added a small map `op_func -> symbol` (e.g. `torch.mul -> "*"`). **`extra_repr(self) -> str`** returns **`op='*'`** (or the matching symbol).

### 4.6 `_TransformDefWrapper` (tensor_term_compiler.py)

- **`__init__`**:  
  - If `len(children) == 1`: set **`self.arg = children[0]`**, **`self.children_modules = None`**.  
  - If `len(children) > 1`: set **`self.arg = None`**, **`self.children_modules = nn.ModuleList(children)`**.  
  - If no children: both `None`.
- **`forward`**: If `self.arg is not None`, call **`self.inner(self.arg(*inputs))`**; else if `self.children_modules` is not None, iterate as before; else **`self.inner(*inputs)`**.

### 4.7 `_ConcatThenModuleWrapper` and `_MultiArgWrapper` (tensor_term_compiler.py)

- **ModuleList** kept (multiple children).
- **`extra_repr(self) -> str`**:  
  - `_ConcatThenModuleWrapper`: return **`module={type(self._module).__name__}`**.  
  - `_MultiArgWrapper`: return **`op={self._op_name!r}`** when `_op_name` is set.

### 4.8 `_ConstantValue` (tensor_term_compiler.py)

- **`extra_repr(self) -> str`**:  
  - If `self.value` is a tensor: if `numel() <= 1` return **`value={v.item()}`**, else **`value=Tensor(shape=(...))`**.  
  - Otherwise return **`value={self.value!r}`** (avoid huge tensor reprs).

### 4.9 `RelNN` (relnn.py and nbs/012_relnn.ipynb)

- **`extra_repr(self) -> str`**:  
  Walk **`self._topo`** and build a de-duplicated pipeline list: for nodes like `transformation_Test1` or `agg_Test1`, take the rule name (e.g. `Test1`); for others (e.g. `Input`) use the node name. Join with **`" -> "`** and return (e.g. **`"Input -> Test1 -> Out"`**).  
  This makes the first line of `print(module)` show the data flow.

---

## 5. Output examples

### 5.1 Minimal example: one rule `Test1(a; transpose(z1)) :- Input(a; z1)`, predict `Out(a; z) :- Test1(a; z)`

After building the module (e.g. with `build_and_run_era_demo_module(session, "Out")`), **`print(module)`** looks like:

```
RelNN(Input -> Test1 -> Out
  (ops): ModuleDict(
    (Input): DataLoader()
    (transformation_Test1): Transformation(
      (transformation): _UnaryOp(
        op='transpose'
        (arg): _InputSelector("z1")
      )
    )
    (agg_Test1): Aggregation()
    (transformation_Out): Transformation(
      (transformation): _InputSelector("z")
    )
    (agg_Out): Aggregation()
  )
)
```

Interpretation:

- **First line**: Pipeline summary `Input -> Test1 -> Out`.
- **`(transformation_Test1)`**: Transformation for rule Test1.
- **`_UnaryOp(op='transpose')`**: DSL-native transpose; **`(arg)`**: single argument.
- **`_InputSelector("z1")`**: Selects the embedding input named `z1`.
- **`(transformation_Out)`**: Pass-through from Test1 to Out; **`_InputSelector("z")`**: selects the single embedding `z`.

### 5.2 With Linear and ReLU (e.g. `Test1(a; ReLU(Linear(4,2)(z1))) :- Input(a; z1)`)

Relevant part of the repr:

```
(transformation_Test1): Transformation(
  (transformation): _SingleChildWrapper(
    op='ReLU'
    (_module): ReLU()
    (input): _SingleChildWrapper(
      op='Linear'
      (_module): Linear(in_features=4, out_features=2, bias=True)
      (input): _InputSelector("z1")
    )
  )
)
```

- **`_SingleChildWrapper(op='ReLU')`**: Wraps `nn.ReLU`; its **`(input)`** is the Linear branch.
- **`_SingleChildWrapper(op='Linear')`**: Wraps `Linear(4, 2)`; **`(input)`** is **`_InputSelector("z1")`**.

### 5.3 Before vs after (conceptual)

**Before** (generic, hard to read):

```
(transformation): _UnaryOp(
  (children_modules): ModuleList(
    (0): _InputSelector()
  )
)
```

**After** (torch-style, readable):

```
(transformation): _UnaryOp(
  op='transpose'
  (arg): _InputSelector("z1")
)
```

---

## 6. How to view the RelNN module in Python

1. **Get a reference to the RelNN module**  
   `session.run(pred_program)` returns only the **predictions** (e.g. an `EmbeddedRelation`), not the module. To inspect the module for a rule, use:

   ```python
   from relann.datasets import build_and_run_era_demo_module

   # After session.run(define_program); "Out" is the predict rule name
   module, cache, nodes = build_and_run_era_demo_module(session, "Out")
   ```

2. **Print the whole module** (torch-style tree + pipeline summary):

   ```python
   print(module)
   ```

3. **Print a specific op** (e.g. the transformation for rule Test1):

   ```python
   print(module.ops["transformation_Test1"])
   ```

4. **Keys in `module.ops`**  
   Names follow the term graph: `Input`, `transformation_Test1`, `agg_Test1`, `transformation_Out`, `agg_Out`, etc.

---

## 7. Testing

- **Existing tests** in `nbs/tests/_feature_tests/test_tensor_term_compiler.py` (and e2e single-op / transpose tests) still pass; they only assert behavior (shapes, values), not repr strings.
- **Repr test** added in `test_tensor_term_compiler.py`: **`test_repr_shows_z1_and_op()`** builds a `transpose(z1)` module and asserts that **`str(module)`** contains `"z1"` and either `"transpose"` or `"op="`, so the new repr is regression-tested.

Run with the project’s conda env, e.g.:

```bash
conda activate parent
pytest nbs/tests/_feature_tests/test_tensor_term_compiler.py -v -k repr
```

---

## 8. Files touched

| File | Changes |
|------|---------|
| `relann/tensor_term_compiler.py` | All wrapper and primitive classes: `extra_repr()`, named single/dual children, compiler passing `name` and `op_name`. |
| `relann/relnn.py` | `RelNN.extra_repr()` for pipeline summary. |
| `nbs/012_relnn.ipynb` | Same `RelNN.extra_repr()` so nbdev export stays in sync. |
| `nbs/tests/_feature_tests/test_tensor_term_compiler.py` | `test_repr_shows_z1_and_op()` and its call in `main()`. |

No changes to `era_operations.Transformation` or other ERA operators; only the **compiler-produced** submodules and the top-level **RelNN** repr were updated.
