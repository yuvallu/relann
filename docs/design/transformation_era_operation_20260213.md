# Transformation ERA Operation and DSL-to-Torch Mapping

This document describes how the **Transformation** Embedded Relational Algebra (ERA) operation works and how a DSL string such as `Linear(in_channels, hidden_channels, False)(z)` is mapped to the underlying PyTorch operation.

## 1. Overview

The Transformation operation is the ERA operator that applies a neural (or arbitrary callable) map to the embeddings of a single embedded relation. The DSL allows users to write expressions like:

- `Linear(1433, 16, False)(z)` — one linear layer applied to embedding variable `z`
- `ReLU(z)` — ReLU activation
- `Linear(16, 7)(ReLU(z))` — composition

The flow from user text to execution is:

1. **Parse** the DSL string into a **TensorTerm** tree (AST).
2. **Term graph** construction adds a **transformation node** that stores that TensorTerm and input-index mapping.
3. **Engine** runs **symbol resolution** (`replace_all_vars_in_tg_using_symbol_table`) and **compilation** (`eval_tensor_terms_on_tg`), which calls **tensor_term_to_module** and stores the resulting `nn.Module` on the node as **torch_transformation**.
4. **RelNN** builds operator modules; for transformation nodes it uses **torch_transformation** and wraps it in the ERA **Transformation** operator.

So: **string → TensorTerm → (replace_all_vars) → tensor_term_to_module → nn.Module → ERA Transformation**.

---

## 2. Parsing: String to TensorTerm

### Grammar

The RelNN grammar (e.g. `parent/relann_grammar.lark`) defines:

- **Embedding expression**: `embedding_expression` contains `tensor_sequential`, which is a list of `tensor_term`s.
- **Module constructor call**:  
  `module_class "(" hyper_param_list ")" "(" module_arguments ")"` → `module_ctor_call`

So `Linear(in_channels, hidden_channels, False)(z)` is parsed as: class name, hyperparameters, then arguments.

### Parser (`parser.py`)

- **module_class(meta, items)**  
  Receives the token (e.g. `"Linear"`). Calls **engine._function_or_nn_module_exists(class_name)** to ensure the name exists in:
  - builtins
  - caller’s globals
  - engine module globals
  - **torch.nn**
  - user-defined subclasses of `nn.Module` (by name in globals)

  If found, it **returns the same string** `"Linear"` (the parser does not resolve to a class object).

- **module_ctor_call(meta, children)**  
  Builds:
  ```text
  TensorTerm(
      op=TensorOp(op=module_class, hyper_params=hyper_param_list),
      sons=module_arguments
  )
  ```
  So `op.op` is the string `"Linear"`, `op.hyper_params` is the list of ArithTerms (e.g. `in_channels`, `hidden_channels`, `False`), and `sons` is the list of tensor terms for the arguments (e.g. one term for `z`).

### Pydantic structures (`pydantic_classes.py`)

- **TensorOp**: `op: str`, `hyper_params: Optional[List[ArithTerm]]`
- **TensorTerm**: `op: Optional[TensorOp]`, `sons: Optional[List[TensorTerm]]`, `value: Optional[Union[Primitive, Var, ...]]`

After parsing, the tree is symbolic: e.g. root `op.op == "Linear"`, `op.hyper_params == [Var("in_channels"), ...]`, `sons == [TensorTerm(value=Var("z"))]`.

---

## 3. Term Graph: Transformation Node

In **term_graph.py**, when building the graph from a rule:

- The **embedding expression** is taken from the rule LHS; its **tensor_term** may be wrapped by an aggregation (e.g. `sum(z*w)`). If so, the aggregation is stripped and the **inner** tensor_term is used for the transformation.
- A node is added with:
  - `type="transformation"`
  - `transformation=tensor_term` (the TensorTerm tree)
  - **var_to_input_index**: mapping from embedding variable names (e.g. `"z"`) to 0-based input indices (remapped when the parent is a Join/Union).

The graph node holds the **symbolic** TensorTerm; no torch module yet.

---

## 4. Engine: Symbol Resolution and Compilation

Before building the RelNN module, the engine:

### 4.1 replace_all_vars_in_tg_using_symbol_table (`engine.py`)

- Walks transformation nodes’ `transformation` TensorTerms.
- **ArithTerms**: If a Var (e.g. `in_channels`) is in the symbol table and refers to a **scalar TransformDef**, it is replaced by that scalar value.
- **TensorTerms**: If `op.op` is a name that refers to a **TransformDef**, the whole term is replaced by that TransformDef’s tensor_term (with current sons as arguments).

After this, e.g. `Linear(in_channels, hidden_channels, False)` has hyperparameters **resolved** to numbers/booleans (1433, 16, False) using the session’s constants.

### 4.2 eval_tensor_terms_on_tg (`engine.py`)

- For each node with `type == 'transformation'` and a `transformation` TensorTerm:
  - Calls **tensor_term_to_module(tterm, var_to_input_index)**.
  - Sets **tg.nodes[node]['torch_transformation'] = torch_module**.
  - Extracts parameters into **engine.parameters** (for loading into the RelNN module later).

The **only** place that maps the string `"Linear"` (and other op names) to a torch callable is inside **tensor_term_to_module**.

---

## 5. tensor_term_to_module: String to nn.Module

**Location**: `engine.py`, method **tensor_term_to_module**.

### 5.1 Hardcoded mapping

At the start of the function:

```python
ops_to_modules = {
    "Linear": nn.Linear,
    "Concat": Concat,
    "ReLU": nn.ReLU,
    "MSELoss": nn.MSELoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "ArgMax": ArgMax,
}
```

So **"Linear"** is mapped to **nn.Linear** only here; there is no generic resolution from the string to `torch.nn` or run scope.

### 5.2 Recursion

- **Leaves** (`op is None`):  
  - `value` is a **Var** (e.g. `z`) → **InputSelector(i)** using `var_to_input_index` (or fallback by parsing `z1`, `z2`, …).  
  - Constants → **ConstantValue**.
- **Binary ops** (`*`, `+`, `-`, `/`, `@`, `**`, `==`): build two child modules and wrap in small nn.Modules that call the corresponding torch function.
- **Unary** (`sqrt`, `transpose`, `view`): one child; for `view`, hyperparameters give the target shape.
- **Named op** (e.g. `Linear`):
  - If `op_name` is in **ops_to_modules**: use that constructor; then a long if/elif chain handles each op (Linear, Concat, ReLU, MSELoss, CrossEntropyLoss, ArgMax) with op-specific logic (e.g. Linear: evaluate hyper_params, instantiate, wrap in **_LinearWrapper**).
  - Else if `op_name` is a **TransformDef** in the symbol table: recursively convert its tensor_term and wrap in **_TransformDefWrapper**.

So the mapping from the **name** to the **torch callable** is entirely in this hardcoded dict and the following if/elif chain. Adding a new op requires editing this dict and the corresponding branch.

---

## 6. RelNN: Using the Built Module (Transformation ERA Op)

In **relnn.py**, **\_build_transformation_operator**:

- Reads **torch_ctor = node.get("torch_transformation", None)** and **dsl_term = node.get("transformation", None)**.
- If **torch_ctor** is not None (normal case after **eval_tensor_terms_on_tg**): it is already an **nn.Module** instance. **\_instantiate_callable_from_hyper** then just (optionally) loads saved parameters and returns it; no re-instantiation from string.
- So the module that runs is the one built by **tensor_term_to_module**; the DSL term is only used for pass-through detection (Var-only) or when there is no **torch_transformation**.
- Finally: **Transformation(transformation=callable_mod, output_schema=...)** is built and returned as the ERA operator for this node.

So the **Transformation** ERA operation is: take one EmbeddedRelation, apply the stored **nn.Module** (from **torch_transformation**) to its embeddings, and return a new EmbeddedRelation with the transformed embeddings and optional schema rename.

---

## 7. ERA Transformation Class (`era_operations.py`)

- **Transformation** is an nn.Module that holds **transformation: nn.Module** and optional **output_schema**.
- **instantiate(sons)**: Takes one son; uses the son’s embedding shapes to build dummy inputs, runs **self.transformation(*dummy_inputs)** to infer output shape, returns an EmbeddedRelation with the same content and the new embedding shapes (no real embeddings yet).
- **forward(sons)**: Takes one son; takes son’s embeddings, moves to module device/dtype, calls **self.transformation(*inputs)**; if the result is a tuple/list, takes the single tensor; returns EmbeddedRelation with **content** (and optional schema rename) and **embeddings = [out]**.

So the “Linear” string ends up as **nn.Linear** inside this **transformation** submodule; the ERA layer only forwards and handles shapes/content.

---

## 8. End-to-End Example (Cora GCN)

Rule:

```text
PapersEmb1(pid; Linear(in_channels, hidden_channels, False)(z)) :- Papers(pid; z) .
```

- **Parser**: TensorTerm with `op.op="Linear"`, `op.hyper_params=[in_channels, hidden_channels, False]`, `sons=[TensorTerm(value=Var("z"))]`.
- **Term graph**: Node `transformation_PapersEmb1` with that TensorTerm and `var_to_input_index={"z": 0}`.
- **replace_all_vars**: `in_channels`→1433, `hidden_channels`→16, `False`→False (from session constants).
- **tensor_term_to_module**: Root op `"Linear"` → nn.Linear(1433, 16, bias=False); one son `z` → InputSelector(0); **_LinearWrapper(linear, [InputSelector(0)])**.
- **eval_tensor_terms_on_tg**: That wrapper is stored as **torch_transformation** and its parameters are registered.
- **RelNN**: **\_build_transformation_operator** gets that module and wraps it in **Transformation(transformation=that_module)**; at runtime **Transformation.forward** runs **self.transformation(*embeddings)** i.e. **Linear(1433,16)(z)**.

---

## 9. Summary: Where “Linear” Becomes Torch

- **Parser**: Keeps “Linear” as a **string** in **TensorTerm.op.op** and only validates existence via **\_function_or_nn_module_exists** (builtins, globals, torch.nn, user nn.Modules).
- **Mapping to torch**: Done **only** in **engine.tensor_term_to_module**, via the **hardcoded** **ops_to_modules** dict and the subsequent if/elif for Linear, ReLU, Concat, losses, ArgMax, etc. There is **no** generic resolution of the string to an nn.Module from the run scope or from `torch.nn`; adding a new op requires editing this dict and the corresponding branch in **tensor_term_to_module**.
