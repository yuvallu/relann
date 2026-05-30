# Row-First Tensor Convention

This document defines the embedding layout convention used throughout RelNN and the smart operations that enforce it.

## The Invariant

Every embedding tensor in RelNN follows the **row-first** layout:

```
(E, *feature_dims)
```

- **E** (dim 0) is the **row dimension** -- one entry per row in the parent `EmbeddedRelation`.
- **feature_dims** (dims 1..N) are the per-row feature dimensions.

This invariant is maintained across all operations: joins, transformations, aggregations, and arithmetic.

## Tensor Taxonomy

Per-row embeddings fall into three categories:

| Category | Shape | Example |
|----------|-------|---------|
| Scalar | `(E,)` or `(E, 1)` | attention logit, loss value |
| Vector | `(E, d)` | node embedding, linear output |
| Matrix | `(E, a, b)` | multi-head reshaped embedding |

Higher-rank tensors (`(E, a, b, c, ...)`) are permitted but rare.

## Why Smart Ops Exist

PyTorch broadcasting is **right-aligned**: it matches dimensions from the trailing end. This is correct for plain batch tensors but wrong for RelNN's row-first layout where dim 0 is a shared batch dim:

```
(E, 1, 1) * (E, 1)    PyTorch native -> (E, E, 1)    WRONG
(E, d) @ (E, d, 1)    PyTorch native -> (E, E, 1)    WRONG
```

Smart ops fix this by aligning dimensions **before** delegating to torch, so dim 0 always stays the row dimension.

## Smart Ops Reference

All smart ops live in `parent/smart_ops.py`. Each preserves dim 0 and operates only on feature dimensions.

### Elementwise arithmetic: `smart_mul`, `smart_add`, `smart_sub`, `smart_div`, `smart_pow`

Pad the lower-ndim operand with **trailing** size-1 dims until both operands have the same ndim, then delegate to the corresponding `torch.*` function. Scalars (0-D) and same-ndim pairs pass through unchanged.

```
(E, d)    * (E, a, b)  ->  (E, d, 1) * (E, a, b)  via trailing unsqueeze
(E, a, b) + (E, d)     ->  (E, a, b) + (E, d, 1)  via trailing unsqueeze
```

### Matrix multiply: `smart_matmul`

Auto-unsqueezes the 2-D operand when mixed with a 3-D operand:

```
(E, d) @ (E, d, 1)    ->  unsqueeze left  -> (E, 1, d) @ (E, d, 1) = (E, 1, 1)
(E, a, b) @ (E, b)    ->  unsqueeze right -> (E, a, b) @ (E, b, 1) = (E, a, 1)
(E, a, b) @ (E, b, c) ->  unchanged       -> (E, a, c)
```

### Transpose: `smart_transpose`

Transposes the feature dimensions only:

```
0-D / 1-D            ->  unchanged (no feature dims)
2-D  (E, d)          ->  (E, d, 1)     column-vector per row
3-D+ (E, a, b, ...)  ->  swap last two dims
```

### View: `smart_view`

Reshapes the feature dimensions into a new shape, preserving the row dimension:

```
0-D / 1-D            ->  reshape entire tensor (no row dim)
2-D+ (E, *features)  ->  (E, *shape)   where prod(features) == prod(shape)
```

Uses `reshape` (not `view`) internally so non-contiguous tensors (e.g. after transpose) work without `.contiguous()`.

## DSL Surface

In the RelNN DSL, smart ops are invoked through standard syntax:

| DSL syntax | Smart op | Example |
|------------|----------|---------|
| `z1 + z2` | `smart_add` | `Output(t; z1 + z2) :- A(t; z1), B(t; z2) .` |
| `z1 * z2` | `smart_mul` | attention weighting |
| `z1 @ z2` | `smart_matmul` | matrix multiply |
| `transpose(z)` | `smart_transpose` | `transpose(Q(z))` |
| `view(h, d/h)(z)` | `smart_view` | reshape vector to multi-head matrix |
| `sqrt(d)` | element-wise | no smart op needed |

Element-wise functions (`sqrt`, `exp`, `ReLU`, etc.) do not need row-first awareness because they operate independently on every element.

## Implementation Location

- Smart ops: `parent/smart_ops.py`
- DSL-native op compilation (view, sqrt, transpose): `parent/tensor_term_compiler.py` (`_UnaryOp`)
- Arithmetic compilation: `parent/tensor_term_compiler.py` (`_ArithmeticWrapper`)
- Op resolution order: `parent/tensor_term_compiler.py` (`resolve_op`)
