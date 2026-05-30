# Encode / decode (content ↔ embedding)

**Historical name:** “shuttling” (early design docs).

RelNN rules normally operate on **embedding tensors** attached to relations. **Encode** and **decode** connect **raw table columns** (pandas/cuDF) to those tensors without hiding behavior: encoders and decoders are ordinary `torch.nn.Module` instances, resolved the same way as `Linear`, `ReLU`, etc.

## RHS encode — `[...]` in embedding expressions

- **`[col]`** — `parent/encode.py::tensorize_column` converts int/float/bool/category to tensors. The first time a column is auto-tensorized this way, `parent/tensor_term_compiler.py::_ColumnExtractModule` emits a **one-shot `INFO` notice** (per extract leaf + dtype) so implicit behavior is visible in logs.
- **`[col]` on text/object** (`object`, pandas `string`, …) — a bare `[text_col]` is **not** auto-tensorized; it raises `EncodeTypeError`. Text columns must use an **explicit encoder**, e.g. `[HashBucketTextEncoder(64, 8)(bio)]`. This applies to single- and multi-item brackets alike.
- **`[Encoder(col)]` / `[Encoder(hps)(col)]`** — the compiler builds `_ColumnExtractModule` + `_EncodeWrapper` around the resolved module (`tensor_term_compiler.py`). Built-in resolution includes `HashBucketTextEncoder` / `HashBucket` (alias), same as `ArgMax`.
- **Multi-item** — `[a, b, Linear(1, d)(c)]` compiles to `_MultiEncodeModule` (concat on the last dimension).

`Transformation` injects the input `EmbeddedRelation` into each `_ColumnExtractModule` (`_source_er`, device, dtype) before running the compiled subgraph. Categorical codes use **`EmbeddedRelation.column_vocabs`** when present (from `RelationSource.load_full` / `_to_er_dict`), not module-local vocab state.

## LHS decode — `[...]` in predict rule LHS

Only **`?pred`** rules: after `module.forward()`, `Engine._apply_lhs_decode` writes decoded values into the output relation’s **content** `DataFrame`, using `resolve_op` + the same `_instantiate` path as tensor terms.

- **Bare `[col]`** performs a **trivial decode only**: a 1-D `(N,)` or 2-D `(N, 1)` tensor is written straight into the column. Any other shape (e.g. `(N, K)` with `K > 1`) raises — specify an explicit decoder such as `[ArgMax()(z)]`.

## Caching

`_ColumnExtractModule` caches tensorized (or text `pd.Series` before an encoder) column data keyed by `(id(df), len(df), df.index[0], data_version)` so repeated `forward` passes on the same data (e.g. epochs) avoid redundant pandas work. The composite key prevents false hits from Python's id() reuse after garbage collection. **`data_version`** bumps when a `RelationSource` reloads (`load_full` / `load_by_keys`). Call `leaf.invalidate()` (alias for `clear_cache`) between mini-batches when a new DataFrame slice arrives. Heavy model work (e.g. sentence-transformers) lives in user encoder modules.

## Key implementation files

| Piece | Location |
|-------|----------|
| Dtypes / `EncodeTypeError` / `build_column_vocabs` | `parent/encode.py` |
| Text encoder (explicit, hash bucket) | `parent/encoders.py` — `HashBucketTextEncoder` |
| Column extract, wrappers, multi concat, `resolve_op` builtins | `parent/tensor_term_compiler.py` |
| Inject ER into leaves; reject bare text; encode-only path | `parent/era_operations.py` — `Transformation` |
| Predict decode | `parent/engine.py` — `_apply_lhs_decode` |
| Grammar | `parent/relann_grammar.lark` — `content_encode`, `content_decode`, `encode_item` |
| AST | `parent/pydantic_classes.py` — `ContentEncode`, `ContentDecode`, `EncodeItem` |

## Next steps (not in this doc)

- Decode as a term-graph op (intermediate use in joins/filters), not only post-`predict`.
- Optional `torch_frame` adapter modules.
- Full minibatching via `BatchSpec` (anchor-based node sampling and per-relation independent sampling) — see `docs/design/data-sources.md`, minibatching section.
