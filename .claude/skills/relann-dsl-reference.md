---
name: relann-dsl-reference
description: Use when writing, reading, or debugging RelaNN DSL programs. Quick-reference for every construct in the RelaNN language - rule syntax (LHS/RHS), transforms, templates with specialization, operators (join `,`, union `|`), fit/predict statements, encode/decode brackets. Read together with write-relnn-program when authoring new programs.
---

# RelNN DSL Syntax Reference

Quick reference for every construct in the RelNN language. For high-level orientation see `relnn-repo-overview.mdc`.

**Style (agents):** When writing RelNN DSL, keep each rule on a single line where possible — avoid breaking in the middle of a rule (e.g. LHS and RHS on separate lines). The language/color server does not yet support mid-rule line breaks well, so prefer one line per rule for correct highlighting.

## Statements

A program is a sequence of statements, each terminated by `.`

| Statement | Syntax |
|-----------|--------|
| Constant | `name = expr .` |
| Transform def | `Name<i> = Linear(d, d/h) .` |
| Rule | `LHS(...) :- RHS(...) .` |
| Fit | `?fit <params> LossRule .` |
| Predict | `?pred OutputRule .` |
| Function def | `def Name<T>(Param: schema) -> schema: body enddef` |

## Rules

```relnn
DerivedRelation(attr1, attr2; embedding_expr) :- Input1(a, b; z1), Input2(b, c; z2) .
```

- **LHS** — output relation name, content attributes, embedding expression (after `;`)
- **RHS** — input relations joined by `,` (join) or `|` (union), each with content attrs and embedding var
- **Join keys** are inferred from shared attribute names across RHS relations
- **Aggregation** occurs when LHS has fewer content attributes than the joined RHS (group-by on LHS attrs)

## Embedding Expressions

After the `;` in the LHS. Supports:

| Form | Example | Notes |
|------|---------|-------|
| Variable passthrough | `z` | |
| Module constructor | `Linear(16, 7, False)(z)` | `ClassName(hyperparams)(inputs)` |
| Named module | `MyLin(z)` | Defined via transform def |
| Arithmetic | `z1 * z2 + z3` | `+`, `-`, `*`, `/`, `@` (matmul), `**` |
| Equality | `z1 == z2` | Returns 0/1 tensor |
| Aggregation | `sum(z * w)` | Wraps the full expression |
| Splat | `*z` | Unpack embedding dims |
| Multi-arg | `CrossEntropyLoss()(z_pred, z_label)` | Comma-separated inside call |

### Aggregation functions

`sum`, `add`, `mean`, `avg`, `min`, `max`, `count`

### Module resolution order

1. DSL-native ops (`view`, `sqrt`, `transpose`, `unsqueeze`)
2. Run scope (user-registered modules)
3. Built-ins (`ArgMax`, `Concat`, `Tensor`)
4. `torch.nn` (e.g. `Linear`, `ReLU`, `CrossEntropyLoss`)
5. `torch` (e.g. `softmax`)

## Transform Definitions (named modules)

The right-hand side is a full tensor expression (same grammar as embedding expressions).

```relnn
K<i> = Linear(d, d/h) .
A_LIN = Linear(d, d) .
Mu<k, i> = Mu_L3<k, i>(ReLU()(Mu_L2<k, i>(ReLU()(Mu_L1<k, i>(inp))))) .
```

- Creates a reusable module binding; use as `K<1>(z)`, `A_LIN(z)` in rules
- Template params `<i>` produce independent weights per instantiation
- Single-paren `Linear(in, out)` and `Linear(in, out, bias)` on the RHS are stored as constructor hyperparameters (not as a “call” with two tensor children), so they compose correctly with `K(z)`.
- For a **composite** transform (e.g. multi-layer MLP), use the reserved leaf `inp` as the formal input in the definition body. At compile time, `MyTransform<...>(z)` replaces every `inp` in that body with the actual argument, so the nested `Mu_L* / ReLU` stack can be written once and used as `Mu<'C2', 0>(z)`.
- `true` / `false` in `Linear(..., false)` are treated as boolean literals for the `bias` argument.

## Template Parameters

```relnn
ATT_Head<i>(s, t; K<i>(z1) * Q<i>(z2) * Mu<i>) :- Emb(s; z1), Edge(s, t; w), Emb(t; z2) .
```

- `<i>` in the definition declares a template parameter
- Instantiate with concrete values: `ATT_Head<1>`, `ATT_Head<2>`, etc.
- Each instantiation gets its own learned parameters
- Works on rules, transform defs, and function defs

### Template Specialization (Recursion Base Cases)

Multiple definitions of the same template name are allowed. The engine dispatches to the most-specific match (C++ template specialization semantics):

```relnn
H<'Author', 0>(id; ReLU(Linear(334, d)(z))) :- Author(id; z) .   # base case
H<'Paper',  0>(id; ReLU(Linear(4231, d)(z))) :- Paper(id; z) .    # base case
def H<type, layer>(): ...  enddef                                   # general case
```

- Concrete values in template params (e.g. `'Author'`, `0`) make the definition more specific
- Variables (e.g. `type`, `layer`) are wildcards that match anything
- Most-specific match wins; equal specificity raises an ambiguity error
- Enables recursive template unrolling with base-case termination

### Bounded Sets (Compile-Time Expansion)

Bounded sets expand a template over a guard relation or integer range at compile time.

**Guard-relation bounding** — iterate over database rows:
```relnn
Agg(t; sum(z)) :- Union(Set(EdgeAgg<L, ts, pe, tt>(t; z) | MetaRel(ts, pe, tt))) .
```
- `MetaRel(ts, pe, tt)` is a DB relation; each row provides concrete values for free vars `ts`, `pe`, `tt`
- Bound variables (quoted strings) filter rows: `MetaRel(ts, pe, 'Author')` keeps only rows where 3rd col = Author

**Condition-only bounding** — iterate over an integer range:
```relnn
AllHeads(s, t; Concat(*z)) :- Join(Set(WMsg<i>(s, t; z) | 1 <= i, i <= h)) .
```
- `1 <= i, i <= h` defines the range for `i` (bounds resolved from constants)
- Each value of `i` produces a concrete ER: `WMsg<1>`, `WMsg<2>`, etc.

**Union vs Join**:
- `Union(Set(...))` — stack rows from all expanded ERs (same embedding var)
- `Join(Set(...))` — join expanded ERs side-by-side; embedding vars are renamed (`z` -> `z_1, z_2, ...`)

**Concat splat** (`*z`): When used with `Join(Set(...))`, `Concat(*z)` on the LHS auto-expands to `Concat(z_1, z_2, ...)` matching the number of expanded ERs.

## Encode / decode (content ↔ embedding)

**Terminology:** this boundary was previously called “shuttling.”

Brackets **`[...]`** mark the boundary between **content columns** (DataFrame) and **embedding tensors**. Everything *outside* brackets is tensor-to-tensor (same resolution order as transformations).

### RHS encode (content → tensor)

| Form | Meaning |
|------|---------|
| `[col]` | Auto-tensorize numeric / bool / categorical (see `relann/encode.py`). Text/object columns **must** use an explicit encoder. |
| `[Encoder(col)]` | Run encoder module with no ctor hyperparameters, e.g. `[GloVe(bio)]`. |
| `[Encoder(hps)(col)]` | Run encoder with hyperparameters, e.g. `[Linear(1, 64)(age)]`, `[Embedding(10, 32)(dept)]`. |
| `[a, b, ...]` | Several **encode items**; outputs are concatenated on the last dim. Items can mix bare columns and encoded columns. |

Encoders are normal `torch.nn.Module` classes resolved like `Linear` / `ReLU` (run scope → built-ins → `torch.nn`).

### LHS decode (predict only; embedding → content)

Used in **`?pred`** rules as derived content attributes:

| Form | Meaning |
|------|---------|
| `[var]` | Trivial decode: a 1-D or `(N, 1)` embedding is written straight into the column. An `(N, K)` (K>1) embedding raises — use an explicit decoder. |
| `[Decoder(var)]` | Apply module, e.g. `[ArgMax(pred)]`. |
| `[Decoder(hps)(var)]` | Apply module with ctor args, e.g. `[Softmax(1)(pred)]`. |

Implemented in `Engine.predict()` as a post-step on the forward `EmbeddedRelation` (`Engine._apply_lhs_decode`).

## Fit & Predict

```relnn
?fit <epochs=200, lr=0.01, weight_decay=0.0005>
Loss(; CrossEntropyLoss()(z_pred, z_label)) :- Output(id; z_pred), Labels(id; z_label) .

?pred Predictions(id; ArgMax()(z)) :- Output(id; z) .
```

Fit params: any `name = arith_expr` pairs (resolved against constants).

## Function Definitions

```relnn
def GCNLayer<T>(Input: (int; T), Edge: (int, int;)) -> (int; T):
    Emb(pid; Linear(T, T)(z)) :- Input(pid; z) .
    Agg(cited; sum(z * w)) :- Emb(citing; z), Edge(citing, cited; w) .
enddef
```

- `<T>` — template params
- `Input: (int; T)` — typed ER parameter with content types and embedding dims
- Body is a sequence of rules; last rule's output is the return value

## Filter Expressions

```relnn
Result(a; z) :- Table1(a, b; z), Table2(b; w), a > 5 .
```

Comparisons: `==`, `!=`, `>`, `<`, `>=`, `<=` — placed after the last RHS relation.

## Tensor Shapes (Row-First Convention)

All embeddings are `(E, *feature_dims)` where E is the row dimension. Smart ops in `relann/smart_ops.py` enforce this:

- Arithmetic (`+`, `-`, `*`, `/`, `**`, `@`) auto-aligns dims, preserving E
- `transpose(z)` — `(E, d)` to `(E, d, 1)`; 3-D+ swaps last two dims
- `view(dims)(z)` — reshapes feature dims only: `(E, *old)` to `(E, *dims)`
- `sqrt`, `exp`, `ReLU`, etc. — element-wise, no special handling needed

## F-String Best Practice

When building RelNN programs with Python f-strings, **inject external values as constant assignments at the top** rather than inlining them in rules. This keeps rules readable and purely in DSL syntax.

```python
# Good: declare dimensions as constants, rules stay clean
session.run(f"""
d_in  = {input_dim} .
d_out = {output_dim} .
hidden = 64 .

Emb(id; ReLU()(Linear(d_in, hidden)(z))) :- Input(id; z) .
Out(id; Linear(hidden, d_out)(z))        :- Emb(id; z) .
""")

# Avoid: f-string interpolations scattered inside rules
session.run(f"""
Emb(id; ReLU()(Linear({input_dim}, 64)(z))) :- Input(id; z) .
Out(id; Linear(64, {output_dim})(z))        :- Emb(id; z) .
""")
```

## Common Patterns

**GCN message passing:**
```relnn
Agg(target; sum(z * w)) :- NodeEmb(source; z), Edge(source, target; w) .
```

**Multi-head attention (HGT-style, generic with bounding):**
```relnn
K_Lin<L, ts, i> = Linear(d, d/h) .
Q_Lin<L, tt, i> = Linear(d, d/h) .

def ATT_Head<L, ts, pe, tt, i>():
    K(s; K_Lin<L, ts, i>(z)) :- H<ts, L-1>(s; z) .
    Q(t; Q_Lin<L, tt, i>(z)) :- H<tt, L-1>(t; z) .
    Dot(s, t; view(1)(z_q @ transpose(z_k)) * Mu<L, pe, i> / sqrt(dh)) :- K(s; z_k), pe(s, t; w), Q(t; z_q) .
    Out(s, t; z) :- Dot(s, t; z) .
enddef

def EdgeAgg<L, ts, pe, tt>():
    WMsg<i>(s, t; z_att * z_msg) :- ATT_Head<L, ts, pe, tt, i>(s, t; z_att), MSG_Head<L, ts, pe, tt, i>(s, t; z_msg) .
    AllHeads(s, t; Concat(*z)) :- Join(Set(WMsg<i>(s, t; z) | 1 <= i, i <= h)) .
    Out(t; sum(z)) :- AllHeads(s, t; z) .
enddef
```

**Residual connection:**
```relnn
Output(t; A_LIN(ReLU(z1)) + z2) :- Aggregated(t; z1), SkipSource(t; z2) .
```

**Classifier head (in fit/predict, not in define):**
```relnn
?fit <...> Loss(; CrossEntropyLoss()(Classifier(z_pred), z)) :- Output(id; z_pred), Labels(id; z) .
?pred Preds(id; ArgMax()(Classifier(z))) :- Output(id; z) .
```
