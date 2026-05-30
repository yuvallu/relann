---
name: write-relnn-program
description: Write, debug, or modify RelNN DSL programs. Use when the user asks to implement a neural network architecture in RelNN, translate PyTorch to RelNN, fix a RelNN program, or write new RelNN rules.
---

# Writing RelNN Programs

## Prerequisites

Before starting, read the DSL syntax reference:

- `.claude/skills/relann-dsl-reference.md` — full syntax quick-ref
- `.claude/skills/relann-repo-overview.md` — repo orientation and execution model

## Writing a New Program

### Step 1: Understand the data schema

Identify the **relations** (tables) available in the database:
- What are the table names?
- What are the content columns (join keys, IDs)?
- What are the embedding shapes (feature dimensions)?

Example: Cora has `Papers(pid; [1433])`, `Citation(citing, target_id; [1])`, `Labels(target_id; [1])`.

### Step 2: Map the architecture to rules

For each layer or operation in the target architecture:

1. **Identify inputs** — which relations feed into this step?
2. **Identify the relational operation** — is it a join (`,`), union (`|`), filter, or just a transformation of one relation?
3. **Identify the embedding operation** — what happens to the embeddings? (linear projection, activation, attention, arithmetic)
4. **Identify aggregation** — does this step reduce rows? If LHS has fewer attrs than the joined RHS, you need an aggregation function (`sum`, `mean`, etc.)
5. **Name the output relation** — pick a descriptive name

**Translation table:**

| PyTorch / math | RelNN |
|----------------|-------|
| `nn.Linear(a, b)(x)` | `Linear(a, b)(z)` in embedding expr |
| `F.relu(x)` | `ReLU(z)` |
| Element-wise multiply | `z1 * z2` |
| Matrix multiply | `z1 @ z2` |
| Concatenation | `Concat(z1, z2, ...)` |
| Scatter-add / sum over neighbors | `sum(...)` aggregation with join |
| Residual / skip connection | `f(z1) + z2` with two RHS relations joined on same key |
| Multi-head (independent weights) | Template params `<i>`, instantiate `<1>`, `<2>`, ... |

### Step 3: Write constants and transform definitions

```relnn
d = 16 .
h = 4 .
K<i> = Linear(d, d/h) .
```

- Use constants for hyperparameters so they're reusable
- Use transform defs for modules referenced in multiple rules or templated per-head
- **Composite transforms** (chained modules): use the reserved leaf `inp` as the formal input. When you call `MyCompositeTransform(z)` in a rule, the engine replaces all `inp` leaves in the body with the actual argument `z`:

```relnn
# Single-layer transform (unapplied ctor)
Linear1 = Linear(16, 32) .

# Multi-layer transform using inp
MLP = Dropout(0.1)(ReLU()(Linear(32, 32)(inp))) .

# Usage in rules: both expand correctly
Layer1(id; Linear1(z)) :- Input(id; z) .
Layer2(id; MLP(z)) :- Layer1(id; z) .
```

- **When using Python f-strings**, inject external values (feature dimensions, dataset sizes) as RelNN constant assignments at the top of the program, not inline inside rules:

```python
# Good: f-string values become RelNN constants at the top
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
""")
```

### Step 4: Write the rules

Work top-down from input to output:

```relnn
Emb(pid; Linear(1433, 16, False)(z)) :- Papers(pid; z) .
Agg(target; sum(z * w)) :- Emb(source; z), Citation(source, target; w) .
Out(target; ReLU(z)) :- Agg(target; z) .
```

**Checklist for each rule:**
- Content attributes match between LHS and RHS where joins should happen
- Embedding variables on the RHS are used in the LHS embedding expression
- If aggregating, the LHS attrs are a subset of the joined RHS attrs
- Aggregation function wraps the full embedding expression when needed

### Step 5: Write fit and predict

```relnn
?fit <epochs=200, lr=0.01> Loss(; CrossEntropyLoss()(z_pred, z)) :- Out(id; z_pred), Labels(id; z) .
?pred Predictions(id; ArgMax()(z)) :- Out(id; z) .
```

- Fit and predict are typically in **separate** `session.run()` calls from the define program
- The classifier layer can live in the define program or inline in fit/predict

### Step 6: Wire it up in Python

```python
from relann.session import Session

session = Session(db=db)
session.run(define_program)
session.run(fit_program)
result = session.run(pred_program)
```

## Debugging an Existing Program

### Common errors and fixes

**"Unknown relation X"** — The relation name in the RHS doesn't match any defined relation or DB table. Check spelling and case.

**Shape mismatch in transformation** — The embedding dimensions going into a `Linear(a, b)` don't match `a`. Trace the embedding shape from the source relation through each rule.

**Wrong aggregation results** — Check that:
- Join keys are correct (shared attribute names must match)
- The aggregation function is applied (LHS has fewer attrs than joined RHS)
- The right embedding variables are used

**Template instantiation issues** — Every `<i>` in a rule/transform must be instantiated with concrete values when used. `ATT_Head<i>` is a template; `ATT_Head<1>` is an instance.

### Debugging workflow

1. **Read the error traceback** — RelNN errors usually point to the specific rule or operation
2. **Check the term graph** — Use `session.engine.term_graphs['RelationName']` to inspect the computation graph for a specific output relation
3. **Trace shapes** — Start from DB tables, follow through rules, check embedding dimensions at each step
4. **Simplify** — Comment out later rules, verify earlier ones produce expected shapes
5. **Run cells incrementally** — Execute define, then fit, then predict separately to isolate which stage fails

## Modifying an Existing Program

1. **Understand the current flow** — Read all rules to map the data flow
2. **Identify the change point** — Which rule(s) need modification?
3. **Check downstream dependencies** — Rules that reference the modified relation's output may need updates (attribute names, embedding dimensions)
4. **Re-run define** — After changing rules, you must re-run the define program (and usually re-create the session or clear state)
5. **Re-run fit** — Model weights are tied to the architecture; changing rules means retraining

## Reference Examples

For complete working programs, see:
- `nbs/demos/001_relnn_hello_world.ipynb` — 2-layer GCN on Cora (simplest)
- `nbs/demos/002_era_join_projection.ipynb` — single-rule ERA walkthrough
- `nbs/demos/003_relnn_hgt.ipynb` — multi-head HGT with templates
