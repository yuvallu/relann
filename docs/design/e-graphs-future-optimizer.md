# E-Graphs as a Future Optimizer Backend

Status: **deferred** (revisit when we have ~8-10 rewrite rules or hit ordering conflicts)

## What are e-graphs?

An **e-graph** (equality graph) is a data structure that compactly represents many equivalent forms of an expression simultaneously. It consists of:

- **E-nodes**: individual operations (like `*`, `join`, `Linear`)
- **E-classes**: groups of e-nodes known to be equivalent

### The problem with sequential rewriting

Our current optimizer applies rewrite rules one at a time in a fixed order. This creates the **phase-ordering problem**: the order in which you apply rewrites matters, and a greedy choice can block a better optimization downstream.

Example with arithmetic rewrites:

```
Expression: (a * 2) / 2

Rules:
  (x * y) / z  =>  x * (y / z)
  x * 2        =>  x << 1
  x / x        =>  1
  x * 1        =>  x

Bad path:  (a * 2) / 2  =>  (a << 1) / 2  =>  stuck
Good path: (a * 2) / 2  =>  a * (2 / 2)   =>  a * 1  =>  a
```

Applying `x * 2 => x << 1` first leads to a dead end. The sequential optimizer has no way to backtrack.

### How e-graphs solve this

Instead of committing to one rewrite, an e-graph **merges** the old and new forms into the same equivalence class. All rewrites are explored simultaneously:

1. **Saturate**: Apply all rewrite rules repeatedly. Each application merges the result into the e-graph without discarding the original. The e-graph grows to contain all reachable equivalent forms.
2. **Extract**: Use a **cost model** to pick the cheapest equivalent form. For us, cost = `wall_ms` from profiling.

This eliminates the phase-ordering problem entirely: you never commit to a single rewrite path.

## Why it's relevant to RelNN

- We plan to add ~10 more optimizations. With sequential application, ordering them correctly becomes fragile (we already saw `MergeConsecutiveGroupBy` interacting with `EliminateIdentityTransformation` in a way that required a bug fix to `input_order` propagation).
- The cost model / extraction step is a natural fit for our profiling infrastructure: `engine.get_profile()` produces per-node `wall_ms` that could directly feed into e-graph extraction.
- Well-studied in compiler literature (LLVM, Cranelift) and ML compilers (TVM, TASO).

## How it would work in RelNN

```
Term graph (nx.DiGraph)
    |
    v
Encode nodes as e-graph terms:
    DataLoader("Author")
    Transformation(Linear(334,64), agg_Author)
    Aggregation(group_by=[author_id], transformation_Author)
    Join(agg_A, agg_B, on=[author_id])
    |
    v
Define rewrite rules as equivalences:
    # Identity elimination
    Transformation(Identity, ?x) <=> ?x

    # Redundant group-by
    Aggregation(keys=K, Transformation(?, Aggregation(keys=K, ?x)))
        <=> Transformation(?, Aggregation(keys=K, ?x))

    # Consecutive group-by merge
    Aggregation(keys=K, Aggregation(keys=K, ?x))
        <=> Aggregation(keys=K, ?x)
    |
    v
Saturate (apply all rules to fixed point)
    |
    v
Extract cheapest graph using cost model:
    cost(node) = profiled wall_ms for that node type + row count
    |
    v
Decode back to nx.DiGraph
```

## Available tooling

- **egglog** (`pip install egglog`): Python bindings to the Rust `egglog` library (successor to `egg`). Supports equality saturation, e-class analysis, custom cost functions, and Datalog integration.
- Mature Rust core, relatively young Python wrapper.

## Why we're deferring

1. **Encoding complexity**: Our term graph has heterogeneous node types (data_loader, join, agg, transformation) with rich metadata (group_by_refs, merge_steps, output_schema). Encoding these as e-graph terms is non-trivial and lossy.
2. **Current scale is fine**: 4 optimizations work well with sequential detect/apply. The phase-ordering problem hasn't bitten us in practice yet.
3. **Debuggability**: Sequential rewriting with logging (`logger.info("removed %s, rewired %s -> %s")`) is easy to trace. E-graph saturation is harder to inspect.
4. **Dependency**: Adding `egglog` as a dependency brings in Rust compilation requirements.

**Trigger to revisit**: when we have ~8-10 rewrite rules and start seeing cases where rule A blocks rule B depending on application order.

## References

- [A Gradual Introduction to E-Graphs](https://www.cole-k.com/2023/07/24/e-graphs-primer/) -- Cole Krumbholz
- [egg: Fast and Extensible Equality Saturation](https://egraphs-good.github.io/) -- Max Willsey et al.
- [egglog Python docs](https://egglog-python.readthedocs.io/)
- [An Introduction to E-Graphs (WiCT 2023)](https://community-dot-o.llvm.org/wict-meetups/) -- Rebecca Swords
- [Equality Saturation: A New Approach to Optimization](https://dl.acm.org/doi/10.1145/1480881.1480915) -- Tate et al. (POPL 2009)
