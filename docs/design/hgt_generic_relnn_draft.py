"""
HGT (Heterogeneous Graph Transformer) in Generic RelNN
=======================================================

This design doc describes how generic HGT is implemented in RelNN using
**recursion base cases** and **bounding**. Each function def in the actual
DSL corresponds to one equation from the HGT paper:
    Hu et al., "Heterogeneous Graph Transformer", WWW 2020
    https://arxiv.org/abs/2003.01332

The same HGT library works for both homogeneous (Cora) and heterogeneous
(DBLP) graphs — only the base cases and graph schema change.

**Current DSL and runnable code:** see `tests/slow/run_compare_dblp_hgt_generic.py`
(RELNN_GENERIC_DEFINE / HGT_LIBRARY). That file is the single source of truth.

STATUS: Recursion base cases and bounded sets are implemented.
    - Recursion base cases: H<'Author', 0>, H<'Paper', 0>, etc.
    - Bounded sets: Union(Set(... | MetaRel(...))), Join(Set(... | 1 <= i, i <= h))
    - MetaRel comes from the DB (load_dblp_dataset / load_cora_dataset)
"""

# ═══════════════════════════════════════════════════════════════════
#  Paper Notation → RelNN Name Mapping
# ═══════════════════════════════════════════════════════════════════
#
#  Paper                     RelNN                  Indexed by
#  ────────────────────────  ─────────────────────  ────────────────
#  K-Linear^i_τ(s)           K_Lin<L, τs, i>        layer, src type, head
#  Q-Linear^i_τ(t)           Q_Lin<L, τt, i>        layer, tgt type, head
#  M-Linear^i_τ(s)           M_Lin<L, τs, i>        layer, src type, head
#  W^ATT_ϕ(e)                W_ATT<L, ϕe, i>        layer, edge type, head
#  W^MSG_ϕ(e)                W_MSG<L, ϕe, i>        layer, edge type, head
#  μ⟨τ(s),ϕ(e),τ(t)⟩        Mu<L, ϕe, i>           layer, edge type, head
#  A-Linear_τ(t)             A_Lin<L, τt>            layer, tgt type
#  skip_τ(t)                 Skip<L, τt>             layer, tgt type
#
# ═══════════════════════════════════════════════════════════════════

# Structure of the generic HGT library (see run_compare_dblp_hgt_generic.py for DSL):
#
#  - Learnable weights: K_Lin, Q_Lin, M_Lin, W_ATT, W_MSG, Mu, A_Lin, Skip (all templated)
#  - Softmax(Scores): reusable 3-rule exp → sum → divide by target
#  - ATT_Head<L, ts, pe, tt, i>: K, Q, Dot (with W_ATT, Mu), Softmax(Dot)
#  - MSG_Head<L, ts, pe, tt, i>: M, then Out with W_MSG
#  - EdgeAgg<L, ts, pe, tt>: WMsg<i> = ATT_Head · MSG_Head; AllHeads = Join(Set(WMsg<i> | 1<=i<=h)); Out = sum
#  - H<tt, L>: Agg = Union(Set(EdgeAgg<L,ts,pe,tt> | MetaRel(ts,pe,tt))); Updated; Out = residual with H<tt,L-1>
#
# Base cases (layer 0): one rule per node type, e.g. H<'Author', 0>(id; ReLU(Linear(334,d)(z))) :- Author(id; z) .

# Compile-time unrolling for Cora with 2 layers:
#
#   Output(t; z) :- H<'Papers', 2>(t; z)
#
#   H<'Papers', 2> → def H<tt='Papers', L=2>():
#     Agg ← Union(EdgeAgg<2,'Papers','Citation','Papers'> | MetaRel(..))
#            only 1 meta relation matches → single EdgeAgg instance
#     Updated ← A_Lin<2,'Papers'>(GELU(Agg))
#     Out ← skip-gate residual with H<'Papers', 1>
#
#   H<'Papers', 1> → same structure, referencing H<'Papers', 0>
#
#   H<'Papers', 0> → BASE CASE rule (Linear projection of raw features)
#                     Recursion stops here.

# Compile-time unrolling for DBLP with H<'Author', 1>:
#
#   H<'Author', 1> → def H<tt='Author', L=1>():
#     Bounding: MetaRel(ts, pe, 'Author')
#       → matches 1 row: ('Paper', 'PaperAuthor', 'Author')
#       → materializes EdgeAgg<1, 'Paper', 'PaperAuthor', 'Author'>
#
#     Inside EdgeAgg<1, 'Paper', 'PaperAuthor', 'Author'>:
#       ATT_Head, MSG_Head (per head), AllHeads = Join(Set(WMsg<i> | 1<=i<=h)), Out = sum
#
#     Updated ← A_Lin<1,'Author'>(GELU(Agg))
#     Out ← skip-gate residual with H<'Author', 0>  ← base case


# ═══════════════════════════════════════════════════════════════════
#  DESIGN OBSERVATIONS
# ═══════════════════════════════════════════════════════════════════
#
#  1. THE HGT LIBRARY IS IDENTICAL FOR CORA AND DBLP.
#     Only the MetaRel facts and H<..., 0> base cases differ.
#
#  2. FULL SPECIALIZATION IS SUFFICIENT FOR BASE CASES.
#     Each node type has different raw features (1433 vs 334 vs 4231
#     vs 50), so H<'Papers', 0>, H<'Author', 0>, etc. are necessarily
#     distinct rules. No partial specialization needed.
#
#  3. MIXED ENTITY TYPES (rule vs function) ARE NEEDED.
#     Base cases are simple one-liner rules: H<'Papers', 0>(...) :- ...
#     Recursive cases are multi-rule function defs: def H<tt, L>(): ...
#     The dispatch picks the most-specific match (concrete beats var).
#
#  4. TEMPLATE ER SUBSTITUTION for edge relations.
#     Inside ATT_Head and MSG_Head, `pe(s, t; w)` uses the template
#     arg as a relation name. After materialization with pe='PaperAuthor',
#     it becomes `PaperAuthor(s, t; w)`. This already works.
#
#  5. BOUNDING FILTERS BY ALREADY-BOUND VARIABLES.
#     In H<'Author', L>, the bounding `MetaRel(ts, pe, 'Author')` only
#     iterates rows where the 3rd column = 'Author'. Free variables
#     ts and pe take each matching row's values.
#
#  6. HEAD CONCAT VIA CONDITION-ONLY BOUNDING.
#     `Concat(*z)` with `Join(Set(WMsg<i>(...) | 1 <= i, i <= h))`
#     generalizes over any number of heads at compile time.
#
#  7. MetaRel COMES FROM THE DATABASE (no grammar change needed).
#     Instead of adding compile-time fact assertions to the grammar,
#     MetaRel is loaded as a regular DB relation by the dataset loader
#     (e.g. load_dblp_dataset adds a MetaRel DataFrame).  Bounding
#     expansion in the engine looks it up via self.db at compile time.
#
# ═══════════════════════════════════════════════════════════════════
#  FEATURES NEEDED (in implementation order)
# ═══════════════════════════════════════════════════════════════════
#
#  Feature 1: RECURSION BASE CASES (engine-only)          [DONE]
#    - Store multiple definitions per template name in symbol table
#    - Dispatch: concrete values match before variables
#    - Recursion depth limit as safety net
#    - Enables: Cora multi-layer HGT (no bounding needed)
#
#  Feature 2: FACT ASSERTIONS — NOT NEEDED
#    - Originally planned as grammar-level fact assertions.
#    - Solved instead by loading MetaRel as a regular DB relation
#      (see observation 7).  No grammar change required.
#
#  Feature 3: BOUNDING (parser + engine)                  [DONE]
#    - Grammar: Set keyword in bounded_rhs rule
#    - Parser: bounded_rule, bounded_rhs, bounding transformers
#    - Engine: _expand_bounded_set (compile-time expansion)
#    - Enables: DBLP full heterogeneous HGT
# ═══════════════════════════════════════════════════════════════════
