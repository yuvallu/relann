# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations
# %%
import networkx as nx
from collections import deque
from typing import Optional, Any, Dict, Iterable, Callable, Tuple, Iterator, Generator
from pydantic import BaseModel
from relann.pydantic_classes import *
from relann.term_graph import TermGraph, make_func_call_full_name
from fastcore.basics import patch
import sys
import types
import torch
import torch.nn as nn
import builtins
from copy import copy, deepcopy
from importlib import import_module
from contextlib import contextmanager
import re
import pandas as pd
import logging

from relann.relnn import term_graph_to_module, RelNNModuleLookupError
from relann.era_operations import ExecutionContext
from relann.torch_utils import full_seed, print_model_params
from relann.arith_eval import evaluate_arith_term

logger = logging.getLogger(__name__)

# %% [markdown]
# # Engine

# %%
class Engine:
    """
    Core stateful engine for RelNN: manages symbol table, relations, term graph, and parameters.
    Inspired by spannerlib.Engine .
    """

    def __init__(self, db=None, seed: Optional[int] = None, debug: bool = False, device=None):
        # Optional explicit seed (e.g. Session(seed=123)). Default 42 is set on session import.
        if seed is not None:
            full_seed(seed)

        # Store the database. RelationSource entries are kept lazy — materialisation
        # happens on first reference via ``_normalize_relation_payload``, so a
        # ``Session(db={"BigTable": SqlSource(...)})`` doesn't pull rows over the
        # wire until a rule actually references "BigTable". Once materialised, the
        # ER-dict replaces the source entry in-place for subsequent lookups.
        # Legacy ``(df, tensor)`` tuples are left as-is and normalised on demand.
        # Shallow-copy so external mutation of the caller's dict doesn't leak in
        # (the previous eager-init code path also produced an isolated dict).
        self.db = dict(db) if db is not None else {}
        self.debug = debug

        # Target device for all db embedding tensors (None = leave as-is, default CPU behavior).
        self.device = torch.device(device) if device is not None else None
        if self.device is not None:
            self._move_db_to_device()
        
        # Symbol table with hierarchical scopes: {scope_name: {var_name: (type, value)}}
        self.symbol_table = {
            "global": {
                # "hgt": Er_function(name, ...)  # Example: a pydantic EmbeddedRelation function object
            }
        }

        # Term graphs: dict mapping names to computational graphs (TermGraph)
        self.term_graphs = {
            "global": TermGraph(db=self.db, debug=self.debug)  # tg with "symbolic" leaves
        }  # {name: TermGraph()}
        
        # Initialize namespace stack with global scope
        self.namespace_stack = ["global"]
        
        # Q: Do I need these: all_functions=None, all_function_objects=None ?
        
        # Persistent parameter store: maps fully-qualified names to nn.Parameter
        # tensors.  Used to save trained weights after fit() and reload them in
        # predict().  This is NOT nn.Module._parameters — Engine is not a Module.
        self.parameter_store: Dict[str, nn.Parameter] = {}
        
        # Trained modules: stores trained RelNN modules after fit, keyed by rule name
        self.trained_modules = {}    # {rule_name: {"module": module, "loss_history": [...], "fit_config": {...}}}
        # Run-scope globals (set by Session.define so op resolution sees caller's Linear, ReLU, etc.)
        self._run_globals: Optional[Dict[str, Any]] = None

        # Last executed module and its grounded term graph (for session.relation() inspection)
        self._last_module: Optional[Any] = None
        self._last_module_tg: Optional[Any] = None
        
        # Template instance cache: maps "Name<arg1,arg2>" -> (kind, materialized_obj)
        # Used for weight sharing across identical template instantiations.
        self._template_instance_cache: Dict[str, Tuple[str, Any]] = {}

        # Template specializations: maps name -> list of (pattern, entity_type, entity_obj).
        # Enables multiple definitions of the same template name (e.g. H<0> and H<L>)
        # with dispatch to the most-specific match at materialization time.
        # NOTE: global-scoped (not per-namespace).  Templated rules inside function
        # bodies (e.g. WMsg<i> inside EdgeAgg) ARE registered here and may be
        # overwritten when the next function is processed.  This is safe because
        # function bodies are processed sequentially, and the in_function guard
        # in _materialize_template_reference forces re-materialization (no caching)
        # for rules inside functions.
        self._template_specializations: Dict[str, list] = {}

        # Recursion depth counter for template materialization (prevents infinite loops).
        self._template_materialization_depth: int = 0

    def _move_db_to_device(self):
        """Move all embedding tensors in self.db to self.device (called once at init)."""
        for key, val in self.db.items():
            if isinstance(val, tuple) and len(val) == 2:
                df, tensor = val
                if isinstance(tensor, torch.Tensor):
                    self.db[key] = (df, tensor.to(self.device))

    def set_run_globals(self, globals_dict: Dict[str, Any]) -> None:
        """Set the run-scope globals used for op resolution (e.g. Linear, ReLU from caller)."""
        self._run_globals = globals_dict

    def get_run_globals(self) -> Dict[str, Any]:
        """Return run-scope globals for op resolution; empty dict if not set."""
        return self._run_globals if self._run_globals is not None else {}

    def evaluate_arith_term_for_hyperparams(self, term: ArithTerm) -> Any:
        """Evaluate an ArithTerm to a Python value (for compiler hyperparams). Uses symbol table for Vars."""
        return self._evaluate_arith_term(term)

    def get_symbol(self, symbol_name: str, namespace: str = None) -> Optional[tuple]:
        """
        Look up a symbol in the current namespace, falling back to global scope if not found.
        
        Args:
            symbol_name: The name of the symbol to look up
            namespace: The namespace to check first (defaults to current namespace)
            
        Returns:
            Tuple of (type, value) if found, None otherwise
        """
        # If no namespace provided, use current namespace
        if namespace is None:
            namespace = self.current_namespace
            
        # First check in provided/current namespace
        if namespace in self.symbol_table and symbol_name in self.symbol_table[namespace]:
            return self.symbol_table[namespace][symbol_name]
            
        # Then check global scope
        if symbol_name in self.symbol_table["global"]:
            return self.symbol_table["global"][symbol_name]
            
        return None
        
    def enter_namespace(self, namespace: str):
        self.namespace_stack.append(namespace)       
    def exit_namespace(self):
        self.namespace_stack.pop()
    @property        
    def current_namespace(self):
        return self.namespace_stack[-1]
    
    # TODO: move to DataLoader from db class interface
    def _load_er_from_db(self, name: str):
        pass
    def _write_er_to_db(self, name: str, er_obj):
        pass
    def _list_ers_in_db(self):
        # If db is a dict or list, interpret keys as ER names
        if isinstance(self.db, dict):
            return list(self.db.keys())
        elif isinstance(self.db, list):
            return self.db
        return []

    def _validate_rule(self, rule: Rule):
        # TODO: remove comment
        # self._check_rhs_ers_exist(rule)
        pass

    def _check_rhs_ers_exist(self, rule: Rule):
        # Build set of ER names known to the symbol table (current + global)
        known_ers = set()
        # Add all names in current namespace
        if self.current_namespace in self.symbol_table:
            known_ers.update(self.symbol_table[self.current_namespace].keys())
        # Add all names in global namespace
        if "global" in self.symbol_table:
            known_ers.update(self.symbol_table["global"].keys())
        # Add ER names in database
        db_ers = set(self._list_ers_in_db())
        # Check each EmbeddedRelation in RHS
        missing = []
        for er in getattr(rule.rhs, "ers", []):
            er_name = getattr(er, "name", None)
            if er_name and er_name not in known_ers and er_name not in db_ers:
                missing.append(er_name)
        if missing:
            raise ValueError(f"RHS contains undefined ERs not found in symbol table or database: {missing}")
    def _validate_transform(self, transform_def: TransformDef):
        pass
    
    def _validate_function(self, function_def: FunctionDef):
        pass

# %% [markdown]
# ## add rule

# %%
@patch
def add_rule(self: Engine, rule: Rule):
    self._validate_rule(rule)
    
    # Set default Parameters:
    # Set default aggregation function if not specified.
    # Global-reduction rules (e.g. loss rules with no group-by columns) default
    # to 'mean' so that per-sample losses are averaged; all others default to 'sum'.
    if rule.lhs.embedding_expression.aggregation_fn is None:
        if not rule.lhs.group_by_column_names:
            rule.lhs.embedding_expression.aggregation_fn = 'mean'
        else:
            rule.lhs.embedding_expression.aggregation_fn = 'sum'
    
    # if all template_params in Rule.lhs are defined:
    #     nx_rule = translate_rule_to_nx(pydantic)
    #     add nx_rule to self.term_graph  # TODO: how - implement subclass of nx with the name TermGraph.
    # else:
    #     pass  # do nothing

    # add rule.lhs.name to symbol_table (store value)
    # self.symbol_table[scope=current_namespace][rule_name] = (ER, rule: Rule)
    
    if rule.lhs.template_params:
        # Templated Rule: store in symbol table but skip term graph addition.
        # Materialization happens when referenced with concrete template args in a RHS.
        self.symbol_table[self.current_namespace][rule.lhs.name] = (Rule, rule)
        self._register_template_specialization(rule.lhs.name, Rule, rule)
        return
    
    # Expand bounded sets before term-graph construction.
    if isinstance(rule.rhs, BoundedRHS):
        expanded_rhs, var_renames = self._expand_bounded_set(rule.rhs)
        updates: dict = {"rhs": expanded_rhs}
        if var_renames:
            updates["lhs"] = _expand_splats_in_lhs(rule.lhs, var_renames)
        rule = rule.model_copy(update=updates)

    # Add rule to the term graph using TermGraph's add_rule method
    current_tg = self.term_graphs[self.current_namespace]
    
    # For each embedded relation in the RHS, handle function calls and template references.
    for er in rule.rhs.ers:
        if getattr(er, "template_args", None) is not None:
            self._materialize_template_reference(er, current_tg)
        elif getattr(er, "arguments", None) is not None:
            self._materialize_function_call(er, current_tg)
    
    # Note: Decided NOT to perform materialization of tensor_term using transform definitions here.
    # Rationale: If we want weight sharing between different uses of the same transform,
    # it's best to postpone this materialization until the end (during fit or predict).
    # If we instead materialize at the time we define the TransformDef, we could give better error messages,
    # but it complicates the termgraph a little. Therefore, materialization is deferred to fit/predict time,
    # where we can set up modules and sharing as needed.
    
    current_tg.add_rule(rule)

    # Add rule.lhs.name to symbol_table (store value)
    self.symbol_table[self.current_namespace][rule.lhs.name] = (DerivedER, rule.lhs)

# %%
if __name__ == "__main__":
    from relann.parser import parse_and_transform_str

# %%
if __name__ == "__main__":
    simple_rule_program_str = '''SimpleEmbedding(X, Y, Z; Linear(3,4)(Concat(z1, z2))) :- InputData1(X, Y;z1), InputData2(Y, Z;z2) .'''
    simple_rule_program_pydantic = parse_and_transform_str(simple_rule_program_str)

# %%
if __name__ == "__main__":
    # Test the EngineWithImplementation class
    print("Testing Engine add_rule...")

    # Create an engine instance
    engine = Engine()

# %%
if __name__ == "__main__":
    rule = simple_rule_program_pydantic.statements[0]
    print(f"Adding rule: {rule.lhs.name}")
    try:
        engine.add_rule(rule)
        print(f"✓ Successfully added rule '{rule.lhs.name}'")
        print(f"  Symbol table now has: {list(engine.symbol_table['global'].keys())}")
        print(f"  Term graph has {engine.term_graphs['global'].number_of_nodes()} nodes")
        print(f"  Term graph nodes: {list(engine.term_graphs['global'].nodes())}")
    except Exception as e:
        assert False, f"✗ Error adding rule: {e}"

# %%
if __name__ == "__main__":
    from relann.term_graph import preety_draw_tg, draw_tg

# %%
if __name__ == "__main__":
    #| eval: false
    preety_draw_tg(engine.term_graphs['global'])
    draw_tg(engine.term_graphs['global'])

# %%
@patch
def add_program(self: Engine, program: Program):
    """
    Add a Program (set of top-level statements) to the engine. 
    Each statement is added to the current namespace's term_graph.
    """
    for stmt in program.statements:
        if isinstance(stmt, Rule):
            self.add_rule(stmt)
        elif isinstance(stmt, TransformDef):
            self.add_transform(stmt)
        elif isinstance(stmt, FunctionDef):
            self.add_function(stmt)
        elif isinstance(stmt, FitStatement):
            self.fit(stmt)
        elif isinstance(stmt, PredictStatement):
            return self.predict(stmt)

# %% [markdown]
# ## test "only rules" program

# %%
if __name__ == "__main__":
    # Test add_program with a program consisting of 2 rules

    # Create the engine
    engine = Engine()

    # Construct a program containing 2 rules using parse_and_transform_str
    prog_3rules_str = """
    SimpleEmbedding(X, Z; Linear(3,4)(Concat(z1, z2))) :- InputData1(X, Y;z1), InputData2(Y, Z;z2) .
    OtherEmbedding(X, W; Linear(5,2)(z3)) :- InputData3(X, W;z3) .
    FromSimpleEmbedding(X; Linear(3,4)(Concat(z1, z2))) :- InputData4(X;z1), SimpleEmbedding(X, Z;z2), X < 10 .
    """
    prog_only_rules = parse_and_transform_str(prog_3rules_str)

    # Add the program to the engine using add_program
    engine.add_program(prog_only_rules)

    # Both rules should be present in the current TermGraph
    tg = engine.term_graphs[engine.current_namespace]

# %%
if __name__ == "__main__":
    prog_only_rules.statements[0]

# %%
if __name__ == "__main__":
    tg.symbol_to_node

# %%
if __name__ == "__main__":
    from relann.term_graph import preety_draw_tg, draw_tg

# %%
if __name__ == "__main__":
    #| eval: false
    preety_draw_tg(tg)
    draw_tg(tg)

# %% [markdown]
# ## Materizalize

# %%
# (deque, Iterator, Generator already imported in cell 2)

# %%
def traverse_tensor_term_bfs(
    term: TensorTerm,
    include_self: bool = True
) -> Generator[TensorTerm, None, None]:
    """
    Breadth-first search traversal (level by level).
    
    Args:
        term: The root TensorTerm to traverse
        include_self: If True, yield the root node; if False, only yield descendants
    
    Yields:
        TensorTerm nodes in breadth-first order (level by level)
    
    Example:
        >>> # For tree: op(+) -> [op(*) -> [value(2), value(3)], value(4)]
        >>> # Yields: +, *, 4, 2, 3
    """
    queue = deque([term] if include_self else [])
    if not include_self and term.sons:
        queue.extend(term.sons)
    
    while queue:
        current = queue.popleft()
        yield current
        
        if current.sons:
            queue.extend(current.sons)

# %%
class Concat(nn.Module):
    """A PyTorch module that concatenates inputs along a specified dimension."""
    def __init__(self):
        super().__init__()
    
    def forward(self, *x):
        # Always concatenate along dim=1 for column-wise concat
        return torch.cat(x, dim=1)

@patch
def _rewrite_param_path_for_fqn(self: Engine, module: nn.Module, full_param_name: str) -> str:
    """Rewrite ``children_modules.<i>`` to a logical name (e.g. ``K``, ``Q``).

    Works on top-level wrappers (``_MultiArgWrapper``) whose ``_child_names``
    list maps each numeric child index to a user-visible name.  When names are
    not unique (e.g. ``["Linear", "Linear"]``), numeric indices are kept so
    FQNs stay distinct.

    Uses a split-on-dot approach so that only exact path segments are matched
    (avoids false substring hits like ``children_modules.1`` inside
    ``children_modules.10``).
    """
    child_names = getattr(module, "_child_names", None)
    if not child_names or not full_param_name.startswith("children_modules."):
        return full_param_name

    non_none = [n for n in child_names if n is not None]
    all_unique = len(non_none) == len(set(non_none))
    index_to_name = {
        str(i): (child_names[i] if child_names[i] is not None and all_unique else str(i))
        for i in range(len(child_names))
    }

    segments = full_param_name.split(".")
    # segments[0] is always "children_modules"; segments[1] is the numeric index.
    if len(segments) >= 2 and segments[0] == "children_modules" and segments[1] in index_to_name:
        segments = [index_to_name[segments[1]]] + segments[2:]
    return ".".join(segments)


@patch
def _resolve_transform_def_name_from_node(self: Engine, node_data: dict) -> Optional[str]:
    """Return the TransformDef name recorded on a term-graph node, or None.

    Checks the cached ``transform_def_name`` key first (set by
    ``replace_all_vars_in_tg_using_symbol_table``), then falls back to
    inspecting the node's ``transformation`` TensorTerm.
    """
    name = node_data.get("transform_def_name")
    if name is not None:
        return name
    tterm = node_data.get("transformation")
    if (
        isinstance(tterm, TensorTerm)
        and isinstance(tterm.op, TensorOp)
        and isinstance(tterm.op.op, str)
    ):
        sym = self.get_symbol(tterm.op.op)
        if sym:
            typ, _ = sym
            if typ == TransformDef:
                return tterm.op.op
    return None


@patch
def _resolve_param_source_node(
    self: Engine,
    node_id: str,
    node_attrs: dict,
) -> tuple:
    """Parameter binding: when a transformation node is tagged with
    ``_param_source_node`` pointing at an original Apply whose Linear was
    copied, this method resolves the lookup so the new node's FQN finds the
    original Linear's weights in parameter_store rather than fresh random
    init. The annotation is currently unused on juplit (the e-graph optimizer
    that produced it was removed); kept for forward compatibility with PR #56.

    Returns (effective_node_id, effective_attrs) for FQN derivation. When no
    annotation is present, returns the inputs unchanged. Recursively resolves
    chains of synthesis (synth -> synth -> origin); termination guaranteed
    since each hop must consume an annotation and we only annotate when
    copying from a non-annotated origin.
    """
    source = node_attrs.get("_param_source_node")
    if source is None:
        return node_id, node_attrs
    # Find the source node in any namespace's term_graph.
    for tg in self.term_graphs.values():
        if source in tg.nodes:
            src_attrs = tg.nodes[source]
            # Recurse — handle synthesis chains.
            return self._resolve_param_source_node(source, src_attrs)
    # Source not found in any tg (shouldn't happen for well-formed graphs);
    # fall back to current node.
    return node_id, node_attrs


@patch
def _build_param_fqn(
    self: Engine,
    module: nn.Module,
    raw_param_name: str,
    namespace: str,
    node_id: str,
    transform_def_name: Optional[str],
    transform_def_child_names: Optional[list],
    transform_def_path_map: Optional[Dict[str, str]] = None,
) -> str:
    """Build the canonical FQN for one parameter.

    Single source of truth used by both ``_extract_and_store_parameters``
    (store path) and ``_load_saved_parameters`` (load path), guaranteeing
    they always produce the same key for the same parameter.

    Shared-child detection uses two module-level annotations set by
    ``eval_tensor_terms_on_tg``:

    * ``_child_names`` (multi-child): first path segment matches a
      TransformDef name in *transform_def_child_names*.
    * ``_transform_def_children`` (single-child): maps the wrapper's
      internal attribute name to the TransformDef name, avoiding any
      dependency on compiler constants.
    * ``transform_def_path_map`` (recursive): maps compiler paths like
      ``left.left`` to TransformDef names like ``K<1>`` for compound
      arithmetic expressions (e.g. ``K<1>(z1) * Q<1>(z2) * Mu<1>``).
    """
    logical_name = self._rewrite_param_path_for_fqn(module, raw_param_name)
    if "." in logical_name:
        module_path, param_name = logical_name.rsplit(".", 1)
    else:
        module_path, param_name = "", logical_name

    effective_transform_def_name = transform_def_name
    effective_module_path = module_path

    # Try recursive path map first (handles nested arithmetic with TransformDefs)
    if transform_def_path_map and module_path:
        best_prefix = ""
        best_td_name = None
        for map_prefix, td_name in transform_def_path_map.items():
            if module_path == map_prefix or module_path.startswith(map_prefix + "."):
                if len(map_prefix) > len(best_prefix):
                    best_prefix = map_prefix
                    best_td_name = td_name
        if best_td_name is not None:
            effective_transform_def_name = best_td_name
            remainder = module_path[len(best_prefix):]
            effective_module_path = remainder.lstrip(".")

    if not (transform_def_path_map and effective_transform_def_name != transform_def_name):
        if transform_def_child_names and module_path:
            first_seg = module_path.split(".", 1)[0]
            if first_seg in transform_def_child_names:
                effective_transform_def_name = first_seg
                effective_module_path = ""
            else:
                td_children = getattr(module, "_transform_def_children", None)
                if td_children and first_seg in td_children:
                    effective_transform_def_name = td_children[first_seg]
                    effective_module_path = ""

    # Strip compiler wrapper segments (inner, _module) when a TransformDef is
    # resolved, so the FQN is stable regardless of surrounding wrapper structure
    # (e.g. Classifier inside CrossEntropyLoss vs inside ArgMax vs standalone).
    if effective_transform_def_name and effective_module_path:
        segs = [s for s in effective_module_path.split(".") if s not in ("inner", "_module")]
        effective_module_path = ".".join(segs)

    base_name = effective_transform_def_name if effective_transform_def_name else node_id
    parts = [namespace, base_name]
    if effective_module_path:
        parts.append(effective_module_path)
    parts.append(param_name)
    return ".".join(parts)


@patch
def _extract_and_store_parameters(
    self: Engine,
    module: nn.Module,
    namespace: str,
    node_id: str,
    transform_def_name: Optional[str] = None,
    transform_def_child_names: Optional[list] = None,
    overwrite: bool = False,
    transform_def_path_map: Optional[Dict[str, str]] = None,
) -> None:
    """Extract parameters from *module* and store them in ``Engine.parameter_store``.

    When *overwrite* is True, existing entries are updated in-place (used
    after training to persist trained values).  Otherwise a parameter is only
    stored if its FQN is not already present (prevents clobbering shared
    weights that were registered from a previous rule).
    """
    for raw_param_name, param in module.named_parameters():
        fqn = self._build_param_fqn(
            module=module,
            raw_param_name=raw_param_name,
            namespace=namespace,
            node_id=node_id,
            transform_def_name=transform_def_name,
            transform_def_child_names=transform_def_child_names,
            transform_def_path_map=transform_def_path_map,
        )
        if fqn not in self.parameter_store:
            self.parameter_store[fqn] = param
        elif overwrite:
            self.parameter_store[fqn].data.copy_(param.data)


@patch
def load_into(self: Engine, module: nn.Module, node_id: str, node: dict) -> None:
    """ParameterLoader protocol — load saved parameters into *module*.

    Uses the same FQN construction logic as ``_extract_and_store_parameters``
    so store and load always agree on keys.
    """
    if not self.parameter_store:
        return

    # Dispatch via _param_source_node so transformation nodes that were
    # tagged with `_param_source_node` (e.g. by an upstream rewrite) inherit
    # the original Apply's parameter FQN. No tagger currently emits this
    # annotation on juplit (the e-graph optimizer that did was removed in
    # 5656608); the dispatch is forward-compat with PR #56.
    eff_node_id, eff_attrs = self._resolve_param_source_node(node_id, node)

    namespace = self.current_namespace
    transform_def_name = self._resolve_transform_def_name_from_node(eff_attrs)
    transform_def_child_names = eff_attrs.get("transform_def_child_names")
    transform_def_path_map = eff_attrs.get("transform_def_path_map")

    for raw_param_name, param in module.named_parameters():
        fqn = self._build_param_fqn(
            module=module,
            raw_param_name=raw_param_name,
            namespace=namespace,
            node_id=eff_node_id,
            transform_def_name=transform_def_name,
            transform_def_child_names=transform_def_child_names,
            transform_def_path_map=transform_def_path_map,
        )
        if fqn in self.parameter_store:
            param.data.copy_(self.parameter_store[fqn].data)

# %%
@patch
def tensor_term_to_module(self: Engine, tterm: TensorTerm, var_to_input_index: Optional[Dict[str, int]] = None) -> nn.Module:
    """
    Convert a TensorTerm tree into a PyTorch nn.Module.

    The module's forward method accepts *inputs and processes them according to
    the TensorTerm tree structure. Leaf nodes with Var values (e.g., "z1", "z2")
    select inputs by index based on the var_to_input_index mapping, which maps
    Var names to their position in the RHS ER list.

    This method uses the Engine's symbol table to resolve named TransformDefs
    (e.g., K_Linear1_Authors) to their underlying tensor terms.

    Args:
        tterm: The TensorTerm to convert
        var_to_input_index: Optional mapping from Var name to input index (0-based).
                          If None or Var not found, falls back to numeric suffix parsing
                          for backward compatibility.

    Returns:
        A PyTorch nn.Module that implements the computation described by the TensorTerm

    Example:
        >>> # Linear(3, 4)(Concat(z1, z2))
        >>> # Creates a module that concatenates inputs[0] and inputs[1], then applies Linear
        >>> module = engine.tensor_term_to_module(linear_term, var_to_input_index={'z1': 0, 'z2': 1})
        >>> result = module(tensor1, tensor2)
    """
    from relann.tensor_term_compiler import TensorTermCompiler
    return TensorTermCompiler(self).compile(tterm, var_to_input_index=var_to_input_index)

# %%
if __name__ == "__main__":
    #| eval: false
    preety_draw_tg(tg)
    draw_tg(tg)

# %%
if __name__ == "__main__":
    from relann.pydantic_classes import TensorTerm, TensorOp, ArithTerm, Var

# %% [markdown]
# test tensor_term_to_module simple tensor arith op - "*"

# %%
if __name__ == "__main__":
    # Test tensor_term_to_module with multiplication operation (z1 * z2)
    # Build a TensorTerm representing: z1 * z2
    mult_term = TensorTerm(
        op=TensorOp(op="*", hyper_params=None),
        sons=[
            TensorTerm(op=None, sons=None, value=Var(name='z1')),
            TensorTerm(op=None, sons=None, value=Var(name='z2')),
        ]
    )

    # Convert to module
    mult_fn = engine.tensor_term_to_module(mult_term, var_to_input_index={'z1': 0, 'z2': 1})

    # Test with two input tensors
    x1 = torch.tensor([[2.0, 3.0, 4.0]])  # Shape: (1, 3)
    x2 = torch.tensor([[1.0, 2.0, 3.0]])  # Shape: (1, 3)

    if isinstance(mult_fn, nn.Module):
        # Test forward pass
        out = mult_fn(x1, x2)
        print("Test Multiplication (z1 * z2):")
        print(f"  Input x1: {x1}")
        print(f"  Input x2: {x2}")
        print(f"  Output: {out}")
        print(f"  Output shape: {out.shape}")
        
        # Verify element-wise multiplication: [2*1, 3*2, 4*3] = [2, 6, 12]
        expected = torch.tensor([[2.0, 6.0, 12.0]])
        assert torch.allclose(out, expected), f"Expected {expected}, got {out}"
        print("✓ Test passed! Module correctly performs element-wise multiplication")
        
        # Test that gradients flow correctly
        x1_grad = torch.tensor([[2.0, 3.0, 4.0]], requires_grad=True)
        x2_grad = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        out_grad = mult_fn(x1_grad, x2_grad)
        loss = out_grad.sum()
        loss.backward()
        
        print(f"  Gradient test:")
        print(f"    x1.grad: {x1_grad.grad}")
        print(f"    x2.grad: {x2_grad.grad}")
        assert x1_grad.grad is not None, "Gradients should flow to x1"
        assert x2_grad.grad is not None, "Gradients should flow to x2"
        print("✓ Gradient flow test passed! Backpropagation works correctly")
    else:
        raise TypeError(f"tensor_term_to_module failed: expected nn.Module, got {type(mult_fn)}")

# %% [markdown]
# test tensor_term_to_module nested ops (Linear on Concat)

# %%
@patch
def _normalize_fit_params(self: Engine, params: Dict[str, ArithTerm]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for name, term in params.items():
        normalized[name.lower()] = self._evaluate_arith_term(term)
    return normalized


def _resolve_var_for_arith(engine: Engine, var: Var) -> Any:
    """Resolve a Var to a value for arithmetic evaluation (symbol table, external scope, TransformDef)."""
    sym = engine.get_symbol(var.name)
    if sym is None:
        resolved = engine._resolve_external_symbol(var.name)
        if resolved is None:
            raise KeyError(
                f"Symbol '{var.name}' referenced in fit params is undefined "
                "and cannot be resolved from Python scope or torch.nn"
            )
        return resolved
    _, sym_value = sym
    if isinstance(sym_value, TransformDef):
        tensor_term = sym_value.tensor_term
        if tensor_term.op is None and tensor_term.value is not None:
            if isinstance(tensor_term.value, (int, float, str, bool)):
                return tensor_term.value
        op_obj = getattr(tensor_term, "op", None)
        op_name = getattr(op_obj, "op", None)
        hp = getattr(op_obj, "hyper_params", None)
        if op_name in {"+", "-", "*", "/", "//", "**"} and hp and len(hp) >= 2:
            return evaluate_arith_term(ArithTerm(op=op_name, sons=hp), lambda v: _resolve_var_for_arith(engine, v))
        from relann.tensor_term_compiler import tensor_term_to_arith_term
        try:
            return evaluate_arith_term(tensor_term_to_arith_term(tensor_term), lambda v: _resolve_var_for_arith(engine, v))
        except ValueError:
            pass
    return sym_value


@patch
def _evaluate_arith_term(self: Engine, term: ArithTerm) -> Any:
    """Evaluate ArithTerm using symbol-table resolver (for compile-time hyperparams)."""
    return evaluate_arith_term(term, lambda v: _resolve_var_for_arith(self, v))

# %%
if __name__ == "__main__":
    # A direct test of tensor_term_to_module for "Linear(Concat(...))" as in cell 16 in @012_relnn.ipynb.
    # Build a TensorTerm representing: Linear(6,4)(Concat(z1, z2))
    # Note: Concatenating two 3D inputs gives 6D, so Linear needs in_features=6
    concat_term = TensorTerm(
        op=TensorOp(op="Concat", hyper_params=None),
        sons=[
            TensorTerm(op=None, sons=None, value=Var(name='z1')),
            TensorTerm(op=None, sons=None, value=Var(name='z2')),
        ]
    )
    linear_term = TensorTerm(
        op=TensorOp(op="Linear", hyper_params=[
            ArithTerm(op=None, sons=None, value=6),  # 3 + 3 = 6 after concat
            ArithTerm(op=None, sons=None, value=4)
        ]),
        sons=[concat_term]
    )

    engine = Engine()
    # NOTE: `var_to_input_index` binds Var('z1') and Var('z2') to input slots
    # 0 and 1 of the resulting nn.Module. Without it, the compiler would try
    # to resolve them as external symbols (Python globals / torch.nn) and fail.
    fn = engine.tensor_term_to_module(linear_term, var_to_input_index={'z1': 0, 'z2': 1})

# %%
if __name__ == "__main__":
    import torch

# %%
if __name__ == "__main__":
    fn

# %%
if __name__ == "__main__":
    x1 = torch.randn(1, 3)  # Add batch dimension: (batch_size, features)
    x2 = torch.randn(1, 3)
    if isinstance(fn, nn.Module):
        # fn is now a proper nn.Module, call it directly
        out = fn(x1, x2)
        print("Test Linear(Concat): Output shape", out.shape)
        assert out.shape == (1, 4), f"Output should be shape (1, 4), got {out.shape}"
        print("✓ Test passed! Module correctly concatenates inputs and applies Linear transformation")
    else:
        print(f"tensor_term_to_module failed: expected nn.Module, got {type(fn)}")

# %%
@patch
def eval_tensor_terms_on_tg(self: Engine, tg: TermGraph):
    """Compile tensor terms on transformation nodes into torch modules.

    Invoked exactly once per fit/predict before module construction. Compiles
    every transformation's DSL TensorTerm against its `var_to_input_index`,
    stamps the result into `torch_transformation`, and persists params via the
    engine's parameter store (`_extract_and_store_parameters`). No partial-recompile
    path.
    """
    # Resolve TransformDef references (e.g. K_Linear1_Author) to their tensor term bodies
    # so tensor_term_to_module (via TensorTermCompiler) sees only primitive ops (Linear, @, etc.).
    self.replace_all_vars_in_tg_using_symbol_table(tg, in_place=True)

    namespace = self.current_namespace

    for node, attrs in tg.nodes(data=True):
        if attrs.get('type') == 'transformation' and 'transformation' in attrs:
            tterm = attrs['transformation']
            # Get var_to_input_index mapping from node attributes
            var_to_input_index = attrs.get('var_to_input_index', {})
            torch_module = self.tensor_term_to_module(tterm, var_to_input_index=var_to_input_index)
            tg.nodes[node]['torch_transformation'] = torch_module
            # Propagate TransformDef child names so FQN/display show K, Q instead of 0, 1.
            transform_def_child_names = attrs.get("transform_def_child_names")
            if isinstance(transform_def_child_names, (list, tuple)):
                if (
                    hasattr(torch_module, "children_modules")
                    and torch_module.children_modules is not None
                    and len(transform_def_child_names) == len(torch_module.children_modules)
                ):
                    # Multi-child wrapper (e.g. Concat(K, Q))
                    torch_module._child_names = list(transform_def_child_names)
                elif len(transform_def_child_names) == 1:
                    # Single-child wrapper (e.g. Linear(K(z))): map the wrapper's
                    # child attr to the TransformDef name so _build_param_fqn can
                    # resolve it without knowing compiler internals.
                    from relann.tensor_term_compiler import SINGLE_CHILD_ATTR
                    torch_module._transform_def_children = {
                        SINGLE_CHILD_ATTR: transform_def_child_names[0]
                    }

            # Dispatch via _param_source_node so any transformation node tagged
            # with that annotation writes its params under the source Apply's
            # FQN. Forward-compat hook with PR #56 (no juplit code emits the
            # tag today).
            eff_node_id, eff_attrs = self._resolve_param_source_node(node, attrs)
            self._extract_and_store_parameters(
                module=torch_module,
                namespace=namespace,
                node_id=eff_node_id,
                transform_def_name=self._resolve_transform_def_name_from_node(eff_attrs),
                transform_def_child_names=eff_attrs.get("transform_def_child_names")
                    or transform_def_child_names,
                transform_def_path_map=eff_attrs.get("transform_def_path_map"),
            )
    return tg

# %%
if __name__ == "__main__":
    prog_3rules_str = """
    SimpleEmbedding(X, Z; Linear(3,4)(Concat(z1, z2))) :- InputData1(X, Y;z1), InputData2(Y, Z;z2) .
    OtherEmbedding(X, W; Linear(5,2)(z3)) :- InputData3(X, W;z3) .
    FromSimpleEmbedding(X; Linear(3,4)(Concat(z1, z2))) :- InputData4(X;z1), SimpleEmbedding(X, Z;z2) .
    """

# %%
if __name__ == "__main__":
    # Test add_program with a program consisting of 1 rule

    # Create the engine
    engine = Engine()

    # Keep only first line in program string and rename variable to prog_one_rule_str and prog_one_rule
    prog_one_rule_str = """
    SimpleEmbedding(X, Y; Linear(3,4)(Concat(z1, z2))) :- InputData1(X, Y;z1) .
    """
    prog_one_rule_str = """
    SimpleEmbedding(X; Linear(3,4)(Concat(z1, z2))) :- InputData1(X, Y;z1) .
    FromSimpleEmbedding(X; Linear(3,4)(Concat(z1, z2))) :- InputData4(X;z1), SimpleEmbedding(X;z2) .
    """
    prog_one_rule = parse_and_transform_str(prog_one_rule_str)

    # Add the program to the engine using add_program
    engine.add_program(prog_one_rule)

    # Resulting TermGraph for this one-rule program
    tg = engine.term_graphs[engine.current_namespace]

# %%
if __name__ == "__main__":
    #| eval: false
    draw_tg(tg)

# %% [markdown]
# ## add transform

# %%
@patch
def add_transform(self: Engine, transform_def: TransformDef):
    self._validate_transform(transform_def)
    
    # Pseudocode for adding a TransformDef to the engine:
    # 1. if not all template_params in Rule.lhs are defined:
    #        pass  # do nothing
    # 2. Add transform_def.name to symbol_table in the current namespace, storing (type, transform_def)
    
    if transform_def.template_params:
        # Templated TransformDef: store in symbol table but skip processing.
        # Materialization happens when referenced with concrete template args.
        self.symbol_table[self.current_namespace][transform_def.name] = (
            TransformDef, transform_def
        )
        self._register_template_specialization(transform_def.name, TransformDef, transform_def)
        return
    
    # Add to symbol table - store both the transform definition and its tensor term
    self.symbol_table[self.current_namespace][transform_def.name] = (
        TransformDef, transform_def
    )

# %%
if __name__ == "__main__":
    from pprint import pprint

    # Test add_transform with a basic example using d = 64 and Lin = Linear(16, 32)
    engine = Engine()

    # Set up variables and a transform definition
    prog_transform_str = """
    d = 64 .
    Lin = Linear(16,32) .

    Embed64(X; Linear(d, d)(z1)) :- InputData(X;z1) .
    TestUseLin(X; Lin(z2)) :- InputData2(X;z2) .
    """
    prog_with_transforms = parse_and_transform_str(prog_transform_str)
    engine.add_program(prog_with_transforms)

    # Check symbol table entries for the transforms
    print("Engine symbol table in current namespace:", engine.symbol_table[engine.current_namespace])

    print(engine.symbol_table['global'].keys())
    pprint(engine.symbol_table['global'])

# %%
if __name__ == "__main__":
    # Test that the values of 'd' and 'Lin' are as expected in the symbol table and print them
    d_obj = engine.symbol_table['global']['d'][1]
    if hasattr(d_obj, "tensor_term") and hasattr(d_obj.tensor_term, "value"):
        d_val = d_obj.tensor_term.value
    elif hasattr(d_obj, "value"):
        d_val = d_obj.value
    else:
        d_val = d_obj
    print("d =", d_val)
    assert d_val == 64, f"Expected d == 64, got {d_val}"

    lin_obj = engine.symbol_table['global']['Lin'][1]
    print("Lin =", lin_obj)

    # Short, robust check: get tensor_term if exists, else use lin_obj directly
    tt = lin_obj.tensor_term if hasattr(lin_obj, "tensor_term") and lin_obj.tensor_term is not None else lin_obj
    op = getattr(tt.op, "op", tt.op) if hasattr(tt, "op") else None
    assert op == "Linear", f"Expected Lin.op == 'Linear', got {op}"
    hyper_params = getattr(tt.op, "hyper_params", getattr(tt, "hyper_params", []))
    hyper_param_vals = [hp.value if hasattr(hp, "value") else hp for hp in hyper_params] if hyper_params else [s.value for s in getattr(tt, "sons", []) if hasattr(s, "value")]
    assert hyper_param_vals == [16, 32], f"Expected Lin.hyper_params == [16, 32], got {hyper_param_vals}"

    draw_tg(engine.term_graphs['global'])

# %% [markdown]
# ## add Function 

# %%
if __name__ == "__main__":
    from relann.term_graph import make_func_call_full_name

# %%
@patch
def _materialize_function_call(self: Engine, er: EmbeddedRelation, current_tg: TermGraph) -> str:
    """
    Materialize a function call by copying nodes from the function's term graph,
    using unique (qualified) names for all nodes within this invocation
    and replacing placeholder nodes with actual argument nodes.

    Args:
        er: The EmbeddedRelation representing the function call
        current_tg: The current term graph to merge into

    Returns:
        The node name that represents the function's output
    """
    # 1. Look up the function definition
    func_symbol = self.get_symbol(er.name)
    if func_symbol is None:
        raise ValueError(f"Function '{er.name}' not found in symbol table")

    func_type, function_def = func_symbol
    from relann.pydantic_classes import FunctionDef
    if func_type != FunctionDef:
        raise ValueError(f"'{er.name}' is not a function definition")

    # 2. Validate arguments
    n_args = len(er.arguments) if er.arguments else 0
    if n_args != len(function_def.er_params):
        raise ValueError(
            f"Function '{er.name}' expects {len(function_def.er_params)} arguments, "
            f"but got {n_args}"
        )

    # 3. Get the function's term graph
    if function_def.name not in self.term_graphs:
        raise ValueError(f"Function '{function_def.name}' has no term graph")

    func_tg = self.term_graphs[function_def.name]

    # 4. Map function parameters to argument nodes
    param_to_arg_node = {}
    for arg_er_ref, param in zip(er.arguments or [], function_def.er_params):
        # Find the argument node in the current term graph
        arg_node = current_tg.get_node_by_symbol(arg_er_ref.name, namespace=self.current_namespace)

        # Check if it's a data loader in the database
        if arg_node is None:
            if arg_er_ref.name in self.db:
                # It's a data loader - ensure it exists
                if arg_er_ref.name not in current_tg.nodes():
                    from relann.pydantic_classes import EmbeddedRelation
                    data_loader_er = EmbeddedRelation(
                        name=arg_er_ref.name,
                        content_attrs=[],
                        embedding_var=None
                    )
                    current_tg._add_data_loader_if_needed(data_loader_er)
                arg_node = arg_er_ref.name
            else:
                raise ValueError(
                    f"Argument '{arg_er_ref.name}' for function '{er.name}' not found "
                    f"in current term graph or database"
                )

        param_to_arg_node[param.name] = arg_node

    # Helpers: function call prefix for qualified names
    func_call_prefix = make_func_call_full_name(er)  # e.g. MyEmbedFunc(InputData1)
    def qualify_node_name(node_name):
        # Only qualify if the node isn't a placeholder. We want: MyEmbedFunc(InputData1).node_name
        return f"{func_call_prefix}.{node_name}"

    # 5. Build qualified node-name map for this invocation
    node_name_map = {}  # unqualified_name -> qualified_name
    _reused_parent_nodes: set = set()
    for node_name, node_data in func_tg.nodes(data=True):
        if node_data.get('type') == 'place_holder':
            continue
        # If a function-local data loader corresponds to a derived ER that
        # already exists in the parent TG, reuse the parent node directly.
        # This avoids creating duplicate weight copies for globally-defined
        # rules (e.g. H<Author,0>) referenced inside nested functions.
        if node_data.get('type') == 'data_loader':
            original_name = node_data.get("name", node_name)
            parent_node = current_tg.get_node_by_symbol(original_name)
            if parent_node is not None:
                parent_type = current_tg.nodes[parent_node].get('type', '')
                if parent_type != 'data_loader':
                    node_name_map[node_name] = parent_node
                    _reused_parent_nodes.add(node_name)
                    continue
        node_name_map[node_name] = qualify_node_name(node_name)

    def _remap_func_node_ref(ref_name: str) -> str:
        """
        Remap a function-local node reference to the invocation graph:
        - placeholders -> call argument nodes
        - regular function nodes -> qualified invocation-local names
        """
        ref_data = func_tg.nodes.get(ref_name)
        if isinstance(ref_data, dict) and ref_data.get('type') == 'place_holder':
            param_name = ref_data.get('er_param').name
            return param_to_arg_node[param_name]
        return node_name_map.get(ref_name, ref_name)

    # 6. Copy all non-placeholder nodes; remap local references inside node attrs
    for node_name, node_data in func_tg.nodes(data=True):
        if node_data.get('type') == 'place_holder':
            continue
        if node_name in _reused_parent_nodes:
            continue
        qualified_node_name = node_name_map[node_name]
        node_data_copy = deepcopy(dict(node_data))
        if "input_order" in node_data_copy and isinstance(node_data_copy["input_order"], list):
            node_data_copy["input_order"] = [_remap_func_node_ref(ref) for ref in node_data_copy["input_order"]]
        current_tg.add_node(qualified_node_name, **node_data_copy)

    # 7. Copy all edges (excluding placeholder nodes), remapping sources/targets
    for source, target, edge_data in func_tg.edges(data=True):
        source_data = func_tg.nodes[source]
        target_data = func_tg.nodes[target]

        # Map source
        if source_data.get('type') == 'place_holder':
            param_name = source_data.get('er_param').name
            if param_name in param_to_arg_node:
                src_qualified = param_to_arg_node[param_name]
            else:
                continue  # skip if placeholder has no mapping
        else:
            src_qualified = node_name_map[source]

        # Map target
        if target_data.get('type') == 'place_holder':
            param_name = target_data.get('er_param').name
            if param_name in param_to_arg_node:
                tgt_qualified = param_to_arg_node[param_name]
            else:
                continue
        else:
            tgt_qualified = node_name_map[target]

        if not current_tg.has_edge(src_qualified, tgt_qualified):
            current_tg.add_edge(src_qualified, tgt_qualified, **copy.deepcopy(edge_data) if edge_data else {})

    # 8. Find the function's output node (last rule's output)
    output_node = None
    if func_tg.symbol_to_node:
        rule_outputs = list(func_tg.symbol_to_node.values())
        if rule_outputs:
            last_output = rule_outputs[-1]
            # The output node gets a qualified name now:
            final_output_node = qualify_node_name(last_output)
            if final_output_node in current_tg.nodes():
                output_node = final_output_node

    if output_node is None:
        raise ValueError(f"Could not determine output node for function '{er.name}'")

    # 9. Create alias for function call name to output node
    current_tg.add_symbol_alias(func_call_prefix, output_node)

    return output_node

# %%
# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def _template_instance_key(name: str, concrete_args: list) -> str:
    """Build a cache key like 'Lin<3,5>' from a base name and evaluated args.

    String values are normalized by stripping surrounding quotes so that
    DSL literals (``'Paper'``) and bounding-derived values (``Paper``)
    produce the same cache key, enabling weight sharing.
    """
    args_str = ",".join(str(_strip_quotes(a)) for a in concrete_args)
    return f"{name}<{args_str}>"


def _evaluate_template_arg(engine: "Engine", arg: ArithTerm):
    """Evaluate a template argument.

    Scalars (``d = 64``) are resolved via ``_evaluate_arith_term``.
    If that raises ``KeyError`` and the arg is a bare ``Var``, we treat the
    name itself as the concrete value (ER references like ``Edges``).
    Quoted strings are preserved as-is (quotes serve as bound-markers
    during bounded set expansion).
    """
    if arg.op is None and isinstance(arg.value, Var) and arg.sons is None:
        try:
            return engine._evaluate_arith_term(arg)
        except (KeyError, TypeError):
            return arg.value.name
    return engine._evaluate_arith_term(arg)


def _build_template_substitution(
    template_params: list, template_args: list, engine: "Engine",
    *, name: str = ""
) -> Dict[str, Any]:
    """Build ``{param_name: concrete_value}`` from params + args."""
    if len(template_params) != len(template_args):
        param_names = [p.name if isinstance(p, Var) else str(p) for p in template_params]
        label = f" '{name}'" if name else ""
        raise ValueError(
            f"Template{label} expects {len(template_params)} params "
            f"({', '.join(param_names)}) but got {len(template_args)} args"
        )
    sub: Dict[str, Any] = {}
    for param, arg in zip(template_params, template_args):
        if isinstance(param, Var):
            sub[param.name] = _evaluate_template_arg(engine, arg)
        # Concrete params (from specialization base cases) are already
        # baked into the definition and don't need substitution.
    return sub


def materialize_pydantic(obj, substitution: Dict[str, Any]):
    """Deep-copy a Pydantic object, replacing template parameter names with
    concrete values according to *substitution*.

    Handles Var.name, EmbeddedRelation.name (ER references),
    ArithTerm.value that is a Var, and recursive structures.
    """
    if obj is None:
        return None

    if isinstance(obj, list):
        return [materialize_pydantic(item, substitution) for item in obj]

    if isinstance(obj, (int, float, str, bool)):
        return obj

    if isinstance(obj, Var):
        if obj.name in substitution:
            val = substitution[obj.name]
            if isinstance(val, (int, float)):
                return val
            if isinstance(val, str):
                return Var(name=val)
            return val
        return obj.model_copy(deep=True)

    if not isinstance(obj, BaseModel):
        return obj

    kwargs: Dict[str, Any] = {}
    for field_name in obj.__class__.model_fields:
        val = getattr(obj, field_name)
        if val is None:
            kwargs[field_name] = None
            continue

        if field_name == "name" and isinstance(val, str) and val in substitution:
            kwargs[field_name] = str(substitution[val])
            continue

        if isinstance(val, (int, float, str, bool)):
            kwargs[field_name] = val
        elif isinstance(val, Var):
            kwargs[field_name] = materialize_pydantic(val, substitution)
        elif isinstance(val, list):
            kwargs[field_name] = materialize_pydantic(val, substitution)
        elif isinstance(val, BaseModel):
            kwargs[field_name] = materialize_pydantic(val, substitution)
        else:
            kwargs[field_name] = val

    return obj.__class__(**kwargs)


# ---------------------------------------------------------------------------
# Template specialization helpers (recursion base cases)
# ---------------------------------------------------------------------------

MAX_TEMPLATE_RECURSION_DEPTH = 50


def _strip_quotes(val):
    """Strip surrounding single or double quotes from a string value."""
    if isinstance(val, str):
        if (val.startswith("'") and val.endswith("'")) or \
           (val.startswith('"') and val.endswith('"')):
            return val[1:-1]
    return val


def _extract_specialization_pattern(template_params: list) -> list:
    """Convert template_params to a pattern for dispatch.

    Each element is either a concrete value (int, float, str, bool)
    or None (wildcard, from a Var parameter).
    String values are normalized by stripping surrounding quotes.

    Examples:
        [Var('L')]        -> [None]
        [0]               -> [0]
        [Var('T'), 0]     -> [None, 0]
        ['Paper']         -> ['Paper']  (quotes stripped)
    """
    pattern = []
    for p in template_params:
        if isinstance(p, Var):
            pattern.append(None)
        elif hasattr(p, "value"):
            pattern.append(_strip_quotes(p.value))
        else:
            pattern.append(_strip_quotes(p))
    return pattern


def _pattern_matches(pattern: list, concrete_args: list) -> bool:
    """Check whether *pattern* matches *concrete_args*.

    Concrete positions must match exactly (after quote normalization);
    None positions match anything.  Arity must match.
    """
    if len(pattern) != len(concrete_args):
        return False
    for p_val, c_val in zip(pattern, concrete_args):
        if p_val is not None and _strip_quotes(p_val) != _strip_quotes(c_val):
            return False
    return True


def _pattern_specificity(pattern: list) -> int:
    """Return the number of concrete (non-None) positions in a pattern.

    Higher specificity = more specific match (preferred in dispatch).
    """
    return sum(1 for p in pattern if p is not None)


def _bounding_value_to_arith(val):
    """Wrap a bounding substitution value as an ArithTerm."""
    if isinstance(val, (int, float)):
        return ArithTerm(op=None, sons=None, value=val)
    return ArithTerm(op=None, sons=None, value=Var(name=str(val)))


def _apply_bounding_sub(template_args, sub):
    """Replace free-variable ArithTerm/Var entries in *template_args* with concrete values from *sub*."""
    if not template_args:
        return
    for j, arg in enumerate(template_args):
        # ArithTerm wrapping a bare Var
        if isinstance(arg, ArithTerm) and arg.op is None and isinstance(arg.value, Var) and arg.sons is None:
            if arg.value.name in sub:
                template_args[j] = _bounding_value_to_arith(sub[arg.value.name])
        elif isinstance(arg, Var) and arg.name in sub:
            template_args[j] = _bounding_value_to_arith(sub[arg.name])


def _expand_er_with_sub(main_er, sub: dict):
    """Deepcopy *main_er* and apply bounding substitution *sub* to its template args."""
    concrete_er = deepcopy(main_er)
    _apply_bounding_sub(concrete_er.template_args, sub)
    if concrete_er.arguments:
        for er_ref in concrete_er.arguments:
            _apply_bounding_sub(er_ref.template_args, sub)
    return concrete_er


@patch
def _expand_bounded_set(
    self: Engine, bounded_rhs: BoundedRHS
) -> Tuple[RHS, Optional[Dict[str, list]]]:
    """Expand a bounded set into a regular RHS at compile time.

    Supports two modes:

    **Guard-relation bounding** (e.g. ``MetaRel(ts, pe, tt)``):
    look up the relation in ``self.db``, classify attrs as bound/free,
    filter rows, substitute free vars into template_args.

    **Condition-only bounding** (e.g. ``1 <= i, i <= h``):
    infer integer range from comparison conditions and substitute.

    For ``Join`` bounded sets the embedding var is renamed per expansion
    (``z`` → ``z_1, z_2, ...``) and a ``var_renames`` dict is returned.
    For ``Union`` bounded sets ``var_renames`` is ``None``.

    Returns:
        ``(rhs, var_renames)`` where *var_renames* maps base var names
        to lists of renamed vars (only for Join), or ``None``.
    """
    main_er = bounded_rhs.main_er

    if bounded_rhs.bounding_ers:
        expanded_ers = self._expand_bounded_set_guard(bounded_rhs)
    elif bounded_rhs.bounding_conditions:
        expanded_ers = self._expand_bounded_set_conditions(bounded_rhs)
    else:
        raise ValueError("Bounded set requires either bounding relations or conditions")

    # For Join bounded sets, rename embedding vars so each ER gets a unique one.
    var_renames: Optional[Dict[str, list]] = None
    if bounded_rhs.rel_op_name == "Join" and main_er.embedding_var is not None:
        base_name = (main_er.embedding_var.name
                     if isinstance(main_er.embedding_var, Var)
                     else str(main_er.embedding_var))
        renamed_vars = []
        for idx, er in enumerate(expanded_ers, 1):
            new_name = f"{base_name}_{idx}"
            er.embedding_var = Var(name=new_name)
            renamed_vars.append(new_name)
        var_renames = {base_name: renamed_vars}
        logger.debug("Join var renames: %s", var_renames)

    rel_op_str = "|" if bounded_rhs.rel_op_name == "Union" else ","
    rel_ops = [rel_op_str] * (len(expanded_ers) - 1) if len(expanded_ers) > 1 else None
    rhs = RHS(ers=expanded_ers, rel_ops=rel_ops, filter_expressions=[])
    return rhs, var_renames


@patch
def _expand_bounded_set_guard(self: Engine, bounded_rhs: BoundedRHS) -> list:
    """Expand using a guard relation looked up in self.db."""
    main_er = bounded_rhs.main_er
    bounding_er = bounded_rhs.bounding_ers[0]
    bounding_name = bounding_er.name

    if bounding_name not in self.db:
        raise ValueError(
            f"Bounding relation '{bounding_name}' not found in database. "
            f"Available relations: {list(self.db.keys())}"
        )

    df, _ = self.db[bounding_name]

    bound_positions = []
    free_positions = []
    for i, attr in enumerate(bounding_er.content_attrs):
        if isinstance(attr, Var):
            stripped = _strip_quotes(attr.name)
            if stripped != attr.name:
                bound_positions.append((i, stripped))
            else:
                free_positions.append((i, attr.name))
        else:
            bound_positions.append((i, _strip_quotes(attr)))

    matching = df
    for col_idx, value in bound_positions:
        col_name = df.columns[col_idx]
        matching = matching[matching[col_name] == value]

    if matching.empty:
        raise ValueError(
            f"No rows in '{bounding_name}' match the bound values "
            f"{[(df.columns[i], v) for i, v in bound_positions]}. "
            f"Available rows:\n{df.to_string()}"
        )

    expanded_ers = []
    for _, row in matching.iterrows():
        sub = {var_name: row.iloc[col_idx] for col_idx, var_name in free_positions}
        expanded_ers.append(_expand_er_with_sub(main_er, sub))

    logger.debug("Guard-bounded set expanded '%s' → %d ERs from '%s'",
                 main_er.name, len(expanded_ers), bounding_name)
    return expanded_ers


@patch
def _expand_bounded_set_conditions(self: Engine, bounded_rhs: BoundedRHS) -> list:
    """Expand using condition-only bounding (e.g. ``1 <= i, i <= h``)."""
    main_er = bounded_rhs.main_er
    ranges = _infer_range_from_conditions(bounded_rhs.bounding_conditions, self)

    if not ranges:
        raise ValueError(
            "Could not infer any variable ranges from bounding conditions: "
            + ", ".join(str(c) for c in bounded_rhs.bounding_conditions)
        )

    if len(ranges) > 1:
        raise ValueError(
            "Condition-only bounding currently supports a single variable, "
            f"got {list(ranges.keys())}"
        )

    var_name, (lo, hi) = next(iter(ranges.items()))
    values = list(range(lo, hi + 1))
    logger.debug("Condition-only bounding: %s in %s", var_name, values)

    expanded_ers = [_expand_er_with_sub(main_er, {var_name: val}) for val in values]
    return expanded_ers


def _infer_range_from_conditions(
    conditions: list, engine: "Engine"
) -> Dict[str, Tuple[int, int]]:
    """Extract integer ranges from simple comparison conditions.

    Supports forms like ``1 <= i``, ``i <= h``, ``i >= 1``, ``h >= i``
    where one side is a bare Var and the other resolves to an int.

    Returns ``{var_name: (lower_inclusive, upper_inclusive)}``.
    """
    lowers: Dict[str, int] = {}
    uppers: Dict[str, int] = {}

    for cond in conditions:
        lhs_var, lhs_val = _resolve_cond_side(cond.lhs, engine)
        rhs_var, rhs_val = _resolve_cond_side(cond.rhs, engine)
        op = cond.comp_op

        if lhs_var and rhs_val is not None:
            # var OP value, e.g. i <= h
            _record_bound(lhs_var, int(rhs_val), op, lowers, uppers)
        elif rhs_var and lhs_val is not None:
            # value OP var, e.g. 1 <= i  →  flip to i >= 1
            flipped = {"<=": ">=", ">=": "<=", "<": ">", ">": "<", "==": "=="}
            _record_bound(rhs_var, int(lhs_val), flipped.get(op, op), lowers, uppers)
        else:
            logger.warning("Cannot interpret bounding condition: %s %s %s",
                           cond.lhs, op, cond.rhs)

    result = {}
    for var in set(lowers) | set(uppers):
        if var not in lowers or var not in uppers:
            raise ValueError(
                f"Condition-only bounding requires both lower and upper bounds "
                f"for variable '{var}'. Got: lower={'yes' if var in lowers else 'MISSING'}, "
                f"upper={'yes' if var in uppers else 'MISSING'}"
            )
        result[var] = (lowers[var], uppers[var])
    return result


def _resolve_cond_side(
    term: ArithTerm, engine: "Engine"
) -> Tuple[Optional[str], Optional[Any]]:
    """Classify one side of a comparison as a variable name or a resolved value.

    Returns ``(var_name, None)`` for a bare variable that is NOT a known scalar,
    or ``(None, resolved_value)`` for a literal/scalar that resolves to a number.
    """
    if term.op is None and isinstance(term.value, Var) and term.sons is None:
        try:
            val = engine._evaluate_arith_term(term)
            if isinstance(val, (int, float)):
                return (None, val)
        except (KeyError, TypeError):
            pass
        return (term.value.name, None)
    try:
        val = engine._evaluate_arith_term(term)
        if isinstance(val, (int, float)):
            return (None, val)
    except (KeyError, TypeError, ValueError):
        pass
    return (None, None)


def _record_bound(var: str, val: int, op: str,
                  lowers: Dict[str, int], uppers: Dict[str, int]):
    """Record a bound for *var* given ``var OP val``."""
    if op in ("<=", "=="):
        uppers[var] = min(uppers.get(var, val), val)
        if op == "==":
            lowers[var] = max(lowers.get(var, val), val)
    elif op == "<":
        uppers[var] = min(uppers.get(var, val - 1), val - 1)
    elif op == ">=":
        lowers[var] = max(lowers.get(var, val), val)
    elif op == ">":
        lowers[var] = max(lowers.get(var, val + 1), val + 1)


def _expand_splats_in_lhs(lhs: DerivedER, var_renames: Dict[str, list]) -> DerivedER:
    """Replace ``Concat(*z)`` splat nodes with expanded var references.

    Walks the LHS embedding expression tree.  When a ``TensorOp(op="splat")``
    is found whose child variable is in *var_renames*, replaces the single
    splat node with N leaf ``TensorTerm(value=Var(name))`` nodes.
    """
    ee = lhs.embedding_expression
    if ee is None or ee.tensor_term is None:
        return lhs

    lhs = deepcopy(lhs)
    _replace_splats_recursive(lhs.embedding_expression.tensor_term, var_renames)
    return lhs


def _replace_splats_recursive(node: TensorTerm, var_renames: Dict[str, list]):
    """In-place recursive replacement of splat children."""
    if node.sons is None:
        return
    new_sons = []
    for son in node.sons:
        if (son.op is not None and son.op.op == "splat"
                and son.sons and len(son.sons) == 1):
            inner = son.sons[0]
            base_name = None
            if isinstance(getattr(inner, "value", None), Var):
                base_name = inner.value.name
            if base_name and base_name in var_renames:
                for renamed in var_renames[base_name]:
                    new_sons.append(TensorTerm(value=Var(name=renamed)))
                continue
        _replace_splats_recursive(son, var_renames)
        new_sons.append(son)
    node.sons = new_sons


@patch
def _register_template_specialization(
    self: Engine, name: str, entity_type, entity_obj
):
    """Register a templated definition for dispatch.

    Extracts the specialization pattern from template_params and stores
    it alongside the entity. Rule and FunctionDef specializations may
    coexist under the same name (e.g. base case Rules + recursive FunctionDef).
    """
    if entity_type == Rule:
        tparams = entity_obj.lhs.template_params
    else:
        tparams = getattr(entity_obj, "template_params", None)

    if not tparams:
        return

    pattern = _extract_specialization_pattern(tparams)

    if name not in self._template_specializations:
        self._template_specializations[name] = []

    specs = self._template_specializations[name]

    # Avoid duplicate patterns
    for existing_pattern, _, _ in specs:
        if existing_pattern == pattern:
            logger.debug(
                "Replacing existing specialization for '%s' with pattern %s", name, pattern
            )
            specs[:] = [(p, t, o) for p, t, o in specs if p != pattern]
            break

    specs.append((pattern, entity_type, entity_obj))


@patch
def _resolve_template_definition(
    self: Engine, name: str, concrete_args: list
) -> Tuple[Any, Any]:
    """Dispatch to the most-specific specialization matching *concrete_args*.

    Returns (entity_type, entity_obj) for the best match.
    Raises ValueError if no definition matches.
    """
    specs = self._template_specializations.get(name)
    if not specs:
        sym = self.get_symbol(name)
        if sym is not None:
            return sym
        raise ValueError(
            f"No template definitions found for '{name}'"
        )

    matches = [
        (pattern, typ, obj)
        for pattern, typ, obj in specs
        if _pattern_matches(pattern, concrete_args)
    ]

    if not matches:
        args_str = ", ".join(str(a) for a in concrete_args)
        available = [str(p) for p, _, _ in specs]
        raise ValueError(
            f"No specialization of '{name}' matches args ({args_str}). "
            f"Available patterns: {available}"
        )

    matches.sort(key=lambda m: _pattern_specificity(m[0]), reverse=True)
    if len(matches) > 1 and _pattern_specificity(matches[0][0]) == _pattern_specificity(matches[1][0]):
        args_str = ", ".join(str(a) for a in concrete_args)
        raise ValueError(
            f"Ambiguous dispatch for '{name}' with args ({args_str}): "
            f"patterns {matches[0][0]} and {matches[1][0]} have equal specificity"
        )
    _, best_type, best_obj = matches[0]
    return best_type, best_obj


@patch
def _get_or_materialize_transform(
    self: Engine, name: str, obj, template_args: list
) -> tuple:
    """Return ``(cache_key, materialized_TransformDef)`` for a templated TransformDef.

    On cache miss, materializes and caches.  On hit, returns the cached object.
    When multiple specializations exist, dispatches to the best match.
    """
    concrete_args = [_evaluate_template_arg(self, a) for a in template_args]
    cache_key = _template_instance_key(name, concrete_args)
    if cache_key not in self._template_instance_cache:
        # If specializations are registered, dispatch to best match
        if name in self._template_specializations:
            _, obj = self._resolve_template_definition(name, concrete_args)
        sub_dict = _build_template_substitution(
            obj.template_params, template_args, self, name=name
        )
        materialized = materialize_pydantic(obj, sub_dict)
        materialized.name = cache_key
        self._template_instance_cache[cache_key] = ("transform", materialized)
    _, td = self._template_instance_cache[cache_key]
    return cache_key, td


@patch
def _materialize_template_reference(
    self: Engine, er: EmbeddedRelation, current_tg: TermGraph
):
    """Handle an EmbeddedRelation in the RHS that has ``template_args``.

    Looks up the templated Rule or FunctionDef, materializes it with
    concrete args, and processes it (``add_rule`` / ``add_function`` +
    ``_materialize_function_call``).

    Supports multiple definitions (specializations) of the same template
    name.  Dispatches to the most-specific match via
    ``_resolve_template_definition``.
    """
    self._template_materialization_depth += 1
    try:
        if self._template_materialization_depth > MAX_TEMPLATE_RECURSION_DEPTH:
            raise RecursionError(
                f"Template recursion depth exceeded {MAX_TEMPLATE_RECURSION_DEPTH} "
                f"while materializing '{er.name}'. Missing base case?"
            )

        concrete_args = [_evaluate_template_arg(self, a) for a in er.template_args]
        cache_key = _template_instance_key(er.name, concrete_args)

        # Dispatch to best matching specialization (or fall back to get_symbol)
        typ, obj = self._resolve_template_definition(er.name, concrete_args)

        if typ == Rule:
            tparams = obj.lhs.template_params
        else:
            tparams = getattr(obj, "template_params", None)

        if not tparams:
            raise ValueError(f"'{er.name}' is not a templated definition")

        in_function = self.current_namespace != "global"
        # A template is "local" if it was defined in the current function scope.
        # Local templates (e.g. WMsg<i> inside EdgeAgg) must be re-materialized
        # for each function invocation.  Global templates (e.g. H<Author,0>)
        # should be materialized once into the global TG for weight sharing.
        is_local_template = (
            in_function and typ == Rule
            and er.name in self.symbol_table.get(self.current_namespace, {})
        )

        need_materialize = (
            cache_key not in self._template_instance_cache
            or is_local_template
        )

        if need_materialize:
            sub_dict = _build_template_substitution(tparams, er.template_args, self, name=er.name)
            materialized = materialize_pydantic(obj, sub_dict)
            if typ == Rule:
                materialized.lhs.template_params = None
                materialized.lhs.name = cache_key
                if not is_local_template:
                    self._template_instance_cache[cache_key] = ("rule", materialized)
            elif typ == FunctionDef:
                materialized.template_params = None
                materialized.name = cache_key
                self._template_instance_cache[cache_key] = ("function", materialized)
            else:
                raise ValueError(f"Cannot materialize template reference of type {typ}")
        else:
            # Cache hit — retrieve previously materialized object
            _, materialized = self._template_instance_cache[cache_key]

        # Preserve the original templated definition in the symbol table —
        # add_rule / add_function will overwrite the entry under er.name with
        # the materialized (non-templated) version.
        original_er_name = er.name
        original_sym = self.symbol_table[self.current_namespace].get(original_er_name)

        if typ == Rule:
            if in_function and not is_local_template:
                # Global rule referenced inside a function: add to the global
                # TG (once) so weights are shared.  The function's TG will see
                # it as a data-loader source referencing the global result.
                global_tg = self.term_graphs["global"]
                if global_tg.get_node_by_symbol(cache_key) is None:
                    rule_to_add = self._template_instance_cache[cache_key][1]
                    saved_ns = list(self.namespace_stack)
                    self.namespace_stack = ["global"]
                    try:
                        self.add_rule(deepcopy(rule_to_add))
                    finally:
                        self.namespace_stack = saved_ns
            else:
                rule_to_add = materialized if is_local_template else self._template_instance_cache[cache_key][1]
                self.add_rule(deepcopy(rule_to_add))
            er.name = cache_key
            er.template_args = None
        elif typ == FunctionDef:
            _, cached_fn = self._template_instance_cache[cache_key]
            self.add_function(deepcopy(cached_fn))
            # Rewrite the ER so downstream term-graph code uses the
            # materialized function name (avoids name collisions in
            # recursive templates).
            er.name = cache_key
            er.template_args = None

        if original_sym is not None:
            self.symbol_table[self.current_namespace][original_er_name] = original_sym
        else:
            self.symbol_table.get(self.current_namespace, {}).pop(original_er_name, None)

        # For FunctionDef, always call _materialize_function_call so the function's
        # output nodes are wired into the *current* rule's term graph.
        if typ == FunctionDef:
            if er.arguments is None:
                er.arguments = []
            self._materialize_function_call(er, current_tg)

    finally:
        self._template_materialization_depth -= 1

# %%
if __name__ == "__main__":
    from pprint import pprint

# %%
@patch
def add_function(self: Engine, function_def: FunctionDef):
    self._validate_function(function_def)
    # Pseudocode for adding a FunctionDef to the engine:
    # 1. Enter the function's namespace (push function_def.name onto namespace stack)
    # 2. For each statement in function_def.function_body:
    #     For each statement that is a TransformDef or Rule:
    #         - Call the appropriate add method (self.add_transform or self.add_rule)
    #         - Add to term_graph as needed
    # 3. Add function_def.name to symbol_table in the current namespace, storing (type, function_def)
    #    self.symbol_table[self.current_namespace][function_def.name] = ("function", function_def)
    # 4. Exit the function's namespace (pop from namespace stack)
    # 5. Q: Do i do this? or only when calling the function?
    #       If all template_params are defined, translate function_def to nx and add to term_graph
    
    if function_def.template_params:
        # Templated FunctionDef: store in symbol table but skip body processing.
        # Materialization happens when called with concrete template args.
        self.symbol_table[self.current_namespace][function_def.name] = (FunctionDef, function_def)
        self._register_template_specialization(function_def.name, FunctionDef, function_def)
        return
    
    # Ensure symbol_table has the function's namespace key
    if function_def.name not in self.symbol_table:
        self.symbol_table[function_def.name] = {}

    # Enter function namespace
    self.enter_namespace(function_def.name)
    
        # Create a new TermGraph for this function's namespace
    self.term_graphs[function_def.name] = TermGraph(db=self.db, debug=self.debug)
    current_tg = self.term_graphs[function_def.name]
    
    # For each er_param in function_def.er_params, add a place holder node to the function's term graph. 
    # It will be replaced by the actual node when the function is called.
    for er_param in function_def.er_params:
        current_tg.add_place_holder_node(er_param)
    
    # Process each statement in the function body
    for stmt in function_def.function_body:
        if isinstance(stmt, Rule):
            # Reuse Engine.add_rule so function-call RHS terms are materialized consistently.
            self.add_rule(stmt)
        elif isinstance(stmt, TransformDef):
            self.add_transform(stmt)
        # You could extend here for more statement types if desired

    # Exit the function's namespace
    self.exit_namespace()
    
    # Add function_def.name to symbol_table in the global namespace
    self.symbol_table[self.current_namespace][function_def.name] = (FunctionDef, function_def)

# %%
if __name__ == "__main__":
    from pprint import pprint
    # Inline test of add_transform with a function in the DSL program string
    engine = Engine()

    prog_transform_str = """
    def MyEmbedFunc(A):
        d = 64 .
        Embed64(a; Linear(d, d)(z1)) :- A(a;z1) .
        TestUseLin(a; Linear(16,32)(z2)) :- Embed64(a;z2) .
    enddef
    """

    prog_with_transforms = parse_and_transform_str(prog_transform_str)
    engine.add_program(prog_with_transforms)

    pprint(list(engine.symbol_table.keys()))
    pprint(engine.symbol_table['global'])
    pprint(engine.symbol_table['MyEmbedFunc'])

# %%
if __name__ == "__main__":
    draw_tg(engine.term_graphs['global'])
    print(engine.term_graphs['MyEmbedFunc'].symbol_to_node)
    draw_tg(engine.term_graphs['MyEmbedFunc'])

# %%
if __name__ == "__main__":

    # function call example
    engine = Engine(db={"InputData1":None, "InputData2":None})

    prog_transform_str = """
    def MyEmbedFunc(A):
        d = 64 .
        Embed64(a; Linear(d, d)(z1)) :- A(a;z1) .
        TestUseLin(a; Linear(16,32)(z2)) :- Embed64(a;z2) .
    enddef
    Rule1(X,Y; Linear(z1)) :- MyEmbedFunc(InputData1)(X,Y;z1), MyEmbedFunc(InputData2)(X;z2) .
    # Rule2(X,Y; Linear(z1)) :- InputData2(X,Y;z1), MyEmbedFunc(Rule1)(X;z2) .
    """

    prog_with_transforms = parse_and_transform_str(prog_transform_str)
    engine.add_program(prog_with_transforms)

# %%
if __name__ == "__main__":
    draw_tg(engine.term_graphs['MyEmbedFunc'])

# %%
if __name__ == "__main__":
    pprint(engine.symbol_table['global'])

# %%
if __name__ == "__main__":
    pprint(engine.term_graphs['global'].symbol_to_node)
    draw_tg(engine.term_graphs['global'])

# %%
if __name__ == "__main__":
    engine.term_graphs['global'].nodes()

# %%
if __name__ == "__main__":
    engine.term_graphs['global'].edges()

# %% [markdown]
# ## Fit and Predict

# %%
@patch
def _build_transform_def_path_map(
    self: Engine,
    term: 'TensorTerm',
    _prefix: str = "",
) -> Dict[str, str]:
    """Recursively walk a *pre-inlined* TensorTerm tree and return a map
    from compiler module paths to TransformDef names.

    Mirrors the compiler's wrapper structure:
      - arithmetic/equality ops: sons[0] → "left", sons[1] → "right"
      - other ops (non-leaf): treated as a single unit for the node
    Only records entries where the sub-tree root is a TransformDef reference
    (either a TensorOp call like ``K<1>(z1)`` or a VarTemplated leaf like ``Mu<1>``).
    """
    from relann.pydantic_classes import TransformDef, TensorOp, VarTemplated

    result: Dict[str, str] = {}

    def _resolve_td_name(t) -> Optional[str]:
        """If *t* is a TransformDef reference, return its (possibly templated) display name."""
        if isinstance(getattr(t, "op", None), TensorOp) and isinstance(t.op.op, str):
            sym = self.get_symbol(t.op.op)
            if sym and sym[0] == TransformDef:
                obj = sym[1]
                if getattr(obj, "template_params", None) and getattr(t.op, "template_args", None):
                    key, _ = self._get_or_materialize_transform(
                        t.op.op, obj, t.op.template_args
                    )
                    return key
                return t.op.op
        if isinstance(getattr(t, "value", None), VarTemplated):
            vt = t.value
            sym = self.get_symbol(vt.name)
            if sym and sym[0] == TransformDef:
                obj = sym[1]
                if getattr(obj, "template_params", None) and vt.template_params:
                    template_args = [
                        ArithTerm(value=p) if not isinstance(p, ArithTerm) else p
                        for p in vt.template_params
                    ]
                    key, _ = self._get_or_materialize_transform(vt.name, obj, template_args)
                    return key
                return vt.name
        if isinstance(getattr(t, "value", None), Var):
            sym = self.get_symbol(t.value.name)
            if sym and sym[0] == TransformDef:
                return t.value.name
        return None

    ARITHMETIC_OPS = {"*", "+", "-", "/", "@", "**", "=="}

    def _walk(t, prefix: str):
        if t is None:
            return
        op = getattr(t, "op", None)
        op_name = getattr(op, "op", None) if isinstance(op, TensorOp) else None

        if op_name in ARITHMETIC_OPS:
            sons = t.sons or []
            if len(sons) >= 1:
                td_name = _resolve_td_name(sons[0])
                child_prefix = f"{prefix}.left" if prefix else "left"
                if td_name:
                    result[child_prefix] = td_name
                else:
                    _walk(sons[0], child_prefix)
            if len(sons) >= 2:
                td_name = _resolve_td_name(sons[1])
                child_prefix = f"{prefix}.right" if prefix else "right"
                if td_name:
                    result[child_prefix] = td_name
                else:
                    _walk(sons[1], child_prefix)
        else:
            # Non-arithmetic ops (e.g. CrossEntropyLoss, Concat): the compiler
            # wraps children with _MultiArgWrapper using children_modules.<i>,
            # or _SingleChildWrapper using SINGLE_CHILD_ATTR ("input").
            sons = t.sons or []
            if len(sons) == 1:
                td_name = _resolve_td_name(sons[0])
                child_prefix = f"{prefix}.input" if prefix else "input"
                if td_name:
                    result[child_prefix] = td_name
                else:
                    _walk(sons[0], child_prefix)
            elif len(sons) >= 2:
                for i, son in enumerate(sons):
                    td_name = _resolve_td_name(son)
                    child_prefix = f"{prefix}.children_modules.{i}" if prefix else f"children_modules.{i}"
                    if td_name:
                        result[child_prefix] = td_name
                    else:
                        _walk(son, child_prefix)

    _walk(term, _prefix)
    return result


def _apply_call_argument(
    subst: TensorTerm,
    arg0: TensorTerm,
    engine: "Engine",
    replace_fn,
    call_sons: list,
) -> TensorTerm:
    """Apply the resolved call argument(s) to a (resolved) TransformDef body.

    Two cases, exactly one of which applies:

    1. **Lambda body** — the body contains formal-parameter ``Var`` leaves
       (e.g. ``Mu = L2(L1(x))`` has the formal ``x``). β-reduce by
       substituting the formal with the call argument.

    2. **Bare unapplied ctor** — the body has no formals
       (e.g. ``K = Linear(1, 20)``). The call argument becomes the module's
       positional input via the existing ``subst.sons = [...]`` convention.

    The formals are inferred fresh from the *resolved* ``subst`` (rather than
    being stored at parse time) because resolution / template materialization
    may have eliminated some of the original Var leaves — e.g. ``Mu<k, i>``
    had template params ``k, i`` that got materialized into literals, leaving
    only the true runtime formal in the body. The β-substitution returns a
    count so we can fall back to the bare-ctor path on a zero-substitution
    result (a defensive check for edge cases like a formal whose name
    happens to match a global scalar that got resolved away).
    """
    formals = collect_formal_vars(subst)
    if len(formals) > 1:
        # The current DSL only supports single-formal TransformDef bodies; a
        # body with multiple free Var leaves usually indicates a DSL typo (extra
        # unresolved Var). Surface it loudly here rather than silently leaving
        # extras unbound and producing confusing downstream errors.
        raise ValueError(
            f"TransformDef body has multiple unresolved formals {formals!r} after "
            f"symbol resolution; only single-formal bodies are supported. Body: {subst!r}"
        )
    if formals:
        new_body, n = _inject_formal_param(subst, formals[0], arg0)
        if n > 0:
            return new_body
    subst = _promote_body_sons_to_ctor_args(subst, engine)
    subst.sons = [replace_fn(s) for s in call_sons]
    return subst


def _inject_formal_param(
    body: Optional[TensorTerm],
    formal_name: str,
    actual: TensorTerm,
) -> tuple:
    """β-reduce: replace every ``TensorTerm(value=Var(formal_name))`` leaf inside
    *body* with *actual*. Returns ``(new_body, n_substitutions)``.

    A ``TransformDef`` body is a lambda; its formal parameters are the ``Var``
    leaves that remain after symbol resolution. At application time
    (``replace_tensor_term`` substituting ``Mu(arg)``), we walk the body and
    bind the formal name to the actual call argument — standard β-reduction.

    The pre-fix engine only looked for the literal name ``"inp"`` (a brittle,
    undocumented convention in the grammar). That broke any DSL body that used
    a different formal name (e.g. ``x``), silently collapsing nested alias
    calls into just the outermost layer. The regression suite
    is in ``tests/repro/test_c2_sparse_matmul_shape_mismatch.py``.
    """
    if body is None:
        return body, 0
    if body.value is not None and isinstance(body.value, Var) and body.value.name == formal_name:
        return actual, 1
    if body.sons:
        new_sons: list[TensorTerm] = []
        total = 0
        for s in body.sons:
            ns, k = _inject_formal_param(s, formal_name, actual)
            new_sons.append(ns)
            total += k
        if total:
            return TensorTerm(op=body.op, sons=new_sons, value=body.value), total
    return body, 0


def _promote_body_sons_to_ctor_args(subst: "TensorTerm", engine: "Engine") -> "TensorTerm":
    """Promote a transform-def body's resolved sons to hyper_params for the unapplied-ctor pattern.

    After ``replace_tensor_term(subst)``, any Var remaining in the body's sons is either a
    bool literal (True/False) or an unresolved name — never a rule variable (those are not in
    the symbol table).  If the op resolves to an ``nn.Module`` class and all sons convert to
    ``ArithTerm`` (i.e., are constants, literals, or arithmetic expressions over constants),
    lift them to ``hyper_params`` so that the subsequent ``subst.sons = [call_arg]`` overwrites
    with only the tensor input and doesn't destroy the constructor arguments.

    Returns ``subst`` unchanged if promotion is not applicable.
    """
    if not subst.sons or not isinstance(subst.op, TensorOp) or (subst.op.hyper_params or []):
        return subst
    # Lazy import to avoid circular dependency (engine ← tensor_term_compiler ← engine).
    from relann.tensor_term_compiler import (  # noqa: PLC0415
        resolve_op,
        tensor_term_to_arith_term,
        _arith_term_bool_var_to_primitive,
    )
    import inspect as _inspect
    import torch.nn as _nn
    resolved = resolve_op(str(subst.op.op), engine.get_run_globals())
    if resolved is None or not (_inspect.isclass(resolved) and issubclass(resolved, _nn.Module)):
        return subst
    try:
        new_hp = [_arith_term_bool_var_to_primitive(tensor_term_to_arith_term(s)) for s in subst.sons]
    except (ValueError, TypeError):
        return subst  # some son is a sub-module-call, not a constant → not a ctor pattern
    new_op = TensorOp(op=subst.op.op, template_args=subst.op.template_args, hyper_params=new_hp)
    return TensorTerm(op=new_op, sons=None, value=None)


@patch
def replace_all_vars_in_tg_using_symbol_table(
    self: Engine,
    tg: TermGraph,
    *,
    in_place: bool = False,
) -> TermGraph:
    """
    Replace symbols in transformation nodes using the engine's symbol table.

    The replacement rules are:
    - ``ArithTerm.value`` that is a ``Var`` referring to a scalar ``TransformDef`` is replaced
      with that scalar value.
    - ``TensorTerm.op`` that names a ``TransformDef`` is replaced by the referenced
      ``TransformDef``'s tensor term, applied to the call argument via
      ``_apply_call_argument`` (see its docstring above for the β-reduction vs.
      bare-ctor decision rule and the rationale for inferring formals on the
      resolved body via ``collect_formal_vars``).

    Args:
        tg: Term graph to operate on.
        in_place: Mutate ``tg`` when ``True``; otherwise (default) operate on and return a deep copy.

    Returns:
        The graph whose transformation nodes have had symbolic references resolved.
    """

    working_tg = tg if in_place else deepcopy(tg)

    def replace_arith_term(a: ArithTerm) -> ArithTerm:
        if a is None:
            return a
        changed = False

        # Recurse into sons (if present)
        new_sons = None
        if a.sons:
            new_sons = [replace_arith_term(s) for s in a.sons]
            if any(ns is not os for ns, os in zip(new_sons, a.sons)):
                changed = True

        # Replace Var-valued hyper-params (e.g., d) with scalar from TransformDef
        new_val = a.value
        if isinstance(a.value, Var):
            sym = self.get_symbol(a.value.name)
            if sym:
                typ, obj = sym
                if typ == TransformDef and getattr(obj, "tensor_term", None) is not None:
                    tt = obj.tensor_term
                    # Only substitute scalars (e.g., d = 64)
                    if getattr(tt, "op", None) is None and getattr(tt, "value", None) is not None:
                        new_val = tt.value
                        changed = True

        if not changed:
            return a
        return ArithTerm(op=a.op, sons=new_sons, value=new_val)

    def replace_tensor_term(t: TensorTerm) -> TensorTerm:
        if t is None:
            return t

        # If op is a symbol referencing a TransformDef (e.g., Lin), substitute whole term
        if isinstance(t.op, TensorOp) and isinstance(t.op.op, str):
            sym = self.get_symbol(t.op.op)
            if sym:
                typ, obj = sym
                if typ == TransformDef and getattr(obj, "tensor_term", None) is not None:
                    # Templated TransformDef: materialize with template_args first
                    if getattr(obj, "template_params", None) and getattr(t.op, "template_args", None):
                        _, td = self._get_or_materialize_transform(
                            t.op.op, obj, t.op.template_args
                        )
                        subst = deepcopy(td.tensor_term)
                        subst = replace_tensor_term(subst)
                        if t.sons:
                            arg0 = replace_tensor_term(t.sons[0])
                            subst = _apply_call_argument(subst, arg0, self, replace_tensor_term, t.sons)
                        return subst

                    subst = deepcopy(obj.tensor_term)
                    # Recursively resolve inside the subst as well
                    subst = replace_tensor_term(subst)
                    if t.sons:
                        arg0 = replace_tensor_term(t.sons[0])
                        subst = _apply_call_argument(subst, arg0, self, replace_tensor_term, t.sons)
                    return subst

        # Otherwise, rebuild this term with possible changes in hyper_params, sons, value
        new_op = t.op
        if isinstance(t.op, TensorOp) and t.op.hyper_params:
            new_hps = [replace_arith_term(hp) for hp in t.op.hyper_params]
            if any(n is not o for n, o in zip(new_hps, t.op.hyper_params)):
                new_op = copy(t.op)
                new_op.hyper_params = new_hps

        new_sons = None
        if t.sons:
            new_sons = [replace_tensor_term(s) for s in t.sons]

        # If value is a Var referencing a TransformDef, replace the term with its tensor_term body.
        # This handles both scalar TransformDefs (like `d = 96`) and non-scalar ones (like `W_ATT = Linear(d/h, d/h, False)`).
        if isinstance(t.value, Var):
            sym = self.get_symbol(t.value.name)
            if sym:
                typ, obj = sym
                if typ == TransformDef and getattr(obj, "tensor_term", None) is not None:
                    tt = deepcopy(obj.tensor_term)
                    tt = replace_tensor_term(tt)
                    return tt

        # VarTemplated: templated TransformDef reference as a leaf (e.g., Lin<64> without call parens)
        if isinstance(t.value, VarTemplated):
            sym = self.get_symbol(t.value.name)
            if sym:
                typ, obj = sym
                if typ == TransformDef and getattr(obj, "template_params", None):
                    template_args = [
                        ArithTerm(value=p) if not isinstance(p, ArithTerm) else p
                        for p in t.value.template_params
                    ]
                    _, td = self._get_or_materialize_transform(
                        t.value.name, obj, template_args
                    )
                    tt = deepcopy(td.tensor_term)
                    tt = replace_tensor_term(tt)
                    return tt

        if new_op is not t.op or (t.sons and any(ns is not os for ns, os in zip(new_sons, t.sons))):
            return TensorTerm(op=new_op, sons=new_sons, value=t.value)
        return t

    # Apply only to transformation nodes
    from relann.pydantic_classes import TransformDef, TensorOp
    for _, data in working_tg.nodes(data=True):
        if data.get("type") == "transformation" and data.get("transformation") is not None:
            term = data["transformation"]
            new_term = replace_tensor_term(term)
            if new_term is not term:
                data["transformation"] = new_term
                # Preserve TransformDef name only for user-defined transforms (weight sharing).
                # Do not set for builtin ops (Linear, ReLU, etc.): they are not in the symbol
                # table as TransformDef, and using the op name as base_name would create a
                # duplicate parameter key (e.g. global.Linear.linear.weight alongside
                # global.transformation_PapersEmb1.linear.weight) for the same tensor.
                if isinstance(term.op, TensorOp) and isinstance(term.op.op, str):
                    sym = self.get_symbol(term.op.op)
                    if sym:
                        typ, obj = sym
                        if typ == TransformDef:
                            if getattr(obj, "template_params", None) and getattr(term.op, "template_args", None):
                                key, _ = self._get_or_materialize_transform(
                                    term.op.op, obj, term.op.template_args
                                )
                                data["transform_def_name"] = key
                            else:
                                data["transform_def_name"] = term.op.op
                # For FQN/display: record TransformDef names so shared params key by name (K, Q) not node path.
                # Allows partial matches: non-TransformDef sons get None.
                if term.sons:
                    child_names = []
                    has_any = False
                    for son in term.sons:
                        if isinstance(getattr(son, "op", None), TensorOp) and isinstance(getattr(son.op, "op", None), str):
                            sym = self.get_symbol(son.op.op)
                            if sym:
                                typ, obj = sym
                                if typ == TransformDef:
                                    if getattr(obj, "template_params", None) and getattr(son.op, "template_args", None):
                                        key, _ = self._get_or_materialize_transform(
                                            son.op.op, obj, son.op.template_args
                                        )
                                        child_names.append(key)
                                    else:
                                        child_names.append(son.op.op)
                                    has_any = True
                                    continue
                        child_names.append(None)
                    if has_any:
                        data["transform_def_child_names"] = child_names

                # Build recursive path→TransformDef map for compound expressions
                # (e.g. K<1>(z1) * Q<1>(z2) * Mu<1>) so _build_param_fqn can
                # resolve names like "left.left" → "K<1>" inside _ArithmeticWrapper trees.
                path_map = self._build_transform_def_path_map(term)
                if path_map:
                    data["transform_def_path_map"] = path_map
    return working_tg

# %%
if __name__ == "__main__":

    # Test add_transform with a basic example using d = 64 and Lin = Linear(16, 32)
    engine = Engine()

    # Set up variables and a transform definition
    prog_transform_str = """
    d = 64 .
    Lin = Linear(16,32) .

    Embed64(X; Linear(d, d)(z1)) :- InputData(X;z1) .
    TestUseLin(X; Lin(z2)) :- InputData2(X;z2) .
    TestUseLin2(X; Linear(16,32)(Lin(z2))) :- InputData3(X;z2) .
    """
    prog_with_transforms = parse_and_transform_str(prog_transform_str)
    engine.add_program(prog_with_transforms)

# %%
if __name__ == "__main__":

    engine.symbol_table['global']

# %%
if __name__ == "__main__":

    # Test the replace_all_vars_in_tg_using_symbol_table function
    tg = engine.term_graphs['global']

    print("=== Term graph before replacing vars ===")
    draw_tg(engine.term_graphs['global'])

    engine.replace_all_vars_in_tg_using_symbol_table(tg, in_place=True)

    print("\n=== Term graph after replacing vars ===")
    draw_tg(engine.term_graphs['global'])

# %%
if __name__ == "__main__":

    def contains_var_d(x):
        if isinstance(x, Var): return x.name == "d"
        if isinstance(x, ArithTerm):
            if isinstance(x.value, Var) and x.value.name == "d": return True
            return any(contains_var_d(s) for s in (x.sons or []))
        if isinstance(x, TensorTerm):
            if isinstance(x.value, Var) and x.value.name == "d": return True
            if isinstance(x.op, TensorOp) and x.op.hyper_params:
                for hp in x.op.hyper_params:
                    if isinstance(hp.value, Var) and hp.value.name == "d": return True
            return any(contains_var_d(s) for s in (x.sons or []))
        return False

    def contains_op_lin(t):
        if isinstance(t, TensorTerm):
            if isinstance(t.op, TensorOp) and t.op.op == "Lin": return True
            return any(contains_op_lin(s) for s in (t.sons or []))
        return False

    # assumes: engine, tg exist and you've already called engine.replace_all_vars_in_tg_using_symbol_table(tg)
    for _, data in tg.nodes(data=True):
        if data.get("type") == "transformation":
            term = data["transformation"]
            assert not contains_var_d(term), "Unresolved Var('d') still present"
            assert not contains_op_lin(term), "Unresolved op 'Lin' still present"
    print("OK: no 'd' Vars and no 'Lin' ops remain in transformation terms.")

# %%
@patch
def _normalize_db_relations(self: Engine, db: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Normalize all relations in the db dictionary to ensure they have proper
    embedding_shapes and embeddings format. This is useful when db contains
    tuples like (DataFrame, tensor) that need to be normalized.
    
    Args:
        db: Dictionary mapping relation names to relation data (DataFrame, tuple, dict, etc.)
        
    Returns:
        Dictionary mapping relation names to normalized relation dicts with
        'content', 'content_schema', 'embedding_shapes', and 'embeddings' keys.
    """
    from relann.era_operations import _to_er_dict
    from relann.data_sources import RelationSource
    normalized_db = {}
    for name, rel_data in db.items():
        if isinstance(rel_data, RelationSource):
            rel_data = rel_data.load_full()
        normalized = _to_er_dict(rel_data)
        # Verify that if we have embeddings, we also have embedding_shapes
        if normalized.get("embeddings") and not normalized.get("embedding_shapes"):
            # If we have embeddings but no shapes, compute shapes from embeddings
            embeddings = normalized["embeddings"]
            if isinstance(embeddings, (list, tuple)):
                normalized["embedding_shapes"] = [e.shape for e in embeddings]
            elif torch.is_tensor(embeddings):
                normalized["embedding_shapes"] = [embeddings.shape]
        normalized_db[name] = normalized
    return normalized_db


@patch
def _ensure_db_relations_normalized(self: Engine):
    """
    Ensure all relations in self.db are properly normalized with embedding_shapes.
    This should be called before using the db in fit/predict operations.
    """
    if self.db:
        normalized_db = self._normalize_db_relations(self.db)
        # Update db with normalized relations
        self.db = normalized_db

# %% [markdown]
# ### fit

# %%
@patch
def fit(self: Engine, fit_stmt: FitStatement):
    self._last_module = None
    # Add the rule and get term graph
    self.add_rule(fit_stmt.rule)
    tg = self.term_graphs["global"]

    # Use induced_subgraph to get the part of the graph relevant to this rule
    rule_name, fit_params = fit_stmt.rule.lhs.name, fit_stmt.fit_params

    tg.mark_phase(rule_name, "fit")  # Visualization metadata: mark nodes involved in this FitStatement.

    sub_tg = tg.induced_subgraph(node_name=rule_name, direction="ancestors", include_root=True)
    
    # After all symbol table variables have been resolved in sub_tg, we can refer to this as a "grounded" or "concrete" term graph.
    ground_sub_tg = self.replace_all_vars_in_tg_using_symbol_table(sub_tg, in_place=False)

    ###################################################################################################################
    # # TODO: collapse the terms in the grounded sub_tg using boaz's method.
    # # also a good name instead of 'collapse' is 'evaluate'.
    # # materialize_sub_tg = self.collapse_terms_in_tg(ground_sub_tg)
    # materialize_sub_tg = ground_sub_tg
    
    # # net = self.er_ref_to_torch(rule_name=rule_name)

    # # materialize data loaders...

    # # for epoch in range(epochs):
    # #     for batch in batches:
    # #         inputs = get_batch_data(data_loaders_dict) # tables we fit on now
    # #         y,y_tag = net.forward(inputs)
    # #         l = loss_function(y,y_tag)
    # #         optimizer.step(l)
    # #
    # # return net
    # return materialize_sub_tg
    ###################################################################################################################

    data_sources = self._collect_data_sources(ground_sub_tg)
    if not data_sources:
        raise ValueError(
            f"No data loaders were found for rule '{rule_name}'. Ensure the fit statement references input data."
        )

    fit_config = self._normalize_fit_params(fit_stmt.fit_params)
    epochs = fit_config.get("epochs", 1)
    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    if logger.isEnabledFor(logging.DEBUG):
        draw_tg(ground_sub_tg)

    # Compile DSL TensorTerms before building the module. The e-graph
    # optimizer was removed from juplit (PR #56 will re-introduce it with
    # proper test coverage).
    ground_sub_tg = self.eval_tensor_terms_on_tg(ground_sub_tg)
    module = term_graph_to_module(
        ground_sub_tg,
        param_loader=self,
        engine=self,
    )
    module.train()

    module.instantiate(data_sources)
    optimizer = self._create_optimizer(module.parameters(), fit_config)
    history: list[float] = []

    # Calculate print interval for debug mode (every 10% of epochs)
    print_interval = max(1, epochs // 10)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = module.forward()
        # Loss is computed as part of the transformation, so extract it from the output embeddings
        if not getattr(out, "embeddings", None) or len(out.embeddings) == 0:
            raise ValueError("Loss computation requires output embeddings from the transformation")
        loss_value = out.embeddings[0].mean()
        if not torch.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}: {loss_value.item()}")
        loss_value.backward()
        optimizer.step()
        current_loss = float(loss_value.detach().cpu())
        history.append(current_loss)

        # Print progress at 10% intervals
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % print_interval == 0:
            print(f"  Epoch {(epoch+1):3d}, Loss: {current_loss:.4f}")

    # Save all trained parameters to the parameter table
    namespace = self.current_namespace
    self._save_module_parameters(module, namespace, rule_name)

    self._last_module = module
    self._last_module_tg = tg

    self.trained_modules[rule_name] = {
        "module": module,
        "loss_history": history,
        "fit_config": fit_config,
    }

    # Print training progress
    if history:
        print(f"Training completed: {history[0]:.4f} -> {history[-1]:.4f} ({len(history)} epochs)")

    return self.trained_modules[rule_name]


@patch
def _materialise_if_source(self: Engine, name: str) -> Any:
    """Materialise a ``RelationSource`` entry once and cache it in-place in ``self.db``.

    Returns the entry currently in ``self.db[name]``. If the entry is a
    ``RelationSource``, calls ``load_full()`` and replaces it with the resulting
    ER-dict so subsequent reads see the materialised view. Legacy tuples / dicts
    are returned unchanged.

    This enables ``Session(db={"BigTable": SqlSource(...)})`` without paying the
    round-trip cost until "BigTable" is actually referenced by a rule. If the
    user reassigns ``engine.db[name]``, the cache is implicitly cleared.
    """
    from relann.data_sources import RelationSource
    entry = self.db[name]
    if isinstance(entry, RelationSource):
        logger.debug("Engine: materialising RelationSource %r via load_full()", name)
        materialised = entry.load_full()
        self.db[name] = materialised
        if self.device is not None:
            embs = materialised.get("embeddings")
            if embs:
                materialised["embeddings"] = [
                    e.to(self.device) if isinstance(e, torch.Tensor) else e for e in embs
                ]
        return materialised
    return entry


@patch
def _collect_data_sources(self: Engine, graph: nx.DiGraph) -> Dict[str, Dict[str, Any]]:
    data_sources: Dict[str, Dict[str, Any]] = {}
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "data_loader":
            continue
        rel_name = data.get("name") or node_id
        if rel_name in data_sources:
            continue
        if self.db is None or rel_name not in self.db:
            raise KeyError(f"Data loader '{rel_name}' not found in Engine.db")
        entry = self._materialise_if_source(rel_name)
        normalized = self._normalize_relation_payload(entry)
        data_sources[rel_name] = normalized
    return data_sources


@patch
def _normalize_relation_payload(self: Engine, rel: Any) -> Dict[str, Any]:
    from relann.era_operations import _to_er_dict  # type: ignore
    from relann.data_sources import RelationSource
    if isinstance(rel, RelationSource):
        rel = rel.load_full()
    return _to_er_dict(rel)


@patch
def _resolve_external_symbol(self: Engine, name: str) -> Any | None:
    fq_name = self._function_or_nn_module_exists(name)
    if not fq_name:
        return None
    module_name, _, attr = fq_name.rpartition(".")
    if not module_name:
        return getattr(builtins, attr, None)
    module = import_module(module_name)
    return getattr(module, attr, None)


@patch
def _create_optimizer(
    self: Engine,
    parameters: Iterable[torch.nn.Parameter],
    fit_config: Dict[str, Any],
) -> torch.optim.Optimizer:
    optimizer_spec = fit_config.get("optimizer", "adam")
    lr = fit_config.get("lr", 1e-3)
    optimizer_cls: type[torch.optim.Optimizer]

    if isinstance(optimizer_spec, str):
        name = optimizer_spec.lower()
        mapping: Dict[str, type[torch.optim.Optimizer]] = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported optimizer '{optimizer_spec}'")
        optimizer_cls = mapping[name]
    elif isinstance(optimizer_spec, type) and issubclass(optimizer_spec, torch.optim.Optimizer):
        optimizer_cls = optimizer_spec
    elif callable(optimizer_spec):
        optimizer = optimizer_spec(parameters, lr=lr)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Custom optimizer factory must return a torch.optim.Optimizer instance")
        return optimizer
    else:
        raise TypeError("optimizer must be a string name, Optimizer subclass, or factory callable")

    extra_kwargs = {}
    if optimizer_cls is torch.optim.SGD:
        momentum = fit_config.get("momentum", 0.0)
        extra_kwargs["momentum"] = momentum
        
    # Add weight_decay if specified (supported by Adam, SGD, AdamW)
    if "weight_decay" in fit_config:
        extra_kwargs["weight_decay"] = fit_config["weight_decay"]
    
    return optimizer_cls(parameters, lr=lr, **extra_kwargs)


@patch
def _create_loss(
    self: Engine,
    fit_config: Dict[str, Any],
) -> Tuple[Optional[Callable[..., torch.Tensor]], Dict[str, Any]]:
    loss_spec = fit_config.get("loss", None)
    loss_kwargs: Dict[str, Any] = {
        "target_column": fit_config.get("target_column"),
        "prediction_index": fit_config.get("prediction_index", 0),
    }

    if loss_spec is None:
        return None, loss_kwargs

    if callable(loss_spec):
        return loss_spec, loss_kwargs

    if isinstance(loss_spec, str):
        name = loss_spec.lower()
        mapping: Dict[str, Callable[..., torch.Tensor]] = {
            "crossentropy": torch.nn.functional.cross_entropy,
            "nllloss": torch.nn.functional.nll_loss,
            "mse": torch.nn.functional.mse_loss,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported loss '{loss_spec}'")
        return mapping[name], loss_kwargs

    raise TypeError("loss must be a callable or string name")


@patch
def _compute_loss(
    self: Engine,
    er: Any,
    loss_fn: Optional[Callable[..., torch.Tensor]],
    loss_kwargs: Dict[str, Any],
) -> torch.Tensor:
    if loss_fn is None:
        if not getattr(er, "embeddings", None):
            raise ValueError("Model output provides no embeddings to compute loss from")
        tensor = er.embeddings[0]
        return tensor.mean() if tensor.ndim > 0 else tensor

    prediction_index = loss_kwargs.get("prediction_index", 0)
    if not getattr(er, "embeddings", None):
        raise ValueError("Loss function requires embeddings but none were produced by the rule")
    if prediction_index >= len(er.embeddings):
        raise IndexError(f"prediction_index {prediction_index} out of range for embeddings list")
    preds = er.embeddings[prediction_index]

    df = getattr(er, "content", None)
    target_column = loss_kwargs.get("target_column")
    targets = None
    if df is not None:
        if hasattr(df, "__class__") and df.__class__.__module__.startswith("cudf"):
            df = df.to_pandas()
        if target_column is None:
            target_column = self._infer_target_column(df)
        if target_column is None:
            raise ValueError(
                "Unable to infer target column for loss computation; specify 'target_column' in fit params"
            )
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found in model output columns {list(df.columns)}")
        targets = torch.as_tensor(df[target_column].to_numpy(), device=preds.device)

    if targets is None:
        raise ValueError("Loss computation requires targets but none were provided or inferred")
    return loss_fn(preds, targets)


@patch
def _infer_target_column(self: Engine, df) -> Optional[str]:
    preferred = ["target", "target_id", "label", "labels", "y", "gt", "ground_truth"]
    for name in preferred:
        if name in df.columns:
            return name
    return None


@patch
def _save_module_parameters(
    self: Engine,
    module: nn.Module,
    namespace: str,
    rule_name: str
) -> None:
    """Save trained parameters from *module* back into ``Engine.parameter_store``.

    Traverses the RelNN graph, finds every transformation node, and calls
    ``_extract_and_store_parameters`` with ``overwrite=True`` so that the
    trained tensor values replace the initial ones.
    """
    if not hasattr(module, 'graph'):
        return

    if not hasattr(module, "module_for_node"):
        raise RuntimeError(
            "Expected RelNN-like module with module_for_node(node_id) accessor"
        )

    for node_id, node_data in module.graph.nodes(data=True):
        if node_data.get('type') == 'transformation':
            transformation_op = None
            # Resolve by graph-node id through the canonical RelNN accessor.
            try:
                transformation_op = module.module_for_node(node_id)
            except RelNNModuleLookupError:
                transformation_op = None
            except Exception as e:
                raise RuntimeError(
                    f"Failed to resolve transformation module for node '{node_id}'"
                ) from e

            if transformation_op is not None and hasattr(transformation_op, 'transformation'):
                # Same source-node dispatch as the initial extract path so
                # trained weights persist back under the source Apply's FQN.
                eff_node_id, eff_attrs = self._resolve_param_source_node(node_id, node_data)
                self._extract_and_store_parameters(
                    module=transformation_op.transformation,
                    namespace=namespace,
                    node_id=eff_node_id,
                    transform_def_name=self._resolve_transform_def_name_from_node(eff_attrs),
                    transform_def_child_names=eff_attrs.get("transform_def_child_names"),
                    overwrite=True,
                    transform_def_path_map=eff_attrs.get("transform_def_path_map"),
                )

# %%
if __name__ == "__main__":

    fit_program_str = """
    Encoder(X; Linear(3, 4)(z_in)) :- InputFeatures(X; z_in) .
    Predict(X; Linear(4, 2)(z_pred)) :- Encoder(X; z_pred) .
    ?fit <epochs=10, batch_size=2> Loss(; MSELoss()(z_pred, z_label)) :- Predict(X; z_pred), Labels(X; z_label) .
    """

    program = parse_and_transform_str(fit_program_str)

    fit_statements = [stmt for stmt in program.statements if isinstance(stmt, FitStatement)]
    assert len(fit_statements) == 1, "Program should contain exactly one fit statement"
    fit_stmt = fit_statements[0]

    assert {k: v.value for k, v in fit_stmt.fit_params.items()} == {"epochs": 10, "batch_size": 2}


    # Provide actual DataFrames and embeddings as tuples (DataFrame, tensor)
    input_content_df = pd.DataFrame({"X": [0, 1, 2]})
    input_embeddings = torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7], [0.5, 0.5, 0.5]], dtype=torch.float32)

    labels_content_df = pd.DataFrame({"X": [0, 1, 2]})
    labels_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    engine = Engine(db={
        "InputFeatures": (input_content_df, input_embeddings),
        "Labels": (labels_content_df, labels_embeddings)
    })
    engine.add_program(program)

# %%
if __name__ == "__main__":
    engine.trained_modules

# %%
if __name__ == "__main__":
    engine.parameter_store

# %% [markdown]
# ### preety print parametersm

# %%
def _clean_display_name(n: str) -> str:
    """Strip engine-internal prefixes and path segments from a parameter FQN."""
    d = n
    for prefix in ["global.", "ops.", "transformation_", "transformation."]:
        if d.startswith(prefix):
            d = d[len(prefix):]
    d = d.replace("transformation.", "")
    d = re.sub(r"children_modules\.(\d+)", r"\1", d)
    d = d.replace("._module.", ".").replace("._module", "")
    d = d.replace(".inner.", ".").replace(".inner", "")
    return d


def _normalize_subpath_for_display(subpath: str) -> str:
    """Normalize compiler-internal subpaths to a stable, Torch-style display path.

    - N.right / N.left -> N (matmul operand internals hidden per Concat child index)
    - right / left -> 0 (single arithmetic wrapper branch)
    - numeric child indices remain unchanged (0,1,2,3,...)
    """
    if not subpath:
        return subpath
    m = re.fullmatch(r"(\d+)\.(left|right)", subpath)
    if m:
        return m.group(1)
    if subpath in ("left", "right"):
        return "0"
    return subpath


def _clean_param_fqn(name: str) -> str:
    """Full cleaning pipeline: _clean_display_name + _normalize_subpath_for_display."""
    cleaned = _clean_display_name(name)
    if "." in cleaned:
        parts = cleaned.split(".")
        if len(parts) >= 2:
            rule, param_name = parts[0], parts[-1]
            subpath = ".".join(parts[1:-1])
            norm = _normalize_subpath_for_display(subpath)
            return f"{rule}.{norm}.{param_name}" if norm else f"{rule}.{param_name}"
    return cleaned


def pretty_print_params(model_or_engine, show_stats: bool = True, max_name_width: int = 50):
    """
    Pretty print model parameters in a clean, minimalistic table format,
    sorted according to insertion order in RelNN models.

    - Removes 'global.' and 'ops.' prefixes as they're not meaningful.
    - Prints either top-level parameter stats or just name and shape, depending on show_stats.
    Removes every occurrence of 'transformation.' from the parameter name for display.
    Adds a final total line with the total number of parameters.
    """
    import torch
    import torch.nn as nn

    def get_insertion_order_params(params_dict):
        """Return (name, param) pairs in dict insertion order (guaranteed in Python 3.7+)."""
        return list(params_dict.items())

    # Determine if input is Engine or Module
    if hasattr(model_or_engine, 'parameter_store') and isinstance(model_or_engine.parameter_store, dict):
        params_dict = model_or_engine.parameter_store
        total_params = sum(p.numel() for p in params_dict.values())
        num_param_tensors = len(params_dict)
        print(f"Model Parameters ({total_params:,} trainable parameters, {num_param_tensors} parameter tensor{'s' if num_param_tensors != 1 else ''})")
        print("=" * 85)
        sorted_params = get_insertion_order_params(params_dict)
    elif isinstance(model_or_engine, nn.Module):
        params_dict = {name: param for name, param in model_or_engine.named_parameters()}
        total_params = sum(p.numel() for p in params_dict.values())
        num_param_tensors = len(params_dict)
        print(f"Model Parameters ({total_params:,} trainable parameters, {num_param_tensors} parameter tensor{'s' if num_param_tensors != 1 else ''})")
        print("=" * 85)
        sorted_params = list(params_dict.items())
    else:
        raise ValueError("Input must be an Engine instance or PyTorch nn.Module")

    # Print column headers
    print(f"{'Name':<{max_name_width}} {'Shape':>20} {'#Params':>10}")
    print("-" * 85)

    display_entries = []
    for name, param in sorted_params:
        cleaned = _clean_display_name(name)
        normalized_display_name = cleaned
        if "." in cleaned:
            parts = cleaned.split(".")
            if len(parts) >= 2:
                rule = parts[0]
                param_name = parts[-1]
                subpath = ".".join(parts[1:-1])
                normalized_subpath = _normalize_subpath_for_display(subpath)
                if normalized_subpath:
                    normalized_display_name = f"{rule}.{normalized_subpath}.{param_name}"
                else:
                    normalized_display_name = f"{rule}.{param_name}"
        display_entries.append(
            {
                "param": param,
                "cleaned": cleaned,
                "display": normalized_display_name,
            }
        )

    # If normalization collapses multiple distinct parameter paths into one display name,
    # preserve readability by falling back to the cleaned path for only those collisions.
    display_name_to_cleaned_paths = {}
    for entry in display_entries:
        display_name_to_cleaned_paths.setdefault(entry["display"], set()).add(entry["cleaned"])
    ambiguous_display_names = {
        display_name
        for display_name, cleaned_paths in display_name_to_cleaned_paths.items()
        if len(cleaned_paths) > 1
    }
    for entry in display_entries:
        if entry["display"] in ambiguous_display_names:
            entry["display"] = entry["cleaned"]

    grand_total = 0
    for entry in display_entries:
        display_name = entry["display"]
        param = entry["param"]

        if len(display_name) > max_name_width:
            display_name = display_name[:max_name_width-3] + "..."

        shape_str = str(tuple(param.shape))
        num_params = param.numel()
        grand_total += num_params

        # Format: Name | Shape | #Params
        print(f"{display_name:<{max_name_width}} {shape_str:>20} {num_params:>10,}")

        # Optionally show statistics
        if show_stats and num_params > 0:
            with torch.no_grad():
                p_data = param.data.flatten()
                p_min = p_data.min().item()
                p_max = p_data.max().item()
                p_mean = p_data.mean().item()
                p_std = p_data.std().item() if p_data.numel() >= 2 else 0.0
                print(f"{'':>{max_name_width}} {'stats:':>20} min={p_min:7.4f}  max={p_max:7.4f}  mean={p_mean:7.4f}  std={p_std:7.4f}")

    print("=" * 85)
    print(f"{'Total':<{max_name_width}} {'':>20} {grand_total:>10,}")
    print("=" * 85)

# %%
if __name__ == "__main__":
    # Show the parameters of the trained model
    print(len(engine.parameter_store))
    engine.parameter_store.keys()

# %%
if __name__ == "__main__":

    pretty_print_params(engine, show_stats=False)

# %%
if __name__ == "__main__":
    import sys
    from pathlib import Path

# %%
if __name__ == "__main__":
    # Set project root to the parent folder (one level up from the current directory)
    # Add project root to sys.path so utils can be imported as a package

    _project_root = Path().resolve().parent
    if _project_root.exists():
        _project_root_str = str(_project_root)
        if _project_root_str not in sys.path:
            sys.path.insert(0, _project_root_str)

# %%
if __name__ == "__main__":
    print_model_params(engine.trained_modules['Loss']['module'])

# %% [markdown]
# ### predict

# %%
@patch
def _apply_lhs_decode(self: Engine, lhs, predictions, *, _is_predict_context: bool = True):
    """Apply predict-rule LHS ``[...]`` decode steps to the forward output ER.

    If the LHS has no decode brackets (no ``ContentDecode`` entries), returns
    ``predictions`` unchanged.

    Args:
        lhs: The LHS relation object containing ``derived_content_attrs``.
        predictions: The ``EmbeddedRelation`` returned by ``module.forward()``.
        _is_predict_context: Must be ``True`` (the default). Passing ``False``
            raises ``NotImplementedError`` because decode-as-a-mid-graph-op is
            not yet supported; this guard prevents accidental misuse.
    """
    if not _is_predict_context:
        raise NotImplementedError(
            "_apply_lhs_decode is only supported after ?pred forward(). "
            "Decode as an intermediate term-graph op is not yet implemented."
        )

    from relann.embedded_relation import EmbeddedRelation
    from relann.tensor_term_compiler import TensorTermCompiler, resolve_op

    decode_attrs = [a for a in lhs.derived_content_attrs if isinstance(a, ContentDecode)]
    if not decode_attrs:
        return predictions
    if not predictions.embeddings:
        raise ValueError(
            "Cannot decode: no embeddings in predict output. "
            "Check that the rule produces an embedding tensor."
        )
    if len(predictions.embeddings) != 1:
        raise ValueError(
            f"Cannot decode: expected exactly 1 embedding in predict output, "
            f"got {len(predictions.embeddings)}. "
            "Decode only supports single-embedding predict rules."
        )
    emb = predictions.embeddings[0]
    tc = TensorTermCompiler(self)
    df = predictions.content.copy() if predictions.content is not None else pd.DataFrame()
    for attr in decode_attrs:
        col_name = attr.column.name
        if attr.decoder_name:
            resolved = resolve_op(attr.decoder_name, self.get_run_globals())
            if resolved is None:
                raise ValueError(
                    f"Unknown decode module {attr.decoder_name!r}. "
                    "Import it in the run scope or use a built-in such as ArgMax."
                )
            mock_tterm = TensorTerm(
                op=TensorOp(op=attr.decoder_name, hyper_params=attr.decoder_params or []),
            )
            dec_mod = tc._instantiate(resolved, mock_tterm)
            with torch.no_grad():
                decoded = dec_mod(emb)
        else:
            # Bare [col]: only a trivial decode — a 1-D (N,) or 2-D (N, 1) tensor
            # written straight into the column. Any other shape needs an explicit
            # decoder; the framework does not guess a reduction.
            if emb.dim() == 1 or (emb.dim() == 2 and emb.size(1) == 1):
                decoded = emb.squeeze(-1) if emb.dim() > 1 else emb
            else:
                raise ValueError(
                    f"Cannot decode an embedding of shape {tuple(emb.shape)} into the single "
                    f"column {col_name!r}. Specify a decoder in the predict-rule bracket, "
                    f"e.g. [ArgMax()({col_name})]."
                )
        arr = decoded.detach().cpu().numpy()
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.reshape(-1)
        df[col_name] = arr
    return EmbeddedRelation(
        content_schema=[str(c) for c in df.columns],
        embedding_shapes=[e.shape for e in predictions.embeddings],
        content=df,
        embeddings=predictions.embeddings,
        column_vocabs=getattr(predictions, "column_vocabs", None),
        data_version=int(getattr(predictions, "data_version", 0)),
    )


@patch
def predict(self: Engine, predict_stmt: PredictStatement):
    """
    Make predictions using saved parameters from the parameter table.
    Returns:
        The output of the module's forward pass (typically an EmbeddedRelation
        with embeddings and/or content attributes)
    """
    self._last_module = None
    # Get the rule name
    rule_name = predict_stmt.rule.lhs.name
    
    # Add the rule to the term graph (if not already)
    self.add_rule(predict_stmt.rule)
    tg = self.term_graphs["global"]

    tg.mark_phase(rule_name, "predict")  # Visualization metadata: mark nodes involved in this PredictStatement.
    
    # Use induced_subgraph to get the part of the graph relevant to this rule
    sub_tg = tg.induced_subgraph(node_name=rule_name, direction="ancestors", include_root=True)
    
    # Replace variables using symbol table to get a grounded term graph
    ground_sub_tg = self.replace_all_vars_in_tg_using_symbol_table(sub_tg, in_place=False)
    
    # Collect data sources from the graph
    data_sources = self._collect_data_sources(ground_sub_tg)
    if not data_sources:
        raise ValueError(
            f"No data loaders were found for rule '{rule_name}'. "
            f"Ensure the predict statement references input data."
        )
    
    # Compile DSL TensorTerms before building the module. Saved parameters
    # will be automatically loaded via the engine reference.
    ground_sub_tg = self.eval_tensor_terms_on_tg(ground_sub_tg)
    module = term_graph_to_module(
        ground_sub_tg,
        param_loader=self,
        engine=self,
    )

    # Set module to evaluation mode (disables dropout, batch norm updates, etc.)
    module.eval()

    # Make predictions with no gradient computation for efficiency
    with torch.no_grad():
        module.instantiate(data_sources)
        # Run forward pass to get predictions
        predictions = module.forward()
        predictions = self._apply_lhs_decode(predict_stmt.rule.lhs, predictions)

    self._last_module = module
    self._last_module_tg = tg

    return predictions

# %%
if __name__ == "__main__":

    # Minimal test for predict method
    predict_test_program_str = """
    Encoder(X; Linear(3, 4)(z_in)) :- InputFeatures(X; z_in) .
    SmallNet(X; Linear(4, 2)(z_pred)) :- Encoder(X; z_pred) .
    ?fit <epochs=10, batch_size=2> Loss(; MSELoss()(z_pred, z_label)) :- SmallNet(X; z_pred), Labels(X; z_label) .
    ?pred Output(X; z) :- SmallNet(X; z) .
    """
    program = parse_and_transform_str(predict_test_program_str)

    input_content_df = pd.DataFrame({"X": [0, 1, 2]})
    input_embeddings = torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7], [0.5, 0.5, 0.5]], dtype=torch.float32)
    labels_content_df = pd.DataFrame({"X": [0, 1, 2]})
    labels_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    engine = Engine(db={
        "InputFeatures": (input_content_df, input_embeddings),
        "Labels": (labels_content_df, labels_embeddings)
    })
    output = engine.add_program(program)

    # The predict statement will be executed during add_program due to the add_program logic.

    # There may be no need to call engine.predict explicitly or check return value here,
    # as add_program already triggers predict statements. All side effects or outputs
    # from prediction should be handled inside the predict/apply logic.

# %%
if __name__ == "__main__":
    print(output)
    print(output.content)
    print(output.embedding_shapes)
    print(output.embeddings)

# %%
if __name__ == "__main__":
    draw_tg(engine.term_graphs["global"])

# %%
if __name__ == "__main__":

    # Test that two different predict statements produce identical results using saved weights
    # Step 1: Create a program with fit statement and train the model
    fit_program_str = """
    Encoder(X; Linear(3, 4)(z_in)) :- InputFeatures(X; z_in) .
    SmallNet(X; Linear(4, 2)(z_pred)) :- Encoder(X; z_pred) .
    ?fit <epochs=10, batch_size=2> Loss(; MSELoss()(z_pred, z_label)) :- SmallNet(X; z_pred), Labels(X; z_label) .
    """

    fit_program = parse_and_transform_str(fit_program_str)

    # Create data
    input_content_df = pd.DataFrame({"X": [0, 1, 2]})
    input_embeddings = torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7], [0.5, 0.5, 0.5]], dtype=torch.float32)
    labels_content_df = pd.DataFrame({"X": [0, 1, 2]})
    labels_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    # Create engine and train
    engine = Engine(db={
        "InputFeatures": (input_content_df, input_embeddings),
        "Labels": (labels_content_df, labels_embeddings)
    })
    engine.add_program(fit_program)

    # Verify that training completed and parameters were saved
    assert "Loss" in engine.trained_modules, "Model should be trained and saved"
    assert len(engine.parameter_store) > 0, "Parameters should be saved after training"
    print(f"✓ Training completed. Saved {len(engine.parameter_store)} parameters")

    # Step 2: Create two different predict statement programs
    predict_program1_str = """
    ?pred Output1(X; z) :- SmallNet(X; z) .
    """
    predict_program1 = parse_and_transform_str(predict_program1_str)

    predict_program2_str = """
    ?pred Output2(X; z) :- SmallNet(X; z) .
    """
    predict_program2 = parse_and_transform_str(predict_program2_str)

    # Step 3: Call add_program with each predict statement
    print("\nRunning first predict (Output1)...")
    output1 = engine.add_program(predict_program1)
    print(f"Output1 embeddings shape: {output1.embedding_shapes}")
    print(f"Output1 embeddings:\n{output1.embeddings[0]}")

    print("\nRunning second predict (Output2)...")
    output2 = engine.add_program(predict_program2)
    print(f"Output2 embeddings shape: {output2.embedding_shapes}")
    print(f"Output2 embeddings:\n{output2.embeddings[0]}")

    # Step 4: Verify both predictions are identical (because weights are saved)
    assert output1.embedding_shapes == output2.embedding_shapes, "Embedding shapes should match"
    assert torch.allclose(output1.embeddings[0], output2.embeddings[0], atol=1e-6), \
        "Predictions should be identical (using saved weights)"
    assert output1.content.equals(output2.content), "Content DataFrames should be identical"

    print("\n✓ SUCCESS: Both predictions (Output1 and Output2) are identical, confirming weights are saved and reused correctly!")

# %%
if __name__ == "__main__":

    # Test: predict -> fit -> predict updates weights as expected

    # Prepare deterministic data
    input_content_df = pd.DataFrame({"X": [0, 1, 2]})
    input_embeddings = torch.tensor([[0.1, -0.2, 0.3],
                                    [0.2, 0.0, 0.8],
                                    [0.5, 0.3, -0.5]], dtype=torch.float32)
    labels_content_df = pd.DataFrame({"X": [0, 1, 2]})
    labels_embeddings = torch.tensor([[1.0, 0.0],
                                     [0.0, 1.0],
                                     [1.0, 0.0]], dtype=torch.float32)

    # Create a new engine instance
    eng = Engine(db={
        "InputFeatures": (input_content_df, input_embeddings),
        "Labels": (labels_content_df, labels_embeddings)
    })

    program_str = """
    Encoder(X; Linear(3, 2)(z_enc)) :- InputFeatures(X; z_enc) .
    ?fit <epochs=5, batch_size=3> Loss(; MSELoss()(z_enc, z_label)) :- Encoder(X; z_enc), Labels(X; z_label) .
    """
    program = parse_and_transform_str(program_str)
    eng.add_program(program)

    # Predict before fitting
    out_before = eng.add_program(parse_and_transform_str("?pred Out1(X; z) :- Encoder(X; z) ."))
    first_pred = out_before.embeddings[0].clone().detach()

    # Fit again for 5 epochs (should update weights)
    eng.add_program(
        parse_and_transform_str(
            "?fit <epochs=5, batch_size=3> Loss(; MSELoss()(z_enc, z_label)) :- Encoder(X; z_enc), Labels(X; z_label) ."
        )
    )

    # Predict again after fitting
    out_after = eng.add_program(parse_and_transform_str("?pred Out2(X; z) :- Encoder(X; z) ."))
    second_pred = out_after.embeddings[0].clone().detach()

    # Predictions should not be exactly the same, weights must have updated
    assert not torch.allclose(first_pred, second_pred, atol=1e-5), \
        "Predictions before and after fit should differ because weights were updated"

    print("✓ Test passed: predict->fit->predict yields updated weights and changed predictions.")

# %%
if __name__ == "__main__":
    eng.parameter_store

# %%
if __name__ == "__main__":
    print(first_pred, "\n", second_pred)
    draw_tg(eng.term_graphs["global"])

# %% [markdown]
# ## Check if Function or Module Exists

# %%
@patch
def relation(self: Engine, name: str):
    """Return the EmbeddedRelation for a named rule from the last fit/predict run."""
    if self._last_module is None:
        raise RuntimeError("No relation data available. Run fit or predict first.")
    tg = self._last_module_tg
    cache = self._last_module._cache_forward
    node_id = tg.get_node_by_symbol(name)
    if node_id is None or node_id not in cache:
        cached_nodes = set(cache.keys())
        available = sorted(
            sym for sym, nid in tg.symbol_to_node.items()
            if nid in cached_nodes
        )
        raise KeyError(
            f"Relation '{name}' not found. "
            f"Available relations from last run: {available}"
        )
    return cache[node_id]


@patch
def _function_or_nn_module_exists(self: Engine, name: str) -> str | None:
    """
    Checks if a function or nn.Module class named `name` exists in Python global scope, 
    is a built-in, in torch.nn, or is a user-defined subclass of nn.Module.
    Returns: the fully qualified name as str (e.g., 'torch.nn.ReLU', 'builtins.len', 'my_module.Linear'), or None if not found.

    Note: Modern Python may implement builtins like `len` as types.BuiltinMethodType (not just BuiltinFunctionType),
    so we expand the check to include BuiltinMethodType as well.
    """

    # 0. RelNN DSL-native callable ops (implemented by the runtime compiler, not Python/torch.nn)
    # This lets the parser validate curried calls like `view(h, d/h)(z)` without hardcoding
    # allowlists in the parser.
    if name in {"view", "sqrt", "transpose"}:
        return f"relann.dsl.{name}"

    # 1. Check builtins (functions only)
    if hasattr(builtins, name):
        obj = getattr(builtins, name)
        # Accept both functions and builtin functions (e.g., len, print, sorted)
        # and also types.BuiltinMethodType for completeness
        if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType, types.BuiltinMethodType)):
            return f"builtins.{name}"

    # 2. Check globals of current scope (user-defined functions/classes)
    frame = sys._getframe(1)
    engine_module = sys.modules[__name__]
    
    # Check calling frame's globals
    if name in frame.f_globals:
        obj = frame.f_globals[name]
        if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType, types.BuiltinMethodType)):
            modname = getattr(obj, "__module__", "__main__")
            return f"{modname}.{name}"
        if isinstance(obj, type):
            modname = getattr(obj, "__module__", "__main__")
            return f"{modname}.{name}"
    
    # Also check this module's (engine.py) globals for imported modules
    if hasattr(engine_module, name):
        obj = getattr(engine_module, name)
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            modname = getattr(obj, "__module__", "__main__")
            return f"{modname}.{name}"

    # 3. Check torch.nn module by attribute
    if hasattr(nn, name):
        obj = getattr(nn, name)
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            return f"torch.nn.{name}"
        if callable(obj):  # fallback for callable things (not usually needed for torch.nn, but safe)
            return f"torch.nn.{name}"

    # 4. Check for user-defined subclasses of nn.Module (by iterating all globals)
    for obj in frame.f_globals.values():
        if isinstance(obj, type) and obj.__name__ == name and issubclass(obj, nn.Module):
            modname = getattr(obj, "__module__", "__main__")
            return f"{modname}.{name}"
    
    # Also check engine module's globals for imported modules
    for obj in engine_module.__dict__.values():
        if isinstance(obj, type) and obj.__name__ == name and issubclass(obj, nn.Module):
            modname = getattr(obj, "__module__", "__main__")
            return f"{modname}.{name}"

    # Not found
    return None

# %%
@patch
def params(self: Engine):
    """Return an OrderedDict of {clean_name: nn.Parameter} from the parameter store."""
    from collections import OrderedDict
    if not self.parameter_store:
        raise RuntimeError("No parameters available. Run fit or define first.")
    return OrderedDict(
        (_clean_param_fqn(k), v) for k, v in self.parameter_store.items()
    )

# %%
if __name__ == "__main__":
    # Inline test for Engine._function_or_nn_module_exists
    engine = Engine()

    assert engine._function_or_nn_module_exists("len") == "builtins.len"
    assert engine._function_or_nn_module_exists("print") == "builtins.print"
    assert engine._function_or_nn_module_exists("sorted") == "builtins.sorted"
    assert engine._function_or_nn_module_exists("NotARealBuiltin") is None

    assert engine._function_or_nn_module_exists("ReLU") == "torch.nn.ReLU"
    assert engine._function_or_nn_module_exists("Linear") == "torch.nn.Linear"
    assert engine._function_or_nn_module_exists("Module") == "torch.nn.Module"

    assert engine._function_or_nn_module_exists("init") is None  # nn doesn't have a direct callable called "init"

    class TestClass(nn.Module):
        def __init__(self): super().__init__()
    modname = TestClass.__module__
    assert engine._function_or_nn_module_exists("TestClass") == f"{modname}.TestClass"

    def foo_fn(x): return x
    foo_modname = foo_fn.__module__
    assert engine._function_or_nn_module_exists("foo_fn") == f"{foo_modname}.foo_fn"

    class Plain:
        pass
    assert engine._function_or_nn_module_exists("Plain") == f"{Plain.__module__}.Plain"

    assert engine._function_or_nn_module_exists("not_in_scope") is None

    assert engine._function_or_nn_module_exists("sum") == "builtins.sum"

    bar = 123
    assert engine._function_or_nn_module_exists("bar") is None

    def inner_scope_test():
        def inside_fn(): pass
        # It should now find inside_fn in the current (local) frame
        modname = inside_fn.__module__
        assert engine._function_or_nn_module_exists("inside_fn") is None
    inner_scope_test()

# %% [markdown]
# ## weigths sharing

# %%
if __name__ == "__main__":
    prog_3rules_str = """
    SharedLin = Linear(16,32) .
    SimpleEmbedding(X, Z; SharedLin(Concat(z1, z2))) :- InputData1(X, Y;z1), InputData2(Y, Z;z2) .
    OtherEmbedding(X, W; SharedLin(z3)) :- InputData3(X, W;z3) .

    SimpleEmbedding2(X, Z; Linear(16,32)(Linear(16,32)(Concat(z1, z2)))) :- InputData1(X, Y;z1), InputData2(Y, Z;z2) .
    OtherEmbedding2(X, W; Linear(16,32)(z3)) :- InputData3(X, W;z3) .
    """

    # """ tensor_term_wo_leaves = operator.
    # OtherEmbedding2(X, W; Relu(Linear(16,32)(z3)))
    # OtherEmbedding2(X, W; Lin(Relu(0)(z3)))
    # """
