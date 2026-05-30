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
# %% [markdown]
# # Term Graph
#
# > A module for handling term graphs in the DSL

# %%
import logging
import networkx as nx
from typing import List, Union, Dict, Any, Callable, Optional
from relann.pydantic_classes import (
    Program, Rule, FitStatement, PredictStatement,
    DerivedER, EmbeddedRelation, TensorTerm, TensorOp, ArithTerm,
    Var, Primitive, ComparisonExpression,  ErParam, 
    ERSchema, EmbeddingExpression, FunctionDef, TransformDef, RHS, BoundedRHS, VarTemplated, ContentEncode
)
from relann.column_ref import ColumnRef

import torch
import torch.nn as nn

from fastcore.basics import patch

logger = logging.getLogger(__name__)

# %%
if __name__ == "__main__":

    from relann.parser import parse_and_transform_str

# %%
from relann.pydantic_classes import ALLOWED_AGGREGATIONS

# %% [markdown]
# ## Column Name and Aggregation Utilities

# %%
# Mapping from relational operator symbols to operation names
REL_OP_MAPPING: Dict[str, str] = {",": "join", "|": "union"}

def _colnames(attrs: List[Union[Var, Primitive]], vars_only: bool = True) -> List[str]:
    """
    Args:
        attrs: List of attributes (Var or Primitive)
        vars_only: If True, only return Var names. If False, return all as strings.
    Returns: List of column names
    """
    if not attrs:
        return []
    
    if vars_only:
        return [a.name for a in attrs if isinstance(a, Var)]
    else:
        return [attr.name if isinstance(attr, Var) else str(attr) for attr in attrs]

def get_column_names(attrs: List[Union[Var, Primitive]]) -> List[str]:
    """Extract column names from attributes, including primitives."""
    return _colnames(attrs, vars_only=False)

def find_col_index(attrs: List[Union[Var, Primitive]], name: str) -> Optional[int]:
    """Find the index of a column by name in attributes list (0-based), or None if not found."""
    col_names = _colnames(attrs, vars_only=True)
    try:
        return col_names.index(name)
    except ValueError:
        return None

# %%
def make_agg_fn(base_fn: Callable) -> Callable[[torch.Tensor, Any, int], torch.Tensor]:
    """Wraps an aggregation function to ensure index tensor is on the correct device and dtype."""
    def agg_fn(src: torch.Tensor, index: Any, dim: int = 0, **kwargs) -> torch.Tensor:
        if isinstance(index, torch.Tensor):
            idx = index if (index.device is src.device and index.dtype is torch.long) \
                  else index.to(device=src.device, dtype=torch.long)
        else:
            idx = torch.as_tensor(index, device=src.device, dtype=torch.long)
        return base_fn(src, idx, dim=dim, **kwargs)
    return agg_fn

# %%
def make_func_call_full_name(er: EmbeddedRelation) -> str:
    """Create a full name for a function call from its name and arguments."""
    if not hasattr(er, "arguments") or er.arguments is None:
        raise ValueError(f"{er.name} is not a function call")
    if not er.arguments:
        return er.name
    return f"{er.name}_of_{','.join([arg.name for arg in er.arguments])}"

# %% [markdown]
# ## ColumnRef

# %% [markdown]
# ## Term Graph

# %%
class TermGraph(nx.DiGraph):
    """
    A specialized NetworkX DiGraph for managing term graphs in RelNN.
    Extends nx.DiGraph with methods for adding rules, functions, and merging with existing graphs.
    """
    
    def __init__(self, db: Optional[Dict] = None, debug: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = db if db is not None else {}
        self.symbol_to_node: Dict[str, str] = {}
        self.debug = debug

# %%
@patch
def get_node_by_symbol(self: TermGraph, symbol_name: str, namespace: str = None) -> Optional[str]:
    """Get the actual node name for a given symbol name."""
    node_name = self.symbol_to_node.get(symbol_name)
    if node_name is not None:
        return node_name
    if symbol_name in self.nodes():
        return symbol_name
    return None

@patch
def add_symbol_alias(self: TermGraph, symbol_name: str, node_name: str) -> None:
    """Add an alias mapping from a symbol name to a node name."""
    self.symbol_to_node[symbol_name] = node_name

# %%
@patch
def _add_data_loader_if_needed(self: TermGraph, embedded_relation: EmbeddedRelation) -> None:
    """Add a data loader node if it doesn't already exist in the graph."""
    if embedded_relation.name not in self.nodes():
        self.add_node(embedded_relation.name, 
                        name=embedded_relation.name, 
                        type="data_loader",
                        content_attrs=embedded_relation.content_attrs,
                        embedding_var=embedded_relation.embedding_var,
                        output_schema=get_column_names(embedded_relation.content_attrs))

# %%
@patch
def add_place_holder_node(self: TermGraph, er_param: ErParam) -> None:
    """Add a placeholder node to the term graph."""
    node_name = er_param.name
    if node_name not in self.nodes():
        self.add_node(
            node_name,
            name=node_name,
            type="place_holder",
            er_param=er_param,
            embedding_var=None
        )

# %%
@patch
def normalize_column_names_to_refs(self: TermGraph, 
                                    column_names: List[str], 
                                    embedded_relations: List[EmbeddedRelation],
                                    input_idx: Optional[int] = None) -> List[ColumnRef]:
    """
    Convert column names to ColumnRef objects for one or all embedded relations. 
    If input_idx is None, search all inputs; otherwise, only search that input.
    """
    if not column_names:
        return []
    
    refs = []
    inputs_to_search = [input_idx] if input_idx is not None else range(len(embedded_relations))
    
    for idx in inputs_to_search:
        if idx >= len(embedded_relations):
            continue
        er = embedded_relations[idx]
        for col_name in column_names:
            col_idx = find_col_index(er.content_attrs, col_name)
            if col_idx is not None:
                refs.append(ColumnRef(idx, col_idx))
    return refs

# %%
@patch
def _find_matching_columns(self: TermGraph, column_name: str, 
                            embedded_relations: List[EmbeddedRelation]) -> List[ColumnRef]:
    # currently not in use
    """Find all occurrences of a column name across input relations."""
    return self.normalize_column_names_to_refs([column_name], embedded_relations)

# %%
def draw_tg(tg, label_key="type", drop_keys=["name"], graph_attrs={'size': '10,8'}):
    from stringdale.viz import draw_nx
    return draw_nx(tg, label_key=label_key, drop_keys=drop_keys, graph_attrs=graph_attrs)

# %% [markdown]
# ## add_rule

# %%
@patch
def add_rule(self: TermGraph, rule: Rule, namespace: Optional[str] = None) -> None:
    """
    Add a Rule to the term graph, creating all necessary nodes and edges.
    
    This method processes a Rule by:
    1. Adding data loader nodes for input relations
    2. Creating relational operator nodes (join, union, etc.) if multiple inputs
    3. Adding transformation nodes for neural network operations
    4. Adding aggregation nodes for grouping operations
    5. Mapping the rule's LHS name to the final output node
    
    Args:
        rule: Rule object to add to the graph
        namespace: Reserved for future namespace support (currently unused)
    
    Raises:
        ValueError: If a function call node doesn't exist in the graph
        RuntimeError: If rule still uses BoundedRHS (should be expanded by engine first)
    """
    lhs = rule.lhs
    rhs = rule.rhs

    # Guard: block rule redefinition with a new data source (corrupts the graph).
    # Future alternative: a DSL `del Rule .` statement to explicitly remove a rule before redefining it.
    existing = self.symbol_to_node.get(lhs.name)
    if existing is not None and existing in self.nodes():
        node_type = self.nodes[existing].get("type", "")
        if node_type in ("transformation", "agg", "orderby"):
            if isinstance(rhs, RHS):
                new_sources = set()
                for er in rhs.ers:
                    resolved = self.get_node_by_symbol(er.name, namespace) if not getattr(er, "arguments", None) else None
                    if resolved is None:
                        raise ValueError(
                            f"Rule '{lhs.name}' is already defined. Adding it again "
                            f"with new data source '{er.name}' would corrupt the term graph. "
                            f"To predict on different data, update "
                            f"session.engine.db[\"{er.name}\"] instead."
                        )
                    new_sources.add(resolved)
            return

    # Handle RHS (regular case)
    if isinstance(rhs, RHS):
        embedded_relations = rhs.ers
        if not embedded_relations:
            raise ValueError("RHS must have at least one embedded relation")
        
        rel_ops = rhs.rel_ops or []
        
        # Track resolved relation names to actual node names
        resolved_relation_nodes = []
        
        # Add data loader nodes for RHS relations that are not already in the term graph
        for rhs_item in embedded_relations:
            # If it's a function call, assume the engine already created it
            if getattr(rhs_item, "arguments", None) is not None:
                func_name = make_func_call_full_name(rhs_item)
                # If func_name doesn't exist in the graph, raise error
                existing_node = self.get_node_by_symbol(func_name, namespace)
                if existing_node is None:
                    raise ValueError(f"Function call node {func_name} does not exist in the term graph.")
                resolved_relation_nodes.append(existing_node)
            else:
                # Not a function call; handle as regular ER
                existing_node = self.get_node_by_symbol(rhs_item.name, namespace)
                if existing_node is None:
                    # This is a new data loader or relation
                    self._add_data_loader_if_needed(rhs_item)
                    data_loader_node = rhs_item.name
                else:
                    # Use the existing node instead of creating a new data loader
                    self.add_symbol_alias(rhs_item.name, existing_node)
                    data_loader_node = existing_node
                # If embedding_var is a constant, insert an explicit Zero node after the DataLoader
                if isinstance(rhs_item.embedding_var, (int, float, bool)):
                    zero_node = f"zero_{data_loader_node}"
                    if zero_node not in self.nodes():
                        self.add_node(zero_node, type="zero")
                    self.add_edge(data_loader_node, zero_node)
                    resolved_relation_nodes.append(zero_node)
                else:
                    resolved_relation_nodes.append(data_loader_node)
        
        # Track content schema at this point in the pipeline (for later ColumnRef normalization)
        current_content_attrs: List[Union[Var, Primitive]] = []
        join_conditions_for_schema: List[Dict[str, Any]] = []

        # 1. Relational operator nodes for combining relations (join, union, product, difference)
        if len(embedded_relations) > 1:
            # Map the new rel_op format to the old format for compatibility
            rel_op_name = REL_OP_MAPPING.get(rel_ops[0] if rel_ops else ",", "join")
            rel_op_id = f"{rel_op_name}_{lhs.name}"

            join_conditions_for_schema = rhs.join_conditions if rel_op_name == "join" else []

            # Use pre-computed output_content_attrs from RHS (uses aliased names from join_conditions).
            # This is used later to normalize group_by indices correctly.
            if rel_op_name == "join":
                current_content_attrs = rhs.output_content_attrs
            elif rel_op_name == "union":
                # For union, all RHS relations should have the same schema; keep that schema.
                #  (Union concatenates rows; it doesn't merge columns.)
                current_content_attrs = list(embedded_relations[0].content_attrs or [])
            else:
                raise ValueError(f"Unsupported relational operator type: {rel_op_name!r}")

            # Build node attributes (only add join_conditions for join operations)
            node_attrs = {
                "type": rel_op_name,
                "output_schema": get_column_names(current_content_attrs),
                "input_order": resolved_relation_nodes.copy(),
            }
            if rel_op_name == "join":
                # Pass pre-computed merge_steps and input_schemas for efficient join execution
                node_attrs["merge_steps"] = rhs.merge_steps
                node_attrs["input_schemas"] = rhs.input_schemas
                # Keep join_conditions for backward compatibility during transition
                node_attrs["join_conditions"] = join_conditions_for_schema

            self.add_node(rel_op_id, **node_attrs)

            # Use resolved node names when creating edges
            for resolved_node in resolved_relation_nodes:
                self.add_edge(resolved_node, rel_op_id)
            last_node = rel_op_id
        else:
            last_node = resolved_relation_nodes[0]
            # Single-input case: carry forward the RHS relation's schema as the current content schema.
            current_content_attrs = list(embedded_relations[0].content_attrs or [])
        
        # Insert Selection node after join or single input if filter_expressions exist
        if rhs.filter_expressions:
            selection_id = f"selection_{lhs.name}"
            self.add_node(
                selection_id,
                type="selection",
                filter_expressions=rhs.filter_expressions,
            )
            self.add_edge(last_node, selection_id)
            last_node = selection_id

    elif isinstance(rhs, BoundedRHS):
        raise RuntimeError(
            "BoundedRHS should have been expanded to RHS by engine._expand_bounded_set "
            "before reaching term_graph.add_rule"
        )
    else:
        raise ValueError(f"Unsupported RHS type: {type(rhs)}")
    
    # 2. Transform node for neural network operations
    tensor_term = lhs.embedding_expression.tensor_term
    
    if tensor_term:
        transform_id = f"transformation_{lhs.name}"
        input_er = EmbeddedRelation(
            name=last_node,
            # Use the current (pre-transform) content schema, not the LHS schema.
            # This ensures downstream group_by normalization matches the runtime content_schema.
            content_attrs=current_content_attrs
        )
        
        input_indices = self.normalize_column_names_to_refs(
            get_column_names(lhs.derived_content_attrs),
            [input_er],
            input_idx=0
        )
        
        # Build var_to_input_index mapping: Var name -> RHS ER index (0-based)
        # This maps embedding variables (e.g., z1, z2) to their position in the RHS ER list.
        var_to_input_index_rhs = {}
        if isinstance(rhs, RHS):
            for idx, er in enumerate(rhs.ers):
                # Only Vars participate in var_to_input_index (constants like 0 should be ignored)
                if isinstance(er.embedding_var, Var):
                    var_name = er.embedding_var.name
                    if var_name in var_to_input_index_rhs:
                        # This shouldn't happen in valid DSL, but warn if it does
                        import warnings
                        warnings.warn(
                            f"Multiple ERs in RHS use the same embedding_var '{var_name}'. "
                            f"Using the first occurrence (index {var_to_input_index_rhs[var_name]})."
                        )
                    else:
                        var_to_input_index_rhs[var_name] = idx
        
        # Remap var_to_input_index from RHS ER indices to Join output indices
        # If the parent node is a Join, we need to map RHS ER indices to Join input indices
        # Join outputs embeddings in the same order as its input relations (see Join.forward)
        var_to_input_index = {}
        # Check if parent is a Join by checking if last_node exists and is a join type
        is_join_parent, is_union_parent = False, False
        if last_node and last_node in self.nodes:
            parent_node_data = self.nodes[last_node]
            parent_type = parent_node_data.get("type") if isinstance(parent_node_data, dict) else getattr(parent_node_data, "get", lambda k, d=None: d)("type")
            if parent_type == "join":
                is_join_parent = True
            elif parent_type == "union":
                is_union_parent = True
        
        if isinstance(rhs, RHS) and len(rhs.ers) > 1 and is_join_parent:
            # Parent is a Join - need to remap indices
            # Get the stored input_order from the Join node (this is the actual order Join receives inputs)
            join_node = self.nodes[last_node]
            join_input_order = join_node.get("input_order", resolved_relation_nodes)
            
            # Build mapping: RHS ER index -> Join input index
            # Match RHS ERs to join_input_order by name
            # The key insight: join_input_order contains the resolved node names in the order they were added
            # We need to match each resolved_node to the corresponding RHS ER
            rhs_idx_to_join_idx = {}
            
            # Build mapping by iterating through join_input_order and finding the matching RHS ER
            # This ensures we preserve the Join input order
            matched_rhs_indices = set()  # Track which RHS ERs have been matched
            for join_idx, resolved_node in enumerate(join_input_order):
                for rhs_idx, er in enumerate(rhs.ers):
                    if rhs_idx in matched_rhs_indices:
                        continue  # Skip already matched RHS ERs
                    
                    # Direct name match
                    if er.name == resolved_node:
                        rhs_idx_to_join_idx[rhs_idx] = join_idx
                        matched_rhs_indices.add(rhs_idx)
                        break
                    # Check if resolved_node is an existing node that matches er.name
                    elif isinstance(resolved_node, str):
                        existing_node = self.get_node_by_symbol(er.name, namespace)
                        if existing_node == resolved_node:
                            rhs_idx_to_join_idx[rhs_idx] = join_idx
                            matched_rhs_indices.add(rhs_idx)
                            break
            
            # Remap var_to_input_index: RHS index -> Join output index.
            # When name-based matching fails (e.g. same function called twice yields
            # duplicate ER names), fall back to positional identity mapping.
            for var_name, rhs_idx in var_to_input_index_rhs.items():
                join_idx = rhs_idx_to_join_idx.get(rhs_idx, rhs_idx)
                var_to_input_index[var_name] = join_idx
        elif is_union_parent:
            # Union outputs exactly one embedding (the concatenated first embedding from each input),
            # so any embedding var should map to input index 0.
            var_to_input_index = {var_name: 0 for var_name in var_to_input_index_rhs.keys()}
        else:
            # Not a Join or single input - use RHS indices directly
            var_to_input_index = var_to_input_index_rhs
        
        self.add_node(transform_id, type="transformation",
                    transformation=tensor_term,
                    input_indices=input_indices,
                    var_to_input_index=var_to_input_index,
                    # Transformation passes content through; keep the same content schema.
                    output_schema=get_column_names(current_content_attrs)
                    )
        self.add_edge(last_node, transform_id)
        last_node = transform_id
    
    # 3. Aggregation node
    agg_fn_to_use = lhs.embedding_expression.aggregation_fn
    if agg_fn_to_use:
        from relann.era_operations import get_aggregation_function
        agg_id = f"agg_{lhs.name}"
        #TODO: continue here
        base_fn = get_aggregation_function(agg_fn_to_use)
        
        input_er = EmbeddedRelation(
            name=last_node,
            # Aggregation groups rows of the *pre-aggregation* content, so use current schema.
            content_attrs=current_content_attrs
        )
        
        group_by_refs = self.normalize_column_names_to_refs(
            lhs.group_by_column_names,
            [input_er],
            input_idx=0
        )
        
        self.add_node(agg_id, type="agg",
                    aggregation_name=agg_fn_to_use,
                    aggregation_function=make_agg_fn(base_fn),
                    group_by_refs=group_by_refs,
                    output_schema=lhs.group_by_column_names
                    )
        self.add_edge(last_node, agg_id)
        last_node = agg_id
    
    # Note: Projection is currently handled within the aggregation node.
    # In the future, we could optimize this by replacing agg with a separate type="project" node.
    
    # If debug mode is enabled, add an OrderBy node after the last node
    if self.debug:
        orderby_id = f"orderby_{lhs.name}"
        self.add_node(orderby_id, type="orderby")
        self.add_edge(last_node, orderby_id)
        last_node = orderby_id
    
    # Add the original rule name as an alias to the last node
    self.symbol_to_node[lhs.name] = last_node

# %% [markdown]
# ### Symbol Mapping Tests

# %%
if __name__ == "__main__":
    # Construct a program containing 2 rules using parse_and_transform_str
    prog_3rules_str = """
    SimpleEmbedding(X, Z; Linear(3,4)(Concat(z1, z2))) :- InputData1(X, Y;z1), InputData2(Y, Z;z2) .
    FromSimpleEmbedding(X; Linear(3,4)(Concat(z1, z2))) :- InputData4(X;z1), SimpleEmbedding(X, Z;z2) .
    """
    prog_only_rules = parse_and_transform_str(prog_3rules_str)

    tg = TermGraph()
    tg.add_rule(prog_only_rules.statements[0])
    tg.add_rule(prog_only_rules.statements[1])

    # Check that only the first and third rule-derived nodes are present
    print(f"Term graph nodes: {list(tg.nodes())}")
    print(f"Symbol mappings: {tg.symbol_to_node}")

    # Check lookup by symbol for each present rule
    for symbol in ["SimpleEmbedding", "FromSimpleEmbedding"]:
        node = tg.get_node_by_symbol(symbol)
        print(f"  Looking up '{symbol}': {node}")
        assert node == tg.symbol_to_node[symbol], f"{symbol} should map to {tg.symbol_to_node[symbol]}"

    # Check that each actual node name is found directly
    for node_name in tg.symbol_to_node.values():
        node_actual = tg.get_node_by_symbol(node_name)
        print(f"  Looking up '{node_name}': {node_actual}")
        assert node_actual == node_name, f"{node_name} should be found directly"

    print("✓ Multi-statement symbol mapping test completed!")

# %%
if __name__ == "__main__":
    from pprint import pprint

# %%
if __name__ == "__main__":
    pprint(dict(tg.symbol_to_node))

    draw_tg(tg)

# %% [markdown]
# ### Fit Predict

# %%
@patch
def add_fit_statement(self: TermGraph, fit_stmt: FitStatement) -> None:
    """Add a FitStatement to the term graph."""
    fit_id = f"fit_{fit_stmt.rule.lhs.name}"
    input_er = EmbeddedRelation(
        name=fit_stmt.rule.lhs.name,
        content_attrs=fit_stmt.rule.lhs.derived_content_attrs
    )
    
    input_indices = self.normalize_column_names_to_refs(
        get_column_names(fit_stmt.rule.lhs.derived_content_attrs),
        [input_er],
        input_idx=0
    )
    
    self.add_node(fit_id, type="fit",
                rule=fit_stmt.rule,
                input_indices=input_indices,
                output_schema=get_column_names(fit_stmt.rule.lhs.derived_content_attrs)
                )
    self.add_edge(fit_stmt.rule.lhs.name, fit_id)

@patch
def add_predict_statement(self: TermGraph, predict_stmt: PredictStatement) -> None:
    """Add a PredictStatement to the term graph."""
    predict_id = f"predict_{predict_stmt.rule.lhs.name}"
    input_er = EmbeddedRelation(
        name=predict_stmt.rule.lhs.name,
        content_attrs=predict_stmt.rule.lhs.derived_content_attrs
    )
    
    input_indices = self.normalize_column_names_to_refs(
        get_column_names(predict_stmt.rule.lhs.derived_content_attrs),
        [input_er],
        input_idx=0
    )
    
    self.add_node(predict_id, type="predict",
                rule=predict_stmt.rule,
                input_indices=input_indices,
                output_schema=get_column_names(predict_stmt.rule.lhs.derived_content_attrs)
                )
    self.add_edge(predict_stmt.rule.lhs.name, predict_id)

# %% [markdown]
# ### utils for creating TGs (for tests)

# %%
# Legacy function for backward compatibility - now uses TermGraph internally
def program_to_graph(program: Program) -> TermGraph:
    """Convert a Program into a TermGraph representation."""
    tg = TermGraph()
    for stmt in program.statements:
        if isinstance(stmt, Rule):
            tg.add_rule(stmt)
        elif isinstance(stmt, FitStatement):
            tg.add_fit_statement(stmt)
        elif isinstance(stmt, PredictStatement):
            tg.add_predict_statement(stmt)
    return tg

# %%
def create_simple_join_program():
    """
    Creates a simple program with one rule that joins two input relations.
    SimpleEmbedding(X,Y,Z; avg(Linear(3,4)(Concat(z1,z2)))) :- InputData1(X,Y;z1), InputData2(Y,Z;z2) .
    """
    from relann.parser import parse_and_transform_str
    
    program_str = "SimpleEmbedding(X,Y,Z; avg(Linear(3,4)(Concat(z1,z2)))) :- InputData1(X,Y;z1), InputData2(Y,Z;z2) ."
    return parse_and_transform_str(program_str)

# %%
if __name__ == "__main__":

    # NOTE: The following Program object is provided for reference/testing purposes only.
    # It is meant to be the expected output of the `create_simple_join_program()` function above.
    # This can be used in tests, or in case the function's behavior changes or breaks in the future,
    # to have a concrete example of the expected object structure for comparison.

    EXPECTED_SIMPLE_JOIN_PROGRAM = Program(
        statements=[
            Rule(
                lhs=DerivedER(
                    name="SimpleEmbedding",
                    template_params=None,
                    derived_content_attrs=[Var(name="X"), Var(name="Y"), Var(name="Z")],
                    embedding_expression=EmbeddingExpression(
                        aggregation_fn="avg",
                        tensor_term=TensorTerm(
                            op=TensorOp(
                                op="Linear",
                                hyper_params=[
                                    ArithTerm(op=None, sons=None, value=3),
                                    ArithTerm(op=None, sons=None, value=4),
                                ],
                            ),
                            sons=[
                                TensorTerm(
                                    op=TensorOp(op="Concat", hyper_params=None),
                                    sons=[
                                        TensorTerm(
                                            op=None, sons=None, value=Var(name="z1")
                                        ),
                                        TensorTerm(
                                            op=None, sons=None, value=Var(name="z2")
                                        ),
                                    ],
                                    value=None,
                                )
                            ],
                            value=None,
                        ),
                    ),
                ),
                rhs=RHS(
                    ers=[
                        EmbeddedRelation(
                            name="InputData1",
                            template_args=None,
                            arguments=None,
                            content_attrs=[Var(name="X"), Var(name="Y")],
                            embedding_var=Var(name="z1"),
                        ),
                        EmbeddedRelation(
                            name="InputData2",
                            template_args=None,
                            arguments=None,
                            content_attrs=[Var(name="Y"), Var(name="Z")],
                            embedding_var=Var(name="z2"),
                        ),
                    ],
                    rel_ops=[","],
                    filter_expressions=[],
                ),
            )
        ]
    )

    create_simple_join_program()

# %%
class ConcatLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        x = xs[0] if len(xs) == 1 else torch.cat(xs, dim=-1)
        self.linear.to(device=x.device, dtype=x.dtype)
        return self.linear(x)

# %% [markdown]
# ### Column Normalization Tests

# %%
if __name__ == "__main__":
    # Test the column normalization functionality
    print("Testing Column Normalization:")
    print("=" * 70)

    # Create a simple program with a join
    tg = TermGraph()

    # Create a rule with a join on column 'Y'
    # InputData1(X, Y) JOIN InputData2(Y, Z)
    # Y is at position 1 in InputData1 and position 0 in InputData2
    simple_program = create_simple_join_program()
    tg.add_rule(simple_program.statements[0])

    print("Term graph nodes:")
    for node_name in tg.nodes():
        node_data = tg.nodes[node_name]
        print(f"  {node_name}:")
        print(f"    type: {node_data.get('type')}")
        print(f"    output_schema: {node_data.get('output_schema')}")
        if node_data.get('type') == 'join':
            join_conditions = node_data.get('join_conditions', [])
            print(f"    join_conditions: {len(join_conditions)} condition(s)")
            for jc in join_conditions:
                print(f"      - key: {jc.get('key_name')}, refs: {jc.get('normalized_refs')}")
    print()

    # Verify the normalization
    join_node = 'join_SimpleEmbedding'
    if join_node in tg.nodes():
        join_data = tg.nodes[join_node]
        join_conditions = join_data.get('join_conditions', [])
        
        print("Detailed join normalization:")
        print(f"  Join conditions: {len(join_conditions)}")
        
        for jc in join_conditions:
            key_name = jc.get('key_name')
            refs = jc.get('normalized_refs', [])
            print(f"\n  Join key '{key_name}':")
            for ref in refs:
                if isinstance(ref, ColumnRef):
                    print(f"    Input {ref.input_idx}, Column {ref.column_idx}")

    print("\n" + "=" * 70)
    print("✓ Column normalization test completed!")
    print("\nKey insight: Join conditions use ColumnRef objects with input_idx and column_idx,")
    print("making them independent of column names. This allows node reuse even")
    print("when different parents reference columns with different names.")
    draw_tg(tg)

# %% [markdown]
# ### Node Reuse Tests

# %%
if __name__ == "__main__":
    # Test node reuse with different column names (the key benefit of normalization!)
    print("\n\nTesting Node Reuse with Different Column Names:")
    print("=" * 70)
    print("This test demonstrates why normalization is crucial:")
    print("Different parents can reference the same join attribute with different names\n")

    # Create a new graph
    tg2 = TermGraph()

    # Create rules using parser

    node_reuse_program_str = """
    BaseRelation(id, value; sum()) :- RawData(id, value) .
    Derived1(user_id, score; sum()) :- BaseRelation(user_id, score), Table1(user_id, extra) .
    Derived2(person_id, amount; sum()) :- BaseRelation(person_id, amount), Table2(person_id, data) .
    """
    prog = parse_and_transform_str(node_reuse_program_str)

    # Add rules to graph
    tg2.add_rule(prog.statements[0])  # BaseRelation
    tg2.add_rule(prog.statements[1])  # Derived1
    tg2.add_rule(prog.statements[2])  # Derived2

    print("Scenario:")
    print("  - BaseRelation has columns (id, value)")
    print("  - Derived1 references it as (user_id, score) and joins on 'user_id'")
    print("  - Derived2 references it as (person_id, amount) and joins on 'person_id'")
    print("\nWith normalization, both joins reference the SAME column by position (index 0),")
    print("not by name, so BaseRelation can be safely reused!\n")

    print("Graph structure:")
    print(f"  Nodes: {list(tg2.nodes())}")
    print()

    # Show the normalized join conditions
    for node_name in ['join_Derived1', 'join_Derived2']:
        if node_name in tg2.nodes():
            node_data = tg2.nodes[node_name]
            print(f"{node_name}:")
            join_conditions = node_data.get('join_conditions', [])
            print(f"  Join conditions: {len(join_conditions)}")
            for jc in join_conditions:
                key_name = jc.get('key_name')
                refs = jc.get('normalized_refs', [])
                print(f"    Key '{key_name}':")
                for ref in refs:
                    if isinstance(ref, ColumnRef):
                        print(f"      -> Input {ref.input_idx}, Column {ref.column_idx}")
            print()

    print("=" * 70)
    print("✓ Node reuse test completed!")
    print("\nNotice how both joins reference BaseRelation's column at index 0,")
    print("even though they use different names ('user_id' vs 'person_id').")
    print("This is the power of normalization - position-based, not name-based!")
    draw_tg(tg2)

# %% [markdown]
# # Induce subgraph

# %%
@patch
def induced_subgraph(self: TermGraph, node_name: str, *, direction: str = "descendants", depth: Optional[int] = None, include_root: bool = True) -> nx.DiGraph:
    """Return the induced subgraph around ``node_name``.

    Args:
        node_name: Symbol or node identifier present in the graph.
        direction: Which neighbours to include. One of ``"descendants"`` (default),
            ``"ancestors"``, or ``"both"``. ``"descendants"`` follows outgoing edges,
            ``"ancestors"`` follows incoming edges.
        depth: Optional non-negative depth limit. ``None`` means unlimited depth.
        include_root: Whether to include ``node_name`` itself in the induced subgraph.

    Returns:
        A new ``nx.DiGraph`` containing the induced subgraph on the collected nodes.
    """
    if direction not in {"descendants", "ancestors", "both"}:
        raise ValueError("direction must be one of {'descendants', 'ancestors', 'both'}")

    if node_name not in self and node_name not in self.symbol_to_node:
        raise KeyError(f"Node '{node_name}' not found in this term graph")

    # Resolve symbol aliases to actual node names.
    resolved_node = self.get_node_by_symbol(node_name)
    if resolved_node is None:
        resolved_node = node_name

    def bfs(start_node: str, forward: bool) -> set[str]:
        """Breadth-first search with optional depth limit."""
        visited: set[str] = set()
        frontier = {start_node}
        current_depth = 0
        while frontier and (depth is None or current_depth < depth):
            next_frontier: set[str] = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                neighbours = self.successors(node) if forward else self.predecessors(node)
                for nb in neighbours:
                    if nb not in visited:
                        next_frontier.add(nb)
            frontier = next_frontier
            current_depth += 1
        return visited

    nodes_to_include: set[str] = set()
    if include_root:
        nodes_to_include.add(resolved_node)

    if direction in {"descendants", "both"}:
        desc = bfs(resolved_node, forward=True)
        if not include_root:
            desc.discard(resolved_node)
        nodes_to_include |= desc
    if direction in {"ancestors", "both"}:
        anc = bfs(resolved_node, forward=False)
        if not include_root:
            anc.discard(resolved_node)
        nodes_to_include |= anc

    if not nodes_to_include:
        raise ValueError(f"No nodes collected when starting from '{node_name}'")

    # Copy to detach from original graph.
    return self.subgraph(nodes_to_include).copy()


@patch
def mark_phase(self: TermGraph, symbol: str, phase: str) -> None:
    """Mark a symbol's *terminal output node* as belonging to fit/predict.

    This is visualization metadata only. It annotates the resolved node with a
    ``phases`` list.

    Args:
        symbol: A rule output symbol (e.g. ``"Loss"``, ``"Output"``) or node id.
        phase: ``"fit"`` or ``"predict"``.
    """
    if phase not in {"fit", "predict"}:
        raise ValueError("phase must be one of {'fit', 'predict'}")

    # Resolve symbol aliases to actual node names (e.g. OrderBy node when debug=True).
    node_id = self.get_node_by_symbol(symbol)
    if node_id is None:
        node_id = symbol

    if node_id not in self.nodes:
        raise KeyError(f"Node '{symbol}' not found in this term graph")

    nd = self.nodes[node_id]
    phases = nd.get("phases", [])

    if phase not in phases:
        phases.append(phase)

    # Dedupe + stable order.
    canonical = ["fit", "predict"]
    nd["phases"] = [p for p in canonical if p in set(phases)]

# %%
if __name__ == "__main__":
    # Example: parse and build program, then induce termgraph for "FromSimpleEmbedding"

    # Build a small program with join and chaining, as in the examples
    example_prog_str = """
    SimpleEmbedding(X, Z; Linear(3,4)(Concat(z1, z2))) :- InputData1(X, Y;z1), InputData2(Y, Z;z2) .
    OtherEmbedding(X, W; Linear(5,2)(z3)) :- InputData3(X, W;z3) .
    FromSimpleEmbedding(X; Linear(3,4)(Concat(z1, z2))) :- InputData4(X;z1), SimpleEmbedding(X, Z;z2) .
    """
    prog = parse_and_transform_str(example_prog_str)
    tg = TermGraph()
    tg.add_rule(prog.statements[0])  # SimpleEmbedding
    tg.add_rule(prog.statements[1])  # OtherEmbedding
    tg.add_rule(prog.statements[2])  # FromSimpleEmbedding

    # Print nodes and symbol table
    print(f"Full term graph nodes: {list(tg.nodes())}")
    print(f"Symbol mapping: {tg.symbol_to_node}")

    # Induce subgraph for FromSimpleEmbedding and print contents
    sg = tg.induced_subgraph("FromSimpleEmbedding", direction="ancestors", include_root=True)
    print(f"Induced subgraph nodes (ancestors of FromSimpleEmbedding): {set(sg.nodes)}")
    print(f"Induced subgraph edges: {set(sg.edges)}")

    # Inline test assertions for the induced subgraph:
    from_simple_node = tg.symbol_to_node["FromSimpleEmbedding"]
    assert from_simple_node in sg.nodes, "Induced subgraph should include the FromSimpleEmbedding derived node"

    # It should *at least* include the transformation node and upstream embedding chain
    assert any(n.startswith("transformation_") for n in sg.nodes), "Induced subgraph should contain transformation node(s)"
    assert any("SimpleEmbedding" in n for n in sg.nodes), "Should contain upstream embedding node(s)"

    # The inputs to both FromSimpleEmbedding and upstream SimpleEmbedding are present
    assert "InputData4" in sg.nodes, "Should include left input node (InputData4) for FromSimpleEmbedding"
    assert any(inp in sg.nodes for inp in {"InputData1", "InputData2"}), "Should include upstream input data nodes"

    print("✓ induced_subgraph FromSimpleEmbedding example passed.")

# %%
if __name__ == "__main__":
    print("Induced subgraph for OtherEmbedding (ancestors):")
    draw_tg(tg.induced_subgraph("OtherEmbedding", direction="ancestors", include_root=True))

# %% [markdown]
# # Preety Draw Term Graph

# %%
# Example usage of the new pretty visualization function
# This creates SQL query plan-like visualizations with:
# - Join nodes showing join keys like "join_Y" or "join_X"  
# - Transformation nodes showing the embedding expression text (e.g., "Linear(3,4)(Concat(z1,z2))")
# - Aggregation nodes showing which columns are aggregated (e.g., "avg(X,Y,Z)")
# - Color-coded nodes by type for easy visual identification

# Example: visualize the term graph with the new function
# preety_draw_tg(tg, graph_attrs={'size': '10,8'})

# %%
def format_tensor_term(tterm, var_to_input_index=None):
    """Format a TensorTerm as a string representation."""
    if tterm is None: return ""
    if tterm.value is not None:
        if isinstance(tterm.value, Var):
            return tterm.value.name
        if isinstance(tterm.value, VarTemplated):
            params_str = ",".join(p.name if isinstance(p, Var) else str(p) for p in tterm.value.template_params)
            return f"{tterm.value.name}<{params_str}>"
        if isinstance(tterm.value, ContentEncode):
            ce = tterm.value
            parts = []
            for it in ce.items:
                if it.encoder_name:
                    if it.encoder_params:
                        hp_s = ",".join(format_arith_term(h) for h in it.encoder_params)
                        parts.append(f"{it.encoder_name}({hp_s})({it.column.name})")
                    else:
                        parts.append(f"{it.encoder_name}({it.column.name})")
                else:
                    parts.append(it.column.name)
            return "[" + ", ".join(parts) + "]"
        return str(tterm.value)
    if tterm.op is not None:
        op_name = tterm.op.op
        hyper_params_str = ""
        if tterm.op.hyper_params:
            hyper_vals = []
            for hp in tterm.op.hyper_params:
                if hp.value is not None: hyper_vals.append(str(hp.value))
                elif hp.sons: hyper_vals.append(f"({format_arith_term(hp)})")
            if hyper_vals: hyper_params_str = "(" + ",".join(hyper_vals) + ")"
        if tterm.sons:
            sons_str = ",".join(format_tensor_term(son, var_to_input_index) for son in tterm.sons)
            return f"{op_name}{hyper_params_str}({sons_str})" if hyper_params_str else f"{op_name}({sons_str})"
        return f"{op_name}{hyper_params_str}" if hyper_params_str else op_name
    return ""

def format_arith_term(aterm):
    """Format an ArithTerm as a string."""
    if aterm.value is not None:
        return aterm.value.name if isinstance(aterm.value, Var) else str(aterm.value)
    if aterm.op and aterm.sons:
        return f"({aterm.op.join(format_arith_term(s) for s in aterm.sons)})"
    return ""

def format_join_keys(join_conditions, node_data=None):
    """Format join conditions for display (key names from join_conditions)."""
    if not join_conditions:
        return "join"
    key_names = []
    for jc in join_conditions:
        if isinstance(jc, dict):
            key = jc.get("key_name")
            if key:
                key_names.append(key)
        elif isinstance(jc, (tuple, list)) and len(jc) == 2:
            key_names.append(str(jc[0]))
    return f"⨝_{','.join(key_names)}" if key_names else "join"

def format_aggregation_info(node_data):
    """Format aggregation information for display."""
    agg_name = node_data.get("aggregation_name", "agg")
    group_by_refs = node_data.get("group_by_refs", [])
    group_by_indices = node_data.get("group_by_indices", [])
    group_by_keys = node_data.get("group_by_keys", [])
    group_cols = []
    if group_by_keys:
        group_cols = group_by_keys if isinstance(group_by_keys, list) else [group_by_keys]
    elif group_by_refs:
        group_cols = [f"col{ref.column_idx}" if hasattr(ref, 'column_idx') else str(ref) for ref in group_by_refs]
    elif group_by_indices:
        group_cols = [f"col{ref.column_idx}" if hasattr(ref, 'column_idx') else str(ref) for ref in group_by_indices]
    if group_cols:
        return f"π_{agg_name}({','.join(group_cols)})"
    return f"{agg_name}()"

def format_node_label_for_relnn(g, node, label_key=None, drop_keys=None, symbol_to_node=None):
    """
    Shorter, more readable version of node label formatting, reducing symbol_name duplication (using context7).
    """
    drop_keys = (drop_keys or []) + ([label_key] if label_key else [])

    # Fast reverse lookup for symbol name (context7: one-pass lookup)
    symbol_name = None
    if symbol_to_node:
        symbol_name = next((sym for sym, n in symbol_to_node.items() if n == node), None)

    nd = g.nodes[node]
    ntype = nd.get("type", "unknown")
    label = None

    if ntype == "join":
        label = format_join_keys(nd.get("join_conditions", []))
    elif ntype == "transformation":
        trans = nd.get("transformation")
        if trans:
            var2idx = nd.get("var_to_input_index", {})
            expr = format_tensor_term(trans, var2idx)
            label = f"Transform: {expr}" if expr else "Transform"
        else:
            label = "Transform"
    elif ntype == "agg":
        label = format_aggregation_info(nd)
    elif ntype == "data_loader":
        name = nd.get("name", node)
        schema = nd.get("output_schema", [])
        label = f"🛢 {name}({', '.join(schema)})" if schema else f"🛢 {name}"
    elif ntype == "selection":
        label = "Filter" if nd.get("filter_expressions", []) else "Selection"
    elif ntype == "orderby":
        label = "OrderBy"
    elif ntype == "fit":
        label = "Fit"
    elif ntype == "predict":
        label = "Predict"
    else:
        label = str(nd[label_key]) if label_key and label_key in nd else ntype

    # PHASES handling, add ## markers for Fit/Predict/Fit+Predict for visual emphasis
    phases = nd.get("phases", [])
    phase_label = None
    if phases:
        if phases == ["fit"]:
            phase_label = "#Fit#"
        elif phases == ["predict"]:
            phase_label = "#Predict#"
        elif ("fit" in phases) and ("predict" in phases):
            phase_label = "#Fit+Predict#"
        else:
            phase_label = "+".join(str(p) for p in phases)

    if symbol_name:
        return f"⟦{symbol_name}⟧\\n{label}\\n{phase_label}" if phase_label else f"⟦{symbol_name}⟧\\n{label}"

    return f"{label}\\n{phase_label}" if phase_label else label

def preety_draw_tg(
    tg,
    label_key="type",
    drop_keys=None,
    graph_attrs=None,
    node_attrs=None,
    edge_attrs=None,
    direction='TB',
    format='svg',
    ret_dot=False,
    **kwargs
):
    """
    Draw a RelNN term graph with SQL query plan-like visualization.

    Pretty styling (context7), colored by node type, concise symbol name handling.
    """
    from stringdale.viz import (
        graph_to_graphviz_spec,
        draw_graphviz,
        check_graphviz_installed,
        display_in_ipython,
    )
    if not check_graphviz_installed():
        return None

    symbol_to_node = getattr(tg, 'symbol_to_node', None)
    node_kwargs = []
    for node in tg.nodes():
        label = format_node_label_for_relnn(
            tg, node, label_key=label_key, drop_keys=drop_keys, symbol_to_node=symbol_to_node
        )
        node_kwargs.append({'name': node, 'label': label})

    edge_kwargs = [{'tail_name': e[0], 'head_name': e[1], 'label': ''} for e in tg.edges()]

    # Default attributes, SQL-style
    graph_attrs = {
        'bgcolor': '#f8f9fa',
        'fontname': 'Arial',
        'fontsize': '12',
        'ranksep': '0.8',
        'nodesep': '0.5',
        'size': '10,8',
        **(graph_attrs or {}),
    }

    node_type_colors = {
        'data_loader': {'fillcolor': '#e3f2fd', 'color': '#1976d2'},      # Blue
        'join': {'fillcolor': '#fff3e0', 'color': '#f57c00'},             # Orange
        'transformation': {'fillcolor': '#f3e5f5', 'color': '#7b1fa2'},   # Purple
        'agg': {'fillcolor': '#e8f5e9', 'color': '#388e3c'},              # Green
        'selection': {'fillcolor': '#fce4ec', 'color': '#c2185b'},        # Pink
        'orderby': {'fillcolor': '#fff9c4', 'color': '#f9a825'},          # Yellow
    }
    # Apply coloring by type
    for node_kw in node_kwargs:
        ntype = tg.nodes[node_kw['name']].get('type', 'unknown')
        colors = node_type_colors.get(ntype, {'fillcolor': '#ececff', 'color': '#9370db'})
        node_kw.update(colors)
        node_kw['style'] = 'filled,rounded'
        node_kw['shape'] = 'box'
        node_kw['fontname'] = 'Arial'
        node_kw['fontsize'] = '11'

    node_attrs = {'style': 'filled,rounded', 'shape': 'box', **(node_attrs or {})}
    edge_attrs = {'color': '#666666', 'arrowsize': '0.8', **(edge_attrs or {})}

    dot = draw_graphviz(
        node_kwargs,
        edge_kwargs,
        name=None,
        direction=direction,
        format=format,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
        graph_attrs=graph_attrs,
        **kwargs
    )
    if ret_dot:
        return dot
    else:
        display_in_ipython(dot)

# %%
if __name__ == "__main__":
    # Test the new visualization function
    # Create a simple term graph to visualize
    test_tg = TermGraph()
    simple_program = create_simple_join_program()
    test_tg.add_rule(simple_program.statements[0])

    # Visualize with the new pretty function
    print("Visualizing term graph with draw_tg:")
    preety_draw_tg(test_tg, graph_attrs={'size': '10,8'})

# %%
if __name__ == "__main__":
    pprint(tg.symbol_to_node)

    preety_draw_tg(sg)

    # draw with attributes
    draw_tg(sg)
