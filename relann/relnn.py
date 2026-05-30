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
# %% [markdown]
# # RelNN — **Term Graph → PyTorch nn.Module**
#
# This notebook provides an implementation for Step 1:
# Converting a **NetworkX term graph** (RelNN ERA graph) into a concrete **PyTorch nn.Module**
# with an instantiate() pass that pre-computes relational ops and a forward() pass that runs the neural ops.
#
# Key features:
# - ERA ops executed in **topological order** and **cached** (instantiate())
# - Torch ops executed on top of cached results (forward())
# - **cuDF** support (GPU) with **pandas** fallback
#
# - ERA operators included: EmbeddedRelation, Transformation, Join, DataLoader, Aggregation, Project
#
# > This is designed to align with the ERA semantics described in story.md.

# %%

from dataclasses import dataclass
import logging
import typing as T
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
import inspect
import re

from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Protocol, Tuple, Any, runtime_checkable

# %%
try:
    import cudf
    _HAS_CUDF = True
except Exception:
    cudf = None
    _HAS_CUDF = False

from relann.term_graph import program_to_graph, create_simple_join_program
from relann.column_ref import ColumnRef
from relann.era_operations import (
    Join,
    EmbeddedRelation,
    Transformation,
    DataLoader,
    Aggregation,
    Project,
    Selection,
    Zero,
    OrderBy,
    Union,
    Rename,
    ExecutionContext,
)

from relann.term_graph import preety_draw_tg, draw_tg

logger = logging.getLogger(__name__)


class RelNNNodeError(Exception):
    """Raised when evaluation fails at a specific term-graph node; preserves node_id, mode, op type, and optional rule."""

    def __init__(
        self,
        node_id: str,
        mode: str,
        op_type: str,
        original_msg: str,
        rule: Optional[Any] = None,
    ) -> None:
        self.node_id = node_id
        self.mode = mode
        self.op_type = op_type
        self.original_msg = original_msg
        self.rule = rule
        super().__init__(self._message())

    def _message(self) -> str:
        msg = (
            f"RelNN evaluation failed at node '{self.node_id}' "
            f"(mode={self.mode}, op={self.op_type}): {self.original_msg}"
        )
        if self.rule is not None:
            msg += f" [rule={self.rule}]"
        return msg


class RelNNModuleLookupError(KeyError):
    """Raised when resolving an operator module by graph node id fails."""

# %% [markdown]
# ## RelNN

# %%
@runtime_checkable
class ParameterLoader(Protocol):
    """Abstraction for loading saved parameters into an nn.Module.

    Engine implements this so RelNN never depends on Engine internals.
    """

    def load_into(self, module: nn.Module, node_id: str, node: dict) -> None: ...

# %%
_OPERATOR_CLASSES: dict[str, type] = {
    "embedded_relation": EmbeddedRelation,
    "transformation":    Transformation,
    "join":              Join,
    "data_loader":       DataLoader,
    "aggregation":       Aggregation,
    "agg":               Aggregation,
    "project":           Project,
    "selection":         Selection,
    "zero":              Zero,
    "orderby":           OrderBy,
    "union":             Union,
    "rename":            Rename,
}


def _is_passthrough_term(term) -> bool:
    """True when *term* is a TensorTerm that simply passes an embedding variable through."""
    from relann.pydantic_classes import TensorTerm, Var
    if not isinstance(term, TensorTerm):
        return False
    sons_empty = term.sons is None or (isinstance(term.sons, (list, tuple)) and len(term.sons) == 0)
    return (
        term.op is None
        and sons_empty
        and hasattr(term, "value")
        and isinstance(term.value, Var)
    )


class RelNN(nn.Module):
    """
    Strict executor for ERA-style term graphs.

    Edge direction is input -> operator (data-flow direction).
    The output/root is the *last* node in topological order (the sink).
    Each node's attributes must contain the exact constructor fields of
    its operator class; unknown node types raise.

    Modules are built lazily on the first call to instantiate(...) or forward(...).
    Each operator is invoked via the ERA interface: .instantiate(sons=...)
    and .__call__(sons=...) (which routes through nn.Module.forward).
    """

    def __init__(self, graph: nx.DiGraph, param_loader: Optional[ParameterLoader] = None) -> None:
        super().__init__()
        self.graph = graph.copy()
        self._param_loader = param_loader

        eval_order = list(nx.topological_sort(self.graph))
        if not eval_order:
            raise ValueError("Empty graph.")
        self._eval_order: List[str] = eval_order
        self.output_node: str = eval_order[-1]

        self._operators: Optional[nn.ModuleDict] = None
        # Graph node ids may contain characters (e.g. ".") that are invalid as
        # torch module names. Keep a stable mapping to ModuleDict-safe keys.
        self._node_to_module_key: Dict[str, str] = {}

        # Precompute ordered inputs per node (static — depends only on graph topology).
        self._node_inputs: Dict[str, List[str]] = {}
        for node_id in self._eval_order:
            node = self.graph.nodes[node_id]
            input_order = node.get("input_order")
            inputs: List[str] = []
            if input_order is not None:
                inputs = [c for c in input_order
                          if c in self.graph and self.graph.has_edge(c, node_id)]
            if not inputs:
                inputs = list(self.graph.predecessors(node_id))
            self._node_inputs[node_id] = inputs

        self._ctx: Optional[ExecutionContext] = None
        # Phase results — also accessible externally for intermediate-result inspection.
        self._cache_instantiate: Dict[str, object] = {}
        self._cache_forward: Dict[str, object] = {}

    # ---------------------------------------------------------------------
    # Operator construction
    # ---------------------------------------------------------------------

    def _make_module_key(self, node_id: str, idx: int) -> str:
        """Build a readable, ModuleDict-safe key while keeping deterministic order."""
        cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(node_id)).strip("_")
        cleaned = re.sub(r"_+", "_", cleaned)
        return f"op_{idx:04d}__{(cleaned or 'node')}"

    def _summarize_node(self, node_id: str) -> str:
        """Compact readable label for one graph node."""
        node = self.graph.nodes[node_id] if node_id in self.graph else {}
        op_type = (node.get("type") or "unknown").lower()
        return f"{op_type}:{node_id}"

    def describe_ops(self, max_entries: int = 20) -> str:
        """
        Human-readable summary of operators (reverse topological for readability).
        """
        if self._operators is None or not self._node_to_module_key:
            return "operators not built yet"
        node_by_key = {v: k for k, v in self._node_to_module_key.items()}
        lines: List[str] = []
        for idx, module_key in enumerate(self._operators.keys()):
            if idx >= max_entries:
                remaining = len(self._operators) - max_entries
                if remaining > 0:
                    lines.append(f"... ({remaining} more ops)")
                break
            node_id = node_by_key.get(module_key, module_key)
            lines.append(f"{module_key} -> {self._summarize_node(node_id)}")
        return "\n".join(lines)

    def module_for_node(self, node_id: str) -> nn.Module:
        """Return the instantiated operator module for a graph node id."""
        module_key = self._node_to_module_key.get(str(node_id))
        if module_key is None:
            raise RelNNModuleLookupError(f"No module key registered for node '{node_id}'")
        if self._operators is None:
            raise RelNNModuleLookupError("Operator modules are not built yet")
        return self._operators[module_key]

    def extra_repr(self) -> str:
        """Keep repr close to native torch style with a tiny header only."""
        return f"output={self._summarize_node(self.output_node)}"

    @staticmethod
    def _inspect_init_args(cls: type) -> Tuple[List[str], List[str]]:
        """
        Return required and optional argument names of cls.__init__,
        excluding 'self' and variadic (*args/**kwargs) parameters.
        """
        sig = inspect.signature(cls.__init__)
        required, optional = [], []
        for name, p in sig.parameters.items():
            if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if p.default is inspect._empty:
                required.append(name)
            else:
                optional.append(name)
        return required, optional

    # ---------------------------------------------------------------------
    # Transformation resolution
    # ---------------------------------------------------------------------

    def _resolve_transformation_callable(self, node_id: str, node: dict) -> nn.Module:
        """Resolve the nn.Module for a transformation node.

        Priority: compiled torch_transformation > callable DSL term > passthrough Identity.
        """
        torch_mod = node.get("torch_transformation")
        if torch_mod is not None:
            if isinstance(torch_mod, nn.Module):
                if self._param_loader is not None:
                    self._param_loader.load_into(torch_mod, node_id, node)
                return torch_mod
            module = torch_mod()
            if self._param_loader is not None:
                self._param_loader.load_into(module, node_id, node)
            return module

        dsl_term = node.get("transformation")
        if callable(dsl_term):
            return dsl_term

        if _is_passthrough_term(dsl_term):
            return nn.Identity()

        raise TypeError(
            f"Node '{node_id}': transformation must be a compiled module, "
            f"callable, or variable passthrough. Got {type(dsl_term)}"
        )

    def _build_transformation_operator(self, node_id: str, node: dict) -> nn.Module:
        """Build a Transformation ERA operator for a graph node."""
        callable_mod = self._resolve_transformation_callable(node_id, node)
        return Transformation(
            transformation=callable_mod,
            output_schema=node.get("output_schema", None),
        )

    def _build_operator_modules(self) -> None:
        """Construct operator modules for every node from graph attributes."""
        if isinstance(self._operators, nn.ModuleDict) and self._node_to_module_key:
            return

        modules: Dict[str, nn.Module] = {}
        self._node_to_module_key = {}
        for idx, node_id in enumerate(self._eval_order):
            node_id_str = str(node_id)
            module_key = self._make_module_key(node_id_str, idx)
            self._node_to_module_key[node_id_str] = module_key

            node = self.graph.nodes[node_id]
            node_type = (node.get("type") or "").lower()

            if node_type == "transformation":
                modules[module_key] = self._build_transformation_operator(node_id_str, node)
            else:
                modules[module_key] = self._build_generic_operator(node_id_str, node, node_type)

        self._operators = nn.ModuleDict(OrderedDict(reversed(list(modules.items()))))

    def _build_generic_operator(self, node_id: str, node: dict, node_type: str) -> nn.Module:
        """Build a non-transformation operator from its registered class and node attrs."""
        op_cls = _OPERATOR_CLASSES.get(node_type)
        if op_cls is None:
            raise NotImplementedError(
                f"Unsupported operator type '{node_type}'. "
                f"Supported: {sorted(_OPERATOR_CLASSES)}"
            )

        required, optional = self._inspect_init_args(op_cls)
        missing = [name for name in required if name not in node]
        if missing:
            raise KeyError(
                f"Node '{node_id}' (type='{node_type}') is missing required "
                f"constructor arguments {missing}. Present keys: {list(node.keys())}"
            )
        kwargs = {name: node[name] for name in (*required, *optional) if name in node}

        try:
            op = op_cls(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Failed to construct {op_cls.__name__} at node '{node_id}' "
                f"with kwargs {kwargs}: {e}"
            ) from e

        if not (hasattr(op, "instantiate") and hasattr(op, "forward")):
            raise TypeError(
                f"Operator {op_cls.__name__} at node '{node_id}' must implement "
                f"'instantiate(sons=...)' and 'forward(sons=...)'."
            )
        return op

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------

    def _run_phase(self, phase: str, ctx: ExecutionContext) -> Dict[str, object]:
        """Evaluate all nodes iteratively in topological order."""
        results: Dict[str, object] = {}
        for node_id in self._eval_order:
            input_results = [results[c] for c in self._node_inputs[node_id]]
            op = self._operators[self._node_to_module_key[node_id]]
            logger.debug("evaluating node_id=%s phase=%s op=%s", node_id, phase, type(op).__name__)
            try:
                if phase == "instantiate":
                    results[node_id] = op.instantiate(sons=input_results, ctx=ctx)
                else:
                    results[node_id] = op(sons=input_results, ctx=ctx)
            except Exception as e:
                node = self.graph.nodes[node_id]
                raise RelNNNodeError(
                    node_id=node_id,
                    mode=phase,
                    op_type=type(op).__name__,
                    original_msg=str(e),
                    rule=node.get("rule"),
                ) from e
        return results

    def instantiate(self, relations: Optional[Dict[str, dict]] = None) -> object:
        ctx = ExecutionContext(relations=relations or {})
        self._ctx = ctx
        self._build_operator_modules()
        self._cache_instantiate = self._run_phase("instantiate", ctx)
        return self._cache_instantiate[self.output_node]

    def forward(self, relations: Optional[Dict[str, dict]] = None) -> object:
        if relations:
            ctx = ExecutionContext(relations=relations)
            self._ctx = ctx
        else:
            ctx = self._ctx
            if ctx is None:
                raise RuntimeError(
                    "forward() called without relations and without prior instantiate(). "
                    "Call instantiate(relations) first or pass relations to forward()."
                )
        self._build_operator_modules()
        self._cache_forward = self._run_phase("forward", ctx)
        return self._cache_forward[self.output_node]

def term_graph_to_module(
    graph: nx.DiGraph,
    param_loader: Optional[ParameterLoader] = None,
    engine: Optional[Any] = None,
) -> RelNN:
    """Wrap an nx.DiGraph term graph as a runnable RelNN module.

    Assumes the caller has already compiled DSL TensorTerms via
    `engine.eval_tensor_terms_on_tg(graph)` — typically done in
    `Engine.fit` / `Engine.predict` before this is reached. `engine` is
    accepted (and inferred from `param_loader` when the caller passed the
    Engine itself) only so future hooks have a reference; the body just
    constructs the `RelNN` and returns.

    The e-graph optimizer that used to live here was removed in `5656608`;
    re-introduction lives in PR #56. If that PR lands, this function will
    grow back its optimize-then-compile branch.
    """
    if engine is None and param_loader is not None and hasattr(param_loader, "eval_tensor_terms_on_tg"):
        engine = param_loader

    return RelNN(graph, param_loader=param_loader)

# %% [markdown]
# # Tests

# %% [markdown]
# ## Single node tree

# %%
if __name__ == "__main__":

    import torch
    import pandas as pd
    import networkx as nx

# %%
if __name__ == "__main__":


    users_df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "age": [25, 31, 42],
    })
    movies_df = pd.DataFrame({
        "movie_id": [10, 11, 12],
        "year": [1998, 2007, 2015],
    })
    ratings_df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "movie_id": [10, 11, 12],
        "rating": [5.0, 3.5, 4.0],
    })

    E = torch.randn(len(ratings_df), 8)  # (N, d)

    relations = {
        "users": {
            "content": users_df,
            "content_schema": list(users_df.columns),
            "embedding_shapes": [],
            "embeddings": None,
        },
        "movies": {
            "content": movies_df,
            "content_schema": list(movies_df.columns),
            "embedding_shapes": [],
            "embeddings": None,
        },
        "ratings": {
            "content": ratings_df,
            "content_schema": list(ratings_df.columns),
            "embedding_shapes": [E.shape],
            "embeddings": [E],
        },
    }

    G = nx.DiGraph()
    G.add_node("ratings_loader", type="data_loader", name="ratings")
    # no edges (single-node graph)

    mod = term_graph_to_module(graph=G)
    preety_draw_tg(G)
    draw_tg(G)

    # seed GLOBAL_EMBEDDED_RELATIONS and run instantiate (no embeddings in result)
    _ = mod.instantiate(relations)

    # forward pass (with embeddings)
    out = mod()

    print(f"DataFrame:\n{out.content}")
    print(f"DataFrame shape: {out.content.shape}")
    print(f"Embeddings: {out.embeddings}")
    print(f"Expected embedding shapes: {out.embedding_shapes}")

    # Assertions
    assert list(out.content.columns) == ["user_id", "movie_id", "rating"], "Content columns should match ratings schema"
    assert out.content.shape == (3, 3), f"Content shape should be (3, 3), got {out.content.shape}"
    assert out.embeddings is not None, "Embeddings should be present"
    assert len(out.embeddings) == 1, f"Should have 1 embedding tensor, got {len(out.embeddings)}"
    assert out.embeddings[0].shape == (3, 8), f"Embedding shape should be (3, 8), got {out.embeddings[0].shape}"
    assert out.embedding_shapes == [torch.Size([3, 8])], f"Embedding shapes should be [torch.Size([3, 8])], got {out.embedding_shapes}"
    assert torch.allclose(out.embeddings[0], E), "Embeddings should match input embeddings"

    mod

# %% [markdown]
# ## Multiple node tree

# %%
if __name__ == "__main__":

    from relann.engine import Engine

# %%
if __name__ == "__main__":


    def materialize_graph_for_tests(g):
        """
        Utility function to apply the same transformations to a test term graph
        as are applied in Engine.fit before module construction. This is used for tests.
        """
        engine = Engine()
        # Typically, you would attach your actual db and symbol table to engine here if needed

        # Since the test graph is already grounded, we can proceed directly.

        # Evaluate tensor terms to create torch modules for transformation nodes
        materialized_g = engine.eval_tensor_terms_on_tg(g)

        return materialized_g

# %%
if __name__ == "__main__":
    # Create the program and convert to graph
    simple_program = create_simple_join_program()
    g = program_to_graph(simple_program)

    # Apply the test utility transformation function
    materialized_g = materialize_graph_for_tests(g)

    # Visualize the graph
    draw_tg(materialized_g)

# %%
if __name__ == "__main__":
    materialized_g.nodes()['transformation_SimpleEmbedding']

# %%
if __name__ == "__main__":

    import pandas as pd
    import torch
    import networkx as nx

# %%
if __name__ == "__main__":


    # InputData1: content schema ["X","Y"] + embeddings z1
    df1 = pd.DataFrame({"X": [1,2,3,4,5],
                        "Y": [10,11,12,13,14]})
    z1 = torch.randn(len(df1), 1)  # z1 dim = 1 (so z1+z2 can match the 3->4 Linear in the figure)

    rel_InputData1 = {
        "content": df1,
        "content_schema": ["X","Y"],
        "embedding_shapes": [z1.shape],
        "embeddings": [z1],
    }

    # InputData2: content schema ["Y","Z"] + embeddings z2
    df2 = pd.DataFrame({"Y": [10,11,12,13,14],
                        "Z": [100,101,102,103,104]})
    z2 = torch.randn(len(df2), 2)  # z2 dim = 2  => z1(1) + z2(2) -> 3 dims total

    rel_InputData2 = {
        "content": df2,
        "content_schema": ["Y","Z"],
        "embedding_shapes": [z2.shape],
        "embeddings": [z2],
    }

    relations = {
        "InputData1": rel_InputData1,
        "InputData2": rel_InputData2,
    }


    # Create graph with join node
    g = nx.DiGraph()
    g.add_node("join", type="join", merge_steps=[{"step": 1, "left_refs": [ColumnRef(0, 1)], "right_refs": [ColumnRef(1, 0)], "key_names": ["Y"]}], input_schemas=[["X", "Y"], ["Y", "Z"]], output_schema=["X", "Y", "Z"])
    g.add_node("dl1", type="data_loader", name="InputData1")
    g.add_node("dl2", type="data_loader", name="InputData2")
    g.add_edge("dl1", "join")
    g.add_edge("dl2", "join")

    mod = term_graph_to_module(g)
    _ = mod.instantiate(relations)

    # forward pass
    out = mod()

    print(f"DataFrame:\n{out.content}")
    print(f"DataFrame shape: {out.content.shape}")
    print(f"Embeddings: {out.embeddings}")
    print(f"Expected embedding shapes: {out.embedding_shapes}")

# %%
if __name__ == "__main__":

    mod

# %% [markdown]
# ## Extended tests for ERA components & wrappers

# %%
if __name__ == "__main__":
    import pandas as pd
    import torch
    import torch.nn as nn
    import networkx as nx
    from relann.term_graph import ConcatLinear

# %%
if __name__ == "__main__":
    #| exec_doc

    torch.manual_seed(0)

    df1 = pd.DataFrame({"X": [1,2,3,4,5],
                        "Y": [10,11,12,13,14]})
    z1 = torch.randn(len(df1), 1)

    rel_InputData1 = {
        "content": df1,
        "content_schema": ["X","Y"],
        "embedding_shapes": [z1.shape],
        "embeddings": [z1],
    }

    df2 = pd.DataFrame({"Y": [10,11,12,13,14],
                        "Z": [100,101,102,103,104]})
    z2 = torch.randn(len(df2), 2)

    rel_InputData2 = {
        "content": df2,
        "content_schema": ["Y","Z"],
        "embedding_shapes": [z2.shape],
        "embeddings": [z2],
    }

    relations = {"InputData1": rel_InputData1, "InputData2": rel_InputData2}

    def show_out(title, out):
        print(f"\n=== {title} ===")
        print("DataFrame:")
        print(out.content)
        print("DataFrame shape:", out.content.shape)
        if hasattr(out, "embeddings"):
            shapes = [tuple(t.shape) for t in out.embeddings]
            print("Embeddings shapes:", shapes)
            outputs = [tuple(t) for t in out.embeddings]
            print("Embeddings outputs:", outputs)
            
        if hasattr(out, "embedding_shapes"):
            print("Expected embedding shapes:", out.embedding_shapes)

    def total_embed_dim(embed_list):
        return 0 if not embed_list else sum(e.shape[1] for e in embed_list)

    # 1) DataLoader (single node) -----------------------------------------------
    g = nx.DiGraph()
    g.add_node("dl1", type="data_loader", name="InputData1")  # output node = dl1
    mod = term_graph_to_module(g)
    mod.instantiate(relations)
    out = mod()
    show_out("DataLoader(InputData1)", out)
    assert list(out.content.columns) == ["X","Y"]
    assert total_embed_dim(out.embeddings) == 1

    # 2) Project(['X']) — wire as dl1 → proj -------------------------------------
    g = nx.DiGraph()
    g.add_node("proj", type="project", project_keys=["X"])  # output node = proj
    g.add_node("dl1",  type="data_loader", name="InputData1")
    g.add_edge("dl1", "proj")  # input → op
    mod = term_graph_to_module(g)
    mod.instantiate(relations)
    out = mod()
    show_out("Project(['X']) on InputData1", out)
    assert list(out.content.columns) == ["X"]
    assert total_embed_dim(out.embeddings) == 1

    # 3) Join on 'Y' — wire as (dl1, dl2) → join --------------------------------
    g = nx.DiGraph()
    g.add_node("join", type="join", merge_steps=[{"step": 1, "left_refs": [ColumnRef(0, 1)], "right_refs": [ColumnRef(1, 0)], "key_names": ["Y"]}], input_schemas=[["X", "Y"], ["Y", "Z"]], output_schema=["X","Y","Z"])  # output = join
    g.add_node("dl1",  type="data_loader", name="InputData1")
    g.add_node("dl2",  type="data_loader", name="InputData2")
    g.add_edge("dl1", "join")  # input → op
    g.add_edge("dl2", "join")
    mod = term_graph_to_module(g)
    mod.instantiate(relations)
    out = mod()
    show_out("Join(InputData1 ⋈ InputData2 on Y)", out)
    assert list(out.content.columns) == ["X","Y","Z"]
    assert total_embed_dim(out.embeddings) == 3  # 1 + 2

    # 4) Transformation 3→4 after Join — wire (dl1, dl2) → join → tr -------------
    lin = ConcatLinear(in_features=3, out_features=4)  # 1 (z1) + 2 (z2) -> 4
    g = nx.DiGraph()
    g.add_node("tr",   type="transformation", transformation=lin)   # output = tr
    g.add_node("join", type="join", merge_steps=[{"step": 1, "left_refs": [ColumnRef(0, 1)], "right_refs": [ColumnRef(1, 0)], "key_names": ["Y"]}], input_schemas=[["X", "Y"], ["Y", "Z"]], output_schema=["X","Y","Z"])
    g.add_node("dl1",  type="data_loader", name="InputData1")
    g.add_node("dl2",  type="data_loader", name="InputData2")
    g.add_edge("join", "tr")    # input → op
    g.add_edge("dl1",  "join")
    g.add_edge("dl2",  "join")
    mod = term_graph_to_module(g)
    mod.instantiate(relations)
    out = mod()
    show_out("Transformation(ConcatLinear 3→4) after Join", out)
    assert list(out.content.columns) == ["X","Y","Z"]
    assert total_embed_dim(out.embeddings) == 4

    # 5) Aggregation(mean) — wire dl1 → agg --------------------------------------
    g = nx.DiGraph()
    g.add_node("agg", type="agg", aggregation_name="mean", group_by_refs=[ColumnRef(0, 1)])  # output = agg (Y = col 1)
    g.add_node("dl1", type="data_loader", name="InputData1")
    g.add_edge("dl1", "agg")  # input → op
    mod = term_graph_to_module(g)
    mod.instantiate(relations)
    out = mod()
    show_out("Aggregation(mean) by ['Y'] on InputData1", out)
    assert total_embed_dim(out.embeddings) == 1

    print("\nAll ERA single-op tests completed ✓")

# %% [markdown]
# ## Additional tests

# %%
if __name__ == "__main__":
    lin = ConcatLinear(in_features=3, out_features=4)  # 1 (z1) + 2 (z2) -> 4
    g = nx.DiGraph()
    g.add_node("tr",   type="transformation", transformation=lin)   # output = tr
    g.add_node("join", type="join", merge_steps=[{"step": 1, "left_refs": [ColumnRef(0, 1)], "right_refs": [ColumnRef(1, 0)], "key_names": ["Y"]}], input_schemas=[["X", "Y"], ["Y", "Z"]], output_schema=["X","Y","Z"])
    g.add_node("dl1",  type="data_loader", name="InputData1")
    g.add_node("dl2",  type="data_loader", name="InputData2")
    g.add_edge("join", "tr")    # input → op
    g.add_edge("dl1",  "join")
    g.add_edge("dl2",  "join")
    mod = term_graph_to_module(g)
    mod.instantiate(relations)
    out = mod()
    show_out("Transformation(ConcatLinear 3→4) after Join", out)
    assert list(out.content.columns) == ["X","Y","Z"]
    assert total_embed_dim(out.embeddings) == 4

    # Grab the inner module actually used by the graph
    inner = mod.module_for_node("tr").transformation   # your Transformation ERA op
    assert isinstance(inner, ConcatLinear)

    # Build the same inputs the graph used (order matters!)
    # EITHER use the join’s output from the same graph:
    #   (best because it’s exactly what 'tr' saw)
    g_join = nx.DiGraph()
    g_join.add_node("join", type="join", merge_steps=[{"step": 1, "left_refs": [ColumnRef(0, 1)], "right_refs": [ColumnRef(1, 0)], "key_names": ["Y"]}], input_schemas=[["X", "Y"], ["Y", "Z"]], output_schema=["X","Y","Z"])
    g_join.add_node("dl1",  type="data_loader", name="InputData1")
    g_join.add_node("dl2",  type="data_loader", name="InputData2")
    g_join.add_edge("dl1", "join")
    g_join.add_edge("dl2", "join")
    mod_join = term_graph_to_module(g_join)
    mod_join.instantiate(relations)
    join_out = mod_join()                          # (5,1) and (5,2)

    # Use the same device/dtype as the *inner* linear
    W = inner.linear.weight
    cat_in = torch.cat([t.to(W.device, W.dtype) for t in join_out.embeddings], dim=1)
    ref = inner.linear(cat_in)

    # Now compare to the graph output
    torch.testing.assert_close(out.embeddings[0], ref, atol=1e-6, rtol=0)
    print("✓ Transformation matches inner linear on join outputs")

# %% [markdown]
# ### CPU vs. GPU Test

# %%
if __name__ == "__main__":
    # TODO: is this cell - #| hide or |# export ?

    import networkx as nx
    import torch
    from relann.era_operations import HAS_CUDF, HAS_PANDAS

# %%
if __name__ == "__main__":
    # TODO: is this cell - #| hide or |# export ?


    def _is_cudf_df(obj) -> bool:
        return HAS_CUDF and cudf is not None and isinstance(obj, cudf.DataFrame)  # type: ignore

    def _is_pandas_df(obj) -> bool:
        return HAS_PANDAS and isinstance(obj, pd.DataFrame)

    def _build_relations(N=5, device=torch.device("cpu"), use_cudf: bool = False):
        # Deterministic embeddings created on CPU then moved
        g = torch.Generator(device="cpu").manual_seed(12345)
        z1 = torch.randn(N, 1, generator=g).to(device)
        z2 = torch.randn(N, 2, generator=g).to(device)

        # Content backend
        df1_pd = pd.DataFrame({"X": list(range(1, N + 1)), "Y": list(range(10, 10 + N))})
        df2_pd = pd.DataFrame({"Y": list(range(10, 10 + N)), "Z": list(range(100, 100 + N))})
        if use_cudf:
            if not HAS_CUDF:
                raise RuntimeError("Requested cuDF backend but cuDF is not available")
            df1 = cudf.DataFrame.from_pandas(df1_pd)  # type: ignore
            df2 = cudf.DataFrame.from_pandas(df2_pd)  # type: ignore
        else:
            df1, df2 = df1_pd, df2_pd

        rels = {
            "InputData1": {
                "content": df1,
                "content_schema": ["X", "Y"],
                "embedding_shapes": [z1.shape],
                "embeddings": [z1],
            },
            "InputData2": {
                "content": df2,
                "content_schema": ["Y", "Z"],
                "embedding_shapes": [z2.shape],
                "embeddings": [z2],
            },
        }
        return rels


    def _build_graph_with_tr(tr_module):
        g = nx.DiGraph()
        g.add_node("tr",   type="transformation", transformation=tr_module)
        g.add_node("join", type="join", merge_steps=[{"step": 1, "left_refs": [ColumnRef(0, 1)], "right_refs": [ColumnRef(1, 0)], "key_names": ["Y"]}], input_schemas=[["X", "Y"], ["Y", "Z"]], output_schema=["X", "Y", "Z"])
        g.add_node("dl1",  type="data_loader", name="InputData1")
        g.add_node("dl2",  type="data_loader", name="InputData2")
        g.add_edge("join", "tr")
        g.add_edge("dl1",  "join")
        g.add_edge("dl2",  "join")
        return g


    def _run_graph(g, relations, device: torch.device):
        mod = term_graph_to_module(g)
        # If RelNN inherits nn.Module, keep ops on the intended device
        try:
            mod.to(device)
        except Exception:
            pass
        _ = mod.instantiate(relations)
        out = mod()
        return mod, out


    # -----------------------
    # 1) CPU-only test
    # -----------------------
    def test_cpu_only_graph_term():
        cpu = torch.device("cpu")
        relations = _build_relations(device=cpu, use_cudf=False)  # pandas + CPU tensors

        # DataLoader only
        g0 = nx.DiGraph()
        g0.add_node("dl1", type="data_loader", name="InputData1")
        _, out0 = _run_graph(g0, relations, cpu)
        assert _is_pandas_df(out0.content), "CPU test: content must be pandas.DataFrame"
        assert not _is_cudf_df(out0.content), "CPU test: cuDF must not be used"
        assert all((not e.is_cuda) for e in (out0.embeddings or [])), "CPU test: embeddings must be on CPU"

        # Join → (dl1, dl2)
        g1 = nx.DiGraph()
        g1.add_node("join", type="join", merge_steps=[{"step": 1, "left_refs": [ColumnRef(0, 1)], "right_refs": [ColumnRef(1, 0)], "key_names": ["Y"]}], input_schemas=[["X", "Y"], ["Y", "Z"]], output_schema=["X", "Y", "Z"])
        g1.add_node("dl1", type="data_loader", name="InputData1")
        g1.add_node("dl2", type="data_loader", name="InputData2")
        g1.add_edge("dl1", "join")
        g1.add_edge("dl2", "join")
        _, out1 = _run_graph(g1, relations, cpu)
        assert _is_pandas_df(out1.content), "CPU test: Join content must be pandas"
        assert all((not e.is_cuda) for e in (out1.embeddings or [])), "CPU test: Join embeddings must be on CPU"

        # Transformation(ConcatLinear 3→4) after Join
        tr = ConcatLinear(in_features=3, out_features=4)
        g2 = _build_graph_with_tr(tr)
        _, out2 = _run_graph(g2, relations, cpu)
        assert _is_pandas_df(out2.content), "CPU test: Transform content must be pandas"
        assert len(out2.embeddings) == 1 and out2.embeddings[0].shape[1] == 4, "CPU test: transform output dim mismatch"
        assert not out2.embeddings[0].is_cuda, "CPU test: transform output must be on CPU"

        print("✓ CPU-only graph_term test passed (no CUDA, no cuDF).")


    # -----------------------
    # 2) GPU test (and optional cuDF)
    # -----------------------
    def test_gpu_graph_term():
        if not torch.cuda.is_available():
            print("- SKIP GPU test: CUDA not available.")
            return

        gpu = torch.device("cuda:0")

        # (A) GPU with pandas backend
        relations_gpu_pd = _build_relations(device=gpu, use_cudf=False)  # pandas + CUDA tensors
        tr_gpu = ConcatLinear(in_features=3, out_features=4).to(gpu)
        g = _build_graph_with_tr(tr_gpu)
        _, out_pd = _run_graph(g, relations_gpu_pd, gpu)
        assert _is_pandas_df(out_pd.content), "GPU test (pandas): content should remain pandas"
        assert len(out_pd.embeddings) == 1 and out_pd.embeddings[0].is_cuda, "GPU test (pandas): embeddings must be CUDA"

        # (B) GPU with cuDF backend (if available)
        if HAS_CUDF:
            relations_gpu_cu = _build_relations(device=gpu, use_cudf=True)  # cuDF + CUDA tensors
            tr_gpu2 = ConcatLinear(in_features=3, out_features=4).to(gpu)
            g2 = _build_graph_with_tr(tr_gpu2)
            _, out_cu = _run_graph(g2, relations_gpu_cu, gpu)
            assert _is_cudf_df(out_cu.content), "GPU test (cuDF): content should be cuDF.DataFrame"
            assert len(out_cu.embeddings) == 1 and out_cu.embeddings[0].is_cuda, "GPU test (cuDF): embeddings must be CUDA"
            print("✓ GPU graph_term test passed (pandas & cuDF backends).")
        else:
            print("✓ GPU graph_term test passed (pandas backend). cuDF not available.")


    # -----------------------
    # Run both tests
    # -----------------------
    test_cpu_only_graph_term()
    test_gpu_graph_term()

