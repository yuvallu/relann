"""
DHN (Deep Homomorphism Networks) utilities for RelNN.

Implements the paper: "Deep Homomorphism Networks" (Maehara & NT, NeurIPS 2024).
Reference: https://proceedings.neurips.cc/paper_files/paper/2024/file/
           65f54fdf62cd5614dc5715ae7ece4ef6-Paper-Conference.pdf

Contains:
  - Homomorphism pre-computation (cycles C_k, cliques K_k)
  - DHN RelNN DSL program generator
  - Dataset loaders (CSL, EXP, SR25)
  - DB builder for RelNN Session
"""

import itertools
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Pattern definitions
# ---------------------------------------------------------------------------

PATTERN_SIZES = {
    "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6,
    "C7": 7, "C8": 8, "C9": 9, "C10": 10,
    "K3": 3, "K4": 4, "K5": 5,
}

POSITION_LABELS = list("uvwxyzabcdefghij")


def pattern_size(name: str) -> int:
    return PATTERN_SIZES[name]


def is_cycle(name: str) -> bool:
    return name.startswith("C")


def is_clique(name: str) -> bool:
    return name.startswith("K")


# ---------------------------------------------------------------------------
#  Homomorphism enumeration
# ---------------------------------------------------------------------------

def _simple_cycle_index(G: nx.Graph, length_bound: int = 10) -> Dict[int, List[Tuple[int, ...]]]:
    """Compute rooted simple-cycle mappings for all nodes, matching the
    reference implementation (gear/dhn).

    Uses `nx.simple_cycles` (injective cycles only), then generates all
    rooted rotations so that each node in the cycle serves as root once.

    Returns dict mapping cycle length k -> list of tuples (root, v1, ..., v_{k-1}).
    """
    result: Dict[int, List[Tuple[int, ...]]] = {}
    if length_bound < 2:
        return result

    # C2 = edges (each direction)
    c2_list: List[Tuple[int, ...]] = []
    for u, v in G.edges():
        c2_list.append((u, v))
        c2_list.append((v, u))
    if c2_list:
        result[2] = c2_list

    if length_bound < 3:
        return result

    # C3..Ck via nx.simple_cycles (injective cycles, each found once)
    raw: Dict[int, List[List[int]]] = {}
    for cyc in nx.simple_cycles(G, length_bound=length_bound):
        k = len(cyc)
        if k < 3:
            continue
        raw.setdefault(k, [])
        raw[k].append(cyc)
        raw[k].append(list(reversed(cyc)))

    for k, cycles in raw.items():
        rooted: List[Tuple[int, ...]] = []
        for cyc in cycles:
            for shift in range(k):
                rotated = cyc[shift:] + cyc[:shift]
                rooted.append(tuple(rotated))
        result[k] = rooted

    return result


def enumerate_cycle_homomorphisms(
    G: nx.Graph, k: int, root: int,
) -> List[Tuple[int, ...]]:
    """Enumerate rooted simple-cycle mappings of length k rooted at `root`.

    Uses injective cycle enumeration (nx.simple_cycles) matching the reference
    DHN implementation, NOT all closed walks.

    Returns list of tuples (root, v1, v2, ..., v_{k-1}).
    """
    if k < 2:
        raise ValueError(f"Cycle must have k >= 2, got {k}")

    if not hasattr(G, "_dhn_cycle_cache"):
        G._dhn_cycle_cache = {}

    max_k = max(PATTERN_SIZES.get(f"C{i}", 0) for i in range(2, 11))
    if not G._dhn_cycle_cache:
        all_cycles = _simple_cycle_index(G, length_bound=max_k)
        for kk, entries in all_cycles.items():
            by_root: Dict[int, List[Tuple[int, ...]]] = {}
            for entry in entries:
                by_root.setdefault(entry[0], []).append(entry)
            G._dhn_cycle_cache[kk] = by_root

    by_root = G._dhn_cycle_cache.get(k, {})
    return by_root.get(root, [])


def enumerate_clique_homomorphisms(
    G: nx.Graph, k: int, root: int,
) -> List[Tuple[int, ...]]:
    """Enumerate rooted homomorphisms from clique K_k to G, rooted at `root`.

    For simple graphs (no self-loops), a K_k hom must be injective, so this
    finds all k-cliques containing `root` and returns all permutations with
    root fixed at position 0 (as the reference impl does).

    Returns list of tuples (root, v1, ..., v_{k-1}).
    """
    if k < 2:
        raise ValueError(f"Clique must have k >= 2, got {k}")
    if k == 2:
        return [(root, v) for v in G.neighbors(root)]

    cliques = []
    root_neighbors = set(G.neighbors(root))
    candidates = sorted(root_neighbors)

    def _find_cliques(current_clique: List[int], remaining_candidates: List[int]):
        if len(current_clique) == k - 1:
            cliques.append(tuple([root] + current_clique))
            return
        for i, v in enumerate(remaining_candidates):
            if all(G.has_edge(v, u) for u in current_clique):
                _find_cliques(
                    current_clique + [v],
                    [c for c in remaining_candidates[i + 1:] if G.has_edge(c, v)],
                )

    _find_cliques([], candidates)

    results = []
    for clique in cliques:
        others = list(clique[1:])
        for perm in itertools.permutations(others):
            results.append((root,) + perm)
    return results


def enumerate_homomorphisms(
    G: nx.Graph, pattern: str, root: int,
) -> List[Tuple[int, ...]]:
    if pattern not in PATTERN_SIZES:
        raise ValueError(f"Unknown pattern: {pattern}. Known: {list(PATTERN_SIZES.keys())}")
    k = pattern_size(pattern)
    if is_cycle(pattern):
        return enumerate_cycle_homomorphisms(G, k, root)
    elif is_clique(pattern):
        return enumerate_clique_homomorphisms(G, k, root)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


# ---------------------------------------------------------------------------
#  Pre-compute homomorphism tables for a batch of graphs
# ---------------------------------------------------------------------------

def precompute_hom_tables(
    graphs: List[nx.Graph],
    patterns: List[str],
    graph_ids: Optional[List[int]] = None,
) -> Dict[str, Tuple[pd.DataFrame, torch.Tensor]]:
    """Pre-compute homomorphism mapping tables for all graphs and patterns.

    Returns dict mapping pattern name (e.g. "C3") to (DataFrame, embedding_tensor).
    DataFrame columns: [graph_id, u, v, w, ...] (position labels depend on pattern size).
    Embedding tensor: 1.0 for real homomorphisms, 0.0 for dummy padding rows.

    Dummy rows (embedding=0) are added for every (graph_id, node) pair that has
    no real homomorphisms for a given pattern, ensuring that inner joins in
    RelNN don't drop nodes. All positions in a dummy row point to the root node.
    """
    if graph_ids is None:
        graph_ids = list(range(len(graphs)))

    tables: Dict[str, List[List]] = {p: [] for p in patterns}
    masks: Dict[str, List[float]] = {p: [] for p in patterns}

    for gid, G in zip(graph_ids, graphs):
        for pat in patterns:
            k = pattern_size(pat)
            nodes_with_homs: set = set()
            for root in G.nodes():
                homs = enumerate_homomorphisms(G, pat, root)
                for mapping in homs:
                    tables[pat].append([gid] + list(mapping))
                    masks[pat].append(1.0)
                if homs:
                    nodes_with_homs.add(root)
            for root in G.nodes():
                if root not in nodes_with_homs:
                    tables[pat].append([gid] + [root] * k)
                    masks[pat].append(0.0)

    result = {}
    for pat in patterns:
        k = pattern_size(pat)
        col_names = ["graph_id"] + POSITION_LABELS[:k]
        if tables[pat]:
            df = pd.DataFrame(tables[pat], columns=col_names)
            emb = torch.tensor(masks[pat], dtype=torch.float32).unsqueeze(1)
        else:
            df = pd.DataFrame(columns=col_names)
            emb = torch.ones(0, 1)
        result[pat] = (df, emb)
    return result


def precompute_hom_tables_simple_cycles(
    graphs: List[nx.Graph],
    patterns: List[str],
    graph_ids: Optional[List[int]] = None,
) -> Dict[str, Tuple[pd.DataFrame, torch.Tensor]]:
    """Pre-compute homomorphism tables using simple cycles (nx.simple_cycles).
    
    For cycles (C2, C3, ..., C10), enumerate via nx.simple_cycles(G, length_bound=k)
    plus all rotations (matching gear/dhn cycle_mapping_index).
    For cliques (K2, K3, ...), fall back to enumerate_homomorphisms (unchanged).
    
    Returns same format as precompute_hom_tables: dict of (DataFrame, embedding_tensor).
    """
    if graph_ids is None:
        graph_ids = list(range(len(graphs)))

    tables: Dict[str, List[List]] = {p: [] for p in patterns}
    masks: Dict[str, List[float]] = {p: [] for p in patterns}

    for gid, G in zip(graph_ids, graphs):
        for pat in patterns:
            k = pattern_size(pat)
            
            if is_cycle(pat):
                # Enumerate via simple_cycles + rotations (matches gear/dhn)
                nodes_with_homs: set = set()
                try:
                    cycles = list(nx.simple_cycles(G, length_bound=k))
                except Exception:
                    # If simple_cycles fails, fall back to enumerate_homomorphisms
                    cycles = []
                
                # Add cycles and all rotations
                hom_tuples_seen = set()
                for cycle in cycles:
                    if len(cycle) == k:
                        for rot in range(k):
                            rotated = tuple(cycle[rot:] + cycle[:rot])
                            if rotated not in hom_tuples_seen:
                                hom_tuples_seen.add(rotated)
                                tables[pat].append([gid] + list(rotated))
                                masks[pat].append(1.0)
                                root = rotated[0]
                                nodes_with_homs.add(root)
                
                # For C2 (edges), also add undirected edges as directed both ways
                if k == 2:
                    edges = set()
                    for u, v in G.edges():
                        if (u, v) not in hom_tuples_seen:
                            hom_tuples_seen.add((u, v))
                            tables[pat].append([gid, u, v])
                            masks[pat].append(1.0)
                            nodes_with_homs.add(u)
                        if (v, u) not in hom_tuples_seen:
                            hom_tuples_seen.add((v, u))
                            tables[pat].append([gid, v, u])
                            masks[pat].append(1.0)
                            nodes_with_homs.add(v)
                
                # Add dummy rows for nodes without homomorphisms
                for root in G.nodes():
                    if root not in nodes_with_homs:
                        tables[pat].append([gid] + [root] * k)
                        masks[pat].append(0.0)
            else:
                # For cliques, use the standard enumerate_homomorphisms
                nodes_with_homs: set = set()
                for root in G.nodes():
                    homs = enumerate_homomorphisms(G, pat, root)
                    for mapping in homs:
                        tables[pat].append([gid] + list(mapping))
                        masks[pat].append(1.0)
                    if homs:
                        nodes_with_homs.add(root)
                for root in G.nodes():
                    if root not in nodes_with_homs:
                        tables[pat].append([gid] + [root] * k)
                        masks[pat].append(0.0)

    result = {}
    for pat in patterns:
        k = pattern_size(pat)
        col_names = ["graph_id"] + POSITION_LABELS[:k]
        if tables[pat]:
            df = pd.DataFrame(tables[pat], columns=col_names)
            emb = torch.tensor(masks[pat], dtype=torch.float32).unsqueeze(1)
        else:
            df = pd.DataFrame(columns=col_names)
            emb = torch.ones(0, 1)
        result[pat] = (df, emb)
    return result


# ---------------------------------------------------------------------------
#  DHN RelNN DSL program generator
# ---------------------------------------------------------------------------

@dataclass
class DHNConfig:
    """Configuration for generating a DHN RelNN program."""
    patterns_per_layer: List[List[str]]
    d_in: int
    d_k: int = 10
    d_hidden: int = 20
    num_classes: int = 10
    readout: str = "sum"
    lr: float = 0.001
    epochs: int = 1200
    dropout: float = 0.0
    injective: bool = False  # False = paper's definition (closed walks)
    mu_n_layers: int = 3     # 2 = official HomConv (Linear->ReLU->Drop->Linear)
    mu_dropout: float = 0.0  # Dropout within Mu transforms (official: 0.05)
    pattern_combine: str = "sum"  # "sum" (default, uses Union) or "concat" (official)


def _mlp_expr(z_var: str, d_in: int, d_hidden: int, d_out: int,
              name_prefix: str, n_layers: int = 3,
              dropout: float = 0.0) -> Tuple[str, str]:
    """Generate an MLP as chained module calls + transform defs.

    n_layers=3 (default): Linear(in,hidden)->ReLU->Linear(hidden,hidden)->ReLU->Linear(hidden,out)
    n_layers=2: Linear(in,out)->ReLU->[Dropout]->Linear(out,out) — matches official HomConv.

    Returns (transform_defs_str, call_expr_str).
    """
    if n_layers == 2:
        l1 = f"{name_prefix}_L1"
        l2 = f"{name_prefix}_L2"
        defs = (
            f"{l1} = Linear({d_in}, {d_out}) .\n"
            f"{l2} = Linear({d_out}, {d_out}) .\n"
        )
        if dropout > 0:
            drop = f"{name_prefix}_Drop"
            defs += f"{drop} = Dropout({dropout}) .\n"
            call = f"{l2}({drop}(ReLU()({l1}({z_var}))))"
        else:
            call = f"{l2}(ReLU()({l1}({z_var})))"
        return defs, call
    else:
        l1 = f"{name_prefix}_L1"
        l2 = f"{name_prefix}_L2"
        l3 = f"{name_prefix}_L3"
        defs = (
            f"{l1} = Linear({d_in}, {d_hidden}) .\n"
            f"{l2} = Linear({d_hidden}, {d_hidden}) .\n"
            f"{l3} = Linear({d_hidden}, {d_out}) .\n"
        )
        call = f"{l3}(ReLU()({l2}(ReLU()({l1}({z_var})))))"
        return defs, call


def _all_position_labels(patterns: List[str]) -> List[str]:
    """Collect unique non-root position labels needed for a set of patterns."""
    max_k = max(pattern_size(p) for p in patterns)
    return POSITION_LABELS[1:max_k]


def generate_dhn_program(config: DHNConfig) -> Tuple[str, str, str]:
    """Generate (define_program, fit_program, predict_program) RelNN DSL strings.

    The define program builds the full DHN architecture:
      - H0: initial node features (identity or projection)
      - For each layer L, for each pattern P:
          - Position alias rules
          - Per-position lookup + transform (Mu)
          - Element-wise product + aggregation
      - Combine across patterns with Rho
      - Graph readout + classifier
    """
    lines = []
    transform_defs = []
    num_layers = len(config.patterns_per_layer)

    lines.append(f"d_in = {config.d_in} .")
    lines.append(f"d_k = {config.d_k} .")
    lines.append(f"d_hidden = {config.d_hidden} .")
    lines.append(f"num_classes = {config.num_classes} .")
    lines.append("")

    # H0: initial embedding (identity -- just pass through node features)
    lines.append("H0(graph_id, n; z) :- Node(graph_id, n; z) .")
    lines.append("")

    for layer_idx, patterns in enumerate(config.patterns_per_layer):
        layer_num = layer_idx + 1
        h_in = f"H{layer_idx}"
        h_out = f"H{layer_num}"
        d_layer_in = config.d_in if layer_idx == 0 else config.d_k

        # Position alias rules for this layer's input
        needed_pos = _all_position_labels(patterns)
        for pos in needed_pos:
            lines.append(
                f"{h_in}_{pos}(graph_id, {pos}; z) :- {h_in}(graph_id, {pos}; z) ."
            )
        if needed_pos:
            lines.append("")

        agg_names = []
        for pat in patterns:
            k = pattern_size(pat)
            pos_labels = POSITION_LABELS[:k]
            hom_cols = ", ".join(["graph_id"] + pos_labels)
            hom_rel = f"Hom_{pat}"
            pat_safe = pat.replace("C", "Cyc").replace("K", "Kli")
            agg_name = f"{pat_safe}_Agg_{layer_num}"
            agg_names.append(agg_name)

            # Per-position transform rules.
            # Position 0 (root) multiplies by w_hom so dummy rows (w_hom=0)
            # contribute zero to the aggregation.
            t_names = []
            for pos_idx, pos in enumerate(pos_labels):
                t_name = f"{pat_safe}_T{pos_idx}_{layer_num}"
                t_names.append(t_name)
                mu_prefix = f"Mu_{pat_safe}_{pos_idx}_{layer_num}"

                mu_defs, mu_call = _mlp_expr(
                    "z", d_layer_in, config.d_hidden, config.d_k, mu_prefix,
                    n_layers=config.mu_n_layers, dropout=config.mu_dropout,
                )
                transform_defs.append(mu_defs)

                if pos_idx == 0:
                    node_rel = f"{h_in}(graph_id, {pos}; z)"
                    lines.append(
                        f"{t_name}({hom_cols}; {mu_call} * w_hom) :- "
                        f"{hom_rel}({hom_cols}; w_hom), {node_rel} ."
                    )
                else:
                    node_rel = f"{h_in}_{pos}(graph_id, {pos}; z)"
                    lines.append(
                        f"{t_name}({hom_cols}; {mu_call}) :- "
                        f"{hom_rel}({hom_cols}; _), {node_rel} ."
                    )

            # Product + aggregate
            z_vars = [f"z{i}" for i in range(k)]
            t_joins = ", ".join(
                f"{t_names[i]}({hom_cols}; {z_vars[i]})" for i in range(k)
            )
            product = " * ".join(z_vars)
            lines.append(
                f"{agg_name}(graph_id, u; sum({product})) :- {t_joins} ."
            )
            lines.append("")

        # Combine all pattern aggregations with Rho
        use_concat = config.pattern_combine == "concat" and len(agg_names) > 1
        rho_in_dim = len(patterns) * config.d_k if use_concat else config.d_k
        d_out = config.d_k
        rho_prefix = f"Rho_{layer_num}"
        rho_defs, rho_call_template = _mlp_expr(
            "__Z__", rho_in_dim, config.d_hidden, d_out, rho_prefix
        )
        transform_defs.append(rho_defs)

        if use_concat:
            z_agg_vars = [f"z_{i}" for i in range(len(agg_names))]
            concat_expr = f"Concat({', '.join(z_agg_vars)})"
            rho_input = rho_call_template.replace("__Z__", concat_expr)
            join_parts = ", ".join(
                f"{agg_names[i]}(graph_id, n; z_{i})"
                for i in range(len(agg_names))
            )
            lines.append(
                f"{h_out}(graph_id, n; {rho_input}) :- {join_parts} ."
            )
        else:
            if len(agg_names) == 1:
                rho_input = rho_call_template.replace("__Z__", "z")
                lines.append(
                    f"{h_out}(graph_id, n; {rho_input}) :- {agg_names[0]}(graph_id, n; z) ."
                )
            else:
                all_pats_name = f"AllPats_{layer_num}"
                sum_pats_name = f"SumPats_{layer_num}"
                union_parts = " | ".join(
                    f"{agg_names[i]}(graph_id, n; z)" for i in range(len(agg_names))
                )
                lines.append(f"{all_pats_name}(graph_id, n; z) :- {union_parts} .")
                lines.append(f"{sum_pats_name}(graph_id, n; sum(z)) :- {all_pats_name}(graph_id, n; z) .")
                rho_input = rho_call_template.replace("__Z__", "z")
                lines.append(f"{h_out}(graph_id, n; {rho_input}) :- {sum_pats_name}(graph_id, n; z) .")
        lines.append("")

    # Graph readout
    final_h = f"H{num_layers}"
    lines.append(
        f"GraphEmb(graph_id; {config.readout}(z)) :- {final_h}(graph_id, n; z) ."
    )
    # Classifier
    classifier_def = f"Classifier = Linear({config.d_k}, {config.num_classes}) .\n"
    transform_defs.append(classifier_def)
    lines.append("GraphLogits(graph_id; Classifier(z)) :- GraphEmb(graph_id; z) .")
    lines.append("")

    # Assemble define program
    all_defs = "".join(transform_defs)
    all_rules = "\n".join(lines)
    define_program = f"#lang:relnn\n{all_defs}\n{all_rules}"

    # Fit program
    fit_program = (
        f"#lang:relnn\n"
        f"?fit <epochs={config.epochs}, lr={config.lr}> "
        f"Loss(; CrossEntropyLoss()(z_pred, z_label)) :- "
        f"GraphLogits(graph_id; z_pred), GraphLabel(graph_id; z_label) .\n"
    )

    # Predict program
    predict_program = (
        f"#lang:relnn\n"
        f"?pred Predictions(graph_id; ArgMax()(z)) :- GraphLogits(graph_id; z) .\n"
    )

    return define_program, fit_program, predict_program


# ---------------------------------------------------------------------------
#  DB builder: combine node features + Hom tables into RelNN db dict
# ---------------------------------------------------------------------------

def build_dhn_db(
    graphs: List[nx.Graph],
    patterns: List[str],
    node_features: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    graph_ids: Optional[List[int]] = None,
    simple_cycles: bool = False,
) -> dict:
    """Build a RelNN-compatible db dict for DHN.

    Args:
        graphs: list of NetworkX graphs.
        patterns: list of pattern names (e.g. ["C2", "C3", "K3"]).
        node_features: tensor of shape (total_nodes, d_in), concatenated across graphs.
            If None, uses constant 1 features of shape (total_nodes, 1).
        labels: tensor of shape (num_graphs,) with integer class labels.
        graph_ids: optional list of integer graph IDs; defaults to range(len(graphs)).
        simple_cycles: if True, enumerate cycles via nx.simple_cycles (matches gear/dhn);
            if False, use standard enumerate_homomorphisms.

    Returns:
        dict suitable for Session(db=...).
    """
    if graph_ids is None:
        graph_ids = list(range(len(graphs)))

    # Build Node relation
    node_rows = []
    for gid, G in zip(graph_ids, graphs):
        for n in sorted(G.nodes()):
            node_rows.append({"graph_id": gid, "n": n})
    node_df = pd.DataFrame(node_rows)

    if node_features is None:
        node_features = torch.ones(len(node_df), 1)

    db: dict = {"Node": (node_df, node_features)}

    # Pre-compute and add Hom tables
    unique_patterns = sorted(set(patterns))
    if simple_cycles:
        hom_tables = precompute_hom_tables_simple_cycles(graphs, unique_patterns, graph_ids)
    else:
        hom_tables = precompute_hom_tables(graphs, unique_patterns, graph_ids)
    for pat, (df, emb) in hom_tables.items():
        hom_key = f"Hom_{pat}"
        db[hom_key] = (df, emb)
        n_real = int((emb > 0).sum().item())
        n_dummy = len(df) - n_real
        logger.info(f"  {hom_key}: {n_real} real + {n_dummy} dummy = {len(df)} rows")

    # Add labels
    if labels is not None:
        label_df = pd.DataFrame({"graph_id": graph_ids})
        db["GraphLabel"] = (label_df, labels.view(-1, 1).float())

    return db


def build_dhn_db_edge(
    graphs: List[nx.Graph],
    node_features: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    graph_ids: Optional[List[int]] = None,
    pad_dim: int = 10,
) -> dict:
    """Build a RelNN db with **Node**, **Edge**, and **AllNodePad** (no ``Hom_*``).

    ``Edge`` is directed: for every undirected ``{a,b}`` in ``G`` we add
    ``(graph_id, a, b)`` and ``(graph_id, b, a)`` with embedding **1.0** (no
    synthetic self-loops). **AllNodePad** has the same keys as ``Node``
    ``(graph_id, n)`` and **zero** embeddings of shape ``(total_nodes, pad_dim)``
    (GHL ``d_k``). The static program ``dhn_ghl_csl_c2_4_edge.relnn`` uses
    **Union(AllNodePad, *_Sparse)** then **sum** so every root node gets a row
    without dummy self-loops in ``Edge``.

    Args:
        graphs: list of NetworkX graphs.
        node_features: optional ``(total_nodes, d_in)`` tensor; default all-ones
            of shape ``(total_nodes, 1)``.
        labels: optional ``(num_graphs,)`` class indices.
        graph_ids: optional per-graph integer ids; default ``0..len(graphs)-1``.
        pad_dim: trailing feature dim for ``AllNodePad`` (must match ``Mu`` output width).

    Returns:
        ``dict`` with ``"Node"``, ``"Edge"``, ``"AllNodePad"``, and optionally
        ``"GraphLabel"``, suitable for ``Session(db=...)``.
    """
    if graph_ids is None:
        graph_ids = list(range(len(graphs)))

    node_rows: List[dict] = []
    for gid, G in zip(graph_ids, graphs):
        for n in sorted(G.nodes()):
            node_rows.append({"graph_id": gid, "n": n})
    node_df = pd.DataFrame(node_rows)

    if node_features is None:
        node_features = torch.ones(len(node_df), 1)

    db: dict = {"Node": (node_df, node_features)}

    edge_rows: List[dict] = []
    edge_ones: List[float] = []
    for gid, G in zip(graph_ids, graphs):
        for u, v in G.edges():
            edge_rows.append({"graph_id": gid, "u": u, "v": v})
            edge_ones.append(1.0)
            edge_rows.append({"graph_id": gid, "u": v, "v": u})
            edge_ones.append(1.0)

    edge_df = pd.DataFrame(edge_rows)
    edge_emb = torch.tensor(edge_ones, dtype=torch.float32).unsqueeze(1)
    db["Edge"] = (edge_df, edge_emb)
    logger.info(f"  Edge: {len(edge_df)} directed rows (2|undir E|)")

    pad_emb = torch.zeros(len(node_df), pad_dim, dtype=torch.float32)
    db["AllNodePad"] = (node_df, pad_emb)
    logger.info(f"  AllNodePad: {len(node_df)} rows × {pad_dim} (zeros)")

    if labels is not None:
        label_df = pd.DataFrame({"graph_id": graph_ids})
        db["GraphLabel"] = (label_df, labels.view(-1, 1).float())

    return db


# ---------------------------------------------------------------------------
#  Dataset loaders
# ---------------------------------------------------------------------------

def _pyg_to_nx(data) -> nx.Graph:
    """Convert a PyG Data object to a NetworkX undirected graph."""
    G = nx.Graph()
    num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.shape[0]
    G.add_nodes_from(range(num_nodes))
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        if u != v:
            G.add_edge(int(u), int(v))
    return G


def load_csl_dataset(root: str = "./data") -> Tuple[List[nx.Graph], torch.Tensor, int, int]:
    """Load the CSL (Circular Skip Links) dataset via PyTorch Geometric.

    Returns (graphs, labels, num_classes, num_features).
    CSL: 150 graphs, 41 nodes, degree 4, 10 isomorphism classes, no node features.
    """
    from torch_geometric.datasets import GNNBenchmarkDataset
    dataset = GNNBenchmarkDataset(root=root, name="CSL")
    graphs = [_pyg_to_nx(data) for data in dataset]
    labels = torch.tensor([data.y.item() for data in dataset], dtype=torch.long)
    return graphs, labels, 10, 1


def _pyg_data_to_nx(d) -> nx.Graph:
    """Convert an old-format PyG Data object (stored in __dict__) to nx.Graph."""
    dd = d.__dict__
    x = dd.get("x")
    ei = dd.get("edge_index")
    num_nodes = x.shape[0] if x is not None else int(ei.max()) + 1
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    ei_np = ei.numpy()
    for col in range(ei_np.shape[1]):
        u, v = int(ei_np[0, col]), int(ei_np[1, col])
        if u != v:
            G.add_edge(u, v)
    return G


def _download_exp_pkl(data_dir: str) -> str:
    """Download the EXP GRAPHSAT.pkl from GNN-RNI repo if not cached."""
    import urllib.request
    pkl_dir = os.path.join(data_dir, "EXP", "raw")
    pkl_path = os.path.join(pkl_dir, "GRAPHSAT.pkl")
    if os.path.exists(pkl_path):
        return pkl_path
    os.makedirs(pkl_dir, exist_ok=True)
    url = "https://github.com/ralphabb/GNN-RNI/raw/main/Data/EXP/raw/GRAPHSAT.pkl"
    logger.info(f"Downloading EXP dataset from {url}")
    urllib.request.urlretrieve(url, pkl_path)
    return pkl_path


def load_exp_dataset(root: str = "./data") -> Tuple[List[nx.Graph], torch.Tensor, int, int]:
    """Load the real EXP dataset from Abboud et al. (IJCAI 2021).

    The EXP dataset contains 1200 WL-indistinguishable planar SAT/UNSAT graph
    pairs (600 SAT + 600 UNSAT). Downloaded from the GNN-RNI repository on
    first use.

    Returns (graphs, labels, num_classes=2, num_features=1).
    Node features are binary (0=clause, 1=variable).
    """
    import pickle

    pkl_path = _download_exp_pkl(root)
    with open(pkl_path, "rb") as f:
        data_list = pickle.load(f)

    graphs = []
    labels = []
    for d in data_list:
        G = _pyg_data_to_nx(d)
        graphs.append(G)
        y = d.__dict__["y"]
        labels.append(int(y.item()) if y.numel() == 1 else int(y[0].item()))

    label_tensor = torch.tensor(labels, dtype=torch.long)
    num_features = 1
    return graphs, label_tensor, 2, num_features


_PAULUS_SR25_ADJ = [
    # All 15 Paulus SR(25,12,5,6) graphs from Brouwer's database:
    # https://aeb.win.tue.nl/graphs/paulus/p25_01 .. p25_15
    # Each entry is a list of 25 neighbor-lists (0-indexed).
    [[1,2,3,4,5,6,7,8,9,10,11,12],[0,2,3,4,5,6,13,14,15,16,17,18],[0,1,3,4,5,6,19,20,21,22,23,24],[0,1,2,7,8,9,13,14,15,19,20,21],[0,1,2,7,10,11,13,16,17,19,22,23],[0,1,2,8,10,12,14,16,18,20,22,24],[0,1,2,9,11,12,15,17,18,21,23,24],[0,3,4,8,9,10,13,15,18,22,23,24],[0,3,5,7,9,11,14,16,17,21,22,24],[0,3,6,7,8,12,16,17,18,19,20,23],[0,4,5,7,11,12,13,14,18,20,21,23],[0,4,6,8,10,12,14,15,17,19,21,22],[0,5,6,9,10,11,13,15,16,19,20,24],[1,3,4,7,10,12,15,16,17,20,21,24],[1,3,5,8,10,11,15,16,18,19,21,23],[1,3,6,7,11,12,13,14,18,19,22,24],[1,4,5,8,9,12,13,14,17,19,23,24],[1,4,6,8,9,11,13,16,18,20,21,22],[1,5,6,7,9,10,14,15,17,20,22,23],[2,3,4,9,11,12,14,15,16,20,22,23],[2,3,5,9,10,12,13,17,18,19,21,22],[2,3,6,8,10,11,13,14,17,20,23,24],[2,4,5,7,8,11,15,17,18,19,20,24],[2,4,6,7,9,10,14,16,18,19,21,24],[2,5,6,7,8,12,13,15,16,21,22,23]],
    [[1,2,3,4,5,6,7,8,9,10,11,12],[0,2,3,4,5,6,13,14,15,16,17,18],[0,1,3,4,5,6,19,20,21,22,23,24],[0,1,2,7,8,9,13,14,15,19,20,21],[0,1,2,7,10,11,13,16,17,19,22,23],[0,1,2,8,10,12,14,16,18,20,22,24],[0,1,2,9,11,12,15,17,18,21,23,24],[0,3,4,8,9,10,13,15,18,22,23,24],[0,3,5,7,9,12,14,16,17,21,22,23],[0,3,6,7,8,11,16,17,18,19,20,24],[0,4,5,7,11,12,13,14,18,19,21,24],[0,4,6,9,10,12,14,15,16,19,20,23],[0,5,6,8,10,11,13,15,17,20,21,22],[1,3,4,7,10,12,15,16,17,20,21,24],[1,3,5,8,10,11,15,16,18,19,21,23],[1,3,6,7,11,12,13,14,18,20,22,23],[1,4,5,8,9,11,13,14,17,20,23,24],[1,4,6,8,9,12,13,16,18,19,21,22],[1,5,6,7,9,10,14,15,17,19,22,24],[2,3,4,9,10,11,14,17,18,20,21,22],[2,3,5,9,11,12,13,15,16,19,22,24],[2,3,6,8,10,12,13,14,17,19,23,24],[2,4,5,7,8,12,15,17,18,19,20,23],[2,4,6,7,8,11,14,15,16,21,22,24],[2,5,6,7,9,10,13,16,18,20,21,23]],
    [[1,2,3,4,5,6,7,8,9,10,11,12],[0,2,3,4,5,6,13,14,15,16,17,18],[0,1,3,4,5,6,19,20,21,22,23,24],[0,1,2,7,8,9,13,14,15,19,20,21],[0,1,2,7,10,11,13,16,17,19,22,23],[0,1,2,8,10,12,14,16,18,20,22,24],[0,1,2,9,11,12,15,17,18,21,23,24],[0,3,4,8,9,10,13,15,18,22,23,24],[0,3,5,7,9,11,14,16,17,20,23,24],[0,3,6,7,8,12,16,17,18,19,21,22],[0,4,5,7,11,12,13,14,18,19,21,24],[0,4,6,8,10,12,14,15,17,19,20,23],[0,5,6,9,10,11,13,15,16,20,21,22],[1,3,4,7,10,12,15,16,17,20,21,24],[1,3,5,8,10,11,15,16,18,19,21,23],[1,3,6,7,11,12,13,14,18,20,22,23],[1,4,5,8,9,12,13,14,17,21,22,23],[1,4,6,8,9,11,13,16,18,19,20,24],[1,5,6,7,9,10,14,15,17,19,22,24],[2,3,4,9,10,11,14,17,18,20,21,22],[2,3,5,8,11,12,13,15,17,19,22,24],[2,3,6,9,10,12,13,14,16,19,23,24],[2,4,5,7,9,12,15,16,18,19,20,23],[2,4,6,7,8,11,14,15,16,21,22,24],[2,5,6,7,8,10,13,17,18,20,21,23]],
    [[1,2,3,4,5,6,7,8,9,10,11,12],[0,2,3,4,5,6,13,14,15,16,17,18],[0,1,3,4,5,6,19,20,21,22,23,24],[0,1,2,7,8,9,13,14,15,19,20,21],[0,1,2,7,10,11,13,16,17,19,22,23],[0,1,2,8,10,12,14,16,18,20,22,24],[0,1,2,9,11,12,15,17,18,21,23,24],[0,3,4,8,9,10,13,16,18,21,23,24],[0,3,5,7,9,12,15,16,17,19,22,24],[0,3,6,7,8,11,14,17,18,20,22,23],[0,4,5,7,11,12,13,14,15,20,23,24],[0,4,6,9,10,12,14,16,17,19,20,21],[0,5,6,8,10,11,13,15,18,19,21,22],[1,3,4,7,10,12,15,17,18,20,21,22],[1,3,5,9,10,11,15,16,18,19,20,23],[1,3,6,8,10,12,13,14,17,19,23,24],[1,4,5,7,8,11,14,17,18,19,21,24],[1,4,6,8,9,11,13,15,16,20,22,24],[1,5,6,7,9,12,13,14,16,21,22,23],[2,3,4,8,11,12,14,15,16,21,22,23],[2,3,5,9,10,11,13,14,17,21,22,24],[2,3,6,7,11,12,13,16,18,19,20,24],[2,4,5,8,9,12,13,17,18,19,20,23],[2,4,6,7,9,10,14,15,18,19,22,24],[2,5,6,7,8,10,15,16,17,20,21,23]],
    [[1,2,3,4,5,6,7,8,9,10,11,12],[0,2,3,4,5,6,13,14,15,16,17,18],[0,1,3,4,5,6,19,20,21,22,23,24],[0,1,2,7,8,9,13,14,15,19,20,21],[0,1,2,7,10,11,13,16,17,19,22,23],[0,1,2,8,10,12,14,16,18,20,22,24],[0,1,2,9,11,12,15,17,18,21,23,24],[0,3,4,8,9,10,13,15,18,22,23,24],[0,3,5,7,9,12,16,17,18,19,21,22],[0,3,6,7,8,11,14,16,17,20,23,24],[0,4,5,7,11,12,14,15,18,19,20,23],[0,4,6,9,10,12,13,14,17,20,21,22],[0,5,6,8,10,11,13,15,16,19,21,24],[1,3,4,7,11,12,14,15,16,21,22,24],[1,3,5,9,10,11,13,16,18,20,21,23],[1,3,6,7,10,12,13,17,18,19,20,24],[1,4,5,8,9,12,13,14,17,19,23,24],[1,4,6,8,9,11,15,16,18,19,20,22],[1,5,6,7,8,10,14,15,17,21,22,23],[2,3,4,8,10,12,15,16,17,20,21,23],[2,3,5,9,10,11,14,15,17,19,22,24],[2,3,6,8,11,12,13,14,18,19,22,23],[2,4,5,7,8,11,13,17,18,20,21,24],[2,4,6,7,9,10,14,16,18,19,21,24],[2,5,6,7,9,12,13,15,16,20,22,23]],
    [[1,2,3,4,5,6,7,8,9,10,11,12],[0,2,3,4,5,6,13,14,15,16,17,18],[0,1,3,4,5,7,13,19,20,21,22,23],[0,1,2,6,8,9,14,15,19,20,21,24],[0,1,2,7,10,11,14,16,17,19,22,24],[0,1,2,8,10,12,15,16,18,20,22,23],[0,1,3,8,10,11,13,17,18,21,23,24],[0,2,4,9,11,12,13,15,17,20,23,24],[0,3,5,6,9,12,16,17,19,22,23,24],[0,3,7,8,11,12,13,14,15,18,19,22],[0,4,5,6,11,12,14,18,20,21,22,24],[0,4,6,7,9,10,15,16,18,19,21,23],[0,5,7,8,9,10,13,14,16,17,20,21],[1,2,6,7,9,12,14,17,18,21,22,23],[1,3,4,9,10,12,13,15,16,21,22,24],[1,3,5,7,9,11,14,16,18,20,23,24],[1,4,5,8,11,12,14,15,17,19,21,23],[1,4,6,7,8,12,13,16,18,19,20,24],[1,5,6,9,10,11,13,15,17,19,20,22],[2,3,4,8,9,11,16,17,18,20,21,22],[2,3,5,7,10,12,15,17,18,19,21,24],[2,3,6,10,11,12,13,14,16,19,20,23],[2,4,5,8,9,10,13,14,18,19,23,24],[2,5,6,7,8,11,13,15,16,21,22,24],[3,4,6,7,8,10,14,15,17,20,22,23]],
    [[1,2,3,4,5,6,7,8,9,10,11,12],[0,2,3,4,5,6,13,14,15,16,17,18],[0,1,3,4,5,6,19,20,21,22,23,24],[0,1,2,7,8,9,13,14,15,19,20,21],[0,1,2,7,10,11,13,16,17,19,22,23],[0,1,2,8,10,12,14,16,18,20,22,24],[0,1,2,9,11,12,15,17,18,21,23,24],[0,3,4,8,9,10,13,16,18,21,23,24],[0,3,5,7,9,12,15,16,17,19,22,24],[0,3,6,7,8,11,14,17,18,20,22,23],[0,4,5,7,11,12,13,15,18,20,21,22],[0,4,6,9,10,12,14,16,17,19,20,21],[0,5,6,8,10,11,13,14,15,19,23,24],[1,3,4,7,10,12,14,15,17,20,23,24],[1,3,5,9,11,12,13,16,18,19,20,23],[1,3,6,8,10,12,13,17,18,19,21,22],[1,4,5,7,8,11,14,17,18,19,21,24],[1,4,6,8,9,11,13,15,16,20,22,24],[1,5,6,7,9,10,14,15,16,21,22,23],[2,3,4,8,11,12,14,15,16,21,22,23],[2,3,5,9,10,11,13,14,17,21,22,24],[2,3,6,7,10,11,15,16,18,19,20,24],[2,4,5,8,9,10,15,17,18,19,20,23],[2,4,6,7,9,12,13,14,18,19,22,24],[2,5,6,7,8,12,13,16,17,20,21,23]],
    [[1,2,3,4,5,6,7,8,9,10,11,12],[0,2,3,4,5,6,13,14,15,16,17,18],[0,1,3,4,5,6,19,20,21,22,23,24],[0,1,2,7,8,9,13,14,15,19,20,21],[0,1,2,7,10,11,13,16,17,19,22,23],[0,1,2,8,10,12,14,16,18,20,22,24],[0,1,2,9,11,12,15,17,18,21,23,24],[0,3,4,8,9,10,13,16,18,21,23,24],[0,3,5,7,9,12,15,16,17,20,22,23],[0,3,6,7,8,11,14,17,18,19,22,24],[0,4,5,7,11,12,13,14,15,21,22,24],[0,4,6,9,10,12,14,16,17,19,20,21],[0,5,6,8,10,11,13,15,18,19,20,23],[1,3,4,7,10,12,15,17,18,19,20,24],[1,3,5,9,10,11,15,16,18,19,21,22],[1,3,6,8,10,12,13,14,17,21,22,23],[1,4,5,7,8,11,14,17,18,20,21,23],[1,4,6,8,9,11,13,15,16,20,22,24],[1,5,6,7,9,12,13,14,16,19,23,24],[2,3,4,9,11,12,13,14,18,20,22,23],[2,3,5,8,11,12,13,16,17,19,21,24],[2,3,6,7,10,11,14,15,16,20,23,24],[2,4,5,8,9,10,14,15,17,19,23,24],[2,4,6,7,8,12,15,16,18,19,21,22],[2,5,6,7,9,10,13,17,18,20,21,22]],
    [[1,2,3,4,5,6,7,8,9,10,11,12],[0,2,3,4,5,6,13,14,15,16,17,18],[0,1,3,4,5,6,19,20,21,22,23,24],[0,1,2,7,8,9,13,14,15,19,20,21],[0,1,2,7,10,11,13,16,17,19,22,23],[0,1,2,8,10,12,14,16,18,20,22,24],[0,1,2,9,11,12,15,17,18,21,23,24],[0,3,4,8,9,12,13,16,18,19,23,24],[0,3,5,7,9,11,14,17,18,20,22,23],[0,3,6,7,8,10,15,16,17,21,22,24],[0,4,5,9,11,12,13,15,16,20,21,22],[0,4,6,8,10,12,14,15,17,19,20,23],[0,5,6,7,10,11,13,14,18,19,21,24],[1,3,4,7,10,12,14,15,18,21,22,23],[1,3,5,8,11,12,13,15,17,19,22,24],[1,3,6,9,10,11,13,14,16,20,23,24],[1,4,5,7,9,10,15,17,18,19,20,24],[1,4,6,8,9,11,14,16,18,19,21,22],[1,5,6,7,8,12,13,16,17,20,21,23],[2,3,4,7,11,12,14,16,17,20,21,24],[2,3,5,8,10,11,15,16,18,19,21,23],[2,3,6,9,10,12,13,17,18,19,20,22],[2,4,5,8,9,10,13,14,17,21,23,24],[2,4,6,7,8,11,13,15,18,20,22,24],[2,5,6,7,9,12,14,15,16,19,22,23]],
    [[1,2,3,4,5,6,7,8,9,10,11,12],[0,2,3,4,5,6,13,14,15,16,17,18],[0,1,3,4,5,7,13,19,20,21,22,23],[0,1,2,6,8,9,14,15,19,20,21,24],[0,1,2,7,10,11,14,16,17,19,22,24],[0,1,2,8,10,12,15,16,18,20,22,23],[0,1,3,9,10,12,13,17,18,19,23,24],[0,2,4,8,11,12,13,14,18,21,23,24],[0,3,5,7,10,11,15,17,18,20,21,24],[0,3,6,10,11,12,13,14,16,20,21,22],[0,4,5,6,8,9,16,17,21,22,23,24],[0,4,7,8,9,12,13,15,16,17,19,20],[0,5,6,7,9,11,14,15,18,19,22,23],[1,2,6,7,9,11,16,17,18,20,21,23],[1,3,4,7,9,12,15,16,18,21,22,24],[1,3,5,8,11,12,14,16,17,19,21,23],[1,4,5,9,10,11,13,14,15,20,23,24],[1,4,6,8,10,11,13,15,18,19,21,22],[1,5,6,7,8,12,13,14,17,20,22,24],[2,3,4,6,11,12,15,17,20,22,23,24],[2,3,5,8,9,11,13,16,18,19,22,24],[2,3,7,8,9,10,13,14,15,17,22,23],[2,4,5,9,10,12,14,17,18,19,20,21],[2,5,6,7,10,12,13,15,16,19,21,24],[3,4,6,7,8,10,14,16,18,19,20,23]],
    [[1,2,3,4,5,6,7,8,9,10,11,12],[0,2,3,4,5,6,13,14,15,16,17,18],[0,1,3,4,5,6,19,20,21,22,23,24],[0,1,2,7,8,9,13,14,15,19,20,21],[0,1,2,7,10,11,13,16,17,19,22,23],[0,1,2,8,10,12,14,16,18,20,22,24],[0,1,2,9,11,12,15,17,18,21,23,24],[0,3,4,8,9,12,14,16,17,21,22,23],[0,3,5,7,9,11,15,16,18,19,22,24],[0,3,6,7,8,10,13,17,18,20,23,24],[0,4,5,9,11,12,13,14,17,19,20,24],[0,4,6,8,10,12,13,15,18,19,21,22],[0,5,6,7,10,11,14,15,16,20,21,23],[1,3,4,9,10,11,14,15,18,20,22,23],[1,3,5,7,10,12,13,15,17,21,22,24],[1,3,6,8,11,12,13,14,16,19,23,24],[1,4,5,7,8,12,15,17,18,19,20,23],[1,4,6,7,9,10,14,16,18,19,21,24],[1,5,6,8,9,11,13,16,17,20,21,22],[2,3,4,8,10,11,15,16,17,20,21,24],[2,3,5,9,10,12,13,16,18,19,21,23],[2,3,6,7,11,12,14,17,18,19,20,22],[2,4,5,7,8,11,13,14,18,21,23,24],[2,4,6,7,9,12,13,15,16,20,22,24],[2,5,6,8,9,10,14,15,17,19,22,23]],
    [[1,2,3,4,5,6,7,8,9,10,11,12],[0,2,3,4,5,6,13,14,15,16,17,18],[0,1,3,4,7,8,13,14,19,20,21,22],[0,1,2,4,9,10,15,16,19,20,23,24],[0,1,2,3,11,12,17,18,21,22,23,24],[0,1,6,7,9,11,13,15,17,19,21,23],[0,1,5,8,10,12,14,16,18,19,21,23],[0,2,5,8,9,11,14,15,18,20,21,24],[0,2,6,7,10,12,13,15,18,19,22,24],[0,3,5,7,10,11,14,16,17,19,22,24],[0,3,6,8,9,12,14,15,17,20,22,23],[0,4,5,7,9,12,13,16,18,20,22,23],[0,4,6,8,10,11,13,16,17,20,21,24],[1,2,5,8,11,12,15,16,17,19,20,22],[1,2,6,7,9,10,16,17,18,20,21,22],[1,3,5,7,8,10,13,17,18,20,23,24],[1,3,6,9,11,12,13,14,18,19,20,24],[1,4,5,9,10,12,13,14,15,21,22,24],[1,4,6,7,8,11,14,15,16,22,23,24],[2,3,5,6,8,9,13,16,21,22,23,24],[2,3,7,10,11,12,13,14,15,16,21,23],[2,4,5,6,7,12,14,17,19,20,23,24],[2,4,8,9,10,11,13,14,17,18,19,23],[3,4,5,6,10,11,15,18,19,20,21,22],[3,4,7,8,9,12,15,16,17,18,19,21]],
    [[1,2,3,4,5,6,7,8,9,10,11,12],[0,2,3,4,5,6,13,14,15,16,17,18],[0,1,3,4,5,6,19,20,21,22,23,24],[0,1,2,7,8,9,13,14,15,19,20,21],[0,1,2,7,10,11,13,16,17,19,22,23],[0,1,2,8,10,12,14,16,18,20,22,24],[0,1,2,9,11,12,15,17,18,21,23,24],[0,3,4,8,9,10,13,15,18,22,23,24],[0,3,5,7,9,12,14,16,17,19,23,24],[0,3,6,7,8,11,16,17,18,20,21,22],[0,4,5,7,11,12,13,14,18,20,21,23],[0,4,6,9,10,12,14,15,16,19,21,22],[0,5,6,8,10,11,13,15,17,19,20,24],[1,3,4,7,10,12,15,16,17,20,21,24],[1,3,5,8,10,11,15,16,18,19,21,23],[1,3,6,7,11,12,13,14,18,19,22,24],[1,4,5,8,9,11,13,14,17,21,22,24],[1,4,6,8,9,12,13,16,18,19,20,23],[1,5,6,7,9,10,14,15,17,20,22,23],[2,3,4,8,11,12,14,15,17,20,22,23],[2,3,5,9,10,12,13,17,18,19,21,22],[2,3,6,9,10,11,13,14,16,20,23,24],[2,4,5,7,9,11,15,16,18,19,20,24],[2,4,6,7,8,10,14,17,18,19,21,24],[2,5,6,7,8,12,13,15,16,21,22,23]],
    [[1,2,3,4,5,6,7,8,9,10,11,12],[0,2,3,4,5,6,13,14,15,16,17,18],[0,1,3,4,5,6,19,20,21,22,23,24],[0,1,2,7,8,9,13,14,15,19,20,21],[0,1,2,7,10,11,13,16,17,19,22,23],[0,1,2,8,10,12,14,16,18,20,22,24],[0,1,2,9,11,12,15,17,18,21,23,24],[0,3,4,8,9,10,13,16,18,21,23,24],[0,3,5,7,9,12,15,16,17,20,22,23],[0,3,6,7,8,11,14,17,18,19,22,24],[0,4,5,7,11,12,13,15,18,19,20,24],[0,4,6,9,10,12,14,16,17,19,20,21],[0,5,6,8,10,11,13,14,15,21,22,23],[1,3,4,7,10,12,14,15,17,21,22,24],[1,3,5,9,11,12,13,16,18,19,21,22],[1,3,6,8,10,12,13,17,18,19,20,23],[1,4,5,7,8,11,14,17,18,20,21,23],[1,4,6,8,9,11,13,15,16,20,22,24],[1,5,6,7,9,10,14,15,16,19,23,24],[2,3,4,9,10,11,14,15,18,20,22,23],[2,3,5,8,10,11,15,16,17,19,21,24],[2,3,6,7,11,12,13,14,16,20,23,24],[2,4,5,8,9,12,13,14,17,19,23,24],[2,4,6,7,8,12,15,16,18,19,21,22],[2,5,6,7,9,10,13,17,18,20,21,22]],
    [[1,2,3,4,5,6,7,8,9,10,11,12],[0,2,3,4,5,6,13,14,15,16,17,18],[0,1,3,4,7,8,13,14,19,20,21,22],[0,1,2,4,9,10,15,16,19,20,23,24],[0,1,2,3,11,12,17,18,21,22,23,24],[0,1,7,8,9,11,13,15,16,17,21,23],[0,1,7,8,10,12,14,15,16,18,22,24],[0,2,5,6,9,11,14,15,19,21,22,24],[0,2,5,6,10,12,13,16,20,21,22,23],[0,3,5,7,11,12,13,16,18,19,20,24],[0,3,6,8,11,12,14,15,17,19,20,23],[0,4,5,7,9,10,14,17,18,20,22,23],[0,4,6,8,9,10,13,17,18,19,21,24],[1,2,5,8,9,12,15,17,18,19,20,22],[1,2,6,7,10,11,16,17,18,19,20,21],[1,3,5,6,7,10,13,17,19,22,23,24],[1,3,5,6,8,9,14,18,20,21,23,24],[1,4,5,10,11,12,13,14,15,20,21,24],[1,4,6,9,11,12,13,14,16,19,22,23],[2,3,7,9,10,12,13,14,15,18,21,23],[2,3,8,9,10,11,13,14,16,17,22,24],[2,4,5,7,8,12,14,16,17,19,23,24],[2,4,6,7,8,11,13,15,18,20,23,24],[3,4,5,8,10,11,15,16,18,19,21,22],[3,4,6,7,9,12,15,16,17,20,21,22]],
]


def _validate_sr25_graph(adj: List[List[int]]) -> bool:
    """Check that an adjacency list satisfies SR(25,12,5,6) parameters."""
    n = len(adj)
    if n != 25:
        return False
    A = np.zeros((n, n), dtype=int)
    for i, nbrs in enumerate(adj):
        if len(nbrs) != 12:
            return False
        for j in nbrs:
            A[i][j] = 1
    if not np.array_equal(A, A.T):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            common = int((A[i] * A[j]).sum())
            if A[i][j] == 1 and common != 5:
                return False
            if A[i][j] == 0 and common != 6:
                return False
    return True


def load_sr25_dataset() -> Tuple[List[nx.Graph], torch.Tensor, int, int]:
    """Load the SR25 dataset: all 15 Paulus SR(25,12,5,6) graphs.

    Data source: A.E. Brouwer's database (https://aeb.win.tue.nl/graphs/paulus/).
    These are the complete set of strongly regular graphs with parameters
    (25, 12, 5, 6), classified by Paulus (1973).

    Returns (graphs, labels, num_classes=15, num_features=1).
    Each graph is its own class (the task is to distinguish all 15).
    """
    graphs = []
    for idx, adj in enumerate(_PAULUS_SR25_ADJ):
        G = nx.Graph()
        G.add_nodes_from(range(25))
        for i, nbrs in enumerate(adj):
            for j in nbrs:
                if j > i:
                    G.add_edge(i, j)
        graphs.append(G)

    labels = torch.arange(15, dtype=torch.long)
    return graphs, labels, 15, 1


def _load_tu_dataset(
    name: str, num_classes: int, root: str = "./data",
) -> Tuple[List[nx.Graph], torch.Tensor, int, int, Optional[torch.Tensor]]:
    """Load a TUDataset and return graphs, labels, classes, features dim, and
    a concatenated node-feature tensor (or None if no features).

    Returns (graphs, labels, num_classes, num_features, node_features_tensor).
    """
    from torch_geometric.datasets import TUDataset
    dataset = TUDataset(root=root, name=name, use_node_attr=True)
    graphs = [_pyg_to_nx(data) for data in dataset]
    labels = torch.tensor([data.y.item() for data in dataset], dtype=torch.long)
    num_features = dataset.num_node_features or 1

    if num_features > 1:
        feat_list = []
        for data in dataset:
            if data.x is not None:
                feat_list.append(data.x.float())
            else:
                feat_list.append(torch.ones(data.num_nodes, 1))
        node_features = torch.cat(feat_list, dim=0)
    else:
        node_features = None

    return graphs, labels, num_classes, num_features, node_features


def load_enzymes_dataset(root: str = "./data"):
    """Load the ENZYMES dataset (600 graphs, 6 classes).

    Returns (graphs, labels, 6, num_features, node_features_tensor).
    """
    return _load_tu_dataset("ENZYMES", 6, root)


def load_proteins_dataset(root: str = "./data"):
    """Load the PROTEINS dataset (1113 graphs, 2 classes).

    Returns (graphs, labels, 2, num_features, node_features_tensor).
    """
    return _load_tu_dataset("PROTEINS", 2, root)


def _load_dataset(name: str, root: str = "./data"):
    """Unified dataset loader.

    Returns (graphs, labels, num_classes, num_features, node_features_or_None).
    For datasets without explicit node features, node_features is None.
    """
    name = name.upper()
    if name == "CSL":
        g, l, nc, nf = load_csl_dataset(root=root)
        return g, l, nc, nf, None
    elif name == "EXP":
        g, l, nc, nf = load_exp_dataset(root=root)
        return g, l, nc, nf, None
    elif name == "SR25":
        g, l, nc, nf = load_sr25_dataset()
        return g, l, nc, nf, None
    elif name == "ENZYMES":
        return load_enzymes_dataset(root=root)
    elif name == "PROTEINS":
        return load_proteins_dataset(root=root)
    else:
        raise ValueError(f"Unknown dataset: {name}. "
                         f"Supported: CSL, EXP, SR25, ENZYMES, PROTEINS")


# ---------------------------------------------------------------------------
#  Count-based DHN: pre-compute hom counts as node features
# ---------------------------------------------------------------------------
# With constant node features, the full DHN reduces to counting homomorphisms
# per node. This variant avoids massive Hom tables by precomputing per-node
# counts and encoding them as features, making large patterns practical.

def precompute_hom_counts(
    graphs: List[nx.Graph],
    patterns: List[str],
    graph_ids: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """Precompute per-node homomorphism counts for all graphs and patterns.

    Returns (node_df, features_tensor) where features_tensor has shape
    (total_nodes, 1 + len(patterns)).  Column 0 is a constant 1 feature,
    columns 1..P are the log1p(hom_count) for each pattern.
    """
    if graph_ids is None:
        graph_ids = list(range(len(graphs)))

    rows = []
    for gid, G in zip(graph_ids, graphs):
        counts_per_node: Dict[int, List[int]] = {}
        for pat in patterns:
            for node in sorted(G.nodes()):
                if node not in counts_per_node:
                    counts_per_node[node] = []
                homs = enumerate_homomorphisms(G, pat, node)
                counts_per_node[node].append(len(homs))
        for node in sorted(G.nodes()):
            rows.append({"graph_id": gid, "n": node, "_counts": counts_per_node[node]})

    node_df = pd.DataFrame([{"graph_id": r["graph_id"], "n": r["n"]} for r in rows])
    counts = np.array([r["_counts"] for r in rows], dtype=np.float32)
    features = np.column_stack([np.ones(len(rows), dtype=np.float32), np.log1p(counts)])
    return node_df, torch.from_numpy(features)


def precompute_walk_counts(
    graphs: List[nx.Graph],
    patterns: List[str],
    graph_ids: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """Precompute per-node non-injective closed walk counts via matrix powers.

    For cycle pattern Ck, the non-injective count at node u is [A^k]_{u,u}.
    For clique pattern Kk, falls back to injective enumeration (cliques are
    inherently injective on simple graphs).

    Returns (node_df, features_tensor) matching the format of precompute_hom_counts.
    """
    if graph_ids is None:
        graph_ids = list(range(len(graphs)))

    rows = []
    for gid, G in zip(graph_ids, graphs):
        A = nx.adjacency_matrix(G).toarray().astype(np.float64)
        n = A.shape[0]
        counts_per_node: Dict[int, List[float]] = {node: [] for node in sorted(G.nodes())}
        for pat in patterns:
            k = pattern_size(pat)
            if is_cycle(pat):
                Ak = np.linalg.matrix_power(A, k)
                for node in sorted(G.nodes()):
                    counts_per_node[node].append(float(Ak[node, node]))
            else:
                for node in sorted(G.nodes()):
                    homs = enumerate_homomorphisms(G, pat, node)
                    counts_per_node[node].append(float(len(homs)))
        for node in sorted(G.nodes()):
            rows.append({"graph_id": gid, "n": node, "_counts": counts_per_node[node]})

    node_df = pd.DataFrame([{"graph_id": r["graph_id"], "n": r["n"]} for r in rows])
    counts = np.array([r["_counts"] for r in rows], dtype=np.float32)
    features = np.column_stack([np.ones(len(rows), dtype=np.float32), np.log1p(counts)])
    return node_df, torch.from_numpy(features)


def build_walk_count_db(
    graphs: List[nx.Graph],
    patterns: List[str],
    labels: Optional[torch.Tensor] = None,
    graph_ids: Optional[List[int]] = None,
    node_features: Optional[torch.Tensor] = None,
) -> dict:
    """Build a RelNN db using non-injective closed walk counts as node features.

    If node_features is provided, they are concatenated after the walk counts.
    """
    if graph_ids is None:
        graph_ids = list(range(len(graphs)))

    node_df, features = precompute_walk_counts(graphs, patterns, graph_ids)
    if node_features is not None:
        features = torch.cat([features, node_features], dim=1)
    db: dict = {"Node": (node_df, features)}

    if labels is not None:
        label_df = pd.DataFrame({"graph_id": graph_ids})
        db["GraphLabel"] = (label_df, labels.view(-1, 1).float())

    return db


def generate_count_dhn_program(
    d_in: int,
    d_hidden: int = 64,
    num_classes: int = 10,
    readout: str = "sum",
    lr: float = 0.001,
    epochs: int = 500,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    dropout: float = 0.0,
) -> Tuple[str, str, str]:
    """Generate a simple MLP-on-counts RelNN program.

    The node features already encode hom counts, so we just need an MLP
    to transform them and a graph-level readout.

    Optional training alignment with official DHN (gear/dhn): optimizer=\"adamw\",
    weight_decay=0.01, dropout=0.05.
    """
    # MLP body: optionally wrap with Dropout after each hidden layer
    if dropout > 0:
        drop_def = f"Dropout1 = Dropout({dropout}) .\n"
        # ReLU -> Dropout after L1 and L2
        mlp_body = (
            f"MLP_L3(ReLU()(Dropout1(MLP_L2(ReLU()(Dropout1(MLP_L1(z)))))))"
        )
    else:
        drop_def = ""
        mlp_body = "MLP_L3(ReLU()(MLP_L2(ReLU()(MLP_L1(z)))))"

    define_prog = f"""#lang:relnn
{drop_def}MLP_L1 = Linear({d_in}, {d_hidden}) .
MLP_L2 = Linear({d_hidden}, {d_hidden}) .
MLP_L3 = Linear({d_hidden}, {d_hidden}) .
Classifier = Linear({d_hidden}, {num_classes}) .

H(graph_id, n; {mlp_body}) :- Node(graph_id, n; z) .
GraphEmb(graph_id; {readout}(z)) :- H(graph_id, n; z) .
GraphLogits(graph_id; Classifier(z)) :- GraphEmb(graph_id; z) .
"""

    fit_params = f"epochs={epochs}, lr={lr}, optimizer=\"{optimizer}\""
    if weight_decay != 0:
        fit_params += f", weight_decay={weight_decay}"
    fit_prog = (
        f"#lang:relnn\n"
        f"?fit <{fit_params}> "
        f"Loss(; CrossEntropyLoss()(z_pred, z_label)) :- "
        f"GraphLogits(graph_id; z_pred), GraphLabel(graph_id; z_label) .\n"
    )

    pred_prog = (
        f"#lang:relnn\n"
        f"?pred Predictions(graph_id; ArgMax()(z)) :- GraphLogits(graph_id; z) .\n"
    )

    return define_prog, fit_prog, pred_prog


def build_count_dhn_db(
    graphs: List[nx.Graph],
    patterns: List[str],
    labels: Optional[torch.Tensor] = None,
    graph_ids: Optional[List[int]] = None,
) -> dict:
    """Build a RelNN db using precomputed hom counts as node features."""
    if graph_ids is None:
        graph_ids = list(range(len(graphs)))

    node_df, features = precompute_hom_counts(graphs, patterns, graph_ids)
    db: dict = {"Node": (node_df, features)}

    if labels is not None:
        label_df = pd.DataFrame({"graph_id": graph_ids})
        db["GraphLabel"] = (label_df, labels.view(-1, 1).float())

    return db


# ---------------------------------------------------------------------------
#  Pure-RelNN DHN: compute homomorphisms via cyclic joins (no Python precompute)
# ---------------------------------------------------------------------------

def _cycle_join_rules(pat: str, h_in: str, d_in: int, d_hidden: int,
                      d_k: int, layer_num: int,
                      injective: bool = False,
                      mu_n_layers: int = 3,
                      mu_dropout: float = 0.0) -> Tuple[str, str, str]:
    """Generate RelNN rules that compute Hom table + transforms for a cycle
    pattern using cyclic self-joins on the Edge relation.

    When injective=False (default, matching the paper), no inequality filters
    are added -- homomorphisms are general (closed walks).
    When injective=True, pairwise != filters enforce simple cycles.

    Returns (transform_defs, rules, agg_name).
    """
    k = pattern_size(pat)
    pos_labels = POSITION_LABELS[:k]
    pat_safe = pat.replace("C", "Cyc").replace("K", "Kli")
    agg_name = f"{pat_safe}_Agg_{layer_num}"

    defs_parts = []
    rule_lines = []

    if k == 2:
        # C2: just Edge itself -- each directed edge is a rooted C2 homomorphism
        # Hom_C2(graph_id, u, v) := Edge(graph_id, u, v)
        # Per-position transforms join directly with Edge and H

        for pos_idx, pos in enumerate(pos_labels):
            t_name = f"{pat_safe}_T{pos_idx}_{layer_num}"
            mu_prefix = f"Mu_{pat_safe}_{pos_idx}_{layer_num}"
            mu_defs, mu_call = _mlp_expr("z", d_in, d_hidden, d_k, mu_prefix,
                                         n_layers=mu_n_layers, dropout=mu_dropout)
            defs_parts.append(mu_defs)

            if pos_idx == 0:
                node_rel = f"{h_in}(graph_id, {pos}; z)"
                rule_lines.append(
                    f"{t_name}(graph_id, u, v; {mu_call}) :- "
                    f"Edge(graph_id, u, v; _), {node_rel} ."
                )
            else:
                node_rel = f"{h_in}_{pos}(graph_id, {pos}; z)"
                rule_lines.append(
                    f"{t_name}(graph_id, u, v; {mu_call}) :- "
                    f"Edge(graph_id, u, v; _), {node_rel} ."
                )

        z_vars = [f"z{i}" for i in range(k)]
        t_names = [f"{pat_safe}_T{i}_{layer_num}" for i in range(k)]
        t_joins = ", ".join(
            f"{t_names[i]}(graph_id, u, v; {z_vars[i]})" for i in range(k)
        )
        product = " * ".join(z_vars)
        rule_lines.append(
            f"{agg_name}(graph_id, u; sum({product})) :- {t_joins} ."
        )
    else:
        # Ck (k >= 3): cyclic join on k Edge aliases with inequality filters.
        # Edge_1(graph_id, u, v), Edge_2(graph_id, v, w), ...,
        # Edge_k(graph_id, <last>, u)  -- closes the cycle back to root u.
        # Injectivity: all position vars must be pairwise distinct.

        # Edge aliases (one per position in the cycle)
        for i in range(k):
            src = pos_labels[i]
            dst = pos_labels[(i + 1) % k]
            rule_lines.append(
                f"Edge_{pat_safe}_{i}(graph_id, {src}, {dst}; z) :- "
                f"Edge(graph_id, {src}, {dst}; z) ."
            )

        # Hom relation via cyclic join.
        # Pass through the first edge's embedding (always 1.0) so each
        # join row gets its own embedding row.
        # When injective=True, add pairwise != filters for simple cycles.
        hom_cols = ", ".join(["graph_id"] + pos_labels)
        edge_join_parts = []
        for i in range(k):
            emb_var = "z" if i == 0 else "_"
            edge_join_parts.append(
                f"Edge_{pat_safe}_{i}(graph_id, {pos_labels[i]}, {pos_labels[(i+1)%k]}; {emb_var})"
            )
        edge_joins = ", ".join(edge_join_parts)
        if injective:
            ineq_filters = ", ".join(
                f"{pos_labels[i]} != {pos_labels[j]}"
                for i in range(k) for j in range(i + 1, k)
            )
            rule_lines.append(
                f"Hom_{pat_safe}({hom_cols}; z) :- "
                f"{edge_joins}, {ineq_filters} ."
            )
        else:
            rule_lines.append(
                f"Hom_{pat_safe}({hom_cols}; z) :- "
                f"{edge_joins} ."
            )
        rule_lines.append("")

        # Per-position transforms on the Hom relation
        hom_rel = f"Hom_{pat_safe}"
        for pos_idx, pos in enumerate(pos_labels):
            t_name = f"{pat_safe}_T{pos_idx}_{layer_num}"
            mu_prefix = f"Mu_{pat_safe}_{pos_idx}_{layer_num}"
            mu_defs, mu_call = _mlp_expr("z", d_in, d_hidden, d_k, mu_prefix,
                                         n_layers=mu_n_layers, dropout=mu_dropout)
            defs_parts.append(mu_defs)

            if pos_idx == 0:
                node_rel = f"{h_in}(graph_id, {pos}; z)"
            else:
                node_rel = f"{h_in}_{pos}(graph_id, {pos}; z)"
            rule_lines.append(
                f"{t_name}({hom_cols}; {mu_call}) :- "
                f"{hom_rel}({hom_cols}; _), {node_rel} ."
            )

        # Product + aggregate
        z_vars = [f"z{i}" for i in range(k)]
        t_names = [f"{pat_safe}_T{i}_{layer_num}" for i in range(k)]
        t_joins = ", ".join(
            f"{t_names[i]}({hom_cols}; {z_vars[i]})" for i in range(k)
        )
        product = " * ".join(z_vars)
        rule_lines.append(
            f"{agg_name}(graph_id, u; sum({product})) :- {t_joins} ."
        )

    return "".join(defs_parts), "\n".join(rule_lines), agg_name


def _clique_join_rules(pat: str, h_in: str, d_in: int, d_hidden: int,
                       d_k: int, layer_num: int,
                       injective: bool = False,
                       mu_n_layers: int = 3,
                       mu_dropout: float = 0.0) -> Tuple[str, str, str]:
    """Generate RelNN rules that compute Hom table + transforms for a clique
    pattern using all-pairs Edge joins.

    A Kk clique requires C(k,2) edge joins (every pair of nodes connected).
    When injective=True, pairwise != filters are added.

    Returns (transform_defs, rules, agg_name).
    """
    k = pattern_size(pat)
    pos_labels = POSITION_LABELS[:k]
    pat_safe = pat.replace("C", "Cyc").replace("K", "Kli")
    agg_name = f"{pat_safe}_Agg_{layer_num}"

    defs_parts = []
    rule_lines = []

    if k == 2:
        # K2 = edge, same as C2
        for pos_idx, pos in enumerate(pos_labels):
            t_name = f"{pat_safe}_T{pos_idx}_{layer_num}"
            mu_prefix = f"Mu_{pat_safe}_{pos_idx}_{layer_num}"
            mu_defs, mu_call = _mlp_expr("z", d_in, d_hidden, d_k, mu_prefix,
                                         n_layers=mu_n_layers, dropout=mu_dropout)
            defs_parts.append(mu_defs)

            if pos_idx == 0:
                node_rel = f"{h_in}(graph_id, {pos}; z)"
            else:
                node_rel = f"{h_in}_{pos}(graph_id, {pos}; z)"
            rule_lines.append(
                f"{t_name}(graph_id, u, v; {mu_call}) :- "
                f"Edge(graph_id, u, v; _), {node_rel} ."
            )

        z_vars = [f"z{i}" for i in range(k)]
        t_names = [f"{pat_safe}_T{i}_{layer_num}" for i in range(k)]
        t_joins = ", ".join(
            f"{t_names[i]}(graph_id, u, v; {z_vars[i]})" for i in range(k)
        )
        product = " * ".join(z_vars)
        rule_lines.append(
            f"{agg_name}(graph_id, u; sum({product})) :- {t_joins} ."
        )
    else:
        # Kk (k >= 3): all-pairs edge joins.
        # One Edge alias per pair (i, j) where i < j.
        edge_pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        for idx, (i, j) in enumerate(edge_pairs):
            src, dst = pos_labels[i], pos_labels[j]
            rule_lines.append(
                f"Edge_{pat_safe}_{i}_{j}(graph_id, {src}, {dst}; z) :- "
                f"Edge(graph_id, {src}, {dst}; z) ."
            )

        hom_cols = ", ".join(["graph_id"] + pos_labels)
        edge_join_parts = []
        for idx, (i, j) in enumerate(edge_pairs):
            emb_var = "z" if idx == 0 else "_"
            edge_join_parts.append(
                f"Edge_{pat_safe}_{i}_{j}(graph_id, "
                f"{pos_labels[i]}, {pos_labels[j]}; {emb_var})"
            )
        edge_joins = ", ".join(edge_join_parts)
        if injective:
            ineq_filters = ", ".join(
                f"{pos_labels[i]} != {pos_labels[j]}"
                for i in range(k) for j in range(i + 1, k)
            )
            rule_lines.append(
                f"Hom_{pat_safe}({hom_cols}; z) :- "
                f"{edge_joins}, {ineq_filters} ."
            )
        else:
            rule_lines.append(
                f"Hom_{pat_safe}({hom_cols}; z) :- "
                f"{edge_joins} ."
            )
        rule_lines.append("")

        hom_rel = f"Hom_{pat_safe}"
        for pos_idx, pos in enumerate(pos_labels):
            t_name = f"{pat_safe}_T{pos_idx}_{layer_num}"
            mu_prefix = f"Mu_{pat_safe}_{pos_idx}_{layer_num}"
            mu_defs, mu_call = _mlp_expr("z", d_in, d_hidden, d_k, mu_prefix,
                                         n_layers=mu_n_layers, dropout=mu_dropout)
            defs_parts.append(mu_defs)

            if pos_idx == 0:
                node_rel = f"{h_in}(graph_id, {pos}; z)"
            else:
                node_rel = f"{h_in}_{pos}(graph_id, {pos}; z)"
            rule_lines.append(
                f"{t_name}({hom_cols}; {mu_call}) :- "
                f"{hom_rel}({hom_cols}; _), {node_rel} ."
            )

        z_vars = [f"z{i}" for i in range(k)]
        t_names = [f"{pat_safe}_T{i}_{layer_num}" for i in range(k)]
        t_joins = ", ".join(
            f"{t_names[i]}({hom_cols}; {z_vars[i]})" for i in range(k)
        )
        product = " * ".join(z_vars)
        rule_lines.append(
            f"{agg_name}(graph_id, u; sum({product})) :- {t_joins} ."
        )

    return "".join(defs_parts), "\n".join(rule_lines), agg_name


def _pattern_join_rules(pat: str, h_in: str, d_in: int, d_hidden: int,
                        d_k: int, layer_num: int,
                        injective: bool = False,
                        mu_n_layers: int = 3,
                        mu_dropout: float = 0.0) -> Tuple[str, str, str]:
    """Dispatch to the appropriate join rule generator for a pattern."""
    kw = dict(injective=injective, mu_n_layers=mu_n_layers, mu_dropout=mu_dropout)
    if is_cycle(pat):
        return _cycle_join_rules(pat, h_in, d_in, d_hidden, d_k, layer_num, **kw)
    elif is_clique(pat):
        return _clique_join_rules(pat, h_in, d_in, d_hidden, d_k, layer_num, **kw)
    else:
        raise ValueError(f"Unknown pattern type: {pat}")


def generate_pure_dhn_program(config: DHNConfig) -> Tuple[str, str, str]:
    """Generate a DHN program that computes homomorphisms via RelNN joins.

    Unlike generate_dhn_program(), this version does NOT require pre-computed
    Hom tables.  It only needs Node and Edge as input relations, and derives
    the homomorphism tables internally using cyclic/clique self-joins on Edge.

    Returns (define_program, fit_program, predict_program).
    """
    lines = []
    transform_defs = []
    num_layers = len(config.patterns_per_layer)

    lines.append(f"d_in = {config.d_in} .")
    lines.append(f"d_k = {config.d_k} .")
    lines.append(f"d_hidden = {config.d_hidden} .")
    lines.append(f"num_classes = {config.num_classes} .")
    lines.append("")
    lines.append("H0(graph_id, n; z) :- Node(graph_id, n; z) .")
    lines.append("")

    prev_out_dim = config.d_in  # tracks output dim of previous layer
    for layer_idx, patterns in enumerate(config.patterns_per_layer):
        layer_num = layer_idx + 1
        h_in = f"H{layer_idx}"
        d_layer_in = prev_out_dim

        # Position alias rules
        needed_pos = _all_position_labels(patterns)
        for pos in needed_pos:
            lines.append(
                f"{h_in}_{pos}(graph_id, {pos}; z) :- {h_in}(graph_id, {pos}; z) ."
            )
        if needed_pos:
            lines.append("")

        agg_names = []
        for pat in patterns:
            defs, rules, agg_name = _pattern_join_rules(
                pat, h_in, d_layer_in, config.d_hidden, config.d_k, layer_num,
                injective=config.injective,
                mu_n_layers=config.mu_n_layers,
                mu_dropout=config.mu_dropout,
            )
            transform_defs.append(defs)
            lines.append(rules)
            lines.append("")
            agg_names.append(agg_name)

        h_out = f"H{layer_num}"
        use_concat = config.pattern_combine == "concat" and len(agg_names) > 1

        if use_concat:
            rho_in_dim = len(patterns) * config.d_k
        else:
            rho_in_dim = config.d_k

        # For concat with 2 layers, the next layer's input dim is the concat dim
        d_out_for_next = rho_in_dim if use_concat and config.mu_n_layers == 2 else config.d_k

        rho_prefix = f"Rho_{layer_num}"
        rho_defs, rho_call_template = _mlp_expr(
            "__Z__", rho_in_dim, config.d_hidden, config.d_k, rho_prefix
        )
        transform_defs.append(rho_defs)

        if use_concat:
            z_agg_vars = [f"z_{i}" for i in range(len(agg_names))]
            concat_expr = f"Concat({', '.join(z_agg_vars)})"
            rho_input = rho_call_template.replace("__Z__", concat_expr)
            join_parts = ", ".join(
                f"{agg_names[i]}(graph_id, n; z_{i})"
                for i in range(len(agg_names))
            )
            lines.append(f"{h_out}(graph_id, n; {rho_input}) :- {join_parts} .")
        else:
            all_pats_name = f"AllPats_{layer_num}"
            sum_pats_name = f"SumPats_{layer_num}"
            union_parts = " | ".join(
                f"{agg_names[i]}(graph_id, n; z)" for i in range(len(agg_names))
            )
            lines.append(f"{all_pats_name}(graph_id, n; z) :- {union_parts} .")
            lines.append(f"{sum_pats_name}(graph_id, n; sum(z)) :- {all_pats_name}(graph_id, n; z) .")
            rho_input = rho_call_template.replace("__Z__", "z")
            lines.append(f"{h_out}(graph_id, n; {rho_input}) :- {sum_pats_name}(graph_id, n; z) .")
        lines.append("")
        prev_out_dim = config.d_k  # Rho always outputs d_k

    # Graph readout + classifier
    final_h = f"H{num_layers}"
    classifier_def = f"Classifier = Linear({config.d_k}, {config.num_classes}) .\n"
    transform_defs.append(classifier_def)
    lines.append(f"GraphEmb(graph_id; {config.readout}(z)) :- {final_h}(graph_id, n; z) .")
    lines.append("GraphLogits(graph_id; Classifier(z)) :- GraphEmb(graph_id; z) .")
    lines.append("")

    all_defs = "".join(transform_defs)
    all_rules = "\n".join(lines)
    define_program = f"#lang:relnn\n{all_defs}\n{all_rules}"

    fit_program = (
        f"#lang:relnn\n"
        f"?fit <epochs={config.epochs}, lr={config.lr}> "
        f"Loss(; CrossEntropyLoss()(z_pred, z_label)) :- "
        f"GraphLogits(graph_id; z_pred), GraphLabel(graph_id; z_label) .\n"
    )

    predict_program = (
        f"#lang:relnn\n"
        f"?pred Predictions(graph_id; ArgMax()(z)) :- GraphLogits(graph_id; z) .\n"
    )

    return define_program, fit_program, predict_program


def build_pure_dhn_db(
    graphs: List[nx.Graph],
    labels: Optional[torch.Tensor] = None,
    node_features: Optional[torch.Tensor] = None,
    graph_ids: Optional[List[int]] = None,
) -> dict:
    """Build a RelNN db with only Node and Edge (no pre-computed Hom tables).

    Use with generate_pure_dhn_program() which computes homomorphisms
    via cyclic joins inside RelNN.
    """
    if graph_ids is None:
        graph_ids = list(range(len(graphs)))

    # Node relation
    node_rows = []
    for gid, G in zip(graph_ids, graphs):
        for n in sorted(G.nodes()):
            node_rows.append({"graph_id": gid, "n": n})
    node_df = pd.DataFrame(node_rows)

    if node_features is None:
        node_features = torch.ones(len(node_df), 1)

    db: dict = {"Node": (node_df, node_features)}

    # Edge relation (both directions for undirected graphs)
    edge_rows = []
    for gid, G in zip(graph_ids, graphs):
        for u, v in G.edges():
            edge_rows.append({"graph_id": gid, "src": u, "dst": v})
            edge_rows.append({"graph_id": gid, "src": v, "dst": u})
    edge_df = pd.DataFrame(edge_rows)
    edge_emb = torch.ones(len(edge_df), 1)
    db["Edge"] = (edge_df, edge_emb)

    # Labels
    if labels is not None:
        label_df = pd.DataFrame({"graph_id": graph_ids})
        db["GraphLabel"] = (label_df, labels.view(-1, 1).float())

    return db
