"""
Minimal repro for join-key/schema aliasing bug (2025-01-17).

Goal:
- Ensure `orderby_NodesEmbedding2` has `content_schema == ['node_id']`
- Ensure `join_MsgPassing2` produces a merged content DF without duplicated join-key columns
  (no unexpected *_x/*_y artifacts from a wrong left_on/right_on merge).
"""

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from relann.engine import Engine
from relann.era_operations import _to_er_dict
from relann.parser import parse_and_transform_str
from relann.relnn import term_graph_to_module
from relann.torch_utils import full_seed


def main() -> None:
    full_seed(0)

    device = torch.device("cpu")
    num_nodes = 5
    in_channels = 3
    hidden_channels = 4
    out_channels = 2

    # Small synthetic relations
    nodes_all_df = pd.DataFrame({"node_id": list(range(num_nodes))})
    nodes_all_emb = torch.randn(num_nodes, in_channels, device=device)

    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3)]
    edges_df = pd.DataFrame(
        {"node_id": [s for s, _t in edges], "target_id": [t for _s, t in edges]}
    )
    edges_w = torch.ones(len(edges), 1, device=device)

    program_str = "\n".join(
        [
            # Alias rule: map nodes to nodes_all
            "nodes(node_id; z) :- nodes_all(node_id; z) .",
            # Layer 1
            f"NodesEmbedding1(node_id; Linear({in_channels}, {hidden_channels}, False)(z)) :- nodes(node_id; z) .",
            "MsgPassing1(target_id; sum(z * w)) :- NodesEmbedding1(node_id; z), edges(node_id, target_id; w) .",
            "ReLULayer1(node_id; ReLU(z)) :- MsgPassing1(node_id; z) .",
            # Layer 2
            f"NodesEmbedding2(node_id; Linear({hidden_channels}, {out_channels}, False)(z)) :- ReLULayer1(node_id; z) .",
            "MsgPassing2(target_id; sum(z * w)) :- NodesEmbedding2(node_id; z), edges(node_id, target_id; w) .",
            "ReLULayer2(node_id; ReLU(z)) :- MsgPassing2(node_id; z) .",
            "Output(node_id; z) :- ReLULayer2(node_id; z) .",
        ]
    )

    program = parse_and_transform_str(program_str)
    engine = Engine(db={"nodes_all": (nodes_all_df, nodes_all_emb), "edges": (edges_df, edges_w)}, debug=True)
    engine.add_program(program)

    tg = engine.term_graphs["global"]
    tg = engine.eval_tensor_terms_on_tg(tg)

    model = term_graph_to_module(tg, param_loader=engine).to(device)
    rels = {
        "nodes_all": _to_er_dict((nodes_all_df, nodes_all_emb)),
        "edges": _to_er_dict((edges_df, edges_w)),
    }
    model.instantiate(rels)

    er = model._cache_instantiate.get("orderby_NodesEmbedding2")
    if er is None:
        raise RuntimeError("Expected 'orderby_NodesEmbedding2' in instantiate cache.")

    print("orderby_NodesEmbedding2.content_schema =", er.content_schema)
    print("orderby_NodesEmbedding2.content.columns =", list(er.content.columns))

    join = model._operators["join_MsgPassing2"]
    df_out, _ = join._cached_join
    print("join_MsgPassing2 joined columns =", list(df_out.columns))


if __name__ == "__main__":
    main()

