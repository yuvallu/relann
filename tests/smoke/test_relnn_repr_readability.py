"""Smoke tests for RelNN readable repr output."""

import sys
from pathlib import Path

import networkx as nx
from relann.relnn import RelNN

def test_relnn_repr_shows_semantic_module_key_for_qualified_nodes():
    """Repr should include readable semantic keys for dotted node ids."""
    g = nx.DiGraph()
    g.add_node(
        "GCN_of_Papers,Citation.Papers",
        type="data_loader",
        name="Papers",
        output_schema=["pid"],
    )
    module = RelNN(g)
    module._build_operator_modules()
    resolved = module.module_for_node("GCN_of_Papers,Citation.Papers")

    text = repr(module)
    assert "output=data_loader:GCN_of_Papers,Citation.Papers" in text
    assert "op_0000__GCN_of_Papers_Citation_Papers" in text
    assert "DataLoader()" in text
    assert resolved is module._operators["op_0000__GCN_of_Papers_Citation_Papers"]
    assert "GCN_of_Papers,Citation.Papers" not in module._operators
