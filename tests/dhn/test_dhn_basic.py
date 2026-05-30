"""
Unit tests for DHN (Deep Homomorphism Networks) in RelNN.

Tests:
  - Homomorphism enumeration correctness
  - DHN RelNN program generation and parsing
  - DHN forward pass on toy data
  - Graph-level aggregation in RelNN
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import networkx as nx
sys.path.insert(0, str(Path(__file__).resolve().parent))

from relann.session import Session
from relann.torch_utils import full_seed

from dhn_utils import (
    _load_dataset,
    _PAULUS_SR25_ADJ,
    _validate_sr25_graph,
    enumerate_cycle_homomorphisms,
    enumerate_clique_homomorphisms,
    enumerate_homomorphisms,
    precompute_hom_tables,
    precompute_walk_counts,
    generate_dhn_program,
    generate_pure_dhn_program,
    build_dhn_db,
    build_pure_dhn_db,
    build_walk_count_db,
    DHNConfig,
    precompute_hom_counts,
    build_count_dhn_db,
    generate_count_dhn_program,
    load_sr25_dataset,
    load_exp_dataset,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _triangle_graph() -> nx.Graph:
    """Single triangle: 0-1-2-0."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return G

def _square_graph() -> nx.Graph:
    """Square: 0-1-2-3-0."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    return G

def _k4_graph() -> nx.Graph:
    """Complete graph K4."""
    return nx.complete_graph(4)

def _two_triangles_graph() -> nx.Graph:
    """Two triangles sharing edge 0-1: nodes 0,1,2,3."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 3), (3, 1)])
    return G

# ── Homomorphism enumeration tests ──────────────────────────────────────────

class TestCycleHomomorphisms:
    def test_c2_is_edge_count(self):
        """C2 homs rooted at u = neighbors of u (edges incident to u)."""
        G = _triangle_graph()
        homs = enumerate_cycle_homomorphisms(G, 2, root=0)
        assert len(homs) == 2  # neighbors of 0 are {1, 2}
        roots = [h[0] for h in homs]
        assert all(r == 0 for r in roots)

    def test_c3_triangle_in_triangle(self):
        """In a triangle graph, C3 rooted at any node should find exactly 2 homs
        (the triangle traversed CW and CCW)."""
        G = _triangle_graph()
        for node in G.nodes():
            homs = enumerate_cycle_homomorphisms(G, 3, root=node)
            assert len(homs) == 2, f"Expected 2 C3 homs at node {node}, got {len(homs)}"

    def test_c3_in_square(self):
        """A square has no triangles, so C3 homs should be 0."""
        G = _square_graph()
        for node in G.nodes():
            homs = enumerate_cycle_homomorphisms(G, 3, root=node)
            assert len(homs) == 0

    def test_c4_in_square(self):
        """In a square, C4 homs from any node should exist."""
        G = _square_graph()
        homs = enumerate_cycle_homomorphisms(G, 4, root=0)
        assert len(homs) > 0
        for h in homs:
            assert h[0] == 0
            assert len(h) == 4

    def test_c2_in_k4(self):
        """In K4, each node has degree 3, so C2 homs = 3."""
        G = _k4_graph()
        homs = enumerate_cycle_homomorphisms(G, 2, root=0)
        assert len(homs) == 3

    def test_c3_in_k4(self):
        """K4 has 4 triangles. Each node is in 3 of them.
        Each triangle gives 2 directed traversals, so 6 C3 homs per node."""
        G = _k4_graph()
        homs = enumerate_cycle_homomorphisms(G, 3, root=0)
        assert len(homs) == 6

    def test_hom_tuple_structure(self):
        """Each hom should be a tuple starting with root."""
        G = _triangle_graph()
        homs = enumerate_cycle_homomorphisms(G, 3, root=1)
        for h in homs:
            assert isinstance(h, tuple)
            assert h[0] == 1
            assert len(h) == 3

class TestCliqueHomomorphisms:
    def test_k3_equals_triangles(self):
        """K3 homs in a triangle graph from node 0: the triangle in all permutations
        of the non-root vertices."""
        G = _triangle_graph()
        homs = enumerate_clique_homomorphisms(G, 3, root=0)
        assert len(homs) == 2  # (0,1,2) and (0,2,1)
        for h in homs:
            assert h[0] == 0
            assert set(h) == {0, 1, 2}

    def test_k3_in_square(self):
        """No triangles in a square."""
        G = _square_graph()
        homs = enumerate_clique_homomorphisms(G, 3, root=0)
        assert len(homs) == 0

    def test_k4_in_k4(self):
        """K4 has exactly one 4-clique. From root 0, there are 3! = 6 permutations."""
        G = _k4_graph()
        homs = enumerate_clique_homomorphisms(G, 4, root=0)
        assert len(homs) == 6
        for h in homs:
            assert h[0] == 0
            assert set(h) == {0, 1, 2, 3}

    def test_k2_is_edges(self):
        """K2 homs = edges from root."""
        G = _triangle_graph()
        homs = enumerate_clique_homomorphisms(G, 2, root=0)
        assert len(homs) == 2

class TestEnumerateHomomorphisms:
    def test_dispatch_cycle(self):
        G = _triangle_graph()
        homs = enumerate_homomorphisms(G, "C3", root=0)
        assert len(homs) == 2

    def test_dispatch_clique(self):
        G = _k4_graph()
        homs = enumerate_homomorphisms(G, "K4", root=0)
        assert len(homs) == 6

    def test_unknown_pattern(self):
        G = _triangle_graph()
        with pytest.raises(ValueError):
            enumerate_homomorphisms(G, "Z5", root=0)

# ── Pre-compute tables ──────────────────────────────────────────────────────

class TestPrecomputeHomTables:
    def test_basic_structure(self):
        graphs = [_triangle_graph(), _square_graph()]
        tables = precompute_hom_tables(graphs, ["C3", "C2"])
        assert "C3" in tables
        assert "C2" in tables
        c3_df, c3_emb = tables["C3"]
        c2_df, c2_emb = tables["C2"]
        assert list(c3_df.columns) == ["graph_id", "u", "v", "w"]
        assert list(c2_df.columns) == ["graph_id", "u", "v"]

    def test_graph_id_present(self):
        graphs = [_triangle_graph(), _square_graph()]
        tables = precompute_hom_tables(graphs, ["C3"], graph_ids=[10, 20])
        c3_df, c3_emb = tables["C3"]
        mask = (c3_emb.squeeze() > 0).numpy()
        real_rows = c3_df[mask]
        assert set(real_rows["graph_id"].unique()) == {10}

    def test_dummy_rows_for_missing_nodes(self):
        """Graph with no triangles → only dummy rows (embedding=0) in C3 table."""
        graphs = [_square_graph()]
        tables = precompute_hom_tables(graphs, ["C3"])
        c3_df, c3_emb = tables["C3"]
        assert len(c3_df) == 4  # 4 nodes, each gets a dummy row
        assert (c3_emb == 0).all()  # all are dummies

# ── DSL program generation ──────────────────────────────────────────────────

class TestGenerateDHNProgram:
    def test_generates_parseable_program(self):
        """Generated program should parse without error in RelNN."""
        config = DHNConfig(
            patterns_per_layer=[["C2", "C3"]],
            d_in=1, d_k=4, d_hidden=8, num_classes=2,
            epochs=1,
        )
        define_prog, fit_prog, pred_prog = generate_dhn_program(config)
        assert "#lang:relnn" in define_prog
        assert "?fit" in fit_prog
        assert "?pred" in pred_prog
        assert "H0(" in define_prog
        assert "H1(" in define_prog
        assert "GraphEmb(" in define_prog
        assert "GraphLogits(" in define_prog

    def test_two_layer_program(self):
        config = DHNConfig(
            patterns_per_layer=[["C2", "C3"], ["C2"]],
            d_in=1, d_k=4, d_hidden=8, num_classes=3,
            epochs=1,
        )
        define_prog, _, _ = generate_dhn_program(config)
        assert "H0(" in define_prog
        assert "H1(" in define_prog
        assert "H2(" in define_prog

    def test_single_pattern(self):
        config = DHNConfig(
            patterns_per_layer=[["C2"]],
            d_in=1, d_k=4, d_hidden=8, num_classes=2,
            epochs=1,
        )
        define_prog, _, _ = generate_dhn_program(config)
        assert "Hom_C2" in define_prog

# ── End-to-end: DHN forward pass in RelNN ────────────────────────────────────

class TestDHNForwardPass:
    def test_toy_dhn_define_and_predict(self):
        """Minimal DHN: 2 triangles graph, C3 pattern, 1 layer, predict shapes."""
        full_seed(42)
        G = _two_triangles_graph()

        config = DHNConfig(
            patterns_per_layer=[["C2"]],
            d_in=1, d_k=4, d_hidden=8, num_classes=2,
            epochs=1,
        )
        db = build_dhn_db(
            graphs=[G],
            patterns=["C2"],
            labels=torch.tensor([0]),
        )

        define_prog, fit_prog, pred_prog = generate_dhn_program(config)

        session = Session(db=db)
        session.run(define_prog)
        session.run(fit_prog)
        result = session.run(pred_prog)

        assert result is not None
        assert result.embeddings is not None
        assert result.embeddings[0].shape[0] == 1  # 1 graph → 1 prediction

    def test_two_graph_classification(self):
        """Two graphs (triangle vs square) should be classifiable."""
        full_seed(42)
        graphs = [_triangle_graph(), _square_graph()]
        labels = torch.tensor([0, 1])

        config = DHNConfig(
            patterns_per_layer=[["C2"]],
            d_in=1, d_k=4, d_hidden=8, num_classes=2,
            epochs=5, lr=0.01,
        )
        all_patterns = ["C2"]
        db = build_dhn_db(graphs, all_patterns, labels=labels)

        define_prog, fit_prog, pred_prog = generate_dhn_program(config)

        session = Session(db=db)
        session.run(define_prog)
        session.run(fit_prog)
        result = session.run(pred_prog)

        assert result is not None
        preds = result.embeddings[0]
        assert preds.shape[0] == 2  # 2 graphs → 2 predictions

class TestGraphAggregation:
    def test_graph_level_aggregation_in_relnn(self):
        """Verify that RelNN can aggregate node embeddings to graph level."""
        full_seed(42)
        G1 = _triangle_graph()   # 3 nodes
        G2 = _square_graph()     # 4 nodes

        node_df = pd.DataFrame({
            "graph_id": [0, 0, 0, 1, 1, 1, 1],
            "n": [0, 1, 2, 0, 1, 2, 3],
        })
        node_z = torch.randn(7, 2)

        label_df = pd.DataFrame({"graph_id": [0, 1]})
        labels = torch.tensor([[0], [1]], dtype=torch.float32)

        db = {
            "Node": (node_df, node_z),
            "GraphLabel": (label_df, labels),
        }

        session = Session(db=db)
        session.run("""
#lang:relnn
GraphEmb(graph_id; mean(z)) :- Node(graph_id, n; z) .
""")
        result = session.run("""
#lang:relnn
?pred Out(graph_id; z) :- GraphEmb(graph_id; z) .
""")
        assert result is not None
        assert result.embeddings[0].shape == (2, 2)  # 2 graphs, 2-dim embedding

class TestCountBasedDHN:
    def test_hom_count_features(self):
        """Precomputed hom counts should include counts for each pattern."""
        full_seed(42)
        G1 = _triangle_graph()
        G2 = _square_graph()
        node_df, features = precompute_hom_counts([G1, G2], ["C2", "C3"])
        assert features.shape[1] == 3  # 1 constant + 2 patterns
        assert (features[:, 0] == 1.0).all()  # constant feature
        # Triangle graph nodes have C3 > 0
        g1_mask = node_df["graph_id"] == 0
        g1_c3 = features[g1_mask.values, 2]  # C3 counts for triangle graph
        assert (g1_c3 > 0).all()
        # Square graph nodes have C3 = 0
        g2_mask = node_df["graph_id"] == 1
        g2_c3 = features[g2_mask.values, 2]
        assert (g2_c3 == 0).all()

    def test_count_based_forward_pass(self):
        """Count-based variant should run end-to-end."""
        full_seed(42)
        graphs = [_triangle_graph(), _square_graph()]
        labels = torch.tensor([0, 1])
        db = build_count_dhn_db(graphs, ["C2", "C3"], labels=labels)
        d_in = db["Node"][1].shape[1]

        define_prog, fit_prog, pred_prog = generate_count_dhn_program(
            d_in=d_in, d_hidden=8, num_classes=2, epochs=5, lr=0.01,
        )

        session = Session(db=db)
        session.run(define_prog)
        session.run(fit_prog)
        result = session.run(pred_prog)
        assert result is not None
        assert result.embeddings[0].shape[0] == 2

# ── Pure-RelNN tests (no Python preprocessing) ──────────────────────────────

class TestPureDHN:
    """End-to-end tests for pure-RelNN DHN (homomorphisms via Edge joins)."""

    def test_pure_cycle_forward_pass(self):
        """Pure-RelNN with cycle patterns should classify triangle vs square."""
        full_seed(42)
        graphs = [_triangle_graph(), _square_graph()]
        labels = torch.tensor([0, 1])

        config = DHNConfig(
            patterns_per_layer=[["C2", "C3"]],
            d_in=1, d_k=5, d_hidden=10, num_classes=2,
            readout="sum", lr=0.01, epochs=50,
        )
        db = build_pure_dhn_db(graphs, labels=labels)
        define, fit, pred = generate_pure_dhn_program(config)

        session = Session(db=db)
        session.run(define)
        session.run(fit)
        result = session.run(pred)
        preds = result.embeddings[0].view(-1).long()
        assert preds.shape[0] == 2

    def test_pure_two_layer(self):
        """Two-layer pure-RelNN DHN should produce valid predictions."""
        full_seed(42)
        graphs = [_triangle_graph(), _square_graph(), _k4_graph()]
        labels = torch.tensor([0, 1, 2])

        config = DHNConfig(
            patterns_per_layer=[["C2", "C3"], ["C2"]],
            d_in=1, d_k=5, d_hidden=10, num_classes=3,
            readout="sum", lr=0.01, epochs=30,
        )
        db = build_pure_dhn_db(graphs, labels=labels)
        define, fit, pred = generate_pure_dhn_program(config)

        session = Session(db=db)
        session.run(define)
        session.run(fit)
        result = session.run(pred)
        preds = result.embeddings[0].view(-1).long()
        assert preds.shape[0] == 3

    def test_pure_db_has_no_hom_tables(self):
        """build_pure_dhn_db should only contain Node, Edge, GraphLabel."""
        graphs = [_triangle_graph()]
        labels = torch.tensor([0])
        db = build_pure_dhn_db(graphs, labels=labels)
        assert set(db.keys()) == {"Node", "Edge", "GraphLabel"}

    def test_pure_program_contains_edge_joins(self):
        """Generated pure-RelNN program should reference Edge aliases.
        Default (injective=False) should NOT have inequality filters."""
        config = DHNConfig(
            patterns_per_layer=[["C2", "C3"]],
            d_in=1, d_k=4, d_hidden=8, num_classes=2, epochs=1,
        )
        define, _, _ = generate_pure_dhn_program(config)
        assert "Edge_Cyc3_0" in define
        assert "Hom_Cyc3" in define
        assert "u != v" not in define  # non-injective by default

    def test_pure_program_injective_has_filters(self):
        """With injective=True, inequality filters should appear."""
        config = DHNConfig(
            patterns_per_layer=[["C2", "C3"]],
            d_in=1, d_k=4, d_hidden=8, num_classes=2, epochs=1,
            injective=True,
        )
        define, _, _ = generate_pure_dhn_program(config)
        assert "u != v" in define
        assert "u != w" in define
        assert "v != w" in define

class TestWalkCounts:
    """Tests for non-injective walk count computation via matrix powers."""

    def test_walk_counts_match_matrix_power(self):
        """Walk counts for cycles should equal A^k diagonal entries."""
        import numpy as np
        G = _k4_graph()  # K4: 4 nodes, degree 3
        A = nx.adjacency_matrix(G).toarray().astype(float)

        node_df, features = precompute_walk_counts([G], ["C2", "C3", "C4"])
        # features: [constant_1, log1p(C2_count), log1p(C3_count), log1p(C4_count)]
        for node_idx in range(4):
            for k_idx, k in enumerate([2, 3, 4]):
                Ak = np.linalg.matrix_power(A, k)
                expected = Ak[node_idx, node_idx]
                actual_log = features[node_idx, 1 + k_idx].item()
                actual = np.expm1(actual_log)
                assert abs(actual - expected) < 0.5, \
                    f"Node {node_idx}, C{k}: expected {expected}, got {actual}"

    def test_walk_counts_differ_from_injective(self):
        """Non-injective C4 counts should differ from injective on a square graph."""
        G = _square_graph()
        # Injective C4: simple 4-cycles (2 per node in a square: CW and CCW)
        injective_c4 = len(enumerate_cycle_homomorphisms(G, 4, root=0))
        # Non-injective C4: A^4 diagonal
        import numpy as np
        A = nx.adjacency_matrix(G).toarray().astype(float)
        non_injective_c4 = int(np.linalg.matrix_power(A, 4)[0, 0])
        # A^4[0,0] includes degenerate walks like 0->1->0->1->0
        assert non_injective_c4 > injective_c4, \
            f"Non-injective ({non_injective_c4}) should be > injective ({injective_c4})"

    def test_csl_walk_counts_distinguish_classes(self):
        """Non-injective walk counts C2:10 should produce 10 distinct groups for CSL."""
        try:
            graphs, labels, nc, nf, node_features = _load_dataset(
                'CSL', root=str(Path(__file__).resolve().parent / 'data')
            )
        except Exception:
            pytest.skip("CSL dataset not available")

        import numpy as np
        class_vecs = {}
        for i, (G, label) in enumerate(zip(graphs, labels)):
            A = nx.adjacency_matrix(G).toarray().astype(float)
            vec = tuple(int(np.linalg.matrix_power(A, k)[0, 0]) for k in range(2, 11))
            lab = label.item()
            if lab not in class_vecs:
                class_vecs[lab] = vec
        unique = len(set(class_vecs.values()))
        assert unique == 10, f"Expected 10 distinct groups, got {unique}"

    def test_build_walk_count_db(self):
        """build_walk_count_db should produce correct DB structure."""
        graphs = [_triangle_graph(), _square_graph()]
        labels = torch.tensor([0, 1])
        db = build_walk_count_db(graphs, ["C2", "C3"], labels=labels)
        assert "Node" in db
        assert "GraphLabel" in db
        node_df, feats = db["Node"]
        assert feats.shape[1] == 3  # constant + 2 patterns
        assert len(node_df) == 7   # 3 + 4 nodes

class TestCliqueJoins:
    """Tests for pure-RelNN clique detection via all-pairs Edge joins."""

    def test_k3_detection(self):
        """K3 join should detect triangles in K4 but not in a square."""
        full_seed(42)
        g1 = nx.complete_graph(4)  # has triangles
        g2 = nx.cycle_graph(4)    # no triangles
        graphs = [g1, g2]
        labels = torch.tensor([0, 1])

        config = DHNConfig(
            patterns_per_layer=[["C2", "K3"]],
            d_in=1, d_k=5, d_hidden=10, num_classes=2,
            readout="sum", lr=0.01, epochs=50,
        )
        db = build_pure_dhn_db(graphs, labels=labels)
        define, fit, pred = generate_pure_dhn_program(config)

        session = Session(db=db)
        session.run(define)
        session.run(fit)
        result = session.run(pred)
        preds = result.embeddings[0].view(-1).long()
        assert preds.shape[0] == 2

    def test_k3_program_structure(self):
        """Generated program for K3 should have all-pairs Edge aliases."""
        config = DHNConfig(
            patterns_per_layer=[["K3"]],
            d_in=1, d_k=4, d_hidden=8, num_classes=2, epochs=1,
        )
        define, _, _ = generate_pure_dhn_program(config)
        assert "Edge_Kli3_0_1" in define
        assert "Edge_Kli3_0_2" in define
        assert "Edge_Kli3_1_2" in define
        assert "Hom_Kli3" in define

    def test_mixed_cycles_and_cliques(self):
        """C2 + K3:5 pattern mix should produce valid program and predictions."""
        full_seed(42)
        graphs = [nx.complete_graph(5), _triangle_graph(), _square_graph()]
        labels = torch.tensor([0, 1, 2])

        config = DHNConfig(
            patterns_per_layer=[["C2", "K3", "K4"]],
            d_in=1, d_k=5, d_hidden=10, num_classes=3,
            readout="sum", lr=0.01, epochs=30,
        )
        db = build_pure_dhn_db(graphs, labels=labels)
        define, fit, pred = generate_pure_dhn_program(config)

        session = Session(db=db)
        session.run(define)
        session.run(fit)
        result = session.run(pred)
        preds = result.embeddings[0].view(-1).long()
        assert preds.shape[0] == 3

class TestSR25Dataset:
    """Validate that hardcoded SR25 graphs satisfy SR(25,12,5,6) parameters."""

    def test_exactly_15_graphs(self):
        assert len(_PAULUS_SR25_ADJ) == 15

    def test_all_graphs_have_25_nodes(self):
        for i, adj in enumerate(_PAULUS_SR25_ADJ):
            assert len(adj) == 25, f"Graph {i}: expected 25 nodes, got {len(adj)}"

    def test_all_graphs_are_12_regular(self):
        for i, adj in enumerate(_PAULUS_SR25_ADJ):
            for node, nbrs in enumerate(adj):
                assert len(nbrs) == 12, (
                    f"Graph {i}, node {node}: degree {len(nbrs)} != 12"
                )

    def test_sr25_parameters(self):
        """Validate lambda=5, mu=6 for a sample of graphs."""
        import numpy as np
        for idx in [0, 7, 14]:  # first, middle, last
            assert _validate_sr25_graph(_PAULUS_SR25_ADJ[idx]), (
                f"Graph {idx} fails SR(25,12,5,6) validation"
            )

    def test_graphs_are_pairwise_non_isomorphic(self):
        graphs, _, _, _ = load_sr25_dataset()
        for i in range(len(graphs)):
            for j in range(i + 1, len(graphs)):
                assert not nx.is_isomorphic(graphs[i], graphs[j]), (
                    f"Graphs {i} and {j} are isomorphic"
                )

    def test_load_sr25_returns_correct_structure(self):
        graphs, labels, nc, nf = load_sr25_dataset()
        assert len(graphs) == 15
        assert nc == 15
        assert nf == 1
        assert labels.shape == (15,)
        assert set(labels.tolist()) == set(range(15))

class TestEXPDataset:
    """Validate that the real EXP dataset loads correctly."""

    def test_exp_dataset_has_1200_graphs(self):
        data_root = str(Path(__file__).resolve().parent / "data")
        try:
            graphs, labels, nc, nf = load_exp_dataset(root=data_root)
        except Exception:
            pytest.skip("EXP dataset not available (download required)")
        assert len(graphs) == 1200

    def test_exp_balanced_classes(self):
        data_root = str(Path(__file__).resolve().parent / "data")
        try:
            graphs, labels, nc, nf = load_exp_dataset(root=data_root)
        except Exception:
            pytest.skip("EXP dataset not available")
        assert nc == 2
        assert (labels == 0).sum().item() == 600
        assert (labels == 1).sum().item() == 600

    def test_exp_graph_sizes(self):
        data_root = str(Path(__file__).resolve().parent / "data")
        try:
            graphs, labels, nc, nf = load_exp_dataset(root=data_root)
        except Exception:
            pytest.skip("EXP dataset not available")
        sizes = [G.number_of_nodes() for G in graphs]
        assert min(sizes) >= 20
        assert max(sizes) <= 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
