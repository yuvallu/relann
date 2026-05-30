"""
Parity: rooted cycle tuples from ``precompute_hom_tables`` (Hom_Ck semantics)
vs edge-join enumeration matching ``dhn_ghl_csl_c2_4_edge.relnn`` (directed
Edge, simple cycles only).

If these diverge, the edge RelNN and Hom-based RelNN are not the same object.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

_REPO = Path(__file__).resolve().parents[3]
if str(_DHN := _REPO / "nbs" / "tests" / "dhn") not in sys.path:
    sys.path.insert(0, str(_DHN))

from dhn_utils import precompute_hom_tables  # noqa: E402


def _directed_edge_set(G: nx.Graph) -> set[tuple[int, int]]:
    d: set[tuple[int, int]] = set()
    for a, b in G.edges():
        d.add((a, b))
        d.add((b, a))
    return d


def c3_tuples_from_edge_join(G: nx.Graph, gid: int = 0) -> set[tuple]:
    """
    Tuples (graph_id, n, v, w) as in the RelNN C3_Sparse join:
    Edge(n,v) & Edge(v,w) & Edge(w,n) with n,v,w distinct.
    """
    d = _directed_edge_set(G)
    s: set[tuple] = set()
    for n in G.nodes():
        for v in G[n]:
            if (n, v) not in d:
                continue
            for w in G[v]:
                if w in (n, v):
                    continue
                if (v, w) in d and (w, n) in d and len({n, v, w}) == 3:
                    s.add((gid, n, v, w))
    return s


def c4_tuples_from_edge_join(G: nx.Graph, gid: int = 0) -> set[tuple]:
    """
    (graph_id, n, v, w, p) for Edge(n,v) & Edge(v,w) & Edge(w,p) & Edge(p,n)
    with n,v,w,p all distinct.
    """
    d = _directed_edge_set(G)
    s: set[tuple] = set()
    for n in G.nodes():
        for v in G[n]:
            if (n, v) not in d:
                continue
            for w in G[v]:
                if w in (n, v):
                    continue
                for p in G[w]:
                    if p in (n, v, w):
                        continue
                    if (w, p) in d and (p, n) in d and len({n, v, w, p}) == 4:
                        s.add((gid, n, v, w, p))
    return s


def c2_tuples_from_edge_join(G: nx.Graph, gid: int = 0) -> set[tuple]:
    return {(gid, a, b) for (a, b) in _directed_edge_set(G)}


def _real_hom_tuples(df, emb) -> set[tuple]:
    m = (emb.squeeze() > 0.5).numpy()
    rows = df.loc[m]
    return {tuple(int(x) for x in r) for r in rows.itertuples(index=False, name=None)}


def test_c3_parity_k3():
    G = nx.complete_graph(3)  # triangle 0,1,2
    h = _real_hom_tuples(*precompute_hom_tables([G], ["C3"])["C3"])
    e1 = c3_tuples_from_edge_join(G, 0)
    assert h == e1, f"Hom C3 and edge-join C3 differ: hom={len(h)} edge={len(e1)}"


def test_c2_parity_k3():
    G = nx.complete_graph(3)
    h = _real_hom_tuples(*precompute_hom_tables([G], ["C2"])["C2"])
    e = c2_tuples_from_edge_join(G, 0)
    assert h == e


def test_c4_parity_c4_graph():
    """The 4-cycle graph C4: four nodes in a single ring (no diagonals)."""
    G = nx.cycle_graph(4)  # 0-1-2-3-0
    h = _real_hom_tuples(*precompute_hom_tables([G], ["C4"])["C4"])
    e = c4_tuples_from_edge_join(G, 0)
    assert h == e, f"Hom C4 and edge-join C4 on C4 graph: hom={len(h)} edge={len(e)}"


def test_c3_tuples_two_triangles_sharing_edge():
    """Bowtie-like: two triangles sharing a vertex has two triangular faces; hom should match join."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)])
    h = _real_hom_tuples(*precompute_hom_tables([G], ["C3"])["C3"])
    e = c3_tuples_from_edge_join(G, 0)
    assert h == e
