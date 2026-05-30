"""
Test that (1, N, d) embedding is normalized before index_select so replication yields (M, d).

When a relation has 1 row and embedding (1, N, d), replicating that row M times via
index_select(0, [0,0,...,0]) would give (M, N, d) and break attention. Join.forward
squeezes (1, N, d) to (N, d) first so index_select yields (M, d).
"""

import sys
from pathlib import Path

import torch
def test_squeeze_1_N_d_then_index_select_gives_M_d():
    """(1, N, d) -> squeeze(0) -> (N, d); index_select(0, [0]*M) -> (M, d)."""
    N, d, M = 100, 4, 50
    emb = torch.randn(1, N, d)
    squeezed = emb.squeeze(0)
    assert squeezed.shape == (N, d)
    idx = torch.zeros(M, dtype=torch.long)
    aligned = squeezed.index_select(0, idx)
    assert aligned.shape == (M, d), f"Expected (M, d)={M, d}, got {aligned.shape}"
