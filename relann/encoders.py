"""
Built-in content encoders for RelNN RHS ``[...]`` brackets.

These are standard ``torch.nn.Module``s resolved like ``Linear`` / ``ArgMax`` (see
``tensor_term_compiler.resolve_op``). Prefer explicit encoders for production;
``HashBucketTextEncoder`` is the zero-dependency default used for bare ``[text_col]``.
"""

from __future__ import annotations

import zlib
from typing import Any, List

import torch
import torch.nn as nn

__all__ = ["HashBucketTextEncoder"]


def _stable_bucket(token: str, num_buckets: int) -> int:
    """CRC32-based bucket (stable across runs; not Python's salted ``hash()``)."""
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    return int(zlib.crc32(token.encode("utf-8")) % num_buckets)


class HashBucketTextEncoder(nn.Module):
    """
    Lightweight text encoder: whitespace tokenization, CRC32 hash into ``nn.Embedding`` rows,
    mean-pool tokens per row → ``(N, dim)``. No external model download.

    Use explicitly::

        Feat(id; [HashBucketTextEncoder(1024, 16)(bio)]) :- T(id, bio; z) .

    Same class is used as the implicit default for bare ``[bio]`` on text/object columns.
    """

    def __init__(self, num_buckets: int = 1024, dim: int = 16):
        super().__init__()
        if num_buckets <= 0 or dim <= 0:
            raise ValueError("num_buckets and dim must be positive")
        self.num_buckets = int(num_buckets)
        self.dim = int(dim)
        self.embed = nn.Embedding(self.num_buckets, self.dim)

    def extra_repr(self) -> str:
        return f"num_buckets={self.num_buckets}, dim={self.dim}"

    def forward(self, x: Any) -> torch.Tensor:
        # pd.Series (strings) or list-like
        if hasattr(x, "tolist"):
            texts = [str(v) if v is not None and not (isinstance(v, float) and str(v) == "nan") else "" for v in x.tolist()]
        elif isinstance(x, str):
            texts = [x]
        else:
            texts = [str(v) for v in x]

        device = self.embed.weight.device
        rows: List[torch.Tensor] = []
        for t in texts:
            toks = t.split()
            if not toks:
                toks = ["<empty>"]
            idx = torch.tensor(
                [_stable_bucket(tok, self.num_buckets) for tok in toks],
                dtype=torch.long,
                device=device,
            )
            vecs = self.embed(idx)
            rows.append(vecs.mean(dim=0))
        return torch.stack(rows, dim=0)
