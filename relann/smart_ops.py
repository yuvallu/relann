"""
Smart operations for RelNN's row-first embedding convention.

RelNN stores per-row embeddings as (E, d) where E is the row/batch dimension.
PyTorch's native broadcasting (right-aligned) doesn't know dim 0 is a shared
batch dim, so mixed-ndim operations produce wrong results:

    (E, 1, 1) * (E, 1)  ->  (E, E, 1)   # PyTorch native -- WRONG
    (E, d) @ (E, d, 1)  ->  (E, E, 1)   # PyTorch native -- WRONG

Smart ops fix this by aligning dimensions before delegating to torch:

    smart_mul:    trailing-unsqueeze the lower-dim operand
    smart_matmul: position-specific unsqueeze (-2 for left, -1 for right)
    smart_transpose: (E, d) -> (E, d, 1) column-vector per row
    smart_view:   reshape feature dims only, preserve row dim

See docs/design/row-first-tensor-convention.md for the full convention.
"""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align_dims_for_elementwise(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the lower-dim operand with trailing size-1 dims until ndims match.

    Scalars (0-D) and same-ndim pairs pass through unchanged so native
    PyTorch broadcasting handles them.

    NOTE: Trailing unsqueeze is correct for all current RelNN patterns where
    the lower-dim operand is a per-row vector (E, d) being combined with a
    per-row matrix (E, a, b).  If a future pattern requires right-aligned
    broadcasting (e.g. (E, 3, 4) + (E, 4) where 4 should match the last dim),
    this function is the single place to adjust.
    """
    if a.ndim == b.ndim or a.ndim == 0 or b.ndim == 0:
        return a, b
    while a.ndim < b.ndim:
        a = a.unsqueeze(-1)
    while b.ndim < a.ndim:
        b = b.unsqueeze(-1)
    return a, b


# ---------------------------------------------------------------------------
# Smart arithmetic
# ---------------------------------------------------------------------------

def smart_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-first batch matmul with auto-unsqueeze.

    When one operand is 2-D and the other is 3-D the 2-D operand is
    unsqueezed so both participate in proper batched matrix multiply:

        (E, d) @ (E, d, 1)  ->  unsqueeze left  -> (E, 1, d) @ (E, d, 1) = (E, 1, 1)
        (E, a, b) @ (E, b)  ->  unsqueeze right  -> (E, a, b) @ (E, b, 1) = (E, a, 1)

    Standard 2-D @ 2-D and 3-D @ 3-D are delegated unchanged.

    Raises ValueError for operands with ndim > 3 (no RelNN pattern should
    produce 4-D+ embeddings in a matmul context).

    TODO: Generalize to ndim > 3 by changing `== 3` guards to `>= 3`.
    torch.matmul already broadcasts batch dims for tensors >= 3-D; only the
    2-D operand needs unsqueezing.  Element-wise ops and smart_transpose
    already handle arbitrary ndim.
    """
    if a.ndim > 3 or b.ndim > 3:
        raise ValueError(
            f"smart_matmul expects at most 3-D operands (E, ...), "
            f"got {a.shape} and {b.shape}"
        )
    if a.ndim == 2 and b.ndim == 3:
        return torch.matmul(a.unsqueeze(-2), b)
    if a.ndim == 3 and b.ndim == 2:
        return torch.matmul(a, b.unsqueeze(-1))
    return torch.matmul(a, b)


def smart_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _align_dims_for_elementwise(a, b)
    return torch.mul(a, b)


def smart_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _align_dims_for_elementwise(a, b)
    return torch.add(a, b)


def smart_sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _align_dims_for_elementwise(a, b)
    return torch.sub(a, b)


def smart_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _align_dims_for_elementwise(a, b)
    return torch.div(a, b)


def smart_pow(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _align_dims_for_elementwise(a, b)
    return torch.pow(a, b)


# ---------------------------------------------------------------------------
# Smart transpose
# ---------------------------------------------------------------------------

def smart_transpose(x: torch.Tensor) -> torch.Tensor:
    """Row-first transpose.

    0-D / 1-D            ->  unchanged (no feature dims to transpose)
    2-D  (E, d)          ->  (E, d, 1)   column-vector per row
    3-D+ (E, a, b, ...)  ->  swap last two dims (standard matrix transpose per row)
    """
    if x.ndim <= 1:
        return x
    if x.ndim == 2:
        return x.unsqueeze(-1)
    return x.transpose(-1, -2)


# ---------------------------------------------------------------------------
# Smart view (reshape feature dims)
# ---------------------------------------------------------------------------

def smart_view(x: torch.Tensor, *shape: int) -> torch.Tensor:
    """Row-first view: reshape feature dimensions, preserve the row dimension.

    0-D / 1-D            ->  reshape entire tensor (no row dim to preserve)
    2-D+ (E, *features)  ->  (E, *shape)  where prod(features) == prod(shape)

    Because dim 0 is always the row dimension, *shape* applies only to
    the feature dims.  This means you cannot use smart_view to reshape
    across the row boundary -- e.g. turning (E, d) into (E*d,).  If you
    need that, use ``torch.reshape`` directly outside the smart-ops layer.

    Uses ``reshape`` internally so non-contiguous tensors (e.g. after
    transpose) work without an explicit ``.contiguous()`` call.
    """
    if not shape:
        raise ValueError("smart_view requires at least one target dimension")
    if x.ndim <= 1:
        return x.reshape(*shape)
    # Preserve dim 0 (row dimension), reshape only feature dims
    feature_numel = 1
    for s in x.shape[1:]:
        feature_numel *= s
    target_numel = 1
    for s in shape:
        target_numel *= s
    if feature_numel != target_numel:
        raise RuntimeError(
            f"smart_view: cannot reshape features {tuple(x.shape[1:])} "
            f"(numel={feature_numel}) into {shape} (numel={target_numel})"
        )
    return x.reshape(x.shape[0], *shape)


__all__ = [
    "smart_matmul",
    "smart_mul",
    "smart_add",
    "smart_sub",
    "smart_div",
    "smart_pow",
    "smart_transpose",
    "smart_view",
]
