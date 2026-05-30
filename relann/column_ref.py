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

# %% [markdown]
# # column_ref
#
# > ColumnRef: normalized column reference by input and column indices. Used by join_conditions and group_by_indices.

# %%
"""ColumnRef: normalized column reference by input and column indices. Used by join_conditions and group_by_indices."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ColumnRef:
    """
    Represents a normalized column reference using input and column indices.

    Args:
        input_idx: Which input this references (0 = first input, 1 = second input, etc.)
        column_idx: Which column in that input (0-based)
    """
    input_idx: int
    column_idx: int

    def __str__(self) -> str:
        return f"input{self.input_idx}.{self.column_idx}"

    def __repr__(self) -> str:
        return f"ColumnRef({self.input_idx}, {self.column_idx})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ColumnRef):
            return NotImplemented
        return (self.input_idx == other.input_idx and self.column_idx == other.column_idx)

    def __hash__(self) -> int:
        return hash((self.input_idx, self.column_idx))
