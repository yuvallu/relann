"""Session.run() should resolve ops from the *caller's* globals (not session.py)."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from relann.session import Session

def test_torch_nn_linear_resolves_from_caller():
    """`Linear` must come from the test module's `import torch.nn as nn`, not from relann.session."""
    s = Session(db={})
    s.run(
        """
#lang:relnn
L = Linear(4, 2) .
"""
    )
    assert "L" in s.engine.symbol_table.get("global", {})

def test_custom_module_class_from_caller():
    class MyMod(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            return x * self.w

    s = Session(db={})
    s.run(
        """
#lang:relnn
M = MyMod() .
"""
    )
    assert "M" in s.engine.symbol_table.get("global", {})
