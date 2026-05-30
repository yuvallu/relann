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
# # Torch Utils
#
# > Utility functions for PyTorch operations including model inspection, seeding, and tensor comparison

# %%
import logging
import torch
import random
import numpy as np
import os
from typing import Union, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# %% [markdown]
# ## Path Utilities

# %%
def get_project_root() -> Path:
    """Return project root by searching up for `pyproject.toml` from CWD or __file__ location.

    Falls back to legacy nbdev markers (`settings.ini`, `setup.py`) so the function
    still works if it ever runs from an old checkout. Raises FileNotFoundError if
    nothing is found.
    """
    # Start from current directory or __file__ location
    try:
        start_path = Path(__file__).resolve().parent
    except NameError:
        # Interactive session / notebook
        start_path = Path.cwd().resolve()

    MARKERS = ("pyproject.toml", "settings.ini", "setup.py")

    current = start_path
    while current != current.parent:  # stop at filesystem root
        if any((current / m).exists() for m in MARKERS):
            return current
        current = current.parent

    raise FileNotFoundError(
        f"Could not find project root (looking for any of {MARKERS}). "
        f"Searched from: {start_path}"
    )

# %% [markdown]
# ## Model Utilities

# %%
def get_model_weights(model):
    weights = []
    for param in model.parameters():
        weights.extend(param.detach().cpu().numpy().flatten())
    return np.array(weights)

# %%
def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    # Group parameters by module hierarchy
    param_dict = {}
    for name, param in model.named_parameters():
        parts = name.split('.')
        current = param_dict
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = param.numel()
    
    def print_hierarchy(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print('  ' * indent + f"{key}:")
                print_hierarchy(value, indent + 1)
            else:
                print('  ' * indent + f"{key}: {value} parameters")
    
    print("\nParameter hierarchy:")
    print_hierarchy(param_dict)

# %% [markdown]
# ## Seeding Utilities

# %%
def full_seed(seed: Optional[int] = None) -> int:
    """
    Set random seeds for reproducibility. If seed is None, uses PARENT_SEED env var or defaults to 42.
    """
    if seed is None:
        seed_str = os.environ.get("PARENT_SEED", "42")
        try:
            seed = int(seed_str)
        except ValueError:
            seed = 42
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)          # guarantees determinism in Adam etc.
    os.environ["PYTHONHASHSEED"] = str(seed)           # pandas / groupby
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Required for deterministic CuBLAS operations
    
    return seed

# %% [markdown]
# ## Tensor Comparison Utilities

# %%
def equal_up_to_permutation(
    a: Union[torch.Tensor, torch.Tensor],
    b: Union[torch.Tensor, torch.Tensor],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> bool:
    """
    Return True iff `a` and `b` contain the *same multiset of elements*
    (order doesn't matter).

    Parameters
    ----------
    a, b : torch.Tensor (any shape, same `dtype`)
    atol, rtol : float
        Absolute / relative tolerance for floating types
        (set both to zero for bit-exact matching).

    Technical notes
    ---------------
    * All work is delegated to fast C++/CUDA kernels (`unique`, `sort`).
    * For floating tensors we first **bucketise** each value to the nearest
      grid point of size `atol` (i.e. `round(x / atol)`); this guarantees that
      any two numbers that would pass `torch.allclose` land in the same bucket.
      If `atol==rtol==0` the function degrades to exact equality.
    """
    # ---------------------------------------------------------------- basic checks
    if a.shape != b.shape:
        return False
    if a.numel() == 0:          # same empty shape ⇒ equal
        return True
    if a.dtype != b.dtype:
        return False

    # ------------------------------------------------------- flatten once, stay on device
    a1 = a.reshape(-1)
    b1 = b.reshape(-1)

    # ------------------------------------------------------- step 1: (optional) quantise floats
    if a1.is_floating_point() and (atol > 0 or rtol > 0):
        scale = max(atol, rtol * torch.max(a1.abs().max(), b1.abs().max()).item())
        if scale == 0:                      # tensors are all-zero
            return True
        a1 = torch.round(a1 / scale).to(torch.int64)
        b1 = torch.round(b1 / scale).to(torch.int64)

    # ------------------------------------------------------- step 2: unique + counts
    # (works for all integer-convertible dtypes incl. bool)
    uniq_a, cnt_a = torch.unique(a1, sorted=True, return_counts=True)
    uniq_b, cnt_b = torch.unique(b1, sorted=True, return_counts=True)

    return torch.equal(uniq_a, uniq_b) and torch.equal(cnt_a, cnt_b)
