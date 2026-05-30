"""Scaffold-bucket conftest: force a non-interactive matplotlib backend
before any scaffold test module is imported.

Why:
    tests/scaffold/test_912_scaffold_gcn_cora.py has an inline-demo
    `if test():` block that calls ``matplotlib.pyplot.show()`` to render
    a training-loss comparison plot. Under an interactive backend (the
    OS default on workstations), ``plt.show()`` BLOCKS the import
    waiting for the user to close the figure window. pytest's
    collection step never returns, so combined runs
    (``pytest tests/scaffold/``) wedge forever even though each file
    runs fine alone.

    Forcing the ``Agg`` backend here makes ``plt.show()`` a no-op
    (Agg is non-interactive and renders to memory). The functional
    intent of the demo cell — exercising matplotlib without crashing
    — is preserved. Plots can still be saved via ``plt.savefig`` if
    a future test wants visible output.

Why a conftest rather than the module:
    conftest.py is imported by pytest BEFORE any test module in the
    same directory is collected. Setting the backend here guarantees
    the choice takes effect before any scaffold file runs
    `import matplotlib.pyplot as plt`. Setting it inside the module
    would race with `if test():` evaluation at collection time.

Why not just delete `plt.show()`:
    The visualization cell is a real part of the literate-programming
    intent of the scaffold tests (they read like the original notebook
    they were converted from). Forcing the Agg backend keeps the
    semantics — the code runs, a figure is built, the engine writes
    the data — without the side effect of blocking the test runner.
"""
from __future__ import annotations

import matplotlib

# Must run before any `import matplotlib.pyplot` in this dir's test
# modules; pytest evaluates conftest.py first.
matplotlib.use("Agg", force=True)
