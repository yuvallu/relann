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

# %%
from __future__ import annotations

# %% [markdown]
# # Log Utils
#
# > Logging utilities: `checkLogs` context manager for temporary log-level overrides.

# %%
import logging
from contextlib import contextmanager
from pathlib import Path

# %%
class FlushingStreamHandler(logging.StreamHandler):
    """A `StreamHandler` that flushes after every emit so output appears immediately."""
    def emit(self, record):
        super().emit(record)
        self.flush()

# %%
DEFAULT_LOG_FORMAT = "%(name)s - %(levelname)s - %(message)s"

@contextmanager
def checkLogs(
    level: int = logging.DEBUG,
    name: str = '__main__',
    toFile: str | Path | None = None,
    format: str = DEFAULT_LOG_FORMAT,
):
    """Context manager that temporarily raises the logging level for a namespace.

    Args:
        level:  Logging level to set (default ``logging.DEBUG``).
        name:   Logger name / module namespace to adjust (default root).
        toFile: Optional path; if given, logs are also written to this file.
        format: Log format string.

    Yields:
        The :class:`logging.Logger` whose level was raised.
    """
    target = logging.getLogger(name)
    previous_level = target.getEffectiveLevel()
    target.setLevel(level)

    added_sh = False
    if not target.handlers:
        sh = FlushingStreamHandler()
        sh.setFormatter(logging.Formatter(format))
        target.addHandler(sh)
        added_sh = True

    fh = None
    if toFile is not None:
        fh = logging.FileHandler(toFile)
        fh.setFormatter(logging.Formatter(format))
        target.addHandler(fh)

    try:
        yield target
    finally:
        target.setLevel(previous_level)
        if fh is not None:
            target.removeHandler(fh)
        if added_sh:
            target.removeHandler(sh)

# %% [markdown]
# ### Usage example
#
# ```python
# from relann.log_utils import checkLogs
#
# # See all DEBUG output from the engine module:
# with checkLogs(name='relann.engine'):
#     engine.fit(...)
#
# # Write logs to a file:
# with checkLogs(name='relann.era_operations', toFile='era_debug.log'):
#     engine.fit(...)
# ```
