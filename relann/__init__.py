__version__ = "0.1.0"

# On Windows, the default console encoding (cp1252) can't print common Unicode
# characters used in our demo cells (e.g. ✓, →, ⇒, em-dashes). Reconfigure
# stdout/stderr to UTF-8 so `print(...)` works the same as on Linux/macOS.
# Jupyter wraps stdout in its own stream which already handles UTF-8 and lacks
# `.reconfigure`; we silently skip that case.
import sys as _sys
if _sys.platform == "win32":
    for _stream in (_sys.stdout, _sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass
    del _stream
del _sys

from relann.scaffold import Scaffold, WeightMapper, ForwardHook, ComparisonResult, scaffold_decorator
from relann.comparison import SessionComparison, TrainResult, EvalResult, ForwardResult, SyncResult
from relann.encoders import HashBucketTextEncoder
