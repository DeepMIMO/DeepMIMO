"""Safe import helpers for optional AODT dependencies.

This module centralizes optional imports used by AODT modules, so that
import-time failures are avoided and users receive a simple, consistent
message on how to install extras.
"""

# Keep this minimal: only import and print warnings.
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore
    print(
        "Warning: AODT features require pandas/pyarrow.\nInstall with: pip install 'deepmimo[aodt]'",
    )
