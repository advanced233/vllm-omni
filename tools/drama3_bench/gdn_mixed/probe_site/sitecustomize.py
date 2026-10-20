# SPDX-License-Identifier: Apache-2.0
"""Loads the Drama 3 GDN probe in every Python process started with this directory on PYTHONPATH.

Does nothing unless DRAMA3_GDN_PROBE=1. Then runs any other sitecustomize this
one shadows, so the environment's own start-up code still runs.
"""

import importlib.machinery
import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))

if os.environ.get("DRAMA3_GDN_PROBE") == "1":
    sys.path.insert(0, os.path.dirname(_here))
    try:
        import drama3_gdn_probe

        drama3_gdn_probe.install()
    except Exception as exc:  # noqa: BLE001 - never break interpreter start-up
        print(f"[drama3-gdn-probe] not installed: {type(exc).__name__}: {exc}", file=sys.stderr)

_rest = [p for p in sys.path if os.path.abspath(p or ".") != _here]
_spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _rest)
if _spec is not None and _spec.loader is not None:
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
