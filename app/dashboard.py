"""Deprecated launcher — kept for backward compatibility.

The dashboard and the agent API are now a single process. Use ``run.py``
(``make server``) which serves the WebUI at ``/webui`` and the agent API at
``/docs`` on the same port, sharing one LightRAG + RAGAnything instance.

Running this module simply delegates to that unified entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _prepare_sys_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    modules_dir = repo_root / "app" / "modules"
    for p in (str(modules_dir), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


_prepare_sys_path()


def main() -> None:
    from run import main as unified_main

    print(
        "[dashboard] The dashboard is now served by the unified app. "
        "Delegating to run.py (WebUI at /webui, API docs at /docs)."
    )
    unified_main()


if __name__ == "__main__":
    main()
