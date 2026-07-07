"""Runner: unified API on port 8000.

Boots a single FastAPI process that serves both the LightRAG dashboard
(WebUI + native ``/documents``, ``/query``, ``/graph`` routes) and the agent
surface (``/rag-anything/*``, ``/agent/*``). One ``LightRAG`` instance is
shared across both, wrapped by one ``RAGAnything`` pipeline, so ingestion and
retrieval are single-source-of-truth. See ``app/api/main.py``.

    WebUI       → http://localhost:8000/webui
    API docs    → http://localhost:8000/docs
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

def _prepare_sys_path() -> None:
    """Ensure both the repo root and ``app/modules`` are on ``sys.path``."""
    repo_root = Path(__file__).resolve().parent
    modules_dir = repo_root / "app" / "modules"
    for p in (str(modules_dir), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

_prepare_sys_path()

import uvicorn
from app.api.main import create_app

app = create_app()

def main() -> None:
    host = os.getenv("AGENT_API_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_API_PORT", "8000"))
    log_level = os.getenv("AGENT_API_LOG_LEVEL", "info").lower()
    # Pass the app instance directly (not "run:app") so the module is not
    # re-imported and the dashboard app is not built twice.
    uvicorn.run(app, host=host, port=port, log_level=log_level)

if __name__ == "__main__":
    main()
