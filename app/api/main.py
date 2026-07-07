"""Unified API — single FastAPI process (default port 8000).

One process, one ``LightRAG`` instance, one ``RAGAnything`` pipeline. It serves:

    * The LightRAG **dashboard**: WebUI at ``/webui`` plus the native
      ``/documents``, ``/query``, ``/graph`` and Ollama-emulation routes.
    * The **agent / programmatic** surface: ``/rag-anything/*`` and ``/agent/*``.

Single source of truth
-----------------------
The dashboard and the agent API share the *same* initialized ``LightRAG``
instance, wrapped by the *same* ``RAGAnything``:

    * **Ingestion** — the dashboard's ``/documents/upload`` is routed through
      ``app.rag_pipeline.ingest_file`` (the exact function agents call), so both
      paths use RAG-Anything's multimodal pipeline. See the ``multimodal_ingest``
      hook wired below.
    * **Retrieval** — because the dashboard's native ``/query`` routes and
      ``rag_pipeline.retrieve`` both call ``LightRAG.aquery`` on the *same*
      instance over the *same* storages, results are identical. The native
      query routes are intentionally left untouched to preserve the WebUI's
      streaming / structured-response contract.

Lifecycle
---------
The dashboard's native lifespan keeps ownership of the ``LightRAG`` instance
(storage init + data migration + finalize). ``RAGAnything`` is layered on top
via ``build_pipeline(lightrag=...)`` which reuses that instance without
re-initializing or double-finalizing it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI

from app.api.routes.agent import router as agent_router
from app.api.routes.rag import router as rag_router
from app.rag_pipeline import build_pipeline, ingest_file


def _build_dashboard_app() -> FastAPI:
    """Construct the LightRAG dashboard app from the vendored fork.

    This gives us the WebUI mount, native document/query/graph routes, auth,
    LLM/embedding wiring and the storage-owning lifespan — all reused as-is.
    """
    from app.modules.lightrag.api.config import global_args, initialize_config
    from app.modules.lightrag.api.lightrag_server import create_app as create_dashboard_app

    initialize_config()
    return create_dashboard_app(global_args)

def _make_multimodal_ingester(app: FastAPI):
    async def _ingest(file_path: Path, *, track_id: Optional[str] = None) -> None:
        rag_anything = app.state.rag_anything
        await ingest_file(
            rag_anything,
            file_path,
            file_name=file_path.name,   
        )
    return _ingest


def create_app() -> FastAPI:
    """Assemble the unified application."""
    app = _build_dashboard_app()

    # The fork's create_app installed a lifespan that owns the LightRAG
    # instance. Wrap it so we can layer the shared RAGAnything on top without
    # touching the fork's storage init / migration / finalize logic.
    dashboard_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with dashboard_lifespan(app):
            # LightRAG is now initialized + migrated by the native lifespan.
            shared_lightrag = app.state.rag
            async with build_pipeline(lightrag=shared_lightrag) as rag_anything:
                app.state.rag_anything = rag_anything
                # Wire dashboard uploads into the shared multimodal pipeline.
                shared_lightrag.multimodal_ingest = _make_multimodal_ingester(app)
                try:
                    yield
                finally:
                    shared_lightrag.multimodal_ingest = None

    app.router.lifespan_context = lifespan

    # Agent + programmatic RAG endpoints. get_pipeline() reads
    # app.state.rag_anything, set above in the lifespan.
    app.include_router(rag_router)
    app.include_router(agent_router)

    return app
