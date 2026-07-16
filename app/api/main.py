from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional
from fastapi import FastAPI
from app.rag_pipeline import build_pipeline, ingest_file
from app.api.routes import ingestion_router, chat_router, deletion_router

def _build_dashboard_app() -> FastAPI:
    """Construct the LightRAG dashboard app from the vendored fork"""
    from app.modules.lightrag.api.config import global_args, initialize_config
    from app.modules.lightrag.api.lightrag_server import create_app as create_dashboard_app
    from app.core.logging import setup_logging
    initialize_config()
    setup_logging()
    return create_dashboard_app(global_args)

def _make_multimodal_ingester(app: FastAPI):
    async def _ingest(file_path: Path, *, track_id: Optional[str] = None) -> None:
        rag_anything = app.state.rag_anything
        await ingest_file(rag_anything, file_path, file_name=file_path.name)
    return _ingest

def create_app() -> FastAPI:
    """Assemble the unified application."""
    app = _build_dashboard_app()
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
                    from app.core.workspace_pool import workspace_pool
                    await workspace_pool.shutdown()
    app.router.lifespan_context = lifespan
    app.include_router(chat_router)
    app.include_router(ingestion_router)
    app.include_router(deletion_router)
    return app