"""FastAPI dependencies.

The pipeline lifecycle is owned by ``app.api.main.lifespan``: one
``RAGAnything`` instance is built at startup and stashed on
``app.state.rag_anything``. Route handlers pull it via :func:`get_pipeline`.
"""
from __future__ import annotations
from fastapi import HTTPException, Request
from app.modules.raganything import RAGAnything

def get_pipeline(request: Request) -> RAGAnything:
    """Return the shared pipeline stored on ``app.state`` by the lifespan."""
    rag = getattr(request.app.state, "rag_anything", None)
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialised. Server is still starting.",
        )
    return rag