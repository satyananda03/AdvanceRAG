"""RAG-Anything endpoints — the single-source-of-truth surface for agents
and any client that needs multimodal ingestion / retrieval.
"""

from fastapi import APIRouter

from app.api.routes.rag.documents import router as documents_router
from app.api.routes.rag.query import router as query_router

router = APIRouter(prefix="/rag-anything", tags=["rag-anything"])
router.include_router(documents_router)
router.include_router(query_router)


__all__ = ["router"]
