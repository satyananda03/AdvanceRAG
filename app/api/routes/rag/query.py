from __future__ import annotations
from typing import Any, List, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from app.api.dependencies import get_pipeline
from app.rag_pipeline import retrieve, retrieve_multimodal

router = APIRouter(tags=["query"])

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = Field(default="mix", description="query mode local, global, hybrid, naive, or mix")
    top_k: Optional[int] = None
    chunk_top_k: Optional[int] = None
    enable_rerank: Optional[bool] = None
    user_prompt: Optional[str] = None


class MultimodalQueryRequest(QueryRequest):
    multimodal_content: List[dict] = Field(
        ...,
        min_length=1,
        description="Multimodal items (equation / table / image).",
    )


class QueryResponse(BaseModel):
    answer: str
    mode: str


def _forwarded_kwargs(payload: QueryRequest) -> dict[str, Any]:
    return {
        k: v
        for k, v in {
            "top_k": payload.top_k,
            "chunk_top_k": payload.chunk_top_k,
            "enable_rerank": payload.enable_rerank,
            "user_prompt": payload.user_prompt,
        }.items()
        if v is not None
    }


@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, rag=Depends(get_pipeline)) -> QueryResponse:
    answer = await retrieve(
        rag, payload.query, mode=payload.mode, **_forwarded_kwargs(payload)
    )
    return QueryResponse(answer=answer, mode=payload.mode)


@router.post("/query-multimodal", response_model=QueryResponse)
async def query_multimodal(
    payload: MultimodalQueryRequest, rag=Depends(get_pipeline)
) -> QueryResponse:
    answer = await retrieve_multimodal(
        rag,
        payload.query,
        payload.multimodal_content,
        mode=payload.mode,
        **_forwarded_kwargs(payload),
    )
    return QueryResponse(answer=answer, mode=payload.mode)