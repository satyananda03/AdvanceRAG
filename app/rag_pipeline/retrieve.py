from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from app.modules.lightrag.utils import logger
from app.modules.lightrag.base import QueryParam
from app.modules.raganything import RAGAnything

VALID_MODES = ("local", "global", "hybrid", "naive", "mix", "bypass")
IMAGE_PATH_PATTERN = re.compile(r"Image Path:\s*([^\r\n]+?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif)(?:\?[^\r\n]*)?)")

@dataclass
class RetrievalResult:
    context: str
    """Raw retrieved context string from LightRAG (chunks, entities, relations)."""
    image_paths: List[str]
    """Absolute image paths extracted from retrieved chunks."""
    raw_data: Optional[Dict[str, Any]] = None
    """Optional raw data dict from LightRAG (entities, relationships, chunks)."""

def check_mode(mode: str) -> None:
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode {mode!r}. Expected one of {VALID_MODES}.")

# ====== Return Retrieved Context Only ======== #
async def retrieve_context(rag: RAGAnything, query: str, *, mode: str = "mix", include_raw_data: bool = False, **query_kwargs: Any) -> RetrievalResult:
    check_mode(mode)
    if rag.lightrag is None:
        raise ValueError("No LightRAG instance available. Process documents first.")
    query_param = QueryParam(mode=mode, only_need_context=True, **query_kwargs)
    query_result = await rag.lightrag.aquery(query, param=query_param)
    # extract content
    if hasattr(query_result, "content"):
        context = query_result.content or ""
        raw_data = query_result.raw_data if include_raw_data else None
    else:
        # Fallback if aquery returns plain string
        context = str(query_result) if query_result else ""
        raw_data = None
    # Extract image paths from the context
    image_paths = IMAGE_PATH_PATTERN.findall(context)
    logger.info(f"[retrieve_context] context_len = {len(context)} images_found = {image_paths}")
    return RetrievalResult(
        context=context,
        image_paths=image_paths,
        raw_data=raw_data,
    )

async def retrieve_multimodal_context(rag: RAGAnything, query: str, multimodal_content: List[dict], *, mode: str = "mix", include_raw_data: bool = False, **query_kwargs: Any) -> RetrievalResult:
    """multimodal_content -> list of query-time multimodal item, contoh :
    [
        {"type": "image", "img_path": "/path/to/new_chart.png"},
        {"type": "table", "table_body": "| A | B |\\n|---|---|\\n| 1 | 2 |"},
    ]"""
    check_mode(mode)
    if not multimodal_content:
        raise ValueError("multimodal_content is empty; use retrieve_context() for text-only queries.")
    if rag.lightrag is None:
        raise ValueError("No LightRAG instance available. Process documents first.")
    enhanced_query = await rag._process_multimodal_query_content(query, multimodal_content)
    query_param = QueryParam(mode=mode, only_need_context=True, **query_kwargs)
    query_result = await rag.lightrag.aquery(enhanced_query, param=query_param)
    # Extract content
    if hasattr(query_result, "content"):
        context = query_result.content or ""
        raw_data = query_result.raw_data if include_raw_data else None
    else:
        context = str(query_result) if query_result else ""
        raw_data = None
    # Extract image paths from context
    image_paths = IMAGE_PATH_PATTERN.findall(context)
    return RetrievalResult(
        context=context,
        image_paths=image_paths,
        raw_data=raw_data,
    )


# ====== Return Final Answer (Include Generation) ======== #
async def retrieve(rag: RAGAnything, query: str, *, mode: str = "mix", **query_kwargs: Any) -> str:
    check_mode(mode)
    return await rag.aquery(query, mode=mode, **query_kwargs)

async def retrieve_multimodal(rag: RAGAnything, query: str, multimodal_content: List[dict], *, mode: str = "mix", **query_kwargs: Any) -> str:
    check_mode(mode)
    if not multimodal_content:
        raise ValueError("multimodal_content is empty; use retrieve() for text-only queries.")
    return await rag.aquery_with_multimodal(query, multimodal_content, mode=mode, **query_kwargs)




