"""Assemble ``LightRAG`` + ``RAGAnything`` from ``PipelineSettings``"""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional
from app.modules.lightrag import LightRAG
from app.modules.lightrag.utils import logger
from app.modules.raganything import RAGAnything, RAGAnythingConfig
from raganything.parser import Parser, register_parser, get_parser
from app.core.config import PipelineSettings
from app.rag_pipeline.model_funcs import (build_embedding_func, build_llm_func, build_vision_func)
import os
import raganything.utils as _rag_utils
import raganything.processor as _rag_processor

_CHUNK_STRATEGY = os.getenv("LIGHTRAG_CHUNK_STRATEGY", "F").strip().upper()

class _CustomParseAdapter(Parser):
    def check_installation(self) -> bool:
        return True
    def parse_document(self, file_path, output_dir=None, **kwargs):
        raise NotImplementedError("CustomParseAdapter is a stub — parsing happens before insert_content_list.")

register_parser("custom", _CustomParseAdapter)

async def _insert_text_content_via_pipeline(lightrag, input, split_by_character=None, split_by_character_only=False, ids=None, file_paths=None):
    """Enqueue text via the pipeline so ``process_options`` can select Fixed/Recursive/Semantic/Paragraph."""
    _rag_utils.logger.info(f"Starting text content insertion into LightRAG (strategy={_CHUNK_STRATEGY})...")
    await lightrag.apipeline_enqueue_documents(
        input=input,
        ids=ids,
        file_paths=file_paths,
        process_options=_CHUNK_STRATEGY,
    )
    await lightrag.apipeline_process_enqueue_documents()
    _rag_utils.logger.info("Text content insertion complete")

# Patch both the definition site and processor's imported reference.
_rag_utils.insert_text_content = _insert_text_content_via_pipeline
_rag_processor.insert_text_content = _insert_text_content_via_pipeline

def _build_rag_anything_config(settings: PipelineSettings) -> RAGAnythingConfig:
    """Project ``PipelineSettings`` onto ``RAGAnythingConfig``."""
    return RAGAnythingConfig(
        working_dir=settings.working_dir,
        parser=settings.parser,
        parse_method=settings.parse_method,
        parser_output_dir=settings.parser_output_dir,
        enable_image_processing=settings.image_processing_enabled,
        enable_table_processing=settings.enable_table_processing,
        enable_equation_processing=settings.enable_equation_processing,
    )


def _build_lightrag_kwargs(settings: PipelineSettings) -> Dict[str, Any]:
    """Project ``PipelineSettings`` onto ``LightRAG`` kwargs."""
    kwargs: Dict[str, Any] = {
        "working_dir": settings.working_dir,
        "workspace": settings.workspace,
        "kv_storage": settings.kv_storage,
        "vector_storage": settings.vector_storage,
        "graph_storage": settings.graph_storage,
        "doc_status_storage": settings.doc_status_storage,
        "llm_model_name": settings.llm_model,
        "llm_model_max_async": settings.llm_max_async,
        "enable_llm_cache": settings.enable_llm_cache,
        "max_parallel_insert": settings.max_parallel_insert,
    }
    # Escape hatch overrides anything above.
    kwargs.update(settings.extra_lightrag_kwargs)
    return kwargs


async def _construct_lightrag(settings: PipelineSettings) -> LightRAG:
    """Build a ``LightRAG`` instance and initialize its storages + pipeline."""
    llm_func = build_llm_func(settings)
    embedding_func = build_embedding_func(settings)
    rag = LightRAG(
        **_build_lightrag_kwargs(settings),
        llm_model_func=llm_func,
        embedding_func=embedding_func,
    )
    await rag.initialize_storages()
    return rag

@asynccontextmanager
async def build_pipeline(settings: Optional[PipelineSettings] = None, *, lightrag: Optional[LightRAG] = None) -> AsyncIterator[RAGAnything]:
    """Async context manager yielding a ready ``RAGAnything`` instance."""
    settings = settings or PipelineSettings()
    settings.require()

    owns_lightrag = lightrag is None
    if owns_lightrag:
        lightrag = await _construct_lightrag(settings)

    rag = RAGAnything(
        lightrag=lightrag,
        llm_model_func=build_llm_func(settings),
        vision_model_func=build_vision_func(settings),
        embedding_func=build_embedding_func(settings),
        config=_build_rag_anything_config(settings),
    )

    try:
        yield rag
    finally:
        try:
            if owns_lightrag:
                await rag.finalize_storages()
            else:
                for cache in (
                    getattr(rag, "parse_cache", None),
                    getattr(rag, "multimodal_status_cache", None),
                ):
                    if cache is not None:
                        await cache.finalize()
        except Exception:
            logger.debug("finalize_storages raised on shutdown", exc_info=True)
