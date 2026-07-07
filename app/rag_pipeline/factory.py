"""Assemble ``LightRAG`` + ``RAGAnything`` from ``PipelineSettings``.

Public entry point is :func:`build_pipeline`, an async context manager that
yields a ready-to-use ``RAGAnything`` instance and guarantees storage
finalisation on exit.

Rationale for a single builder:

* Every downstream caller (LangGraph agent, FastAPI server, CLI) uses the same
  initialization / teardown path, so bugs get fixed once.
* Model funcs are built once and shared by both frameworks, avoiding
  duplicated OpenAI client state.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from app.modules.lightrag import LightRAG
from app.modules.lightrag.kg.shared_storage import initialize_pipeline_status
from app.modules.lightrag.utils import logger
from app.modules.raganything import RAGAnything, RAGAnythingConfig

from app.core.config import PipelineSettings
from app.rag_pipeline.model_funcs import (
    build_embedding_func,
    build_llm_func,
    build_vision_func,
)


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
    """Project ``PipelineSettings`` onto ``LightRAG`` kwargs.

    Keeps the constructor call site trivial: ``LightRAG(**kwargs)``.
    """
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
    await initialize_pipeline_status()
    return rag


@asynccontextmanager
async def build_pipeline(
    settings: Optional[PipelineSettings] = None,
    *,
    lightrag: Optional[LightRAG] = None,
) -> AsyncIterator[RAGAnything]:
    """Async context manager yielding a ready ``RAGAnything`` instance.

    Args:
        settings: pipeline settings. Defaults to ``PipelineSettings()`` which
            loads from ``.env``.
        lightrag: reuse an existing initialized LightRAG (e.g. the one the
            FastAPI server already built). When ``None``, a new one is built.

    Example::

        async with build_pipeline() as rag:
            await rag.process_document_complete("./doc.pdf")
            answer = await rag.aquery("summarize", mode="mix")
    """
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
                # We built the LightRAG, so finalize everything RAGAnything
                # touches (its caches + the underlying LightRAG storages).
                await rag.finalize_storages()
            else:
                # The caller owns the injected LightRAG's lifecycle (e.g. the
                # dashboard server's lifespan). Finalize only the extra caches
                # RAGAnything created so we never double-finalize the shared
                # LightRAG storages.
                for cache in (
                    getattr(rag, "parse_cache", None),
                    getattr(rag, "multimodal_status_cache", None),
                ):
                    if cache is not None:
                        await cache.finalize()
        except Exception:
            # ``finalize_storages`` may run during interpreter shutdown; log
            # but never mask an in-flight exception.
            logger.debug("finalize_storages raised on shutdown", exc_info=True)
