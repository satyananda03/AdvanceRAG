from __future__ import annotations
from typing import Any, List, Optional
from app.modules.lightrag.utils import logger
from app.modules.raganything import RAGAnything


VALID_MODES = ("local", "global", "hybrid", "naive", "mix", "bypass")


def _check_mode(mode: str) -> None:
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid mode {mode!r}. Expected one of {VALID_MODES}."
        )


async def retrieve(
    rag: RAGAnything,
    query: str,
    *,
    mode: str = "mix",
    **query_kwargs: Any,
) -> str:
    """Text-only retrieval.

    Delegates to ``RAGAnything.aquery`` which itself delegates to
    ``LightRAG.aquery``. That means results are identical to whatever the
    LightRAG WebUI returns for the same query.

    Args:
        rag: initialised ``RAGAnything`` from :func:`build_pipeline`.
        query: user question.
        mode: retrieval mode. Defaults to ``mix`` (recommended with reranker).
        **query_kwargs: forwarded to ``QueryParam`` (``top_k``, ``chunk_top_k``,
            ``enable_rerank``, ``max_total_tokens``, ``user_prompt``, ...).
    """
    _check_mode(mode)
    logger.debug(f"[retrieve] mode={mode} query={query[:80]!r}")
    return await rag.aquery(query, mode=mode, **query_kwargs)

async def retrieve_multimodal(
        rag: RAGAnything,
        query: str,
        multimodal_content: List[dict],
        *,
        mode: str = "mix",
        **query_kwargs: Any,
    ) -> str:
    """Retrieval augmented with query-time multimodal payloads.

    ``multimodal_content`` is a list of dicts like::

        [
            {"type": "equation", "latex": "P(d|q) = …", "equation_caption": "…"},
            {"type": "table", "table_body": "…", "table_caption": "…"},
            {"type": "image", "img_path": "/abs/or/rel.png", "img_caption": "…"},
        ]

    The multimodal items are processed by RAG-Anything's specialised
    processors before being folded into the query context.
    """
    _check_mode(mode)
    if not multimodal_content:
        raise ValueError(
            "multimodal_content is empty; use retrieve() for text-only queries."
        )
    logger.debug(
        f"[retrieve-mm] mode={mode} n_items={len(multimodal_content)} "
        f"query={query[:80]!r}"
    )
    return await rag.aquery_with_multimodal(
        query, multimodal_content, mode=mode, **query_kwargs
    )
