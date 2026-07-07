"""Builders for LLM, vision, and embedding functions.

These wrap ``lightrag.llm.openai``'s primitives with settings pulled from
``PipelineSettings``. Both LightRAG and RAG-Anything share the same callables,
so a single ``PipelineSettings`` drives the entire stack.

Design notes
------------
* All functions returned are OpenAI-compatible and forward ``api_key`` +
  ``base_url`` on every call.
* ``build_embedding_func`` returns a ``lightrag.utils.EmbeddingFunc`` which is
  the shape both LightRAG and RAG-Anything expect.
* ``build_vision_func`` mirrors the pattern from RAG-Anything's example: it
  supports three call shapes (``messages=`` for VLM-enhanced query,
  ``image_data=`` for a single-image caption, and plain text fallback).
"""

from __future__ import annotations
from functools import partial
from typing import Any, Callable, List, Optional
from app.modules.lightrag.llm.openai import openai_complete_if_cache, openai_embed
from app.modules.lightrag.utils import EmbeddingFunc
from app.core.config import PipelineSettings

def build_llm_func(settings: PipelineSettings) -> Callable[..., Any]:
    """Return an async LLM callable compatible with LightRAG + RAG-Anything.

    The callable signature matches ``llm_model_func`` expected by both
    frameworks: ``(prompt, system_prompt=None, history_messages=[], **kwargs)``.
    """

    async def llm_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> str:
        return await openai_complete_if_cache(
            settings.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=settings.llm_binding_api_key,
            base_url=settings.llm_binding_host,
            **kwargs,
        )

    return llm_model_func


def build_vision_func(settings: PipelineSettings) -> Optional[Callable[..., Any]]:
    if not settings.vlm_enabled:
        return None

    llm_func = build_llm_func(settings)
    vlm_model = settings.vlm_model
    api_key = settings.resolved_vlm_binding_api_key
    base_url = settings.resolved_vlm_binding_host

    async def vision_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[dict]] = None,
        image_data: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> str:
        # Case 1: caller composed messages themselves (VLM-enhanced query).
        if messages:
            return await openai_complete_if_cache(
                vlm_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        # Case 2: single-image caption.
        if image_data:
            composed: List[dict] = []
            if system_prompt:
                composed.append({"role": "system", "content": system_prompt})
            composed.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                }
            )
            return await openai_complete_if_cache(
                vlm_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=composed,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        # Case 3: no image — degrade to text LLM.
        return await llm_func(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            **kwargs,
        )

    return vision_model_func


def build_embedding_func(settings: PipelineSettings) -> EmbeddingFunc:
    """Return an ``EmbeddingFunc`` sharing ``.env`` credentials.

    We use ``partial`` on ``openai_embed.func`` (the underlying, undecorated
    coroutine) — attempting to wrap an already-wrapped ``EmbeddingFunc``
    fails. See AGENTS.md "Custom Embedding Functions" for the rationale.
    """
    return EmbeddingFunc(
        embedding_dim=settings.embedding_dim,
        max_token_size=settings.embedding_token_limit,
        func=partial(
            openai_embed.func,
            model=settings.embedding_model,
            api_key=settings.embedding_binding_api_key,
            base_url=settings.embedding_binding_host,
        ),
    )
