"""Multimodal RAG pipeline library.

Public API used by both HTTP routes (``app.api``) and agent code:

    build_pipeline()             -> async context manager -> RAGAnything
    ingest_file(pipeline, path)  -> single-source-of-truth ingestion
    retrieve(pipeline, query)
    retrieve_multimodal(pipeline, query, multimodal_content)

Any change made here is reflected in the WebUI dashboard AND every agent
that imports this module.
"""

from app.core.config import PipelineSettings
from app.rag_pipeline.factory import build_pipeline
from app.rag_pipeline.ingest import ingest_file
from app.rag_pipeline.retrieve import retrieve, retrieve_multimodal

__all__ = [
    "PipelineSettings",
    "build_pipeline",
    "ingest_file",
    "retrieve",
    "retrieve_multimodal",
]
