from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class PipelineSettings(BaseSettings):
    # =====================================================================
    # STORAGE
    # =====================================================================
    working_dir: str = "./rag_storage"
    workspace: str = ""
    kv_storage: str = Field(
        default="PGKVStorage", validation_alias="LIGHTRAG_KV_STORAGE"
    )
    vector_storage: str = Field(
        default="PGVectorStorage", validation_alias="LIGHTRAG_VECTOR_STORAGE"
    )
    graph_storage: str = Field(
        default="PGGraphStorage", validation_alias="LIGHTRAG_GRAPH_STORAGE"
    )
    doc_status_storage: str = Field(
        default="PGDocStatusStorage",
        validation_alias="LIGHTRAG_DOC_STATUS_STORAGE",
    )

    # =====================================================================
    # LLM (OpenAI-compatible) — main / extract role
    # =====================================================================
    llm_binding: str = Field(default="openai", validation_alias="LLM_BINDING")
    llm_binding_host: Optional[str] = Field(
        default=None, validation_alias="LLM_BINDING_HOST"
    )
    llm_binding_api_key: Optional[str] = Field(
        default=None, validation_alias="LLM_BINDING_API_KEY"
    )
    llm_model: str = Field(default="gpt-4o-mini", validation_alias="LLM_MODEL")
    llm_max_async: int = Field(default=4, validation_alias="MAX_ASYNC_LLM")

    # =====================================================================
    # VISION / VLM — uses LightRAG role-specific env vars (VLM_LLM_*)
    # =====================================================================
    vlm_process_enable: bool = Field(
        default=True, validation_alias="VLM_PROCESS_ENABLE"
    )
    vlm_model: Optional[str] = Field(
        default=None, validation_alias="VLM_LLM_MODEL"
    )
    vlm_binding_host: Optional[str] = Field(
        default=None, validation_alias="VLM_LLM_BINDING_HOST"
    )
    vlm_binding_api_key: Optional[str] = Field(
        default=None, validation_alias="VLM_LLM_BINDING_API_KEY"
    )
    vlm_binding: Optional[str] = Field(
        default=None, validation_alias="VLM_LLM_BINDING"
    )
    vlm_max_async: Optional[int] = Field(
        default=None, validation_alias="VLM_MAX_ASYNC_LLM"
    )
    vlm_timeout: Optional[int] = Field(
        default=None, validation_alias="VLM_LLM_TIMEOUT"
    )
    vlm_max_image_bytes: int = Field(
        default=5242880, validation_alias="VLM_MAX_IMAGE_BYTES"
    )
    vlm_min_image_pixel: int = Field(
        default=64, validation_alias="VLM_MIN_IMAGE_PIXEL"
    )

    # =====================================================================
    # EMBEDDING (OpenAI-compatible)
    # =====================================================================
    embedding_binding: str = Field(
        default="openai", validation_alias="EMBEDDING_BINDING"
    )
    embedding_binding_host: Optional[str] = Field(
        default=None, validation_alias="EMBEDDING_BINDING_HOST"
    )
    embedding_binding_api_key: Optional[str] = Field(
        default=None, validation_alias="EMBEDDING_BINDING_API_KEY"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small", validation_alias="EMBEDDING_MODEL"
    )
    embedding_dim: int = Field(default=1536, validation_alias="EMBEDDING_DIM")
    embedding_token_limit: int = Field(
        default=8192, validation_alias="EMBEDDING_TOKEN_LIMIT"
    )
    embedding_batch_num: int = Field(
        default=32, validation_alias="EMBEDDING_BATCH_NUM"
    )

    # =====================================================================
    # CHUNKING — Accuracy-critical tuning parameters
    # =====================================================================
    chunk_size: int = Field(default=1200, validation_alias="CHUNK_SIZE")
    chunk_overlap_size: int = Field(
        default=100, validation_alias="CHUNK_OVERLAP_SIZE"
    )

    # -- Recursive character chunker (process_options=R) --
    chunk_r_size: Optional[int] = Field(
        default=None, validation_alias="CHUNK_R_SIZE"
    )
    chunk_r_overlap_size: Optional[int] = Field(
        default=None, validation_alias="CHUNK_R_OVERLAP_SIZE"
    )

    # -- Paragraph semantic chunker (process_options=P) --
    chunk_p_size: Optional[int] = Field(
        default=None, validation_alias="CHUNK_P_SIZE"
    )
    chunk_p_overlap_size: Optional[int] = Field(
        default=None, validation_alias="CHUNK_P_OVERLAP_SIZE"
    )

    # =====================================================================
    # ENTITY EXTRACTION — Accuracy-critical tuning parameters
    # =====================================================================
    entity_extraction_use_json: bool = Field(
        default=True, validation_alias="ENTITY_EXTRACTION_USE_JSON"
    )
    force_llm_summary_on_merge: int = Field(
        default=8, validation_alias="FORCE_LLM_SUMMARY_ON_MERGE"
    )
    summary_max_tokens: int = Field(
        default=1200, validation_alias="SUMMARY_MAX_TOKENS"
    )
    summary_language: str = Field(
        default="English", validation_alias="SUMMARY_LANGUAGE"
    )
    max_extract_input_tokens: int = Field(
        default=20480, validation_alias="MAX_EXTRACT_INPUT_TOKENS"
    )
    max_extraction_records: int = Field(
        default=100, validation_alias="MAX_EXTRACTION_RECORDS"
    )
    max_extraction_entities: int = Field(
        default=40, validation_alias="MAX_EXTRACTION_ENTITIES"
    )

    # =====================================================================
    # RETRIEVAL — Accuracy-critical tuning parameters
    # =====================================================================
    top_k: int = Field(default=40, validation_alias="TOP_K")
    chunk_top_k: int = Field(default=20, validation_alias="CHUNK_TOP_K")
    cosine_threshold: float = Field(
        default=0.2, validation_alias="COSINE_THRESHOLD"
    )
    max_entity_tokens: int = Field(
        default=6000, validation_alias="MAX_ENTITY_TOKENS"
    )
    max_relation_tokens: int = Field(
        default=8000, validation_alias="MAX_RELATION_TOKENS"
    )
    max_total_tokens: int = Field(
        default=30000, validation_alias="MAX_TOTAL_TOKENS"
    )
    enable_content_headings: bool = Field(
        default=True, validation_alias="ENABLE_CONTENT_HEADINGS"
    )

    # =====================================================================
    # MULTIMODAL CONTEXT — surrounding text budget for VLM prompts
    # =====================================================================
    surrounding_leading_max_tokens: int = Field(
        default=2000, validation_alias="SURROUNDING_LEADING_MAX_TOKENS"
    )
    surrounding_trailing_max_tokens: int = Field(
        default=2000, validation_alias="SURROUNDING_TRAILING_MAX_TOKENS"
    )

    # =====================================================================
    # RAG-ANYTHING PARSER + MULTIMODAL
    # =====================================================================
    parser: str = Field(default="mineru", validation_alias="LIGHTRAG_PARSER_ENGINE")
    parse_method: str = "auto"
    parser_output_dir: str = Field(default="./output", validation_alias="OUTPUT_DIR")
    enable_image_processing: bool = True
    enable_table_processing: bool = True
    enable_equation_processing: bool = True

    # =====================================================================
    # RERANKING
    # =====================================================================
    rerank_binding: Optional[str] = Field(
        default=None, validation_alias="RERANK_BINDING"
    )
    rerank_model: Optional[str] = Field(
        default=None, validation_alias="RERANK_MODEL"
    )

    # =====================================================================
    # PIPELINE CONCURRENCY & CACHE
    # =====================================================================
    enable_llm_cache: bool = Field(
        default=True, validation_alias="ENABLE_LLM_CACHE"
    )
    max_parallel_insert: int = Field(
        default=3, validation_alias="MAX_PARALLEL_INSERT"
    )

    # Escape hatch: extra kwargs forwarded to ``LightRAG(**kwargs)`` beyond
    # the fields above. Prefer adding a proper field over using this.
    extra_lightrag_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # =====================================================================
    # EXTERNAL SERVICE KEYS
    # =====================================================================
    llamaparse_api_key: str = Field(
        default="", validation_alias="LLAMAPARSE_API_KEY"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"

    # =====================================================================
    # DERIVED / COMPUTED PROPERTIES
    # =====================================================================
    @property
    def vlm_enabled(self) -> bool:
        """VLM is only 'on' when both the flag and a model are set."""
        return self.vlm_process_enable and bool(self.vlm_model)

    @property
    def image_processing_enabled(self) -> bool:
        """Image processor requires a working VLM."""
        return self.enable_image_processing and self.vlm_enabled

    @property
    def resolved_vlm_binding_host(self) -> Optional[str]:
        """Fall back to main LLM host if VLM host not explicitly set."""
        return self.vlm_binding_host or self.llm_binding_host

    @property
    def resolved_vlm_binding_api_key(self) -> Optional[str]:
        """Fall back to main LLM API key if VLM key not explicitly set."""
        return self.vlm_binding_api_key or self.llm_binding_api_key

    # =====================================================================
    # VALIDATION
    # =====================================================================
    def require(self) -> None:
        """Fail fast on missing critical fields with an actionable message."""
        missing = []
        if not self.llm_binding_host:
            missing.append("LLM_BINDING_HOST")
        if not self.llm_binding_api_key:
            missing.append("LLM_BINDING_API_KEY")
        if not self.embedding_binding_host:
            missing.append("EMBEDDING_BINDING_HOST")
        if not self.embedding_binding_api_key:
            missing.append("EMBEDDING_BINDING_API_KEY")
        if missing:
            raise RuntimeError(
                "Pipeline is missing required env vars: "
                + ", ".join(missing)
                + ". Set them in .env or the shell before starting."
            )


settings = PipelineSettings()
