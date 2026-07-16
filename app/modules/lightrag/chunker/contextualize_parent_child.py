from __future__ import annotations
import asyncio
import os
from typing import Any
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import Tokenizer, logger
import os
from dotenv import load_dotenv
load_dotenv()

PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "2500"))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "200"))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "310"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))
CONTEXTUALIZE_ENABLED = "true"
CONTEXTUALIZE_MAX_ASYNC = 10
CONTEXTUALIZE_MODEL = "amazon.nova-pro-v1:0"

def chunking_parent_child(
    tokenizer: Tokenizer,
    content: str,
    parent_chunk_size: int | None = None,
    child_chunk_size: int | None = None,
    *,
    parent_overlap: int | None = None,
    child_overlap: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[int]]]:
    """Split content into parent chunks, then sub-split each parent into children.

    Uses LangChain's RecursiveCharacterTextSplitter for both levels.

    Returns:
        parent_chunks: list of ``{tokens, content, chunk_order_index}``
        child_chunks:  list of ``{tokens, content, chunk_order_index, parent_index}``
        parent_to_children: mapping ``{parent_index: [child_global_indices]}``
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    parent_size = parent_chunk_size or PARENT_CHUNK_SIZE
    child_size = child_chunk_size or CHILD_CHUNK_SIZE
    p_overlap = parent_overlap if parent_overlap is not None else PARENT_CHUNK_OVERLAP
    c_overlap = child_overlap if child_overlap is not None else CHILD_CHUNK_OVERLAP

    # Estimate chars per token (rough: 4 chars/token for mixed Indonesian/English)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=p_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=c_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Split into parents
    parent_texts = parent_splitter.split_text(content)

    parent_chunks: list[dict[str, Any]] = []
    child_chunks: list[dict[str, Any]] = []
    parent_to_children: dict[int, list[int]] = {}

    child_global_index = 0

    for parent_idx, parent_text in enumerate(parent_texts):
        parent_text_stripped = parent_text.strip()
        if not parent_text_stripped:
            continue

        parent_tokens = len(tokenizer.encode(parent_text_stripped))
        parent_chunks.append(
            {
                "tokens": parent_tokens,
                "content": parent_text_stripped,
                "chunk_order_index": parent_idx,
            }
        )

        # Split parent into children
        child_texts = child_splitter.split_text(parent_text_stripped)
        children_indices: list[int] = []

        for child_text in child_texts:
            child_text_stripped = child_text.strip()
            if not child_text_stripped:
                continue

            child_tokens = len(tokenizer.encode(child_text_stripped))
            child_chunks.append(
                {
                    "tokens": child_tokens,
                    "content": child_text_stripped,
                    "chunk_order_index": child_global_index,
                    "parent_index": parent_idx,
                }
            )
            children_indices.append(child_global_index)
            child_global_index += 1

        parent_to_children[parent_idx] = children_indices

    logger.info(
        f"[parent-child] Split into {len(parent_chunks)} parents, "
        f"{len(child_chunks)} children "
        f"(parent_size={parent_size}, child_size={child_size})"
    )

    return parent_chunks, child_chunks, parent_to_children

CONTEXTUALIZE_SYSTEM_PROMPT = """
    ROLE : You are an expert AI assistant specializing in Document Analysis and Contextual Retrieval. Your primary task is to generate succinct context that situates a specific chunk of text within its parent document for search retrieval.
    LANGUAGE : ALWAYS RESPONSE in Bahasa Indonesia
    """

CONTEXTUALIZE_USER_TEMPLATE = (
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{child_content}\n</chunk>\n\n"
    "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."
)

async def _invoke_contextualize_llm(
        parent_content: str,
        child_content: str,
    ) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    base_url = os.getenv("LLM_BINDING_HOST", "")
    api_key = os.getenv("LLM_BINDING_API_KEY", "")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=CONTEXTUALIZE_MODEL,
        temperature=0.0,
    )

    messages = [
        SystemMessage(content=CONTEXTUALIZE_SYSTEM_PROMPT),
        HumanMessage(content=[
            {
                "type": "text",
                "text": f"<document>\n{parent_content}\n</document>\n\n",
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": CONTEXTUALIZE_USER_TEMPLATE.format(child_content=child_content),
            },
        ]),
    ]

    response = await llm.ainvoke(messages)
    return response.content.strip()


async def contextualize_children(
    parent_chunks: list[dict[str, Any]],
    child_chunks: list[dict[str, Any]],
    llm_func=None,
    *,
    max_concurrent: int | None = None,
) -> list[dict[str, Any]]:
    """Contextualize each child chunk using its parent as context via LLM.

    Menggunakan fungsi LLM khusus (_invoke_contextualize_llm) dengan model
    amazon nova pro yang mendukung prompt caching. Parameter llm_func
    dipertahankan untuk backward compatibility tapi tidak digunakan.
    """
    if not CONTEXTUALIZE_ENABLED:
        logger.info("[parent-child] Contextualization disabled, skipping.")
        return child_chunks

    concurrency = max_concurrent or CONTEXTUALIZE_MAX_ASYNC
    semaphore = asyncio.Semaphore(concurrency)

    async def _contextualize_one(child: dict[str, Any]) -> dict[str, Any]:
        parent_idx = child.get("parent_index", 0)
        parent_content = (
            parent_chunks[parent_idx]["content"]
            if parent_idx < len(parent_chunks)
            else ""
        )

        async with semaphore:
            try:
                context = await _invoke_contextualize_llm(
                    parent_content=parent_content,
                    child_content=child["content"],
                )
                context = (context or "").strip().rstrip(".")
                if context:
                    new_content = f"{context}. {child['content']}"
                else:
                    new_content = child["content"]
            except Exception as e:
                logger.warning(
                    f"[parent-child] Contextualization failed for child "
                    f"{child['chunk_order_index']}: {e}. Using original content."
                )
                new_content = child["content"]

        new_child = dict(child)
        new_child["_original_content"] = child["content"]
        new_child["content"] = new_content
        return new_child

    logger.info(
        f"[parent-child] Contextualizing {len(child_chunks)} children "
        f"(concurrency={concurrency})..."
    )

    tasks = [_contextualize_one(child) for child in child_chunks]
    contextualized = await asyncio.gather(*tasks)

    logger.info("[parent-child] Contextualization complete.")
    return list(contextualized)

def build_parent_to_child_map(
    parent_chunks_dict: dict[str, dict[str, Any]],
    child_chunks_dict: dict[str, dict[str, Any]],
    parent_to_children: dict[int, list[int]],
) -> dict[str, list[str]]:
    """Build mapping from parent chunk_key to list of child chunk_keys.

    Used after entity extraction to redirect source_id from parent → children.

    Args:
        parent_chunks_dict: Output of build_chunks_dict_from_chunking_result for parents.
        child_chunks_dict: Output of build_chunks_dict_from_chunking_result for children.
        parent_to_children: Mapping {parent_index: [child_global_indices]} from chunking step.

    Returns:
        Dict mapping each parent chunk_key to its corresponding child chunk_keys.
    """
    parent_keys = list(parent_chunks_dict.keys())
    child_keys = list(child_chunks_dict.keys())

    mapping: dict[str, list[str]] = {}
    for parent_idx, child_indices in parent_to_children.items():
        if parent_idx >= len(parent_keys):
            continue
        parent_key = parent_keys[parent_idx]
        mapped_children = [
            child_keys[ci] for ci in child_indices if ci < len(child_keys)
        ]
        if mapped_children:
            mapping[parent_key] = mapped_children

    logger.info(
        f"[parent-child] Built source_id redirect map: "
        f"{len(mapping)} parents → {sum(len(v) for v in mapping.values())} children"
    )
    return mapping


def redirect_source_ids(
    chunk_results: list,
    parent_to_child_map: dict[str, list[str]],
) -> None:
    """In-place replace source_id in extraction results from parent → child keys.

    After entity/relation extraction runs on parent chunks, each entity's
    source_id points to a parent chunk_key. This function redirects those
    to the child chunk_keys so that graph retrieval returns child chunks.

    Args:
        chunk_results: List of (maybe_nodes, maybe_edges) from _process_extract_entities.
        parent_to_child_map: Output of build_parent_to_child_map.
    """
    redirected_count = 0

    for maybe_nodes, maybe_edges in chunk_results:
        for _entity_name, entities in maybe_nodes.items():
            for entity in entities:
                source_id = entity.get("source_id", "")
                if source_id in parent_to_child_map:
                    entity["source_id"] = GRAPH_FIELD_SEP.join(
                        parent_to_child_map[source_id]
                    )
                    redirected_count += 1

        for _edge_key, edges in maybe_edges.items():
            for edge in edges:
                source_id = edge.get("source_id", "")
                if source_id in parent_to_child_map:
                    edge["source_id"] = GRAPH_FIELD_SEP.join(
                        parent_to_child_map[source_id]
                    )
                    redirected_count += 1

    logger.info(f"[parent-child] Redirected {redirected_count} source_ids")