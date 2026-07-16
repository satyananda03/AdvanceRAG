from __future__ import annotations
import asyncio
import shutil
import tempfile
import os
import httpx
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from app.modules.raganything import RAGAnything
from app.modules.lightrag.utils import logger
from app.modules.parser.amazon_textract import parse_document_to_json, reconstruct_content_list

async def _download_image(url: str, output_path: Path) -> bool:
    """Download gambar dari URL ke local path."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            output_path.write_bytes(response.content)
        return True
    except Exception as e:
        logger.error(f"Failed to download image {url}: {e}")
        return False


async def _process_images_to_local(content_list: list, tmp_dir: Path) -> list:
    tasks = []
    image_indices = []

    for i, item in enumerate(content_list):
        if item.get("type") == "image" and item.get("img_path", "").startswith("http"):
            url = item["img_path"]
            # Simpan URL asli sebelum di-replace
            item["_original_url"] = url

            parsed_url = urlparse(url)
            ext = os.path.splitext(parsed_url.path)[1] or ".jpg"
            local_filename = f"image_{i}{ext}"
            local_path = tmp_dir / local_filename

            tasks.append(_download_image(url, local_path))
            image_indices.append((i, local_path))

    if not tasks:
        return content_list

    # Download paralel
    results = await asyncio.gather(*tasks)

    for (idx, local_path), success in zip(image_indices, results):
        if success:
            content_list[idx]["img_path"] = str(local_path.resolve())
            logger.debug(f"Image downloaded: {local_path.name}")
        else:
            content_list[idx] = None

    # Filter out item yang None (gagal download)
    return [item for item in content_list if item is not None]


async def ingest_file(
        rag: RAGAnything,
        file_path: str | Path,
        *,
        file_name: Optional[str] = None,
        doc_id: Optional[str] = None,
        split_by_character: Optional[str] = None,
        split_by_character_only: bool = False,
        cleanup: bool = True,
    ) -> bool:
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    citation_name = file_name or path.name
    logger.info(f"[ingest] starting file={citation_name} (tmp={path.name})")

    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_assets_"))

    try:
        # ── Step 1: Parse document via Textract ──
        raw_parse = await asyncio.to_thread(
            parse_document_to_json,
            file_path=str(path),
            output_file=str(tmp_dir / "raw_parsing_result.json"),
        )

        # ── Step 2: Reconstruct content_list ──
        # Images auto-upload ke S3 jika env S3_BUCKET di-set
        content_list = await asyncio.to_thread(
            reconstruct_content_list,
            raw_parse,
        )

        # ── Step 2.5: Download S3 images ke local untuk VLM analysis ──
        content_list = await _process_images_to_local(content_list, tmp_dir)
        logger.info(f"[ingest] prepared {len(content_list)} items (images localized)")

        # ── Step 3: Insert into RAG ──
        await rag.insert_content_list(
            content_list=content_list,
            file_path=citation_name,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            doc_id=doc_id,
            display_stats=True,
        )

        logger.info(f"[ingest] completed file={citation_name}")
        return True

    except Exception as e:
        logger.error(f"[ingest] failed file={citation_name}: {e}", exc_info=True)
        raise
    finally:
        if cleanup:
            path.unlink(missing_ok=True)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.debug(f"[ingest] cleaned up tmp assets dir: {tmp_dir}")
