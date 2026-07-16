import os
import re
import asyncio
import time
from collections import OrderedDict
from typing import Dict, Tuple
from app.core.config import settings as base_settings
from app.rag_pipeline.factory import _construct_lightrag, _build_rag_anything_config
from app.rag_pipeline.model_funcs import build_llm_func, build_vision_func, build_embedding_func
from app.modules.raganything import RAGAnything
from app.modules.lightrag.utils import logger

# Regex untuk memastikan nama workspace aman untuk filesystem/DB (tanpa prefix paksa)
_WS_RE = re.compile(r"[^a-zA-Z0-9_\-]")

def sanitize_workspace(ws_name: str) -> str:
    """Normalize workspace name to be filesystem/DB safe."""
    if not ws_name:
        return "default_workspace"
    # Hapus karakter ilegal, ganti dengan underscore
    cleaned = _WS_RE.sub("_", ws_name.strip())
    return cleaned.strip("_-") or "default_workspace"

class WorkspacePool:
    def __init__(self, max_workspaces: int = 32):
        self._max = max_workspaces
        self._entries: "OrderedDict[str, Tuple[RAGAnything, float]]" = OrderedDict()
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def acquire(self, ws_name: str) -> RAGAnything:
        ws = sanitize_workspace(ws_name)

        async with self._global_lock:
            lock = self._locks.setdefault(ws, asyncio.Lock())

        async with lock:
            async with self._global_lock:
                entry = self._entries.get(ws)
                if entry is not None:
                    rag, _ = entry
                    self._entries[ws] = (rag, time.time())
                    self._entries.move_to_end(ws)
                    return rag

            logger.info(f"[WorkspacePool] Initializing new RAG instance for: {ws}")
            
            user_settings = base_settings.model_copy()
            user_settings.workspace = ws  # Set nama workspace apa adanya

            lightrag = await _construct_lightrag(user_settings)
            rag = RAGAnything(
                lightrag=lightrag,
                llm_model_func=build_llm_func(user_settings),
                vision_model_func=build_vision_func(user_settings),
                embedding_func=build_embedding_func(user_settings),
                config=_build_rag_anything_config(user_settings),
            )

            async with self._global_lock:
                self._entries[ws] = (rag, time.time())
                self._entries.move_to_end(ws)
                
                while len(self._entries) > self._max:
                    old_ws, (old_rag, _) = self._entries.popitem(last=False)
                    logger.info(f"[WorkspacePool] Evicting workspace: {old_ws}")
                    asyncio.create_task(self._safe_finalize(old_rag))
            
            return rag

    async def _safe_finalize(self, rag: RAGAnything):
        """Helper untuk menutup storage connection dengan aman."""
        try:
            await rag.finalize_storages()
        except Exception:
            logger.error("[WorkspacePool] Error finalizing RAG instance", exc_info=True)

    async def shutdown(self):
        """Dipanggil saat aplikasi FastAPI dimatikan."""
        async with self._global_lock:
            entries = list(self._entries.values())
            self._entries.clear()
        
        for rag, _ in entries:
            await self._safe_finalize(rag)

# Inisialisasi Singleton Pool, batas maksimal workspace aktif bersamaan di memori
pool_size = int(os.getenv("RAG_MAX_WORKSPACES", "32"))
workspace_pool = WorkspacePool(max_workspaces=pool_size)