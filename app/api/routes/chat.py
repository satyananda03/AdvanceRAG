from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
import tempfile, shutil
from app.core.workspace_pool import workspace_pool
from app.rag_pipeline.retrieve import retrieve_context
from app.rag_pipeline.ingest import ingest_file

router = APIRouter(prefix = "/api/v1", tags=["chat"])

@router.post("/chat")
async def retrieve_endpoint(
    workspace: str = Form(...), # Bisa "policy_legal", "evidence_indicator_1", atau "consult_userA"
    query: str = Form(...),
    mode: str = Form("mix"),
):
    rag = await workspace_pool.acquire(workspace)
    result = await retrieve_context(rag, query, mode=mode, include_raw_data=False)
    return {
        "context": result.context,
        "image_paths": result.image_paths,
        "workspace": workspace,
    }