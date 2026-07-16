from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
import tempfile, shutil
from app.core.workspace_pool import workspace_pool
from app.rag_pipeline.retrieve import retrieve_context
from app.rag_pipeline.ingest import ingest_file

router = APIRouter(prefix = "/api/v1/ingest", tags=["ingestion"])

@router.post("/policy")
async def ingest_policy(
        doc_id: str = Form(...),
        file: UploadFile = File(...)
    ):
    """Khusus upload dokumen hukum/policy. Workspace statis."""
    ws = "policy_legal"
    rag = await workspace_pool.acquire(ws)
    tmp = Path(tempfile.mktemp(prefix="policy_", suffix=f"_{file.filename}"))
    try:
        with tmp.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        ok = await ingest_file(rag=rag, file_path=tmp, file_name=file.filename, doc_id=doc_id)
        return {"status": "ok" if ok else "failed", "workspace": ws, "doc_id": doc_id}
    finally:
        tmp.unlink(missing_ok=True)

@router.post("/evidence")
async def ingest_evidence(
        indicator_id: str = Form(...),
        doc_id: str = Form(...),
        file: UploadFile = File(...)
    ):
    """Khusus upload evidence sistem per indikator."""
    ws = f"evidence_indicator_{indicator_id}"
    rag = await workspace_pool.acquire(ws)
    tmp = Path(tempfile.mktemp(prefix="evidence_", suffix=f"_{file.filename}"))
    try:
        with tmp.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        ok = await ingest_file(rag=rag, file_path=tmp, file_name=file.filename, doc_id=doc_id)
        return {"status": "ok" if ok else "failed", "workspace": ws, "doc_id": doc_id}
    finally:
        tmp.unlink(missing_ok=True)

@router.post("/consult")
async def consult_ingest(
        user_id: str = Form(...),
        doc_id: str = Form(...),
        file: UploadFile = File(...)
    ):
    """User upload dokumen untuk konsultasi."""
    ws = f"consult_{user_id}"
    rag = await workspace_pool.acquire(ws)
    tmp = Path(tempfile.mktemp(prefix="consult_", suffix=f"_{file.filename}"))
    try:
        with tmp.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        ok = await ingest_file(rag=rag, file_path=tmp, file_name=file.filename, doc_id=doc_id)
        return {"status": "ok" if ok else "failed", "workspace": ws, "doc_id": doc_id}
    finally:
        tmp.unlink(missing_ok=True)