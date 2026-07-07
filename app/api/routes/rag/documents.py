from __future__ import annotations
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel, Field

from app.api.dependencies import get_pipeline
from app.rag_pipeline import ingest_file


router = APIRouter(tags=["documents"])


class IngestResponse(BaseModel):
    status: str = Field(default="accepted", description="Ingestion status.")
    file_name: str
    detail: Optional[str] = None

async def _run_ingest(rag, tmp_path: Path, file_name: str) -> None:
    """Background task: run the pipeline. Cleanup handled by ingest_file."""
    await ingest_file(
        rag,
        tmp_path,
        file_name=file_name,
    )

@router.post("/ingest", response_model=IngestResponse)
async def ingest(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        parse_method: Optional[str] = Form(default=None),
        doc_id: Optional[str] = Form(default=None),
        rag=Depends(get_pipeline),
    ) -> IngestResponse:
    """Multipart upload -> pipeline ingestion in the background."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    background_tasks.add_task(_run_ingest, rag, tmp_path, file.filename)
    return IngestResponse(
        file_name=file.filename,
        detail="Ingestion running in background.",
    )
