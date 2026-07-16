from fastapi import APIRouter, Query, HTTPException
from app.core.workspace_pool import workspace_pool

router = APIRouter(prefix="/api/v1/delete", tags=["deletion"])

@router.delete("/policy")
async def delete_policy(doc_id: str = Query(..., description="Document ID to delete")):
    ws = "policy_legal"
    rag = await workspace_pool.acquire(ws)
    try:
        result = await rag.lightrag.adelete_by_doc_id(doc_id = doc_id, delete_llm_cache = True)
        return {"status": "ok" if result.status == "success" else "failed", "workspace": ws, "doc_id": doc_id, "message": getattr(result, "message", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/evidence")
async def delete_evidence(indicator_id: str = Query(..., description="Indicator ID"), doc_id: str = Query(..., description="Document ID to delete")):
    ws = f"evidence_indicator_{indicator_id}"
    rag = await workspace_pool.acquire(ws)
    try:
        result = await rag.lightrag.adelete_by_doc_id(doc_id = doc_id, delete_llm_cache = True)
        return {"status": "ok" if result.status == "success" else "failed", "workspace": ws, "doc_id": doc_id, "message": getattr(result, "message", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/consult")
async def delete_consult(user_id: str = Query(..., description="User ID"), doc_id: str = Query(..., description="Document ID to delete")):
    ws = f"consult_{user_id}"
    rag = await workspace_pool.acquire(ws)
    try:
        result = await rag.lightrag.adelete_by_doc_id(doc_id = doc_id, delete_llm_cache = True)
        return {"status": "ok" if result.status == "success" else "failed", "workspace": ws, "doc_id": doc_id, "message": getattr(result, "message", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))