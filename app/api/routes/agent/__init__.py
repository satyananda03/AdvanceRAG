"""Agent-facing routes (LangGraph et al.). Populated by the user.

To add a router:
    from fastapi import APIRouter
    router = APIRouter(prefix="/agent", tags=["agent"])
    ...
    include it below.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/agent", tags=["agent"])

# TODO: user to include their agent sub-routers here, e.g.:
#     from app.api.routes.agent.mcp import router as mcp_router
#     router.include_router(mcp_router)


__all__ = ["router"]
