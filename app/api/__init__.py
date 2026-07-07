"""HTTP layer.

Groups routes into ``routes/lightrag`` (dashboard-facing) and
``routes/agent`` (for LangGraph et al.). All routes converge on the same
``app.core.pipeline`` library, so behaviour stays aligned across surfaces.
"""
