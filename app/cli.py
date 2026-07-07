"""Command-line interface for the multimodal pipeline.

Two subcommands so you can sanity-check ingestion and retrieval end-to-end
without spinning up the FastAPI server::

    python -m app.cli ingest ./docs/report.pdf
    python -m app.cli ingest-folder ./docs --recursive
    python -m app.cli query "What are the main findings?"
    python -m app.cli query "Explain P(d|q)=..." --mode mix

Configuration is picked up from ``.env`` via ``PipelineSettings``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure app/modules is on the import path so bare lightrag/raganything
# imports resolve to the forked packages.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULES = _REPO_ROOT / "app" / "modules"
if str(_MODULES) not in sys.path:
    sys.path.insert(0, str(_MODULES))

from app.rag_pipeline import (  # noqa: E402
    build_pipeline,
    ingest_file,
    retrieve,
    retrieve_multimodal,
)


async def _cmd_ingest(args: argparse.Namespace) -> int:
    async with build_pipeline() as rag:
        await ingest_file(
            rag,
            args.path,
            parse_method=args.parse_method,
            doc_id=args.doc_id,
        )
    print(f"[cli] ingested {args.path}")
    return 0

async def _cmd_query(args: argparse.Namespace) -> int:
    multimodal = _load_multimodal(args.multimodal)
    async with build_pipeline() as rag:
        if multimodal:
            answer = await retrieve_multimodal(
                rag, args.query, multimodal, mode=args.mode, top_k=args.top_k
            )
        else:
            answer = await retrieve(
                rag, args.query, mode=args.mode, top_k=args.top_k
            )
    print(answer)
    return 0


def _load_multimodal(value: str | None) -> list[dict] | None:
    if not value:
        return None
    payload = Path(value).read_text() if Path(value).exists() else value
    parsed = json.loads(payload)
    if not isinstance(parsed, list):
        raise ValueError(
            "--multimodal must be a JSON array of items (or a path to one)."
        )
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest a single file.")
    p_ing.add_argument("path", type=Path)
    p_ing.add_argument("--parse-method", default=None)
    p_ing.add_argument("--doc-id", default=None)
    p_ing.set_defaults(func=_cmd_ingest)

    p_qry = sub.add_parser("query", help="Ask a question against the knowledge base.")
    p_qry.add_argument("query")
    p_qry.add_argument(
        "--mode",
        default="mix",
        choices=["local", "global", "hybrid", "naive", "mix", "bypass"],
    )
    p_qry.add_argument("--top-k", type=int, default=None)
    p_qry.add_argument(
        "--multimodal",
        default=None,
        help="Path to a JSON file OR an inline JSON array of multimodal items.",
    )
    p_qry.set_defaults(func=_cmd_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
