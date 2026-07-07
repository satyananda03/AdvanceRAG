SHELL := /bin/bash
VENV  := .venv/bin
PYTHON := $(VENV)/python

# ── colours ──────────────────────────────────────────────────────────────────
COLOR_RESET  := \033[0m
COLOR_BOLD   := \033[1m
COLOR_GREEN  := \033[32m
COLOR_BLUE   := \033[34m
COLOR_YELLOW := \033[33m
ifeq ($(NO_COLOR),1)
COLOR_RESET := COLOR_BOLD := COLOR_GREEN := COLOR_BLUE := COLOR_YELLOW :=
endif

.PHONY: help dev webui server server-reload lint test

# ── help ─────────────────────────────────────────────────────────────────────
help:
	@printf "$(COLOR_BOLD)Usage$(COLOR_RESET)\n"
	@printf "  $(COLOR_GREEN)make dev$(COLOR_RESET)                Install backend deps + build WebUI\n"
	@printf "  $(COLOR_GREEN)make webui$(COLOR_RESET)              (Re)build the LightRAG dashboard bundle\n"
	@printf "  $(COLOR_GREEN)make server$(COLOR_RESET)             Start the unified server (port 8000)\n"
	@printf "  $(COLOR_GREEN)make server-reload$(COLOR_RESET)      Unified server with hot-reload\n"
	@printf "  $(COLOR_GREEN)make lint$(COLOR_RESET)               Run ruff\n"
	@printf "  $(COLOR_GREEN)make test$(COLOR_RESET)               Run pytest\n"
	@printf "\n"
	@printf "$(COLOR_BOLD)Services$(COLOR_RESET) (single process, port 8000)\n"
	@printf "  LightRAG dashboard  → http://localhost:8000/webui\n"
	@printf "  API docs            → http://localhost:8000/docs\n"
	@printf "  Ingest endpoint     → POST http://localhost:8000/rag-anything/ingest\n"
	@printf "  Query endpoint      → POST http://localhost:8000/rag-anything/query\n"

# ── install ──────────────────────────────────────────────────────────────────
dev:
	@command -v uv  >/dev/null 2>&1 || { \
		printf "$(COLOR_YELLOW)uv not found. Install: https://docs.astral.sh/uv/$(COLOR_RESET)\n"; exit 1; }
	@command -v bun >/dev/null 2>&1 || { \
		printf "$(COLOR_YELLOW)bun not found. Install: https://bun.sh$(COLOR_RESET)\n"; exit 1; }
	@printf "$(COLOR_BLUE)Installing backend deps...$(COLOR_RESET)\n"
	uv sync
	@$(MAKE) webui
	@printf "$(COLOR_GREEN)Done. Try: make server$(COLOR_RESET)\n"

# ── webui build ──────────────────────────────────────────────────────────────
webui:
	@printf "$(COLOR_BLUE)Building LightRAG WebUI...$(COLOR_RESET)\n"
	cd lightrag_webui && bun install --frozen-lockfile && bun run build
	@printf "$(COLOR_GREEN)WebUI built.$(COLOR_RESET)\n"

# ── Unified server (port 8000) ───────────────────────────────────────────────
server:
	@printf "$(COLOR_BLUE)Starting unified server (WebUI http://localhost:8000/webui, docs /docs)...$(COLOR_RESET)\n"
	$(PYTHON) run.py

server-reload:
	@printf "$(COLOR_BLUE)Unified server with hot-reload...$(COLOR_RESET)\n"
	PYTHONPATH=app/modules:. \
	$(VENV)/uvicorn run:app \
		--host "$${AGENT_API_HOST:-0.0.0.0}" \
		--port "$${AGENT_API_PORT:-8000}" \
		--reload \
		--reload-dir app

# ── quality ──────────────────────────────────────────────────────────────────
lint:
	$(VENV)/ruff check app run.py

test:
	$(PYTHON) -m pytest tests -q
