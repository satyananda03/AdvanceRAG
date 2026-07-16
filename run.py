from __future__ import annotations
import os
import sys
from pathlib import Path

def prepare_sys_path() -> None:
    repo_root = Path(__file__).resolve().parent
    modules_dir = repo_root / "app" / "modules"
    for p in (str(modules_dir), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

prepare_sys_path()

import uvicorn
from app.api.main import create_app

app = create_app()

def main() -> None:
    host = "0.0.0.0"
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    uvicorn.run(app, host=host, port=port, log_level=log_level)

if __name__ == "__main__":
    main()