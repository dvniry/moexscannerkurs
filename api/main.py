"""MOEX Scanner — точка входа Litestar."""
import os
import sys
import threading
import webbrowser
from pathlib import Path

from litestar import Litestar, MediaType, get, Router
from litestar.config.cors import CORSConfig
from litestar.static_files import StaticFilesConfig
from litestar.openapi import OpenAPIConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.routes.candles import get_candles
from api.routes.formula import calculate_formula
from api.routes.ws      import price_ws

FRONTEND_DIR = str(Path(__file__).parent.parent / "frontend")


@get("/", media_type=MediaType.HTML)
async def index() -> bytes:
    return (Path(FRONTEND_DIR) / "index.html").read_bytes()


@get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.5.0"}


# ── Роутер с префиксом /api ───────────────────────────────
api_router = Router(
    path="/api",
    route_handlers=[get_candles, calculate_formula],
)

# ── Приложение ────────────────────────────────────────────
app = Litestar(
    route_handlers=[
        index,
        health,
        api_router,   # ← /api/candles, /api/formula
        price_ws,     # ← /ws/price
    ],
    static_files_config=[
        StaticFilesConfig(
            directories=[FRONTEND_DIR],
            path="/static",
            name="static",
        )
    ],
    cors_config=CORSConfig(allow_origins=["*"]),
    openapi_config=OpenAPIConfig(
        title="MOEX Scanner API",
        version="0.5.0",
    ),
    debug=True,   # ← включаем для видимости ошибок
)


if __name__ == "__main__":
    import uvicorn

    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open("http://localhost:8050")

    threading.Thread(target=open_browser, daemon=True).start()

    print("\n" + "=" * 50)
    print("🚀  MOEX Scanner")
    print("    http://localhost:8050")
    print("    API docs: http://localhost:8050/schema")
    print("=" * 50 + "\n")

    uvicorn.run(
        app,               # ← передаём объект напрямую, не строку
        host="0.0.0.0",
        port=8050,
        reload=False,
    )
