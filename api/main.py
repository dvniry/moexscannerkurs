"""MOEX Scanner — точка входа Litestar."""
from __future__ import annotations

import os
_cert = os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer')
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = os.path.abspath(_cert)
import sys
import threading
import webbrowser
from pathlib import Path

from litestar import Litestar, MediaType, get, Router
from litestar.config.cors import CORSConfig
from litestar.static_files import StaticFilesConfig
from litestar.openapi import OpenAPIConfig
from litestar.middleware.rate_limit import RateLimitConfig
from litestar.stores.memory import MemoryStore
from litestar.middleware.base import MiddlewareProtocol
from litestar.types import ASGIApp, Receive, Scope, Send
from litestar.exceptions import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.exception_handlers import http_exception_handler, internal_exception_handler
from api.routes.candles  import get_candles
from api.routes.formula  import calculate_formula
from api.routes.ws       import price_ws
from api.routes.scanner  import run_scanner, get_tickers
from api.routes.backtest import run_backtest_endpoint
from api.routes.strategy import (
    run_strategy_endpoint,
    post_order,
    sandbox_topup,
    sandbox_portfolio,
)

FRONTEND_DIR = str(Path(__file__).parent.parent / "frontend")


@get("/", media_type=MediaType.HTML)
async def index() -> bytes:
    return (Path(FRONTEND_DIR) / "index.html").read_bytes()


@get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.8.0"}

class SecurityHeadersMiddleware(MiddlewareProtocol):
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"x-content-type-options"] = b"nosniff"
                headers[b"x-frame-options"]         = b"DENY"
                headers[b"x-xss-protection"]        = b"1; mode=block"
                headers[b"referrer-policy"]          = b"strict-origin-when-cross-origin"
                headers[b"permissions-policy"]       = b"geolocation=(), microphone=()"
                message = {**message, "headers": list(headers.items())}
            await send(message)

        await self.app(scope, receive, send_with_headers)

api_router = Router(
    path="/api",
    route_handlers=[
        get_candles,
        calculate_formula,
        run_scanner,
        get_tickers,
        run_strategy_endpoint,
        post_order,
        sandbox_topup,
        sandbox_portfolio,
        run_backtest_endpoint,
    ],
)

rate_limit_config = RateLimitConfig(
    rate_limit=("minute", 60),   # 10 для теста
    exclude=["/schema"],
    store="rate_limit",
)

app = Litestar(
    route_handlers=[index, health, api_router, price_ws],
    middleware=[SecurityHeadersMiddleware, rate_limit_config.middleware,],
    stores={"rate_limit": MemoryStore()},
        exception_handlers={
        HTTPException: http_exception_handler,
        Exception:     internal_exception_handler,
    },
    static_files_config=[
        StaticFilesConfig(
            directories=[FRONTEND_DIR],
            path="/static",
            name="static",
        )
    ],
    cors_config=CORSConfig(
        allow_origins=["http://localhost:8050"],   # 7.2: убрали *
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    ),
    openapi_config=OpenAPIConfig(
        title="MOEX Scanner API",
        version="0.8.0",
    ),
    debug=False,   # в продакшне False
)

if __name__ == "__main__":
    import uvicorn

    def _open_browser() -> None:
        import time
        time.sleep(1.5)
        webbrowser.open("http://localhost:8050")

    threading.Thread(target=_open_browser, daemon=True).start()

    print("\nMOEX Scanner v0.8")
    print("http://localhost:8050")
    print("API docs: http://localhost:8050/schema\n")

    uvicorn.run(app, host="0.0.0.0", port=8050, reload=False)
