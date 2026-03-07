"""Strategy API — генерация сигналов + sandbox ордера."""
import os, sys, asyncio, uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from litestar import get, post
from litestar.exceptions import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.routes.candles import get_client, _to_unix, _executor
from strategy.base      import FormulaStrategy, Signal
from strategy.engine    import run_strategy

MAX_DAYS = {
    "1m": 1, "5m": 3, "15m": 7, "1h": 30, "1d": 365,
}


# ── Датаклассы запросов/ответов ───────────────────────────

@dataclass
class StrategyRequest:
    ticker:        str
    name:          str
    entry_formula: str
    exit_formula:  str
    stop_formula:  Optional[str]  = None
    size:          float          = 10.0
    interval:      str            = "1h"
    days:          Optional[int]  = None
    params:        Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalPoint:
    time:   int
    price:  float
    action: str     # BUY | SELL | STOP
    size:   float
    reason: str


@dataclass
class StrategyResponse:
    name:        str
    ticker:      str
    signals:     List[SignalPoint]
    in_position: bool
    stats:       Dict[str, Any]
    error:       Optional[str] = None


@dataclass
class OrderRequest:
    figi:      str
    direction: str      # BUY | SELL
    lots:      int
    price:     float    = 0.0   # 0 = market
    account_id: str     = ""


@dataclass
class OrderResponse:
    order_id: str
    status:   str
    lots_exec: int
    price:    float
    account_id: str = ""


# ── Endpoints ─────────────────────────────────────────────

def _run_strategy_sync(req: StrategyRequest):
    client   = get_client()
    figi     = client.find_figi(req.ticker.upper())
    days     = req.days or MAX_DAYS.get(req.interval, 7)
    df       = client.get_candles(figi=figi, interval=req.interval, days_back=days)

    strategy = FormulaStrategy(
        name          = req.name,
        entry_formula = req.entry_formula,
        exit_formula  = req.exit_formula,
        stop_formula  = req.stop_formula,
        size          = req.size,
        interval      = req.interval,
        params        = req.params,
    )
    result = run_strategy(strategy, df)
    return result


@post("/strategy/run")
async def run_strategy_endpoint(data: StrategyRequest) -> StrategyResponse:
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: _run_strategy_sync(data)
        )
    except (SyntaxError, ValueError, RuntimeError) as e:
        return StrategyResponse(
            name=data.name, ticker=data.ticker,
            signals=[], in_position=False,
            stats={}, error=str(e)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    signals = [
        SignalPoint(
            time=s.time, price=s.price,
            action=s.action, size=s.size, reason=s.reason
        )
        for s in result.signals
    ]

    return StrategyResponse(
        name=data.name,
        ticker=data.ticker.upper(),
        signals=signals,
        in_position=result.in_position,
        stats=result.stats,
        error=None,
    )


@post("/strategy/order")
async def post_order(data: OrderRequest) -> OrderResponse:
    """Отправить ордер в Tinkoff Sandbox."""
    try:
        from strategy.orders import SandboxOrders
        orders   = SandboxOrders()
        order_id = str(uuid.uuid4())
        loop     = asyncio.get_event_loop()
        resp     = await loop.run_in_executor(
            _executor,
            lambda: orders.post_order(
                figi=data.figi,
                direction=data.direction,
                lots=data.lots,
                price=data.price,
                order_id=order_id,
            )
        )
        return OrderResponse(**resp)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@get("/strategy/sandbox/topup")
async def sandbox_topup() -> dict:
    """Пополнить sandbox на 1 000 000 ₽."""
    try:
        from strategy.orders import SandboxOrders
        orders = SandboxOrders()
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, lambda: orders.top_up(1_000_000))
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@get("/strategy/sandbox/portfolio")
async def sandbox_portfolio() -> dict:
    """Получить портфель sandbox."""
    try:
        from strategy.orders import SandboxOrders
        orders = SandboxOrders()
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, lambda: orders.get_portfolio())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
