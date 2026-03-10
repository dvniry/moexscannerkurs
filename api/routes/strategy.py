"""Strategy API — генерация сигналов + sandbox ордера."""
import os, sys, asyncio, uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from litestar import get, post
from litestar.exceptions import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.routes.candles    import get_client, _executor
from api.routes.validation import (
    validate_ticker, validate_formula, validate_days,
    validate_lots, validate_direction, validate_name,
)
from strategy.base   import FormulaStrategy
from strategy.engine import run_strategy

MAX_DAYS = {
    "1m": 1, "5m": 3, "15m": 7, "1h": 30, "1d": 365,
}


# ── Датаклассы ────────────────────────────────────────────

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
    action: str
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
    figi:       str
    direction:  str
    lots:       int
    price:      float = 0.0
    account_id: str   = ""


@dataclass
class OrderResponse:
    order_id:   str
    status:     str
    lots_exec:  int
    price:      float
    account_id: str = ""


# ── Логика ────────────────────────────────────────────────

def _run_strategy_sync(req: StrategyRequest):
    ticker = validate_ticker(req.ticker)
    entry  = validate_formula(req.entry_formula, 'entry_formula')
    exit_  = validate_formula(req.exit_formula,  'exit_formula')
    stop   = validate_formula(req.stop_formula,  'stop_formula')
    days   = validate_days(req.days)
    name   = validate_name(req.name)

    client   = get_client()
    figi     = client.find_figi(ticker)
    df       = client.get_candles(figi=figi, interval=req.interval, days_back=days)

    strategy = FormulaStrategy(
        name          = name,
        entry_formula = entry,
        exit_formula  = exit_,
        stop_formula  = stop,
        size          = req.size,
        interval      = req.interval,
        params        = req.params,
    )
    return run_strategy(strategy, df)


# ── Endpoints ─────────────────────────────────────────────

@post("/strategy/run")
async def run_strategy_endpoint(data: StrategyRequest) -> StrategyResponse:
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, lambda: _run_strategy_sync(data)
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
        name        = data.name,
        ticker      = data.ticker.upper(),
        signals     = signals,
        in_position = result.in_position,
        stats       = result.stats,
        error       = None,
    )


@post("/strategy/order")
async def post_order(data: OrderRequest) -> OrderResponse:
    validate_direction(data.direction)
    validate_lots(data.lots)
    if not data.figi:
        raise HTTPException(status_code=422, detail="figi обязателен.")

    try:
        from strategy.orders import SandboxOrders
        orders   = SandboxOrders()
        order_id = str(uuid.uuid4())
        loop     = asyncio.get_event_loop()
        resp     = await loop.run_in_executor(
            _executor,
            lambda: orders.post_order(
                figi      = data.figi,
                direction = data.direction,
                lots      = data.lots,
                price     = data.price,
                order_id  = order_id,
            )
        )
        return OrderResponse(**resp)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@get("/strategy/sandbox/topup")
async def sandbox_topup() -> dict:
    try:
        from strategy.orders import SandboxOrders
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, lambda: SandboxOrders().top_up(1_000_000)
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@get("/strategy/sandbox/portfolio")
async def sandbox_portfolio() -> dict:
    try:
        from strategy.orders import SandboxOrders
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, lambda: SandboxOrders().get_portfolio()
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
