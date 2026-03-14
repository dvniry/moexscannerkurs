"""Strategy API — генерация сигналов + sandbox ордера."""
import os, sys, uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from litestar import get, post
from litestar.exceptions import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.routes._common    import MAX_DAYS, fetch_df, run_in_thread
from api.routes.validation import (
    validate_ticker, validate_formula, validate_days,
    validate_lots, validate_direction, validate_name,
    validate_interval, validate_size,
)
from strategy.base   import FormulaStrategy
from strategy.engine import run_strategy
from strategy.orders import SandboxOrders


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
    ticker   = validate_ticker(req.ticker)
    entry    = validate_formula(req.entry_formula, 'entry_formula')
    exit_    = validate_formula(req.exit_formula,  'exit_formula')
    stop     = validate_formula(req.stop_formula,  'stop_formula')
    days     = validate_days(req.days) or MAX_DAYS.get(req.interval, 30)
    name     = validate_name(req.name)
    interval = validate_interval(req.interval)
    size     = validate_size(req.size)

    _, df = fetch_df(ticker, interval, days)

    strategy = FormulaStrategy(
        name=name, entry_formula=entry, exit_formula=exit_,
        stop_formula=stop, size=size, interval=interval,
        params=req.params,
    )
    return run_strategy(strategy, df)


# ── Endpoints ─────────────────────────────────────────────

@post("/strategy/run")
async def run_strategy_endpoint(data: StrategyRequest) -> StrategyResponse:
    try:
        result = await run_in_thread(lambda: _run_strategy_sync(data))
    except (SyntaxError, ValueError, RuntimeError) as e:
        return StrategyResponse(
            name=data.name, ticker=data.ticker,
            signals=[], in_position=False,
            stats={}, error=str(e),
        )

    return StrategyResponse(
        name        = data.name,
        ticker      = data.ticker.upper(),
        signals     = [
            SignalPoint(
                time=s.time, price=s.price,
                action=s.action, size=s.size, reason=s.reason,
            )
            for s in result.signals
        ],
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

    order_id = str(uuid.uuid4())
    resp = await run_in_thread(lambda: SandboxOrders().post_order(
        figi=data.figi, direction=data.direction,
        lots=data.lots, price=data.price, order_id=order_id,
    ))
    return OrderResponse(**resp)


@get("/strategy/sandbox/topup")
async def sandbox_topup() -> dict:
    return await run_in_thread(lambda: SandboxOrders().top_up(1_000_000))


@get("/strategy/sandbox/portfolio")
async def sandbox_portfolio() -> dict:
    return await run_in_thread(lambda: SandboxOrders().get_portfolio())
