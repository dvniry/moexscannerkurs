"""POST /api/backtest — бэктест стратегии."""
import os, sys, asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from litestar import post
from litestar.exceptions import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.routes.candles    import get_client, _executor
from api.routes.validation import (
    validate_ticker, validate_formula, validate_days,
    validate_capital, validate_name,validate_interval, validate_size,
)
from strategy.base     import FormulaStrategy
from strategy.backtest import run_backtest

MAX_DAYS = {
    "1m": 1, "5m": 3, "15m": 7, "1h": 30, "1d": 365,
}


# ── Датаклассы ────────────────────────────────────────────

@dataclass
class BacktestRequest:
    ticker:        str
    name:          str
    entry_formula: str
    exit_formula:  str
    stop_formula:  Optional[str]  = None
    size:          float          = 10.0
    interval:      str            = "1d"
    days:          Optional[int]  = None
    capital:       float          = 100_000.0
    params:        Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradePoint:
    entry_time:  int
    exit_time:   int
    entry_price: float
    exit_price:  float
    action:      str
    pnl:         float
    pnl_pct:     float
    size:        float


@dataclass
class EquityPoint:
    time:   int
    equity: float


@dataclass
class BacktestResponse:
    ticker:       str
    name:         str
    trades:       List[TradePoint]
    equity_curve: List[EquityPoint]
    initial:      float
    final:        float
    total_return: float
    max_drawdown: float
    sharpe:       float
    winrate:      float
    total_trades: int
    wins:         int
    losses:       int
    avg_profit:   float
    avg_loss:     float
    best_trade:   float
    worst_trade:  float
    commission:   float
    error:        Optional[str] = None


# ── Логика ────────────────────────────────────────────────

def _run_bt(req: BacktestRequest):
    ticker  = validate_ticker(req.ticker)
    entry   = validate_formula(req.entry_formula, 'entry_formula')
    exit_   = validate_formula(req.exit_formula,  'exit_formula')
    stop    = validate_formula(req.stop_formula,  'stop_formula')
    days = validate_days(req.days) or MAX_DAYS.get(req.interval, 365)
    capital = validate_capital(req.capital)
    name    = validate_name(req.name)
    interval = validate_interval(req.interval)
    size     = validate_size(req.size) 
    client   = get_client()
    figi     = client.find_figi(ticker)
    df       = client.get_candles(figi=figi, interval=req.interval, days_back=days)

    if df is None or df.empty:          # ← добавить
        raise ValueError(f"Нет данных для {ticker} / {req.interval}")

    strategy = FormulaStrategy(
        name          = name,
        entry_formula = entry,
        exit_formula  = exit_,
        stop_formula  = stop,
        size          = req.size,
        interval      = req.interval,
        params        = req.params,
    )
    return run_backtest(strategy, df, initial_capital=capital)


# ── Endpoint ──────────────────────────────────────────────

@post("/backtest")
async def run_backtest_endpoint(data: BacktestRequest) -> BacktestResponse:
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, lambda: _run_bt(data))
    except (SyntaxError, ValueError, RuntimeError) as e:
        return BacktestResponse(
            ticker=data.ticker, name=data.name,
            trades=[], equity_curve=[],
            initial=data.capital, final=data.capital,
            total_return=0, max_drawdown=0, sharpe=0,
            winrate=0, total_trades=0, wins=0, losses=0,
            avg_profit=0, avg_loss=0, best_trade=0,
            worst_trade=0, commission=0, error=str(e)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return BacktestResponse(
        ticker       = data.ticker.upper(),
        name         = data.name,
        trades       = [TradePoint(
            entry_time  = t.entry_time,  exit_time   = t.exit_time,
            entry_price = t.entry_price, exit_price  = t.exit_price,
            action      = t.action,      pnl         = t.pnl,
            pnl_pct     = t.pnl_pct,     size        = t.size,
        ) for t in result.trades],
        equity_curve = [
            EquityPoint(time=e.time, equity=e.equity)
            for e in result.equity_curve
        ],
        initial      = result.initial,
        final        = result.final,
        total_return = result.total_return,
        max_drawdown = result.max_drawdown,
        sharpe       = result.sharpe,
        winrate      = result.winrate,
        total_trades = result.total_trades,
        wins         = result.wins,
        losses       = result.losses,
        avg_profit   = result.avg_profit,
        avg_loss     = result.avg_loss,
        best_trade   = result.best_trade,
        worst_trade  = result.worst_trade,
        commission   = result.commission,
        error        = None,
    )
