"""POST /api/scanner — сканирование тикеров по формуле."""
import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from litestar import post, get
from litestar.exceptions import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.routes.candles    import get_client, _executor
from api.routes.validation import validate_ticker, validate_formula
from indicators.formula    import Formula


MAX_DAYS = {
    "1m": 1, "5m": 3, "15m": 7, "1h": 30, "1d": 365,
}


@dataclass
class ScannerRequest:
    tickers:  List[str]
    formula:  str
    interval: str            = "1h"
    params:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class TickerResult:
    ticker:  str
    signal:  bool
    value:   float
    price:   float
    change:  float
    error:   Optional[str] = None


@dataclass
class ScannerResponse:
    results: List[TickerResult]
    total:   int
    signals: int


def _scan_ticker(ticker: str, formula: str, interval: str, params: dict) -> TickerResult:
    try:
        client = get_client()
        figi   = client.find_figi(ticker.upper())
        days   = MAX_DAYS.get(interval, 7)
        df     = client.get_candles(figi=figi, interval=interval, days_back=days)

        if df.empty:
            return TickerResult(
                ticker=ticker, signal=False, value=0.0,
                price=0.0, change=0.0, error="Нет данных"
            )

        ind    = Formula(name='scan', formula=formula, params=params)
        result = ind(df)

        last_val    = float(result.dropna().iloc[-1]) if not result.dropna().empty else 0.0
        last_price  = float(df['close'].iloc[-1])
        first_price = float(df['close'].iloc[0])
        change      = round((last_price - first_price) / first_price * 100, 2)

        return TickerResult(
            ticker=ticker.upper(),
            signal=bool(last_val),
            value=round(last_val, 4),
            price=last_price,
            change=change,
            error=None,
        )

    except (SyntaxError, ValueError, RuntimeError) as e:
        return TickerResult(
            ticker=ticker, signal=False, value=0.0,
            price=0.0, change=0.0, error=f"Формула: {str(e)[:80]}"
        )
    except Exception as e:
        return TickerResult(
            ticker=ticker, signal=False, value=0.0,
            price=0.0, change=0.0, error=str(e)[:80]
        )


@post("/scanner/run")
async def run_scanner(data: ScannerRequest) -> ScannerResponse:
    # Валидация тикеров
    valid_tickers = []
    for t in data.tickers:
        try:
            valid_tickers.append(validate_ticker(t))
        except HTTPException:
            pass

    if not valid_tickers:
        raise HTTPException(status_code=422, detail="Нет валидных тикеров.")

    # Валидация формулы
    formula = validate_formula(data.formula, 'formula')
    if not formula:
        raise HTTPException(status_code=422, detail="Формула обязательна.")

    # ← фикс: получаем loop здесь
    loop = asyncio.get_event_loop()

    tasks = [
        loop.run_in_executor(
            _executor,
            lambda t=ticker: _scan_ticker(t, formula, data.interval, data.params)
        )
        for ticker in valid_tickers
    ]

    results = await asyncio.gather(*tasks)
    signals = sum(1 for r in results if r.signal)

    return ScannerResponse(
        results=list(results),
        total=len(results),
        signals=signals,
    )


MOEX_TICKERS = [
    "SBER", "GAZP", "LKOH", "GMKN", "NVTK",
    "ROSN", "TATN", "MGNT", "MTSS", "TCSG",
    "ALRS", "PLZL", "SNGS", "VTBR", "AFLT",
    "MAGN", "NLMK", "CHMF", "PHOR", "CBOM",
]


@get("/scanner/tickers")
async def get_tickers() -> dict:
    return {"tickers": MOEX_TICKERS}
