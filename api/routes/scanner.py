"""POST /api/scanner — сканирование тикеров по формуле."""
import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from litestar import post, get
from litestar.exceptions import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.routes.candles import get_client, _to_unix, _executor
from indicators.formula import Formula

MAX_DAYS = {
    "1m":  1,
    "5m":  3,
    "15m": 7,
    "1h":  30,
    "1d":  365,
}


@dataclass
class ScannerRequest:
    tickers:  List[str]
    formula:  str                  # RESULT = RSI(14) < 30  → bool Series
    interval: str          = "1h"
    params:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class TickerResult:
    ticker:   str
    signal:   bool          # True = условие выполнено
    value:    float         # последнее значение формулы
    price:    float         # последняя цена
    change:   float         # изменение за период в %
    error:    Optional[str] = None


@dataclass
class ScannerResponse:
    results:  List[TickerResult]
    total:    int
    signals:  int           # сколько тикеров дали сигнал


def _scan_ticker(ticker: str, formula: str, interval: str, params: dict) -> TickerResult:
    """Синхронно: загружаем данные + считаем формулу для одного тикера."""
    try:
        client = get_client()
        figi   = client.find_figi(ticker.upper())

        days = MAX_DAYS.get(interval, 7)
        df   = client.get_candles(figi=figi, interval=interval, days_back=days)

        if df.empty:
            return TickerResult(
                ticker=ticker, signal=False, value=0.0,
                price=0.0, change=0.0, error="Нет данных"
            )

        # Считаем формулу
        ind    = Formula(name='scan', formula=formula, params=params)
        result = ind(df)

        last_val   = float(result.dropna().iloc[-1]) if not result.dropna().empty else 0.0
        last_price = float(df['close'].iloc[-1])
        first_price = float(df['close'].iloc[0])
        change     = round((last_price - first_price) / first_price * 100, 2)

        # Сигнал: True если последнее значение > 0 или True
        signal = bool(last_val)

        return TickerResult(
            ticker=ticker.upper(),
            signal=signal,
            value=round(last_val, 4),
            price=last_price,
            change=change,
            error=None
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


@post("/scanner")
async def run_scanner(data: ScannerRequest) -> ScannerResponse:
    """Запустить сканер по списку тикеров."""
    if not data.tickers:
        raise HTTPException(status_code=400, detail="Список тикеров пуст")
    if not data.formula:
        raise HTTPException(status_code=400, detail="Формула не задана")

    loop = asyncio.get_event_loop()

    # Запускаем все тикеры параллельно через ThreadPoolExecutor
    tasks = [
        loop.run_in_executor(
            _executor,
            lambda t=ticker: _scan_ticker(
                t, data.formula, data.interval, data.params
            )
        )
        for ticker in data.tickers
    ]

    results = await asyncio.gather(*tasks)

    signals = sum(1 for r in results if r.signal)

    return ScannerResponse(
        results=list(results),
        total=len(results),
        signals=signals,
    )


# Список популярных тикеров MOEX
MOEX_TICKERS = [
    "SBER", "GAZP", "LKOH", "GMKN", "NVTK",
    "ROSN", "TATN", "MGNT", "MTSS", "TCSG",   # TCSG вместо YNDX
    "ALRS", "PLZL", "SNGS", "VTBR", "AFLT",
    "MAGN", "NLMK", "CHMF", "PHOR", "CBOM",   # CBOM вместо POLY
]

@get("/scanner/tickers")
async def get_tickers() -> dict:
    """Список доступных тикеров MOEX."""
    return {"tickers": MOEX_TICKERS}
