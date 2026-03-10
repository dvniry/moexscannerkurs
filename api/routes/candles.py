"""GET /api/candles — свечи OHLCV."""
import os
import sys
import asyncio
from dataclasses import dataclass
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from litestar import get
from litestar.exceptions import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.tinkoff_client import TinkoffDataClient
from config import config

# Синглтон клиента и пул потоков
_client: TinkoffDataClient | None = None
_executor = ThreadPoolExecutor(max_workers=4)


def get_client() -> TinkoffDataClient:
    global _client
    if _client is None:
        _client = TinkoffDataClient(token=config.tinkoff.token)
    return _client


def _fetch_candles(ticker: str, interval: str, days: int) -> tuple:
    """Синхронный вызов — выполняется в потоке, не блокирует event loop."""
    client = get_client()
    figi   = client.find_figi(ticker.upper())

    if not figi:                          # ← добавь
        raise ValueError(f"Тикер '{ticker}' не найден. Проверьте правильность написания.")
    
    df     = client.get_candles(figi=figi, interval=interval, days_back=days)
    return figi, df


@dataclass
class Candle:
    time:   int
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


@dataclass
class CandlesResponse:
    ticker:   str
    figi:     str
    interval: str
    candles:  List[Candle]


def _to_unix(df: pd.DataFrame) -> list[int]:
    idx = pd.to_datetime(df.index)
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_convert('UTC').tz_localize(None)
    return (idx.astype('datetime64[ns]').astype('int64') // 10**9).tolist()


@get("/candles")
async def get_candles(
    ticker:   str,
    interval: str = "1h",
    days:     int = 30,
) -> CandlesResponse:
    """Получить свечи по тикеру."""
    try:
        # Запускаем синхронный gRPC вызов в отдельном потоке
        loop       = asyncio.get_event_loop()
        figi, df   = await loop.run_in_executor(
            _executor,
            lambda: _fetch_candles(ticker, interval, days)
        )
    except Exception as e:
        global _client
        _client = None  # сброс при ошибке
        raise HTTPException(status_code=400, detail=str(e))

    unix = _to_unix(df)

    candles = [
        Candle(
            time=t,
            open=round(float(r.open),   2),
            high=round(float(r.high),   2),
            low=round(float(r.low),     2),
            close=round(float(r.close), 2),
            volume=round(float(r.volume), 0),
        )
        for t, r in zip(unix, df.itertuples())
    ]

    return CandlesResponse(
        ticker=ticker.upper(),
        figi=figi,
        interval=interval,
        candles=candles,
    )
