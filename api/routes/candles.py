"""GET /api/candles — свечи OHLCV."""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import List

import pandas as pd
from litestar import get
from litestar.exceptions import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from data.tinkoff_client import TinkoffDataClient
from config import config


@lru_cache(maxsize=1)
def get_client() -> TinkoffDataClient:
    return TinkoffDataClient(token=config.tinkoff.token)


def _fetch_candles(ticker: str, interval: str, days: int) -> tuple[str, pd.DataFrame]:
    client = get_client()
    figi = client.find_figi(ticker.upper())
    if not figi:
        raise ValueError(f"Тикер '{ticker}' не найден. Проверьте правильность написания.")
    df = client.get_candles(figi=figi, interval=interval, days_back=days)
    return figi, df


@dataclass
class Candle:
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class CandlesResponse:
    ticker: str
    figi: str
    interval: str
    candles: List[Candle]


def _to_unix(df: pd.DataFrame) -> list[int]:
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return (idx.astype("datetime64[ns]").astype("int64") // 10**9).tolist()


@get("/candles")
async def get_candles(
    ticker: str,
    interval: str = "1h",
    days: int = 30,
) -> CandlesResponse:
    try:
        figi, df = await asyncio.to_thread(_fetch_candles, ticker, interval, days)
    except Exception as e:
        get_client.cache_clear()
        raise HTTPException(status_code=400, detail=str(e))

    unix = _to_unix(df)
    candles = [
        Candle(
            time=t,
            open=round(float(r.open), 2),
            high=round(float(r.high), 2),
            low=round(float(r.low), 2),
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