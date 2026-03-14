"""Общие утилиты для роутов."""
import asyncio
from litestar.exceptions import HTTPException
from api.routes.candles import get_client, _executor

MAX_DAYS = {
    "1m": 1, "5m": 3, "15m": 7, "1h": 30, "1d": 365,
}

def fetch_df(ticker: str, interval: str, days: int):
    """Загрузить DataFrame — вызывать только из потока (_executor)."""
    client = get_client()
    figi   = client.find_figi(ticker)
    if not figi:
        raise ValueError(f"Тикер '{ticker}' не найден")
    df = client.get_candles(figi=figi, interval=interval, days_back=days)
    if df is None or df.empty:
        raise ValueError(f"Нет данных для {ticker} / {interval}")
    return figi, df

async def run_in_thread(fn):
    """Запустить синхронную функцию в executor без блокировки event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn)
