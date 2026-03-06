"""WebSocket /ws/price — real-time цена."""
import asyncio
import os
import sys

import pandas as pd
from litestar import WebSocket, websocket

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.routes.candles import get_client, _executor


def _fetch_price(ticker: str, interval: str) -> dict:
    """Синхронный вызов в потоке."""
    client = get_client()
    figi   = client.find_figi(ticker)
    df     = client.get_candles(figi=figi, interval=interval, days_back=1)

    last   = df.iloc[-1]
    idx    = pd.to_datetime(df.index[-1])
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_convert('UTC').tz_localize(None)
    unix = int(idx.to_datetime64().astype('int64') // 10**9)

    return {
        "price":  round(float(last['close']), 2),
        "time":   unix,
        # Текущая свеча для обновления графика
        "candle": {
            "time":  unix,
            "open":  round(float(last['open']),  2),
            "high":  round(float(last['high']),  2),
            "low":   round(float(last['low']),   2),
            "close": round(float(last['close']), 2),
        }
    }



@websocket("/ws/price")
async def price_ws(socket: WebSocket) -> None:
    await socket.accept()

    ticker     = "SBER"
    interval   = "1m"
    prev_price: float | None = None
    loop       = asyncio.get_event_loop()

    try:
        while True:
            try:
                msg      = await asyncio.wait_for(socket.receive_json(), timeout=0.1)
                ticker   = msg.get("ticker",   ticker).upper()
                interval = msg.get("interval", interval)
            except asyncio.TimeoutError:
                pass

            try:
                data   = await loop.run_in_executor(
                    _executor,
                    lambda: _fetch_price(ticker, interval)
                )
                price  = data["price"]
                change = round(price - prev_price, 2) if prev_price is not None else 0.0
                prev_price = price

                await socket.send_json({
                    "ticker": ticker,
                    "price":  price,
                    "change": change,
                })
            except Exception as e:
                await socket.send_json({"error": str(e)})

            await asyncio.sleep(5)

    except Exception:
        pass
    finally:
        try:
            await socket.close()
        except Exception:
            pass
