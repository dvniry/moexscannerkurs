# data/tinkoff_client.py
import logging
import os
import random
import time
from datetime import timedelta

import pandas as pd

from t_tech.invest import Client, CandleInterval, Quotation
from t_tech.invest.exceptions import RequestError
from t_tech.invest.utils import now

logger = logging.getLogger(__name__)

TARGET = "invest-public-api.tbank.ru:443"

INTERVALS = {
    "1min": CandleInterval.CANDLE_INTERVAL_1_MIN,
    "5min": CandleInterval.CANDLE_INTERVAL_5_MIN,
    "15min": CandleInterval.CANDLE_INTERVAL_15_MIN,
    "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
    "1d": CandleInterval.CANDLE_INTERVAL_DAY,
}


def _is_rate_limit_error(exc: Exception) -> bool:
    s = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in s
        or "request limit exceeded" in s
        or "ratelimit_remaining=0" in s
    )


def _extract_retry_seconds(exc: Exception, default: int = 20) -> int:
    try:
        for arg in getattr(exc, "args", ()):
            reset = getattr(arg, "ratelimit_reset", None)
            if reset is not None:
                return max(int(reset), 1)
    except Exception:
        pass
    return default


class TinkoffDataClient:
    def __init__(self, token: str):
        self.token = token
        self._figi_cache: dict[str, str] = {}
        self._uid_cache: dict[str, str] = {}
        self._indicative_uid_cache: dict[str, str] = {}
        self._candle_cache: dict[str, pd.DataFrame] = {}

    @staticmethod
    def _q(q: Quotation) -> float:
        return q.units + q.nano / 1e9

    def find_figi(self, ticker: str) -> str | None:
        if ticker in self._figi_cache:
            return self._figi_cache[ticker]

        with Client(self.token, target=TARGET) as api:
            response = api.instruments.find_instrument(query=ticker)
            for inst in response.instruments:
                if getattr(inst, "ticker", None) == ticker and getattr(inst, "figi", None):
                    self._figi_cache[ticker] = inst.figi
                    if getattr(inst, "uid", None):
                        self._uid_cache[ticker] = inst.uid
                    return inst.figi

            for inst in response.instruments:
                if getattr(inst, "ticker", None) == ticker:
                    figi = getattr(inst, "figi", None)
                    if figi:
                        self._figi_cache[ticker] = figi
                        if getattr(inst, "uid", None):
                            self._uid_cache[ticker] = inst.uid
                        return figi

        logger.warning("FIGI not found for ticker=%s", ticker)
        return None

    def find_uid(self, ticker: str) -> str | None:
        if ticker in self._uid_cache:
            return self._uid_cache[ticker]

        with Client(self.token, target=TARGET) as api:
            response = api.instruments.find_instrument(query=ticker)
            for inst in response.instruments:
                if getattr(inst, "ticker", None) == ticker and getattr(inst, "uid", None):
                    self._uid_cache[ticker] = inst.uid
                    if getattr(inst, "figi", None):
                        self._figi_cache[ticker] = inst.figi
                    return inst.uid

            for inst in response.instruments:
                if getattr(inst, "ticker", None) == ticker:
                    uid = getattr(inst, "uid", None)
                    if uid:
                        self._uid_cache[ticker] = uid
                        if getattr(inst, "figi", None):
                            self._figi_cache[ticker] = inst.figi
                        return uid

        logger.warning("UID not found for ticker=%s", ticker)
        return None

    def find_indicative_uid(self, ticker: str) -> str | None:
        if ticker in self._indicative_uid_cache:
            return self._indicative_uid_cache[ticker]

        with Client(self.token, target=TARGET) as api:
            response = api.instruments.find_instrument(query=ticker)
            for inst in response.instruments:
                if getattr(inst, "ticker", None) == ticker:
                    uid = getattr(inst, "uid", None)
                    if uid:
                        self._indicative_uid_cache[ticker] = uid
                        return uid

        logger.warning("Indicative UID not found for ticker=%s", ticker)
        return None

    def _load_candles_chunked(
        self,
        *,
        interval: str,
        days_back: int,
        figi: str | None = None,
        uid: str | None = None,
    ) -> pd.DataFrame:
        if interval not in INTERVALS:
            raise ValueError(f"Неизвестный интервал '{interval}'. Доступны: {list(INTERVALS)}")
        if not figi and not uid:
            raise ValueError("Нужно передать figi или uid")

        chunk_days = 365 if interval == "1d" else 85 if interval == "1h" else days_back
        all_frames = []
        end = now()
        remaining = int(days_back)
        ident = figi or uid or "unknown"

        with Client(self.token, target=TARGET) as api:
            while remaining > 0:
                chunk = min(remaining, chunk_days)
                start = end - timedelta(days=chunk)

                kwargs = {
                    "from_": start,
                    "to": end,
                    "interval": INTERVALS[interval],
                }
                if figi is not None:
                    kwargs["figi"] = figi
                else:
                    kwargs["instrument_id"] = uid

                candles = None
                last_exc = None

                for attempt in range(6):
                    try:
                        candles = api.market_data.get_candles(**kwargs).candles
                        last_exc = None
                        break
                    except RequestError as e:
                        last_exc = e
                        if _is_rate_limit_error(e):
                            reset_s = _extract_retry_seconds(e, default=20)
                            sleep_s = min(max(reset_s + random.uniform(0.3, 1.5), 1.0), 65.0)
                            logger.warning(
                                "Rate limit on get_candles(%s, %s) %s→%s, attempt %d/6, sleep %.1fs",
                                ident, interval, start.date(), end.date(), attempt + 1, sleep_s
                            )
                            time.sleep(sleep_s)
                            continue
                        raise
                    except Exception as e:
                        last_exc = e
                        if "RESOURCE_EXHAUSTED" in str(e):
                            sleep_s = 20.0 + random.uniform(0.3, 1.5)
                            logger.warning(
                                "RESOURCE_EXHAUSTED on get_candles(%s, %s) %s→%s, attempt %d/6, sleep %.1fs",
                                ident, interval, start.date(), end.date(), attempt + 1, sleep_s
                            )
                            time.sleep(sleep_s)
                            continue
                        raise

                if last_exc is not None and candles is None:
                    raise last_exc

                if candles:
                    df_chunk = pd.DataFrame({
                        "time":   [c.time for c in candles],
                        "open":   [self._q(c.open) for c in candles],
                        "high":   [self._q(c.high) for c in candles],
                        "low":    [self._q(c.low) for c in candles],
                        "close":  [self._q(c.close) for c in candles],
                        "volume": [c.volume for c in candles],
                    }).set_index("time")
                    all_frames.append(df_chunk)

                end = start
                remaining -= chunk

                time.sleep(0.08 if interval == "1d" else 0.12)

        if not all_frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        result = pd.concat(all_frames).sort_index()
        return result[~result.index.duplicated(keep="first")]

    def get_candles(
        self,
        figi: str,
        interval: str = "1h",
        days_back: int = 100,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_key = f"{figi}_{interval}_{days_back}"
        if use_cache and cache_key in self._candle_cache:
            return self._candle_cache[cache_key].copy()

        df = self._load_candles_chunked(
            figi=figi,
            interval=interval,
            days_back=days_back,
        )

        if use_cache:
            self._candle_cache[cache_key] = df

        logger.info("Loaded %d candles: %s %s %dd", len(df), figi, interval, days_back)
        return df.copy()

    def get_candles_by_uid(
        self,
        uid: str,
        interval: str = "1d",
        days_back: int = 730,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_key = f"uid_{uid}_{interval}_{days_back}"
        if use_cache and cache_key in self._candle_cache:
            return self._candle_cache[cache_key].copy()

        df = self._load_candles_chunked(
            uid=uid,
            interval=interval,
            days_back=days_back,
        )

        if use_cache:
            self._candle_cache[cache_key] = df

        logger.info("Loaded %d candles by uid: %s %s %dd", len(df), uid, interval, days_back)
        return df.copy()