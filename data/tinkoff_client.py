"""Клиент для работы с T-Bank Invest API."""
from __future__ import annotations

from api.logger import get_logger
from datetime import timedelta
from functools import lru_cache
from typing import Dict, Optional

import pandas as pd

from t_tech.invest import Client, CandleInterval, InstrumentIdType
from t_tech.invest.utils import now

logger = get_logger(__name__)

# ── Константы ─────────────────────────────────────────────
TARGET   = "invest-public-api.tbank.ru:443"   # актуальный домен
SANDBOX  = "sandbox-invest-public-api.tbank.ru:443"

INTERVALS: Dict[str, CandleInterval] = {
    "1m":  CandleInterval.CANDLE_INTERVAL_1_MIN,
    "5m":  CandleInterval.CANDLE_INTERVAL_5_MIN,
    "15m": CandleInterval.CANDLE_INTERVAL_15_MIN,
    "1h":  CandleInterval.CANDLE_INTERVAL_HOUR,
    "1d":  CandleInterval.CANDLE_INTERVAL_DAY,
}


class TinkoffDataClient:
    """Клиент для получения рыночных данных из T-Bank Invest API.

    Особенности:
    - FIGI кэш в памяти (загружаем список акций один раз за сессию)
    - Свечи кэшируются по ключу figi+interval+days
    - Все print убраны — только logger
    """

    def __init__(self, token: str):
        if not token:
            raise ValueError("T-Bank token is required")

        self.token = token
        self._candle_cache: Dict[str, pd.DataFrame] = {}
        self._figi_cache:   Dict[str, str]          = {}   # ticker → figi
        self._figi_loaded   = False

        logger.info("TinkoffDataClient initialized (target=%s)", TARGET)

    # ── FIGI ──────────────────────────────────────────────

    def _load_figi_cache(self) -> None:
        """Загружаем все акции один раз и кэшируем ticker→figi."""
        if self._figi_loaded:
            return
        try:
            with Client(self.token, target=TARGET) as client:
                instruments = client.instruments.shares(
                    instrument_status=1
                ).instruments
            self._figi_cache = {i.ticker: i.figi for i in instruments}
            self._figi_loaded = True
            logger.info("FIGI cache loaded: %d instruments", len(self._figi_cache))
        except Exception as e:
            logger.error("Failed to load FIGI cache: %s", e)
            raise

    def find_figi(self, ticker: str) -> Optional[str]:
        """Найти FIGI по тикеру (с кэшем — O(1) после первой загрузки)."""
        ticker = ticker.upper()
        if ticker not in self._figi_cache:
            self._load_figi_cache()
        figi = self._figi_cache.get(ticker)
        if figi:
            logger.debug("FIGI found: %s → %s", ticker, figi)
        else:
            logger.warning("Ticker not found: %s", ticker)
        return figi

    def get_ticker_by_figi(self, figi: str) -> Optional[str]:
        """Обратный поиск: figi → ticker."""
        if not self._figi_loaded:
            self._load_figi_cache()
        for ticker, f in self._figi_cache.items():
            if f == figi:
                return ticker
        return None

    # ── Свечи ─────────────────────────────────────────────

    def get_candles(
        self,
        figi:      str,
        interval:  str = "1h",
        days_back: int = 100,
    ) -> pd.DataFrame:
        """Получить исторические свечи.

        Returns:
            DataFrame: index=time, columns=[open, high, low, close, volume]
        """
        if interval not in INTERVALS:
            raise ValueError(f"Неизвестный интервал '{interval}'. Доступны: {list(INTERVALS)}")

        cache_key = f"{figi}_{interval}_{days_back}"
        if cache_key in self._candle_cache:
            logger.debug("Candle cache hit: %s", cache_key)
            return self._candle_cache[cache_key].copy()

        try:
            with Client(self.token, target=TARGET) as client:
                candles = client.market_data.get_candles(
                    figi     = figi,
                    from_    = now() - timedelta(days=days_back),
                    to       = now(),
                    interval = INTERVALS[interval],
                ).candles
        except Exception as e:
            logger.error("Failed to get candles for %s: %s", figi, e)
            raise

        if not candles:
            logger.warning("No candles returned for %s (%s, %dd)", figi, interval, days_back)
            return pd.DataFrame()

        df = pd.DataFrame({
            "time":   [c.time for c in candles],
            "open":   [self._q(c.open)   for c in candles],
            "high":   [self._q(c.high)   for c in candles],
            "low":    [self._q(c.low)    for c in candles],
            "close":  [self._q(c.close)  for c in candles],
            "volume": [c.volume          for c in candles],
        }).set_index("time")

        self._candle_cache[cache_key] = df
        logger.info("Loaded %d candles: %s %s %dd", len(df), figi, interval, days_back)
        return df.copy()

    # ── Список инструментов ───────────────────────────────

    def get_all_tickers(self) -> pd.DataFrame:
        """Все акции: ticker, figi, name, currency."""
        if not self._figi_loaded:
            self._load_figi_cache()
        try:
            with Client(self.token, target=TARGET) as client:
                instruments = client.instruments.shares(
                    instrument_status=1
                ).instruments
            return pd.DataFrame([{
                "ticker":   i.ticker,
                "figi":     i.figi,
                "name":     i.name,
                "currency": i.currency,
            } for i in instruments])
        except Exception as e:
            logger.error("Failed to get all tickers: %s", e)
            return pd.DataFrame()

    # ── Кэш ───────────────────────────────────────────────

    def clear_cache(self) -> None:
        """Очистить кэш свечей (FIGI кэш не сбрасывается)."""
        self._candle_cache.clear()
        logger.info("Candle cache cleared")

    def clear_all_cache(self) -> None:
        """Полный сброс всех кэшей."""
        self._candle_cache.clear()
        self._figi_cache.clear()
        self._figi_loaded = False
        logger.info("All caches cleared")

    # ── Утилиты ───────────────────────────────────────────

    @staticmethod
    def _q(quotation) -> float:
        """Quotation → float."""
        return quotation.units + quotation.nano / 1e9
