"""Клиент для работы с Tinkoff Invest API (t-tech-investments)."""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

# ПРАВИЛЬНЫЙ ИМПОРТ для t-tech-investments
from t_tech.invest import Client, CandleInterval, InstrumentIdType
from t_tech.invest.utils import now

logger = logging.getLogger(__name__)


class TinkoffDataClient:
    """Клиент для получения данных из Tinkoff API.
    
    Принципы:
    - Векторизация: возвращаем DataFrame, не списки
    - Кеширование: избегаем повторных запросов
    - ООП: инкапсуляция работы с API
    """
    
    # Маппинг интервалов
    INTERVALS = {
        '1m': CandleInterval.CANDLE_INTERVAL_1_MIN,
        '5m': CandleInterval.CANDLE_INTERVAL_5_MIN,
        '15m': CandleInterval.CANDLE_INTERVAL_15_MIN,
        '1h': CandleInterval.CANDLE_INTERVAL_HOUR,
        '1d': CandleInterval.CANDLE_INTERVAL_DAY,
    }
    
    def __init__(self, token: str):
        """Инициализация клиента.
        
        Args:
            token: API токен Tinkoff
        """
        if not token:
            raise ValueError("Tinkoff token is required!")
        
        self.token = token
        self._cache: Dict[str, pd.DataFrame] = {}
        logger.info("TinkoffDataClient initialized")
        print("✅ TinkoffDataClient инициализирован")
        
    def get_candles(
        self, 
        figi: str, 
        interval: str = '1h',
        days_back: int = 100
    ) -> pd.DataFrame:
        """Получить исторические свечи.
        
        Args:
            figi: FIGI инструмента
            interval: Интервал (1m, 5m, 15m, 1h, 1d)
            days_back: Количество дней назад
            
        Returns:
            DataFrame с колонками: time, open, high, low, close, volume
        """
        cache_key = f"{figi}_{interval}_{days_back}"
        
        # Проверка кеша
        if cache_key in self._cache:
            logger.debug(f"Cache hit: {cache_key}")
            return self._cache[cache_key].copy()
        
        # Запрос к API
        try:
            with Client(self.token) as client:
                candles = client.market_data.get_candles(
                    figi=figi,
                    from_=now() - timedelta(days=days_back),
                    to=now(),
                    interval=self.INTERVALS[interval]
                ).candles
        except Exception as e:
            logger.error(f"Failed to get candles: {e}")
            print(f"❌ Ошибка получения свечей: {e}")
            raise
            
        if not candles:
            logger.warning(f"No candles returned for {figi}")
            print(f"⚠️  Свечи не получены для {figi}")
            return pd.DataFrame()
        
        # Векторизованная обработка: создаем DataFrame за один проход
        data = {
            'time': [c.time for c in candles],
            'open': [self._quotation_to_float(c.open) for c in candles],
            'high': [self._quotation_to_float(c.high) for c in candles],
            'low': [self._quotation_to_float(c.low) for c in candles],
            'close': [self._quotation_to_float(c.close) for c in candles],
            'volume': [c.volume for c in candles],
        }
        
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        
        # Кешируем
        self._cache[cache_key] = df
        
        logger.info(f"Loaded {len(df)} candles for {figi} ({interval})")
        print(f"✅ Загружено {len(df)} свечей")
        
        return df.copy()
    
    def get_ticker_by_figi(self, figi: str) -> Optional[str]:
        """Получить тикер по FIGI.
        
        Args:
            figi: FIGI инструмента
            
        Returns:
            Тикер или None
        """
        try:
            with Client(self.token) as client:
                instrument = client.instruments.get_instrument_by(
                    id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI,
                    id=figi
                ).instrument
                return instrument.ticker if instrument else None
        except Exception as e:
            logger.error(f"Failed to get ticker: {e}")
            return None
    
    def find_figi(self, ticker: str) -> Optional[str]:
        """Найти FIGI по тикеру.
        
        Args:
            ticker: Тикер (например, SBER)
            
        Returns:
            FIGI или None
        """
        try:
            with Client(self.token) as client:
                instruments = client.instruments.shares(
                    instrument_status=1  # INSTRUMENT_STATUS_BASE
                ).instruments
                
                for inst in instruments:
                    if inst.ticker == ticker:
                        logger.info(f"Found {ticker}: {inst.figi}")
                        print(f"✅ Найден {ticker}: {inst.figi}")
                        return inst.figi
                        
            logger.warning(f"Ticker {ticker} not found")
            print(f"⚠️  Тикер {ticker} не найден")
            return None
            
        except Exception as e:
            logger.error(f"Failed to find FIGI: {e}")
            print(f"❌ Ошибка поиска FIGI: {e}")
            return None
    
    def get_all_tickers(self) -> pd.DataFrame:
        """Получить список всех акций.
        
        Returns:
            DataFrame с колонками: ticker, figi, name, currency
        """
        try:
            with Client(self.token) as client:
                instruments = client.instruments.shares(
                    instrument_status=1
                ).instruments
                
            # Векторизация: сразу DataFrame
            data = {
                'ticker': [i.ticker for i in instruments],
                'figi': [i.figi for i in instruments],
                'name': [i.name for i in instruments],
                'currency': [i.currency for i in instruments],
            }
            
            df = pd.DataFrame(data)
            print(f"✅ Получено {len(df)} акций")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get tickers: {e}")
            print(f"❌ Ошибка получения списка акций: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def _quotation_to_float(quotation) -> float:
        """Конвертация Quotation в float.
        
        Args:
            quotation: Объект Quotation из API
            
        Returns:
            Цена в float
        """
        return quotation.units + quotation.nano / 1e9
    
    def clear_cache(self):
        """Очистить кеш."""
        self._cache.clear()
        logger.info("Cache cleared")
        print("✅ Кеш очищен")
