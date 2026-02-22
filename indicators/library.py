"""Библиотека стандартных индикаторов (векторизация, без Python-циклов)."""
import pandas as pd
import numpy as np
from typing import Optional
from .base import Indicator


class SMA(Indicator):
    """Simple Moving Average."""

    def __init__(self, period: int = 20):
        super().__init__(name=f'SMA_{period}', params={'period': period})

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['close'].rolling(window=self.params['period']).mean()


class EMA(Indicator):
    """Exponential Moving Average."""

    def __init__(self, period: int = 12):
        super().__init__(name=f'EMA_{period}', params={'period': period})

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['close'].ewm(span=self.params['period'], adjust=False).mean()


class RSI(Indicator):
    """Relative Strength Index (полностью векторизованный)."""

    def __init__(self, period: int = 14):
        super().__init__(name=f'RSI_{period}', params={'period': period})

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        period = self.params['period']
        delta = df['close'].diff()

        # Разделяем приросты и убытки через where (без циклов)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # EWM-сглаживание
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))


class MACD(Indicator):
    """Moving Average Convergence/Divergence.

    Возвращает DataFrame с колонками: macd, signal, histogram.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(
            name='MACD',
            params={'fast': fast, 'slow': slow, 'signal': signal}
        )

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = self.params['fast']
        slow = self.params['slow']
        sig = self.params['signal']

        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=sig, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }, index=df.index)


class BollingerBands(Indicator):
    """Bollinger Bands.

    Возвращает DataFrame с колонками: upper, middle, lower, bandwidth, %b.
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(
            name=f'BB_{period}',
            params={'period': period, 'std_dev': std_dev}
        )

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.params['period']
        std_dev = self.params['std_dev']

        middle = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()

        upper = middle + std * std_dev
        lower = middle - std * std_dev

        # Дополнительные метрики (без циклов)
        bandwidth = (upper - lower) / middle
        pct_b = (df['close'] - lower) / (upper - lower)

        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'bandwidth': bandwidth,
            'pct_b': pct_b
        }, index=df.index)


class ATR(Indicator):
    """Average True Range."""

    def __init__(self, period: int = 14):
        super().__init__(name=f'ATR_{period}', params={'period': period})

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        period = self.params['period']

        # True Range (векторизованный)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()

        # Max по трём колонкам сразу
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        return tr.ewm(span=period, adjust=False).mean()


class Stochastic(Indicator):
    """Stochastic Oscillator.

    Возвращает DataFrame с колонками: k, d.
    """

    def __init__(self, k_period: int = 14, d_period: int = 3):
        super().__init__(
            name='Stochastic',
            params={'k_period': k_period, 'd_period': d_period}
        )

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        k = self.params['k_period']
        d = self.params['d_period']

        lowest_low = df['low'].rolling(window=k).min()
        highest_high = df['high'].rolling(window=k).max()

        k_line = 100.0 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        d_line = k_line.rolling(window=d).mean()

        return pd.DataFrame({'k': k_line, 'd': d_line}, index=df.index)


class IndicatorFactory:
    """Фабрика для создания индикаторов по строковому имени."""

    _registry = {
        'sma': SMA,
        'ema': EMA,
        'rsi': RSI,
        'macd': MACD,
        'bb': BollingerBands,
        'atr': ATR,
        'stoch': Stochastic,
    }

    @classmethod
    def create(cls, name: str, **params) -> Indicator:
        """Создать индикатор по имени.

        Пример: IndicatorFactory.create('rsi', period=14)
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(f"Неизвестный индикатор: {name!r}. Доступны: {available}")
        return cls._registry[name_lower](**params)

    @classmethod
    def register(cls, name: str, indicator_class: type):
        """Зарегистрировать новый индикатор в фабрике."""
        cls._registry[name.lower()] = indicator_class
        print(f"✅ Индикатор '{name}' зарегистрирован")

    @classmethod
    def list_all(cls):
        """Список всех доступных индикаторов."""
        return list(cls._registry.keys())
