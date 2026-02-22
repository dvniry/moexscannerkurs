"""Программируемые пользовательские индикаторы.

Два стиля написания:
  1. Класс (MQL5-like): наследуем CustomIndicator, пишем on_calculate
  2. Код (Pine Script-like): передаём строку с Python-кодом
"""
import pandas as pd
import numpy as np
from abc import abstractmethod
from typing import Any, Dict, Optional, Union
from .base import Indicator
from .library import SMA, EMA, RSI, MACD, BollingerBands, ATR, Stochastic
from .bar import Bar, BarSeries
import logging

logger = logging.getLogger(__name__)


# ─── Безопасный namespace для строкового стиля ─────────────

SAFE_NAMESPACE = {
    'np': np,
    'pd': pd,
    'abs': abs, 'round': round,
    'min': min, 'max': max,
    'sum': sum, 'len': len,
    # Готовые индикаторы
    'SMA': SMA, 'EMA': EMA, 'RSI': RSI,
    'MACD': MACD, 'BB': BollingerBands,
    'ATR': ATR, 'Stoch': Stochastic,
}


# ════════════════════════════════════════════════════════════
# СТИЛЬ 1: Класс (MQL5-like)
# ════════════════════════════════════════════════════════════

class CustomIndicator(Indicator):
    """Базовый класс для пользовательских индикаторов (MQL5-стиль).
    
    Использование:
        class MyIndicator(CustomIndicator):
            period: int = 14        # параметры как атрибуты класса
            
            def on_init(self):      # аналог OnInit()
                self.ema = EMA(self.period)
            
            def on_calculate(self, bar: Bar) -> float:  # аналог OnCalculate()
                return bar.indicator['ema']
        
        ind = MyIndicator(period=21)
        series = ind(df)  # → pd.Series
    """
    
    def __init__(self, **params):
        """Инициализация с параметрами.
        
        Параметры передаются как kwargs и переопределяют атрибуты класса.
        Пример: MyIndicator(period=21) — переопределит period=14
        """
        # Берём дефолтные параметры из аннотаций класса
        defaults = {
            key: getattr(self.__class__, key)
            for key in getattr(self.__class__, '__annotations__', {})
            if hasattr(self.__class__, key)
        }
        
        # Переопределяем переданными значениями
        merged = {**defaults, **params}
        
        # Применяем как атрибуты
        for key, val in merged.items():
            setattr(self, key, val)
        
        name = self.__class__.__name__
        super().__init__(name=name, params=merged)
        
        # Вызываем on_init (аналог OnInit)
        self.on_init()
    
    def on_init(self):
        """Инициализация индикаторов. Переопределить при необходимости."""
        pass
    
    @abstractmethod
    def on_calculate(self, bar: Bar) -> float:
        """Расчёт значения для одного бара. ОБЯЗАТЕЛЬНО переопределить.
        
        Args:
            bar: Текущий бар с OHLCV и индикаторами
            
        Returns:
            float — значение индикатора на этом баре
        """
        pass
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Запуск on_calculate для всех баров (вызывается из __call__).
        
        Векторизация: предрассчитываем все индикаторы заранее,
        затем проходим по барам один раз.
        """
        # Предрасчёт всех зарегистрированных индикаторов
        indicators = self._precalculate_indicators(df)
        
        # Строим серию баров с индикаторами
        bar_series = BarSeries(df, indicators=indicators)
        
        # Применяем on_calculate к каждому бару
        values = [
            self.on_calculate(bar) for bar in bar_series
        ]
        
        return pd.Series(values, index=df.index, name=self.name)
    
    def _precalculate_indicators(
        self, 
        df: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """Предрасчёт всех Indicator-атрибутов.
        
        Автоматически находит атрибуты типа Indicator
        и рассчитывает их для всего DataFrame.
        """
        result = {}
        
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue
            attr = getattr(self, attr_name, None)
            if isinstance(attr, Indicator):
                calc = attr(df)
                if isinstance(calc, pd.DataFrame):
                    # Для MACD/BB разворачиваем колонки
                    for col in calc.columns:
                        result[f"{attr_name}_{col}"] = calc[col]
                else:
                    result[attr_name] = calc
        
        return result


# ════════════════════════════════════════════════════════════
# СТИЛЬ 2: Строка кода (Pine Script-like)
# ════════════════════════════════════════════════════════════

class CodeIndicator(Indicator):
    """Индикатор, заданный строкой Python-кода (Pine Script-like).
    
    Использование:
        ind = CodeIndicator(
            name='EMA Diff',
            code='''
            fast = EMA(period=params['fast'])(df)
            slow = EMA(period=params['slow'])(df)
            result = fast - slow
            ''',
            params={'fast': 9, 'slow': 21}
        )
        series = ind(df)
    
    Правила:
        - Входные данные: df, close, open, high, low, volume, params
        - Финальный результат: переменная result (pd.Series)
        - Доступны: np, pd, SMA, EMA, RSI, MACD, BB, ATR, Stoch
    """
    
    def __init__(
        self,
        name: str,
        code: str,
        params: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name=name, params=params or {})
        self.code = code.strip()
        
        # Компилируем сразу — ловим SyntaxError при создании
        try:
            self._compiled = compile(
                self.code, f'<indicator:{name}>', 'exec'
            )
        except SyntaxError as e:
            raise SyntaxError(
                f"Синтаксическая ошибка в '{name}':\n{e}"
            )
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Выполнить код и вернуть result."""
        namespace = {**SAFE_NAMESPACE}
        namespace.update({
            'df':     df.copy(),
            'params': self.params,
            'close':  df['close'],
            'open':   df['open'],
            'high':   df['high'],
            'low':    df['low'],
            'volume': df['volume'],
        })
        
        try:
            exec(self._compiled, namespace)
        except Exception as e:
            raise RuntimeError(
                f"Ошибка в '{self.name}':\n"
                f"{type(e).__name__}: {e}"
            )
        
        if 'result' not in namespace:
            raise ValueError(
                f"'{self.name}': переменная 'result' не найдена. "
                f"Добавьте: result = ..."
            )
        
        result = namespace['result']
        if isinstance(result, (pd.Series, pd.DataFrame)):
            result.index = df.index
        
        return result
    
    def __repr__(self) -> str:
        first_line = self.code.split('\n')[0].strip()
        return f"CodeIndicator(name={self.name!r}, code='{first_line}...')"
