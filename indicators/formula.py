"""Formula Language — математический язык для создания индикаторов.

Синтаксис специально упрощён для нематематиков:
    RESULT = EMA(9) - EMA(21)
    RESULT = RSI(14) > 70
    RESULT = (HIGH + LOW + CLOSE) / 3
    
Доступные функции:
    EMA(period)              — exponential moving average
    SMA(period)              — simple moving average
    RSI(period)              — relative strength index
    MACD(fast, slow, signal) — MACD линия
    BB_UPPER(period, std)    — верхняя полоса Боллинджера
    BB_LOWER(period, std)    — нижняя полоса Боллинджера
    BB_MIDDLE(period)        — средняя полоса
    ATR(period)              — average true range
    MAX(series, period)      — максимум за период
    MIN(series, period)      — минимум за период
    STD(series, period)      — стандартное отклонение
    ABS(series)              — модуль
    PREV(series)             — предыдущее значение
    CROSS_UP(a, b)           — пересечение вверх (1 / 0)
    CROSS_DOWN(a, b)         — пересечение вниз (1 / 0)
    IF(cond, true, false)    — условие

Встроенные переменные:
    CLOSE, OPEN, HIGH, LOW, VOLUME
"""

import re
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
import logging

from .base import Indicator
from .library import SMA, EMA, RSI, MACD, BollingerBands, ATR

logger = logging.getLogger(__name__)


# ─── Встроенные функции формул ────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=int(period), adjust=False).mean()

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=int(period)).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=int(period), adjust=False).mean()
    avg_loss = loss.ewm(span=int(period), adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def _macd_line(series: pd.Series,
               fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    ema_fast = series.ewm(span=int(fast), adjust=False).mean()
    ema_slow = series.ewm(span=int(slow), adjust=False).mean()
    return ema_fast - ema_slow

def _bb_upper(series: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
    m = series.rolling(int(period)).mean()
    s = series.rolling(int(period)).std()
    return m + s * float(std)

def _bb_lower(series: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
    m = series.rolling(int(period)).mean()
    s = series.rolling(int(period)).std()
    return m - s * float(std)

def _bb_middle(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(int(period)).mean()

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=int(period), adjust=False).mean()

def _rolling_max(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(int(period)).max()

def _rolling_min(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(int(period)).min()

def _std(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(int(period)).std()

def _prev(series: pd.Series) -> pd.Series:
    return series.shift(1)

def _cross_up(a, b) -> pd.Series:
    """Пересечение вверх: a пересекает b снизу вверх.
    a и b могут быть Series или скалярами.
    """
    # Приводим к Series с общим индексом
    if isinstance(a, pd.Series):
        idx = a.index
    elif isinstance(b, pd.Series):
        idx = b.index
    else:
        raise TypeError("CROSS_UP ожидает хотя бы один аргумент типа Series")

    if not isinstance(a, pd.Series):
        a = pd.Series(float(a), index=idx)
    if not isinstance(b, pd.Series):
        b = pd.Series(float(b), index=idx)

    return ((a.shift(1) <= b.shift(1)) & (a > b)).astype(float)


def _cross_down(a, b) -> pd.Series:
    """Пересечение вниз: a пересекает b сверху вниз.
    a и b могут быть Series или скалярами.
    """
    if isinstance(a, pd.Series):
        idx = a.index
    elif isinstance(b, pd.Series):
        idx = b.index
    else:
        raise TypeError("CROSS_DOWN ожидает хотя бы один аргумент типа Series")

    if not isinstance(a, pd.Series):
        a = pd.Series(float(a), index=idx)
    if not isinstance(b, pd.Series):
        b = pd.Series(float(b), index=idx)

    return ((a.shift(1) >= b.shift(1)) & (a < b)).astype(float)


def _if(cond, true_val, false_val) -> pd.Series:
    """Условный выбор: IF(RSI(14) > 70, 1, 0)."""
    if isinstance(cond, pd.Series):
        return pd.Series(
            np.where(cond, true_val, false_val),
            index=cond.index
        )
    return true_val if cond else false_val


# ─── Класс Formula ────────────────────────────────────────

class Formula(Indicator):
    """Индикатор, заданный формулой на Formula Language.
    
    Примеры:
        # Разность EMA
        Formula('EMA Diff', 'RESULT = EMA(9) - EMA(21)')
        
        # RSI зоны
        Formula('RSI Zone', 'RESULT = IF(RSI(14) > 70, 1, IF(RSI(14) < 30, -1, 0))')
        
        # Собственный индекс волатильности
        Formula('Volatility', '''
            RANGE = HIGH - LOW
            RESULT = SMA(RANGE, 20)
        ''')
        
        # Объёмное давление
        Formula('Vol Pressure', '''
            HL = HIGH - LOW
            RESULT = (CLOSE - LOW) / HL * VOLUME
        ''')
        
        # С параметрами
        Formula('Custom EMA', 'RESULT = EMA(CLOSE, params.fast) - EMA(CLOSE, params.slow)',
                params={'fast': 9, 'slow': 21})
    """
    
    def __init__(
        self,
        name: str,
        formula: str,
        params: Optional[Dict[str, Any]] = None
    ):
        """Инициализация формулы.
        
        Args:
            name:    Название индикатора
            formula: Текст формулы
            params:  Параметры (доступны как params.fast, params.slow и т.д.)
        """
        super().__init__(name=name, params=params or {})
        self.formula = formula.strip()
        
        # Параметры как объект (params.fast вместо params['fast'])
        self._params_obj = type('P', (), params or {})()
        
        # Компилируем при создании
        self._python_code = self._translate(self.formula)
        try:
            self._compiled = compile(
                self._python_code,
                f'<formula:{name}>',
                'exec'
            )
        except SyntaxError as e:
            raise SyntaxError(
                f"Ошибка в формуле '{name}':\n"
                f"  Формула: {formula}\n"
                f"  Код:     {self._python_code}\n"
                f"  Ошибка:  {e}"
            )
    
    def _translate(self, formula: str) -> str:
        """Переводим Formula Language в Python.
        
        EMA(9)           → _ema(CLOSE, 9)
        SMA(VOLUME, 20)  → _sma(VOLUME, 20)
        RSI(14)          → _rsi(CLOSE, 14)
        CROSS_UP(A, B)   → _cross_up(A, B)
        IF(cond, t, f)   → _if(cond, t, f)
        params.fast      → _params.fast
        """
        code = formula
        
        # Функции без явной series → используют CLOSE по умолчанию
        code = re.sub(
            r'\bEMA\((\d+)\)',
            r'_ema(CLOSE, \1)',
            code
        )
        code = re.sub(
            r'\bSMA\((\d+)\)',
            r'_sma(CLOSE, \1)',
            code
        )
        code = re.sub(
            r'\bRSI\((\d+)\)',
            r'_rsi(CLOSE, \1)',
            code
        )
        code = re.sub(
            r'\bATR\((\d+)\)',
            r'_atr(_df, \1)',
            code
        )
        
        # Функции с явной series
        code = re.sub(r'\bEMA\(', '_ema(', code)
        code = re.sub(r'\bSMA\(', '_sma(', code)
        code = re.sub(r'\bRSI\(', '_rsi(', code)
        code = re.sub(r'\bMACD\(', '_macd_line(CLOSE, ', code)
        code = re.sub(r'\bBB_UPPER\(', '_bb_upper(CLOSE, ', code)
        code = re.sub(r'\bBB_LOWER\(', '_bb_lower(CLOSE, ', code)
        code = re.sub(r'\bBB_MIDDLE\(', '_bb_middle(CLOSE, ', code)
        code = re.sub(r'\bMAX\(', '_rolling_max(', code)
        code = re.sub(r'\bMIN\(', '_rolling_min(', code)
        code = re.sub(r'\bSTD\(', '_std(', code)
        code = re.sub(r'\bPREV\(', '_prev(', code)
        code = re.sub(r'\bCROSS_UP\(', '_cross_up(', code)
        code = re.sub(r'\bCROSS_DOWN\(', '_cross_down(', code)
        code = re.sub(r'\bIF\(', '_if(', code)
        
        # params.fast → _params.fast
        code = re.sub(r'\bparams\.', '_params.', code)
        
        return code
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Выполнить формулу и вернуть RESULT."""
        
        namespace = {
            # Встроенные функции
            '_ema': _ema, '_sma': _sma, '_rsi': _rsi,
            '_macd_line': _macd_line, '_atr': _atr,
            '_bb_upper': _bb_upper, '_bb_lower': _bb_lower,
            '_bb_middle': _bb_middle,
            '_rolling_max': _rolling_max, '_rolling_min': _rolling_min,
            '_std': _std, '_prev': _prev,
            '_cross_up': _cross_up, '_cross_down': _cross_down,
            '_if': _if,
            
            # Данные OHLCV
            'CLOSE':  df['close'],
            'OPEN':   df['open'],
            'HIGH':   df['high'],
            'LOW':    df['low'],
            'VOLUME': df['volume'],
            '_df':    df,
            
            # Параметры
            '_params': self._params_obj,
            
            # Базовые
            'np': np,
            'pd': pd,
            'abs': abs,
        }
        
        try:
            exec(self._compiled, namespace)
        except Exception as e:
            raise RuntimeError(
                f"Ошибка в формуле '{self.name}':\n"
                f"  {type(e).__name__}: {e}\n"
                f"  Формула: {self.formula}"
            )
        
        if 'RESULT' not in namespace:
            raise ValueError(
                f"Формула '{self.name}': не найдена переменная RESULT.\n"
                f"Добавьте строку: RESULT = ..."
            )
        
        result = namespace['RESULT']
        if isinstance(result, (pd.Series, pd.DataFrame)):
            result.index = df.index
        
        return result
    
    def __repr__(self) -> str:
        return f"Formula(name={self.name!r}, formula={self.formula!r})"
