"""Базовый класс стратегии и Formula-стратегия."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class Signal:
    """Один торговый сигнал."""
    time:       int             # unix timestamp
    price:      float
    action:     str             # "BUY" | "SELL" | "STOP"
    size:       float   = 1.0   # % от депо или лоты
    reason:     str     = ""


class Strategy(ABC):
    name:     str = "Base"
    interval: str = "1h"

    @abstractmethod
    def entry(self, df: pd.DataFrame) -> pd.Series:
        """Возвращает boolean Series — True там где вход."""
        ...

    @abstractmethod
    def exit(self, df: pd.DataFrame) -> pd.Series:
        """Возвращает boolean Series — True там где выход."""
        ...

    def stop(self, df: pd.DataFrame) -> pd.Series:
        """Опционально: стоп-лосс условие."""
        return pd.Series(False, index=df.index)

    def position_size(self) -> float:
        """Размер позиции в % от баланса."""
        return 10.0


class FormulaStrategy(Strategy):
    """
    Стратегия через Formula Language прямо из UI.

    entry_formula  = "RESULT = CROSS_UP(EMA(9), EMA(21))"
    exit_formula   = "RESULT = CROSS_DOWN(EMA(9), EMA(21))"
    stop_formula   = "RESULT = CLOSE < EMA(21) * 0.97"   (опционально)
    """

    def __init__(
        self,
        name:          str,
        entry_formula: str,
        exit_formula:  str,
        stop_formula:  Optional[str] = None,
        size:          float         = 10.0,
        interval:      str           = "1h",
        params:        dict          = None,
    ):
        # импорт здесь чтобы не было циклических зависимостей
        from indicators.formula import Formula

        self.name          = name
        self.interval      = interval
        self._size         = size
        self._params       = params or {}

        self._entry_ind = Formula("entry", entry_formula, self._params)
        self._exit_ind  = Formula("exit",  exit_formula,  self._params)
        self._stop_ind  = Formula("stop",  stop_formula,  self._params) \
                          if stop_formula else None

    def entry(self, df: pd.DataFrame) -> pd.Series:
        return self._entry_ind(df).fillna(0).astype(bool)

    def exit(self, df: pd.DataFrame) -> pd.Series:
        return self._exit_ind(df).fillna(0).astype(bool)

    def stop(self, df: pd.DataFrame) -> pd.Series:
        if self._stop_ind:
            return self._stop_ind(df).fillna(0).astype(bool)
        return pd.Series(False, index=df.index)

    def position_size(self) -> float:
        return self._size
