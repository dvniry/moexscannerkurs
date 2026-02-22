"""Базовый класс для всех индикаторов."""
import pandas as pd
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union


class Indicator(ABC):
    """Базовый класс для всех индикаторов.

    Принципы:
    - Единый интерфейс через __call__
    - Кэширование: не пересчитываем одни и те же данные
    - Векторизация: calculate работает только с Series/DataFrame, без циклов
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self._cache: Dict[str, Any] = {}

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """Расчёт индикатора. Переопределяется в каждом подклассе."""
        pass

    def __call__(self, df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """Вызов с автоматическим кэшированием."""
        key = self._cache_key(df)
        if key not in self._cache:
            self._cache[key] = self.calculate(df)
        return self._cache[key]

    def _cache_key(self, df: pd.DataFrame) -> str:
        """Уникальный ключ: имя + параметры + размер данных + последняя метка времени."""
        raw = f"{self.name}_{sorted(self.params.items())}_{len(df)}_{df.index[-1]}"
        return hashlib.md5(raw.encode()).hexdigest()

    def clear_cache(self):
        """Очистить кэш."""
        self._cache.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, params={self.params})"
