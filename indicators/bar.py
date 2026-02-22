"""Bar — объект одного бара.

Аналог структуры MqlRates в MQL5.
Даёт удобный доступ к OHLCV + предрассчитанным индикаторам.
"""
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Bar:
    """Один бар (свеча) с OHLCV и значениями индикаторов.
    
    Аналог MQL5:
        rates[i].close  →  bar.close
        rates[i].open   →  bar.open
        iMA(...)        →  bar.indicator['ema_9']
        rates[i-1]      →  bar.prev
    """
    # OHLCV (аналог MqlRates)
    time:   pd.Timestamp
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float
    
    # Индекс в DataFrame (0 = первый бар)
    index: int = 0
    
    # Значения индикаторов: {'ema_9': 310.5, 'rsi': 58.2, ...}
    indicator: Dict[str, Any] = field(default_factory=dict)
    
    # Предыдущий бар (bar.prev.close, bar.prev.indicator['rsi'])
    prev: Optional['Bar'] = field(default=None, repr=False)
    
    # ─── Свойства для удобства ─────────────────────────────
    
    @property
    def is_bullish(self) -> bool:
        """Бычья свеча (close > open)."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Медвежья свеча (close < open)."""
        return self.close < self.open
    
    @property
    def body(self) -> float:
        """Размер тела свечи."""
        return abs(self.close - self.open)
    
    @property
    def shadow_upper(self) -> float:
        """Верхняя тень."""
        return self.high - max(self.close, self.open)
    
    @property
    def shadow_lower(self) -> float:
        """Нижняя тень."""
        return min(self.close, self.open) - self.low
    
    @property
    def range(self) -> float:
        """Полный диапазон (high - low)."""
        return self.high - self.low
    
    def __repr__(self) -> str:
        return (
            f"Bar({self.time} | "
            f"O={self.open:.2f} H={self.high:.2f} "
            f"L={self.low:.2f} C={self.close:.2f} | "
            f"V={self.volume:,.0f})"
        )


class BarSeries:
    """Серия баров из DataFrame.
    
    Строит список Bar-объектов с предрассчитанными индикаторами.
    Каждый bar.prev указывает на предыдущий бар.
    """
    
    def __init__(
        self, 
        df: pd.DataFrame,
        indicators: Optional[Dict[str, pd.Series]] = None
    ):
        """Инициализация серии баров.
        
        Args:
            df: DataFrame с OHLCV
            indicators: словарь {название: Series} предрассчитанных индикаторов
        """
        self._df = df.copy()
        self._indicators = indicators or {}
        self._bars = self._build()
    
    def _build(self):
        """Построение списка Bar-объектов (векторизованно через itertuples)."""
        bars = []
        prev_bar = None
        
        # itertuples быстрее iterrows в ~10 раз
        for i, row in enumerate(self._df.itertuples()):
            # Значения индикаторов для этого бара
            ind_values = {
                name: float(series.iloc[i])
                for name, series in self._indicators.items()
                if i < len(series) and not pd.isna(series.iloc[i])
            }
            
            bar = Bar(
                time=row.Index,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
                index=i,
                indicator=ind_values,
                prev=prev_bar
            )
            
            bars.append(bar)
            prev_bar = bar
        
        return bars
    
    def __len__(self) -> int:
        return len(self._bars)
    
    def __getitem__(self, idx: int) -> Bar:
        return self._bars[idx]
    
    def __iter__(self):
        return iter(self._bars)
    
    @property
    def last(self) -> Bar:
        """Последний (текущий) бар."""
        return self._bars[-1]
    
    @property
    def first(self) -> Bar:
        """Первый бар."""
        return self._bars[0]
