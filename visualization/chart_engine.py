"""Графический движок на базе lightweight-charts."""
import pandas as pd
from typing import Optional, Dict
import logging

try:
    from lightweight_charts import Chart
    CHART_AVAILABLE = True
except ImportError:
    print("⚠️  lightweight-charts не установлен")
    CHART_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChartEngine:
    """Графический движок с поддержкой индикаторов."""

    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
        theme: str = 'dark'
    ):
        if not CHART_AVAILABLE:
            raise ImportError("pip install lightweight-charts")

        self.chart = Chart(toolbox=True)

        bg = '#1e1e1e' if theme == 'dark' else '#ffffff'
        fg = '#d1d4dc' if theme == 'dark' else '#191919'
        self.chart.layout(background_color=bg, text_color=fg)

        # Легенда с OHLC данными
        self.chart.legend(visible=True, font_size=13)

        # Показываем время на оси
        self.chart.time_scale(time_visible=True, seconds_visible=False)

        self._indicators: Dict[str, any] = {}
        self._current_data: Optional[pd.DataFrame] = None

        print(f"✅ ChartEngine инициализирован ({width}x{height}, {theme})")

    @staticmethod
    def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
        """Приводим DatetimeIndex к datetime64[ns] naive UTC.

        ПРИЧИНА ДЕФЕКТА:
        - Tinkoff API → pandas 2.x → datetime64[us, UTC] (микросекунды)
        - lightweight-charts-python ожидает datetime64[ns] (наносекунды)
        - Внутри библиотека делает int64 // 10^9 для получения секунд
        - С us: 1_737_302_400_000_000 // 10^9 = 1_737_302 сек = январь 1970 ❌
        - С ns: 1_737_302_400_000_000_000 // 10^9 = 1_737_302_400 сек = 2026 ✅
        """
        df = df.copy()
        idx = pd.to_datetime(df.index)

        # Убираем timezone (приводим к UTC)
        if getattr(idx, 'tz', None) is not None:
            idx = idx.tz_convert('UTC').tz_localize(None)

        # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: us → ns
        df.index = idx.astype('datetime64[ns]')
        return df

    def plot_candles(self, df: pd.DataFrame, title: str = "Price"):
        """Отобразить свечной график."""

        # Нормализуем ОДИН РАЗ — отсюда же берут add_line
        df = self._normalize_index(df)
        self._current_data = df

        self.chart.set(df[['open', 'high', 'low', 'close', 'volume']])
        self.chart.watermark(title)

        print(f"✅ График построен: {len(df)} свечей")
        print(f"   Период: {df.index[0]} → {df.index[-1]}")

    def add_line(
        self,
        name: str,
        data: pd.Series,
        color: str = '#2196F3',
        width: int = 2
    ):
        """Добавить линию индикатора на основной график."""
        if self._current_data is None:
            raise ValueError("Сначала вызовите plot_candles()")

        # Нормализуем индекс Series так же, как график
        s = data.copy()
        idx = pd.to_datetime(s.index)
        if getattr(idx, 'tz', None) is not None:
            idx = idx.tz_convert('UTC').tz_localize(None)
        s.index = idx.astype('datetime64[ns]')

        # Выравниваем по индексу основного графика
        s = s.reindex(self._current_data.index)

        line_data = pd.DataFrame(
            {name: s.values},
            index=self._current_data.index
        ).dropna()

        if line_data.empty:
            print(f"⚠️  {name!r}: нет данных после reindex")
            return

        # ИСПРАВЛЕНИЕ: color и width прямо в create_line [web:42]
        line = self.chart.create_line(name, color=color, width=width)
        line.set(line_data)

        self._indicators[name] = line
        print(f"✅ Индикатор: {name} ({len(line_data)} точек) [{color}]")

    def show(self, block: bool = True):
        """Показать график."""
        print("\n📊 Открытие графика... (закройте окно для продолжения)")
        self.chart.show(block=block)

    def clear_indicators(self):
        """Очистить все индикаторы."""
        self._indicators.clear()
        print("✅ Индикаторы очищены")
