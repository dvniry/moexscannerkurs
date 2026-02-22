"""Тест программируемых индикаторов: класс-стиль и код-стиль."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from data.tinkoff_client import TinkoffDataClient
from visualization.chart_engine import ChartEngine
from indicators.library import EMA, RSI, SMA
from indicators.bar import Bar, BarSeries
from indicators.custom import CustomIndicator, CodeIndicator
from config import config


# ════════════════════════════════════════════
# СТИЛЬ 1: Класс (MQL5-like)
# ════════════════════════════════════════════

class EMACrossIndicator(CustomIndicator):
    """Пересечение двух EMA → +1 (бычье) / -1 (медвежье) / 0 (нет).
    
    Аналог iCustom в MQL5:
        Параметры заданы как атрибуты класса.
        on_init: создаём индикаторы.
        on_calculate: логика одного бара.
    """
    # Параметры (аналог input в MQL5)
    fast_period: int = 9
    slow_period: int = 21
    
    def on_init(self):
        """Аналог OnInit() — создаём нужные индикаторы."""
        self.fast = EMA(self.fast_period)
        self.slow = EMA(self.slow_period)
    
    def on_calculate(self, bar: Bar) -> float:
        """Аналог OnCalculate() — значение для одного бара."""
        if bar.prev is None:
            return 0.0
        
        # Текущие значения
        fast_cur = bar.indicator.get('fast', 0)
        slow_cur = bar.indicator.get('slow', 0)
        
        # Предыдущие значения
        fast_prev = bar.prev.indicator.get('fast', 0)
        slow_prev = bar.prev.indicator.get('slow', 0)
        
        # Пересечение вверх
        if fast_prev <= slow_prev and fast_cur > slow_cur:
            return 1.0
        
        # Пересечение вниз
        if fast_prev >= slow_prev and fast_cur < slow_cur:
            return -1.0
        
        return 0.0


class RSIZoneIndicator(CustomIndicator):
    """RSI зона: +1 (перекупленность), -1 (перепроданность), 0 (нейтрально)."""
    
    period:     int   = 14
    overbought: float = 70.0
    oversold:   float = 30.0
    
    def on_init(self):
        self.rsi = RSI(self.period)
    
    def on_calculate(self, bar: Bar) -> float:
        rsi_val = bar.indicator.get('rsi', 50)
        
        if rsi_val >= self.overbought:
            return 1.0   # Перекупленность
        if rsi_val <= self.oversold:
            return -1.0  # Перепроданность
        return 0.0


# ════════════════════════════════════════════
# СТИЛЬ 2: Код (Pine Script-like)
# ════════════════════════════════════════════

TREND_STRENGTH_CODE = """
ema_fast = EMA(period=params['fast'])(df)
ema_slow = EMA(period=params['slow'])(df)
rsi      = RSI(period=14)(df)

# Сила тренда: нормализованная разность EMA * RSI-фактор
ema_diff   = (ema_fast - ema_slow) / ema_slow * 100
rsi_factor = (rsi - 50) / 50   # от -1 до +1

result = ema_diff * rsi_factor
"""

VOLATILITY_INDEX_CODE = """
import numpy as np

# Простой индекс волатильности: rolling std / rolling mean * 100
rolling_std  = close.rolling(params['period']).std()
rolling_mean = close.rolling(params['period']).mean()
result = (rolling_std / rolling_mean * 100).fillna(0)
"""


def test_step_4():
    print("=" * 60)
    print("ТЕСТ ШАГ 4: Программируемые индикаторы")
    print("=" * 60)
    
    # 1. Загрузка данных
    print("\n📊 Загрузка данных...")
    client = TinkoffDataClient(token=config.tinkoff.token)
    figi   = client.find_figi("SBER")
    df     = client.get_candles(figi=figi, interval='1h', days_back=60)
    print(f"✅ Загружено {len(df)} свечей")
    
    # ────────────────────────────────────────
    # СТИЛЬ 1: Класс
    # ────────────────────────────────────────
    print("\n── СТИЛЬ 1: Класс (MQL5-like) ──")
    
    ema_cross = EMACrossIndicator(fast_period=9, slow_period=21)
    rsi_zone  = RSIZoneIndicator(period=14, overbought=70, oversold=30)
    
    cross_vals = ema_cross(df)
    rsi_vals   = rsi_zone(df)
    
    # Подсчёт сигналов
    buy_signals  = (cross_vals == 1.0).sum()
    sell_signals = (cross_vals == -1.0).sum()
    
    print(f"✅ EMA Cross (9/21):")
    print(f"   BUY  пересечений: {buy_signals}")
    print(f"   SELL пересечений: {sell_signals}")
    print(f"   Текущий сигнал:   {cross_vals.iloc[-1]:+.0f}")
    
    print(f"\n✅ RSI Zone (14 | OB=70 | OS=30):")
    print(f"   Перекуплен:   {(rsi_vals == 1.0).sum()} баров")
    print(f"   Перепродан:   {(rsi_vals == -1.0).sum()} баров")
    print(f"   Текущая зона: {rsi_vals.iloc[-1]:+.0f}")
    
    # ────────────────────────────────────────
    # СТИЛЬ 2: Код
    # ────────────────────────────────────────
    print("\n── СТИЛЬ 2: Код (Pine Script-like) ──")
    
    trend_strength = CodeIndicator(
        name='Trend Strength',
        code=TREND_STRENGTH_CODE,
        params={'fast': 9, 'slow': 21}
    )
    
    volatility_idx = CodeIndicator(
        name='Volatility Index',
        code=VOLATILITY_INDEX_CODE,
        params={'period': 20}
    )
    
    trend_vals = trend_strength(df)
    vol_vals   = volatility_idx(df)
    
    print(f"✅ Trend Strength: {trend_vals.iloc[-1]:+.4f}")
    print(f"✅ Volatility Index: {vol_vals.iloc[-1]:.4f}%")
    
    # ────────────────────────────────────────
    # Тест Bar объекта
    # ────────────────────────────────────────
    print("\n── Тест Bar объекта ──")
    
    indicators = {
        'ema_9':  EMA(9)(df),
        'ema_21': EMA(21)(df),
        'rsi':    RSI(14)(df),
    }
    
    bar_series = BarSeries(df, indicators=indicators)
    last_bar   = bar_series.last
    
    print(f"✅ Последний бар: {last_bar}")
    print(f"   Бычья свеча:  {last_bar.is_bullish}")
    print(f"   Тело:         {last_bar.body:.4f}")
    print(f"   Диапазон:     {last_bar.range:.4f}")
    print(f"   EMA 9:        {last_bar.indicator.get('ema_9', 'n/a'):.2f}")
    print(f"   RSI:          {last_bar.indicator.get('rsi', 'n/a'):.2f}")
    print(f"   Пред. Close:  {last_bar.prev.close:.2f}")
    
    # ────────────────────────────────────────
    # Визуализация
    # ────────────────────────────────────────
    print("\n📈 Построение графика...")
    
    chart = ChartEngine(theme='dark')
    chart.plot_candles(df, title="SBER (1H) — Custom Indicators")
    
    # EMA для контекста
    chart.add_line('EMA 9',  EMA(9)(df),  color='#FF9800', width=1)
    chart.add_line('EMA 21', EMA(21)(df), color='#2196F3', width=1)
    
    # Кастомные
    chart.add_line('Trend Strength', trend_vals,  color='#9C27B0', width=2)
    chart.add_line('EMA Cross',      cross_vals,  color='#4CAF50', width=2)
    
    print("\n" + "=" * 60)
    print("🎉 ШАГ 4 ЗАВЕРШЕН: Оба стиля работают!")
    print("   Класс-стиль (MQL5):      EMACrossIndicator, RSIZoneIndicator")
    print("   Код-стиль (Pine Script): TrendStrength, VolatilityIndex")
    print("=" * 60)
    
    chart.show(block=True)


if __name__ == '__main__':
    test_step_4()
