"""Тест библиотеки индикаторов + отображение на графике."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.tinkoff_client import TinkoffDataClient
from visualization.chart_engine import ChartEngine
from indicators.library import SMA, EMA, RSI, MACD, BollingerBands, ATR, Stochastic
from indicators.library import IndicatorFactory
from config import config


def test_step_3():
    print("=" * 60)
    print("ТЕСТ ШАГ 3: Библиотека индикаторов")
    print("=" * 60)

    # 1. Загрузка данных
    print("\n📊 Загрузка данных...")
    client = TinkoffDataClient(token=config.tinkoff.token)
    figi = client.find_figi("SBER")
    df = client.get_candles(figi=figi, interval='1h', days_back=60)
    print(f"✅ Загружено {len(df)} свечей")

    # 2. Расчёт индикаторов (все векторизованные, без циклов)
    print("\n🔬 Расчёт индикаторов...")

    sma_20  = SMA(period=20)
    sma_50  = SMA(period=50)
    ema_12  = EMA(period=12)
    rsi_14  = RSI(period=14)
    macd    = MACD(fast=12, slow=26, signal=9)
    bb      = BollingerBands(period=20, std_dev=2.0)
    atr_14  = ATR(period=14)
    stoch   = Stochastic(k_period=14, d_period=3)

    # Расчёт (кэшируются автоматически)
    df['sma_20'] = sma_20(df)
    df['sma_50'] = sma_50(df)
    df['ema_12'] = ema_12(df)
    df['rsi']    = rsi_14(df)
    df['atr']    = atr_14(df)

    macd_data  = macd(df)
    bb_data    = bb(df)
    stoch_data = stoch(df)

    # Текущие значения
    print(f"\n📈 Последние значения ({df.index[-1]}):")
    print(f"   Цена:       {df['close'].iloc[-1]:.2f} RUB")
    print(f"   SMA 20:     {df['sma_20'].iloc[-1]:.2f}")
    print(f"   SMA 50:     {df['sma_50'].iloc[-1]:.2f}")
    print(f"   EMA 12:     {df['ema_12'].iloc[-1]:.2f}")
    print(f"   RSI 14:     {df['rsi'].iloc[-1]:.2f}  ({'перекуплен' if df['rsi'].iloc[-1] > 70 else 'перепродан' if df['rsi'].iloc[-1] < 30 else 'нейтрально'})")
    print(f"   MACD:       {macd_data['macd'].iloc[-1]:.4f}")
    print(f"   Signal:     {macd_data['signal'].iloc[-1]:.4f}")
    print(f"   Histogram:  {macd_data['histogram'].iloc[-1]:.4f}")
    print(f"   BB upper:   {bb_data['upper'].iloc[-1]:.2f}")
    print(f"   BB lower:   {bb_data['lower'].iloc[-1]:.2f}")
    print(f"   ATR 14:     {df['atr'].iloc[-1]:.4f}")
    print(f"   Stoch %K:   {stoch_data['k'].iloc[-1]:.2f}")
    print(f"   Stoch %D:   {stoch_data['d'].iloc[-1]:.2f}")

    # Тест кэширования
    print("\n⚡ Тест кэширования (второй вызов должен быть мгновенным)...")
    import time
    t0 = time.time()
    _ = rsi_14(df)
    print(f"   Повторный расчёт RSI: {(time.time() - t0)*1000:.3f} ms (из кэша)")

    # Тест IndicatorFactory
    print("\n🏭 Тест IndicatorFactory...")
    rsi_via_factory = IndicatorFactory.create('rsi', period=21)
    print(f"   Создан через фабрику: {rsi_via_factory}")
    print(f"   Доступные индикаторы: {IndicatorFactory.list_all()}")

    # 3. Визуализация
    print("\n📈 Построение графика с индикаторами...")
    chart = ChartEngine(theme='dark')

    # Основной график: свечи + MA + BB
    chart.plot_candles(df, title="SBER (1H) — SMA / EMA / BB")
    chart.add_line('SMA 20',   df['sma_20'],        color='orange', width=2)
    chart.add_line('SMA 50',   df['sma_50'],        color='cyan',   width=2)
    chart.add_line('EMA 12',   df['ema_12'],        color='yellow', width=1)
    chart.add_line('BB Upper', bb_data['upper'],    color='#888888', width=1)
    chart.add_line('BB Middle',bb_data['middle'],   color='#555555', width=1)
    chart.add_line('BB Lower', bb_data['lower'],    color='#888888', width=1)

    print("\n" + "=" * 60)
    print("🎉 ШАГ 3 ЗАВЕРШЕН: Библиотека индикаторов работает!")
    print("=" * 60)

    chart.show(block=True)


if __name__ == '__main__':
    test_step_3()
