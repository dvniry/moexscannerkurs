"""Тест Formula Language + видимость индикаторов."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.tinkoff_client import TinkoffDataClient
from visualization.chart_engine import ChartEngine
from indicators.library import EMA, RSI
from indicators.formula import Formula
from config import config


def test_step_4():
    print("=" * 60)
    print("ТЕСТ ШАГ 4: Formula Language")
    print("=" * 60)
    
    # Данные
    client = TinkoffDataClient(token=config.tinkoff.token)
    figi   = client.find_figi("SBER")
    df     = client.get_candles(figi=figi, interval='1h', days_back=60)
    print(f"✅ Загружено {len(df)} свечей\n")
    
    # ── Формулы ─────────────────────────────────────────────
    print("── Создание формул ──")
    
    # 1. Простая разность EMA
    ema_diff = Formula('EMA Diff', 'RESULT = EMA(9) - EMA(21)')
    
    # 2. RSI зона сигналов
    rsi_zone = Formula('RSI Zone',
        'RESULT = IF(RSI(14) > 70, 1, IF(RSI(14) < 30, -1, 0))')
    
    # 3. Типичная цена
    typical = Formula('Typical Price',
        'RESULT = (HIGH + LOW + CLOSE) / 3')
    
    # 4. Волатильность
    volatility = Formula('Volatility %', '''
RANGE = HIGH - LOW
AVG   = SMA(CLOSE, 20)
RESULT = RANGE / AVG * 100
    ''')
    
    # 5. Объёмное давление
    vol_pressure = Formula('Vol Pressure', '''
HL = HIGH - LOW
RESULT = (CLOSE - LOW) / HL * VOLUME
    ''')
    
    # 6. С параметрами
    custom_ma = Formula('Custom MA Cross',
        'RESULT = EMA(CLOSE, params.fast) - EMA(CLOSE, params.slow)',
        params={'fast': 9, 'slow': 21}
    )
    
    # 7. Пересечение EMA
    ema_cross = Formula('EMA Cross Signal',
        'RESULT = CROSS_UP(EMA(9), EMA(21)) - CROSS_DOWN(EMA(9), EMA(21))')
    
    # ── Расчёт ──────────────────────────────────────────────
    print("── Расчёт формул ──")
    
    results = {}
    formulas = [ema_diff, rsi_zone, typical, volatility,
                vol_pressure, custom_ma, ema_cross]
    
    for f in formulas:
        try:
            results[f.name] = f(df)
            val = results[f.name].iloc[-1]
            print(f"  ✅ {f.name:<22} → {val:+.4f}")
        except Exception as e:
            print(f"  ❌ {f.name}: {e}")
    
    # ── Визуализация ────────────────────────────────────────
    print("\n── Визуализация ──")
    chart = ChartEngine(theme='dark')
    chart.plot_candles(df, title="SBER (1H) — Formula Indicators")
    
    # На основном графике: скользящие + типичная цена
    chart.add_line('EMA 9',         EMA(9)(df),       color='#FF9800', width=1)
    chart.add_line('EMA 21',        EMA(21)(df),       color='#2196F3', width=1)
    chart.add_line('Typical Price', results['Typical Price'], color='#ffffff', width=1)
    
    # В субграфиках: осциллаторы
    chart.add_subchart_line('RSI 14',      RSI(14)(df),             color='#9C27B0', width=2)
    chart.add_subchart_line('EMA Diff',    results['EMA Diff'],     color='#4CAF50', width=2)
    chart.add_subchart_line('EMA Cross',   results['EMA Cross Signal'], color='#F44336', width=2)
    chart.add_subchart_line('Volatility %',results['Volatility %'], color='#FF5722', width=2)
    
    print("\n" + "=" * 60)
    print("🎉 ШАГ 4 ЗАВЕРШЕН!")
    print("\nПамятка Formula Language:")
    print("  RESULT = EMA(9) - EMA(21)")
    print("  RESULT = IF(RSI(14) > 70, 1, -1)")
    print("  RESULT = (HIGH + LOW + CLOSE) / 3")
    print("  RESULT = CROSS_UP(EMA(9), EMA(21))")
    print("=" * 60)
    
    chart.show(block=True)


if __name__ == '__main__':
    test_step_4()
