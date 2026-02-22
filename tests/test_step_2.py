"""Тест графического движка."""
import os
import sys

# Добавляем путь к модулям
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.tinkoff_client import TinkoffDataClient
from visualization.chart_engine import ChartEngine
from config import config


def test_step_2():
    """Проверка отображения графиков."""
    
    print("="*60)
    print("ТЕСТ ШАГ 2: Интерактивные графики")
    print("="*60)
    
    # 1. Получаем данные
    print("\n📊 Шаг 1: Загрузка данных...")
    client = TinkoffDataClient(token=config.tinkoff.token)
    figi = client.find_figi("SBER")
    df = client.get_candles(figi=figi, interval='1h', days_back=30)
    
    print(f"✅ Загружено {len(df)} свечей")
    print(f"   Период: {df.index[0]} - {df.index[-1]}")
    print(f"   Цена: {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f} RUB")
    
    # 2. Создаем график
    print("\n📈 Шаг 2: Создание графика...")
    chart = ChartEngine(width=1200, height=800, theme='dark')
    chart.plot_candles(df, title="SBER (1H)")
    
    # 3. Добавляем простую скользящую среднюю
    print("\n📉 Шаг 3: Добавление индикаторов...")
    
    # SMA 20 (векторизованный расчет)
    df['sma_20'] = df['close'].rolling(window=20).mean()
    chart.add_line('SMA 20', df['sma_20'], color='orange', width=2)
    
    # SMA 50
    df['sma_50'] = df['close'].rolling(window=50).mean()
    chart.add_line('SMA 50', df['sma_50'], color='cyan', width=2)
    
    # 4. Показываем
    print("\n" + "="*60)
    print("🎉 ГРАФИК ГОТОВ!")
    print("="*60)
    print("\nВы должны увидеть:")
    print("  ✅ Свечной график SBER")
    print("  ✅ Оранжевую линию SMA 20")
    print("  ✅ Голубую линию SMA 50")
    print("  ✅ Инструменты масштабирования (zoom, pan)")
    print("\n⚠️  Закройте окно графика для продолжения...")
    
    chart.show(block=True)
    
    print("\n" + "="*60)
    print("🎉 ШАГ 2 ЗАВЕРШЕН: Графический движок работает!")
    print("="*60)


if __name__ == '__main__':
    test_step_2()
