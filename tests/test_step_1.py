"""Тест Tinkoff клиента."""
import os
import sys

# ИСПРАВЛЕНИЕ: добавляем родительскую директорию в путь
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.tinkoff_client import TinkoffDataClient
from config import config


def test_step_1():
    """Проверка получения данных."""
    
    print("🔍 Проверка токена Tinkoff...")
    if not config.tinkoff.token:
        print("❌ ОШИБКА: TINKOFF_TOKEN не найден в .env файле!")
        print("   Создайте файл .env с содержимым:")
        print("   TINKOFF_TOKEN=ваш_токен_здесь")
        return
    
    print(f"✅ Токен найден: {config.tinkoff.token[:10]}...")
    
    try:
        # Инициализация
        print("\n🔄 Инициализация клиента...")
        client = TinkoffDataClient(token=config.tinkoff.token)
        
        # 1. Поиск FIGI
        print("\n🔍 Поиск FIGI для SBER...")
        figi = client.find_figi("SBER")
        if not figi:
            print("❌ Не удалось найти SBER. Проверьте токен и доступ к API.")
            return
        print(f"✅ SBER FIGI: {figi}")
        
        # 2. Получение свечей
        print("\n📊 Загрузка свечей (30 дней, 1h)...")
        df = client.get_candles(figi=figi, interval='1h', days_back=30)
        print(f"✅ Загружено {len(df)} свечей")
        print("\nПервые 5 свечей:")
        print(df.head())
        print(f"\nПоследняя цена: {df['close'].iloc[-1]:.2f} RUB")
        
        # 3. Список акций
        print("\n📋 Получение списка доступных акций...")
        tickers = client.get_all_tickers()
        print(f"✅ Доступно акций: {len(tickers)}")
        print("\nПервые 10 акций:")
        print(tickers.head(10))
        
        print("\n" + "="*60)
        print("🎉 ШАГ 1 ЗАВЕРШЕН: Tinkoff клиент работает!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        print("\nВозможные причины:")
        print("1. Неверный токен в .env файле")
        print("2. Нет доступа к интернету")
        print("3. API Tinkoff недоступен")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_step_1()
