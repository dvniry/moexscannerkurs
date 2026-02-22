"""Централизованная конфигурация."""
import os
from dataclasses import dataclass, field
from typing import Optional

# Безопасная загрузка .env с обработкой ошибок кодировки
try:
    from dotenv import load_dotenv
    load_dotenv()
except UnicodeDecodeError:
    print("⚠️  Ошибка чтения .env файла (неправильная кодировка)")
    print("   Создайте .env файл в кодировке UTF-8")
except FileNotFoundError:
    print("⚠️  Файл .env не найден")
    print("   Создайте файл .env с содержимым: TINKOFF_TOKEN=ваш_токен")
except Exception as e:
    print(f"⚠️  Ошибка загрузки .env: {e}")


@dataclass
class TinkoffConfig:
    """Настройки Tinkoff API."""
    token: str = ""
    timeout: int = 30
    sandbox: bool = False
    
    def __post_init__(self):
        # Загружаем токен из переменной окружения
        if not self.token:
            self.token = os.getenv("TINKOFF_TOKEN", "")


@dataclass
class ChartConfig:
    """Настройки графиков."""
    theme: str = "dark"  # dark/light
    width: int = 1200
    height: int = 800
    default_interval: str = "1h"  # 1m, 5m, 15m, 1h, 1d


@dataclass
class Config:
    """Главная конфигурация."""
    tinkoff: TinkoffConfig = field(default_factory=TinkoffConfig)
    chart: ChartConfig = field(default_factory=ChartConfig)


# Создаем глобальный экземпляр конфигурации
config = Config()

# Проверка токена при загрузке модуля
if not config.tinkoff.token:
    print("\n⚠️  ВНИМАНИЕ: TINKOFF_TOKEN не найден!")
    print("   Создайте файл .env с содержимым:")
    print("   TINKOFF_TOKEN=ваш_токен_здесь\n")
