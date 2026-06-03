"""TinkoffDataClient factory — нейтральная точка для ML pipeline.

Раньше ML модули тянули `get_client` из `api/routes/candles.py`, что подтягивало
litestar (web framework) в pipeline. Вынес сюда, чтобы training-код не зависел
от слоя API.
"""
from __future__ import annotations

from functools import lru_cache

from data.tinkoff_client import TinkoffDataClient
from config import config


@lru_cache(maxsize=1)
def get_client() -> TinkoffDataClient:
    return TinkoffDataClient(token=config.tinkoff.token)
