# ══════════════════════════════════════════════════════════════════
# ml/cache_manager.py  — умный менеджер кэша
# ══════════════════════════════════════════════════════════════════
import os, json, random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

CACHE_META_PATH = "ml/cache_v3/_meta.json"
PROBE_TICKERS   = 5          # сколько тикеров проверяем для freshness
PROBE_CANDLES   = 10         # сколько последних свечей сравниваем


def _load_meta() -> dict:
    """Загружает метаданные кэша."""
    if os.path.exists(CACHE_META_PATH):
        with open(CACHE_META_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_meta(meta: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_META_PATH), exist_ok=True)
    with open(CACHE_META_PATH, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def _last_trading_date() -> str:
    """Возвращает дату последнего торгового дня MOEX (Пн-Пт до 18:50 МСК).
    Упрощённо: если сейчас > 19:00 МСК — сегодня, иначе — вчера.
    """
    from datetime import timezone
    now_msk = datetime.now(timezone.utc) + timedelta(hours=3)
    if now_msk.weekday() >= 5:  # суббота/воскресенье
        days_back = now_msk.weekday() - 4
        last = now_msk - timedelta(days=days_back)
    elif now_msk.hour < 19:
        last = now_msk - timedelta(days=1)
        if last.weekday() >= 5:
            last -= timedelta(days=last.weekday() - 4)
    else:
        last = now_msk
    return last.strftime("%Y-%m-%d")


def ticker_cache_valid(ticker: str, meta: dict) -> bool:
    """Проверяет достаточность кэша для тикера по метаданным.
    
    Не делает никаких API-запросов — только проверяет файлы + meta.
    """
    from ml.config import SCALES
    from ml.dataset_v3 import _img_path, _num_path, _cls_path, _ohlc_path

    # 1. Все файлы кэша существуют?
    files_ok = (
        all(os.path.exists(_img_path(ticker, W)) for W in SCALES) and
        all(os.path.exists(_num_path(ticker, W)) for W in SCALES) and
        os.path.exists(_cls_path(ticker)) and
        os.path.exists(_ohlc_path(ticker))
    )
    if not files_ok:
        return False

    # 2. Есть метаданные и они свежие?
    if ticker not in meta:
        return False

    t_meta = meta[ticker]
    last_trading = _last_trading_date()

    if t_meta.get("last_date") < last_trading:
        return False   # кэш устарел

    # 3. Количество сэмплов разумно?
    cached_n = t_meta.get("n_samples", 0)
    if cached_n < 100:
        return False

    return True


def probe_freshness(client, tickers: list, meta: dict,
                    n_probe: int = PROBE_TICKERS) -> bool:
    """Проверяет актуальность кэша через n_probe случайных тикеров.
    
    Скачивает только последние PROBE_CANDLES свечей и сравнивает
    с хвостом кэша. Занимает ~1-2 секунды вместо минут.
    
    Returns True если кэш актуален (можно пропустить скачивание).
    """
    from ml.config import CFG
    from ml.dataset_v3 import _cls_path

    # Выбираем тикеры у которых есть кэш
    cached = [t for t in tickers if ticker_cache_valid(t, meta)]
    if not cached:
        print("  Probe: нет кэшированных тикеров → полная загрузка")
        return False

    probe_set = random.sample(cached, min(n_probe, len(cached)))
    print(f"  Probe freshness: проверяем {probe_set}...")

    matches = 0
    for ticker in probe_set:
        try:
            figi = client.find_figi(ticker)
            if not figi:
                continue

            # Скачиваем только последние ~15 дней (быстро)
            df_fresh = client.get_candles(
                figi=figi, interval=CFG.interval, days_back=20
            )
            if df_fresh is None or df_fresh.empty:
                continue

            last_fresh_date = str(df_fresh.index[-1].date())
            last_cached_date = meta.get(ticker, {}).get("last_date", "")
            last_cached_close = meta.get(ticker, {}).get("last_close", None)

            date_ok  = (last_fresh_date == last_cached_date or
                        last_fresh_date <= last_cached_date)
            close_ok = True
            if last_cached_close is not None:
                fresh_close = float(df_fresh["close"].iloc[-1])
                # Допуск 0.01% — защита от незначительных пересчётов
                close_ok = abs(fresh_close - last_cached_close) / (last_cached_close + 1e-9) < 0.0001

            if date_ok and close_ok:
                matches += 1
                print(f"    {ticker}: ✓ кэш актуален ({last_cached_date})")
            else:
                print(f"    {ticker}: ✗ устарел "
                      f"(кэш={last_cached_date}, api={last_fresh_date})")

        except Exception as e:
            print(f"    {ticker}: probe error — {e}")

    ratio = matches / max(len(probe_set), 1)
    fresh = ratio >= 0.6   # 60% тикеров совпали → считаем кэш актуальным
    print(f"  Probe результат: {matches}/{len(probe_set)} совпало → "
          f"{'кэш АКТУАЛЕН ✓' if fresh else 'нужна загрузка ✗'}")
    return fresh


def update_meta(ticker: str, df: pd.DataFrame, n_samples: int,
                meta: dict) -> None:
    """Обновляет метаданные после успешной загрузки/рендера."""
    meta[ticker] = {
        "last_date":       str(df.index[-1].date()),
        "last_close":      float(df["close"].iloc[-1]),
        "n_samples":       n_samples,
        "n_candles":       len(df),
        "downloaded_at":   datetime.now().isoformat(),
        "interval":        "1d",
    }