"""Sprint 4: Часовой датасет для HourlySpecialist.

Загружает часовые OHLCV для всех тикеров, вычисляет 37 индикаторов
(те же что в V3 дневной модели, но на часовой гранулярности),
строит скользящее окно W=45 баров (≈5 торговых дней MOEX × 9h).

Метка: residual direction на следующий торговый день
  y = 1 если (close_D+1/close_D - 1) > imoex_return_D+1, иначе 0

Запуск smoke-test:
    python -m ml.hourly_only_dataset --smoke
    python -m ml.hourly_only_dataset --rebuild   # пересборка кэша
"""
from __future__ import annotations

import argparse
import os
import time

os.environ["GRPC_DNS_RESOLVER"] = "native"
_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "russian_ca.cer"))
if os.path.exists(_cert):
    os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = _cert

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset

from ml.config import CFG
from ml.dataset_v3 import add_indicators, INDICATOR_COLS

HOURLY_CACHE_VERSION = "h1.0.0"
HOURLY_WINDOW       = 45      # = 5 trading days × 9h/day
HOURLY_STEP         = 9       # advance one full trading day per sample
N_HOURLY_FEAT       = len(INDICATOR_COLS)   # 37 features
HOURLY_DAYS_BACK    = 1825    # 5 years of hourly data
CACHE_DIR           = os.path.join(os.path.dirname(__file__), "cache", "hourly")


def _get_client():
    """Ленивая инициализация TinkoffDataClient."""
    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv("TINKOFF_TOKEN", "")
    if not token:
        raise RuntimeError("TINKOFF_TOKEN не задан в .env")
    from data.tinkoff_client import TinkoffDataClient
    return TinkoffDataClient(token)


def _load_hourly_df(ticker: str, client, days_back: int = HOURLY_DAYS_BACK) -> pd.DataFrame:
    """Загружает часовые свечи тикера. Возвращает DataFrame с OHLCV."""
    figi = client.find_figi(ticker)
    if figi is None:
        raise ValueError(f"FIGI не найден для {ticker}")
    df = client._load_candles_chunked(figi=figi, interval="1h", days_back=days_back)
    if df.empty:
        raise ValueError(f"Нет часовых данных для {ticker}")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    # Убираем нулевые/невалидные свечи
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    return df


def _load_imoex_hourly(client, days_back: int = HOURLY_DAYS_BACK) -> pd.DataFrame | None:
    """Загружает часовые свечи IMOEX для RS-признаков."""
    try:
        uid = client.find_indicative_uid("IMOEX")
        if not uid:
            return None
        df = client._load_candles_chunked(uid=uid, interval="1h", days_back=days_back)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df = df[(df["close"] > 0)]
        return df
    except Exception as e:
        print(f"  [WARN] IMOEX hourly: {e}")
        return None


def _load_imoex_daily(client, days_back: int = HOURLY_DAYS_BACK) -> pd.Series | None:
    """Загружает дневные цены IMOEX для вычисления ежедневной доходности."""
    try:
        uid = client.find_indicative_uid("IMOEX")
        if not uid:
            return None
        df = client._load_candles_chunked(uid=uid, interval="1d", days_back=days_back)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index, utc=True).normalize()
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df["close"].astype(np.float32)
    except Exception as e:
        print(f"  [WARN] IMOEX daily: {e}")
        return None


def _build_features(df: pd.DataFrame, imoex_h: pd.DataFrame | None) -> np.ndarray:
    """Вычисляет 37 индикаторов на часовом датафрейме.

    Возвращает float32[N, 37] (NaN → 0, inf → clip±10).
    """
    df_feat = df[["open", "high", "low", "close", "volume"]].copy()

    if imoex_h is not None:
        # Выровнять IMOEX по тем же timestamps (только колонка close)
        imoex_aligned = imoex_h[["close"]].reindex(df_feat.index, method="ffill")
        add_indicators(df_feat, imoex=imoex_aligned)
    else:
        add_indicators(df_feat, imoex=None)

    # Выбрать только INDICATOR_COLS
    for col in INDICATOR_COLS:
        if col not in df_feat.columns:
            df_feat[col] = 0.0

    arr = df_feat[INDICATOR_COLS].values.astype(np.float32)
    arr = np.where(np.isnan(arr), 0.0, arr)
    arr = np.clip(arr, -10.0, 10.0)
    return arr


def _build_daily_close(df: pd.DataFrame) -> pd.Series:
    """Дневные цены закрытия из часовых данных (last close of each day)."""
    close = df["close"].copy()
    close.index = pd.to_datetime(close.index).normalize()
    # Last close per day
    daily = close.groupby(close.index).last().astype(np.float32)
    return daily


def _build_labels(
    hourly_dates: pd.DatetimeIndex,
    daily_close: pd.Series,
    imoex_daily: pd.Series | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Строит метки и маску для каждого возможного окна.

    Возвращает:
        labels     [N_windows] int8 — 1=UP (outperform IMOEX), 0=DOWN
        valid_mask [N_windows] bool — True если D+1 существует в обоих рядах
    """
    n_windows = max(0, len(hourly_dates) - HOURLY_WINDOW + 1)
    labels    = np.zeros(n_windows, dtype=np.int8)
    valid     = np.zeros(n_windows, dtype=bool)

    daily_idx = daily_close.index

    for w in range(n_windows):
        # Последний бар окна: hourly_dates[w + HOURLY_WINDOW - 1]
        last_hour = hourly_dates[w + HOURLY_WINDOW - 1]
        day_D     = last_hour.normalize()

        # Ищем следующий торговый день D+1
        pos = daily_idx.searchsorted(day_D)
        if pos >= len(daily_idx):
            continue
        # Убедимся что pos указывает на сам day_D (или позже)
        if daily_idx[pos] < day_D:
            pos += 1
        if pos + 1 >= len(daily_idx):
            continue

        close_D  = float(daily_close.iloc[pos])
        close_D1 = float(daily_close.iloc[pos + 1])
        if close_D < 1e-9:
            continue

        ticker_ret = (close_D1 - close_D) / close_D

        if imoex_daily is not None:
            date_D  = daily_idx[pos]
            date_D1 = daily_idx[pos + 1]
            if date_D in imoex_daily.index and date_D1 in imoex_daily.index:
                im_D  = float(imoex_daily[date_D])
                im_D1 = float(imoex_daily[date_D1])
                if im_D > 1e-9:
                    imoex_ret = (im_D1 - im_D) / im_D
                else:
                    imoex_ret = 0.0
            else:
                imoex_ret = 0.0
        else:
            imoex_ret = 0.0

        labels[w] = 1 if ticker_ret > imoex_ret else 0
        valid[w]  = True

    return labels, valid


def _cache_path(ticker: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{ticker}.npz")


def _build_ticker_cache(
    ticker: str,
    client,
    imoex_h: pd.DataFrame | None,
    imoex_daily: pd.Series | None,
    force: bool = False,
) -> dict | None:
    """Загружает/строит кэш для одного тикера. Возвращает dict с arrays."""
    path = _cache_path(ticker)
    if not force and os.path.exists(path):
        try:
            data = np.load(path, allow_pickle=True)
            version = str(data["version"]) if "version" in data.files else ""
            if version == HOURLY_CACHE_VERSION:
                return dict(data)
        except Exception:
            pass

    try:
        t0 = time.time()
        df = _load_hourly_df(ticker, client)
        if len(df) < HOURLY_WINDOW + 20:
            print(f"  [SKIP] {ticker}: слишком мало свечей ({len(df)})")
            return None

        feats  = _build_features(df, imoex_h)             # [N, 37]
        daily_close = _build_daily_close(df)               # pd.Series
        labels, valid = _build_labels(
            df.index, daily_close, imoex_daily)            # [N_windows]

        n_valid = int(valid.sum())
        if n_valid < 50:
            print(f"  [SKIP] {ticker}: мало валидных окон ({n_valid})")
            return None

        dates_arr = np.array([
            df.index[w + HOURLY_WINDOW - 1].strftime("%Y-%m-%d %H:%M")
            for w in range(len(labels))
        ], dtype=object)

        np.savez(path,
                 feats   = feats,
                 labels  = labels,
                 valid   = valid,
                 dates   = dates_arr,
                 version = np.array(HOURLY_CACHE_VERSION),
                 n_hours = np.array(len(df)))

        elapsed = time.time() - t0
        print(f"  {ticker}: {len(df)} h-bars → {n_valid} окон ({elapsed:.1f}s)")
        return {"feats": feats, "labels": labels, "valid": valid, "dates": dates_arr}

    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════

class HourlyDataset(TorchDataset):
    """Dataset с часовыми окнами [N, 45, 37] для HourlySpecialist.

    __getitem__ возвращает (x, y, date_str):
        x:    torch.float32  [45, 37]
        y:    int             0=DOWN, 1=UP (residual vs IMOEX)
        date: str             'YYYY-MM-DD HH:MM' последнего бара окна
    """

    def __init__(self, tickers: list[str] | None = None, rebuild: bool = False):
        self.records: list[tuple[str, int]] = []   # (ticker, window_idx)
        self._data:   dict[str, dict]       = {}   # ticker → cached arrays

        if tickers is None:
            tickers = CFG.tickers

        client = _get_client()
        imoex_h     = _load_imoex_hourly(client)
        imoex_daily = _load_imoex_daily(client)
        print(f"  IMOEX hourly: {'загружен' if imoex_h is not None else 'N/A'}  "
              f"IMOEX daily: {'загружен' if imoex_daily is not None else 'N/A'}")

        for ticker in tickers:
            data = _build_ticker_cache(ticker, client, imoex_h, imoex_daily, force=rebuild)
            if data is None:
                continue
            self._data[ticker] = data
            valid_idx = np.where(data["valid"])[0]
            # Семплируем с шагом HOURLY_STEP (≈1 торговый день)
            sampled = valid_idx[::HOURLY_STEP]
            for w_idx in sampled:
                self.records.append((ticker, int(w_idx)))

        n_up   = sum(self._data[t]["labels"][w] == 1 for t, w in self.records)
        n_down = len(self.records) - n_up
        print(f"\n  Итого: {len(self.records)} сэмплов  "
              f"UP={n_up} ({100*n_up/max(len(self.records),1):.1f}%)  "
              f"DOWN={n_down}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        ticker, w_idx = self.records[idx]
        data = self._data[ticker]
        feats  = data["feats"]             # [N_hours, 37]
        labels = data["labels"]
        dates  = data["dates"]

        # Извлечь окно [w_idx : w_idx + HOURLY_WINDOW]
        start = w_idx
        end   = w_idx + HOURLY_WINDOW
        x = feats[start:end].copy()        # [45, 37]

        if x.shape[0] < HOURLY_WINDOW:
            pad = np.zeros((HOURLY_WINDOW - x.shape[0], N_HOURLY_FEAT), dtype=np.float32)
            x = np.concatenate([pad, x], axis=0)

        y    = int(labels[w_idx])
        date = str(dates[w_idx])[:10] if w_idx < len(dates) else ""
        # B-13: возвращаем ticker — нужно для (date, ticker) join в meta_features
        return torch.tensor(x, dtype=torch.float32), y, date, ticker


# ══════════════════════════════════════════════════════════════════
# temporal_split — аналог dataset_v3.temporal_split
# ══════════════════════════════════════════════════════════════════

def temporal_split(
    dataset:    HourlyDataset,
    val_frac:   float = 0.15,
    test_frac:  float = 0.15,
) -> tuple[list[int], list[int], list[int]]:
    """Временное разбиение: сортировка по дате.

    Возвращает (train_indices, val_indices, test_indices).
    """
    dated = []
    for i, (ticker, w_idx) in enumerate(dataset.records):
        date_str = str(dataset._data[ticker]["dates"][w_idx])
        dated.append((date_str, i))

    dated.sort(key=lambda x: x[0])
    n = len(dated)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    train_idx = [dated[i][1] for i in range(n_train)]
    val_idx   = [dated[i][1] for i in range(n_train, n_train + n_val)]
    test_idx  = [dated[i][1] for i in range(n_train + n_val, n)]

    print(f"  temporal_split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    return train_idx, val_idx, test_idx


# ══════════════════════════════════════════════════════════════════
# CLI smoke-test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",   action="store_true", help="Тест на 3 тикерах")
    parser.add_argument("--rebuild", action="store_true", help="Пересборка кэша")
    parser.add_argument("--tickers", nargs="+", default=None)
    args = parser.parse_args()

    tickers = args.tickers or (["SBER", "LKOH", "GAZP"] if args.smoke else None)
    ds = HourlyDataset(tickers=tickers, rebuild=args.rebuild)

    print(f"\nDataset size: {len(ds)}")
    if len(ds) > 0:
        x, y, date = ds[0]
        print(f"Sample 0: x.shape={x.shape}  y={y}  date={date}")
        print(f"  x[:3] = {x[:3, :5]}")

    if len(ds) >= 10:
        tr, va, te = temporal_split(ds)
        print(f"Split: train={len(tr)} val={len(va)} test={len(te)}")
