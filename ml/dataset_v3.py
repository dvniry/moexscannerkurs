"""Мультимасштабный датасет v3 — factual рендер + OHLC labels + hourly candles."""
import os
os.environ['GRPC_DNS_RESOLVER'] = 'native'

import sys, hashlib, traceback
_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'russian_ca.cer'))
if os.path.exists(_cert):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _cert

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset as TorchDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.config import CFG, SCALES
from ml.candle_render_v2 import render_candles, N_RENDER_CHANNELS
from ml.dataset import add_indicators, load_imoex, INDICATOR_COLS
from ml.labels_ohlc import build_ohlc_labels
from ml.context_loader import load_context_series, build_context_features
from ml.hourly_encoder import (
    render_hourly_candles, N_HOURLY_CHANNELS,
    N_HOURS_PER_DAY, N_INTRADAY_DAYS,
)


# ── Пути кэша ────────────────────────────────────────────────────

def _cache_dir():
    d = "ml/cache_v3"
    os.makedirs(d, exist_ok=True)
    return d

def _img_path(ticker, W):
    return f"{_cache_dir()}/imgs_{ticker}_{W}.npy"

def _num_path(ticker, W):
    return f"{_cache_dir()}/nums_{ticker}_{W}.npy"

def _cls_path(ticker):
    return f"{_cache_dir()}/cls_{ticker}.npy"

def _ohlc_path(ticker):
    return f"{_cache_dir()}/ohlc_{ticker}.npy"

def _ctx_path(ticker):
    return f"{_cache_dir()}/ctx_{ticker}.npy"

def _hourly_path(ticker):
    return f"{_cache_dir()}/hourly_{ticker}.npy"


# ── Загрузка часовых свечей ──────────────────────────────────────

def _load_hourly_candles(client, figi: str, days_back: int = None):
    """Загружает часовые свечи для тикера ЧАНКАМИ.

    T-Bank API ограничивает период для часовых свечей до ~90 дней.
    Загружаем чанками по CHUNK_DAYS дней и склеиваем.

    Возвращает DataFrame с колонками open/high/low/close/volume
    и DateTimeIndex (по часам).
    """
    import time as _time
    from datetime import datetime, timedelta, timezone
    from t_tech.invest import Client, CandleInterval
    from t_tech.invest.utils import now as _now

    if days_back is None:
        days_back = CFG.days_back

    CHUNK_DAYS = 85  # API лимит ~90 дней для HOUR, берём с запасом
    TARGET = "invest-public-api.tbank.ru:443"
    all_frames = []
    end = _now()
    remaining = days_back
    chunk_num = 0

    while remaining > 0:
        chunk = min(remaining, CHUNK_DAYS)
        start = end - timedelta(days=chunk)
        try:
            with Client(client.token, target=TARGET) as api:
                candles = api.market_data.get_candles(
                    figi=figi,
                    from_=start,
                    to=end,
                    interval=CandleInterval.CANDLE_INTERVAL_HOUR,
                ).candles

            if candles:
                df_chunk = pd.DataFrame({
                    "time":   [cdl.time for cdl in candles],
                    "open":   [client._q(cdl.open)   for cdl in candles],
                    "high":   [client._q(cdl.high)   for cdl in candles],
                    "low":    [client._q(cdl.low)    for cdl in candles],
                    "close":  [client._q(cdl.close)  for cdl in candles],
                    "volume": [cdl.volume             for cdl in candles],
                }).set_index("time")
                all_frames.append(df_chunk)
                chunk_num += 1
                if chunk_num % 5 == 0:
                    print(f"      hourly chunk {chunk_num}: "
                          f"{len(df_chunk)} candles, remaining {remaining-chunk}d")
        except Exception as e:
            print(f"    [WARN] hourly chunk {start.date()}→{end.date()}: {e}")

        end = start
        remaining -= chunk
        # Rate limit: ~600 req/min, sleep between chunks
        _time.sleep(0.15)

    if not all_frames:
        print(f"    [WARN] Часовые свечи: ни один чанк не загружен")
        return None

    result = pd.concat(all_frames).sort_index()
    result = result[~result.index.duplicated(keep='first')]  # дедупликация
    print(f"    Часовые свечи загружены: {len(result)} свечей "
          f"({chunk_num} чанков, {days_back} дней)")
    return result


def _build_hourly_for_day(hourly_df: pd.DataFrame,
                           daily_date,
                           daily_high: float,
                           daily_low: float,
                           daily_close: float) -> np.ndarray:
    """Извлекает часовые свечи для одного дня и рендерит их.

    daily_date: дата дневной свечи (используется для фильтрации)
    Возвращает: (N_HOURLY_CHANNELS, N_HOURS_PER_DAY)
    """
    if hourly_df is None or hourly_df.empty:
        return np.zeros((N_HOURLY_CHANNELS, N_HOURS_PER_DAY), dtype=np.float32)

    # Фильтруем часовые свечи этого дня
    if hasattr(daily_date, 'date'):
        day = daily_date.date()
    else:
        day = pd.Timestamp(daily_date).date()

    mask = hourly_df.index.date == day
    day_hourly = hourly_df[mask]

    if len(day_hourly) == 0:
        return np.zeros((N_HOURLY_CHANNELS, N_HOURS_PER_DAY), dtype=np.float32)

    return render_hourly_candles(day_hourly, daily_close, daily_high, daily_low)


def _build_hourly_windows(hourly_df: pd.DataFrame,
                           daily_df: pd.DataFrame,
                           valid_indices: list) -> np.ndarray:
    """Строит массив часовых окон для всех валидных сэмплов.

    Для каждого сэмпла (дневная свеча) берём N_INTRADAY_DAYS последних дней
    часовых данных.

    Возвращает: np.ndarray shape (N_samples, N_INTRADAY_DAYS, N_HOURLY_CHANNELS, N_HOURS_PER_DAY)
    """
    results = []

    for idx in valid_indices:
        # Последние N_INTRADAY_DAYS дней (включая текущий)
        start_day_idx = max(0, idx - N_INTRADAY_DAYS + 1)
        days_window = daily_df.iloc[start_day_idx:idx + 1]

        window_renders = []
        for _, row_date in enumerate(days_window.index):
            d_high  = float(days_window.loc[row_date, 'high'])
            d_low   = float(days_window.loc[row_date, 'low'])
            d_close = float(days_window.loc[row_date, 'close'])
            render = _build_hourly_for_day(
                hourly_df, row_date, d_high, d_low, d_close)
            window_renders.append(render)

        # Паддинг если дней < N_INTRADAY_DAYS (начало истории)
        while len(window_renders) < N_INTRADAY_DAYS:
            window_renders.insert(
                0, np.zeros((N_HOURLY_CHANNELS, N_HOURS_PER_DAY), dtype=np.float32))

        # Берём последние N_INTRADAY_DAYS
        window_renders = window_renders[-N_INTRADAY_DAYS:]
        results.append(np.stack(window_renders))  # (N_days, C, T)

    return np.array(results, dtype=np.float32)  # (N, N_days, C, T)


# ── Dataset ──────────────────────────────────────────────────────

class LazyMultiScaleDatasetV3(TorchDataset):
    """
    Ленивая загрузка данных из кэша.
    Возвращает: (imgs_dict, nums_dict, cls_label, ohlc_label, ctx, hourly)
    """
    def __init__(self, records: list, ctx_dim: int, use_hourly: bool = True):
        self.records    = records
        self.ctx_dim    = ctx_dim
        self.use_hourly = use_hourly
        self._cache     = {}

    def _load(self, ticker: str):
        if ticker in self._cache:
            return self._cache[ticker]
        data = {}
        for W in SCALES:
            data[W]         = np.load(_img_path(ticker, W), mmap_mode='r')
            data[f'num_{W}'] = np.load(_num_path(ticker, W), mmap_mode='r') \
                               if os.path.exists(_num_path(ticker, W)) else None
        data['cls']  = np.load(_cls_path(ticker),  mmap_mode='r')
        data['ohlc'] = np.load(_ohlc_path(ticker), mmap_mode='r')
        ctx_p = _ctx_path(ticker)
        data['ctx']  = np.load(ctx_p, mmap_mode='r') \
                       if os.path.exists(ctx_p) else None

        # Hourly data
        hp = _hourly_path(ticker)
        if self.use_hourly and os.path.exists(hp):
            data['hourly'] = np.load(hp, mmap_mode='r')
        else:
            data['hourly'] = None

        self._cache[ticker] = data
        return data

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        ticker, local_idx = self.records[idx]
        data = self._load(ticker)

        # Factual рендеры: (N_RENDER_CHANNELS, W)
        imgs = {W: torch.tensor(data[W][local_idx]).float() for W in SCALES}

        # Числовые ряды: (W, n_indicator_cols)
        nums = {W: torch.tensor(data[f'num_{W}'][local_idx]).float()
                for W in SCALES} \
               if data[f'num_{SCALES[0]}'] is not None else None

        cls_y  = int(data['cls'][local_idx])
        ohlc_y = torch.tensor(data['ohlc'][local_idx]).float()

        ctx = torch.tensor(data['ctx'][local_idx]).float() \
              if data['ctx'] is not None \
              else torch.zeros(self.ctx_dim)

        # Hourly data: (N_INTRADAY_DAYS, N_HOURLY_CHANNELS, N_HOURS_PER_DAY)
        if self.use_hourly and data['hourly'] is not None:
            hourly = torch.tensor(data['hourly'][local_idx]).float()
        else:
            hourly = torch.zeros(N_INTRADAY_DAYS, N_HOURLY_CHANNELS,
                                 N_HOURS_PER_DAY)

        return imgs, nums, cls_y, ohlc_y, ctx, hourly


# ── Построение датасета ──────────────────────────────────────────

def build_multiscale_dataset_v3(
    df, imoex=None, context=None, ticker: str = "unknown",
    hourly_df: pd.DataFrame = None,
    force_rebuild: bool = False,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray],
           np.ndarray, np.ndarray, Optional[np.ndarray],
           Optional[np.ndarray]]:
    """
    Возвращает: (imgs, nums, cls_labels, ohlc_labels, ctx, hourly)
    hourly: (N, N_INTRADAY_DAYS, N_HOURLY_CHANNELS, N_HOURS_PER_DAY)
    """
    df = add_indicators(df.copy(), imoex).dropna()
    W_max = max(SCALES)
    F = CFG.future_bars

    # Проверяем кэш
    cache_ok = (
        all(os.path.exists(_img_path(ticker, W)) for W in SCALES) and
        all(os.path.exists(_num_path(ticker, W)) for W in SCALES) and
        os.path.exists(_cls_path(ticker)) and
        os.path.exists(_ohlc_path(ticker)) and
        os.path.exists(_hourly_path(ticker))
    )

    if cache_ok and not force_rebuild:
        print(f"    Кэш v3 найден для {ticker}")
        imgs = {W: np.load(_img_path(ticker, W)) for W in SCALES}
        nums = {W: np.load(_num_path(ticker, W)) for W in SCALES}
        cls  = np.load(_cls_path(ticker))
        ohlc = np.load(_ohlc_path(ticker))
        hourly = np.load(_hourly_path(ticker)) \
                 if os.path.exists(_hourly_path(ticker)) else None
        # Контекст
        ctx_list = []
        if context is not None:
            for i in range(W_max, len(df) - F):
                ctx_list.append(context[i])
        ctx = np.array(ctx_list, dtype=np.float32) if ctx_list else None
        return imgs, nums, cls, ohlc, ctx, hourly

    # ── Строим OHLC labels ────────────────────────────────────
    ohlc_all, cls_all, valid_all = build_ohlc_labels(df)

    # ── Рендерим и собираем ───────────────────────────────────
    scale_imgs = {W: [] for W in SCALES}
    scale_nums = {W: [] for W in SCALES}
    ctx_list   = []
    y_cls_list = []
    y_ohlc_list = []
    valid_daily_indices = []  # для часовых свечей

    # Нормируем числовые признаки по всему df
    num_arr  = df[INDICATOR_COLS].values.astype(np.float32)
    num_min  = num_arr.min(axis=0, keepdims=True)
    num_max  = num_arr.max(axis=0, keepdims=True)
    num_norm = (num_arr - num_min) / (num_max - num_min + 1e-9)

    total = len(df) - W_max - F
    for i, idx in enumerate(range(W_max, len(df) - F)):
        if not valid_all[idx]:
            continue
        if i % 100 == 0:
            print(f"    {ticker}: рендер v3 {i}/{total}", end='\r')
        for W in SCALES:
            window = df.iloc[idx - W : idx]
            scale_imgs[W].append(render_candles(window))
            scale_nums[W].append(num_norm[idx - W : idx])
        if context is not None:
            ctx_list.append(context[idx])
        y_cls_list.append(cls_all[idx])
        y_ohlc_list.append(ohlc_all[idx])
        valid_daily_indices.append(idx)

    print()

    imgs = {W: np.array(scale_imgs[W], dtype=np.float32) for W in SCALES}
    nums = {W: np.array(scale_nums[W], dtype=np.float32) for W in SCALES}
    cls  = np.array(y_cls_list, dtype=np.int64)
    ohlc = np.array(y_ohlc_list, dtype=np.float32)
    ctx  = np.array(ctx_list, dtype=np.float32) if ctx_list else None

    # ── Часовые свечи ────────────────────────────────────────
    hourly = None
    if hourly_df is not None and len(valid_daily_indices) > 0:
        print(f"    {ticker}: рендер часовых свечей...")
        hourly = _build_hourly_windows(hourly_df, df, valid_daily_indices)
        print(f"    {ticker}: hourly shape = {hourly.shape}")

    # Сохраняем кэш
    for W in SCALES:
        np.save(_img_path(ticker, W), imgs[W])
        np.save(_num_path(ticker, W), nums[W])
    np.save(_cls_path(ticker), cls)
    np.save(_ohlc_path(ticker), ohlc)
    if hourly is not None:
        np.save(_hourly_path(ticker), hourly)

    return imgs, nums, cls, ohlc, ctx, hourly


def build_full_multiscale_dataset_v3(force_rebuild: bool = False,
                                      use_hourly: bool = True):
    """Точка входа: загружаем все тикеры, строим датасет v3.

    Возвращает: (dataset, y_all, ctx_dim, ticker_lengths)
    """
    from api.routes.candles import get_client
    client = get_client()

    print("  Загружаем IMOEX...")
    imoex = load_imoex()

    records = []
    all_cls = []
    ctx_dim = None
    ticker_lengths = []

    for ticker in CFG.tickers:
        print(f"  Загружаем {ticker}...")
        try:
            figi = client.find_figi(ticker)
            if not figi:
                continue
            df = client.get_candles(figi=figi, interval=CFG.interval,
                                    days_back=CFG.days_back)
            if df is None or df.empty:
                continue

            # Часовые свечи
            hourly_df = None
            if use_hourly:
                print(f"    Часовые свечи для {ticker}...")
                hourly_df = _load_hourly_candles(client, figi)

            print(f"    Контекст для {ticker}...")
            ctx_series = load_context_series(ticker)
            ctx_feats  = build_context_features(
                ctx_series, df.index,
                ticker=ticker,
                imoex_close=imoex['close'] if imoex is not None else None,
            ) if ctx_series is not None else None

            imgs, nums, cls, ohlc, ctx, hourly = build_multiscale_dataset_v3(
                df, imoex, ctx_feats, ticker=ticker,
                hourly_df=hourly_df,
                force_rebuild=force_rebuild)
            if len(cls) == 0:
                continue

            ctx_p = _ctx_path(ticker)
            if ctx is not None:
                np.save(ctx_p, ctx)
                if ctx_dim is None:
                    ctx_dim = ctx.shape[1]
            elif ctx_dim is not None:
                np.save(ctx_p,
                        np.zeros((len(cls), ctx_dim), dtype=np.float32))

            for local_idx in range(len(cls)):
                records.append((ticker, local_idx))
            all_cls.append(cls)
            ticker_lengths.append((ticker, len(cls)))
            h_info = f"hourly={hourly.shape}" if hourly is not None else "no hourly"
            print(f"  {ticker}: {len(cls)} сэмплов, {h_info}")

        except Exception as e:
            print(f"  {ticker}: ошибка — {e}")
            traceback.print_exc()

    if not records:
        raise RuntimeError("Не удалось загрузить данные.")

    ctx_dim = ctx_dim or 0
    y_all   = np.concatenate(all_cls)
    dataset = LazyMultiScaleDatasetV3(records, ctx_dim, use_hourly=use_hourly)
    return dataset, y_all, ctx_dim, ticker_lengths


# ── temporal_split (same as v2) ──────────────────────────────────

def temporal_split(ticker_lengths: list,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   purge_bars: int = None) -> tuple:
    """Temporal split с purge gap — для каждого тикера отдельно."""
    if purge_bars is None:
        purge_bars = CFG.future_bars

    idx_train = []
    idx_val   = []
    idx_test  = []

    global_offset = 0

    for ticker, n in ticker_lengths:
        n_test  = max(int(n * test_ratio), 1)
        n_val   = max(int(n * val_ratio), 1)

        test_start = n - n_test
        val_end   = test_start - purge_bars
        val_start = val_end - n_val
        train_end = val_start - purge_bars

        if train_end < 10:
            for i in range(n):
                idx_train.append(global_offset + i)
            global_offset += n
            continue

        for i in range(0, train_end):
            idx_train.append(global_offset + i)
        for i in range(max(val_start, 0), max(val_end, 0)):
            idx_val.append(global_offset + i)
        for i in range(test_start, n):
            idx_test.append(global_offset + i)

        global_offset += n

    return (
        np.array(idx_train, dtype=np.int64),
        np.array(idx_val,   dtype=np.int64),
        np.array(idx_test,  dtype=np.int64),
    )
