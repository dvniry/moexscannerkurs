"""Мультимасштабный датасет v2 — factual рендер + OHLC labels."""
import os
os.environ['GRPC_DNS_RESOLVER'] = 'native'

import sys, hashlib
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


# ── Пути кэша ────────────────────────────────────────────────────

def _cache_dir():
    d = "ml/cache_v2"
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


# ── Dataset ──────────────────────────────────────────────────────

class LazyMultiScaleDatasetV2(TorchDataset):
    """
    Ленивая загрузка данных из кэша.
    Возвращает: (imgs_dict, nums_dict, cls_label, ohlc_label, ctx)
    """
    def __init__(self, records: list, ctx_dim: int):
        self.records = records
        self.ctx_dim = ctx_dim
        self._cache  = {}

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
        ohlc_y = torch.tensor(data['ohlc'][local_idx]).float()  # (F, 4)

        ctx = torch.tensor(data['ctx'][local_idx]).float() \
              if data['ctx'] is not None \
              else torch.zeros(self.ctx_dim)

        return imgs, nums, cls_y, ohlc_y, ctx


# ── Построение датасета ──────────────────────────────────────────

def build_multiscale_dataset_v2(
    df, imoex=None, context=None, ticker: str = "unknown",
    force_rebuild: bool = False,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray],
           np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Возвращает: (imgs, nums, cls_labels, ohlc_labels, ctx)
    """
    df = add_indicators(df.copy(), imoex).dropna()
    W_max = max(SCALES)
    F = CFG.future_bars

    # Проверяем кэш
    cache_ok = (
        all(os.path.exists(_img_path(ticker, W)) for W in SCALES) and
        all(os.path.exists(_num_path(ticker, W)) for W in SCALES) and
        os.path.exists(_cls_path(ticker)) and
        os.path.exists(_ohlc_path(ticker))
    )

    if cache_ok and not force_rebuild:
        print(f"    Кэш v2 найден для {ticker}")
        imgs = {W: np.load(_img_path(ticker, W)) for W in SCALES}
        nums = {W: np.load(_num_path(ticker, W)) for W in SCALES}
        cls  = np.load(_cls_path(ticker))
        ohlc = np.load(_ohlc_path(ticker))
        # Контекст
        ctx_list = []
        if context is not None:
            for i in range(W_max, len(df) - F):
                ctx_list.append(context[i])
        ctx = np.array(ctx_list, dtype=np.float32) if ctx_list else None
        return imgs, nums, cls, ohlc, ctx

    # ── Строим OHLC labels ────────────────────────────────────
    ohlc_all, cls_all, valid_all = build_ohlc_labels(df)

    # ── Рендерим и собираем ───────────────────────────────────
    scale_imgs = {W: [] for W in SCALES}
    scale_nums = {W: [] for W in SCALES}
    ctx_list   = []
    y_cls_list = []
    y_ohlc_list = []

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
            print(f"    {ticker}: рендер v2 {i}/{total}", end='\r')
        for W in SCALES:
            window = df.iloc[idx - W : idx]
            scale_imgs[W].append(render_candles(window))        # (N_ch, W)
            scale_nums[W].append(num_norm[idx - W : idx])       # (W, 30)
        if context is not None:
            ctx_list.append(context[idx])
        y_cls_list.append(cls_all[idx])
        y_ohlc_list.append(ohlc_all[idx])                       # (F, 4)

    print()

    imgs = {W: np.array(scale_imgs[W], dtype=np.float32) for W in SCALES}
    nums = {W: np.array(scale_nums[W], dtype=np.float32) for W in SCALES}
    cls  = np.array(y_cls_list, dtype=np.int64)
    ohlc = np.array(y_ohlc_list, dtype=np.float32)
    ctx  = np.array(ctx_list, dtype=np.float32) if ctx_list else None

    # Сохраняем кэш
    for W in SCALES:
        np.save(_img_path(ticker, W), imgs[W])
        np.save(_num_path(ticker, W), nums[W])
    np.save(_cls_path(ticker), cls)
    np.save(_ohlc_path(ticker), ohlc)

    return imgs, nums, cls, ohlc, ctx


def build_full_multiscale_dataset_v2(force_rebuild: bool = False):
    """Точка входа: загружаем все тикеры, строим датасет.

    Возвращает: (dataset, y_all, ctx_dim, ticker_lengths)
    ticker_lengths — список (ticker, n_samples) для temporal split.

    force_rebuild: принудительно перестроить кэш (нужно при смене days_back).
    """
    from api.routes.candles import get_client
    client = get_client()

    print("  Загружаем IMOEX...")
    imoex = load_imoex()

    records = []
    all_cls = []
    ctx_dim = None
    ticker_lengths = []  # [(ticker, n_samples), ...]

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

            print(f"    Контекст для {ticker}...")
            ctx_series = load_context_series(ticker)
            ctx_feats  = build_context_features(
                ctx_series, df.index,
                ticker=ticker,
                imoex_close=imoex['close'] if imoex is not None else None,
            ) if ctx_series is not None else None

            imgs, nums, cls, ohlc, ctx = build_multiscale_dataset_v2(
                df, imoex, ctx_feats, ticker=ticker,
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
            print(f"  {ticker}: {len(cls)} сэмплов")

        except Exception as e:
            print(f"  {ticker}: ошибка — {e}")
            import traceback; traceback.print_exc()

    if not records:
        raise RuntimeError("Не удалось загрузить данные.")

    ctx_dim = ctx_dim or 0
    y_all   = np.concatenate(all_cls)
    dataset = LazyMultiScaleDatasetV2(records, ctx_dim)
    return dataset, y_all, ctx_dim, ticker_lengths


def temporal_split(ticker_lengths: list,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   purge_bars: int = None) -> tuple:
    """
    Temporal split с purge gap — для каждого тикера отдельно.

    Для каждого тикера сэмплы идут хронологически (local_idx 0..N-1).
    Разбиваем:
      [0 ... train_end] | purge | [val_start ... val_end] | purge | [test_start ... N-1]

    purge_bars: зазор между split'ами (default = CFG.future_bars).

    Возвращает: (idx_train, idx_val, idx_test) — глобальные индексы.
    """
    if purge_bars is None:
        purge_bars = CFG.future_bars

    idx_train = []
    idx_val   = []
    idx_test  = []

    global_offset = 0

    for ticker, n in ticker_lengths:
        # Для каждого тикера: хронологический split
        n_test  = max(int(n * test_ratio), 1)
        n_val   = max(int(n * val_ratio), 1)

        # test = последние n_test сэмплов
        test_start = n - n_test

        # val = перед test, с purge gap
        val_end   = test_start - purge_bars
        val_start = val_end - n_val

        # train = все до val_start, с purge gap
        train_end = val_start - purge_bars

        # Защита от слишком маленьких тикеров
        if train_end < 10:
            # Слишком мало данных — всё в train (не используем для val/test)
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
