"""Мультимасштабный датасет с рыночным и отраслевым контекстом."""
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
from ml.candle_render import render_candles
from ml.dataset import add_indicators, label_candles, load_imoex, INDICATOR_COLS
from ml.context_loader import load_context_series, build_context_features


def _cache_path(ticker: str, W: int) -> str:
    return f"ml/cache/imgs_{ticker}_{W}.npy"

def _num_cache_path(ticker: str, W: int) -> str:
    return f"ml/cache/nums_{ticker}_{W}.npy"


class LazyMultiScaleDataset(TorchDataset):
    def __init__(self, records: list, ctx_dim: int):
        self.records = records
        self.ctx_dim = ctx_dim
        self._cache  = {}

    def _load(self, ticker: str):
        if ticker in self._cache:
            return self._cache[ticker]
        data = {W: np.load(_cache_path(ticker, W), mmap_mode='r')
                for W in SCALES}
        # Числовые ряды (могут отсутствовать в старом кэше)
        for W in SCALES:
            np_path = _num_cache_path(ticker, W)
            data[f'num_{W}'] = np.load(np_path, mmap_mode='r') \
                               if os.path.exists(np_path) else None
        data['y']   = np.load(f"ml/cache/labels_{ticker}.npy",  mmap_mode='r')
        ctx_path    = f"ml/cache/ctx_{ticker}.npy"
        data['ctx'] = np.load(ctx_path, mmap_mode='r') \
                      if os.path.exists(ctx_path) else None
        self._cache[ticker] = data
        return data

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        ticker, local_idx = self.records[idx]
        data = self._load(ticker)
        imgs = {W: torch.tensor(data[W][local_idx]).float() for W in SCALES}
        nums = {W: torch.tensor(data[f'num_{W}'][local_idx]).float()
                for W in SCALES} \
               if data[f'num_{SCALES[0]}'] is not None else None
        y   = int(data['y'][local_idx])
        ctx = torch.tensor(data['ctx'][local_idx]).float() \
              if data['ctx'] is not None \
              else torch.zeros(self.ctx_dim)
        return imgs, nums, y, ctx


def build_multiscale_dataset(
    df, imoex=None, context=None,
    ticker: str = "unknown",
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray],
           np.ndarray, Optional[np.ndarray]]:

    df     = add_indicators(df.copy(), imoex).dropna()
    labels = label_candles(df)
    W_max  = max(SCALES)

    cache_ok  = all(os.path.exists(_cache_path(ticker, W)) for W in SCALES)
    nums_ok   = all(os.path.exists(_num_cache_path(ticker, W)) for W in SCALES)
    label_path = f"ml/cache/labels_{ticker}.npy"

    if cache_ok and nums_ok and os.path.exists(label_path):
        print(f"    Кэш найден для {ticker}")
        imgs = {W: np.load(_cache_path(ticker, W))     for W in SCALES}
        nums = {W: np.load(_num_cache_path(ticker, W)) for W in SCALES}
        y    = np.load(label_path)
        ctx_list = []
        if context is not None:
            for i in range(W_max, len(df) - CFG.future_bars):
                ctx_list.append(context[i])
        ctx = np.array(ctx_list, dtype=np.float32) if ctx_list else None
        return imgs, nums, y, ctx

    os.makedirs("ml/cache", exist_ok=True)
    scale_imgs  = {W: [] for W in SCALES}
    scale_nums  = {W: [] for W in SCALES}
    ctx_list    = []
    y_list      = []

    # Нормируем числовые признаки по всему df сразу
    num_arr = df[INDICATOR_COLS].values.astype(np.float32)
    num_min = num_arr.min(axis=0, keepdims=True)
    num_max = num_arr.max(axis=0, keepdims=True)
    num_norm = (num_arr - num_min) / (num_max - num_min + 1e-9)

    total = len(df) - W_max - CFG.future_bars
    for i, idx in enumerate(range(W_max, len(df) - CFG.future_bars)):
        if i % 100 == 0:
            print(f"    {ticker}: рендер {i}/{total}", end='\r')
        for W in SCALES:
            window = df.iloc[idx - W : idx]
            scale_imgs[W].append(render_candles(window))
            # Числовой ряд: последние W строк нормированных признаков
            scale_nums[W].append(num_norm[idx - W : idx])
        if context is not None:
            ctx_list.append(context[idx])
        y_list.append(labels.iloc[idx])

    print()
    imgs = {W: np.array(scale_imgs[W], dtype=np.float32) for W in SCALES}
    nums = {W: np.array(scale_nums[W], dtype=np.float32) for W in SCALES}
    y    = np.array(y_list, dtype=np.int64)
    ctx  = np.array(ctx_list, dtype=np.float32) if ctx_list else None

    for W in SCALES:
        np.save(_cache_path(ticker, W),     imgs[W])
        np.save(_num_cache_path(ticker, W), nums[W])
    np.save(label_path, y)

    return imgs, nums, y, ctx


def build_full_multiscale_dataset():
    from api.routes.candles import get_client
    client = get_client()

    print("  Загружаем IMOEX...")
    imoex = load_imoex()

    records = []
    all_y   = []
    ctx_dim = None

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

            imgs, nums, y, ctx = build_multiscale_dataset(
                df, imoex, ctx_feats, ticker=ticker)
            if len(y) == 0:
                continue

            ctx_path = f"ml/cache/ctx_{ticker}.npy"
            if ctx is not None:
                np.save(ctx_path, ctx)
                if ctx_dim is None:
                    ctx_dim = ctx.shape[1]
            elif ctx_dim is not None:
                np.save(ctx_path,
                        np.zeros((len(y), ctx_dim), dtype=np.float32))

            for local_idx in range(len(y)):
                records.append((ticker, local_idx))
            all_y.append(y)
            print(f"  {ticker}: {len(y)} сэмплов")

        except Exception as e:
            print(f"  {ticker}: ошибка — {e}")
            import traceback; traceback.print_exc()

    if not records:
        raise RuntimeError("Не удалось загрузить данные.")

    ctx_dim = ctx_dim or 0
    y_all   = np.concatenate(all_y)
    dataset = LazyMultiScaleDataset(records, ctx_dim)
    return dataset, y_all, ctx_dim
