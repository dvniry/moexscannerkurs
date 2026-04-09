# ml/dataset_v3.py
"""Мультимасштабный датасет v3 — самодостаточный (без зависимости от dataset.py).

Изменения v3.2 (nan-guard):
- __getitem__: nan_to_num на ohlc_y, imgs, nums, hourly
- build_multiscale_dataset_v3: nan_to_num на ohlc сразу после build_ohlc_labels
- add_indicators: защита от деления на ноль через clip(1e-9)
- _build_hourly_windows: защита от nan в рендере

Изменения v3.1 (consolidation):
- add_indicators, load_imoex, INDICATOR_COLS, class_distribution перенесены сюда
- from ml.dataset import ... полностью удалён
- dataset.py / dataset_v2_ohlc.py больше не нужны

ИСПРАВЛЕНИЕ: передаём train_end_idx в build_context_features для HMM без leakage.
ИСПРАВЛЕНИЕ 2: выравнивание массивов по минимальному размеру (cache desync fix).
ИСПРАВЛЕНИЕ 3: imgs сохраняются как (N, C=3, W) — mean-pool по H + transpose,
               совместимо с FactualBackbone(Conv1d, in_channels=3).
"""
import os
os.environ["GRPC_DNS_RESOLVER"] = "native"

import sys, hashlib, traceback
_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "russian_ca.cer"))
if os.path.exists(_cert):
    os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = _cert

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset as TorchDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.config import CFG, SCALES
from ml.candle_render_v2 import render_candles, N_RENDER_CHANNELS
from ml.labels_ohlc import build_ohlc_labels
from ml.context_loader import load_context_series, build_context_features
from ml.hourly_encoder import (
    render_hourly_candles, N_HOURLY_CHANNELS,
    N_HOURS_PER_DAY, N_INTRADAY_DAYS,
)


# ══════════════════════════════════════════════════════════════════
# Вспомогательная функция: (H, W, C) → (C, W)
# ══════════════════════════════════════════════════════════════════

def _hwc_to_cw(img: np.ndarray) -> np.ndarray:
    """Конвертирует рендер (H, W, C) → (C, W) mean-pool по оси H."""
    return img.mean(axis=0).T.astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# Индикаторы
# ══════════════════════════════════════════════════════════════════

INDICATOR_COLS = [
    "ema9", "ema21", "ema50", "macd", "macd_sig",
    "rsi", "rsi_pct", "stoch",
    "bb_upper", "bb_lower", "bb_pct",
    "atr", "range_norm", "range_atr_ratio",
    "vol_ratio", "sent_vol_price",
    "roc5", "roc10",
    "dist_fib_382", "dist_fib_618",
    "close",
    "rs_5d", "rs_20d", "imoex_ret5", "imoex_ret20", "imoex_vol20",
    "day_of_week", "month", "is_monday", "is_friday",
]


def add_indicators(df: pd.DataFrame, imoex: pd.DataFrame = None) -> pd.DataFrame:
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)

    df["ema9"]     = c.ewm(span=9).mean()
    df["ema21"]    = c.ewm(span=21).mean()
    df["ema50"]    = c.ewm(span=50).mean()
    df["macd"]     = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    df["macd_sig"] = df["macd"].ewm(span=9).mean()

    delta  = c.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    # FIX v3.2: clip(lower=1e-9) вместо replace(0, 1e-9) — надёжнее
    loss_s = (-delta.clip(upper=0)).rolling(14).mean().clip(lower=1e-9)
    df["rsi"]     = 100 - (100 / (1 + gain / loss_s))
    rsi_min       = df["rsi"].rolling(50).min()
    rsi_max       = df["rsi"].rolling(50).max()
    df["rsi_pct"] = (df["rsi"] - rsi_min) / (rsi_max - rsi_min + 1e-9)

    mid            = c.rolling(20).mean()
    std            = c.rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    tr          = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"]   = tr.rolling(14).mean()
    # FIX v3.2: защита от нулевого close
    c_safe = c.clip(lower=1e-9)
    df["range_norm"]      = (h - l) / c_safe
    df["range_atr_ratio"] = (h - l) / (df["atr"].clip(lower=1e-9))

    df["vol_ma"]         = v.rolling(20).mean()
    df["vol_ratio"]      = v / (df["vol_ma"].clip(lower=1e-9))
    df["ret1"]           = c.pct_change()
    df["sent_vol_price"] = df["ret1"] * df["vol_ratio"]

    df["roc5"]  = c.pct_change(5)
    df["roc10"] = c.pct_change(10)

    low14       = l.rolling(14).min()
    high14      = h.rolling(14).max()
    df["stoch"] = (c - low14) / (high14 - low14 + 1e-9) * 100

    roll_max           = c.rolling(30).max()
    roll_min           = c.rolling(30).min()
    fib_range          = (roll_max - roll_min).clip(lower=1e-9)
    df["dist_fib_382"] = (c - (roll_max - 0.382 * fib_range)) / fib_range
    df["dist_fib_618"] = (c - (roll_max - 0.618 * fib_range)) / fib_range

    if imoex is not None:
        idx_c             = imoex["close"].astype(float).reindex(df.index).ffill()
        # FIX v3.2: clip знаменатель RS чтобы не делить на 0
        df["rs_5d"]       = c.pct_change(5)  / (idx_c.pct_change(5).abs().clip(lower=1e-9))
        df["rs_20d"]      = c.pct_change(20) / (idx_c.pct_change(20).abs().clip(lower=1e-9))
        df["imoex_ret5"]  = idx_c.pct_change(5)
        df["imoex_ret20"] = idx_c.pct_change(20)
        df["imoex_vol20"] = idx_c.pct_change().rolling(20).std()
    else:
        for col in ("rs_5d", "rs_20d", "imoex_ret5", "imoex_ret20", "imoex_vol20"):
            df[col] = 0.0

    if hasattr(df.index, "dayofweek"):
        df["day_of_week"] = df.index.dayofweek / 4.0
        df["month"]       = df.index.month / 12.0
        df["is_monday"]   = (df.index.dayofweek == 0).astype(float)
        df["is_friday"]   = (df.index.dayofweek == 4).astype(float)
    else:
        df["day_of_week"] = df["month"] = 0.5
        df["is_monday"]   = df["is_friday"] = 0.0

    return df


def load_imoex() -> 'pd.DataFrame | None':
    from api.routes.candles import get_client
    import time as _time
    from datetime import timedelta
    from t_tech.invest import Client, CandleInterval
    from t_tech.invest.utils import now as _now

    client = get_client()
    TARGET = "invest-public-api.tbank.ru:443"

    try:
        uid = client.find_indicative_uid("IMOEX")
        if not uid:
            raise ValueError("UID для IMOEX не найден")

        CHUNK_DAYS = 365
        all_frames = []
        end        = _now()
        remaining  = CFG.days_back
        chunk_num  = 0

        while remaining > 0:
            chunk = min(remaining, CHUNK_DAYS)
            start = end - timedelta(days=chunk)
            try:
                with Client(client.token, target=TARGET) as api:
                    candles = api.market_data.get_candles(
                        instrument_id=uid,
                        from_=start,
                        to=end,
                        interval=CandleInterval.CANDLE_INTERVAL_DAY,
                    ).candles

                if candles:
                    df_chunk = pd.DataFrame({
                        "time":   [cdl.time   for cdl in candles],
                        "open":   [client._q(cdl.open)  for cdl in candles],
                        "high":   [client._q(cdl.high)  for cdl in candles],
                        "low":    [client._q(cdl.low)   for cdl in candles],
                        "close":  [client._q(cdl.close) for cdl in candles],
                        "volume": [cdl.volume for cdl in candles],
                    }).set_index("time")
                    all_frames.append(df_chunk)
                chunk_num += 1

            except Exception as e:
                print(f"  [WARN] IMOEX chunk {start.date()}→{end.date()}: {e}")

            end        = start
            remaining -= chunk
            _time.sleep(0.1)

        if not all_frames:
            raise ValueError("Ни один чанк IMOEX не загружен")

        df = pd.concat(all_frames).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        print(f"  IMOEX загружен: {len(df)} свечей ({chunk_num} чанков)")
        return df

    except Exception as e:
        print(f"  [WARN] IMOEX: {e}")
        print("  [WARN] IMOEX не загружен — RS признаки будут нулями")
        return None


def class_distribution(y: np.ndarray) -> None:
    total = len(y)
    for label, name in {0: "BUY", 1: "HOLD", 2: "SELL"}.items():
        count = int((y == label).sum())
        print(f"  {name:4s}: {count:5d} ({count / total * 100:.1f}%)")


# ══════════════════════════════════════════════════════════════════
# Утилита выравнивания кэша
# ══════════════════════════════════════════════════════════════════

def _align_arrays(imgs, nums, cls, ohlc, ctx=None, hourly=None):
    """Обрезает все массивы до минимального размера (защита от cache desync)."""
    n = min(len(cls), len(ohlc))
    for W in SCALES:
        n = min(n, len(imgs[W]), len(nums[W]))
    if ctx    is not None: n = min(n, len(ctx))
    if hourly is not None: n = min(n, len(hourly))
    cls  = cls[:n];  ohlc = ohlc[:n]
    imgs = {W: imgs[W][:n] for W in SCALES}
    nums = {W: nums[W][:n] for W in SCALES}
    if ctx    is not None: ctx    = ctx[:n]
    if hourly is not None: hourly = hourly[:n]
    return imgs, nums, cls, ohlc, ctx, hourly


# ══════════════════════════════════════════════════════════════════
# Пути кэша
# ══════════════════════════════════════════════════════════════════

def _cache_dir():
    d = "ml/cache_v3"; os.makedirs(d, exist_ok=True); return d

def _img_path(ticker, W):  return f"{_cache_dir()}/imgs_{ticker}_{W}.npy"
def _num_path(ticker, W):  return f"{_cache_dir()}/nums_{ticker}_{W}.npy"
def _cls_path(ticker):     return f"{_cache_dir()}/cls_{ticker}.npy"
def _ohlc_path(ticker):    return f"{_cache_dir()}/ohlc_{ticker}.npy"
def _ctx_path(ticker):     return f"{_cache_dir()}/ctx_{ticker}.npy"
def _hourly_path(ticker):  return f"{_cache_dir()}/hourly_{ticker}.npy"


# ══════════════════════════════════════════════════════════════════
# Часовые свечи
# ══════════════════════════════════════════════════════════════════

def _load_hourly_candles(client, figi: str, days_back: int = None):
    """Загружает часовые свечи ЧАНКАМИ (обход лимита 90 дней API)."""
    import time as _time
    from datetime import timedelta
    from t_tech.invest import Client, CandleInterval
    from t_tech.invest.utils import now as _now

    if days_back is None: days_back = CFG.days_back
    CHUNK_DAYS = 85
    TARGET     = "invest-public-api.tbank.ru:443"
    all_frames = []; end = _now(); remaining = days_back; chunk_num = 0

    while remaining > 0:
        chunk = min(remaining, CHUNK_DAYS); start = end - timedelta(days=chunk)
        try:
            with Client(client.token, target=TARGET) as api:
                candles = api.market_data.get_candles(
                    figi=figi, from_=start, to=end,
                    interval=CandleInterval.CANDLE_INTERVAL_HOUR,
                ).candles
            if candles:
                df_chunk = pd.DataFrame({
                    "time":   [c.time for c in candles],
                    "open":   [client._q(c.open)  for c in candles],
                    "high":   [client._q(c.high)  for c in candles],
                    "low":    [client._q(c.low)   for c in candles],
                    "close":  [client._q(c.close) for c in candles],
                    "volume": [c.volume            for c in candles],
                }).set_index("time")
                all_frames.append(df_chunk); chunk_num += 1
        except Exception as e:
            print(f"  [WARN] hourly chunk {start.date()}→{end.date()}: {e}")
        end = start; remaining -= chunk; _time.sleep(0.15)

    if not all_frames:
        print("  [WARN] Часовые свечи: ни один чанк не загружен"); return None
    result = pd.concat(all_frames).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    print(f"  Часовые свечи: {len(result)} свечей ({chunk_num} чанков)")
    return result


def _build_hourly_for_day(hourly_by_date, daily_date, d_high, d_low, d_close):
    if hourly_by_date is None:
        return np.zeros((N_HOURLY_CHANNELS, N_HOURS_PER_DAY), dtype=np.float32)
    day = daily_date.date() if hasattr(daily_date, "date") else pd.Timestamp(daily_date).date()
    day_h = hourly_by_date.get(day)
    if day_h is None or len(day_h) == 0:
        return np.zeros((N_HOURLY_CHANNELS, N_HOURS_PER_DAY), dtype=np.float32)
    result = render_hourly_candles(day_h, d_close, d_high, d_low)
    # FIX v3.2: защита от nan в часовом рендере
    if not np.isfinite(result).all():
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def _build_hourly_windows(hourly_df, daily_df, valid_indices):
    """Строит (N, N_INTRADAY_DAYS, N_HOURLY_CHANNELS, N_HOURS_PER_DAY)."""
    hourly_by_date = None
    if hourly_df is not None and not hourly_df.empty:
        hourly_by_date = {d: g for d, g in hourly_df.groupby(hourly_df.index.date)}
    results = []
    for i, idx in enumerate(valid_indices):
        if i % 500 == 0 and i: print(f"  hourly render {i}/{len(valid_indices)}", end="\r")
        start_day = max(0, idx - N_INTRADAY_DAYS + 1)
        days_win  = daily_df.iloc[start_day : idx + 1]
        renders   = []
        for row_date in days_win.index:
            renders.append(_build_hourly_for_day(
                hourly_by_date, row_date,
                float(days_win.loc[row_date, "high"]),
                float(days_win.loc[row_date, "low"]),
                float(days_win.loc[row_date, "close"]),
            ))
        while len(renders) < N_INTRADAY_DAYS:
            renders.insert(0, np.zeros((N_HOURLY_CHANNELS, N_HOURS_PER_DAY), dtype=np.float32))
        results.append(np.stack(renders[-N_INTRADAY_DAYS:]))
    print()
    return np.array(results, dtype=np.float32)


def _load_daily_candles_chunked(client, figi: str, days_back: int = None) -> pd.DataFrame:
    """Загружает дневные свечи ЧАНКАМИ по 365 дней."""
    import time as _time
    from datetime import timedelta
    from t_tech.invest import Client, CandleInterval
    from t_tech.invest.utils import now as _now

    if days_back is None:
        days_back = CFG.days_back

    CHUNK_DAYS = 365
    TARGET     = "invest-public-api.tbank.ru:443"
    all_frames = []
    end        = _now()
    remaining  = days_back
    chunk_num  = 0

    while remaining > 0:
        chunk = min(remaining, CHUNK_DAYS)
        start = end - timedelta(days=chunk)
        try:
            with Client(client.token, target=TARGET) as api:
                candles = api.market_data.get_candles(
                    figi=figi,
                    from_=start,
                    to=end,
                    interval=CandleInterval.CANDLE_INTERVAL_DAY,
                ).candles

            if candles:
                df_chunk = pd.DataFrame({
                    "time":   [cdl.time   for cdl in candles],
                    "open":   [client._q(cdl.open)  for cdl in candles],
                    "high":   [client._q(cdl.high)  for cdl in candles],
                    "low":    [client._q(cdl.low)   for cdl in candles],
                    "close":  [client._q(cdl.close) for cdl in candles],
                    "volume": [cdl.volume for cdl in candles],
                }).set_index("time")
                all_frames.append(df_chunk)
            chunk_num += 1

        except Exception as e:
            print(f"  [WARN] daily chunk {start.date()}→{end.date()}: {e}")

        end        = start
        remaining -= chunk
        _time.sleep(0.1)

    if not all_frames:
        print(f"  [WARN] Дневные свечи: ни один чанк не загружен")
        return pd.DataFrame()

    result = pd.concat(all_frames).sort_index()
    result = result[~result.index.duplicated(keep='first')]
    print(f"  Дневные свечи загружены: {len(result)} свечей "
          f"({chunk_num} чанков, {days_back} дней)")
    return result


# ══════════════════════════════════════════════════════════════════
# Dataset (lazy)
# ══════════════════════════════════════════════════════════════════

class LazyMultiScaleDatasetV3(TorchDataset):
    def __init__(self, records: list, ctx_dim: int, use_hourly: bool = True):
        self.records    = records
        self.ctx_dim    = ctx_dim
        self.use_hourly = use_hourly
        self._cache     = {}

    def _load(self, ticker: str):
        if ticker in self._cache: return self._cache[ticker]
        data = {}
        for W in SCALES:
            data[W]          = np.load(_img_path(ticker, W),  mmap_mode="r")
            p = _num_path(ticker, W)
            data[f"num_{W}"] = np.load(p, mmap_mode="r") if os.path.exists(p) else None
        data["cls"]  = np.load(_cls_path(ticker),  mmap_mode="r")
        data["ohlc"] = np.load(_ohlc_path(ticker), mmap_mode="r")
        cp = _ctx_path(ticker)
        data["ctx"]  = np.load(cp, mmap_mode="r") if os.path.exists(cp) else None
        hp = _hourly_path(ticker)
        data["hourly"] = np.load(hp, mmap_mode="r") if (self.use_hourly and os.path.exists(hp)) else None
        # выравниваем по минимальному n
        n = min(len(data["cls"]), len(data["ohlc"]))
        for W in SCALES:
            n = min(n, len(data[W]))
            if data[f"num_{W}"] is not None: n = min(n, len(data[f"num_{W}"]))
        if data["ctx"]    is not None: n = min(n, len(data["ctx"]))
        if data["hourly"] is not None: n = min(n, len(data["hourly"]))
        data["_n"] = n
        self._cache[ticker] = data
        return data

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        ticker, local_idx = self.records[idx]
        data = self._load(ticker)
        n = data["_n"]; local_idx = min(local_idx, n - 1)

        imgs = {W: torch.tensor(data[W][local_idx]).float() for W in SCALES}
        nums = ({W: torch.tensor(data[f"num_{W}"][local_idx]).float() for W in SCALES}
                if data[f"num_{SCALES[0]}"] is not None else None)
        cls_y  = int(data["cls"][local_idx])

        # FIX v3.2: nan/inf guard на ohlc_y — главный источник NaN loss
        ohlc_raw = data["ohlc"][local_idx]
        if not np.isfinite(ohlc_raw).all():
            ohlc_raw = np.nan_to_num(ohlc_raw, nan=0.0, posinf=0.0, neginf=0.0)
        ohlc_y = torch.tensor(ohlc_raw).float()

        # FIX v3.2: nan/inf guard на imgs
        for W in SCALES:
            if not torch.isfinite(imgs[W]).all():
                imgs[W] = torch.nan_to_num(imgs[W], nan=0.0, posinf=0.0, neginf=0.0)

        # FIX v3.2: nan/inf guard на nums
        if nums is not None:
            for W in SCALES:
                if not torch.isfinite(nums[W]).all():
                    nums[W] = torch.nan_to_num(nums[W], nan=0.0, posinf=1.0, neginf=0.0)

        ctx_arr = data["ctx"]
        if ctx_arr is not None and len(ctx_arr) > 0:
            ctx_idx = min(local_idx, len(ctx_arr) - 1)
            ctx = torch.tensor(ctx_arr[ctx_idx]).float()
            if not torch.isfinite(ctx).all():
                ctx = torch.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            ctx_dim_fallback = self.ctx_dim if self.ctx_dim > 0 else 1
            ctx = torch.zeros(ctx_dim_fallback)

        if self.use_hourly and data["hourly"] is not None:
            hourly = torch.tensor(data["hourly"][local_idx]).float()
            # FIX v3.2: nan/inf guard на hourly
            if not torch.isfinite(hourly).all():
                hourly = torch.nan_to_num(hourly, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            hourly = torch.zeros(N_INTRADAY_DAYS, N_HOURLY_CHANNELS, N_HOURS_PER_DAY)

        return imgs, nums, cls_y, ohlc_y, ctx, hourly


# ══════════════════════════════════════════════════════════════════
# Построение датасета для одного тикера
# ══════════════════════════════════════════════════════════════════

def build_multiscale_dataset_v3(
    df, imoex=None, context=None, ticker: str = "unknown",
    hourly_df: pd.DataFrame = None, force_rebuild: bool = False, 
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray],
           np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    df = add_indicators(df.copy(), imoex).dropna()
    W_max = max(SCALES); F = CFG.future_bars

    cache_ok = (
        all(os.path.exists(_img_path(ticker, W)) for W in SCALES) and
        all(os.path.exists(_num_path(ticker, W)) for W in SCALES) and
        os.path.exists(_cls_path(ticker)) and
        os.path.exists(_ohlc_path(ticker)) and
        os.path.exists(_hourly_path(ticker))
    )
    if cache_ok and not force_rebuild:
        print(f"  Кэш v3 найден для {ticker}")
        imgs   = {W: np.load(_img_path(ticker, W)) for W in SCALES}
        nums   = {W: np.load(_num_path(ticker, W)) for W in SCALES}
        cls    = np.load(_cls_path(ticker))
        ohlc   = np.load(_ohlc_path(ticker))
        hourly = np.load(_hourly_path(ticker)) if os.path.exists(_hourly_path(ticker)) else None

        # ── Авто-миграция старого кэша (H, W, C) → (C, W) ──────────
        for W in SCALES:
            if imgs[W].ndim == 4:
                print(f"  [migrate] {ticker} W={W}: (N,H,W,C) → (N,C,W)")
                imgs[W] = np.stack([_hwc_to_cw(imgs[W][i]) for i in range(len(imgs[W]))])
                np.save(_img_path(ticker, W), imgs[W])

        # FIX v3.2: санация кэшированного ohlc
        nan_count = (~np.isfinite(ohlc)).sum()
        if nan_count > 0:
            print(f"  [FIX] {ticker}: ohlc кэш содержит {nan_count} nan/inf → заменяем нулями")
            ohlc = np.nan_to_num(ohlc, nan=0.0, posinf=0.0, neginf=0.0)
            np.save(_ohlc_path(ticker), ohlc)

        ctx_list = []
        if context is not None:
            for i in range(W_max, len(df) - F): ctx_list.append(context[i])
        ctx = np.array(ctx_list, dtype=np.float32) if ctx_list else None
        imgs, nums, cls, ohlc, ctx, hourly = _align_arrays(imgs, nums, cls, ohlc, ctx, hourly)
        return imgs, nums, cls, ohlc, ctx, hourly

    ohlc_all, cls_all, valid_all, _atr_ratio = build_ohlc_labels(df)

    # FIX v3.2: санируем ohlc_all сразу после build_ohlc_labels
    # Это главный источник nan — ATR=0 на выходных или первых барах
    nan_before = (~np.isfinite(ohlc_all)).sum()
    if nan_before > 0:
        print(f"  [FIX] {ticker}: build_ohlc_labels вернул {nan_before} nan/inf "
              f"из {len(ohlc_all)} → заменяем нулями и инвалидируем метки")
        # Инвалидируем соответствующие valid-флаги чтобы не учить на мусоре
        for i in range(len(ohlc_all)):
            if not np.isfinite(ohlc_all[i]).all():
                valid_all[i] = False
        ohlc_all = np.nan_to_num(ohlc_all, nan=0.0, posinf=0.0, neginf=0.0)

    scale_imgs = {W: [] for W in SCALES}
    scale_nums = {W: [] for W in SCALES}
    ctx_list = []; y_cls_list = []; y_ohlc_list = []; valid_daily_indices = []

    num_arr  = df[INDICATOR_COLS].values.astype(np.float32)
    _col_min = np.nanmin(num_arr, axis=0, keepdims=True)
    _col_max = np.nanmax(num_arr, axis=0, keepdims=True)
    num_norm = (num_arr - _col_min) / (_col_max - _col_min + 1e-9)
    num_norm = np.nan_to_num(num_norm, nan=0.0, posinf=1.0, neginf=0.0)

    total = len(df) - W_max - F
    for i, idx in enumerate(range(W_max, len(df) - F)):
        if not valid_all[idx]: continue
        if i % 100 == 0: print(f"  {ticker}: рендер v3 {i}/{total}", end="\r")
        for W in SCALES:
            img_hwc = render_candles(df.iloc[idx - W : idx])
            img_cw  = _hwc_to_cw(img_hwc)
            # FIX v3.2: защита от nan в рендере свечей
            if not np.isfinite(img_cw).all():
                img_cw = np.nan_to_num(img_cw, nan=0.0, posinf=0.0, neginf=0.0)
            scale_imgs[W].append(img_cw)
            scale_nums[W].append(num_norm[idx - W : idx])
        if context is not None: ctx_list.append(context[idx])
        y_cls_list.append(cls_all[idx]); y_ohlc_list.append(ohlc_all[idx])
        valid_daily_indices.append(idx)
    print()

    imgs   = {W: np.array(scale_imgs[W], dtype=np.float32) for W in SCALES}
    nums   = {W: np.array(scale_nums[W], dtype=np.float32) for W in SCALES}
    cls    = np.array(y_cls_list,  dtype=np.int64)
    ohlc   = np.array(y_ohlc_list, dtype=np.float32)
    ctx    = np.array(ctx_list,    dtype=np.float32) if ctx_list else None
    hourly = None
    if hourly_df is not None and len(valid_daily_indices) > 0:
        print(f"  {ticker}: рендер часовых свечей...")
        hourly = _build_hourly_windows(hourly_df, df, valid_daily_indices)

    imgs, nums, cls, ohlc, ctx, hourly = _align_arrays(imgs, nums, cls, ohlc, ctx, hourly)

    for W in SCALES:
        np.save(_img_path(ticker, W), imgs[W])
        np.save(_num_path(ticker, W), nums[W])
    np.save(_cls_path(ticker),  cls)
    np.save(_ohlc_path(ticker), ohlc)
    if hourly is not None: np.save(_hourly_path(ticker), hourly)

    return imgs, nums, cls, ohlc, ctx, hourly


# ══════════════════════════════════════════════════════════════════
# Точка входа — все тикеры
# ══════════════════════════════════════════════════════════════════

def build_full_multiscale_dataset_v3(
    force_rebuild: bool = False, use_hourly: bool = True, ticker_filter: Optional[list[str]] = None,
):
    from api.routes.candles import get_client
    from ml.cache_manager import (
        _load_meta, _save_meta,
        ticker_cache_valid, probe_freshness, update_meta,
    )
    client = get_client()
    imoex  = load_imoex()
    meta   = _load_meta()

    all_cached = all(ticker_cache_valid(t, meta) for t in CFG.tickers)

    if all_cached and not force_rebuild:
        print("  Все тикеры в кэше. Запускаем probe-проверку актуальности...")
        cache_fresh = probe_freshness(client, CFG.tickers, meta)
        if cache_fresh:
            print("  ✓ Кэш актуален — пропускаем скачивание, загружаем из файлов")
            return _load_all_from_cache(meta, use_hourly=use_hourly)
        else:
            print("  Кэш устарел — обновляем только изменившиеся тикеры")

    records = []; all_cls = []; ctx_dim = None; ticker_lengths = []

    tickers_to_use = (
        CFG.tickers
        if ticker_filter is None
        else [t for t in CFG.tickers if t in ticker_filter]
    )
    if not tickers_to_use:
        raise RuntimeError(f"ticker_filter={ticker_filter} не совпал ни с одним из CFG.tickers")

    for ticker in tickers_to_use:
        if not force_rebuild and ticker_cache_valid(ticker, meta):
            print(f"  {ticker}: кэш актуален, пропускаем скачивание ✓")
            cls = np.load(_cls_path(ticker), mmap_mode="r")

            # FIX v3.2: санируем ohlc при загрузке из кэша
            ohlc_p = _ohlc_path(ticker)
            if os.path.exists(ohlc_p):
                ohlc_cached = np.load(ohlc_p, mmap_mode="r")
                nan_cnt = (~np.isfinite(ohlc_cached)).sum()
                if nan_cnt > 0:
                    print(f"  [FIX] {ticker}: кэш ohlc содержит {nan_cnt} nan/inf → перезаписываем")
                    ohlc_fixed = np.nan_to_num(np.array(ohlc_cached),
                                               nan=0.0, posinf=0.0, neginf=0.0)
                    np.save(ohlc_p, ohlc_fixed)

            n = len(cls)
            for local_idx in range(n):
                records.append((ticker, local_idx))
            all_cls.append(cls)
            ticker_lengths.append((ticker, n))
            cp = _ctx_path(ticker)
            if os.path.exists(cp) and ctx_dim is None:
                ctx_dim = np.load(cp, mmap_mode="r").shape[1]
            continue

        print(f"  Загружаем {ticker} из API...")
        try:
            figi = client.find_figi(ticker)
            if not figi: continue

            df = _load_daily_candles_chunked(client, figi, days_back=CFG.days_back)
            if df is None or df.empty:
                continue

            hourly_df = _load_hourly_candles(client, figi) if use_hourly else None

            ctx_series = load_context_series(ticker)
            ctx_feats  = build_context_features(
                ctx_series, df.index, ticker=ticker,
                imoex_close=imoex["close"] if imoex is not None else None,
            ) if ctx_series is not None else None

            imgs, nums, cls, ohlc, ctx, hourly = build_multiscale_dataset_v3(
                df, imoex, ctx_feats, ticker=ticker,
                hourly_df=hourly_df, force_rebuild=force_rebuild,
            )
            if len(cls) == 0: continue

            # ↓ ДОБАВИТЬ: сохраняем ctx на диск
            if ctx is not None and len(ctx) > 0:
                np.save(_ctx_path(ticker), ctx)
                print(f"  {ticker}: ctx сохранён, shape={ctx.shape}")

            update_meta(ticker, df, len(cls), meta)
            _save_meta(meta)

            cp = _ctx_path(ticker)
            from ml.context_loader import get_context_dim
            ctx_dim = get_context_dim(ticker)

            for local_idx in range(len(cls)):
                records.append((ticker, local_idx))
            all_cls.append(cls)
            ticker_lengths.append((ticker, len(cls)))
            print(f"  {ticker}: {len(cls)} сэмплов ✓")

        except Exception as e:
            print(f"  {ticker}: ошибка — {e}"); traceback.print_exc()

    if not records: raise RuntimeError("Не удалось загрузить данные.")
    ctx_dim = ctx_dim or 0
    y_all   = np.concatenate(all_cls)
    dataset = LazyMultiScaleDatasetV3(records, ctx_dim, use_hourly=use_hourly)
    return dataset, y_all, ctx_dim, ticker_lengths


def _load_all_from_cache(meta: dict, use_hourly: bool = True):
    """Мгновенная загрузка когда весь кэш актуален — без API."""
    records = []; all_cls = []; ctx_dim = None; ticker_lengths = []
    for ticker in CFG.tickers:
        cp  = _cls_path(ticker)
        if not os.path.exists(cp): continue
        cls = np.load(cp, mmap_mode="r")
        n   = len(cls)
        for local_idx in range(n):
            records.append((ticker, local_idx))
        all_cls.append(cls)
        ticker_lengths.append((ticker, n))
        ctx_p = _ctx_path(ticker)
        if os.path.exists(ctx_p) and ctx_dim is None:
            ctx_dim = np.load(ctx_p, mmap_mode="r").shape[1]
    ctx_dim = ctx_dim or 0
    y_all   = np.concatenate(all_cls)
    dataset = LazyMultiScaleDatasetV3(records, ctx_dim, use_hourly=use_hourly)
    print(f"  ✓ Загружено из кэша: {len(records)} сэмплов, "
          f"{len(ticker_lengths)} тикеров")
    return dataset, y_all, ctx_dim, ticker_lengths


# ══════════════════════════════════════════════════════════════════
# Temporal split с purge gap
# ══════════════════════════════════════════════════════════════════

def temporal_split(
    ticker_lengths: list,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    purge_bars:  int   = None,
) -> tuple:
    if purge_bars is None: purge_bars = CFG.future_bars
    idx_train = []; idx_val = []; idx_test = []
    global_offset = 0

    for ticker, n in ticker_lengths:
        n_test     = max(int(n * test_ratio), 1)
        n_val      = max(int(n * val_ratio),  1)
        test_start = n - n_test
        val_end    = test_start - purge_bars
        val_start  = val_end - n_val
        train_end  = val_start - purge_bars

        if train_end < 10:
            for i in range(n): idx_train.append(global_offset + i)
            global_offset += n; continue

        for i in range(0, train_end):                        idx_train.append(global_offset + i)
        for i in range(max(val_start, 0), max(val_end, 0)): idx_val.append(global_offset + i)
        for i in range(test_start, n):                       idx_test.append(global_offset + i)
        global_offset += n

    return (np.array(idx_train, dtype=np.int64),
            np.array(idx_val,   dtype=np.int64),
            np.array(idx_test,  dtype=np.int64))