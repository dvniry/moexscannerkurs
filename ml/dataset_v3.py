# ml/dataset_v3.py
"""Мультимасштабный датасет v3 — самодостаточный (без зависимости от dataset.py).

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
    """Конвертирует рендер (H, W, C) → (C, W) mean-pool по оси H.

    FactualBackbone использует Conv1d: ожидает (batch, C, T).
    img: float32 (H, W, C) в диапазоне [0, 1]
    return: float32 (C, W)
    """
    # img shape: (H, W, C) → mean по H → (W, C) → transpose → (C, W)
    return img.mean(axis=0).T.astype(np.float32)   # (C=3, W=img_width)


# ══════════════════════════════════════════════════════════════════
# Индикаторы (перенесены из dataset.py)
# ══════════════════════════════════════════════════════════════════

# 30 признаков: тренд + осцилляторы + Bollinger + ATR + объём +
# Фибоначчи + RS vs IMOEX + календарь
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
    loss_s = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]     = 100 - (100 / (1 + gain / loss_s.replace(0, 1e-9)))
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
    df["range_norm"]      = (h - l) / (c + 1e-9)
    df["range_atr_ratio"] = (h - l) / (df["atr"] + 1e-9)

    df["vol_ma"]         = v.rolling(20).mean()
    df["vol_ratio"]      = v / (df["vol_ma"] + 1e-9)
    df["ret1"]           = c.pct_change()
    df["sent_vol_price"] = df["ret1"] * df["vol_ratio"]

    df["roc5"]  = c.pct_change(5)
    df["roc10"] = c.pct_change(10)

    low14       = l.rolling(14).min()
    high14      = h.rolling(14).max()
    df["stoch"] = (c - low14) / (high14 - low14 + 1e-9) * 100

    roll_max           = c.rolling(30).max()
    roll_min           = c.rolling(30).min()
    fib_range          = roll_max - roll_min
    df["dist_fib_382"] = (c - (roll_max - 0.382 * fib_range)) / (fib_range + 1e-9)
    df["dist_fib_618"] = (c - (roll_max - 0.618 * fib_range)) / (fib_range + 1e-9)

    if imoex is not None:
        idx_c             = imoex["close"].astype(float).reindex(df.index).ffill()
        df["rs_5d"]       = c.pct_change(5)  / (idx_c.pct_change(5).abs()  + 1e-9)
        df["rs_20d"]      = c.pct_change(20) / (idx_c.pct_change(20).abs() + 1e-9)
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


def load_imoex() -> "pd.DataFrame | None":
    from api.routes.candles import get_client
    client = get_client()
    try:
        uid = client.find_indicative_uid("IMOEX")
        if uid:
            df = client.get_candles_by_uid(uid=uid, interval=CFG.interval, days_back=CFG.days_back)
            if df is not None and not df.empty:
                print(f"  IMOEX загружен: {len(df)} свечей")
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
    return render_hourly_candles(day_h, d_close, d_high, d_low)


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

        # imgs[W] хранится как (N, C=3, img_width) — уже готово для Conv1d
        imgs = {W: torch.tensor(data[W][local_idx]).float() for W in SCALES}
        nums = ({W: torch.tensor(data[f"num_{W}"][local_idx]).float() for W in SCALES}
                if data[f"num_{SCALES[0]}"] is not None else None)
        cls_y  = int(data["cls"][local_idx])
        ohlc_y = torch.tensor(data["ohlc"][local_idx]).float()
        ctx_arr = data["ctx"]
        ctx_idx = min(local_idx, len(ctx_arr) - 1) if ctx_arr is not None else 0
        ctx = (torch.tensor(ctx_arr[ctx_idx]).float() if ctx_arr is not None
               else torch.zeros(self.ctx_dim))
        if self.use_hourly and data["hourly"] is not None:
            hourly = torch.tensor(data["hourly"][local_idx]).float()
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
    """Возвращает: (imgs, nums, cls_labels, ohlc_labels, ctx, hourly)

    imgs[W]: shape (N, C=3, img_width)  ← Conv1d-совместимый формат
    nums[W]: shape (N, window=W, n_indicators=30)
    """
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
            if imgs[W].ndim == 4:  # старый формат (N, H, img_W, C)
                print(f"  [migrate] {ticker} W={W}: (N,H,W,C) → (N,C,W)")
                imgs[W] = np.stack([_hwc_to_cw(imgs[W][i]) for i in range(len(imgs[W]))])
                np.save(_img_path(ticker, W), imgs[W])
        # ────────────────────────────────────────────────────────────

        ctx_list = []
        if context is not None:
            for i in range(W_max, len(df) - F): ctx_list.append(context[i])
        ctx = np.array(ctx_list, dtype=np.float32) if ctx_list else None
        imgs, nums, cls, ohlc, ctx, hourly = _align_arrays(imgs, nums, cls, ohlc, ctx, hourly)
        return imgs, nums, cls, ohlc, ctx, hourly

    ohlc_all, cls_all, valid_all = build_ohlc_labels(df)

    scale_imgs = {W: [] for W in SCALES}
    scale_nums = {W: [] for W in SCALES}
    ctx_list = []; y_cls_list = []; y_ohlc_list = []; valid_daily_indices = []

    num_arr  = df[INDICATOR_COLS].values.astype(np.float32)
    num_norm = (num_arr - num_arr.min(0, keepdims=True)) / \
               (num_arr.max(0, keepdims=True) - num_arr.min(0, keepdims=True) + 1e-9)

    total = len(df) - W_max - F
    for i, idx in enumerate(range(W_max, len(df) - F)):
        if not valid_all[idx]: continue
        if i % 100 == 0: print(f"  {ticker}: рендер v3 {i}/{total}", end="\r")
        for W in SCALES:
            # render_candles → (H, W_img, C=3) → _hwc_to_cw → (C=3, W_img)
            img_hwc = render_candles(df.iloc[idx - W : idx])   # (H, W_img, 3)
            scale_imgs[W].append(_hwc_to_cw(img_hwc))          # (3, W_img)
            scale_nums[W].append(num_norm[idx - W : idx])      # (W, n_ind)
        if context is not None: ctx_list.append(context[idx])
        y_cls_list.append(cls_all[idx]); y_ohlc_list.append(ohlc_all[idx])
        valid_daily_indices.append(idx)
    print()

    # imgs[W]: (N, C=3, img_width)   ← Conv1d ready
    # nums[W]: (N, window, n_indicators)
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
    force_rebuild: bool = False, use_hourly: bool = True
):
    """Загружает все тикеры, строит LazyMultiScaleDatasetV3.

    Returns: (dataset, y_all, ctx_dim, ticker_lengths)
    """
    from api.routes.candles import get_client
    client = get_client()

    print("  Загружаем IMOEX...")
    imoex = load_imoex()

    records = []; all_cls = []; ctx_dim = None; ticker_lengths = []

    for ticker in CFG.tickers:
        print(f"  Загружаем {ticker}...")
        try:
            figi = client.find_figi(ticker)
            if not figi: continue
            df = client.get_candles(figi=figi, interval=CFG.interval, days_back=CFG.days_back)
            if df is None or df.empty: continue

            hourly_df = _load_hourly_candles(client, figi) if use_hourly else None

            print(f"  Контекст для {ticker}...")
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

            ctx_p = _ctx_path(ticker)
            if ctx is not None:
                np.save(ctx_p, ctx)
                if ctx_dim is None: ctx_dim = ctx.shape[1]
            elif ctx_dim is not None:
                np.save(ctx_p, np.zeros((len(cls), ctx_dim), dtype=np.float32))

            for local_idx in range(len(cls)):
                records.append((ticker, local_idx))
            all_cls.append(cls); ticker_lengths.append((ticker, len(cls)))
            print(f"  {ticker}: {len(cls)} сэмплов")
        except Exception as e:
            print(f"  {ticker}: ошибка — {e}"); traceback.print_exc()

    if not records: raise RuntimeError("Не удалось загрузить данные.")
    ctx_dim = ctx_dim or 0
    y_all   = np.concatenate(all_cls)
    dataset = LazyMultiScaleDatasetV3(records, ctx_dim, use_hourly=use_hourly)
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
    """Temporal split с purge gap — для каждого тикера отдельно.

    Возвращает: (idx_train, idx_val, idx_test) — глобальные индексы.
    """
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
