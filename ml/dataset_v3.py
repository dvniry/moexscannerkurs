# ml/dataset_v3.py
"""Мультимасштабный датасет v3.4.1 — fix DWT look-ahead leak.

Изменения v3.4.1 (патч поверх v3.4):
- [LEAK FIX] add_indicators() больше НЕ применяет DWT к полной серии close.
  Ранее pywt.wavedec на всей серии → значение в точке t зависело от t+k.
  Теперь c = df["close"] напрямую. Если нужен шумоподавитель —
  применяйте WaveletDenoise как слой модели (там окно фиксировано).
- CACHE_VERSION остаётся "v3.4" (структура не менялась), но индикаторы
  пересчитаются → требуется один --rebuild после обновления.
"""
import os
os.environ["GRPC_DNS_RESOLVER"] = "native"

import sys, traceback
_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "russian_ca.cer"))
if os.path.exists(_cert):
    os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = _cert

import numpy as np
import pandas as pd
from typing import Optional
import torch
from torch.utils.data import Dataset as TorchDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.config import CFG, SCALES
from ml.candle_render_v2 import (
    render_candles as _render_candles_orig,
    N_RENDER_CHANNELS as _N_RENDER_CHANNELS_ORIG,
)
from ml.labels_ohlc import build_ohlc_labels
from ml.context_loader import load_context_series, build_context_features
from ml.hourly_encoder import (
    render_hourly_candles, N_HOURLY_CHANNELS,
    N_HOURS_PER_DAY, N_INTRADAY_DAYS,
)

# ══════════════════════════════════════════════════════════════════
# [3.4] Heikin-Ashi — 4-й канал свечного рендера
# ══════════════════════════════════════════════════════════════════

CACHE_VERSION = "v3.4"
N_RENDER_CHANNELS = _N_RENDER_CHANNELS_ORIG + 1   # 3 → 4


def render_candles(df_window) -> np.ndarray:
    """Обёртка над candle_render_v2: добавляет 4-й канал — нормализованный HA-close."""
    orig = _render_candles_orig(df_window)

    o = df_window["open"].values.astype(np.float64)
    h = df_window["high"].values.astype(np.float64)
    l = df_window["low"].values.astype(np.float64)
    c = df_window["close"].values.astype(np.float64)

    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty(len(o), dtype=np.float64)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(o)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    price_min = l.min(); price_max = h.max()
    rng = max(price_max - price_min, 1e-8)
    ha_norm = ((ha_close - price_min) / rng).astype(np.float32)

    H_dim, W_dim, _ = orig.shape
    x_src = np.linspace(0, 1, len(ha_norm))
    x_dst = np.linspace(0, 1, W_dim)
    ha_col = np.interp(x_dst, x_src, ha_norm).astype(np.float32)
    ha_channel = np.tile(ha_col[np.newaxis, :, np.newaxis], (H_dim, 1, 1))

    return np.concatenate([orig, ha_channel], axis=2)


def _hwc_to_cw(img: np.ndarray) -> np.ndarray:
    return img.mean(axis=0).T.astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# v3.4.1 — wavelet_denoise оставлен как утилита, НЕ применяется в add_indicators
# ══════════════════════════════════════════════════════════════════

def wavelet_denoise(series: np.ndarray,
                    wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """DWT денойзинг с MAD-порогом. НЕ-каузальный — не использовать для фичей!
    Оставлен для возможного применения к историческим исследованиям.
    """
    try:
        import pywt
        arr = np.asarray(series, dtype=np.float64)
        if len(arr) < 2 ** (level + 1):
            return arr.astype(np.float32)
        coeffs = pywt.wavedec(arr, wavelet, level=level)
        sigma  = np.median(np.abs(coeffs[-1])) / 0.6745
        thr    = sigma * np.sqrt(2 * np.log(max(len(arr), 2)))
        ct     = [pywt.threshold(c, thr, mode="soft") for c in coeffs]
        out    = pywt.waverec(ct, wavelet)[: len(arr)]
        return np.where(np.isfinite(out), out, arr).astype(np.float32)
    except Exception:
        return np.asarray(series, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
# Индикаторы (37 признаков) — v3.4.1 БЕЗ DWT LEAK
# ══════════════════════════════════════════════════════════════════

INDICATOR_COLS = [
    "ema9", "ema21", "ema50", "macd", "macd_sig",
    "rsi", "rsi_pct", "stoch",
    "bb_upper", "bb_lower", "bb_pct",
    "atr", "range_norm", "range_atr_ratio",
    "vol_ratio", "sent_vol_price",
    "roc5", "roc10",
    "dist_fib_382", "dist_fib_618",
    "close_rel",
    "rs_5d", "rs_20d", "imoex_ret5", "imoex_ret20", "imoex_vol20",
    "day_of_week", "month", "is_monday", "is_friday",
    "volume_imbalance",
    "overnight_gap",
    "williams_r",
    "cci",
    "rolling_skew5",
    "spread_hl_norm",
    "week_number",
]


def add_indicators(df: pd.DataFrame, imoex: pd.DataFrame = None) -> pd.DataFrame:
    # ── v3.4.1 FIX: БЕЗ DWT LEAK ──
    # Раньше: denoised = wavelet_denoise(df["close"]) — НЕ каузально!
    # Теперь: работаем с сырым close. Все индикаторы — каузальные по построению.
    c = df["close"].astype(float)

    h = df["high"].astype(float)
    l = df["low"].astype(float)
    o = df["open"].astype(float)
    v = df["volume"].astype(float)
    c_safe = c.clip(lower=1e-9)

    # ── Trend ──
    ema9_abs  = c.ewm(span=9,  adjust=False).mean()
    ema21_abs = c.ewm(span=21, adjust=False).mean()
    ema50_abs = c.ewm(span=50, adjust=False).mean()
    df["ema9"]  = (ema9_abs  - c) / c_safe
    df["ema21"] = (ema21_abs - c) / c_safe
    df["ema50"] = (ema50_abs - c) / c_safe
    macd_raw    = (c.ewm(span=12, adjust=False).mean()
                 - c.ewm(span=26, adjust=False).mean())
    df["macd"]     = macd_raw / c_safe
    df["macd_sig"] = macd_raw.ewm(span=9, adjust=False).mean() / c_safe

    # ── Momentum ──
    delta  = c.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss_s = (-delta.clip(upper=0)).rolling(14).mean().clip(lower=1e-9)
    rsi_raw    = 100 - (100 / (1 + gain / loss_s))
    df["rsi"]  = rsi_raw / 100.0
    rsi_min    = rsi_raw.rolling(50).min()
    rsi_max    = rsi_raw.rolling(50).max()
    df["rsi_pct"] = (rsi_raw - rsi_min) / (rsi_max - rsi_min + 1e-9)

    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["stoch"]      = (c - low14) / (high14 - low14 + 1e-9)
    df["williams_r"] = (high14 - c) / (high14 - low14 + 1e-9) * (-100)

    # ── Bollinger ──
    mid = c.rolling(20).mean()
    std = c.rolling(20).std().clip(lower=1e-9)
    bb_upper_abs = mid + 2 * std
    bb_lower_abs = mid - 2 * std
    df["bb_upper"] = (bb_upper_abs - c) / c_safe
    df["bb_lower"] = (c - bb_lower_abs) / c_safe
    df["bb_pct"]   = (c - bb_lower_abs) / (bb_upper_abs - bb_lower_abs + 1e-9)

    # ── CCI ──
    sma20    = c.rolling(20).mean()
    mean_dev = (c - sma20).abs().rolling(20).mean().clip(lower=1e-9)
    df["cci"] = (c - sma20) / (0.015 * mean_dev)

    # ── Volatility ──
    tr = pd.concat([h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_abs = tr.rolling(14).mean()
    df["atr"]              = atr_abs / c_safe
    df["range_norm"]       = (h - l) / c_safe
    df["range_atr_ratio"]  = (h - l) / atr_abs.clip(lower=1e-9)

    roll5_std = c.rolling(5).std().clip(lower=1e-9)
    df["spread_hl_norm"] = (h - l) / roll5_std

    # ── Volume ──
    vol_ma = v.rolling(20).mean()
    df["vol_ma"]         = vol_ma
    df["vol_ratio"]      = (v / vol_ma.clip(lower=1e-9)).clip(0, 10)
    df["ret1"]           = c.pct_change()
    df["sent_vol_price"] = df["ret1"] * df["vol_ratio"]

    df["volume_imbalance"] = ((c - o) / (h - l + 1e-9)) * df["vol_ratio"]
    df["overnight_gap"]    = (o / c.shift(1).clip(lower=1e-9)) - 1.0

    # ── ROC ──
    df["roc5"]  = c.pct_change(5)
    df["roc10"] = c.pct_change(10)

    df["rolling_skew5"] = c.pct_change().rolling(5).skew().fillna(0.0)

    # ── Fibonacci ──
    roll_max  = c.rolling(30).max()
    roll_min  = c.rolling(30).min()
    fib_range = (roll_max - roll_min).clip(lower=1e-9)
    df["dist_fib_382"] = (c - (roll_max - 0.382 * fib_range)) / fib_range
    df["dist_fib_618"] = (c - (roll_max - 0.618 * fib_range)) / fib_range

    df["close_rel"] = c.pct_change(1).fillna(0.0)

    # ── RS + IMOEX ──
    if imoex is not None:
        idx_c = imoex["close"].astype(float).reindex(df.index).ffill()
        df["rs_5d"]       = (c.pct_change(5)  /
                             idx_c.pct_change(5).abs().clip(lower=1e-9)).clip(-10, 10)
        df["rs_20d"]      = (c.pct_change(20) /
                             idx_c.pct_change(20).abs().clip(lower=1e-9)).clip(-10, 10)
        df["imoex_ret5"]  = idx_c.pct_change(5)
        df["imoex_ret20"] = idx_c.pct_change(20)
        df["imoex_vol20"] = idx_c.pct_change().rolling(20).std()
    else:
        for col in ("rs_5d", "rs_20d", "imoex_ret5", "imoex_ret20", "imoex_vol20"):
            df[col] = 0.0

    # ── Calendar ──
    if hasattr(df.index, "dayofweek"):
        df["day_of_week"] = df.index.dayofweek / 4.0
        df["month"]       = df.index.month / 12.0
        df["is_monday"]   = (df.index.dayofweek == 0).astype(float)
        df["is_friday"]   = (df.index.dayofweek == 4).astype(float)
        try:
            df["week_number"] = df.index.isocalendar().week.astype(float).values / 53.0
        except Exception:
            df["week_number"] = 0.5
    else:
        df["day_of_week"] = df["month"] = 0.5
        df["is_monday"]   = df["is_friday"] = 0.0
        df["week_number"] = 0.5

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
        all_frames = []; end = _now(); remaining = CFG.days_back; chunk_num = 0

        while remaining > 0:
            chunk = min(remaining, CHUNK_DAYS); start = end - timedelta(days=chunk)
            try:
                with Client(client.token, target=TARGET) as api:
                    candles = api.market_data.get_candles(
                        instrument_id=uid, from_=start, to=end,
                        interval=CandleInterval.CANDLE_INTERVAL_DAY,
                    ).candles
                if candles:
                    df_chunk = pd.DataFrame({
                        "time":   [cdl.time for cdl in candles],
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
            end = start; remaining -= chunk; _time.sleep(0.1)

        if not all_frames:
            raise ValueError("Ни один чанк IMOEX не загружен")

        df = pd.concat(all_frames).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        print(f"  IMOEX загружен: {len(df)} свечей ({chunk_num} чанков)")
        return df
    except Exception as e:
        print(f"  [WARN] IMOEX: {e} → RS признаки будут нулями")
        return None


def class_distribution(y: np.ndarray) -> None:
    total = len(y)
    for label, name in {0: "BUY", 1: "HOLD", 2: "SELL"}.items():
        count = int((y == label).sum())
        print(f"  {name:4s}: {count:5d} ({count / total * 100:.1f}%)")


def _align_arrays(imgs, nums, cls, ohlc, ctx=None, hourly=None, aux=None):
    n = min(len(cls), len(ohlc))
    for W in SCALES:
        n = min(n, len(imgs[W]), len(nums[W]))
    if ctx    is not None: n = min(n, len(ctx))
    if hourly is not None: n = min(n, len(hourly))
    if aux    is not None: n = min(n, len(aux))
    cls  = cls[:n]; ohlc = ohlc[:n]
    imgs = {W: imgs[W][:n] for W in SCALES}
    nums = {W: nums[W][:n] for W in SCALES}
    if ctx    is not None: ctx    = ctx[:n]
    if hourly is not None: hourly = hourly[:n]
    if aux    is not None: aux    = aux[:n]
    return imgs, nums, cls, ohlc, ctx, hourly, aux


def _cache_dir():
    d = "ml/cache_v3"; os.makedirs(d, exist_ok=True); return d

def _img_path(ticker, W): return f"{_cache_dir()}/imgs_{ticker}_{W}.npy"
def _num_path(ticker, W): return f"{_cache_dir()}/nums_{ticker}_{W}.npy"
def _cls_path(ticker):    return f"{_cache_dir()}/cls_{ticker}.npy"
def _ohlc_path(ticker):   return f"{_cache_dir()}/ohlc_{ticker}.npy"
def _ctx_path(ticker):    return f"{_cache_dir()}/ctx_{ticker}.npy"
def _hourly_path(ticker): return f"{_cache_dir()}/hourly_{ticker}.npy"
def _aux_path(ticker):    return f"{_cache_dir()}/aux_{ticker}.npy"


# ──────────────────────────────────────────────────────────────────
# Часовые свечи
# ──────────────────────────────────────────────────────────────────

def _load_hourly_candles(client, figi: str, days_back: int = None):
    import time as _time
    from datetime import timedelta
    from t_tech.invest import Client, CandleInterval
    from t_tech.invest.utils import now as _now

    if days_back is None: days_back = CFG.days_back
    CHUNK_DAYS = 85; TARGET = "invest-public-api.tbank.ru:443"
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
    return result[~result.index.duplicated(keep="first")]


def _build_hourly_for_day(hourly_by_date, daily_date, d_high, d_low, d_close):
    if hourly_by_date is None:
        return np.zeros((N_HOURLY_CHANNELS, N_HOURS_PER_DAY), dtype=np.float32)
    day = daily_date.date() if hasattr(daily_date, "date") else pd.Timestamp(daily_date).date()
    day_h = hourly_by_date.get(day)
    if day_h is None or len(day_h) == 0:
        return np.zeros((N_HOURLY_CHANNELS, N_HOURS_PER_DAY), dtype=np.float32)
    result = render_hourly_candles(day_h, d_close, d_high, d_low)
    if not np.isfinite(result).all():
        result = np.nan_to_num(result, nan=0., posinf=0., neginf=0.)
    return result


def _build_hourly_windows(hourly_df, daily_df, valid_indices):
    hourly_by_date = None
    if hourly_df is not None and not hourly_df.empty:
        hourly_by_date = {d: g for d, g in hourly_df.groupby(hourly_df.index.date)}
    results = []
    for i, idx in enumerate(valid_indices):
        if i % 500 == 0 and i: print(f"  hourly render {i}/{len(valid_indices)}", end="\r")
        start_day = max(0, idx - N_INTRADAY_DAYS + 1)
        days_win  = daily_df.iloc[start_day: idx + 1]
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
    import time as _time
    from datetime import timedelta
    from t_tech.invest import Client, CandleInterval
    from t_tech.invest.utils import now as _now

    if days_back is None: days_back = CFG.days_back
    CHUNK_DAYS = 365; TARGET = "invest-public-api.tbank.ru:443"
    all_frames = []; end = _now(); remaining = days_back; chunk_num = 0

    while remaining > 0:
        chunk = min(remaining, CHUNK_DAYS); start = end - timedelta(days=chunk)
        try:
            with Client(client.token, target=TARGET) as api:
                candles = api.market_data.get_candles(
                    figi=figi, from_=start, to=end,
                    interval=CandleInterval.CANDLE_INTERVAL_DAY,
                ).candles
            if candles:
                df_chunk = pd.DataFrame({
                    "time":   [cdl.time for cdl in candles],
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
        end = start; remaining -= chunk; _time.sleep(0.1)

    if not all_frames:
        print(f"  [WARN] Дневные свечи: ни один чанк не загружен")
        return pd.DataFrame()

    result = pd.concat(all_frames).sort_index()
    result = result[~result.index.duplicated(keep='first')]
    print(f"  Дневные свечи: {len(result)} ({chunk_num} чанков, {days_back} дней)")
    return result


# ──────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────

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
        data["ctx"] = np.load(cp, mmap_mode="r") if os.path.exists(cp) else None
        hp = _hourly_path(ticker)
        data["hourly"] = np.load(hp, mmap_mode="r") if (self.use_hourly and os.path.exists(hp)) else None
        ap = _aux_path(ticker)
        data["aux"] = np.load(ap, mmap_mode="r") if os.path.exists(ap) else None

        n = min(len(data["cls"]), len(data["ohlc"]))
        for W in SCALES:
            n = min(n, len(data[W]))
            if data[f"num_{W}"] is not None: n = min(n, len(data[f"num_{W}"]))
        if data["ctx"]    is not None: n = min(n, len(data["ctx"]))
        if data["hourly"] is not None: n = min(n, len(data["hourly"]))
        if data["aux"]    is not None: n = min(n, len(data["aux"]))
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
        cls_y = int(data["cls"][local_idx])

        ohlc_raw = data["ohlc"][local_idx]
        if not np.isfinite(ohlc_raw).all():
            ohlc_raw = np.nan_to_num(ohlc_raw, nan=0., posinf=0., neginf=0.)
        ohlc_y = torch.tensor(ohlc_raw).float()

        for W in SCALES:
            if not torch.isfinite(imgs[W]).all():
                imgs[W] = torch.nan_to_num(imgs[W], nan=0., posinf=0., neginf=0.)
        if nums is not None:
            for W in SCALES:
                if not torch.isfinite(nums[W]).all():
                    nums[W] = torch.nan_to_num(nums[W], nan=0., posinf=1., neginf=0.)

        ctx_arr = data["ctx"]
        if ctx_arr is not None and len(ctx_arr) > 0:
            ctx_idx = min(local_idx, len(ctx_arr) - 1)
            ctx = torch.tensor(ctx_arr[ctx_idx]).float()
            if not torch.isfinite(ctx).all():
                ctx = torch.nan_to_num(ctx, nan=0., posinf=0., neginf=0.)
        else:
            ctx = torch.zeros(self.ctx_dim if self.ctx_dim > 0 else 1)

        if self.use_hourly and data["hourly"] is not None:
            hourly = torch.tensor(data["hourly"][local_idx]).float()
            if not torch.isfinite(hourly).all():
                hourly = torch.nan_to_num(hourly, nan=0., posinf=0., neginf=0.)
        else:
            hourly = torch.zeros(N_INTRADAY_DAYS, N_HOURLY_CHANNELS, N_HOURS_PER_DAY)

        if data["aux"] is not None:
            aux_raw = data["aux"][local_idx]
            # aux_raw может быть [vol, skew] (старый кэш) или [vol, skew, atr] (новый)
            aux_y = torch.tensor([
                float(np.clip(aux_raw[0] * 100.0, 0., 10.)),
                float(np.clip(aux_raw[1], -3., 3.)),
            ]).float()
            # atr_ratio хранится отдельно для бэктеста
            # но не передаём в модель (она его не использует)
        else:
            aux_y = torch.zeros(2)

        return imgs, nums, cls_y, ohlc_y, ctx, hourly, aux_y


# ══════════════════════════════════════════════════════════════════
# ATR-адаптивная разметка
# ══════════════════════════════════════════════════════════════════

def build_labels_atr(df: pd.DataFrame,
                     future_bars: int = None,
                     atr_k: float = None):
    if future_bars is None:
        future_bars = CFG.future_bars
    if atr_k is None:
        atr_k = CFG.label_atr_k

    ohlc_all, _, valid_all, extra = build_ohlc_labels(df)

    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)

    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    )
    tr = np.concatenate([[tr[0] if len(tr) else 0.0], tr])

    atr14 = np.zeros(len(c))
    if len(c) >= 14:
        atr14[13] = tr[:14].mean()
        for i in range(14, len(c)):
            atr14[i] = (atr14[i - 1] * 13.0 + tr[i]) / 14.0
        atr14[:13] = atr14[13]

    cls_all = np.ones(len(c), dtype=np.int64)

    for i in range(len(c) - future_bars):
        if not valid_all[i]:
            continue
        c_now = max(c[i], 1e-9)
        thresh = atr_k * atr14[i] / c_now
        if thresh < 1e-5:
            continue
        ret = (c[i + future_bars] - c_now) / c_now
        if ret > thresh:
            cls_all[i] = 0
        elif ret < -thresh:
            cls_all[i] = 2

    return ohlc_all, cls_all, valid_all, extra


# ══════════════════════════════════════════════════════════════════
# Построение для одного тикера
# ══════════════════════════════════════════════════════════════════

def build_multiscale_dataset_v3(
    df, imoex=None, context=None, ticker: str = "unknown",
    hourly_df: pd.DataFrame = None, force_rebuild: bool = False,
):
    from sklearn.preprocessing import RobustScaler

    df    = add_indicators(df.copy(), imoex).dropna()
    W_max = max(SCALES); F = CFG.future_bars

    def _cache_channels_valid(ticker: str) -> bool:
        p = _img_path(ticker, SCALES[0])
        if not os.path.exists(p):
            return False
        try:
            arr = np.load(p, mmap_mode="r")
            return arr.ndim == 3 and arr.shape[1] == N_RENDER_CHANNELS
        except Exception:
            return False

    cache_ok = (
        _cache_channels_valid(ticker) and
        all(os.path.exists(_img_path(ticker, W)) for W in SCALES) and
        all(os.path.exists(_num_path(ticker, W)) for W in SCALES) and
        os.path.exists(_cls_path(ticker)) and
        os.path.exists(_ohlc_path(ticker)) and
        os.path.exists(_hourly_path(ticker)) and
        os.path.exists(_aux_path(ticker))
    )

    if cache_ok and not force_rebuild:
        print(f"  Кэш v3.4 найден для {ticker}")
        imgs   = {W: np.load(_img_path(ticker, W)) for W in SCALES}
        nums   = {W: np.load(_num_path(ticker, W)) for W in SCALES}
        cls    = np.load(_cls_path(ticker))
        ohlc   = np.load(_ohlc_path(ticker))
        hourly = np.load(_hourly_path(ticker)) if os.path.exists(_hourly_path(ticker)) else None
        aux    = np.load(_aux_path(ticker))

        for W in SCALES:
            if imgs[W].ndim == 4:
                imgs[W] = np.stack([_hwc_to_cw(imgs[W][i]) for i in range(len(imgs[W]))])
                np.save(_img_path(ticker, W), imgs[W])
        nan_cnt = (~np.isfinite(ohlc)).sum()
        if nan_cnt > 0:
            ohlc = np.nan_to_num(ohlc, nan=0., posinf=0., neginf=0.)
            np.save(_ohlc_path(ticker), ohlc)

        ctx_list = []
        if context is not None:
            for i in range(W_max, len(df) - F): ctx_list.append(context[i])
        ctx = np.array(ctx_list, dtype=np.float32) if ctx_list else None
        return _align_arrays(imgs, nums, cls, ohlc, ctx, hourly, aux)

    # ── Полное построение ──
    ohlc_all, cls_all, valid_all, atr_ratio_full = build_labels_residual(
        df, imoex=imoex)
    # atr_ratio_full: [N] — ATR(14)/close для каждого бара (из build_ohlc_labels)

    nan_before = (~np.isfinite(ohlc_all)).sum()
    if nan_before > 0:
        for i in range(len(ohlc_all)):
            if not np.isfinite(ohlc_all[i]).all():
                valid_all[i] = False
        ohlc_all = np.nan_to_num(ohlc_all, nan=0., posinf=0., neginf=0.)

    scale_imgs = {W: [] for W in SCALES}
    scale_nums = {W: [] for W in SCALES}
    ctx_list = []; y_cls_list = []; y_ohlc_list = []
    valid_daily_indices = []

    num_arr      = df[INDICATOR_COLS].values.astype(np.float32)
    num_arr_safe = np.nan_to_num(num_arr, nan=0.)
    _train_end   = max(int(len(num_arr_safe) * 0.70), 10)
    assert _train_end < len(num_arr_safe), \
        f"[LEAK] train_end={_train_end} >= len={len(num_arr_safe)}"
    scaler   = RobustScaler()
    scaler.fit(num_arr_safe[:_train_end])
    num_norm = np.clip(
        scaler.transform(num_arr_safe), -10., 10.).astype(np.float32)

    total = len(df) - W_max - F
    for i, idx in enumerate(range(W_max, len(df) - F)):
        if not valid_all[idx]:
            continue
        if i % 100 == 0:
            print(f"  {ticker}: рендер v3.4.1 {i}/{total}", end="\r")
        for W in SCALES:
            img_cw = _hwc_to_cw(render_candles(df.iloc[idx - W: idx]))
            if not np.isfinite(img_cw).all():
                img_cw = np.nan_to_num(img_cw, nan=0., posinf=0., neginf=0.)
            scale_imgs[W].append(img_cw)
            scale_nums[W].append(num_norm[idx - W: idx])
        if context is not None:
            ctx_list.append(context[idx])
        y_cls_list.append(cls_all[idx])
        y_ohlc_list.append(ohlc_all[idx])
        valid_daily_indices.append(idx)
    print()

    imgs = {W: np.array(scale_imgs[W], dtype=np.float32) for W in SCALES}
    nums = {W: np.array(scale_nums[W], dtype=np.float32) for W in SCALES}
    cls  = np.array(y_cls_list,  dtype=np.int64)
    ohlc = np.array(y_ohlc_list, dtype=np.float32)
    ctx  = np.array(ctx_list, dtype=np.float32) if ctx_list else None

    hourly = None
    if hourly_df is not None and len(valid_daily_indices) > 0:
        print(f"  {ticker}: рендер часовых свечей...")
        hourly = _build_hourly_windows(hourly_df, df, valid_daily_indices)

    # ── Aux labels: [vol, skew, atr_ratio] ──────────────────────
    close_arr = df["close"].values.astype(np.float64)
    log_rets  = np.diff(np.log(np.clip(close_arr, 1e-9, None)))
    aux_list  = []

    for idx in valid_daily_indices:
        end_idx = min(idx + F, len(log_rets))
        fut     = log_rets[idx: end_idx]

        # atr_ratio для этого бара
        atr_v = (float(atr_ratio_full[idx])
                 if idx < len(atr_ratio_full) and np.isfinite(atr_ratio_full[idx])
                 else 0.018)
        atr_v = float(np.clip(atr_v, 0.001, 0.15))  # 0.1% - 15% диапазон

        if len(fut) < 2:
            aux_list.append([0., 0., atr_v])
        else:
            vol  = float(np.std(fut))
            mean = np.mean(fut)
            m2   = np.mean((fut - mean) ** 2)
            m3   = np.mean((fut - mean) ** 3)
            skew = float(np.clip(m3 / (m2 ** 1.5 + 1e-12), -5., 5.))
            aux_list.append([vol, skew, atr_v])

    aux = np.array(aux_list, dtype=np.float32)  # [N, 3]

    result = _align_arrays(imgs, nums, cls, ohlc, ctx, hourly, aux)
    imgs, nums, cls, ohlc, ctx, hourly, aux = result

    for W in SCALES:
        np.save(_img_path(ticker, W), imgs[W])
        np.save(_num_path(ticker, W), nums[W])
    np.save(_cls_path(ticker),  cls)
    np.save(_ohlc_path(ticker), ohlc)
    np.save(_aux_path(ticker),  aux)   # теперь [N, 3]
    if hourly is not None:
        np.save(_hourly_path(ticker), hourly)

    return imgs, nums, cls, ohlc, ctx, hourly, aux


def build_full_multiscale_dataset_v3(
    force_rebuild: bool = False,
    use_hourly: bool = True,
    ticker_filter: Optional[list] = None,
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
        print("  Все тикеры в кэше. probe-проверка актуальности...")
        if probe_freshness(client, CFG.tickers, meta):
            print("  ✓ Кэш актуален")
            return _load_all_from_cache(meta, use_hourly=use_hourly)
        print("  Кэш устарел — обновляем изменившиеся тикеры")

    records = []; all_cls = []; ctx_dim = None; ticker_lengths = []
    tickers_to_use = (CFG.tickers if ticker_filter is None
                      else [t for t in CFG.tickers if t in ticker_filter])
    if not tickers_to_use:
        raise RuntimeError(f"ticker_filter={ticker_filter} не совпал ни с одним тикером")

    for ticker in tickers_to_use:
        has_aux = os.path.exists(_aux_path(ticker))
        if not force_rebuild and ticker_cache_valid(ticker, meta) and has_aux:
            print(f"  {ticker}: кэш v3.4 актуален ✓")
            cls = np.load(_cls_path(ticker), mmap_mode="r")
            ohlc_p = _ohlc_path(ticker)
            if os.path.exists(ohlc_p):
                oc = np.load(ohlc_p, mmap_mode="r")
                if (~np.isfinite(oc)).sum() > 0:
                    np.save(ohlc_p, np.nan_to_num(np.array(oc), nan=0., posinf=0., neginf=0.))
            n = len(cls)
            for local_idx in range(n): records.append((ticker, local_idx))
            all_cls.append(cls); ticker_lengths.append((ticker, n))
            cp = _ctx_path(ticker)
            if os.path.exists(cp) and ctx_dim is None:
                ctx_dim = np.load(cp, mmap_mode="r").shape[1]
            continue

        if not has_aux:
            print(f"  {ticker}: aux не найден → принудительный rebuild")

        print(f"  Загружаем {ticker} из API...")
        try:
            figi = client.find_figi(ticker)
            if not figi: continue

            df = _load_daily_candles_chunked(client, figi)
            if df is None or df.empty: continue

            hourly_df  = _load_hourly_candles(client, figi) if use_hourly else None
            ctx_series = load_context_series(ticker)
            ctx_feats  = build_context_features(
                ctx_series, df.index, ticker=ticker,
                imoex_close=imoex["close"] if imoex is not None else None,
            ) if ctx_series is not None else None

            imgs, nums, cls, ohlc, ctx, hourly, aux = build_multiscale_dataset_v3(
                df, imoex, ctx_feats, ticker=ticker,
                hourly_df=hourly_df, force_rebuild=force_rebuild,
            )
            if len(cls) == 0: continue

            if ctx is not None and len(ctx) > 0:
                np.save(_ctx_path(ticker), ctx)

            update_meta(ticker, df, len(cls), meta); _save_meta(meta)

            cp = _ctx_path(ticker)
            if os.path.exists(cp) and ctx_dim is None:
                ctx_dim = np.load(cp, mmap_mode="r").shape[1]

            for local_idx in range(len(cls)): records.append((ticker, local_idx))
            all_cls.append(cls); ticker_lengths.append((ticker, len(cls)))
            print(f"  {ticker}: {len(cls)} сэмплов ✓")

        except Exception as e:
            print(f"  {ticker}: ошибка — {e}"); traceback.print_exc()

    if not records: raise RuntimeError("Не удалось загрузить данные.")
    ctx_dim = ctx_dim or 0
    y_all   = np.concatenate(all_cls)
    dataset = LazyMultiScaleDatasetV3(records, ctx_dim, use_hourly=use_hourly)
    return dataset, y_all, ctx_dim, ticker_lengths


def _load_all_from_cache(meta: dict, use_hourly: bool = True):
    records = []; all_cls = []; ctx_dim = None; ticker_lengths = []
    for ticker in CFG.tickers:
        cp = _cls_path(ticker)
        if not os.path.exists(cp): continue
        cls = np.load(cp, mmap_mode="r"); n = len(cls)
        for local_idx in range(n): records.append((ticker, local_idx))
        all_cls.append(cls); ticker_lengths.append((ticker, n))
        ctx_p = _ctx_path(ticker)
        if os.path.exists(ctx_p) and ctx_dim is None:
            ctx_dim = np.load(ctx_p, mmap_mode="r").shape[1]
    ctx_dim = ctx_dim or 0
    y_all   = np.concatenate(all_cls)
    dataset = LazyMultiScaleDatasetV3(records, ctx_dim, use_hourly=use_hourly)
    print(f"  ✓ Кэш: {len(records)} сэмплов, {len(ticker_lengths)} тикеров")
    return dataset, y_all, ctx_dim, ticker_lengths


def temporal_split(ticker_lengths, val_ratio=0.15, test_ratio=0.15, purge_bars=None):
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
        for i in range(max(val_start,0), max(val_end,0)):    idx_val.append(global_offset + i)
        for i in range(test_start, n):                       idx_test.append(global_offset + i)
        global_offset += n
    return (np.array(idx_train, dtype=np.int64),
            np.array(idx_val,   dtype=np.int64),
            np.array(idx_test,  dtype=np.int64))

def build_labels_residual(df: pd.DataFrame,
                          imoex: pd.DataFrame = None,
                          future_bars: int = None,
                          atr_k: float = None):
    """v3.4.2: метки по residual return (stock - imoex).
    
    На h=1 residual даёт ~50/50 распределение UP/DOWN с предсказуемым
    сигналом. Убирает bull-bias российского рынка (~55-65% UP на длинных
    горизонтах, который перекрывает любой сигнал индикаторов).
    """
    if future_bars is None:
        future_bars = CFG.future_bars
    if atr_k is None:
        atr_k = CFG.label_atr_k

    ohlc_all, _, valid_all, extra = build_ohlc_labels(df)

    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)

    # Market drift (IMOEX)
    if imoex is not None:
        idx_c = imoex["close"].astype(float).reindex(df.index).ffill()
        idx_vals = idx_c.values.astype(np.float64)
    else:
        idx_vals = None
        print("  [WARN] IMOEX не передан в build_labels_residual — "
              "метки будут по raw return (с bull-bias)")

    # ATR
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[tr[0] if len(tr) else 0.0], tr])
    atr14 = np.zeros(len(c))
    if len(c) >= 14:
        atr14[13] = tr[:14].mean()
        for i in range(14, len(c)):
            atr14[i] = (atr14[i - 1] * 13.0 + tr[i]) / 14.0
        atr14[:13] = atr14[13]

    cls_all = np.ones(len(c), dtype=np.int64)   # HOLD по умолчанию

    for i in range(len(c) - future_bars):
        if not valid_all[i]:
            continue
        c_now = max(c[i], 1e-9)
        c_fut = c[i + future_bars]
        stock_ret = (c_fut - c_now) / c_now

        # Residual
        if idx_vals is not None:
            idx_now = idx_vals[i]
            idx_fut = idx_vals[min(i + future_bars, len(idx_vals) - 1)]
            if np.isfinite(idx_now) and np.isfinite(idx_fut) and idx_now > 0:
                market_ret = (idx_fut - idx_now) / idx_now
                signal_ret = stock_ret - market_ret   # ← residual
            else:
                signal_ret = stock_ret
        else:
            signal_ret = stock_ret

        thresh = atr_k * atr14[i] / c_now
        if thresh < 1e-6:
            continue
        if signal_ret > thresh:
            cls_all[i] = 0   # BUY (outperform market)
        elif signal_ret < -thresh:
            cls_all[i] = 2   # SELL (underperform market)

    return ohlc_all, cls_all, valid_all, extra