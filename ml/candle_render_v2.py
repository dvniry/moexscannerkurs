"""Рендер свечей OHLCV → многоканальный тензор (factual, без визуального шума).

Каждый канал — один нормированный временной ряд.
Выход: (N_channels, W) — не картинка, а фактические данные.

Каналы:
  0: Open   (norm)
  1: High   (norm)
  2: Low    (norm)
  3: Close  (norm)
  4: Volume (norm to [0,1] by max)
  5: ATR(14)  (norm)
  6: RSI(14)  (/ 100)
  7: MA5    (norm)
  8: MA20   (norm)
  9: Keltner Upper (norm)
 10: Keltner Lower (norm)

Нормировка OHLC/MA/Keltner: min-max по (low.min, high.max) окна.
Это сохраняет относительные пропорции между O/H/L/C/MA/Keltner.
"""
import numpy as np
import pandas as pd

N_RENDER_CHANNELS = 11


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, p: int = 14) -> np.ndarray:
    """Average True Range."""
    n = len(c)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return pd.Series(tr).rolling(p, min_periods=1).mean().values


def _rsi(c: np.ndarray, p: int = 14) -> np.ndarray:
    """RSI в диапазоне [0, 100]."""
    d = pd.Series(c).diff()
    gain = d.clip(lower=0).rolling(p, min_periods=1).mean()
    loss = (-d.clip(upper=0)).rolling(p, min_periods=1).mean()
    rsi = 100 - 100 / (1 + gain / (loss + 1e-9))
    return rsi.fillna(50.0).values


def render_candles(df_window) -> np.ndarray:
    """
    Вход:  df_window — DataFrame с колонками open/high/low/close/volume, длина W.
    Выход: np.ndarray shape (N_RENDER_CHANNELS, W), dtype float32.
    """
    o = df_window['open'].values.astype(np.float64)
    h = df_window['high'].values.astype(np.float64)
    l = df_window['low'].values.astype(np.float64)
    c = df_window['close'].values.astype(np.float64)
    v = df_window['volume'].values.astype(np.float64)
    W = len(o)

    # ── Индикаторы ────────────────────────────────────────────
    atr_vals = _atr(h, l, c, 14)
    rsi_vals = _rsi(c, 14)
    ma5      = pd.Series(c).rolling(5,  min_periods=1).mean().values
    ma20     = pd.Series(c).rolling(20, min_periods=1).mean().values
    kelt_up  = ma20 + 1.5 * atr_vals
    kelt_dn  = ma20 - 1.5 * atr_vals

    # ── Нормировка OHLC-группы по общему диапазону ────────────
    price_min = l.min()
    price_max = h.max()
    price_rng = price_max - price_min + 1e-9

    def norm_price(arr):
        return ((arr - price_min) / price_rng).astype(np.float32)

    # ── Нормировка остальных ──────────────────────────────────
    vol_norm = (v / (v.max() + 1e-9)).astype(np.float32)
    atr_norm = norm_price(atr_vals + price_min)  # ATR — в единицах цены, нормируем
    # Более корректно: ATR / price_rng
    atr_norm = (atr_vals / price_rng).astype(np.float32)
    rsi_norm = (rsi_vals / 100.0).astype(np.float32)

    # ── Собираем каналы ───────────────────────────────────────
    channels = np.stack([
        norm_price(o),          # 0: Open
        norm_price(h),          # 1: High
        norm_price(l),          # 2: Low
        norm_price(c),          # 3: Close
        vol_norm,               # 4: Volume
        atr_norm,               # 5: ATR
        rsi_norm,               # 6: RSI
        norm_price(ma5),        # 7: MA5
        norm_price(ma20),       # 8: MA20
        norm_price(kelt_up),    # 9: Keltner Upper
        norm_price(kelt_dn),    # 10: Keltner Lower
    ], axis=0)  # (11, W)

    return channels.astype(np.float32)
