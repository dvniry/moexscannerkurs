# ml/candle_render_v2.py
"""Рендеринг свечей v2.2: ATR нормировка с clip для предотвращения выбросов.

Изменения v2.2:
- Добавлена константа N_RENDER_CHANNELS = 3 (нужна multiscale_cnn_v3)
- atr_norm: np.clip(..., 0.0, 2.0) — предотвращает значения >1
  при аномально высокой волатильности (дни ГЭПов, кризисов)
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# ── Константы ──────────────────────────────────────────────────────────────
CANDLE_UP_COLOR   = (80,  200, 120)   # зелёный
CANDLE_DOWN_COLOR = (220,  60,  60)   # красный
CANDLE_DOJI_COLOR = (180, 180, 180)   # серый
WICK_COLOR        = (200, 200, 200)

VOLUME_UP_COLOR   = (80,  200, 120, 80)
VOLUME_DOWN_COLOR = (220,  60,  60, 80)

BG_COLOR = (15, 15, 20)  # тёмный фон

# Количество каналов, которое возвращает render_candles:
#   канал 0 = серая интенсивность свечей
#   канал 1 = объём
#   канал 2 = ATR
N_RENDER_CHANNELS = 3


def _compute_atr(highs: np.ndarray, lows: np.ndarray,
                 closes: np.ndarray, period: int = 14) -> np.ndarray:
    n  = len(closes)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().values
    return atr


def render_candles(df: pd.DataFrame,
                   width:       int  = 64,
                   height:      int  = 64,
                   show_volume: bool = True,
                   show_atr:    bool = True) -> np.ndarray:
    """Рендерит свечной график в numpy-массив (H, W, N_RENDER_CHANNELS).

    Returns:
        img: (H, W, 3) float32 в диапазоне [0, 1]
    """
    n = len(df)
    if n == 0:
        return np.zeros((height, width, N_RENDER_CHANNELS), dtype=np.float32)

    opens  = df["open"].values.astype(np.float64)
    highs  = df["high"].values.astype(np.float64)
    lows   = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    vols   = (df["volume"].values.astype(np.float64)
              if "volume" in df.columns else np.zeros(n))

    # ── Нормировка цены ──────────────────────────────────────────
    price_min = lows.min()
    price_max = highs.max()
    price_rng = price_max - price_min + 1e-9

    def price_to_y(p: float) -> int:
        frac = (p - price_min) / price_rng
        return int(height - 1 - frac * (height - 1))

    # ── Рендер свечей ────────────────────────────────────────────
    candle_img = Image.new("RGB", (width, height), BG_COLOR)
    draw       = ImageDraw.Draw(candle_img)

    candle_w = max(1, width // n - 1)
    gap      = max(1, (width - candle_w * n) // max(n - 1, 1))

    for i in range(n):
        x_center = int(i * (candle_w + gap) + candle_w / 2)
        x_left   = x_center - candle_w // 2
        x_right  = x_left + candle_w

        y_high = price_to_y(highs[i])
        y_low  = price_to_y(lows[i])
        y_open = price_to_y(opens[i])
        y_cls  = price_to_y(closes[i])

        is_up  = closes[i] >= opens[i]
        color  = (CANDLE_UP_COLOR   if is_up else
                  CANDLE_DOWN_COLOR if closes[i] != opens[i] else
                  CANDLE_DOJI_COLOR)

        draw.line([(x_center, y_high), (x_center, y_low)],
                  fill=WICK_COLOR, width=1)
        y_top = min(y_open, y_cls)
        y_bot = max(y_open, y_cls)
        if y_top == y_bot:
            y_bot += 1
        draw.rectangle([x_left, y_top, x_right, y_bot], fill=color)

    candle_arr = np.array(candle_img, dtype=np.float32) / 255.0  # (H, W, 3)

    # ── Канал объёма ─────────────────────────────────────────────
    vol_channel = np.zeros((height, width), dtype=np.float32)
    if show_volume and vols.max() > 0:
        vol_norm = vols / (vols.max() + 1e-9)
        for i in range(n):
            x_center = int(i * (candle_w + gap) + candle_w / 2)
            x_left   = x_center - candle_w // 2
            x_right  = x_left + candle_w
            bar_h    = int(vol_norm[i] * height * 0.3)
            y_top    = height - bar_h
            y_bot    = height - 1
            col_v    = 0.3 if closes[i] >= opens[i] else 0.8
            for y in range(max(0, y_top), min(height, y_bot + 1)):
                for x in range(max(0, x_left), min(width, x_right + 1)):
                    vol_channel[y, x] = col_v

    # ── Канал ATR ─────────────────────────────────────────────────
    atr_channel = np.zeros((height, width), dtype=np.float32)
    if show_atr:
        atr_vals = _compute_atr(highs, lows, closes)
        # FIX v2.1: clip нормированного ATR в [0, 2]
        atr_norm = np.clip(atr_vals / (price_rng + 1e-9), 0.0, 2.0)
        for i in range(n):
            x_center = int(i * (candle_w + gap) + candle_w / 2)
            x_left   = x_center - candle_w // 2
            x_right  = x_left + candle_w
            atr_y    = int((1.0 - min(atr_norm[i], 1.0)) * (height - 1))
            atr_y    = max(0, min(height - 1, atr_y))
            for x in range(max(0, x_left), min(width, x_right + 1)):
                atr_channel[atr_y, x] = atr_norm[i]

    # ── Собираем N_RENDER_CHANNELS-канальное изображение ─────────
    gray = (0.299 * candle_arr[:, :, 0]
            + 0.587 * candle_arr[:, :, 1]
            + 0.114 * candle_arr[:, :, 2])
    out = np.stack([gray, vol_channel, atr_channel], axis=2)  # (H, W, 3)
    return out.astype(np.float32)


def render_candles_batch(dfs: list,
                         width:  int = 64,
                         height: int = 64,
                         **kwargs) -> np.ndarray:
    """Батч-рендер: list[DataFrame] → (N, H, W, N_RENDER_CHANNELS)."""
    imgs = [render_candles(df, width, height, **kwargs) for df in dfs]
    return np.stack(imgs, axis=0)
