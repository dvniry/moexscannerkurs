"""Рендер свечей OHLCV в RGB изображение 224×224 для EfficientNet."""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

IMG_SIZE = 224


def render_candles(df_window) -> np.ndarray:
    """
    Свечной график + MA5 + MA20 + объём → (3, 224, 224) float32, ImageNet norm.
    """
    W, H = IMG_SIZE, IMG_SIZE
    img  = Image.new("RGB", (W, H), color=(15, 15, 15))
    draw = ImageDraw.Draw(img)

    o = df_window['open'].values.astype(float)
    h = df_window['high'].values.astype(float)
    l = df_window['low'].values.astype(float)
    c = df_window['close'].values.astype(float)
    v = df_window['volume'].values.astype(float)
    n = len(o)

    price_min = l.min()
    price_max = h.max()
    price_rng = price_max - price_min + 1e-9

    MARGIN_TOP    = 10
    MARGIN_BOTTOM = 20
    chart_h = H - MARGIN_TOP - MARGIN_BOTTOM

    def py(p):
        return int(H - MARGIN_BOTTOM - (p - price_min) / price_rng * chart_h)

    cw     = max(2, (W - 8) // n - 2)
    stride = (W - 8) // n

    # Свечи
    for i in range(n):
        xc = 4 + i * stride + stride // 2
        draw.line([(xc, py(h[i])), (xc, py(l[i]))],
                  fill=(180, 180, 180), width=1)
        yo    = py(o[i])
        yc    = py(c[i])
        color = (80, 200, 80) if c[i] >= o[i] else (200, 60, 60)
        body_top = min(yo, yc)
        body_bot = max(yo, yc) + 1
        if body_bot - body_top < 2:
            body_bot = body_top + 2
        draw.rectangle([xc - cw // 2, body_top,
                        xc + cw // 2, body_bot], fill=color)

    # Объём
    vol_max = v.max() + 1e-9
    for i in range(n):
        xc    = 4 + i * stride + stride // 2
        vol_h = int((v[i] / vol_max) * (MARGIN_BOTTOM - 4))
        if vol_h > 0:
            draw.line([(xc, H - 2), (xc, H - 2 - vol_h)],
                      fill=(100, 100, 200), width=max(1, cw - 1))

    # MA линии
    def _draw_ma(period: int, color: tuple):
        ma = pd.Series(c).rolling(period, min_periods=1).mean().values
        prev = None
        for i in range(n):
            xc = 4 + i * stride + stride // 2
            cur = (xc, py(ma[i]))
            if prev is not None:
                draw.line([prev, cur], fill=color, width=1)
            prev = cur

    _draw_ma(5,  (255, 200,  40))   # MA5  — жёлтый
    _draw_ma(20, (40,  160, 255))   # MA20 — синий

    arr  = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std

    return arr.transpose(2, 0, 1)   # → (3, 224, 224)
