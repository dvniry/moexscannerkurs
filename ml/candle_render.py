"""Рендер свечей OHLCV в 2D изображение 64×64."""
import numpy as np
from PIL import Image, ImageDraw


def render_candles(df_window) -> np.ndarray:
    """
    Нарисовать свечной график → (64, 64) float32 [0,1].
    Белая свеча = рост, серая = падение.
    Нижняя полоска = нормализованный объём.
    """
    W, H = 64, 64
    img  = Image.new("L", (W, H), color=0)
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

    def py(p):
        return int(H - 4 - (p - price_min) / price_rng * (H - 12))

    cw = max(1, (W - 4) // n - 1)

    for i in range(n):
        xc = 2 + i * (W - 4) // n + cw // 2

        # Фитиль
        draw.line([(xc, py(h[i])), (xc, py(l[i]))], fill=200, width=1)

        # Тело
        yo = py(o[i])
        yc = py(c[i])
        color = 255 if c[i] >= o[i] else 128
        draw.rectangle(
            [xc - cw // 2, min(yo, yc), xc + cw // 2, max(yo, yc) + 1],
            fill=color,
        )

    # Объём
    vol_max = v.max() + 1e-9
    for i in range(n):
        xc    = 2 + i * (W - 4) // n + cw // 2
        vol_h = int((v[i] / vol_max) * 7)
        if vol_h > 0:
            draw.line(
                [(xc, H - 1), (xc, H - 1 - vol_h)],
                fill=160, width=cw,
            )

    return np.array(img, dtype=np.float32) / 255.0
