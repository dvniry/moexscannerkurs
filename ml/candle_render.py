"""Рендер свечей OHLCV → RGB 224×224.
По методам Лиховидова:
  - ADX-фон: зелёный (бычий тренд) / красный (медвежий) / серый (флэт)
  - ATR-канал Keltner (MA20 ± 1.5×ATR14) вместо Bollinger
  - RSI(14) панель с 4 линиями (30/50/70 + динамическая медиана)
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

IMG_SIZE = 224

# ── Раскладка панелей ─────────────────────────────────────────────
PRICE_TOP  = 0
PRICE_BOT  = 155   # 155px — свечи + индикаторы
RSI_TOP    = 159
RSI_BOT    = 198   # 39px  — RSI панель
VOL_TOP    = 201
VOL_BOT    = 223   # 22px  — объём
ML         = 4     # margin left/right
MR         = 4


# ── Расчёт индикаторов ────────────────────────────────────────────

def _atr(h, l, c, p=14):
    n = len(c)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    return pd.Series(tr).rolling(p, min_periods=1).mean().values


def _adx(h, l, c, p=14):
    n = len(c)
    pdm, mdm, tr = np.zeros(n), np.zeros(n), np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        up, dn = h[i]-h[i-1], l[i-1]-l[i]
        pdm[i] = up   if (up > dn  and up > 0)   else 0.0
        mdm[i] = dn   if (dn > up  and dn > 0)   else 0.0
    s = pd.Series
    atr_ = s(tr).rolling(p, min_periods=1).mean().values
    pdi  = 100 * s(pdm).rolling(p, min_periods=1).mean().values / (atr_ + 1e-9)
    mdi  = 100 * s(mdm).rolling(p, min_periods=1).mean().values / (atr_ + 1e-9)
    dx   = 100 * np.abs(pdi - mdi) / (pdi + mdi + 1e-9)
    adx_ = s(dx).rolling(p, min_periods=1).mean().values
    # Заменить NaN нулями (флэт = нет тренда)
    adx_ = np.nan_to_num(adx_, nan=0.0)
    pdi  = np.nan_to_num(pdi,  nan=0.0)
    mdi  = np.nan_to_num(mdi,  nan=0.0)
    return adx_, pdi, mdi


def _rsi(c, p=14):
    d    = pd.Series(c).diff()
    gain = d.clip(lower=0).rolling(p, min_periods=1).mean()
    loss = (-d.clip(upper=0)).rolling(p, min_periods=1).mean()
    rsi  = 100 - 100 / (1 + gain / (loss + 1e-9))
    return rsi.fillna(50.0).values  # ← добавить .fillna(50.0)



# ── Основная функция ──────────────────────────────────────────────

def render_candles(df_window) -> np.ndarray:
    W, H = IMG_SIZE, IMG_SIZE
    img  = Image.new("RGB", (W, H), color=(15, 15, 15))
    draw = ImageDraw.Draw(img)

    o = df_window['open'].values.astype(float)
    h = df_window['high'].values.astype(float)
    l = df_window['low'].values.astype(float)
    c = df_window['close'].values.astype(float)
    v = df_window['volume'].values.astype(float)
    n = len(o)

    # Геометрия
    price_rng = h.max() - l.min() + 1e-9
    price_min = l.min()
    chart_h   = PRICE_BOT - PRICE_TOP - 8
    stride    = (W - ML - MR) // n
    cw        = max(2, stride - 2)

    def xc(i):  return ML + i * stride + stride // 2
    def py(p):  return int(PRICE_BOT - 4 - (p - price_min) / price_rng * chart_h)

    # ── Индикаторы ────────────────────────────────────────────────
    atr_vals         = _atr(h, l, c, 14)
    adx_vals, pdi, mdi = _adx(h, l, c, 14)
    rsi_vals         = _rsi(c, 14)
    ma5              = pd.Series(c).rolling(5,  min_periods=1).mean().values
    ma20             = pd.Series(c).rolling(20, min_periods=1).mean().values
    kelt_up          = ma20 + 1.5 * atr_vals
    kelt_dn          = ma20 - 1.5 * atr_vals

    # ── 1. ADX-фон (Лиховидов: тренд/флэт) ──────────────────────
    for i in range(n):
        x0, x1 = ML + i * stride, ML + (i + 1) * stride - 1
        if adx_vals[i] > 20:
            bg = (14, 34, 14) if pdi[i] > mdi[i] else (34, 14, 14)
        else:
            bg = (22, 22, 28)
        draw.rectangle([x0, PRICE_TOP, x1, PRICE_BOT], fill=bg)

    # ── 2. Keltner Channel (Лиховидов: ATR вместо σ) ─────────────
    prev_u = prev_d = None
    for i in range(n):
        xu = (xc(i), py(kelt_up[i]))
        xd = (xc(i), py(kelt_dn[i]))
        if prev_u:
            draw.line([prev_u, xu], fill=(0, 190, 210), width=1)
            draw.line([prev_d, xd], fill=(0, 190, 210), width=1)
        prev_u, prev_d = xu, xd

    # ── Свечи ────────────────────────────────────────────────────
    for i in range(n):
        x = xc(i)
        draw.line([(x, py(h[i])), (x, py(l[i]))], fill=(180, 180, 180), width=1)
        yo, yc_ = py(o[i]), py(c[i])
        col = (80, 200, 80) if c[i] >= o[i] else (200, 60, 60)
        bt, bb = min(yo, yc_), max(yo, yc_) + 1
        if bb - bt < 2: bb = bt + 2
        draw.rectangle([x - cw//2, bt, x + cw//2, bb], fill=col)

    # ── MA5 + MA20 ───────────────────────────────────────────────
    def _draw_line(vals, color, w=1):
        prev = None
        for i in range(n):
            cur = (xc(i), py(vals[i]))
            if prev: draw.line([prev, cur], fill=color, width=w)
            prev = cur

    _draw_line(ma5,  (255, 200, 40))
    _draw_line(ma20, (40,  160, 255))

    # ── Разделители ──────────────────────────────────────────────
    draw.line([(0, RSI_TOP - 2), (W, RSI_TOP - 2)], fill=(45, 45, 45))
    draw.line([(0, VOL_TOP - 2), (W, VOL_TOP - 2)], fill=(45, 45, 45))

    # ── 3. RSI панель с 4 линиями (Лиховидов) ────────────────────
    rsi_h = RSI_BOT - RSI_TOP

    def ry(val):
        if np.isnan(val): val = 50.0  # нет данных → середина
        return int(RSI_BOT - np.clip(val, 0, 100) / 100.0 * rsi_h)

    draw.rectangle([0, RSI_TOP, W, RSI_BOT], fill=(18, 18, 22))

    # Статичные уровни: 30/70
    draw.line([(0, ry(70)), (W, ry(70))], fill=(120, 40,  40),  width=1)
    draw.line([(0, ry(50)), (W, ry(50))], fill=(80,  80,  50),  width=1)
    draw.line([(0, ry(30)), (W, ry(30))], fill=(40,  120, 40),  width=1)

    # 4-я линия — скользящая медиана RSI (динамическая зона Лиховидова)
    rsi_med = pd.Series(rsi_vals).rolling(min(n, 20), min_periods=3).median().values
    prev = None
    for i in range(n):
        if not np.isnan(rsi_med[i]):
            cur = (xc(i), ry(rsi_med[i]))
            if prev: draw.line([prev, cur], fill=(210, 160, 40), width=1)
            prev = cur
        else:
            prev = None

    # Линия RSI (с цветом по зоне)
    prev = None
    for i in range(n):
        cur = (xc(i), ry(rsi_vals[i]))
        if prev:
            col = (220, 90, 90) if rsi_vals[i] > 70 else \
                  (90, 220, 90) if rsi_vals[i] < 30 else \
                  (170, 170, 170)
            draw.line([prev, cur], fill=col, width=1)
        prev = cur

    # ── Объём ────────────────────────────────────────────────────
    vol_max = v.max() + 1e-9
    vol_h   = VOL_BOT - VOL_TOP
    for i in range(n):
        bar = int(v[i] / vol_max * vol_h)
        if bar > 0:
            draw.line([(xc(i), VOL_BOT), (xc(i), VOL_BOT - bar)],
                      fill=(100, 100, 200), width=max(1, cw - 1))

    # ── ImageNet нормировка → (3, 224, 224) ──────────────────────
    arr  = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    return arr.transpose(2, 0, 1)
