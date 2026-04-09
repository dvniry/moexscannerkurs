# ml/labels_ohlc.py
"""OHLC-лейблы v3.5: UP/FLAT/DOWN + ATR-нормировка регрессионных целей.

Изменения v3.5 (bugfix):
- guard atr_i < 5e-4 (было 1e-5) → исключаем бумаги с ATR/close < 0.05%
- ohlc_labels[i] клампируем в [-5, +5] после нормировки → нет взрыва Huber
"""
import numpy as np
import pandas as pd
from ml.config import CFG

CLS_UP   = 0
CLS_FLAT = 1
CLS_DOWN = 2
CLS_NAMES = ['UP', 'FLAT', 'DOWN']


def _compute_adaptive_threshold(df: pd.DataFrame) -> np.ndarray:
    high  = df['high'].values.astype(np.float64)
    low   = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)

    tr = np.zeros(len(df))
    tr[0] = high[0] - low[0]
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i]  - close[i - 1]),
        )

    atr = pd.Series(tr).rolling(20, min_periods=5).mean().ffill()
    vol_ratio = atr / (close + 1e-9)

    k          = CFG.adaptive_k
    min_thr    = CFG.adaptive_min_thr
    commission = 2 * CFG.broker_commission
    thresholds = np.maximum(min_thr, k * vol_ratio) + commission
    return thresholds.astype(np.float64)


def _compute_atr_ratio(df: pd.DataFrame) -> np.ndarray:
    high  = df['high'].values.astype(np.float64)
    low   = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)

    tr = np.zeros(len(df))
    tr[0] = high[0] - low[0]
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i]  - close[i - 1]),
        )

    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    return atr / (close + 1e-9)


def build_ohlc_labels(df: pd.DataFrame):
    F     = CFG.future_bars
    N     = len(df)
    close = df['close'].values.astype(np.float64)
    open_ = df['open'].values.astype(np.float64)
    high  = df['high'].values.astype(np.float64)
    low   = df['low'].values.astype(np.float64)

    ohlc_labels = np.zeros((N, F * 4), dtype=np.float32)
    cls_labels  = np.ones(N, dtype=np.int64)
    valid_mask  = np.zeros(N, dtype=bool)

    thresholds = (
        _compute_adaptive_threshold(df)
        if CFG.use_adaptive_threshold
        else np.full(N, CFG.effective_profit_thr)
    )

    atr_ratio = _compute_atr_ratio(df)

    for i in range(N - F):
        c = close[i]
        if c <= 0:
            continue

        atr_i = atr_ratio[i]
        # FIX v3.5: порог повышен 1e-5 → 5e-4 (ATR/close < 0.05% = нулевой диапазон)
        if not np.isfinite(atr_i) or atr_i < 5e-4:
            continue

        valid_mask[i] = True

        norm = atr_i * np.sqrt(F)

        for j in range(F):
            fi = i + j + 1
            if fi >= N:
                valid_mask[i] = False
                break
            ohlc_labels[i, j * 4 + 0] = (open_[fi] - c) / c / norm
            ohlc_labels[i, j * 4 + 1] = (high[fi]  - c) / c / norm
            ohlc_labels[i, j * 4 + 2] = (low[fi]   - c) / c / norm
            ohlc_labels[i, j * 4 + 3] = (close[fi] - c) / c / norm

        if not valid_mask[i]:
            continue

        # FIX v3.5: clamp выбросов → max=9.97 из лога → обрезаем в ±5
        ohlc_labels[i] = np.clip(ohlc_labels[i], -5.0, 5.0)

        net_return = (close[i + F] - c) / c
        thr = thresholds[i]

        if np.isfinite(thr) and net_return > thr:
            cls_labels[i] = CLS_UP
        elif np.isfinite(thr) and net_return < -thr:
            cls_labels[i] = CLS_DOWN
        else:
            cls_labels[i] = CLS_FLAT

    return ohlc_labels, cls_labels, valid_mask, atr_ratio.astype(np.float32)


def denormalize_ohlc(ohlc_norm: np.ndarray, atr_ratio: float,
                     future_bars: int) -> np.ndarray:
    norm = atr_ratio * np.sqrt(future_bars)
    return ohlc_norm * norm


def ohlc_to_strategy_features(ohlc_pred: np.ndarray,
                               atr_ratio: float = None,
                               future_bars: int = None) -> dict:
    if atr_ratio is not None and future_bars is not None:
        ohlc_pred = denormalize_ohlc(ohlc_pred, atr_ratio, future_bars)

    ohlc = np.asarray(ohlc_pred, dtype=np.float32).reshape(-1, 4)

    max_high    = float(ohlc[:, 1].max())
    min_low     = float(ohlc[:, 2].min())
    final_close = float(ohlc[-1, 3])
    avg_close   = float(ohlc[:, 3].mean())

    max_upside   = max(max_high, 0.0)
    max_downside = min(min_low,  0.0)

    return {
        'max_upside':    max_upside,
        'max_downside':  max_downside,
        'final_return':  final_close,
        'avg_close':     avg_close,
        'expected_move': float(max(abs(max_upside), abs(max_downside))),
        'range':         float(max_upside - max_downside),
        'max_high':      max_high,
        'min_low':       min_low,
        'final_close':   final_close,
    }