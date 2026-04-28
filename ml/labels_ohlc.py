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
    if df is None or len(df) == 0:
        return np.empty((0,), dtype=np.float64)
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
    if df is None or len(df) == 0:
        return (
            np.empty((0, CFG.future_bars * 4), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=bool),
            np.empty((0,), dtype=np.float64),
        )
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


# ══════════════════════════════════════════════════════════════════
# Sprint 2 — экономические таргеты (MFE / MAE / fill / net edge)
# ══════════════════════════════════════════════════════════════════

ECON_N_COLS = 11
ECON_COL_NAMES = (
    'future_ret',
    'mfe_long', 'mae_long',
    'mfe_short', 'mae_short',
    'rr_long', 'rr_short',
    'fill_long', 'fill_short',
    'net_edge_long', 'net_edge_short',
)


def build_economic_targets(
    df: pd.DataFrame,
    valid_mask: np.ndarray,
    future_bars: int = None,
    commission: float = 0.0005,
    slippage:   float = 0.0003,
    spread:     float = 0.0002,
) -> np.ndarray:
    """Sprint 2: экономические таргеты для каждого бара.

    Returns: [N, 11] float32. Колонки см. ECON_COL_NAMES.

    Логика:
      future_ret      = (close[i+F] - close[i]) / close[i]
      mfe_long        = (max(highs[i+1..i+F]) - c0) / c0,  clamp[0, 0.20]
      mae_long        = max(0, (c0 - min(lows[i+1..i+F])) / c0),  clamp[0, 0.20]
      mfe_short       = (c0 - min(lows[i+1..i+F])) / c0,   clamp[0, 0.20]
      mae_short       = max(0, (max(highs[i+1..i+F]) - c0) / c0), clamp[0, 0.20]
      rr_long/short   = mfe / max(mae, 1e-6),               clamp[0, 10]
      fill_long       = 1 if low[i+1] <= c0 * (1 - 2*slippage) else 0
      fill_short      = 1 if high[i+1] >= c0 * (1 + 2*slippage) else 0
      net_edge_long   = future_ret  - (commission + slippage)
      net_edge_short  = -future_ret - (commission + slippage)

    Для баров где valid_mask[i] = False — все нули (бар не пойдёт в обучение
    из-за фильтра valid_all в dataset_v3.py).
    """
    if future_bars is None:
        future_bars = CFG.future_bars

    N = len(df)
    out = np.zeros((N, ECON_N_COLS), dtype=np.float32)

    if N == 0:
        return out

    close = df['close'].values.astype(np.float64)
    high  = df['high'].values.astype(np.float64)
    low   = df['low'].values.astype(np.float64)

    cost_one_side = commission + slippage
    slip2 = 2.0 * slippage
    F = int(future_bars)
    last_i = N - F

    for i in range(last_i):
        if not valid_mask[i]:
            continue
        c0 = close[i]
        if c0 <= 1e-9:
            continue

        h_max = float(np.max(high[i + 1: i + 1 + F]))
        l_min = float(np.min(low[i + 1: i + 1 + F]))
        c_fut = float(close[i + F])

        future_ret = (c_fut - c0) / c0

        mfe_long  = (h_max - c0) / c0                         # >= 0 если был рост
        mae_long  = max(0.0, (c0 - l_min) / c0)
        mfe_short = max(0.0, (c0 - l_min) / c0)
        mae_short = max(0.0, (h_max - c0) / c0)

        mfe_long  = min(max(mfe_long, 0.0), 0.20)
        mae_long  = min(mae_long, 0.20)
        mfe_short = min(mfe_short, 0.20)
        mae_short = min(mae_short, 0.20)

        rr_long  = min(mfe_long  / max(mae_long,  1e-6), 10.0)
        rr_short = min(mfe_short / max(mae_short, 1e-6), 10.0)

        first_low  = float(low[i + 1])
        first_high = float(high[i + 1])
        fill_long  = 1.0 if first_low  <= c0 * (1.0 - slip2) else 0.0
        fill_short = 1.0 if first_high >= c0 * (1.0 + slip2) else 0.0

        net_edge_long  =  future_ret - cost_one_side
        net_edge_short = -future_ret - cost_one_side

        out[i, 0]  = future_ret
        out[i, 1]  = mfe_long
        out[i, 2]  = mae_long
        out[i, 3]  = mfe_short
        out[i, 4]  = mae_short
        out[i, 5]  = rr_long
        out[i, 6]  = rr_short
        out[i, 7]  = fill_long
        out[i, 8]  = fill_short
        out[i, 9]  = net_edge_long
        out[i, 10] = net_edge_short

    return out


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