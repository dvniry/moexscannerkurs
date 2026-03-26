# ml/labels_ohlc.py
"""OHLC-лейблы v3.3: классификация направления первой свечи вместо торгового сигнала.

Изменения v3.2 → v3.3:
- cls_labels теперь UP(0) / FLAT(1) / DOWN(2) по ΔClose[+1]
- Порог адаптивный (ATR-based) как раньше, но применяется к одной свече
- Убран net_return за future_bars — торговая логика остаётся в Lua-стратегии
- ohlc_to_strategy_features без изменений (совместимость с backtest.py)
"""
import numpy as np
import pandas as pd
from ml.config import CFG


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

    atr       = pd.Series(tr).rolling(20, min_periods=5).mean().ffill()
    vol_ratio = atr / (close + 1e-9)
    vol_ratio = np.clip(vol_ratio, 0, 2.0)   # защита от выбросов

    k          = CFG.adaptive_k
    min_thr    = CFG.adaptive_min_thr
    commission = 2 * CFG.broker_commission

    thresholds = np.maximum(min_thr, k * vol_ratio) + commission
    return thresholds.astype(np.float64)


def build_ohlc_labels(df: pd.DataFrame):
    """Строит OHLC regression labels + direction classification labels.

    Returns:
        ohlc_labels : (N, future_bars * 4) — ΔO, ΔH, ΔL, ΔC для каждого бара
                      все дельты относительно close[i] текущей свечи
        cls_labels  : (N,) — 0=UP, 1=FLAT, 2=DOWN
                      направление ПЕРВОЙ следующей свечи (ΔClose[i+1])
        valid_mask  : (N,) — True если label валиден
    """
    F = CFG.future_bars
    N = len(df)

    close = df['close'].values.astype(np.float64)
    open_ = df['open'].values.astype(np.float64)
    high  = df['high'].values.astype(np.float64)
    low   = df['low'].values.astype(np.float64)

    ohlc_labels = np.zeros((N, F * 4), dtype=np.float32)
    cls_labels  = np.ones(N, dtype=np.int64)   # default = FLAT
    valid_mask  = np.zeros(N, dtype=bool)

    if CFG.use_adaptive_threshold:
        thresholds = _compute_adaptive_threshold(df)
    else:
        thresholds = np.full(N, CFG.effective_profit_thr)

    for i in range(N - F):
        c = close[i]
        if c <= 0:
            continue

        valid_mask[i] = True

        # ── OHLC регрессия: F свечей вперёд ──────────────────────
        for j in range(F):
            fi = i + j + 1
            if fi >= N:
                valid_mask[i] = False
                break
            ohlc_labels[i, j * 4 + 0] = (open_[fi] - c) / c   # ΔOpen
            ohlc_labels[i, j * 4 + 1] = (high[fi]  - c) / c   # ΔHigh
            ohlc_labels[i, j * 4 + 2] = (low[fi]   - c) / c   # ΔLow
            ohlc_labels[i, j * 4 + 3] = (close[fi] - c) / c   # ΔClose

        if not valid_mask[i]:
            continue

        # ── Классификация: направление ПЕРВОЙ свечи ──────────────
        # Используем ΔClose[i+1] — самый надёжный сигнал для aux loss
        delta_close_1 = (close[i + 1] - c) / c
        thr = thresholds[i]

        if delta_close_1 > thr:
            cls_labels[i] = 0   # UP
        elif delta_close_1 < -thr:
            cls_labels[i] = 2   # DOWN
        else:
            cls_labels[i] = 1   # FLAT

    return ohlc_labels, cls_labels, valid_mask


def ohlc_to_strategy_features(ohlc_pred: np.ndarray) -> dict:
    """Преобразует OHLC предсказание в торговые фичи для backtest/Lua.

    Args:
        ohlc_pred: (F*4,) или (F, 4) — ΔO, ΔH, ΔL, ΔC

    Returns:
        dict с ключами совместимыми с backtest.direction_risk_reward
    """
    ohlc = np.asarray(ohlc_pred, dtype=np.float32).reshape(-1, 4)  # (F, 4)

    max_high    = float(ohlc[:, 1].max())
    min_low     = float(ohlc[:, 2].min())
    final_close = float(ohlc[-1, 3])
    avg_close   = float(ohlc[:, 3].mean())
    next_close  = float(ohlc[0, 3])   # ΔClose первой свечи — новый ключ

    max_upside   = max(max_high, 0.0)
    max_downside = min(min_low,  0.0)

    return {
        # ── первичные (используются в backtest.py) ────────────────
        'max_upside':   max_upside,
        'max_downside': max_downside,
        'final_return': final_close,
        # ── новые ────────────────────────────────────────────────
        'next_close':   next_close,     # ΔClose завтра — для Lua
        'momentum':     avg_close,      # средний тренд за горизонт
        # ── дополнительные ───────────────────────────────────────
        'avg_close':    avg_close,
        'expected_move': float(max(abs(max_upside), abs(max_downside))),
        'range':        float(max_upside - max_downside),
        # ── legacy aliases ────────────────────────────────────────
        'max_high':     max_high,
        'min_low':      min_low,
        'final_close':  final_close,
    }
