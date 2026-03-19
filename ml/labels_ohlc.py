"""OHLC-регрессионная разметка.

Для каждого момента t предсказываем future_bars свечей (по умолчанию 5).
Выход: (future_bars, 4) = 20 значений — относительные изменения O/H/L/C
от текущего close[t].

  ΔO[t+k] = (Open[t+k]  - Close[t]) / Close[t]
  ΔH[t+k] = (High[t+k]  - Close[t]) / Close[t]
  ΔL[t+k] = (Low[t+k]   - Close[t]) / Close[t]
  ΔC[t+k] = (Close[t+k] - Close[t]) / Close[t]

Дополнительно сохраняем классификационную метку (BUY/HOLD/SELL)
для multi-task обучения.
"""
import numpy as np
import pandas as pd
from ml.config import CFG


def build_ohlc_labels(df: pd.DataFrame) -> tuple:
    """
    Вход:  df с колонками open/high/low/close (после add_indicators и dropna).
    Выход: (ohlc_labels, cls_labels, valid_mask)
        ohlc_labels: np.ndarray (N, future_bars, 4) — ΔO, ΔH, ΔL, ΔC
        cls_labels:  np.ndarray (N,) — 0=BUY, 1=HOLD, 2=SELL
        valid_mask:  np.ndarray (N,) — bool, True если все future_bars доступны
    """
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    N = len(df)
    F = CFG.future_bars  # 5

    ohlc = np.zeros((N, F, 4), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)

    for t in range(N - F):
        base = c[t]
        if base < 1e-9:
            continue
        for k in range(F):
            idx = t + 1 + k
            ohlc[t, k, 0] = (o[idx] - base) / base   # ΔO
            ohlc[t, k, 1] = (h[idx] - base) / base   # ΔH
            ohlc[t, k, 2] = (l[idx] - base) / base   # ΔL
            ohlc[t, k, 3] = (c[idx] - base) / base   # ΔC
        valid[t] = True

    # ── Классификационная метка (для multi-task) ──────────────
    # На основе net return последнего close
    net_ret = np.zeros(N, dtype=np.float64)
    net_ret[:N - F] = ohlc[:N - F, -1, 3] - 2 * CFG.broker_commission
    thr = CFG.effective_profit_thr

    cls = np.ones(N, dtype=np.int64)       # HOLD
    cls[net_ret > thr] = 0                 # BUY
    cls[net_ret < -thr] = 2                # SELL

    return ohlc, cls, valid


def ohlc_to_strategy_features(ohlc_pred: np.ndarray) -> dict:
    """
    Из предсказанных (future_bars, 4) извлекает торговые метрики.

    Вход:  ohlc_pred shape (F, 4) — ΔO, ΔH, ΔL, ΔC для F будущих свечей.
    Выход: dict с ключами:
      - max_upside:     max(ΔH) — максимальный потенциал роста
      - max_downside:   min(ΔL) — максимальная просадка
      - final_return:   ΔC[-1]  — return к концу горизонта
      - risk_reward:    max_upside / abs(max_downside)
      - best_entry_bar: бар с min(ΔL) — лучшая точка входа
      - best_exit_bar:  бар с max(ΔH) — лучшая точка выхода
      - direction:      BUY / HOLD / SELL (на основе final_return и risk/reward)
      - volatility:     std по всем ΔC — ожидаемая волатильность
    """
    dH = ohlc_pred[:, 1]  # ΔHigh
    dL = ohlc_pred[:, 2]  # ΔLow
    dC = ohlc_pred[:, 3]  # ΔClose

    max_up   = float(dH.max())
    max_down = float(dL.min())
    final_ret = float(dC[-1])
    vol      = float(dC.std())

    # Risk/reward
    risk = abs(max_down) if max_down < 0 else 1e-9
    rr   = max(max_up, 0) / risk   # если upside < 0, rr = 0

    # Направление: учитываем и final return, и risk/reward
    thr = CFG.effective_profit_thr
    commission = 2 * CFG.broker_commission
    net_final = final_ret - commission

    if net_final > thr and rr > 1.5:
        direction = "BUY"
    elif net_final < -thr and rr < 0.67:
        direction = "SELL"
    else:
        direction = "HOLD"

    return {
        "max_upside":     max_up,
        "max_downside":   max_down,
        "final_return":   final_ret,
        "risk_reward":    rr,
        "best_entry_bar": int(dL.argmin()),
        "best_exit_bar":  int(dH.argmax()),
        "direction":      direction,
        "volatility":     vol,
    }
