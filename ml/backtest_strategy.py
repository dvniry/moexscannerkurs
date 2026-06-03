"""Backtest v4 — market-entry для правдивой оценки edge.

Ключевые изменения v4:
- [FIX] entry_mode: 'market' | 'limit_close' | 'limit_tp_sl'
        - 'market': вход на close дня T, выход на close дня T+5
          → устраняет adverse selection от limit-fill
        - 'limit_close': старая логика limit-входа без TP/SL
        - 'limit_tp_sl': старая логика с TP/SL (с close-tiebreaker)
- [FIX] SL+TP конфликт: tiebreaker по знаку real_C
        (close up → TP первый, close down → SL первый)
- [FIX] Диагностика fill-rate отдельно для LONG/SHORT
- [FIX] trading_days: автовыбор из samples_per_ticker
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from ml.decision_layer import SIG_BUY, SIG_SELL

MAX_DAILY_MOVE = 0.15       # 15% для 5-барного окна
MIN_TRADE_MOVE = 0.001
ANNUALIZER     = 252


def _clip_preds(ohlc):
    r = ohlc.copy()
    r[:, 1] = np.clip(ohlc[:, 1],  0.,  MAX_DAILY_MOVE)
    r[:, 2] = np.clip(ohlc[:, 2], -MAX_DAILY_MOVE, 0.)
    r[:, 3] = np.clip(ohlc[:, 3], -MAX_DAILY_MOVE, MAX_DAILY_MOVE)
    return r


def composite_signal(p_dir, cls_probs, ohlc_pred):
    p_buy  = cls_probs[:, 0]
    p_sell = cls_probs[:, 2]
    dC     = np.clip(ohlc_pred[:, 3], -MAX_DAILY_MOVE, MAX_DAILY_MOVE)
    sig = (
          0.50 * (p_dir - 0.5) * 2
        + 0.30 * (p_buy - p_sell)
        + 0.20 * dC / MAX_DAILY_MOVE
    )
    return np.clip(sig, -1., 1.)


# ────────────────────────────────────────────────────────────
# Sharpe
# ────────────────────────────────────────────────────────────

def _sharpe_per_trade(pnl, trading_days):
    if len(pnl) < 2: return 0.0
    years = max(trading_days / ANNUALIZER, 1/ANNUALIZER)
    tpy   = len(pnl) / years
    mu, s = pnl.mean(), pnl.std(ddof=1)
    return float(mu / s * np.sqrt(tpy)) if s > 1e-12 else 0.0


def _sharpe_daily(trades, trading_days, total_samples):
    if not trades or trading_days < 2: return 0.0
    K = max(total_samples / trading_days, 1.0)
    daily = np.zeros(trading_days, dtype=np.float64)
    for tr in trades:
        d = min(int(tr['t'] / K), trading_days - 1)
        daily[d] += tr['pnl_capital']
    mu, s = daily.mean(), daily.std(ddof=1)
    return float(mu / s * np.sqrt(ANNUALIZER)) if s > 1e-12 else 0.0

# ────────────────────────────────────────────────────────────
# SL+TP conflict resolver (для 5-барного горизонта)
# ────────────────────────────────────────────────────────────

def _resolve_sltp_conflict(is_long, entry_delta, sl_price, tp_price,
                            real_C, sl_mult, tp_mult,
                            mode='bm_formula', rng=None):
    """Разрешает одновременный SL+TP hit за future_bars период.

    modes:
      'sl_first'    — консервативный, всегда SL (pessimistic bound)
      'bm_formula'  — P(TP first) = sl_mult / (sl_mult + tp_mult) ← recommended
      'coin_flip'   — 50/50 neutral
      'close_based' — по знаку close (BIAS, только для сравнения)
    """
    def _pnl_tp():
        return (tp_price - entry_delta) if is_long else (entry_delta - tp_price)
    def _pnl_sl():
        return (sl_price - entry_delta) if is_long else (entry_delta - sl_price)

    if mode == 'sl_first':
        return _pnl_sl(), 'SL_worst'

    if mode == 'close_based':
        close_favorable = (real_C >= entry_delta) if is_long else (real_C <= entry_delta)
        if close_favorable:
            return _pnl_tp(), 'TP_close'
        return _pnl_sl(), 'SL_close'

    if mode == 'bm_formula':
        p_tp = sl_mult / (sl_mult + tp_mult)
    else:  # coin_flip
        p_tp = 0.5

    if rng is None:
        rng = np.random
    if rng.random() < p_tp:
        return _pnl_tp(), 'TP_bm'
    return _pnl_sl(), 'SL_bm'
# ────────────────────────────────────────────────────────────
# Симуляция
# ────────────────────────────────────────────────────────────

def simulate_strategy(
    p_dir, cls_probs, ohlc_pred_raw, ohlc_true_raw, y_true,
    entry_mode:       str   = 'market',
    entry_depth:      float = 0.5,
    tp_mult:          float = 1.5,
    sl_mult:          float = 1.0,
    signal_threshold: float = 0.25,
    max_position_pct: float = 0.02,
    fee:              float = 0.001,
    use_composite:    bool  = True,
    min_pred_move:    float = 0.003,
    conflict_mode:    str   = 'bm_formula',   # <<< NEW
):
    ohlc_pred = _clip_preds(ohlc_pred_raw)
    ohlc_true = ohlc_true_raw

    sig = (composite_signal(p_dir, cls_probs, ohlc_pred)
           if use_composite else (p_dir - 0.5) * 2)

    trades = []
    n_long_signals  = 0; n_long_filled  = 0
    n_short_signals = 0; n_short_filled = 0

    for t in range(len(p_dir)):
        s = sig[t]
        if abs(s) < signal_threshold:
            continue

        pred_H, pred_L, pred_C = ohlc_pred[t, 1], ohlc_pred[t, 2], ohlc_pred[t, 3]
        real_H, real_L, real_C = ohlc_true[t, 1], ohlc_true[t, 2], ohlc_true[t, 3]

        if abs(pred_C) < min_pred_move:
            continue

        position_size = max_position_pct * min(abs(s), 1.0)

        # ═══ MARKET MODE — вход на close_prev, выход на real_C ═══
        if entry_mode == 'market':
            if s > 0:
                n_long_signals += 1
                entry_delta = 0.0
                gross_pnl   = real_C - entry_delta   # PnL LONG = real_C
                exit_type   = 'close_long'
                n_long_filled += 1
            else:
                n_short_signals += 1
                entry_delta = 0.0
                gross_pnl   = entry_delta - real_C   # PnL SHORT = -real_C
                exit_type   = 'close_short'
                n_short_filled += 1

        # ═══ LIMIT + CLOSE EXIT — вход на откате, без TP/SL ═══
        elif entry_mode == 'limit_close':
            if s > 0:
                n_long_signals += 1
                entry_delta = pred_L * entry_depth
                if abs(entry_delta) < MIN_TRADE_MOVE: continue
                if real_L > entry_delta: continue
                n_long_filled += 1
                gross_pnl = real_C - entry_delta
                exit_type = 'close_long'
            else:
                n_short_signals += 1
                entry_delta = pred_H * entry_depth
                if abs(entry_delta) < MIN_TRADE_MOVE: continue
                if real_H < entry_delta: continue
                n_short_filled += 1
                gross_pnl = entry_delta - real_C
                exit_type = 'close_short'

        # ═══ LIMIT + TP/SL (close-tiebreaker) ═══
        elif entry_mode == 'limit_tp_sl':
            if s > 0:
                n_long_signals += 1
                entry_delta = pred_L * entry_depth
                if abs(entry_delta) < MIN_TRADE_MOVE: continue
                if real_L > entry_delta: continue
                n_long_filled += 1

                ed = abs(entry_delta)
                sl_price = entry_delta - ed * sl_mult
                tp_price = entry_delta + ed * tp_mult
                sl_hit = real_L <= sl_price
                tp_hit = real_H >= tp_price

                if sl_hit and tp_hit:
                    gross_pnl, exit_type = _resolve_sltp_conflict(
                        is_long=True, entry_delta=entry_delta,
                        sl_price=sl_price, tp_price=tp_price, real_C=real_C,
                        sl_mult=sl_mult, tp_mult=tp_mult,
                        mode=conflict_mode,
                        rng=np.random.RandomState(t))
                elif tp_hit:
                    gross_pnl = tp_price - entry_delta; exit_type = 'TP'
                elif sl_hit:
                    gross_pnl = sl_price - entry_delta; exit_type = 'SL'
                else:
                    gross_pnl = real_C - entry_delta;    exit_type = 'close'

            else:   # SHORT
                n_short_signals += 1
                entry_delta = pred_H * entry_depth
                if abs(entry_delta) < MIN_TRADE_MOVE: continue
                if real_H < entry_delta: continue
                n_short_filled += 1

                ed = abs(entry_delta)
                sl_price = entry_delta + ed * sl_mult
                tp_price = entry_delta - ed * tp_mult
                sl_hit = real_H >= sl_price
                tp_hit = real_L <= tp_price

                if sl_hit and tp_hit:
                    gross_pnl, exit_type = _resolve_sltp_conflict(
                        is_long=False, entry_delta=entry_delta,
                        sl_price=sl_price, tp_price=tp_price, real_C=real_C,
                        sl_mult=sl_mult, tp_mult=tp_mult,
                        mode=conflict_mode,
                        rng=np.random.RandomState(t + 1_000_000))
                elif tp_hit:
                    gross_pnl = entry_delta - tp_price; exit_type = 'TP'
                elif sl_hit:
                    gross_pnl = entry_delta - sl_price; exit_type = 'SL'
                else:
                    gross_pnl = entry_delta - real_C;    exit_type = 'close'
        else:
            raise ValueError(f'Unknown entry_mode: {entry_mode}')

        net_pnl_pct = gross_pnl - 2 * fee
        pnl_capital = net_pnl_pct * position_size
        if abs(pnl_capital) > 0.05: continue

        trades.append({
            't': int(t),
            'direction': 'LONG' if s > 0 else 'SHORT',
            'signal': float(s), 'size': float(position_size),
            'entry_delta': float(entry_delta),
            'gross_pnl': float(gross_pnl),
            'net_pnl_pct': float(net_pnl_pct),
            'pnl_capital': float(pnl_capital),
            'exit': exit_type, 'true_cls': int(y_true[t]),
        })

    # Диагностика fill rate
    diag = {
        'long_fill_rate':  n_long_filled  / max(n_long_signals, 1),
        'short_fill_rate': n_short_filled / max(n_short_signals, 1),
        'n_signals_long':  n_long_signals,
        'n_signals_short': n_short_signals,
    }
    return trades, diag


# ────────────────────────────────────────────────────────────
# Sprint 11.2 D5 — quantile-driven execution
# ────────────────────────────────────────────────────────────

def simulate_quantile_strategy(
    p_dir, cls_probs, quantile_pred_atr, ohlc_true_pct, atr_ratio, y_true,
    *,
    signal_threshold: float = 0.25,
    max_position_pct: float = 0.02,
    fee:              float = 0.001,
    use_composite:    bool  = True,
    future_bars:      int   = 5,
    entry_q_idx:      int   = 0,   # 0 = q10 (pessimistic price), 1 = q50, 2 = q90
    exit_q_idx:       int   = 2,   # для LONG TP — H_q90; для SHORT TP — L_q10
    sl_buf_mult:      float = 1.5, # SL = entry * sl_buf_mult по другую сторону
):
    """Sprint 11.2 — D5_quantile execution. Использует quantile_pred напрямую.

    Layout quantile_pred_atr [N, 60] = [O || H || L || C], каждый chunk [3*fb]:
        [q10 × fb | q50 × fb | q90 × fb]
    Значения в ATR-norm — умножаются на atr_ratio*sqrt(fb) для % через `norm_factor`.

    Логика (entry_q_idx=0, exit_q_idx=2):
      LONG  (s>0): entry = L_q10[t+1] (pessimistic low — низкая цена входа)
                   TP    = H_q90[t+fb-1] (90-й перцентиль макс. за горизонт)
                   SL    = L_q10[t+fb-1] * sl_buf_mult (худший low)
                   fill: real_L[t+1..t+fb] <= entry в любом баре
      SHORT (s<0): entry = H_q90[t+1] (optimistic high)
                   TP    = L_q10[t+fb-1]
                   SL    = H_q90[t+fb-1] * sl_buf_mult

    Q-9 закрыт (O/C coverage 28-38%), но L/H остаются основными квантилями
    для entry/exit (H/L coverage 45-48% > O/C). Используем L для LONG entry,
    H для SHORT entry. Symmetric.
    """
    fb       = future_bars
    n_q      = 3                                                # q10, q50, q90
    chunk    = n_q * fb                                          # 15
    B        = quantile_pred_atr.shape[0]
    # Per-channel layout
    quant    = quantile_pred_atr.reshape(B, 4, n_q, fb)          # [N, 4, 3, fb]
    O, H, L, C = quant[:, 0], quant[:, 1], quant[:, 2], quant[:, 3]   # каждый [N, 3, fb]

    # norm_factor приводит из ATR-norm в % (как и `ohlc_pred_pct`)
    norm = (atr_ratio.astype(np.float64) * np.sqrt(fb))[:, None]       # [N, 1]

    # Точечные предсказания (берём q50) для composite_signal — модель направления
    ohlc_pred_q50_pct = np.stack(
        [O[:, 1, 0] * atr_ratio,    # ΔO t+1 (q50, не пишется в общий норматив с fb)
         H[:, 1, 0] * atr_ratio,
         L[:, 1, 0] * atr_ratio,
         C[:, 1, -1] * (atr_ratio * np.sqrt(fb))], axis=1)             # ΔC t+fb scaled
    sig = (composite_signal(p_dir, cls_probs, ohlc_pred_q50_pct)
           if use_composite else (p_dir - 0.5) * 2)

    trades = []
    n_long_signals = n_long_filled = 0
    n_short_signals = n_short_filled = 0

    for t in range(B):
        s = sig[t]
        if abs(s) < signal_threshold:
            continue
        nf = float(norm[t, 0])

        if s > 0:
            n_long_signals += 1
            # entry: L_q[entry_q_idx] на t+1, в %
            entry_delta = float(L[t, entry_q_idx, 0]) * nf
            if abs(entry_delta) < MIN_TRADE_MOVE:
                continue
            # Fill check: actual real_L (t+1..t+fb) пробил entry?
            # ohlc_true_pct содержит агрегированные ΔO, ΔH, ΔL, ΔC за весь горизонт fb.
            # Используем real_L (samplewise low за период) как proxy.
            real_L_pct = float(ohlc_true_pct[t, 2])
            if real_L_pct > entry_delta:
                continue
            n_long_filled += 1
            # TP — H_q[exit_q_idx] за t+fb (берём последний бар окна — оптимистично)
            tp_price = float(H[t, exit_q_idx, -1]) * nf
            # SL — L_q[0] на горизонте, но с buffer (pessimistic floor)
            sl_price = float(L[t, 0, -1]) * nf * sl_buf_mult
            real_H_pct = float(ohlc_true_pct[t, 1])
            real_C_pct = float(ohlc_true_pct[t, 3])
            sl_hit = real_L_pct <= sl_price
            tp_hit = real_H_pct >= tp_price
            if sl_hit and tp_hit:
                # tie-break по близости — кто ближе к entry по target
                rng = np.random.RandomState(t)
                # BM-formula: P(TP) = |sl − entry| / (|sl − entry| + |tp − entry|)
                ds = abs(sl_price - entry_delta); dt = abs(tp_price - entry_delta)
                p_tp = ds / (ds + dt + 1e-9)
                if rng.random() < p_tp:
                    gross_pnl = tp_price - entry_delta; exit_type = 'TP_bm'
                else:
                    gross_pnl = sl_price - entry_delta; exit_type = 'SL_bm'
            elif tp_hit:
                gross_pnl = tp_price - entry_delta; exit_type = 'TP'
            elif sl_hit:
                gross_pnl = sl_price - entry_delta; exit_type = 'SL'
            else:
                gross_pnl = real_C_pct - entry_delta; exit_type = 'close'

        else:
            n_short_signals += 1
            entry_delta = float(H[t, exit_q_idx, 0]) * nf
            if abs(entry_delta) < MIN_TRADE_MOVE:
                continue
            real_H_pct = float(ohlc_true_pct[t, 1])
            if real_H_pct < entry_delta:
                continue
            n_short_filled += 1
            tp_price = float(L[t, entry_q_idx, -1]) * nf
            sl_price = float(H[t, exit_q_idx, -1]) * nf * sl_buf_mult
            real_L_pct = float(ohlc_true_pct[t, 2])
            real_C_pct = float(ohlc_true_pct[t, 3])
            sl_hit = real_H_pct >= sl_price
            tp_hit = real_L_pct <= tp_price
            if sl_hit and tp_hit:
                rng = np.random.RandomState(t + 1_000_000)
                ds = abs(sl_price - entry_delta); dt = abs(tp_price - entry_delta)
                p_tp = ds / (ds + dt + 1e-9)
                if rng.random() < p_tp:
                    gross_pnl = entry_delta - tp_price; exit_type = 'TP_bm'
                else:
                    gross_pnl = entry_delta - sl_price; exit_type = 'SL_bm'
            elif tp_hit:
                gross_pnl = entry_delta - tp_price; exit_type = 'TP'
            elif sl_hit:
                gross_pnl = entry_delta - sl_price; exit_type = 'SL'
            else:
                gross_pnl = entry_delta - real_C_pct; exit_type = 'close'

        position_size = max_position_pct * min(abs(s), 1.0)
        net_pnl_pct   = gross_pnl - 2 * fee
        pnl_capital   = net_pnl_pct * position_size
        if abs(pnl_capital) > 0.05:
            continue
        trades.append({
            't': int(t),
            'direction': 'LONG' if s > 0 else 'SHORT',
            'signal': float(s), 'size': float(position_size),
            'entry_delta': float(entry_delta),
            'gross_pnl': float(gross_pnl),
            'net_pnl_pct': float(net_pnl_pct),
            'pnl_capital': float(pnl_capital),
            'exit': exit_type, 'true_cls': int(y_true[t]),
        })

    diag = {
        'long_fill_rate':  n_long_filled  / max(n_long_signals, 1),
        'short_fill_rate': n_short_filled / max(n_short_signals, 1),
        'n_signals_long':  n_long_signals,
        'n_signals_short': n_short_signals,
    }
    return trades, diag


# ────────────────────────────────────────────────────────────
# Sprint 11.3 D7 — Chronos quantile-band directional + calibrated TP/SL
# ────────────────────────────────────────────────────────────

def simulate_chronos_strategy(
    chronos_q10, chronos_q50, chronos_q90,    # [N, fb] ATR-norm relative ΔC
    has_chronos,                               # [N] bool
    ohlc_true_pct,                             # [N, fb*4] row-major OHLC future
    atr_ratio,                                 # [N]
    y_true,
    *,
    fb: int = 5,
    rr_ratio: float = 2.0,     # risk-reward: TP/SL distance ratio
    fee: float = 0.001,
    max_position_pct: float = 0.02,
    close_only: bool = False,  # True = exit на close, без TP/SL (чистый directional test)
    invert: str = "none",      # "none" | "long_only" | "short_only" | "both"
):
    """D7_chronos: использует Chronos quantile band как direction signal.

    Direction:
      c_q10, c_q50, c_q90 — относительные ΔC в ATR-norm на t+fb−1
      - q10 > 0  → 80%+ prob close выше → LONG
      - q90 < 0  → 80%+ prob close ниже → SHORT
      - иначе    → HOLD (band straddles zero)

    Exit:
      close_only=True   → market entry, выход на close дня t+fb (чистый directional)
      close_only=False  → TP = q50 (median expected), SL = −TP/rr_ratio (2:1 RR)
                          В отличие от старой версии, SL гарантированно на противоположной
                          стороне от entry: для LONG отрицательный, для SHORT положительный.
    """
    trades = []
    N = len(chronos_q50)
    norm = (atr_ratio.astype(np.float64) * np.sqrt(fb))
    n_signals = 0; n_filled = 0

    # Используем квантиль за t+fb−1 (последний бар горизонта)
    last_bar = fb - 1

    # Реальный ΔC за горизонт — последний C в окне
    ohlc_3d = ohlc_true_pct.reshape(N, fb, 4)
    real_C_last = ohlc_3d[:, last_bar, 3]   # ΔC в %
    real_H_last = ohlc_3d[:, last_bar, 1]   # ΔH
    real_L_last = ohlc_3d[:, last_bar, 2]   # ΔL

    for t in range(N):
        if not has_chronos[t]:
            continue

        # Chronos quantiles в ATR-norm → переводим в % (как ohlc_true_pct)
        q10_atr = float(chronos_q10[t, last_bar])
        q50_atr = float(chronos_q50[t, last_bar])
        q90_atr = float(chronos_q90[t, last_bar])
        if not (np.isfinite(q10_atr) and np.isfinite(q50_atr) and np.isfinite(q90_atr)):
            continue

        nf = float(norm[t])
        q10_pct = q10_atr * nf
        q50_pct = q50_atr * nf
        q90_pct = q90_atr * nf

        # Directional decision из chronos-band
        if q10_pct > 0:
            direction = "LONG"
        elif q90_pct < 0:
            direction = "SHORT"
        else:
            continue   # band straddles 0 → HOLD

        # Sprint 11.3 D7c: проверка anti-signal hypothesis — инвертируем direction.
        # Сохраняем оригинал чтобы избежать cascade-flip (LONG→SHORT→LONG).
        orig = direction
        if invert in ("long_only", "both") and orig == "LONG":
            direction = "SHORT"
        elif invert in ("short_only", "both") and orig == "SHORT":
            direction = "LONG"

        n_signals += 1
        position_size = max_position_pct   # fixed размер (Chronos calibrated, без sizing)

        rh = float(real_H_last[t])
        rl = float(real_L_last[t])
        rc = float(real_C_last[t])

        if direction == "LONG":
            n_filled += 1
            entry_delta = 0.0   # market
            if close_only:
                gross_pnl = rc; exit_type = "close_long"
            else:
                tp_price = q50_pct                       # > 0
                sl_price = -abs(q50_pct) / rr_ratio      # ВСЕГДА отрицательный (под entry)
                tp_hit = rh >= tp_price
                sl_hit = rl <= sl_price
                if sl_hit and tp_hit:
                    ds = abs(sl_price); dt = abs(tp_price)
                    rng = np.random.RandomState(t + 7_000_000)
                    p_tp = ds / (ds + dt + 1e-9)
                    if rng.random() < p_tp:
                        gross_pnl = tp_price; exit_type = "TP_bm"
                    else:
                        gross_pnl = sl_price; exit_type = "SL_bm"
                elif tp_hit:
                    gross_pnl = tp_price; exit_type = "TP"
                elif sl_hit:
                    gross_pnl = sl_price; exit_type = "SL"
                else:
                    gross_pnl = rc; exit_type = "close"
        else:   # SHORT
            n_filled += 1
            entry_delta = 0.0
            if close_only:
                gross_pnl = -rc; exit_type = "close_short"
            else:
                tp_price = q50_pct                       # < 0 (т.к. q90 < 0 → q50 < 0)
                sl_price = abs(q50_pct) / rr_ratio       # ВСЕГДА положительный (над entry)
                tp_hit = rl <= tp_price                   # цена пошла вниз достаточно
                sl_hit = rh >= sl_price                   # цена пошла ВВЕРХ выше SL → стоп
                if sl_hit and tp_hit:
                    ds = abs(sl_price); dt = abs(tp_price)
                    rng = np.random.RandomState(t + 8_000_000)
                    p_tp = ds / (ds + dt + 1e-9)
                    if rng.random() < p_tp:
                        gross_pnl = -tp_price; exit_type = "TP_bm"
                    else:
                        gross_pnl = -sl_price; exit_type = "SL_bm"
                elif tp_hit:
                    gross_pnl = -tp_price; exit_type = "TP"
                elif sl_hit:
                    gross_pnl = -sl_price; exit_type = "SL"
                else:
                    gross_pnl = -rc; exit_type = "close"

        net_pnl_pct = gross_pnl - 2 * fee
        pnl_capital = net_pnl_pct * position_size
        if abs(pnl_capital) > 0.05:
            continue
        trades.append({
            't': int(t),
            'direction': direction,
            'signal': float(q50_atr),    # для контекста — chronos median
            'size': float(position_size),
            'entry_delta': float(entry_delta),
            'gross_pnl': float(gross_pnl),
            'net_pnl_pct': float(net_pnl_pct),
            'pnl_capital': float(pnl_capital),
            'exit': exit_type, 'true_cls': int(y_true[t]),
        })

    diag = {
        'long_fill_rate':  1.0 if n_signals else 0.0,
        'short_fill_rate': 1.0 if n_signals else 0.0,
        'n_signals_long':  sum(1 for tr in trades if tr['direction'] == 'LONG'),
        'n_signals_short': sum(1 for tr in trades if tr['direction'] == 'SHORT'),
    }
    return trades, diag


# ────────────────────────────────────────────────────────────
# Sprint 2: симуляция на основе DecisionLayer
# ────────────────────────────────────────────────────────────

def simulate_decision_strategy(
    decision_signal:  np.ndarray,    # [N] 0=BUY, 1=HOLD, 2=SELL
    decision_conf:    np.ndarray,    # [N] confidence (used for sizing)
    mfe_mae_pred:     np.ndarray,    # [N, 4] mfe_l, mae_l, mfe_s, mae_s (доли)
    ohlc_true_pct:    np.ndarray,    # [N, 4] real ΔO ΔH ΔL ΔC (доли)
    y_true:           np.ndarray,    # [N] для exit диагностики
    fee:              float = 0.001,
    max_position_pct: float = 0.02,
    use_predicted_tp_sl: bool = True,
):
    """Decision-aware backtest.

    Решение от DecisionLayer (BUY/HOLD/SELL). Размер позиции
    масштабируется по confidence (clip[0,1]). Для long-входа TP=pred_mfe_long,
    SL=-pred_mae_long; иначе exit на close. Симметрично для short.
    """
    trades = []
    n_long_signals = n_long_filled = 0
    n_short_signals = n_short_filled = 0
    n_dropped_oversize = 0   # B-6: счётчик молча отброшенных сделок |pnl|>5%

    for t in range(len(decision_signal)):
        sig = int(decision_signal[t])
        if sig == 1:   # HOLD
            continue

        conf = float(np.clip(decision_conf[t], 0.0, 1.0))
        position_size = max_position_pct * max(conf, 1e-3)

        real_H = float(ohlc_true_pct[t, 1])
        real_L = float(ohlc_true_pct[t, 2])
        real_C = float(ohlc_true_pct[t, 3])

        mfe_l = float(mfe_mae_pred[t, 0])
        mae_l = float(mfe_mae_pred[t, 1])
        mfe_s = float(mfe_mae_pred[t, 2])
        mae_s = float(mfe_mae_pred[t, 3])

        rng = np.random.default_rng(t)   # детерминировано per-sample

        if sig == 0:   # BUY
            n_long_signals += 1
            n_long_filled  += 1   # market entry → всегда заполнено
            entry_delta = 0.0
            if use_predicted_tp_sl and mfe_l > 1e-4 and mae_l > 1e-4:
                tp = entry_delta + mfe_l
                sl = entry_delta - mae_l
                tp_hit = real_H >= tp
                sl_hit = real_L <= sl
                if tp_hit and sl_hit:
                    # bm_formula: P(TP first) = sl_mult/(sl_mult+tp_mult) ≈ mae/(mae+mfe)
                    p_tp = mae_l / (mae_l + mfe_l)
                    if rng.random() < p_tp:
                        gross = tp - entry_delta
                        exit_t = 'TP_bm'
                    else:
                        gross = sl - entry_delta
                        exit_t = 'SL_bm'
                elif tp_hit:
                    gross = tp - entry_delta
                    exit_t = 'TP'
                elif sl_hit:
                    gross = sl - entry_delta
                    exit_t = 'SL'
                else:
                    gross = real_C - entry_delta
                    exit_t = 'close'
            else:
                gross = real_C - entry_delta
                exit_t = 'close'

        elif sig == 2:   # SELL
            n_short_signals += 1
            n_short_filled  += 1
            entry_delta = 0.0
            if use_predicted_tp_sl and mfe_s > 1e-4 and mae_s > 1e-4:
                tp = entry_delta - mfe_s
                sl = entry_delta + mae_s
                tp_hit = real_L <= tp
                sl_hit = real_H >= sl
                if tp_hit and sl_hit:
                    p_tp = mae_s / (mae_s + mfe_s)
                    if rng.random() < p_tp:
                        gross = entry_delta - tp
                        exit_t = 'TP_bm'
                    else:
                        gross = entry_delta - sl
                        exit_t = 'SL_bm'
                elif tp_hit:
                    gross = entry_delta - tp
                    exit_t = 'TP'
                elif sl_hit:
                    gross = entry_delta - sl
                    exit_t = 'SL'
                else:
                    gross = entry_delta - real_C
                    exit_t = 'close'
            else:
                gross = entry_delta - real_C
                exit_t = 'close'
        else:
            continue

        net_pnl_pct = gross - 2 * fee
        pnl_capital = net_pnl_pct * position_size
        if abs(pnl_capital) > 0.05:
            # Аномально крупный PnL (>5% капитала) — типично гэп / стейл цена.
            # Отбрасываем, но считаем для диагностики (B-6).
            n_dropped_oversize += 1
            continue

        trades.append({
            't': int(t),
            'direction':   'LONG' if sig == 0 else 'SHORT',
            'signal':      conf,
            'size':        float(position_size),
            'entry_delta': 0.0,
            'gross_pnl':   float(gross),
            'net_pnl_pct': float(net_pnl_pct),
            'pnl_capital': float(pnl_capital),
            'exit':        exit_t,
            'true_cls':    int(y_true[t]),
        })

    diag = {
        'long_fill_rate':  n_long_filled  / max(n_long_signals, 1),
        'short_fill_rate': n_short_filled / max(n_short_signals, 1),
        'n_signals_long':  n_long_signals,
        'n_signals_short': n_short_signals,
        'n_dropped_oversize': n_dropped_oversize,
    }
    if n_dropped_oversize > 0:
        n_total = n_long_signals + n_short_signals
        print(f"  [decision_strategy] отброшено |pnl|>5%: "
              f"{n_dropped_oversize}/{n_total} ({n_dropped_oversize/max(n_total,1):.1%})")
    return trades, diag


# ────────────────────────────────────────────────────────────
# Sprint 1.5: Intraday Refinement Simulation
# ────────────────────────────────────────────────────────────

def simulate_intraday_refinement(
    model,
    device,
    morning_signal: dict,
    imgs_batch: dict,
    nums_batch: dict,
    ctx_batch,
    hourly_batch,
    intraday_feats_full,    # [1, T, 11] full day features
    known_hours_list=(1, 3, 7),
    cancel_threshold: float = 0.6,
):
    """Симулирует авторегрессионное уточнение прогноза в течение дня.

    На каждом шаге `known_hours` обрезает маску и пересчитывает прогноз.
    Возвращает dict {hour: refined_signal} где refined_signal содержит:
        dir_prob, extremes_pred, action ('HOLD'|'UPDATE'|'CANCEL')

    cancel_threshold: если dir_prob упал ниже morning_dir_prob * threshold -> CANCEL.
    """
    import torch

    model.eval()
    results = {0: morning_signal}

    morning_dir = float(morning_signal.get('dir_prob', 0.5))

    with torch.no_grad():
        for known_h in known_hours_list:
            T = intraday_feats_full.shape[1]
            mask = torch.zeros(1, T, device=device)
            mask[:, :min(known_h, T)] = 1.0

            feats = intraday_feats_full.to(device)

            out = model(
                imgs_batch, nums_batch, ctx_batch, hourly_batch,
                intraday_feats=feats, intraday_mask=mask,
            )

            # 8-tuple: logits, ohlc, aux, dir_logit, intraday_pred, econ, next_hr_pred, extremes
            if len(out) >= 8:
                _, _, _, dir_logit, _, econ_p, _, extremes = out
            elif len(out) >= 6:
                _, _, _, dir_logit, _, econ_p = out[:6]
                extremes = None
            else:
                _, _, _, dir_logit = out[:4]
                econ_p = None; extremes = None

            dir_prob = float(torch.sigmoid(dir_logit).mean().item())
            extremes_np = extremes.cpu().numpy() if extremes is not None else None

            action = 'UPDATE'
            if dir_prob < morning_dir * cancel_threshold:
                action = 'CANCEL'

            results[known_h] = {
                'dir_prob': dir_prob,
                'extremes': extremes_np,
                'econ': {k: v.cpu().numpy() for k, v in econ_p.items()} if econ_p else None,
                'action': action,
                'known_hours': known_h,
            }

    return results


# ────────────────────────────────────────────────────────────
# Sprint 8.3: Path-aware execution simulator
# ────────────────────────────────────────────────────────────

def simulate_path_aware_strategy(
    decision_signal:   np.ndarray,    # [N] 0=BUY, 1=HOLD, 2=SELL
    decision_conf:     np.ndarray,    # [N]
    ohlc_pred_pct:     np.ndarray,    # [N, 4+] ΔO ΔH ΔL ΔC — денормализованные
    ohlc_true_pct:     np.ndarray,    # [N, 4+]
    y_true:            np.ndarray,    # [N]
    high_first_prob:   np.ndarray,    # [N] P(high before low) из HighLowOrderHead
    fee:               float = 0.001,
    max_position_pct:  float = 0.02,
    high_first_thr:    float = 0.55,  # порог для high_first режима
    low_first_thr:     float = 0.45,  # порог для low_first режима (round-trip)
):
    """Sprint 8.3: path-aware execution использует high_first_prob.

    Execution modes:
      low_first  (hf_prob < low_first_thr):  entry at pred_low, exit at pred_high (round-trip).
      high_first (hf_prob > high_first_thr): entry at open, exit at pred_high (carry overnight).
      uncertain  (между порогами):           skip — HOLD.
    """
    trades = []
    n_low_first = n_high_first = n_uncertain = n_hold = 0

    for t in range(len(decision_signal)):
        sig = int(decision_signal[t])
        if sig not in (SIG_BUY, SIG_SELL):
            n_hold += 1
            continue

        hf_p = float(high_first_prob[t])
        conf = float(np.clip(decision_conf[t], 0.0, 1.0))
        position_size = max_position_pct * max(conf, 1e-3)

        real_H = float(ohlc_true_pct[t, 1])
        real_L = float(ohlc_true_pct[t, 2])
        real_C = float(ohlc_true_pct[t, 3])

        pred_H = float(ohlc_pred_pct[t, 1])   # ΔHigh (положительный)
        pred_L = float(ohlc_pred_pct[t, 2])   # ΔLow  (отрицательный)

        if hf_p < low_first_thr:
            # low_first: ожидаем сначала low, потом high
            # BUY: entry = pred_low, exit = pred_high (round-trip in day)
            n_low_first += 1
            if sig == SIG_BUY:
                entry_delta = pred_L     # лимит вниз от close
                tp_price    = pred_H
                entry_hit   = real_L <= entry_delta  # low дошёл до входа
                tp_hit      = entry_hit and real_H >= tp_price
                if not entry_hit:
                    continue  # лимитка не исполнена
                gross = (tp_price if tp_hit else real_C) - entry_delta
                exit_t = 'TP' if tp_hit else 'close'
            else:  # SELL
                entry_delta = pred_H
                tp_price    = pred_L
                entry_hit   = real_H >= entry_delta
                tp_hit      = entry_hit and real_L <= tp_price
                if not entry_hit:
                    continue
                gross = entry_delta - (tp_price if tp_hit else real_C)
                exit_t = 'TP' if tp_hit else 'close'

        elif hf_p > high_first_thr:
            # high_first: ожидаем сначала high, потом low → BUY на следующий день
            n_high_first += 1
            if sig == SIG_BUY:
                # entry at open (delta=0), exit at pred_high (TP) или close
                entry_delta = 0.0
                tp_price    = pred_H
                tp_hit      = real_H >= tp_price
                gross = (tp_price if tp_hit else real_C) - entry_delta
                exit_t = 'TP' if tp_hit else 'close'
            else:  # SELL short при high_first — entry at open, TP at pred_low
                entry_delta = 0.0
                tp_price    = pred_L
                tp_hit      = real_L <= tp_price
                gross = entry_delta - (tp_price if tp_hit else real_C)
                exit_t = 'TP' if tp_hit else 'close'
        else:
            # Неопределённость → HOLD
            n_uncertain += 1
            continue

        net_pnl_pct = gross - 2 * fee
        pnl_capital = net_pnl_pct * position_size
        if abs(pnl_capital) > 0.05:
            continue  # аномальный PnL — отбрасываем

        trades.append({
            't':           int(t),
            'direction':   'LONG' if sig == SIG_BUY else 'SHORT',
            'signal':      conf,
            'size':        float(position_size),
            'entry_delta': entry_delta,
            'gross_pnl':   float(gross),
            'net_pnl_pct': float(net_pnl_pct),
            'pnl_capital': float(pnl_capital),
            'exit':        exit_t,
            'true_cls':    int(y_true[t]),
            'mode':        'low_first' if hf_p < low_first_thr else 'high_first',
        })

    diag = {
        'n_low_first':  n_low_first,
        'n_high_first': n_high_first,
        'n_uncertain':  n_uncertain,
        'n_hold':       n_hold,
    }
    print(f'  [path-aware] low_first={n_low_first} high_first={n_high_first} '
          f'uncertain={n_uncertain} hold={n_hold}')
    return trades, diag


# ────────────────────────────────────────────────────────────
# Анализ
# ────────────────────────────────────────────────────────────

def analyze_trades(trades, label='', trading_days=250, total_samples=0, diag=None):
    if not trades:
        print(f'  [{label}] Нет сделок')
        return {}

    pnl   = np.array([t['pnl_capital'] for t in trades])
    gross = np.array([t['gross_pnl']   for t in trades])
    dirs  = np.array([t['direction']   for t in trades])
    exits = np.array([t['exit']        for t in trades])
    n     = len(trades)

    # Conditional WR: на модельной уверенности
    wr_long  = (gross[dirs == 'LONG']  > 0).mean() if (dirs == 'LONG').any()  else 0
    wr_short = (gross[dirs == 'SHORT'] > 0).mean() if (dirs == 'SHORT').any() else 0

    total_sum  = pnl.sum()
    total_comp = (1 + pnl).prod() - 1
    sharpe_pt  = _sharpe_per_trade(pnl, trading_days)
    sharpe_d   = _sharpe_daily(trades, trading_days, total_samples)

    eq = np.cumprod(1 + pnl)
    dd = (eq - np.maximum.accumulate(eq)) / (np.maximum.accumulate(eq) + 1e-9)
    max_dd = dd.min()

    print(f'\n  ═══ {label} ═══')
    print(f'  Сделок:                 {n}')
    print(f'  LONG / SHORT:           {(dirs=="LONG").sum()} / {(dirs=="SHORT").sum()}')
    if diag:
        print(f'  Fill rate LONG:         {diag["long_fill_rate"]:.1%} '
              f'({diag["n_signals_long"]} signals)')
        print(f'  Fill rate SHORT:        {diag["short_fill_rate"]:.1%} '
              f'({diag["n_signals_short"]} signals)')
    print(f'  Win rate OVERALL:       {(pnl > 0).mean():.2%}')
    print(f'  Win rate LONG:          {wr_long:.2%}   ← должно быть ~p_UP_correct')
    print(f'  Win rate SHORT:         {wr_short:.2%}  ← должно быть ~p_DOWN_correct')
    print(f'  Avg gross/trade:        {gross.mean() * 100:+.3f}%')
    print(f'  Total return (sum):     {total_sum * 100:+.2f}%')
    print(f'  Total return (compound):{total_comp * 100:+.2f}%')
    print(f'  Sharpe (per-trade):     {sharpe_pt:+.2f}')
    print(f'  Sharpe (daily bucket):  {sharpe_d:+.2f}  ← main')
    print(f'  Max drawdown:           {max_dd * 100:+.2f}%')

    # Exit breakdown
    if n > 0:
        ex, cnts = np.unique(exits, return_counts=True)
        print(f'  Exits: ' + ', '.join(f'{e}={c/n:.1%}' for e, c in zip(ex, cnts)))

    return {
        'n': n, 'win_rate': float((pnl>0).mean()),
        'wr_long': float(wr_long), 'wr_short': float(wr_short),
        'total_return_sum':  float(total_sum * 100),
        'total_return_comp': float(total_comp * 100),
        'sharpe_per_trade':  sharpe_pt,
        'sharpe_daily':      sharpe_d,
        'max_dd_pct':        float(max_dd * 100),
    }


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', default='ml/ensemble/ensemble_predictions.npz')
    parser.add_argument('--days', type=int, default=None)
    parser.add_argument('--future-bars', type=int, default=5)
    parser.add_argument('--n-tickers', type=int, default=20)
    # Sprint 8: новые режимы
    parser.add_argument('--intraday',   action='store_true',
                        help='Sprint 8.4: intraday cancellation через hourly predictions')
    parser.add_argument('--path-aware', action='store_true',
                        help='Sprint 8.3: path-aware execution (требует high_first_prob в npz)')
    args = parser.parse_args()

    data = np.load(args.preds, allow_pickle=False)
    p_dir     = data['dir_prob']
    cls_probs = data['cls_probs']
    ohlc_pred = data['ohlc_pred']
    ohlc_true = data['ohlc_test']
    y_true    = data['y_test']
    atr_ratio = data['atr_ratio'] if 'atr_ratio' in data.files else np.full(len(p_dir), 0.018)

    # Sprint 5 bull=OFF: загружаем regime если присутствует в npz
    _regime_key = 'test_regime' if 'test_regime' in data.files else ('regime' if 'regime' in data.files else None)
    _regime_arr = data[_regime_key] if _regime_key else None

    # Sprint 2: decision-aware ключи (опционально)
    has_decision = 'decision_signal' in data.files
    if has_decision:
        decision_signal = data['decision_signal']
        decision_conf   = data['decision_confidence']
        mfe_mae_pred    = data['mfe_mae_pred']
        print(f'  ✓ Sprint 2: decision_signal присутствует '
              f'(BUY={int((decision_signal==0).sum())} '
              f'HOLD={int((decision_signal==1).sum())} '
              f'SELL={int((decision_signal==2).sum())})')
        # Sprint 5 bull=OFF: перевести bull-сэмплы в HOLD
        if _regime_arr is not None:
            _bull = (_regime_arr == 2)
            decision_signal = decision_signal.copy()
            decision_signal[_bull] = 1
            print(f'  ✓ bull=OFF: {int(_bull.sum())} bull→HOLD  '
                  f'(BUY={int((decision_signal==0).sum())} '
                  f'SELL={int((decision_signal==2).sum())} остались)')

    # Sprint 8.1: extremes (pred_high, pred_low[, high_low_logit]) — опционально
    extremes_pred = data['extremes_pred'] if 'extremes_pred' in data.files else None
    if extremes_pred is not None:
        print(f'  ✓ Sprint 8.1: extremes_pred присутствует shape={extremes_pred.shape}  '
              f'range_mean={float((extremes_pred[:,0] - extremes_pred[:,1]).mean()):.4f}')
        # Sprint 8.2: если extremes_pred имеет 3 столбца, извлекаем high_first_prob
        if extremes_pred.shape[1] >= 3 and 'high_first_prob' not in data.files:
            _hf = (1.0 / (1.0 + np.exp(-extremes_pred[:, 2].astype(np.float64)))).astype(np.float32)
            data = dict(data)
            data['high_first_prob'] = _hf
            print(f'  ✓ Sprint 8.2: high_first_prob извлечён из extremes_pred[:,2]')
        # Если есть extremes и econ-ключи — пересчитаем decision с учётом extremes
        if has_decision and 'fill_prob' in data.files and 'edge_pred' in data.files:
            from ml.decision_layer import DecisionLayer, costs_from_config
            dl_ext = DecisionLayer(costs_from_config())
            dec_ext = dl_ext.decide_numpy(
                dir_prob  = p_dir,
                mfe_mae   = mfe_mae_pred,
                fill_prob = data['fill_prob'],
                edge_pred = data['edge_pred'],
                extremes  = extremes_pred,
            )
            decision_signal_ext = dec_ext['signal']
            decision_conf_ext   = dec_ext['confidence']
            # bull=OFF для ext сигнала
            if _regime_arr is not None:
                decision_signal_ext = decision_signal_ext.copy()
                decision_signal_ext[_regime_arr == 2] = 1
            n_buy_ext  = int((decision_signal_ext == 0).sum())
            n_hold_ext = int((decision_signal_ext == 1).sum())
            n_sell_ext = int((decision_signal_ext == 2).sum())
            print(f'  ✓ Decision с extremes: BUY={n_buy_ext} HOLD={n_hold_ext} SELL={n_sell_ext}')

    total_samples = len(p_dir)

    if args.days is not None:
        trading_days = args.days
    else:
        # 14669 / 20 ≈ 733 дней на тикер
        trading_days = max(50, total_samples // args.n_tickers)
        print(f'  auto trading_days = {trading_days} (≈ {total_samples}/{args.n_tickers})')

    norm_factor = (atr_ratio * np.sqrt(args.future_bars))[:, None]
    ohlc_pred_pct = ohlc_pred * norm_factor
    ohlc_true_pct = ohlc_true * norm_factor

    print(f'\n  📦 Samples={total_samples}, days={trading_days}, '
          f'future_bars={args.future_bars}')
    print(f'  pred ΔC: {ohlc_pred_pct[:,3].mean()*100:+.2f}% ± '
          f'{ohlc_pred_pct[:,3].std()*100:.2f}%')
    print(f'  true ΔC: {ohlc_true_pct[:,3].mean()*100:+.2f}% ± '
          f'{ohlc_true_pct[:,3].std()*100:.2f}%')

    # Главное — 3 стратегии для сравнения
    _base_d = dict(signal_threshold=0.25, entry_depth=0.5,
                   tp_mult=1.5, sl_mult=1.0)
    strategies = {
        'A_market':           ('market',      dict(signal_threshold=0.25)),
        'C_limit_close':      ('limit_close', dict(signal_threshold=0.25,
                                                    entry_depth=0.5)),
        'D1_sl_first':        ('limit_tp_sl', {**_base_d, 'conflict_mode': 'sl_first'}),
        'D2_bm_formula':      ('limit_tp_sl', {**_base_d, 'conflict_mode': 'bm_formula'}),
        'D3_coin_flip':       ('limit_tp_sl', {**_base_d, 'conflict_mode': 'coin_flip'}),
        'D4_close_based':     ('limit_tp_sl', {**_base_d, 'conflict_mode': 'close_based'}),
    }

    print(f'\n{"═"*70}\n  СРАВНЕНИЕ РЕЖИМОВ ИСПОЛНЕНИЯ\n{"═"*70}')
    stats = {}
    all_trades = {}
    for name, (mode, kw) in strategies.items():
        trades, diag = simulate_strategy(
            p_dir, cls_probs, ohlc_pred_pct, ohlc_true_pct, y_true,
            entry_mode=mode, **kw)
        stats[name] = analyze_trades(
            trades, label=f'{name} [{mode}]',
            trading_days=trading_days, total_samples=total_samples, diag=diag)
        all_trades[name] = trades

    # Sprint 11.2 D5 — quantile-driven execution (если quantile_pred в npz)
    if 'quantile_pred' in data.files:
        qp = data['quantile_pred']
        # Ожидаем shape [N, 4 * 3 * fb]; для текущего fb=5 это [N, 60].
        if qp.shape[1] == 4 * 3 * args.future_bars:
            print(f'\n{"═"*70}\n  D5 quantile variants (Sprint 11.2 — Q-9 unlocked)\n{"═"*70}')
            print(f'  quantile_pred shape={qp.shape}, future_bars={args.future_bars}')
            # D5a (default): L_q10 entry — самый жёсткий, fill ~6%
            # D5b: L_q50 entry — median, fill выше, win-rate ниже
            # D5c: D5a + sl_buf_mult=2.0 — шире SL, реже выбиваемся
            d5_variants = [
                ('D5a_q10',     dict(entry_q_idx=0, exit_q_idx=2, sl_buf_mult=1.5),
                                'L_q10 entry, H_q90 TP, SL×1.5'),
                ('D5b_q50',     dict(entry_q_idx=1, exit_q_idx=2, sl_buf_mult=1.5),
                                'L_q50 entry (median), H_q90 TP, SL×1.5'),
                ('D5c_wide_sl', dict(entry_q_idx=0, exit_q_idx=2, sl_buf_mult=2.0),
                                'L_q10 entry, H_q90 TP, SL×2.0 (шире)'),
            ]
            for name, kw, descr in d5_variants:
                trades, diag = simulate_quantile_strategy(
                    p_dir, cls_probs, qp, ohlc_true_pct, atr_ratio, y_true,
                    signal_threshold=0.25,
                    future_bars=args.future_bars,
                    **kw,
                )
                stats[name] = analyze_trades(
                    trades, label=f'{name} [{descr}]',
                    trading_days=trading_days, total_samples=total_samples, diag=diag)
                all_trades[name] = trades
        else:
            print(f'\n  ⚠️  D5 skip: quantile_pred shape {qp.shape} != ожидаемое '
                  f'[N, {4*3*args.future_bars}]')

    # Sprint 11.3 D7 — Chronos quantile-band directional execution
    if all(k in data.files for k in ('chronos_close_q10', 'chronos_close_q50',
                                       'chronos_close_q90', 'has_chronos')):
        print(f'\n{"═"*70}\n  D7_chronos (Sprint 11.3 — foundation model directional)\n{"═"*70}')
        c_q10 = data['chronos_close_q10']
        c_q50 = data['chronos_close_q50']
        c_q90 = data['chronos_close_q90']
        has_c = data['has_chronos'].astype(bool)
        n_c = int(has_c.sum())
        print(f'  Chronos coverage: {n_c}/{len(has_c)} ({n_c/len(has_c)*100:.1f}%)')

        # D7a: TP/SL — TP=q50, SL=-q50/RR (2:1 risk-reward)
        d7a_trades, d7a_diag = simulate_chronos_strategy(
            c_q10, c_q50, c_q90, has_c, ohlc_true_pct, atr_ratio, y_true,
            fb=args.future_bars, rr_ratio=2.0, close_only=False,
        )
        stats['D7a_chronos_tpsl'] = analyze_trades(
            d7a_trades, label='D7a_chronos [TP=q50, SL=−q50/2 (RR 2:1)]',
            trading_days=trading_days, total_samples=total_samples, diag=d7a_diag)
        all_trades['D7a_chronos_tpsl'] = d7a_trades

        # D7b: close-only — чистый directional test без TP/SL
        d7b_trades, d7b_diag = simulate_chronos_strategy(
            c_q10, c_q50, c_q90, has_c, ohlc_true_pct, atr_ratio, y_true,
            fb=args.future_bars, close_only=True,
        )
        stats['D7b_chronos_close'] = analyze_trades(
            d7b_trades, label='D7b_chronos [close-only, чистый directional]',
            trading_days=trading_days, total_samples=total_samples, diag=d7b_diag)
        all_trades['D7b_chronos_close'] = d7b_trades

        # D7c: LONG-band инвертируется в SHORT (тест anti-signal hypothesis для LONG)
        d7c_trades, d7c_diag = simulate_chronos_strategy(
            c_q10, c_q50, c_q90, has_c, ohlc_true_pct, atr_ratio, y_true,
            fb=args.future_bars, close_only=True, invert="long_only",
        )
        stats['D7c_chronos_invLONG'] = analyze_trades(
            d7c_trades, label='D7c_chronos [LONG→SHORT инверсия, close-only]',
            trading_days=trading_days, total_samples=total_samples, diag=d7c_diag)
        all_trades['D7c_chronos_invLONG'] = d7c_trades

        # D7d: полная инверсия (LONG↔SHORT) — последняя проверка
        d7d_trades, d7d_diag = simulate_chronos_strategy(
            c_q10, c_q50, c_q90, has_c, ohlc_true_pct, atr_ratio, y_true,
            fb=args.future_bars, close_only=True, invert="both",
        )
        stats['D7d_chronos_invBOTH'] = analyze_trades(
            d7d_trades, label='D7d_chronos [LONG↔SHORT full invert, close-only]',
            trading_days=trading_days, total_samples=total_samples, diag=d7d_diag)
        all_trades['D7d_chronos_invBOTH'] = d7d_trades

    # ── Sprint 2: decision-aware стратегия ─────────────────────────────
    if has_decision:
        print(f'\n{"═"*70}\n  E. DECISION LAYER (Sprint 2)\n{"═"*70}')
        # E1: с предсказанными TP/SL (mfe_pred/mae_pred)
        dec_trades, dec_diag = simulate_decision_strategy(
            decision_signal=decision_signal,
            decision_conf=decision_conf,
            mfe_mae_pred=mfe_mae_pred,
            ohlc_true_pct=ohlc_true_pct,
            y_true=y_true,
            use_predicted_tp_sl=True,
        )
        stats['E_decision_layer'] = analyze_trades(
            dec_trades, label='E_decision_layer',
            trading_days=trading_days, total_samples=total_samples, diag=dec_diag)
        all_trades['E_decision_layer'] = dec_trades

        # E2 (B-15): close-only режим — соответствует expectancy из decision_sweep.
        # Tест: если TP/SL miscalibrated (mfe/mae=~2.5% при close-движении ~0.1%),
        # close-only показывает чистый PnL от направленного сигнала.
        dec_trades_co, dec_diag_co = simulate_decision_strategy(
            decision_signal=decision_signal,
            decision_conf=decision_conf,
            mfe_mae_pred=mfe_mae_pred,
            ohlc_true_pct=ohlc_true_pct,
            y_true=y_true,
            use_predicted_tp_sl=False,
        )
        stats['E2_decision_close_only'] = analyze_trades(
            dec_trades_co, label='E2_decision_close_only',
            trading_days=trading_days, total_samples=total_samples, diag=dec_diag_co)
        all_trades['E2_decision_close_only'] = dec_trades_co

        # E3 Sprint 8.1: decision с extremes (если доступны)
        if extremes_pred is not None and 'decision_signal_ext' in dir():
            print(f'\n{"═"*70}\n  E3. DECISION + EXTREMES (Sprint 8.1)\n{"═"*70}')
            e3_trades, e3_diag = simulate_decision_strategy(
                decision_signal=decision_signal_ext,
                decision_conf=decision_conf_ext,
                mfe_mae_pred=mfe_mae_pred,
                ohlc_true_pct=ohlc_true_pct,
                y_true=y_true,
                use_predicted_tp_sl=False,
            )
            stats['E3_extremes_boost'] = analyze_trades(
                e3_trades, label='E3_extremes_boost',
                trading_days=trading_days, total_samples=total_samples, diag=e3_diag)
            all_trades['E3_extremes_boost'] = e3_trades

    # ── Sprint 8.3: Path-aware execution ───────────────────────────────
    if args.path_aware and has_decision:
        print(f'\n{"═"*70}\n  F. PATH-AWARE EXECUTION (Sprint 8.3)\n{"═"*70}')
        if 'high_first_prob' not in data.files:
            print('  Пропуск: high_first_prob не найден в npz.')
            print('  Требуется rebuild V3 с HighLowOrderHead (Sprint 8.2).')
        else:
            hf_prob = data['high_first_prob']
            f_trades, f_diag = simulate_path_aware_strategy(
                decision_signal=decision_signal,
                decision_conf=decision_conf,
                ohlc_pred_pct=ohlc_pred_pct,
                ohlc_true_pct=ohlc_true_pct,
                y_true=y_true,
                high_first_prob=hf_prob,
            )
            stats['F_path_aware'] = analyze_trades(
                f_trades, label='F_path_aware',
                trading_days=trading_days, total_samples=total_samples, diag=f_diag)
            all_trades['F_path_aware'] = f_trades

    # ── Sprint 8.4: Intraday cancellation ──────────────────────────────
    if args.intraday and has_decision:
        print(f'\n{"═"*70}\n  G. INTRADAY CANCELLATION (Sprint 8.4)\n{"═"*70}')
        hourly_npz = 'ml/ensemble/hourly_test_predictions.npz'
        meta_npz   = 'ml/ensemble/meta_features.npz'
        if not os.path.exists(hourly_npz) and not os.path.exists(meta_npz):
            print('  Пропуск: hourly_test_predictions.npz и meta_features.npz не найдены.')
            print('  Запустите py -m ml.meta_ensemble для генерации hourly предсказаний.')
        else:
            # Применяем простой фильтр: cancel BUY если hourly_dir_prob < 0.4
            # для соответствующего (date, ticker) из hourly npz
            h_path = hourly_npz if os.path.exists(hourly_npz) else meta_npz
            try:
                hd = np.load(h_path, allow_pickle=False)
                print(f'  Загружено: {h_path}  shape keys: {list(hd.files)[:6]}')

                test_dates    = data['test_dates']    if 'test_dates'    in data.files else None
                test_tickers  = data['test_tickers']  if 'test_tickers'  in data.files else None
                hourly_dates  = hd['dates']   if 'dates'   in hd.files else None
                hourly_tickers= hd['tickers'] if 'tickers' in hd.files else None
                hourly_dir    = hd['dir_prob'] if 'dir_prob' in hd.files else None

                if (test_dates is not None and test_tickers is not None
                        and hourly_dates is not None and hourly_tickers is not None
                        and hourly_dir is not None):
                    # Строим словарь (date, ticker) → hourly dir_prob
                    key_to_hdir = {
                        (str(hourly_dates[i]), str(hourly_tickers[i])): float(hourly_dir[i])
                        for i in range(len(hourly_dates))
                    }
                    cancel_mask = np.zeros(len(decision_signal), dtype=bool)
                    for i in range(len(decision_signal)):
                        sig = int(decision_signal[i])
                        if sig == 1:
                            continue
                        key = (str(test_dates[i]), str(test_tickers[i]))
                        h_prob = key_to_hdir.get(key, None)
                        if h_prob is None:
                            continue
                        # Отменяем BUY если hourly говорит DOWN (<0.4)
                        if sig == 0 and h_prob < 0.4:
                            cancel_mask[i] = True
                        # Отменяем SELL если hourly говорит UP (>0.6)
                        elif sig == 2 and h_prob > 0.6:
                            cancel_mask[i] = True

                    dec_sig_filtered = decision_signal.copy()
                    dec_sig_filtered[cancel_mask] = 1   # HOLD
                    n_cancelled = int(cancel_mask.sum())
                    print(f'  Отменено сделок: {n_cancelled} / {int((decision_signal!=1).sum())}')

                    g_trades, g_diag = simulate_decision_strategy(
                        decision_signal=dec_sig_filtered,
                        decision_conf=decision_conf,
                        mfe_mae_pred=mfe_mae_pred,
                        ohlc_true_pct=ohlc_true_pct,
                        y_true=y_true,
                        use_predicted_tp_sl=False,
                    )
                    stats['G_intraday_cancel'] = analyze_trades(
                        g_trades, label='G_intraday_cancel',
                        trading_days=trading_days, total_samples=total_samples, diag=g_diag)
                    all_trades['G_intraday_cancel'] = g_trades
                else:
                    print('  Пропуск: не найдены поля dates/tickers/dir_prob в hourly npz.')
            except Exception as e:
                print(f'  Ошибка загрузки hourly: {e}')

    # Equity plot
    plt.figure(figsize=(14, 7))
    for name, trades in all_trades.items():
        if not trades: continue
        pnl = np.array([t['pnl_capital'] for t in trades])
        eq = np.cumprod(1 + pnl)
        sh = _sharpe_daily(trades, trading_days, total_samples)
        plt.plot(eq, label=f'{name} | SR={sh:.2f} | ret={(eq[-1]-1)*100:+.1f}%')
    plt.axhline(1.0, color='gray', ls='--', alpha=0.5)
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig('ml/backtest_equity.png', dpi=100)

    with open('ml/backtest_report.json', 'w') as f:
        json.dump(stats, f, indent=2, default=float)
    print(f'\n  📝 → ml/backtest_report.json')


if __name__ == '__main__':
    main()