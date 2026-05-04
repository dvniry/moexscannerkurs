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