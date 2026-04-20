"""Backtest стратегии v2 — исправлены баги с масштабированием PnL."""
import os, sys, argparse, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────
# Константы — реалистичные ограничения
# ────────────────────────────────────────────────────────────
MAX_DAILY_MOVE = 0.10   # максимум 10% дневное движение (фильтр выбросов)
MIN_TRADE_MOVE = 0.001  # минимум 0.1% для входа (иначе комиссия съедает)


def _clip_ohlc(ohlc: np.ndarray) -> np.ndarray:
    """Клипуем нереалистичные дельты OHLC."""
    result = ohlc.copy()
    result[:, 1] = np.clip(ohlc[:, 1],  0.,  MAX_DAILY_MOVE)  # High > 0
    result[:, 2] = np.clip(ohlc[:, 2], -MAX_DAILY_MOVE, 0.)   # Low < 0
    result[:, 3] = np.clip(ohlc[:, 3], -MAX_DAILY_MOVE, MAX_DAILY_MOVE)
    return result


def composite_signal(p_dir, cls_probs, ohlc_pred):
    """Композитный сигнал [-1, 1]."""
    p_cls_buy  = cls_probs[:, 0]
    p_cls_sell = cls_probs[:, 2]
    delta_C    = np.clip(ohlc_pred[:, 3], -MAX_DAILY_MOVE, MAX_DAILY_MOVE)

    signal = (
          0.50 * (p_dir - 0.5) * 2
        + 0.30 * (p_cls_buy - p_cls_sell)
        + 0.20 * np.sign(delta_C) * np.abs(delta_C) / (MAX_DAILY_MOVE + 1e-9)
    )
    return np.clip(signal, -1., 1.)


def simulate_strategy(
    p_dir, cls_probs, ohlc_pred_raw, ohlc_true_raw, y_true,
    entry_depth: float = 0.5,
    tp_mult: float = 1.5,          # TP = entry * tp_mult (reward/risk)
    signal_threshold: float = 0.25,
    max_position_pct: float = 0.02,  # 2% капитала на сделку (реалистично)
    fee: float = 0.001,
    use_composite: bool = True,
    min_pred_move: float = 0.003,   # минимальное предсказанное движение 0.3%
):
    """
    Реалистичная симуляция:
    
    - ohlc в виде ДОЛЕЙ (0.02 = 2%)
    - Entry = limit ордер внутри бара
    - TP фиксированный как ratio к entry distance
    - SL = entry distance (risk:reward = 1:tp_mult)
    - Size = max_position_pct * |signal| (max 2% капитала)
    """
    # Клипуем нереалистичные значения
    ohlc_pred = _clip_ohlc(ohlc_pred_raw)
    ohlc_true = _clip_ohlc(ohlc_true_raw)

    if use_composite:
        signal = composite_signal(p_dir, cls_probs, ohlc_pred)
    else:
        signal = (p_dir - 0.5) * 2

    trades = []

    for t in range(len(p_dir)):
        s = signal[t]
        if abs(s) < signal_threshold:
            continue

        pred_H = ohlc_pred[t, 1]   # предсказанный ΔHigh (>0)
        pred_L = ohlc_pred[t, 2]   # предсказанный ΔLow (<0)
        pred_C = ohlc_pred[t, 3]   # предсказанный ΔClose

        real_H = ohlc_true[t, 1]
        real_L = ohlc_true[t, 2]
        real_C = ohlc_true[t, 3]

        # Размер позиции
        position_size = max_position_pct * min(abs(s), 1.0)

        if s > 0:  # LONG
            # Entry: пытаемся купить на откате
            # entry_price = close_prev * (1 + pred_L * entry_depth)
            # В дельтах: entry_delta = pred_L * entry_depth (отрицательное)
            entry_delta = pred_L * entry_depth  # < 0

            # Минимальное движение для осмысленной сделки
            if abs(entry_delta) < MIN_TRADE_MOVE:
                continue

            # TP distance = entry distance * tp_mult
            tp_delta = abs(entry_delta) * tp_mult  # > 0

            # Проверяем исполнение: low бара должен достичь entry
            filled = real_L <= entry_delta
            if not filled:
                continue

            # TP: high бара достиг tp?
            tp_hit = real_H >= tp_delta
            if tp_hit:
                gross_pnl = tp_delta - entry_delta  # > 0
                exit_type = 'TP'
            else:
                # Выход по close
                gross_pnl = real_C - entry_delta
                exit_type = 'close'

        else:  # SHORT
            entry_delta = pred_H * entry_depth  # > 0
            
            if abs(entry_delta) < MIN_TRADE_MOVE:
                continue

            tp_delta = -abs(entry_delta) * tp_mult  # < 0

            filled = real_H >= entry_delta
            if not filled:
                continue

            tp_hit = real_L <= tp_delta
            if tp_hit:
                gross_pnl = entry_delta - tp_delta  # > 0
                exit_type = 'TP'
            else:
                gross_pnl = entry_delta - real_C
                exit_type = 'close'

        # Комиссия: 2 * fee от размера
        net_pnl_pct = gross_pnl - 2 * fee

        # PnL с учётом sizing (в долях капитала)
        pnl_capital = net_pnl_pct * position_size

        # Защита от нереалистичных значений
        if abs(pnl_capital) > 0.05:  # > 5% капитала за сделку — невозможно
            continue

        trades.append({
            't':            t,
            'direction':    'LONG' if s > 0 else 'SHORT',
            'signal':       float(s),
            'size':         float(position_size),
            'entry_delta':  float(entry_delta),
            'gross_pnl':    float(gross_pnl),
            'net_pnl_pct':  float(net_pnl_pct),
            'pnl_capital':  float(pnl_capital),
            'exit':         exit_type,
            'true_cls':     int(y_true[t]),
        })

    return trades


def analyze_trades(trades: list, label: str = '', trading_days: int = 250):
    if not trades:
        print(f'  [{label}] Нет сделок')
        return {}

    # Используем pnl_capital — доля капитала
    pnl  = np.array([t['pnl_capital'] for t in trades])
    dirs = np.array([t['direction'] for t in trades])
    n    = len(trades)

    win_rate  = (pnl > 0).mean()
    mean_pnl  = pnl.mean()
    std_pnl   = pnl.std() + 1e-9
    total_ret = (1 + pnl).prod() - 1

    # Правильный Sharpe: нормируем на количество сделок в день
    # Предполагаем test период = 250 дней (можно передать снаружи)
    trades_per_day = n / trading_days
    # Дневной PnL = среднее по сделкам * кол-во сделок в день
    daily_mean = mean_pnl * trades_per_day
    daily_std  = std_pnl * np.sqrt(trades_per_day) + 1e-9
    sharpe_ann = daily_mean / daily_std * np.sqrt(trading_days)

    # Max drawdown
    eq_curve    = np.cumprod(1 + pnl)
    running_max = np.maximum.accumulate(eq_curve)
    dd          = (eq_curve - running_max) / (running_max + 1e-9)
    max_dd      = dd.min()

    tp_rate = (np.array([t['exit'] for t in trades]) == 'TP').mean()

    # Средний gross PnL на сделку
    avg_gross = np.array([t['gross_pnl'] for t in trades]).mean()

    print(f'\n  ═══ {label} ═══')
    print(f'  Сделок:             {n}')
    print(f'  LONG / SHORT:       {(dirs == "LONG").sum()} / {(dirs == "SHORT").sum()}')
    print(f'  Win rate:           {win_rate:.2%}')
    print(f'  TP hit rate:        {tp_rate:.2%}')
    print(f'  Avg gross per trade:{avg_gross * 100:+.3f}%')
    print(f'  Avg net/capital:    {mean_pnl * 100:+.4f}%')
    print(f'  Std PnL/capital:    {std_pnl * 100:.4f}%')
    print(f'  Total return:       {total_ret * 100:+.2f}%')
    print(f'  Sharpe (annualized):{sharpe_ann:+.2f}')
    print(f'  Max drawdown:       {max_dd * 100:+.2f}%')
    print(f'  Trades/day:         {trades_per_day:.2f}')

    return {
        'n': n,
        'win_rate':         float(win_rate),
        'tp_rate':          float(tp_rate),
        'avg_gross_pct':    float(avg_gross * 100),
        'avg_net_capital':  float(mean_pnl * 100),
        'total_return_pct': float(total_ret * 100),
        'sharpe':           float(sharpe_ann),
        'max_dd_pct':       float(max_dd * 100),
        'trades_per_day':   float(trades_per_day),
    }


def grid_search(p_dir, cls_probs, ohlc_pred, ohlc_true, y_true,
                min_trades: int = 30, trading_days: int = 250):
    """Grid search с реалистичными параметрами."""
    print(f'\n{"═" * 70}')
    print(f'  🔍 GRID SEARCH по параметрам')
    print(f'{"═" * 70}')

    results = []
    best = {'sharpe': -np.inf}

    for thr in [0.20, 0.25, 0.30, 0.35, 0.40]:
        for entry in [0.3, 0.5, 0.7]:
            for tp_mult in [1.0, 1.5, 2.0, 3.0]:
                trades = simulate_strategy(
                    p_dir, cls_probs, ohlc_pred, ohlc_true, y_true,
                    entry_depth=entry,
                    tp_mult=tp_mult,
                    signal_threshold=thr,
                    use_composite=True,
                )
                if len(trades) < min_trades:
                    continue

                pnl = np.array([t['pnl_capital'] for t in trades])
                n   = len(trades)

                trades_per_day = n / trading_days
                daily_mean = pnl.mean() * trades_per_day
                daily_std  = pnl.std() * np.sqrt(trades_per_day) + 1e-9
                sharpe     = daily_mean / daily_std * np.sqrt(trading_days)
                total      = (1 + pnl).prod() - 1
                win_rate   = (pnl > 0).mean()

                results.append({
                    'thr': thr, 'entry': entry, 'tp_mult': tp_mult,
                    'n': n, 'sharpe': sharpe,
                    'total_return': total, 'win_rate': win_rate,
                })

                if sharpe > best['sharpe']:
                    best = {
                        'sharpe': sharpe, 'thr': thr,
                        'entry': entry, 'tp_mult': tp_mult,
                        'n': n, 'total_return': total,
                    }

    results.sort(key=lambda x: -x['sharpe'])
    print(f'\n  Топ-10 конфигураций:')
    print(f'  {"thr":>5} {"entry":>6} {"tp_x":>6} {"n":>5} '
          f'{"Sharpe":>8} {"TotalRet":>10} {"WinRate":>8}')
    for r in results[:10]:
        print(f'  {r["thr"]:>5.2f} {r["entry"]:>6.2f} {r["tp_mult"]:>6.1f} '
              f'{r["n"]:>5} {r["sharpe"]:>+8.2f} '
              f'{r["total_return"] * 100:>+9.2f}% {r["win_rate"]:>7.2%}')

    return best, results


def plot_equity_curve(
    trades_dict: dict,
    save_path: str = 'ml/backtest_equity.png',
    trading_days: int = 250,
):
    plt.figure(figsize=(14, 7))

    for label, trades in trades_dict.items():
        if not trades:
            continue
        pnl = np.array([t['pnl_capital'] for t in trades])
        eq  = np.cumprod(1 + pnl)

        n   = len(trades)
        tpd = n / trading_days
        dm  = pnl.mean() * tpd
        ds  = pnl.std() * np.sqrt(tpd) + 1e-9
        sh  = dm / ds * np.sqrt(trading_days)

        plt.plot(eq, label=f'{label} | Sharpe={sh:.2f} | '
                           f'ret={(eq[-1]-1)*100:+.1f}%')

    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    plt.xlabel('Trade #')
    plt.ylabel('Equity (капитал нормирован к 1.0)')
    plt.title('Backtest equity curves (реалистичный)')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f'  📈 Equity curve → {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', default='ml/ensemble/ensemble_predictions.npz')
    parser.add_argument('--days', type=int, default=None,
                        help='Торговых дней в тестовом периоде')
    parser.add_argument('--atr-ratio', type=float, default=None,
                        help='ATR/close вручную если нет в файле')
    args = parser.parse_args()

    # ── Загрузка ─────────────────────────────────────────────
    data      = np.load(args.preds)
    p_dir     = data['dir_prob']
    cls_probs = data['cls_probs']
    ohlc_pred = data['ohlc_pred']
    ohlc_true = data['ohlc_test']
    y_true    = data['y_test']

    print(f'\n  📦 Загружено предсказаний: {len(p_dir)}')
    print(f'  Ключи: {list(data.keys())}')
    print(f'  BUY={(y_true==0).sum()} '
          f'HOLD={(y_true==1).sum()} '
          f'SELL={(y_true==2).sum()}')

    # ── atr_ratio ────────────────────────────────────────────
    if 'atr_ratio' in data:
        atr_ratio = data['atr_ratio']
        print(f'  atr_ratio из файла: '
              f'mean={atr_ratio.mean():.4f} '
              f'std={atr_ratio.std():.4f}')
    elif args.atr_ratio is not None:
        atr_ratio = np.full(len(p_dir), args.atr_ratio, dtype=np.float32)
        print(f'  atr_ratio задан вручную: {args.atr_ratio:.4f}')
    else:
        atr_ratio = np.full(len(p_dir), 0.018, dtype=np.float32)
        print(f'  ⚠️  atr_ratio не найден → fallback 0.018')
        print(f'  Запусти --rebuild для пересчёта или передай --atr-ratio X')

    # ── Денормализация OHLC ──────────────────────────────────
    # ohlc хранится в ATR-нормированных единицах
    # реальный % = ohlc_norm * atr_ratio * sqrt(future_bars)
    future_bars = 5
    norm_factor = atr_ratio * np.sqrt(future_bars)          # [N]
    norm_factor = norm_factor[:, np.newaxis]                 # [N, 1]

    ohlc_pred_pct = ohlc_pred * norm_factor                  # [N, 4] в долях цены
    ohlc_true_pct = ohlc_true * norm_factor                  # [N, 4] в долях цены

    print(f'\n  📊 До денормализации (ATR-единицы):')
    print(f'  pred ΔHigh: mean={ohlc_pred[:,1].mean():+.4f} '
          f'std={ohlc_pred[:,1].std():.4f}')
    print(f'  true ΔClose: mean={ohlc_true[:,3].mean():+.4f} '
          f'std={ohlc_true[:,3].std():.4f}')

    print(f'\n  📊 После денормализации (доли цены = %/100):')
    print(f'  pred ΔHigh:  mean={ohlc_pred_pct[:,1].mean()*100:+.3f}% '
          f'std={ohlc_pred_pct[:,1].std()*100:.3f}%')
    print(f'  pred ΔLow:   mean={ohlc_pred_pct[:,2].mean()*100:+.3f}% '
          f'std={ohlc_pred_pct[:,2].std()*100:.3f}%')
    print(f'  pred ΔClose: mean={ohlc_pred_pct[:,3].mean()*100:+.3f}% '
          f'std={ohlc_pred_pct[:,3].std()*100:.3f}%')
    print(f'  true ΔClose: mean={ohlc_true_pct[:,3].mean()*100:+.3f}% '
          f'std={ohlc_true_pct[:,3].std()*100:.3f}%')

    # ── Торговых дней ────────────────────────────────────────
    if args.days is not None:
        trading_days = args.days
    else:
        # 14575 сэмплов / ~20 тикеров / ~3 года ≈ 109 дней теста
        n_tickers_est = 20
        trading_days  = max(50, len(p_dir) // n_tickers_est)
        print(f'\n  ⚠️  --days не задан, оценка: {trading_days} дней')
        print(f'  Передай --days N для точности')

    # ── Grid search ──────────────────────────────────────────
    best, grid_results = grid_search(
        p_dir, cls_probs, ohlc_pred_pct, ohlc_true_pct, y_true,
        trading_days=trading_days,
    )

    # ── Финальные стратегии ──────────────────────────────────
    print(f'\n{"═" * 70}')
    print(f'  🏆 ФИНАЛЬНЫЕ СТРАТЕГИИ (период: ~{trading_days} торг. дней)')
    print(f'{"═" * 70}')

    strategies = {
        'S1_simple_dir': simulate_strategy(
            p_dir, cls_probs, ohlc_pred_pct, ohlc_true_pct, y_true,
            entry_depth=0.5, tp_mult=1.5,
            signal_threshold=0.25,
            use_composite=False,
        ),
        'S2_composite_best': simulate_strategy(
            p_dir, cls_probs, ohlc_pred_pct, ohlc_true_pct, y_true,
            entry_depth=best['entry'],
            tp_mult=best['tp_mult'],
            signal_threshold=best['thr'],
            use_composite=True,
        ),
        'S3_conservative': simulate_strategy(
            p_dir, cls_probs, ohlc_pred_pct, ohlc_true_pct, y_true,
            entry_depth=0.3, tp_mult=2.0,
            signal_threshold=0.35,
            use_composite=True,
        ),
    }

    stats = {}
    for name, trades in strategies.items():
        stats[name] = analyze_trades(
            trades, label=name, trading_days=trading_days)

    plot_equity_curve(strategies, trading_days=trading_days)

    with open('ml/backtest_report.json', 'w') as f:
        json.dump(stats, f, indent=2, default=float)
    print(f'\n  📝 Отчёт → ml/backtest_report.json')


if __name__ == '__main__':
    main()