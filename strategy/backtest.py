"""Движок бэктеста — максимально реалистичный."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np

from strategy.base import Strategy


@dataclass
class Trade:
    entry_time:  int
    exit_time:   int
    entry_price: float
    exit_price:  float
    action:      str        # EXIT | STOP | OPEN(незакрытая)
    pnl:         float
    pnl_pct:     float
    size:        float


@dataclass
class EquityPoint:
    time:   int
    equity: float


@dataclass
class BacktestResult:
    trades:       List[Trade]
    equity_curve: List[EquityPoint]
    initial:      float
    final:        float
    total_return: float
    max_drawdown: float
    sharpe:       float
    winrate:      float
    total_trades: int
    wins:         int
    losses:       int
    avg_profit:   float
    avg_loss:     float
    best_trade:   float
    worst_trade:  float
    commission:   float


# ── Параметры реализма ─────────────────────────────────────
COMMISSION  = 0.0005   # 0.05% как в Tinkoff Sandbox
SLIPPAGE    = 0.001    # 0.10% проскальзывание (спред + рыночный импакт)


def _exec_buy_price(df: pd.DataFrame, i: int) -> float:
    """
    Реалистичная цена входа:
      - берём OPEN следующей свечи (сигнал виден только после закрытия текущей)
      - добавляем проскальзывание вверх (платим больше при покупке)
    """
    open_next = float(df['open'].iloc[i + 1])
    return open_next * (1 + SLIPPAGE)


def _exec_sell_price(df: pd.DataFrame, i: int) -> float:
    """
    Реалистичная цена выхода:
      - берём OPEN следующей свечи
      - проскальзывание вниз (получаем меньше при продаже)
    """
    open_next = float(df['open'].iloc[i + 1])
    return open_next * (1 - SLIPPAGE)


def run_backtest(
    strategy:        Strategy,
    df:              pd.DataFrame,
    initial_capital: float = 100_000.0,
) -> BacktestResult:

    entry_series = strategy.entry(df)
    exit_series  = strategy.exit(df)
    stop_series  = strategy.stop(df)

    # Unix timestamps
    try:
        idx = pd.to_datetime(df.index)
        if getattr(idx, 'tz', None) is not None:
            idx = idx.tz_convert('UTC').tz_localize(None)
        times = (idx.astype('datetime64[ns]').astype('int64') // 10**9).tolist()
    except Exception:
        times = list(range(len(df)))

    capital          = initial_capital
    in_position      = False
    entry_price      = 0.0
    entry_time       = 0
    position_val     = 0.0
    size_pct         = strategy.position_size() / 100.0
    trades:      List[Trade]       = []
    equity_curve: List[EquityPoint] = []
    total_commission = 0.0

    # -1 т.к. при сигнале на свече i исполняемся на open[i+1]
    for i in range(len(df) - 1):
        t     = times[i]
        close = float(df['close'].iloc[i])

        if not in_position:
            # Текущий капитал = просто деньги
            equity_curve.append(EquityPoint(time=t, equity=round(capital, 2)))

            if entry_series.iloc[i]:
                # Исполняем на open следующей свечи + слипедж
                exec_price   = _exec_buy_price(df, i)
                position_val = capital * size_pct
                commission   = position_val * COMMISSION

                capital -= (position_val + commission)
                total_commission += commission

                in_position  = True
                entry_price  = exec_price   # ← теперь правильно
                entry_time   = times[i + 1] # ← время реального исполнения

        else:
            # Текущая рыночная стоимость позиции (по close текущей свечи)
            current_val = position_val * (close / entry_price)
            equity_now  = capital + current_val
            equity_curve.append(EquityPoint(time=t, equity=round(equity_now, 2)))

            close_reason = None

            # Стоп имеет приоритет над выходом
            if stop_series.iloc[i]:
                close_reason = "STOP"
            elif exit_series.iloc[i]:
                close_reason = "EXIT"

            if close_reason:
                # Выходим на open следующей свечи + слипедж
                exec_exit    = _exec_sell_price(df, i)
                exit_val     = position_val * (exec_exit / entry_price)
                commission   = exit_val * COMMISSION
                pnl          = exit_val - position_val - commission
                pnl_pct      = pnl / position_val * 100

                capital          += position_val + pnl
                total_commission += commission
                in_position       = False

                trades.append(Trade(
                    entry_time  = entry_time,
                    exit_time   = times[i + 1],
                    entry_price = round(entry_price, 4),
                    exit_price  = round(exec_exit, 4),
                    action      = close_reason,
                    pnl         = round(pnl, 2),
                    pnl_pct     = round(pnl_pct, 2),
                    size        = round(position_val, 2),
                ))

    # Незакрытая позиция в конце — закрываем по последнему close
    if in_position:
        last_price = float(df['close'].iloc[-1]) * (1 - SLIPPAGE)
        exit_val   = position_val * (last_price / entry_price)
        commission = exit_val * COMMISSION
        pnl        = exit_val - position_val - commission
        pnl_pct    = pnl / position_val * 100
        capital   += position_val + pnl
        total_commission += commission
        trades.append(Trade(
            entry_time  = entry_time,
            exit_time   = times[-1],
            entry_price = round(entry_price, 4),
            exit_price  = round(last_price, 4),
            action      = "OPEN",
            pnl         = round(pnl, 2),
            pnl_pct     = round(pnl_pct, 2),
            size        = round(position_val, 2),
        ))

    # ── Метрики ───────────────────────────────────────────
    total_return = (capital - initial_capital) / initial_capital * 100

    wins_list   = [t for t in trades if t.pnl > 0]
    losses_list = [t for t in trades if t.pnl <= 0]
    wins        = len(wins_list)
    losses      = len(losses_list)
    total_t     = len(trades)
    winrate     = round(wins / total_t * 100, 1) if total_t else 0

    avg_profit  = round(sum(t.pnl_pct for t in wins_list)   / wins   if wins   else 0, 2)
    avg_loss    = round(sum(t.pnl_pct for t in losses_list) / losses if losses else 0, 2)
    best_trade  = round(max((t.pnl_pct for t in trades), default=0), 2)
    worst_trade = round(min((t.pnl_pct for t in trades), default=0), 2)

    # Макс просадка по equity curve
    equities = [e.equity for e in equity_curve]
    max_dd   = 0.0
    if equities:
        peak = equities[0]
        for e in equities:
            peak  = max(peak, e)
            dd    = (peak - e) / peak * 100 if peak else 0
            max_dd = max(max_dd, dd)

    # Sharpe по дневным доходностям equity (более корректно чем по сделкам)
    sharpe = 0.0
    if len(equities) > 2:
        eq_series = pd.Series(equities)
        daily_ret = eq_series.pct_change().dropna()
        if daily_ret.std() > 0:
            sharpe = round(float(
                daily_ret.mean() / daily_ret.std() * np.sqrt(252)
            ), 2)

    return BacktestResult(
        trades        = trades,
        equity_curve  = equity_curve,
        initial       = initial_capital,
        final         = round(capital, 2),
        total_return  = round(total_return, 2),
        max_drawdown  = round(max_dd, 2),
        sharpe        = sharpe,
        winrate       = winrate,
        total_trades  = total_t,
        wins          = wins,
        losses        = losses,
        avg_profit    = avg_profit,
        avg_loss      = avg_loss,
        best_trade    = best_trade,
        worst_trade   = worst_trade,
        commission    = round(total_commission, 2),
    )
