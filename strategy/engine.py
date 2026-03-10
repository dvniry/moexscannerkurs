"""Движок стратегии — генерирует сигналы и статистику."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd

from strategy.base import Strategy, Signal


@dataclass
class EngineResult:
    signals:     List[Signal]
    in_position: bool           # сейчас открыта позиция?
    last_entry:  Optional[Signal] = None
    stats:       dict = field(default_factory=dict)


def run_strategy(strategy: Strategy, df: pd.DataFrame) -> EngineResult:
    """
    Прогоняем стратегию по DataFrame.
    Возвращает список сигналов BUY/SELL/STOP.
    """
    entry_series = strategy.entry(df)
    exit_series  = strategy.exit(df)
    stop_series  = strategy.stop(df)

    signals:    List[Signal] = []
    in_position = False
    entry_price = 0.0
    wins = losses = 0

    # unix timestamps
    try:
        idx = pd.to_datetime(df.index)
        if getattr(idx, 'tz', None) is not None:
            idx = idx.tz_convert('UTC').tz_localize(None)
        times = (idx.astype('datetime64[ns]').astype('int64') // 10**9).tolist()
    except Exception:
        times = list(range(len(df)))

    for i in range(len(df)):
        t     = times[i]
        price = float(df['close'].iloc[i])
        size  = strategy.position_size()

        if not in_position:
            if entry_series.iloc[i]:
                signals.append(Signal(
                    time=t, price=price,
                    action="BUY", size=size,
                    reason="entry"
                ))
                in_position = True
                entry_price = price

        else:
            # Сначала стоп, потом нормальный выход
            if stop_series.iloc[i]:
                signals.append(Signal(
                    time=t, price=price,
                    action="STOP", size=size,
                    reason="stop-loss"
                ))
                in_position = False
                if price < entry_price:
                    losses += 1
                else:
                    wins += 1

            elif exit_series.iloc[i]:
                signals.append(Signal(
                    time=t, price=price,
                    action="SELL", size=size,
                    reason="exit"
                ))
                in_position = False
                if price > entry_price:
                    wins += 1
                else:
                    losses += 1

    total_trades = wins + losses
    last_entry   = next(
        (s for s in reversed(signals) if s.action == "BUY"), None
    )

    stats = {
        "total_signals": len(signals),
        "total_trades":  total_trades,
        "wins":          wins,
        "losses":        losses,
        "winrate":       round(wins / total_trades * 100, 1) if total_trades else 0,
    }

    profits = []
    losses_list = []
    entry_p = 0.0
    for s in signals:
        if s.action == 'BUY':
            entry_p = s.price
        elif s.action in ('SELL', 'STOP') and entry_p:
            pct = (s.price - entry_p) / entry_p * 100
            if pct > 0: profits.append(pct)
            else:       losses_list.append(pct)

    stats = {
        "total_signals": len(signals),
        "total_trades":  total_trades,
        "wins":          wins,
        "losses":        losses,
        "winrate":       round(wins / total_trades * 100, 1) if total_trades else 0,
        "avg_profit":    round(sum(profits) / len(profits), 2)    if profits      else 0,
        "avg_loss":      round(sum(losses_list) / len(losses_list), 2) if losses_list else 0,
    }


    return EngineResult(
        signals=signals,
        in_position=in_position,
        last_entry=last_entry,
        stats=stats,
    )
