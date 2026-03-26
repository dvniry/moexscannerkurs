# ml/backtest_v3.py
"""Бэктест-модуль v3 — самодостаточный (без зависимости от backtest.py).

Изменения v3 (consolidation):
- StrategyConfig, Trade, direction_risk_reward, Backtester, buy_hold_benchmark
  встроены напрямую — backtest.py больше не нужен
- Добавлена поддержка hourly encoder (batch[5])
- run_backtest_v3() — единственная точка входа

Запуск:
  python -m ml.backtest_v3 --model ml/model_multiscale_v3.pt
  python -m ml.backtest_v3 --model ml/model_multiscale_v3.pt --no-hourly
"""
import os, sys, json, argparse
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.config import CFG, SCALES
from ml.labels_ohlc import ohlc_to_strategy_features

# ══════════════════════════════════════════════════════════════════
# Конфигурация стратегии
# ══════════════════════════════════════════════════════════════════

@dataclass
class StrategyConfig:
    confidence_threshold: float = 0.45
    min_risk_reward:      float = 1.2
    min_expected_move:    float = 0.005
    risk_per_trade:       float = 0.01
    max_position_pct:     float = 0.10
    max_open_positions:   int   = 5
    min_downside_floor:   float = 0.005
    commission:           float = 0.0005
    initial_capital:      float = 1_000_000.0
    hold_bars:            int   = 5


# ══════════════════════════════════════════════════════════════════
# Трейд
# ══════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    idx:          int
    direction:    str
    confidence:   float
    expected_ret: float
    risk_reward:  float
    position_size: float
    capital_used: float
    open_bar:     int
    close_bar:    int
    actual_ret:   float = 0.0
    actual_maxdd: float = 0.0
    pnl:          float = 0.0
    pnl_pct:      float = 0.0


# ══════════════════════════════════════════════════════════════════
# Direction-aware risk/reward
# ══════════════════════════════════════════════════════════════════

def direction_risk_reward(features: dict, direction: str) -> float:
    """BUY: reward = max_upside / risk = |max_downside|
       SELL: reward = |max_downside| / risk = max_upside
    """
    max_up   = features["max_upside"]
    max_down = features["max_downside"]
    if direction == "BUY":
        reward = max(max_up,        0.0)
        risk   = max(abs(max_down), 1e-9)
    else:
        reward = max(abs(max_down), 0.0)
        risk   = max(max_up,        1e-9)
    return reward / risk


# ══════════════════════════════════════════════════════════════════
# Backtester v2.1 (P&L реализуется при закрытии позиции)
# ══════════════════════════════════════════════════════════════════

class Backtester:
    def __init__(self, cfg: StrategyConfig = None):
        self.cfg          = cfg or StrategyConfig()
        self.trades:      List[Trade] = []
        self.equity_curve: List[float] = []

    def run(self,
            cls_probs:  np.ndarray,   # (N, 3)
            ohlc_pred:  np.ndarray,   # (N, F*4) или (N, F, 4)
            ohlc_true:  np.ndarray,   # (N, F, 4)
            cls_true:   np.ndarray = None):
        N = len(cls_probs); cfg = self.cfg
        capital = cfg.initial_capital
        self.equity_curve = [capital]; self.trades = []
        closing_at: dict = {}
        capital_locked = 0.0; n_open = 0

        for i in range(N):
            for trade in closing_at.pop(i, []):
                capital         += trade.pnl
                capital_locked  -= trade.position_size
                n_open          -= 1

            probs       = cls_probs[i]
            pred_class  = int(probs.argmax())
            confidence  = float(probs.max())
            self.equity_curve.append(capital)

            if pred_class == 1: continue
            if confidence < cfg.confidence_threshold: continue

            features  = ohlc_to_strategy_features(ohlc_pred[i])
            direction = "BUY" if pred_class == 0 else "SELL"

            rr = direction_risk_reward(features, direction)
            if rr < cfg.min_risk_reward: continue

            final_ret    = features["final_return"]
            expected_ret = final_ret if direction == "BUY" else -final_ret
            if abs(expected_ret) < cfg.min_expected_move: continue

            if n_open >= cfg.max_open_positions: continue
            available_pct = 1.0 - capital_locked
            if available_pct < 0.01: continue

            downside_risk = (abs(features["max_downside"]) if direction == "BUY"
                             else abs(features["max_upside"]))
            downside_risk = max(downside_risk, cfg.min_downside_floor)
            conf_scale    = (confidence - cfg.confidence_threshold) / (1.0 - cfg.confidence_threshold + 1e-9)
            position_pct  = min(cfg.risk_per_trade / downside_risk * conf_scale,
                                cfg.max_position_pct, available_pct)
            position_pct  = max(position_pct, 0.001)

            real_ohlc = ohlc_true[i]
            if direction == "BUY":
                actual_ret   = float(real_ohlc[-1, 3])
                actual_maxdd = float(real_ohlc[:, 2].min())
            else:
                actual_ret   = -float(real_ohlc[-1, 3])
                actual_maxdd = -float(real_ohlc[:, 1].max())

            net_ret      = actual_ret - 2 * cfg.commission
            capital_used = capital * position_pct
            pnl          = capital_used * net_ret

            close_bar = min(i + cfg.hold_bars, N - 1)
            trade = Trade(
                idx=i, direction=direction, confidence=confidence,
                expected_ret=expected_ret, risk_reward=rr,
                position_size=position_pct, capital_used=capital_used,
                open_bar=i, close_bar=close_bar,
                actual_ret=net_ret, actual_maxdd=actual_maxdd,
                pnl=pnl, pnl_pct=net_ret * 100,
            )
            self.trades.append(trade)
            closing_at.setdefault(close_bar, []).append(trade)
            capital_locked += position_pct; n_open += 1

        for bar_trades in closing_at.values():
            for trade in bar_trades: capital += trade.pnl
        self.equity_curve.append(capital)
        return self.compute_metrics()

    def compute_metrics(self) -> dict:
        cfg    = self.cfg
        equity = np.array(self.equity_curve)
        trades = self.trades
        if not trades: return {"error": "Нет сделок — фильтры слишком жёсткие"}

        total_return   = (equity[-1] / cfg.initial_capital - 1) * 100
        n_trades       = len(trades)
        wins           = [t for t in trades if t.pnl > 0]
        losses         = [t for t in trades if t.pnl <= 0]
        win_rate       = len(wins) / n_trades * 100
        avg_win        = float(np.mean([t.pnl for t in wins]))   if wins   else 0.0
        avg_loss       = float(np.mean([t.pnl for t in losses])) if losses else 0.0
        gross_profit   = sum(t.pnl for t in wins)                if wins   else 0.0
        gross_loss     = sum(abs(t.pnl) for t in losses)         if losses else 1e-9
        profit_factor  = gross_profit / gross_loss

        peak = equity[0]; max_dd = 0.0
        for val in equity:
            if val > peak: peak = val
            dd = (peak - val) / (peak + 1e-9)
            if dd > max_dd: max_dd = dd
        max_dd_pct = max_dd * 100

        returns = np.diff(equity) / (equity[:-1] + 1e-9)
        if len(returns) > 1:
            mean_r = np.mean(returns); std_r = np.std(returns, ddof=1)
            sharpe  = mean_r / (std_r + 1e-9) * np.sqrt(250)
            downside = returns[returns < 0]
            dstd    = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
            sortino = mean_r / (dstd + 1e-9) * np.sqrt(250)
            n_days  = max(len(returns), 1)
            annual_ret = (equity[-1] / cfg.initial_capital) ** (250 / n_days) - 1
            calmar  = annual_ret / (max_dd + 1e-9)
        else:
            sharpe = sortino = calmar = 0.0

        buys   = [t for t in trades if t.direction == "BUY"]
        sells  = [t for t in trades if t.direction == "SELL"]
        buy_wr = len([t for t in buys  if t.pnl > 0]) / max(len(buys),  1) * 100
        sel_wr = len([t for t in sells if t.pnl > 0]) / max(len(sells), 1) * 100

        return {
            "total_return_pct":     round(total_return, 2),
            "n_trades":             n_trades,
            "win_rate_pct":         round(win_rate, 1),
            "profit_factor":        round(profit_factor, 3),
            "sharpe_ratio":         round(sharpe,  3),
            "sortino_ratio":        round(sortino, 3),
            "calmar_ratio":         round(calmar,  3),
            "max_drawdown_pct":     round(max_dd_pct, 2),
            "avg_win":              round(avg_win,  2),
            "avg_loss":             round(avg_loss, 2),
            "expectancy":           round(float(np.mean([t.pnl for t in trades])), 2),
            "expectancy_pct":       round(float(np.mean([t.pnl_pct for t in trades])), 4),
            "avg_hold_days":        cfg.hold_bars,
            "final_equity":         round(float(equity[-1]), 2),
            "initial_capital":      cfg.initial_capital,
            "n_buys":               len(buys),
            "n_sells":              len(sells),
            "buy_win_rate_pct":     round(buy_wr, 1),
            "sell_win_rate_pct":    round(sel_wr, 1),
            "avg_pnl_buy":          round(float(np.mean([t.pnl for t in buys]))  if buys  else 0, 2),
            "avg_pnl_sell":         round(float(np.mean([t.pnl for t in sells])) if sells else 0, 2),
            "avg_confidence_win":   round(float(np.mean([t.confidence for t in wins]))   if wins   else 0, 4),
            "avg_confidence_loss":  round(float(np.mean([t.confidence for t in losses])) if losses else 0, 4),
            "filters": {
                "confidence_thr":     cfg.confidence_threshold,
                "min_risk_reward":    cfg.min_risk_reward,
                "min_expected_move":  cfg.min_expected_move,
                "max_open_positions": cfg.max_open_positions,
                "min_downside_floor": cfg.min_downside_floor,
            },
        }

    def print_report(self, results: dict):
        if "error" in results:
            print(f"\n  ⚠ {results['error']}"); return
        print("\n" + "=" * 60)
        print("  BACKTEST REPORT")
        print("=" * 60)
        print(f"  Капитал : {results['initial_capital']:,.0f} → {results['final_equity']:,.0f}")
        print(f"  Return  : {results['total_return_pct']:+.2f}%")
        print(f"  Sharpe  : {results['sharpe_ratio']:.3f}  Sortino: {results['sortino_ratio']:.3f}  Calmar: {results['calmar_ratio']:.3f}")
        print(f"  Max DD  : {results['max_drawdown_pct']:.2f}%")
        print(f"  Сделок  : {results['n_trades']}  WR: {results['win_rate_pct']:.1f}%  PF: {results['profit_factor']:.3f}")
        print(f"  BUY     : {results['n_buys']} (WR {results['buy_win_rate_pct']:.1f}%)")
        print(f"  SELL    : {results['n_sells']} (WR {results['sell_win_rate_pct']:.1f}%)")
        print(f"  Expect  : {results['expectancy']:+,.2f}₽ ({results['expectancy_pct']:+.4f}%)")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════════════════════════

def buy_hold_benchmark(ohlc_true: np.ndarray) -> dict:
    """Buy & Hold: покупаем каждый сэмпл, держим future_bars баров."""
    final_returns = ohlc_true[:, -1, 3]
    net_returns   = final_returns - 2 * 0.0005
    capital       = 1_000_000.0; equity = [capital]
    for r in net_returns:
        capital *= (1 + r * 0.10); equity.append(capital)
    equity    = np.array(equity)
    total_ret = (equity[-1] / equity[0] - 1) * 100
    peak = equity[0]; max_dd = 0.0
    for v in equity:
        if v > peak: peak = v
        dd = (peak - v) / (peak + 1e-9)
        if dd > max_dd: max_dd = dd
    return {
        "strategy":    "Buy & Hold (10% position)",
        "total_return": round(total_ret, 2),
        "mean_return":  round(float(net_returns.mean() * 100), 4),
        "pct_positive": round(float((net_returns > 0).mean() * 100), 1),
        "max_dd_pct":   round(max_dd * 100, 2),
        "n_samples":    len(ohlc_true),
    }


# ══════════════════════════════════════════════════════════════════
# Основная функция
# ══════════════════════════════════════════════════════════════════

def run_backtest_v3(model_path: str = "ml/model_multiscale_v3.pt",
                    save_json: str = None, use_hourly: bool = True):
    from ml.dataset_v3 import (build_full_multiscale_dataset_v3,
                                temporal_split, class_distribution)
    from ml.multiscale_cnn_v3 import MultiScaleHybridV3, _collate_v3
    from torch.utils.data import Subset, DataLoader

    print("=" * 60 + "\n  BACKTEST v3\n" + "=" * 60)

    dataset, y_all, ctx_dim, ticker_lengths = build_full_multiscale_dataset_v3(use_hourly=use_hourly)
    print(f"\n  Всего сэмплов: {len(y_all)}, ctx_dim={ctx_dim}")

    _, _, idx_test = temporal_split(ticker_lengths, val_ratio=0.15, test_ratio=0.15,
                                     purge_bars=CFG.future_bars)
    y_test = y_all[idx_test]
    te_ds  = Subset(dataset, idx_test.tolist())
    print(f"  Test set: {len(y_test)} сэмплов"); class_distribution(y_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MultiScaleHybridV3(ctx_dim=ctx_dim, use_hourly=use_hourly).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  Модель загружена: {model_path}  Hourly: {'ON' if use_hourly else 'OFF'}")

    loader = DataLoader(te_ds, batch_size=64, shuffle=False,
                        num_workers=0, pin_memory=True, collate_fn=_collate_v3)
    all_probs = []; all_ohlc_pred = []; all_ohlc_true = []

    with torch.no_grad():
        for batch in loader:
            imgs_dict, num_dict, cls_y, ohlc_y, ctx = batch[:5]
            hourly_data = batch[5] if len(batch) > 5 else None
            imgs    = {W: imgs_dict[W].to(device) for W in SCALES}
            ctx_t   = ctx.to(device) if ctx_dim > 0 else None
            hourly_t = hourly_data.to(device) if (use_hourly and hourly_data is not None) else None
            if num_dict is not None:
                nums = {W: num_dict[W].to(device) for W in SCALES}
                cls_logits, ohlc_pred = model(imgs, nums, ctx_t, hourly=hourly_t)
            else:
                cls_logits, ohlc_pred = model(imgs, ctx=ctx_t, hourly=hourly_t)
            all_probs.append(torch.softmax(cls_logits, dim=1).cpu().numpy())
            all_ohlc_pred.append(ohlc_pred.cpu().numpy())
            all_ohlc_true.append(ohlc_y.numpy())

    all_probs     = np.concatenate(all_probs)
    all_ohlc_pred = np.concatenate(all_ohlc_pred)
    all_ohlc_true = np.concatenate(all_ohlc_true)
    print(f"  Inference done: {len(all_probs)} сэмплов")

    configs = {
        "conservative": StrategyConfig(confidence_threshold=0.55, min_risk_reward=1.5,
                                        min_expected_move=0.008, risk_per_trade=0.01,
                                        max_open_positions=3),
        "balanced":     StrategyConfig(confidence_threshold=0.45, min_risk_reward=1.2,
                                        min_expected_move=0.005, risk_per_trade=0.015,
                                        max_open_positions=5),
        "aggressive":   StrategyConfig(confidence_threshold=0.38, min_risk_reward=0.8,
                                        min_expected_move=0.003, risk_per_trade=0.02,
                                        max_open_positions=8),
    }
    all_results = {}
    for name, cfg in configs.items():
        print(f"\n  ── Стратегия: {name.upper()} ──")
        bt = Backtester(cfg)
        res = bt.run(all_probs, all_ohlc_pred, all_ohlc_true, y_test)
        bt.print_report(res); all_results[name] = res

    print("\n  ── BENCHMARK: Buy & Hold ──")
    bh = buy_hold_benchmark(all_ohlc_true)
    print(f"  Return: {bh['total_return']:+.2f}%  Max DD: {bh['max_dd_pct']:.2f}%")
    all_results["benchmark_buy_hold"] = bh

    if save_json is None: save_json = model_path.replace(".pt", "_backtest.json")
    with open(save_json, "w") as f: json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Результаты → {save_json}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="ml/model_multiscale_v3.pt")
    parser.add_argument("--output",    default=None)
    parser.add_argument("--no-hourly", action="store_true")
    args = parser.parse_args()
    run_backtest_v3(args.model, args.output, use_hourly=not args.no_hourly)
